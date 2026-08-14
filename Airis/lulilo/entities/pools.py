"""SoA（Structure of Arrays）对象池：ArmorPool + BoxPool + ProjectilePool。

对应 AGENTS.md 3.1：彻底弃用 Python List/Dict，全量预分配 NumPy 连续数组。
本模块仅包含数据结构与 spawn/destroy/expand，不含任何模拟逻辑（移动/碰撞等）。

容量设计（AGENTS.md 3.4）：
    MAX_ARMOR  = 2048  装甲单位
    MAX_BOXES  = 4096  碰撞箱（每装甲 2~6 个）
    MAX_PROJ   = 16384 飞行中抛射物

动态扩容约束（AGENTS.md 3.1）：
    扩容在主线程 Tick 间隙执行（sim_paused=True），确保所有 Numba 并行循环已返回。
    扩容后引用替换是原子的——Numba 编译与数组大小无关（dtype/维度不变）。
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# 容量常量
# ---------------------------------------------------------------------------

MAX_ARMOR = 2048
MAX_BOXES = 4096
MAX_PROJ = 16384
MAX_EMITTERS = 4096  # Phase 6b 电磁辐射源池（2048 装甲红外 + 导引头/照射器 + 余量）
MAX_FORMATIONS = 256  # Phase 6c 编队 AI 池（docs/phase6-ai-design.md §1.2）

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class PartType(IntEnum):
    """碰撞箱部位类型（AGENTS.md 3.1）。"""
    HULL = 0       # 车体
    TURRET = 1     # 炮塔
    ENGINE = 2     # 引擎
    TRACK = 3      # 履带
    AMMO_RACK = 4  # 弹药架


# 部位满血表（docs/part-health-design.md §2.3）。
# HULL=100 接替原 armor.health 默认值 → AI/触发器阈值语义不变；
# 其余部位按功能重要度递减（炮塔/引擎/履带/弹药架）。
PART_MAX_HP: dict[int, float] = {
    int(PartType.HULL): 100.0,
    int(PartType.TURRET): 60.0,
    int(PartType.ENGINE): 50.0,
    int(PartType.TRACK): 40.0,
    int(PartType.AMMO_RACK): 25.0,
}


class ProjectileType(IntEnum):
    """抛射物类型。"""
    BULLET = 0   # 动能弹（APFSDS/AP）
    MISSILE = 1  # 导弹（ATGM）


class EmitterBand(IntEnum):
    """电磁辐射源波段（Phase 6b：物理上都是电磁波，仅波段不同）。

    射频 = 雷达/主动导引头/ESM；红外 = 目标热辐射/红外诱饵；激光 = 激光照射器/光斑。
    """
    RF = 0      # 射频
    IR = 1      # 红外
    LASER = 2   # 激光


# Emitter kind（docs/phase6-guidance-pipeline.md §2.2）：
#   0=平台雷达  1=主动导引头（/末敏弹毫米波扫描器）  2=激光照射器
#   3=红外辐射源  4=激光漫反射光斑（无父，不走 sync）
EMITTER_KIND_PLATFORM_RADAR = 0
EMITTER_KIND_SEEKER = 1
EMITTER_KIND_ILLUMINATOR = 2
EMITTER_KIND_IR_SOURCE = 3
EMITTER_KIND_SPOT = 4

# 父实体类型（EmitterPool.parent_kind）：0=装甲 1=弹丸 2=无父（光斑）
EMITTER_PARENT_ARMOR = 0
EMITTER_PARENT_PROJECTILE = 1
EMITTER_PARENT_NONE = 2

# Numba 用 int 常量（IntEnum 在 njit 中会报错）
_BAND_IR = int(EmitterBand.IR)
_BAND_RF = int(EmitterBand.RF)
_BAND_LASER = int(EmitterBand.LASER)


# ---------------------------------------------------------------------------
# ArmorPool
# ---------------------------------------------------------------------------


class ArmorPool:
    """装甲单位池（SoA 预分配，MAX_ARMOR=2048）。

    数组布局:
        pos       (N, 3) float32  中心位置 (x, y, z)
        half      (N, 3) float32  碰撞箱半边长
        yaw       (N,)   float32  朝向角 (rad)
        hull_idx  (N,)   int32    该装甲的 HULL 碰撞箱索引（-1 未绑定；
                                  血量权威来源 = BoxPool.hp，见 docs/part-health-design.md）
        active    (N,)   bool_    存活掩码
        wp        (N, 8, 2) float32  路径拐点（世界坐标）
        wp_count  (N,)   uint8    拐点数量
        wp_idx    (N,)   uint8    当前目标拐点索引
        mesh_id   (N,)   int32    所在表面面片 ID（-1=空中/未绑定）
        pitch     (N,)   float32  俯仰角 (rad，预留悬挂姿态)
        roll      (N,)   float32  侧倾角 (rad，预留悬挂姿态)
        suspension(N,)   float32  悬挂压缩量 (m，预留)

    docs/worldmesh-3d-surface-design.md §5：所有单位统一用
    (mesh_id, x, y, z) 表示世界位置。mesh_id >= 0 时 z 由面片
    ray_plane_z 派生；-1 为空中单位（z 自由）。
    """

    def __init__(self, capacity: int = MAX_ARMOR) -> None:
        self.capacity = int(capacity)

        # 主数组
        self.pos = np.zeros((self.capacity, 3), dtype=np.float32)
        self.half = np.zeros((self.capacity, 3), dtype=np.float32)
        self.yaw = np.zeros(self.capacity, dtype=np.float32)
        self.hull_idx = np.full(self.capacity, -1, dtype=np.int32)  # HULL box 索引（-1 未绑定）
        self.active = np.zeros(self.capacity, dtype=np.bool_)

        # 路径系统
        self.wp = np.zeros((self.capacity, 8, 2), dtype=np.float32)
        self.wp_count = np.zeros(self.capacity, dtype=np.uint8)
        self.wp_idx = np.zeros(self.capacity, dtype=np.uint8)

        # 世界模型绑定（WorldMesh）
        self.mesh_id = np.full(self.capacity, -1, dtype=np.int32)
        # 悬挂姿态（docs/suspension-slope-design.md §3，2026-08-13 启用）
        self.pitch = np.zeros(self.capacity, dtype=np.float32)
        self.roll = np.zeros(self.capacity, dtype=np.float32)
        self.suspension = np.zeros(self.capacity, dtype=np.float32)
        # 四触点表面 z + 各自面片（车跨桥头时前/后轮不同面片）
        self.contact_z = np.zeros((self.capacity, 4), dtype=np.float32)
        self.contact_mesh_id = np.full((self.capacity, 4), -1, dtype=np.int32)
        # 弹簧阻尼速度项（方案 B 临界阻尼）
        self.pitch_vel = np.zeros(self.capacity, dtype=np.float32)
        self.roll_vel = np.zeros(self.capacity, dtype=np.float32)
        # 电磁特征（Phase 6b，docs/phase6-guidance-pipeline.md §2.3）
        #   rcs: 雷达截面积（反射特性，末敏弹毫米波阈值 / 射频导引头检测用）
        #   ecm: ECM 干扰强度 0~1（压制敌方射频导引头检测）
        # 注：红外特征已迁移到 EmitterPool（band=IR, kind=3，充能-残热模型）
        self.rcs = np.full(self.capacity, 50.0, dtype=np.float32)
        self.ecm = np.zeros(self.capacity, dtype=np.float32)

        # ---- Phase 6c AI 字段（docs/phase6-ai-design.md §1.1）----
        self.team = np.zeros(self.capacity, dtype=np.int8)          # 阵营：0=友 1=敌 2=中立
        self.role = np.zeros(self.capacity, dtype=np.int8)          # 角色：0坦克 1炮兵 2侦察 3防空 4攻顶ATGM发射器 5预留(空中,Phase7)
        self.order = np.zeros(self.capacity, dtype=np.int8)         # 编队下达的执行指令（UnitOrder，非单位独立意图）
        self.target_idx = np.full(self.capacity, -1, dtype=np.int32)  # 编队分配的目标装甲索引（-1 无）
        self.target_kind = np.zeros(self.capacity, dtype=np.int8)   # 目标类型：0=装甲 1=辐射源 2=光斑 3=地形点
        self.comm = np.ones(self.capacity, dtype=np.float32)        # 通信质量 0~1（数据链，门控命令下达）
        self.contact_age = np.full(self.capacity, 1e9, dtype=np.float32)  # 最近一次接触距今（秒，态势老化）
        self.formation_id = np.full(self.capacity, -1, dtype=np.int32)  # 所属编队 AI 池索引（-1 未编队）

        # 空闲栈：存储未被占用的索引
        self._free_stack: list[int] = list(range(self.capacity))

    # ------------------------------------------------------------------
    # 激活/销毁
    # ------------------------------------------------------------------

    def spawn(
        self,
        x: float,
        y: float,
        z: float,
        half_x: float = 1.0,
        half_y: float = 3.0,
        half_z: float = 1.0,
        yaw: float = 0.0,
        hull_idx: int = -1,
        mesh_id: int = -1,
        rcs: float = 50.0,
        ecm: float = 0.0,
        team: int = 0,
        role: int = 0,
        formation_id: int = -1,
    ) -> int:
        """生成一辆装甲车，返回其池索引。池满则扩容后继续。

        Args:
            x, y, z: 世界坐标。
            half_x, half_y, half_z: 碰撞箱半边长。
            yaw: 朝向角 (rad)。
            hull_idx: HULL 碰撞箱索引（-1 未绑定；创建 HULL box 后调用 bind_hull 填入）。
            mesh_id: 所在表面面片 ID（WorldMesh；-1=空中/未绑定，
                随后由 z 同步自动定位）。
            rcs: 雷达截面积 (m²，反射特性)。
            ecm: ECM 干扰强度 0~1。
        """
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self.pos[idx] = (x, y, z)
        self.half[idx] = (half_x, half_y, half_z)
        self.yaw[idx] = yaw
        self.hull_idx[idx] = int(hull_idx)
        self.active[idx] = True
        self.mesh_id[idx] = int(mesh_id)
        self.pitch[idx] = 0.0
        self.roll[idx] = 0.0
        self.suspension[idx] = 0.0
        self.contact_z[idx] = 0.0
        self.contact_mesh_id[idx] = -1
        self.pitch_vel[idx] = 0.0
        self.roll_vel[idx] = 0.0
        self.rcs[idx] = float(rcs)
        self.ecm[idx] = float(ecm)
        # ---- Phase 6c AI 字段（docs/phase6-ai-design.md §1.1）----
        self.team[idx] = int(team)
        self.role[idx] = int(role)
        self.order[idx] = 0
        self.target_idx[idx] = -1
        self.target_kind[idx] = 0
        self.comm[idx] = 1.0
        self.contact_age[idx] = 1e9
        self.formation_id[idx] = int(formation_id)
        # 清除残留路径
        self.wp[idx] = 0.0
        self.wp_count[idx] = 0
        self.wp_idx[idx] = 0
        return idx

    def destroy(self, idx: int) -> None:
        """销毁装甲（标记 inactive，归还索引）。"""
        self.active[idx] = False
        self.hull_idx[idx] = -1
        self._free_stack.append(idx)

    def bind_hull(self, armor_idx: int, box_idx: int) -> None:
        """绑定该装甲的 HULL 碰撞箱索引（docs/part-health-design.md §2.2）。

        创建 HULL box 后调用；血量权威来源 = BoxPool.hp[box_idx]。
        """
        self.hull_idx[armor_idx] = int(box_idx)

    # ------------------------------------------------------------------
    # 路径设置
    # ------------------------------------------------------------------

    def set_waypoints(self, idx: int, waypoints: list[tuple[float, float]]) -> None:
        """为装甲设置路径拐点（最多 8 个）。

        Args:
            idx: 装甲索引。
            waypoints: 世界坐标 (x, y) 列表。
        """
        n = min(len(waypoints), 8)
        for i in range(n):
            self.wp[idx, i, 0] = waypoints[i][0]
            self.wp[idx, i, 1] = waypoints[i][1]
        self.wp_count[idx] = n
        self.wp_idx[idx] = 0

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """当前存活数。"""
        return int(np.sum(self.active))

    def active_indices(self) -> np.ndarray:
        """返回所有 active=True 的索引数组。"""
        return np.flatnonzero(self.active)

    # ------------------------------------------------------------------
    # 扩容（Tick 间隙调用）
    # ------------------------------------------------------------------

    def _expand(self, factor: float = 2.0) -> None:
        """内部扩容（AGENTS.md 3.1：Tick 间隙 + 原子引用替换）。"""
        new_cap = max(int(self.capacity * factor), self.capacity + 1)

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.pos = _resize(self.pos, (new_cap, 3))
        self.half = _resize(self.half, (new_cap, 3))
        self.yaw = _resize(self.yaw, (new_cap,))
        self.hull_idx = _resize(self.hull_idx, (new_cap,))
        self.active = _resize(self.active, (new_cap,))
        self.wp = _resize(self.wp, (new_cap, 8, 2))
        self.wp_count = _resize(self.wp_count, (new_cap,))
        self.wp_idx = _resize(self.wp_idx, (new_cap,))
        self.mesh_id = _resize(self.mesh_id, (new_cap,))
        self.pitch = _resize(self.pitch, (new_cap,))
        self.roll = _resize(self.roll, (new_cap,))
        self.suspension = _resize(self.suspension, (new_cap,))
        self.contact_z = _resize(self.contact_z, (new_cap, 4))
        self.contact_mesh_id = _resize(self.contact_mesh_id, (new_cap, 4))
        self.pitch_vel = _resize(self.pitch_vel, (new_cap,))
        self.roll_vel = _resize(self.roll_vel, (new_cap,))
        self.rcs = _resize(self.rcs, (new_cap,))
        self.ecm = _resize(self.ecm, (new_cap,))
        # ---- Phase 6c AI 字段（docs/phase6-ai-design.md §1.1）----
        self.team = _resize(self.team, (new_cap,))
        self.role = _resize(self.role, (new_cap,))
        self.order = _resize(self.order, (new_cap,))
        self.target_idx = _resize(self.target_idx, (new_cap,))
        self.target_kind = _resize(self.target_kind, (new_cap,))
        self.comm = _resize(self.comm, (new_cap,))
        self.contact_age = _resize(self.contact_age, (new_cap,))
        self.formation_id = _resize(self.formation_id, (new_cap,))

        old_cap = self.capacity
        # ---- 非零默认值补填（_resize 用 zeros；新区恢复构造默认）----
        self.target_idx[old_cap:] = -1
        self.comm[old_cap:] = 1.0
        self.contact_age[old_cap:] = 1e9
        self.formation_id[old_cap:] = -1
        self.hull_idx[old_cap:] = -1
        self.rcs[old_cap:] = 50.0  # 既有遗留：rcs 构造默认 50.0（非零）

        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))


# ---------------------------------------------------------------------------
# FormationAIPool（Phase 6c，docs/phase6-ai-design.md §1.2）
# ---------------------------------------------------------------------------


class FormationAIPool:
    """编队 AI 池（SoA，MAX_FORMATIONS=256）。一编队 = 单/多单位，战术姿态由编队 AI 决定。

    数组布局:
        active       (N,)   bool_   编队存活
        team         (N,)   int8    阵营
        role         (N,)   int8    编队角色（装甲/炮兵/侦察/混合）
        intent       (N,)   int8    编队意图 FSM（FormationIntent，simulation/ai.py）
        order        (N,)   int8    编队级命令（上级/玩家下达）
        stance       (N,)   int8    战术姿态（进攻/防御）
        target_zone  (N, 3) float32 任务目标区中心
        comm         (N,)   float32 编队通信质量（指挥链门控）
        member_count (N,)   int32   成员数
        contact_set  (N,)   Python  汇总敌接触表（非热路径，Python list）
    """

    def __init__(self, capacity: int = MAX_FORMATIONS) -> None:
        self.capacity = int(capacity)
        self.active = np.zeros(capacity, dtype=np.bool_)
        self.team = np.zeros(capacity, dtype=np.int8)
        self.role = np.zeros(capacity, dtype=np.int8)
        self.intent = np.zeros(capacity, dtype=np.int8)
        self.order = np.zeros(capacity, dtype=np.int8)
        self.stance = np.zeros(capacity, dtype=np.int8)
        self.target_zone = np.zeros((capacity, 3), dtype=np.float32)
        self.comm = np.ones(capacity, dtype=np.float32)
        self.member_count = np.zeros(capacity, dtype=np.int32)
        self.contact_set: list[list | None] = [None] * capacity
        self._free_stack: list[int] = list(range(capacity))

    def spawn(
        self,
        team: int = 0,
        role: int = 0,
        target_zone: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """登记一个编队 AI，返回池索引。池满则扩容后继续。

        Args:
            team: 阵营（0=友 1=敌 2=中立）。
            role: 编队角色（0=装甲 1=炮兵 2=侦察 3=防空 4=攻顶ATGM 5=混合/预留）。
            target_zone: 任务目标区中心 (x, y, z)。
        """
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self.active[idx] = True
        self.team[idx] = int(team)
        self.role[idx] = int(role)
        self.intent[idx] = 0
        self.order[idx] = 0
        self.stance[idx] = 0
        self.target_zone[idx] = target_zone
        self.comm[idx] = 1.0
        self.member_count[idx] = 0
        self.contact_set[idx] = None
        return idx

    def destroy(self, idx: int) -> None:
        """注销编队 AI（标记 inactive，归还索引）。"""
        self.active[idx] = False
        self.contact_set[idx] = None
        self._free_stack.append(idx)

    @property
    def active_count(self) -> int:
        """当前活跃编队数。"""
        return int(np.sum(self.active))

    def active_indices(self) -> np.ndarray:
        """返回所有 active=True 的索引数组。"""
        return np.flatnonzero(self.active)

    def _expand(self, factor: float = 2.0) -> None:
        """内部扩容（Tick 间隙调用，原子引用替换）。"""
        new_cap = max(int(self.capacity * factor), self.capacity + 1)

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.active = _resize(self.active, (new_cap,))
        self.team = _resize(self.team, (new_cap,))
        self.role = _resize(self.role, (new_cap,))
        self.intent = _resize(self.intent, (new_cap,))
        self.order = _resize(self.order, (new_cap,))
        self.stance = _resize(self.stance, (new_cap,))
        self.target_zone = _resize(self.target_zone, (new_cap, 3))
        self.comm = _resize(self.comm, (new_cap,))
        self.member_count = _resize(self.member_count, (new_cap,))
        self.contact_set.extend([None] * (new_cap - self.capacity))

        old_cap = self.capacity
        self.comm[old_cap:] = 1.0  # 非零默认值补填（_resize 用 zeros）
        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))


# ---------------------------------------------------------------------------
# BoxPool
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _sync_boxes_kernel(
    box_local: np.ndarray,     # (M, 3) 局部偏移
    box_parent: np.ndarray,    # (M,) int32
    box_active: np.ndarray,    # (M,) bool_（原地修改：父装甲死亡时失活）
    box_pos: np.ndarray,       # (M, 3) 世界坐标（out，原地写）
    armor_pos: np.ndarray,     # (N, 3) 父装甲位置
    armor_yaw: np.ndarray,     # (N,) 父装甲朝向
    armor_pitch: np.ndarray,   # (N,) 父装甲俯仰（绕车身横轴 ly）
    armor_roll: np.ndarray,    # (N,) 父装甲侧倾（绕车身纵轴 lx）
    armor_active: np.ndarray,  # (N,) 父装甲存活
) -> None:
    """Numba 并行：把所有活跃碰撞箱从局部偏移变换到世界坐标。

    变换 = Rz(yaw)·Rx(roll)·Ry(pitch)·local（docs/suspension-slope-design.md
    §3.8，与渲染 Model 矩阵同一约定）。与 Python 版 sync_from_parent 同语义
    （每元素独立 → 无竞争），但全量并行——Phase 3 开火链每子弹 Tick 调用
    一次（16384 box 级 < 1ms）。
    """
    m = box_pos.shape[0]
    for i in prange(m):
        if not box_active[i]:
            continue
        p = box_parent[i]
        if p < 0 or not armor_active[p]:
            box_active[i] = False
            continue
        lx, ly, lz = box_local[i]
        cos_y = np.cos(armor_yaw[p])
        sin_y = np.sin(armor_yaw[p])
        cos_r = np.cos(armor_roll[p])
        sin_r = np.sin(armor_roll[p])
        cos_p = np.cos(armor_pitch[p])
        sin_p = np.sin(armor_pitch[p])
        # Ry(pitch)：绕车身横轴（y 轴）
        v1x = lx * cos_p + lz * sin_p
        v1y = ly
        v1z = -lx * sin_p + lz * cos_p
        # Rx(roll)：绕车身纵轴（x 轴）
        v2x = v1x
        v2y = v1y * cos_r - v1z * sin_r
        v2z = v1y * sin_r + v1z * cos_r
        # Rz(yaw)
        box_pos[i, 0] = armor_pos[p, 0] + v2x * cos_y - v2y * sin_y
        box_pos[i, 1] = armor_pos[p, 1] + v2x * sin_y + v2y * cos_y
        box_pos[i, 2] = armor_pos[p, 2] + v2z


class BoxPool:
    """碰撞箱池（SoA 预分配，MAX_BOXES=4096）。

    每辆装甲车由 2~6 个碰撞箱组成（车体/炮塔/引擎/履带/弹药架）。
    碰撞箱位置为世界坐标（由装甲 pos + yaw 变换后写入）。

    数组布局（注意：实际分配 capacity+1，末位为哨兵槽，docs/part-health-design.md §2.1）:
        pos       (N, 3) float32  世界坐标
        half      (N, 3) float32  局部半边长（在装甲局部坐标系中为 AABB）
        parent    (N,)   int32    所属装甲索引
        part      (N,)   int32    部位类型（PartType 枚举）
        active    (N,)   bool_    存活掩码
        hp        (N,)   float32  当前部位血量（哨兵槽恒 0）
        hp_max    (N,)   float32  部位满血（spawn 时按 part 设定）
    """

    def __init__(self, capacity: int = MAX_BOXES) -> None:
        self.capacity = int(capacity)
        # 哨兵槽：分配 capacity+1，末位（index==capacity）恒 hp=0、active=False、不进 free_stack。
        # armor.hull_idx=-1 时 fancy indexing boxes.hp[hull_idx] 读出哨兵 0，避免负索引垃圾值。
        alloc = capacity + 1

        self.local = np.zeros((alloc, 3), dtype=np.float32)  # 局部偏移（权威）
        self.pos = np.zeros((alloc, 3), dtype=np.float32)    # 世界坐标（sync 后）
        self.half = np.zeros((alloc, 3), dtype=np.float32)
        self.parent = np.full(alloc, -1, dtype=np.int32)
        self.part = np.zeros(alloc, dtype=np.int32)
        self.active = np.zeros(alloc, dtype=np.bool_)
        self.hp = np.zeros(alloc, dtype=np.float32)
        self.hp_max = np.zeros(alloc, dtype=np.float32)

        self._free_stack: list[int] = list(range(self.capacity))

    # ------------------------------------------------------------------
    # 激活/销毁
    # ------------------------------------------------------------------

    def spawn(
        self,
        parent_idx: int,
        part: int,
        half_x: float,
        half_y: float,
        half_z: float,
        local_x: float = 0.0,
        local_y: float = 0.0,
        local_z: float = 0.0,
    ) -> int:
        """生成碰撞箱（局部偏移 + 半边长，世界位置由父装甲 pos/yaw 变换后填入）。

        注意：调用者应在此后调用 `sync_from_parent(idx, armor_pool)` 更新世界位置。

        Args:
            parent_idx: 所属装甲索引。
            part: 部位类型（PartType 枚举）。
            half_x, half_y, half_z: 碰撞箱局部半边长。
            local_x, local_y, local_z: 相对父装甲中心的局部偏移。

        Returns:
            碰撞箱索引。
        """
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self.parent[idx] = parent_idx
        self.part[idx] = int(part)
        self.half[idx] = (half_x, half_y, half_z)
        # 局部偏移存入 local（权威）；pos 在 sync 前保持局部偏移（兼容旧语义）
        self.local[idx] = (local_x, local_y, local_z)
        self.pos[idx] = (local_x, local_y, local_z)
        self.active[idx] = True
        self.hp[idx] = PART_MAX_HP.get(int(part), 100.0)
        self.hp_max[idx] = self.hp[idx]
        return idx

    def destroy(self, idx: int) -> None:
        """销毁碰撞箱。"""
        self.active[idx] = False
        self.parent[idx] = -1
        self.hp[idx] = 0.0
        self.hp_max[idx] = 0.0
        self._free_stack.append(idx)

    # ------------------------------------------------------------------
    # 同步：将装甲 pos/yaw 变换到世界坐标
    # ------------------------------------------------------------------

    def sync_from_parent(self, idx: int, armor: ArmorPool) -> None:
        """将碰撞箱 idx 的局部偏移变换到世界坐标（基于父装甲 pos/yaw）。

        从 self.local（权威局部偏移）读取，可重复调用（幂等）。
        应在每帧移动积分后调用。
        """
        p = self.parent[idx]
        if p < 0 or not armor.active[p]:
            self.active[idx] = False
            return

        lx, ly, lz = self.local[idx]
        cos_y = np.cos(armor.yaw[p])
        sin_y = np.sin(armor.yaw[p])
        cos_r = np.cos(armor.roll[p])
        sin_r = np.sin(armor.roll[p])
        cos_p = np.cos(armor.pitch[p])
        sin_p = np.sin(armor.pitch[p])
        # Rz(yaw)·Rx(roll)·Ry(pitch)·local（docs §3.8，与内核一致）
        v1x = lx * cos_p + lz * sin_p
        v1y = ly
        v1z = -lx * sin_p + lz * cos_p
        v2x = v1x
        v2y = v1y * cos_r - v1z * sin_r
        v2z = v1y * sin_r + v1z * cos_r
        wx = armor.pos[p, 0] + v2x * cos_y - v2y * sin_y
        wy = armor.pos[p, 1] + v2x * sin_y + v2y * cos_y
        wz = armor.pos[p, 2] + v2z
        self.pos[idx] = (wx, wy, wz)

    def sync_all_from_parent(self, armor: ArmorPool) -> None:
        """批量同步所有活跃碰撞箱（Numba 并行，幂等可重复调用）。"""
        _sync_boxes_kernel(
            self.local, self.parent, self.active, self.pos,
            armor.pos, armor.yaw, armor.pitch, armor.roll, armor.active,
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return int(np.sum(self.active))

    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    # ------------------------------------------------------------------
    # 扩容
    # ------------------------------------------------------------------

    def _expand(self, factor: float = 2.0) -> None:
        new_cap = max(int(self.capacity * factor), self.capacity + 1)
        alloc = new_cap + 1  # 哨兵槽维持 capacity+1

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.local = _resize(self.local, (alloc, 3))
        self.pos = _resize(self.pos, (alloc, 3))
        self.half = _resize(self.half, (alloc, 3))
        self.parent = _resize(self.parent, (alloc,))
        self.part = _resize(self.part, (alloc,))
        self.active = _resize(self.active, (alloc,))
        self.hp = _resize(self.hp, (alloc,))
        self.hp_max = _resize(self.hp_max, (alloc,))

        old_cap = self.capacity
        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))


# ---------------------------------------------------------------------------
# ProjectilePool
# ---------------------------------------------------------------------------


class ProjectilePool:
    """抛射物池（SoA 预分配，MAX_PROJ=16384）。

    数组布局:
        pos       (N, 3) float32  世界位置
        vel       (N, 3) float32  速度向量 (m/s)
        radius    (N,)   float32  碰撞球半径
        dmg       (N,)   float32  伤害值
        kind      (N,)   int32    类型（ProjectileType）
        active    (N,)   bool_    存活掩码
        owner     (N,)   int32    发射者装甲索引（-1 表示无主）
        weapon    (N,)   int32    武器注册表索引（-1 无；下沉自 _proj_meta dict）
        ammo      (N,)   int32    弹药注册表索引（-1 无；下沉自 _proj_meta dict）
        age       (N,)   float32  已飞行秒数（下沉自 _proj_age dict）

    制导字段（Phase 6b，docs/phase6-guidance-pipeline.md §2.1）：
        guid_target_pos      (N, 3) float32 当前检测到的目标世界坐标（detection 输出；
                                          TRACK/TERMINAL 有效；fire 写初始瞄准坐标）
        guid_law             (N,)   int32   控制律注册表索引（-1=无制导；末敏弹也有索引，
                                          BALLISTIC 阶段该函数不制导）
        guid_nav_gain        (N,)   float32 制导增益 N（per-ammo）
        guid_stage           (N,)   int32   GuidanceStage（给控制回调的状态机输入）
        guid_memory_pos      (N, 3) float32 最后已知目标位置（MEMORY 用）
        guid_comm            (N,)   float32 通信质量 0~1（人在回路命令可达性门控）
        guid_confidence      (N,)   float32 友军对导弹位置知晓置信度 0~1（认知孪生，
                                          **与制导无关**，Phase 8 孪生层维护）
        guid_override        (N,)   int32   人在回路推进指令（ManualOverride，已门控）
        guid_search_fov_cos  (N,)   float32 seeker 搜索视场余弦（per-ammo，fire 写）
        guid_track_fov_cos   (N,)   float32 seeker 跟踪视场余弦（per-ammo，fire 写）
        guid_filter_quality  (N,)   float32 滤波能力 0~1（Python 检测层用）
    """

    def __init__(self, capacity: int = MAX_PROJ) -> None:
        self.capacity = int(capacity)

        self.pos = np.zeros((self.capacity, 3), dtype=np.float32)
        self.vel = np.zeros((self.capacity, 3), dtype=np.float32)
        self.radius = np.zeros(self.capacity, dtype=np.float32)
        self.dmg = np.zeros(self.capacity, dtype=np.float32)
        self.kind = np.zeros(self.capacity, dtype=np.int32)
        self.active = np.zeros(self.capacity, dtype=np.bool_)
        self.owner = np.full(self.capacity, -1, dtype=np.int32)
        self.weapon = np.full(self.capacity, -1, dtype=np.int32)
        self.ammo = np.full(self.capacity, -1, dtype=np.int32)
        self.age = np.zeros(self.capacity, dtype=np.float32)

        # ---- 制导字段（docs/phase6-guidance-pipeline.md §2.1）----
        self.guid_target_pos = np.zeros((self.capacity, 3), dtype=np.float32)
        self.guid_law = np.full(self.capacity, -1, dtype=np.int32)
        self.guid_nav_gain = np.full(self.capacity, 3.0, dtype=np.float32)
        self.guid_stage = np.zeros(self.capacity, dtype=np.int32)
        self.guid_memory_pos = np.zeros((self.capacity, 3), dtype=np.float32)
        self.guid_comm = np.ones(self.capacity, dtype=np.float32)
        self.guid_confidence = np.ones(self.capacity, dtype=np.float32)
        self.guid_override = np.zeros(self.capacity, dtype=np.int32)
        self.guid_search_fov_cos = np.ones(self.capacity, dtype=np.float32)
        self.guid_track_fov_cos = np.ones(self.capacity, dtype=np.float32)
        self.guid_filter_quality = np.full(self.capacity, 0.7, dtype=np.float32)

        self._free_stack: list[int] = list(range(self.capacity))

    # ------------------------------------------------------------------
    # 激活/销毁
    # ------------------------------------------------------------------

    def spawn(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        radius: float = 0.05,
        dmg: float = 10.0,
        kind: int = ProjectileType.BULLET,
        owner: int = -1,
        weapon: int = -1,
        ammo: int = -1,
    ) -> int:
        """发射抛射物，返回索引。

        Args:
            weapon/ammo: 武器/弹药注册表索引（-1 无元数据）。
                下沉自骨架期 FiringControl._proj_meta dict，见
                docs/ammunition-expansion-design.md §八。
        """
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self.pos[idx] = (x, y, z)
        self.vel[idx] = (vx, vy, vz)
        self.radius[idx] = radius
        self.dmg[idx] = dmg
        self.kind[idx] = int(kind)
        self.owner[idx] = owner
        self.weapon[idx] = int(weapon)
        self.ammo[idx] = int(ammo)
        self.age[idx] = 0.0
        self.active[idx] = True
        # 制导字段初始化（fire() 随后按 ammo 写实际值）
        self.guid_target_pos[idx] = 0.0
        self.guid_law[idx] = -1
        self.guid_nav_gain[idx] = 3.0
        self.guid_stage[idx] = 0
        self.guid_memory_pos[idx] = 0.0
        self.guid_comm[idx] = 1.0
        self.guid_confidence[idx] = 1.0
        self.guid_override[idx] = 0
        self.guid_search_fov_cos[idx] = 1.0
        self.guid_track_fov_cos[idx] = 1.0
        self.guid_filter_quality[idx] = 0.7
        return idx

    def destroy(self, idx: int) -> None:
        """销毁抛射物（命中后或飞出边界）。"""
        self.active[idx] = False
        self.owner[idx] = -1
        self.weapon[idx] = -1
        self.ammo[idx] = -1
        self.age[idx] = 0.0
        # 制导字段清零（与 weapon/ammo/age 同批）
        self.guid_target_pos[idx] = 0.0
        self.guid_law[idx] = -1
        self.guid_nav_gain[idx] = 3.0
        self.guid_stage[idx] = 0
        self.guid_memory_pos[idx] = 0.0
        self.guid_comm[idx] = 1.0
        self.guid_confidence[idx] = 1.0
        self.guid_override[idx] = 0
        self.guid_search_fov_cos[idx] = 1.0
        self.guid_track_fov_cos[idx] = 1.0
        self.guid_filter_quality[idx] = 0.7
        self._free_stack.append(idx)

    def destroy_batch(self, indices: np.ndarray) -> None:
        """批量销毁（Numba 外部调用后的结果写入）。"""
        self.active[indices] = False
        self.owner[indices] = -1
        self.weapon[indices] = -1
        self.ammo[indices] = -1
        self.age[indices] = 0.0
        self.guid_target_pos[indices] = 0.0
        self.guid_law[indices] = -1
        self.guid_nav_gain[indices] = 3.0
        self.guid_stage[indices] = 0
        self.guid_memory_pos[indices] = 0.0
        self.guid_comm[indices] = 1.0
        self.guid_confidence[indices] = 1.0
        self.guid_override[indices] = 0
        self.guid_search_fov_cos[indices] = 1.0
        self.guid_track_fov_cos[indices] = 1.0
        self.guid_filter_quality[indices] = 0.7
        self._free_stack.extend(indices.tolist())

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return int(np.sum(self.active))

    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    # ------------------------------------------------------------------
    # 扩容
    # ------------------------------------------------------------------

    def _expand(self, factor: float = 2.0) -> None:
        new_cap = max(int(self.capacity * factor), self.capacity + 1)

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.pos = _resize(self.pos, (new_cap, 3))
        self.vel = _resize(self.vel, (new_cap, 3))
        self.radius = _resize(self.radius, (new_cap,))
        self.dmg = _resize(self.dmg, (new_cap,))
        self.kind = _resize(self.kind, (new_cap,))
        self.active = _resize(self.active, (new_cap,))
        self.owner = _resize(self.owner, (new_cap,))
        self.weapon = _resize(self.weapon, (new_cap,))
        self.ammo = _resize(self.ammo, (new_cap,))
        self.age = _resize(self.age, (new_cap,))
        # 制导字段同步扩容（dtype/维度不变，Numba 免重编译）
        self.guid_target_pos = _resize(self.guid_target_pos, (new_cap, 3))
        self.guid_law = _resize(self.guid_law, (new_cap,))
        self.guid_nav_gain = _resize(self.guid_nav_gain, (new_cap,))
        self.guid_stage = _resize(self.guid_stage, (new_cap,))
        self.guid_memory_pos = _resize(self.guid_memory_pos, (new_cap, 3))
        self.guid_comm = _resize(self.guid_comm, (new_cap,))
        self.guid_confidence = _resize(self.guid_confidence, (new_cap,))
        self.guid_override = _resize(self.guid_override, (new_cap,))
        self.guid_search_fov_cos = _resize(self.guid_search_fov_cos, (new_cap,))
        self.guid_track_fov_cos = _resize(self.guid_track_fov_cos, (new_cap,))
        self.guid_filter_quality = _resize(self.guid_filter_quality, (new_cap,))

        old_cap = self.capacity
        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))


# ---------------------------------------------------------------------------
# EmitterPool（Phase 6b 统一电磁辐射源，docs/phase6-guidance-pipeline.md §2.2/§6.2）
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _sync_emitters_kernel(
    epos: np.ndarray,          # (M, 3) float32 辐射源位置（out，父位置同步）
    eactive: np.ndarray,       # (M,) bool_（原地修改：非红外父死亡 → 失活）
    eparent_kind: np.ndarray,  # (M,) int32 父实体类型 0=装甲 1=弹丸 2=无父
    eparent_idx: np.ndarray,   # (M,) int32 父实体索引
    eband: np.ndarray,         # (M,) int32 EmitterBand（0=RF 1=IR 2=LASER）
    epower: np.ndarray,        # (M,) float32 额定强度（红外=充能目标值）
    ecur: np.ndarray,          # (M,) float32 当前强度（红外充能/残热）
    armor_pos: np.ndarray,     # (N, 3) float32
    armor_active: np.ndarray,  # (N,) bool_
    proj_pos: np.ndarray,      # (P, 3) float32
    proj_active: np.ndarray,   # (P,) bool_
) -> None:
    """Numba 并行：辐射源位置同步自父实体 + 级联失效兜底。

    - 父实体（装甲/弹丸）inactive → 非红外 emitter 立即失活（级联兜底，
      Python 层漏关也会被清）；红外 emitter 保留（残热衰减交给 Python 层）。
    - 红外：父存活 → 充能维持额定强度（cur_power = power_db）。
    - 无父（激光漫反射光斑 kind=4）：位置固定，不参与 sync（SAL 系统管理存活）。
    """
    for i in prange(epos.shape[0]):
        if not eactive[i]:
            continue
        pk = int(eparent_kind[i])
        if pk == EMITTER_PARENT_ARMOR:  # 0 父 = 装甲
            p = int(eparent_idx[i])
            if p < 0 or not armor_active[p]:
                if eband[i] != _BAND_IR:
                    eactive[i] = False
                continue  # 红外：交给 Python 层残热衰减
            epos[i, 0] = armor_pos[p, 0]
            epos[i, 1] = armor_pos[p, 1]
            epos[i, 2] = armor_pos[p, 2]
        elif pk == EMITTER_PARENT_PROJECTILE:  # 1 父 = 弹丸
            p = int(eparent_idx[i])
            if p < 0 or not proj_active[p]:
                if eband[i] != _BAND_IR:
                    eactive[i] = False
                continue
            epos[i, 0] = proj_pos[p, 0]
            epos[i, 1] = proj_pos[p, 1]
            epos[i, 2] = proj_pos[p, 2]
        else:  # 2 无父（光斑）：位置固定，不参与 sync
            continue
        if eband[i] == _BAND_IR:
            ecur[i] = epower[i]  # 红外：父存活 → 充能维持额定强度


class EmitterPool:
    """电磁辐射源池：射频 / 红外 / 激光统一管理（物理上都是电磁波，仅波段不同）。

    数组布局:
        pos         (N, 3) float32  世界位置（每 tick sync 自父实体）
        power_db    (N,)   float32  额定发射强度（RF=dBm / 红外=额定热强度 / LASER=功率）
        cur_power   (N,)   float32  当前强度（红外：充能/残热；主动 = power_db）
        decay_per_s (N,)   float32  残热衰减速率（红外用，父死亡后指数衰减）
        band        (N,)   int32    EmitterBand：0=射频 1=红外 2=激光
        kind        (N,)   int32    0=平台雷达 1=主动导引头 2=激光照射器
                                   3=红外辐射源 4=激光漫反射光斑
        active      (N,)   bool_    是否在辐射（主动=开机；红外=cur_power≥检测阈值）
        parent_kind (N,)   int32    父实体类型 0=装甲 1=弹丸 2=无父（光斑，不走 sync）
        parent_idx  (N,)   int32    父实体索引
        normal      (N, 3) float32  光斑法线（kind=4，目标表面法线，半球检测用）

    生命周期三态（docs §2.2）：
        activate(...)  从 _free_stack 弹索引，绑定父实体
        deactivate(idx) 战术关机（保留索引，可 reactivate 重新开机）
        级联失效        sync_from_parent 发现父实体 inactive → 主动辐射立即失活；
                       红外停止充能、残热衰减后失活（索引归还 _free_stack）
    """

    IR_OFF_THRESHOLD = 0.05  # 红外残热低于此值 → 失活归还

    def __init__(self, capacity: int = MAX_EMITTERS) -> None:
        self.capacity = int(capacity)
        self.pos = np.zeros((self.capacity, 3), dtype=np.float32)
        self.power_db = np.zeros(self.capacity, dtype=np.float32)
        self.cur_power = np.zeros(self.capacity, dtype=np.float32)
        self.decay_per_s = np.zeros(self.capacity, dtype=np.float32)
        self.band = np.zeros(self.capacity, dtype=np.int32)
        self.kind = np.zeros(self.capacity, dtype=np.int32)
        self.active = np.zeros(self.capacity, dtype=np.bool_)
        self.parent_kind = np.zeros(self.capacity, dtype=np.int32)
        self.parent_idx = np.full(self.capacity, -1, dtype=np.int32)
        self.normal = np.zeros((self.capacity, 3), dtype=np.float32)
        # 可用索引池：list 供 O(1) pop；set 供 O(1) 成员检查（sync 每帧遍历容量时
        # 若用 `i in list` 是 O(n) 线性搜索 → 4096×4096 比较/帧，性能瓶颈）
        self._free_stack: list[int] = list(range(self.capacity))
        self._free_set: set[int] = set(range(self.capacity))
        # 战术关机保留索引（deactivate 保留、reactivate 复用；级联失效不在此）
        self._deactivated: set[int] = set()

    # ------------------------------------------------------------------
    # 激活
    # ------------------------------------------------------------------

    def _alloc(self) -> int:
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self._free_set.discard(idx)
        return idx

    def activate(
        self,
        parent_kind: int,
        parent_idx: int,
        power_db: float,
        band: int,
        kind: int,
        decay_per_s: float = 0.0,
    ) -> int:
        """从 free_stack 弹索引并绑定父实体（普通激活）。"""
        idx = self._alloc()
        self.parent_kind[idx] = int(parent_kind)
        self.parent_idx[idx] = int(parent_idx)
        self.power_db[idx] = float(power_db)
        self.cur_power[idx] = float(power_db)
        self.decay_per_s[idx] = float(decay_per_s)
        self.band[idx] = int(band)
        self.kind[idx] = int(kind)
        self.active[idx] = True
        self._deactivated.discard(idx)
        return idx

    def activate_from_projectile(
        self,
        proj_idx: int,
        power_db: float,
        band: int = EmitterBand.RF,
        kind: int = EMITTER_KIND_SEEKER,
    ) -> int:
        """主动导引头 / 末敏弹毫米波扫描器：父实体 = 弹丸。"""
        return self.activate(
            EMITTER_PARENT_PROJECTILE, proj_idx, power_db, band, kind
        )

    def activate_ir(
        self, armor_idx: int, ir_strength: float, decay_per_s: float = 0.2
    ) -> int:
        """绑定红外辐射源（band=IR, kind=3）：仅热辐射显著单位（docs §2.2）。

        cur_power 从 0 开始，父存活时由 sync 内核充能到额定 ir_strength。
        """
        idx = self._alloc()
        self.parent_kind[idx] = EMITTER_PARENT_ARMOR
        self.parent_idx[idx] = int(armor_idx)
        self.power_db[idx] = float(ir_strength)
        self.cur_power[idx] = 0.0  # 从 0 充能
        self.decay_per_s[idx] = float(decay_per_s)
        self.band[idx] = _BAND_IR
        self.kind[idx] = EMITTER_KIND_IR_SOURCE
        self.active[idx] = True
        self._deactivated.discard(idx)
        return idx

    def pulse(self, idx: int, peak_power: float) -> None:
        """殉爆充能：一次性给红外源充大量能量（cur_power 瞬时拉高后按 decay 衰减）。

        接入点 = Phase 3 既有 cookoff 判定（docs §2.2 陷阱 §八.17）。
        """
        if 0 <= idx < self.capacity and self.active[idx]:
            self.cur_power[idx] = max(float(self.cur_power[idx]), float(peak_power))

    def designate(
        self,
        target_pos: tuple[float, float, float] | np.ndarray,
        power: float,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        illum_idx: int = -1,
    ) -> int:
        """放置激光漫反射光斑（kind=4，无父，不走 sync）。

        吊舱 LOS 射线检测后调用（docs §7.5）；光斑存活由 SAL 系统（Python 层）
        每 tick 检查「吊舱 active 且 LOS 通」决定保留或 deactivate。
        normal 存目标表面法线（半球漫反射：背面检测不到）。
        """
        idx = self._alloc()
        self.parent_kind[idx] = EMITTER_PARENT_NONE
        self.parent_idx[idx] = int(illum_idx)  # 记录来源照射器（sync 不消费）
        self.power_db[idx] = float(power)
        self.cur_power[idx] = float(power)
        self.decay_per_s[idx] = 0.0
        self.band[idx] = _BAND_LASER
        self.kind[idx] = EMITTER_KIND_SPOT
        self.pos[idx] = tuple(float(v) for v in target_pos)
        self.normal[idx] = tuple(float(v) for v in normal)
        self.active[idx] = True
        self._deactivated.discard(idx)
        return idx

    # ------------------------------------------------------------------
    # 战术关机（保留索引，可重新开机）
    # ------------------------------------------------------------------

    def deactivate(self, idx: int) -> None:
        """战术关机（主动静默，如被 ARM 威胁）；保留索引可 reactivate。"""
        if 0 <= idx < self.capacity:
            self.active[idx] = False
            self._deactivated.add(idx)

    def reactivate(self, idx: int) -> None:
        """重新开机（须是 deactivate 保留的索引）。"""
        if 0 <= idx < self.capacity and idx in self._deactivated:
            self.active[idx] = True
            self._deactivated.discard(idx)

    # ------------------------------------------------------------------
    # 同步（每 tick 末调用：位置同步 + 级联失效 + 红外残热）
    # ------------------------------------------------------------------

    def _parent_alive(
        self, idx: int, armor: ArmorPool, projectiles: ProjectilePool
    ) -> bool:
        pk = int(self.parent_kind[idx])
        pi = int(self.parent_idx[idx])
        if pk == EMITTER_PARENT_ARMOR:
            return pi >= 0 and bool(armor.active[pi])
        if pk == EMITTER_PARENT_PROJECTILE:
            return pi >= 0 and bool(projectiles.active[pi])
        return True  # 无父（光斑）：存活由 SAL 系统管理

    def sync_from_parent(
        self, armor: ArmorPool, projectiles: ProjectilePool, dt: float
    ) -> None:
        """位置同步 + 级联失效兜底 + 红外充能/残热衰减 + 归还 free_stack。"""
        _sync_emitters_kernel(
            self.pos, self.active, self.parent_kind, self.parent_idx,
            self.band, self.power_db, self.cur_power,
            armor.pos, armor.active,
            projectiles.pos, projectiles.active,
        )
        freed: list[int] = []
        for i in range(self.capacity):
            if not self.active[i]:
                # 战术关机保留索引；其余 inactive（级联失效/内核兜底）→ 归还
                if i in self._deactivated or i in self._free_set:
                    continue
                freed.append(i)
                continue
            if self.band[i] == _BAND_IR and not self._parent_alive(i, armor, projectiles):
                # 红外：父死亡 → 停止充能，残热指数衰减，低于阈值失活
                self.cur_power[i] *= np.exp(-self.decay_per_s[i] * float(dt))
                if self.cur_power[i] < self.IR_OFF_THRESHOLD:
                    self.active[i] = False
                    self._deactivated.discard(i)
                    freed.append(i)
        if freed:
            self._free_stack.extend(freed)
            self._free_set.update(freed)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return int(np.sum(self.active))

    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    # ------------------------------------------------------------------
    # 扩容（Tick 间隙调用）
    # ------------------------------------------------------------------

    def _expand(self, factor: float = 2.0) -> None:
        new_cap = max(int(self.capacity * factor), self.capacity + 1)

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.pos = _resize(self.pos, (new_cap, 3))
        self.power_db = _resize(self.power_db, (new_cap,))
        self.cur_power = _resize(self.cur_power, (new_cap,))
        self.decay_per_s = _resize(self.decay_per_s, (new_cap,))
        self.band = _resize(self.band, (new_cap,))
        self.kind = _resize(self.kind, (new_cap,))
        self.active = _resize(self.active, (new_cap,))
        self.parent_kind = _resize(self.parent_kind, (new_cap,))
        self.parent_idx = _resize(self.parent_idx, (new_cap,))
        self.normal = _resize(self.normal, (new_cap, 3))

        old_cap = self.capacity
        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))
        self._free_set.update(range(old_cap, new_cap))
