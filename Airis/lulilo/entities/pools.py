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


class ProjectileType(IntEnum):
    """抛射物类型。"""
    BULLET = 0   # 动能弹（APFSDS/AP）
    MISSILE = 1  # 导弹（ATGM）


# ---------------------------------------------------------------------------
# ArmorPool
# ---------------------------------------------------------------------------


class ArmorPool:
    """装甲单位池（SoA 预分配，MAX_ARMOR=2048）。

    数组布局:
        pos       (N, 3) float32  中心位置 (x, y, z)
        half      (N, 3) float32  碰撞箱半边长
        yaw       (N,)   float32  朝向角 (rad)
        health    (N,)   float32  当前血量
        active    (N,)   bool_    存活掩码
        wp        (N, 8, 2) float32  路径拐点（世界坐标）
        wp_count  (N,)   uint8    拐点数量
        wp_idx    (N,)   uint8    当前目标拐点索引
    """

    def __init__(self, capacity: int = MAX_ARMOR) -> None:
        self.capacity = int(capacity)

        # 主数组
        self.pos = np.zeros((self.capacity, 3), dtype=np.float32)
        self.half = np.zeros((self.capacity, 3), dtype=np.float32)
        self.yaw = np.zeros(self.capacity, dtype=np.float32)
        self.health = np.zeros(self.capacity, dtype=np.float32)
        self.active = np.zeros(self.capacity, dtype=np.bool_)

        # 路径系统
        self.wp = np.zeros((self.capacity, 8, 2), dtype=np.float32)
        self.wp_count = np.zeros(self.capacity, dtype=np.uint8)
        self.wp_idx = np.zeros(self.capacity, dtype=np.uint8)

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
        health: float = 100.0,
    ) -> int:
        """生成一辆装甲车，返回其池索引。池满则扩容后继续。

        Returns:
            装甲索引 (0..capacity-1)。
        """
        if not self._free_stack:
            self._expand()
        idx = self._free_stack.pop()
        self.pos[idx] = (x, y, z)
        self.half[idx] = (half_x, half_y, half_z)
        self.yaw[idx] = yaw
        self.health[idx] = health
        self.active[idx] = True
        # 清除残留路径
        self.wp[idx] = 0.0
        self.wp_count[idx] = 0
        self.wp_idx[idx] = 0
        return idx

    def destroy(self, idx: int) -> None:
        """销毁装甲（标记 inactive，归还索引）。"""
        self.active[idx] = False
        self._free_stack.append(idx)

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
        self.health = _resize(self.health, (new_cap,))
        self.active = _resize(self.active, (new_cap,))
        self.wp = _resize(self.wp, (new_cap, 8, 2))
        self.wp_count = _resize(self.wp_count, (new_cap,))
        self.wp_idx = _resize(self.wp_idx, (new_cap,))

        old_cap = self.capacity
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
    armor_active: np.ndarray,  # (N,) 父装甲存活
) -> None:
    """Numba 并行：把所有活跃碰撞箱从局部偏移变换到世界坐标。

    与 Python 版 sync_from_parent 同语义（每元素独立 → 无竞争），
    但全量并行——Phase 3 开火链每子弹 Tick 调用一次（16384 box 级 < 1ms）。
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
        box_pos[i, 0] = armor_pos[p, 0] + lx * cos_y - ly * sin_y
        box_pos[i, 1] = armor_pos[p, 1] + lx * sin_y + ly * cos_y
        box_pos[i, 2] = armor_pos[p, 2] + lz


class BoxPool:
    """碰撞箱池（SoA 预分配，MAX_BOXES=4096）。

    每辆装甲车由 2~6 个碰撞箱组成（车体/炮塔/引擎/履带/弹药架）。
    碰撞箱位置为世界坐标（由装甲 pos + yaw 变换后写入）。

    数组布局:
        pos       (N, 3) float32  世界坐标
        half      (N, 3) float32  局部半边长（在装甲局部坐标系中为 AABB）
        parent    (N,)   int32    所属装甲索引
        part      (N,)   int32    部位类型（PartType 枚举）
        active    (N,)   bool_    存活掩码
    """

    def __init__(self, capacity: int = MAX_BOXES) -> None:
        self.capacity = int(capacity)

        self.local = np.zeros((self.capacity, 3), dtype=np.float32)  # 局部偏移（权威）
        self.pos = np.zeros((self.capacity, 3), dtype=np.float32)    # 世界坐标（sync 后）
        self.half = np.zeros((self.capacity, 3), dtype=np.float32)
        self.parent = np.full(self.capacity, -1, dtype=np.int32)
        self.part = np.zeros(self.capacity, dtype=np.int32)
        self.active = np.zeros(self.capacity, dtype=np.bool_)

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
        return idx

    def destroy(self, idx: int) -> None:
        """销毁碰撞箱。"""
        self.active[idx] = False
        self.parent[idx] = -1
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
        # 2D 旋转 + 平移
        wx = armor.pos[p, 0] + lx * cos_y - ly * sin_y
        wy = armor.pos[p, 1] + lx * sin_y + ly * cos_y
        wz = armor.pos[p, 2] + lz
        self.pos[idx] = (wx, wy, wz)

    def sync_all_from_parent(self, armor: ArmorPool) -> None:
        """批量同步所有活跃碰撞箱（Numba 并行，幂等可重复调用）。"""
        _sync_boxes_kernel(
            self.local, self.parent, self.active, self.pos,
            armor.pos, armor.yaw, armor.active,
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

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.local = _resize(self.local, (new_cap, 3))
        self.pos = _resize(self.pos, (new_cap, 3))
        self.half = _resize(self.half, (new_cap, 3))
        self.parent = _resize(self.parent, (new_cap,))
        self.part = _resize(self.part, (new_cap,))
        self.active = _resize(self.active, (new_cap,))

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
        return idx

    def destroy(self, idx: int) -> None:
        """销毁抛射物（命中后或飞出边界）。"""
        self.active[idx] = False
        self.owner[idx] = -1
        self.weapon[idx] = -1
        self.ammo[idx] = -1
        self.age[idx] = 0.0
        self._free_stack.append(idx)

    def destroy_batch(self, indices: np.ndarray) -> None:
        """批量销毁（Numba 外部调用后的结果写入）。"""
        self.active[indices] = False
        self.owner[indices] = -1
        self.weapon[indices] = -1
        self.ammo[indices] = -1
        self.age[indices] = 0.0
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

        old_cap = self.capacity
        self.capacity = new_cap
        self._free_stack.extend(range(old_cap, new_cap))
