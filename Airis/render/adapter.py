"""WorldAdapter：渲染数据源抽象层（Phase 5 新增）。

对应 docs/phase5-ui-pipeline-design.md §2.2：
    - Phase 5：透传 SimulationSnapshot 的 SoA 数组，所有 confidence=1.0（上帝视角）
    - Phase 8：切换为 Twin 数组，含传感器融合 + 时间衰减

核心约定（docs §2.0 法则）：
    GLWidget / HUD / DetailPanel 只通过本适配器读取数据，
    绝不直接访问 SimulationSnapshot.pools 或底层 SoA 数组。
    今日它是 Real 数据的透传代理；未来切换为 Twin 数据源时，
    下游消费者零改动。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from Airis.lulilo.entities.pools import PartType
from Airis.lulilo.entities.side import SideModelMapper, SideNameMapper
from Airis.lulilo.simulation.state import SimulationSnapshot

logger = logging.getLogger(__name__)

# 部位名称（对应 PartType 顺序，与 armor_damaged_parts 的 5 列一一对应）
PART_NAMES = ("车体", "炮塔", "引擎", "履带", "弹药架")

# 阵营 → 单位型号（统一走 entities.side SideModelMapper，唯一数据源；
# 不在这里硬编码型号名；Phase 8 接入 per-unit name 字段后替换）
_SIDE_MODEL = SideModelMapper()
# 阵营显示名统一走 entities.side（SideNameMapper：0=北约 1=华约 2=中立）
_SIDE_NAMES = SideNameMapper()

# ---------------------------------------------------------------------------
# 选择句柄编码（2026-08-18：拾取同时覆盖地面装甲与空中单位）
# ---------------------------------------------------------------------------
# 地面装甲索引直接作 handle（0..MAX_ARMOR-1）；空中单位编码为
# AIR_HANDLE_OFFSET + aircraft_idx（aircraft_idx < 128 << 偏移 1<<20，无冲突）。
# GLWidget hover/selected / DetailPanel 一律用 handle，解码后分流查询。
AIR_HANDLE_OFFSET = 1 << 20


def is_air_handle(handle: int) -> bool:
    """handle 是否编码了空中单位。"""
    return int(handle) >= AIR_HANDLE_OFFSET


def decode_handle(handle: int) -> tuple[int, int]:
    """解码选择句柄 → (kind, idx)；kind 0=装甲 1=空中。handle<0 → (-1,-1)。"""
    h = int(handle)
    if h < 0:
        return -1, -1
    if h >= AIR_HANDLE_OFFSET:
        return 1, h - AIR_HANDLE_OFFSET
    return 0, h


class UnitRenderStatus(IntEnum):
    """渲染用单位状态（Phase 5 全为 ACTIVE；Phase 8 扩展 UNKNOWN/KIA/MIA）。"""

    ACTIVE = 0
    UNKNOWN = 1   # Phase 8：变成问号
    KIA = 2       # Phase 8：确认击毁
    MIA = 3       # Phase 8：信号丢失


# Phase 8：孪生状态 + 置信度 → 渲染状态（docs §4.1 映射表）
def map_twin_render_status(status: int, confidence: float) -> UnitRenderStatus:
    """孪生状态 → 渲染状态（TwinStatus + confidence 分档，docs §4.1）。

    | TwinStatus | confidence | 渲染 |
    |---|---|---|
    | ACTIVE | > 0.8 | ACTIVE（完整模型 + 高亮） |
    | ACTIVE | 0.5~0.8 | ACTIVE（半透明 + 抖动，pos_err 驱动） |
    | ACTIVE | < 0.5 | UNKNOWN（模糊接触框） |
    | UNKNOWN | — | UNKNOWN（「?」标记） |
    | KIA | — | KIA（残骸 + 红叉） |
    | MIA | — | MIA（灰色轮廓 + 闪烁） |
    """
    from Airis.lulilo.entities.twin import TwinStatus

    st = int(status)
    if st == int(TwinStatus.KIA):
        return UnitRenderStatus.KIA
    if st == int(TwinStatus.MIA):
        return UnitRenderStatus.MIA
    if st == int(TwinStatus.UNKNOWN):
        return UnitRenderStatus.UNKNOWN
    # ACTIVE：按置信度分档（< 0.5 降为模糊接触框）
    if confidence < 0.5:
        return UnitRenderStatus.UNKNOWN
    return UnitRenderStatus.ACTIVE


def _build_damaged_parts(
    armor: Any,
    n_armor: int,
) -> np.ndarray:
    """聚合每个装甲单位的 5 部位损坏掩码 (N,5) bool（PartType 顺序）。

    数据源 = ArmorPool.parts_destroyed（2026-08-18 起）：
        - 毁伤是**单位状态**，由模拟层（engagement._apply_part_damage /
          _destroy_armor）在部位血归零/整车摧毁时置位；
        - 不再靠 box 槽位（parent/active）推断——部位毁伤会释放槽位
          （parent 置 -1、回 free_stack），旧推断逻辑永远看不到已毁部位。

    Args:
        armor: ArmorPool（可为 None，骨架期未装配）。
        n_armor: 装甲池容量。

    Returns:
        (n_armor, len(PART_NAMES)) bool 数组；True = 该部位已损坏。
    """
    damaged = np.zeros((n_armor, len(PART_NAMES)), dtype=np.bool_)
    if armor is None or n_armor == 0:
        return damaged
    parts = getattr(armor, "parts_destroyed", None)
    if parts is not None:
        m = min(n_armor, parts.shape[0])
        damaged[:m] = parts[:m, : len(PART_NAMES)]
    return damaged


def _aircraft_views(snapshot: SimulationSnapshot) -> dict[str, np.ndarray]:
    """提取空中单位视图；未装配 → 全空数组。

    Phase 7 演示：AircraftPool 可选装配；HUD 用阵营色小圆点标记飞机，
    拾取/详情面板按 handle 查询（aircraft_name/speed/ammo/fuel 等）。
    所有字段 getattr 兜底（旧快照/未装配不崩）。
    """
    ac = snapshot.pools.get("aircraft")
    if ac is None:
        return {
            "pos": np.zeros((0, 3), dtype=np.float32),
            "vel": np.zeros((0, 3), dtype=np.float32),
            "active": np.zeros(0, dtype=np.bool_),
            "side": np.zeros(0, dtype=np.uint8),
            "health": np.zeros(0, dtype=np.float32),
            "fuel": np.zeros(0, dtype=np.float32),
            "aam": np.zeros(0, dtype=np.uint8),
            "agm": np.zeros(0, dtype=np.uint8),
            "state": np.zeros(0, dtype=np.uint8),
            "role": np.zeros(0, dtype=np.uint8),
            "formation_id": np.zeros(0, dtype=np.int32),
        }
    cap = ac.capacity

    def _get(name: str, default: np.ndarray) -> np.ndarray:
        arr = getattr(ac, name, None)
        return arr if arr is not None else default

    return {
        "pos": ac.pos,
        "vel": _get("vel", np.zeros((cap, 3), dtype=np.float32)),
        "active": ac.active,
        "side": _get("side", np.zeros(cap, dtype=np.uint8)),
        "health": _get("health", np.full(cap, 100.0, dtype=np.float32)),
        "fuel": _get("fuel", np.zeros(cap, dtype=np.float32)),
        "aam": _get("aam_left", np.zeros(cap, dtype=np.uint8)),
        "agm": _get("agm_left", np.zeros(cap, dtype=np.uint8)),
        "state": _get("state", np.zeros(cap, dtype=np.uint8)),
        "role": _get("role", np.zeros(cap, dtype=np.uint8)),
        "formation_id": _get("formation_id", np.full(cap, -1, dtype=np.int32)),
    }


def _projectile_views(snapshot: SimulationSnapshot) -> dict[str, np.ndarray]:
    """提取抛射物视图；未装配 → 全空数组。

    抛射体/导弹可视化：渲染层只读，经 WorldAdapter.projectile_markers 消费。
    所有字段 getattr 兜底（旧快照/未装配不崩）。
    """
    pr = snapshot.pools.get("projectile")
    if pr is None:
        return {
            "pos": np.zeros((0, 3), dtype=np.float32),
            "vel": np.zeros((0, 3), dtype=np.float32),
            "active": np.zeros(0, dtype=np.bool_),
            "owner": np.zeros(0, dtype=np.int32),
            "owner_kind": np.zeros(0, dtype=np.int8),
            "guid_law": np.zeros(0, dtype=np.int32),
            "guid_confidence": np.zeros(0, dtype=np.float32),
        }
    cap = pr.capacity

    def _get(name: str, default: np.ndarray) -> np.ndarray:
        arr = getattr(pr, name, None)
        return arr if arr is not None else default

    return {
        "pos": pr.pos,
        "vel": pr.vel,
        "active": pr.active,
        "owner": _get("owner", np.full(cap, -1, dtype=np.int32)),
        "owner_kind": _get("owner_kind", np.zeros(cap, dtype=np.int8)),
        "guid_law": _get("guid_law", np.full(cap, -1, dtype=np.int32)),
        "guid_confidence": _get("guid_confidence", np.ones(cap, dtype=np.float32)),
    }


# formation 池字段名映射（详情面板编队 AI 状态）
_FORMATION_FIELDS = (
    ("formation_intent", "intent", np.int8),
    ("formation_order", "order", np.int8),
    ("formation_stance", "stance", np.int8),
    ("formation_comm", "comm", np.float32),
    ("formation_member_count", "member_count", np.int32),
)


def _formation_views(
    snapshot: SimulationSnapshot, n_armor: int
) -> dict[str, np.ndarray | None]:
    """提取编队 AI 视图（详情面板显示意图/命令/通信质量）；未装配 → None。"""
    armor = snapshot.pools.get("armor")
    order = getattr(armor, "order", None) if armor is not None else None
    fid = getattr(armor, "formation_id", None) if armor is not None else None
    views: dict[str, np.ndarray | None] = {
        "armor_order": (
            order if order is not None else np.zeros(n_armor, dtype=np.uint8)
        ),
        "armor_formation_id": (
            fid if fid is not None else np.full(n_armor, -1, dtype=np.int32)
        ),
    }
    f = snapshot.pools.get("formation")
    for view_key, pool_key, dtype in _FORMATION_FIELDS:
        arr = getattr(f, pool_key, None) if f is not None else None
        if arr is None:
            views[view_key] = None if f is None else np.zeros(f.capacity, dtype=dtype)
        else:
            views[view_key] = arr
    return views


@dataclass
class WorldAdapter:
    """渲染数据源抽象层（Phase 5 透传模式）。

    Phase 5：直接引用 Real SoA 数组（浅视图，避免每帧深拷贝），
    只读语义由约定保证；所有 confidence=1.0、status=ACTIVE。
    Phase 8：切换为 Twin 数组（传感器融合 + 时间衰减）。

    视图引用数组为**只读**——下游消费者不得写入。
    """

    # ── 视图引用（Phase 5：直指 Real 数组；Phase 8：指 Twin 数组）──
    armor_pos: np.ndarray           # (N, 3) float32 世界位置
    armor_yaw: np.ndarray           # (N,)   float32 朝向角
    armor_pitch: np.ndarray         # (N,)   float32 俯仰角（悬挂，绕车身横轴）
    armor_roll: np.ndarray          # (N,)   float32 侧倾角（悬挂，绕车身纵轴）
    armor_health: np.ndarray        # (N,)   float32 血量
    armor_active: np.ndarray        # (N,)   bool   存活掩码
    armor_confidence: np.ndarray    # (N,)   float32 置信度（Phase 5 恒为 1.0）
    armor_status: np.ndarray        # (N,)   uint8   渲染状态（Phase 5 恒为 ACTIVE）
    armor_side: np.ndarray          # (N,)   uint8   绝对阵营（Side：0=北约 1=华约 2=中立）
    armor_damaged_parts: np.ndarray  # (N, 5) bool   每单位 5 部位损坏掩码（PartType 顺序）

    # ── 部件详情（命中联动，HUD/详情面板读取）──
    box_parent: np.ndarray          # (M,) int32 碰撞箱所属装甲
    box_part: np.ndarray            # (M,) int32 部位类型（PartType）
    box_active: np.ndarray          # (M,) bool  部件存活（False=已损坏）
    # ── 单位角色（型号命名按 role 细分：炮兵 ≠ MBT；None = 未装配 → 按坦克默认）──
    armor_role: np.ndarray | None = None  # (N,) uint8 UnitRole（0=坦克 1=炮兵 …）
    # ── 部件装甲厚度（详情面板逐部位显示；0=未设置/已毁 → 不显示数值）──
    box_armor_mm: np.ndarray | None = None  # (M,) float32 部位标称装甲厚度 (mm RHA)
    # ── 单位命令/编队 AI 状态视图（None = 未装配；详情面板显示）──
    armor_order: np.ndarray | None = None          # (N,) uint8 UnitOrder
    armor_formation_id: np.ndarray | None = None   # (N,) int32 编队索引（-1 无）
    formation_intent: np.ndarray | None = None     # (F,) int8 FormationIntent
    formation_order: np.ndarray | None = None      # (F,) int8 编队级命令
    formation_stance: np.ndarray | None = None     # (F,) int8 战术姿态
    formation_comm: np.ndarray | None = None       # (F,) float32 通信质量 0~1
    formation_member_count: np.ndarray | None = None  # (F,) int32 成员数
    # ── 后勤视图（可选；None = 未装配）──
    ammo_reserve: np.ndarray | None = None    # (N,) float32 当前弹药
    ammo_capacity: np.ndarray | None = None   # (N,) float32 弹药上限
    fuel_reserve: np.ndarray | None = None    # (N,) float32 当前燃料
    fuel_capacity: np.ndarray | None = None   # (N,) float32 燃料上限
    # ── 速度视图（可选；None = 未估计，speed_at 回退 0）──
    # Phase 5e：渲染层位置差分估计（EMA 平滑），引擎侧无 velocity 字段
    armor_speed: np.ndarray | None = None     # (N,) float32 水平速度 m/s
    # ── 空中视图（Phase 7 演示；未装配 = 空数组）──
    aircraft_pos: np.ndarray = None           # (K,3) float32 空中单位世界位置
    aircraft_active: np.ndarray = None        # (K,)   bool   存活掩码
    aircraft_side: np.ndarray = None          # (K,)   uint8  阵营（Side）
    aircraft_vel: np.ndarray = None           # (K,3) float32 速度向量（aircraft_speed_at）
    aircraft_health: np.ndarray = None        # (K,)   float32 血量
    aircraft_fuel: np.ndarray = None          # (K,)   float32 归一化燃料 0~1
    aircraft_aam: np.ndarray = None           # (K,)   uint8  AAM 剩余
    aircraft_agm: np.ndarray = None           # (K,)   uint8  AGM 剩余
    aircraft_state: np.ndarray = None         # (K,)   uint8  AircraftState
    aircraft_role: np.ndarray = None          # (K,)   uint8  AircraftRole
    aircraft_formation_id: np.ndarray = None  # (K,)   int32  编队索引
    # ── 抛射物视图（抛射体/导弹可视化；未装配 = None）──
    proj_pos: np.ndarray | None = None          # (P,3) float32 抛射物世界位置
    proj_vel: np.ndarray | None = None          # (P,3) float32 速度向量（导弹尾迹）
    proj_active: np.ndarray | None = None       # (P,)   bool   存活掩码
    proj_owner: np.ndarray | None = None        # (P,)   int32  发射者索引（按 owner_kind 分池）
    proj_owner_kind: np.ndarray | None = None   # (P,)   int8   0=装甲 1=飞机
    proj_guid_law: np.ndarray | None = None     # (P,)   int32  -1=无制导 >=0=制导（导弹）
    proj_guid_confidence: np.ndarray | None = None  # (P,) float32 友军知晓置信度 0~1
    sim_time: float = 0.0
    tick_count: int = 0

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SimulationSnapshot,
        armor_speed: np.ndarray | None = None,
    ) -> "WorldAdapter":
        """从模拟快照构造（Phase 5 透传模式）。

        Args:
            snapshot: 模拟快照。
            armor_speed: 可选的水平速度数组 (N,) float32
                （Phase 5e：渲染层位置差分估计，引擎无 velocity 字段）。

        Note:
            - 直接引用 SoA 数组，不拷贝（浅视图）
            - confidence 恒 1.0 / status 恒 ACTIVE（上帝视角）
            - armor.side：绝对阵营 SoA 字段（entities/side.py Side 枚举）；
              通过 getattr 读取，缺失时全 0（北约）
        """
        armor = snapshot.pools.get("armor")
        box = snapshot.pools.get("box")
        n = armor.capacity if armor is not None else 0

        ammo_mgr = snapshot.logistics.get("ammo")
        fuel_mgr = snapshot.logistics.get("fuel")
        ammo_st = ammo_mgr.state if ammo_mgr is not None else None
        fuel_st = fuel_mgr.state if fuel_mgr is not None else None

        if armor is None:
            empty3 = np.zeros((0, 3), dtype=np.float32)
            empty1 = np.zeros(0, dtype=np.float32)
            emptyb = np.zeros(0, dtype=np.bool_)
            av = _aircraft_views(snapshot)
            pv = _projectile_views(snapshot)
            return cls(
                armor_pos=empty3,
                armor_yaw=empty1,
                armor_pitch=np.zeros(0, dtype=np.float32),
                armor_roll=np.zeros(0, dtype=np.float32),
                armor_health=empty1,
                armor_active=emptyb,
                armor_confidence=np.zeros(0, dtype=np.float32),
                armor_status=np.zeros(0, dtype=np.uint8),
                armor_side=np.zeros(0, dtype=np.uint8),
                armor_damaged_parts=np.zeros((0, len(PartType)), dtype=np.bool_),
                armor_role=np.zeros(0, dtype=np.uint8),
                box_parent=np.zeros(0, dtype=np.int32),
                box_part=np.zeros(0, dtype=np.int32),
                box_active=emptyb,
                box_armor_mm=np.zeros(0, dtype=np.float32),
                aircraft_pos=av["pos"],
                aircraft_active=av["active"],
                aircraft_side=av["side"],
                aircraft_vel=av["vel"],
                aircraft_health=av["health"],
                aircraft_fuel=av["fuel"],
                aircraft_aam=av["aam"],
                aircraft_agm=av["agm"],
                aircraft_state=av["state"],
                aircraft_role=av["role"],
                aircraft_formation_id=av["formation_id"],
                proj_pos=pv["pos"],
                proj_vel=pv["vel"],
                proj_active=pv["active"],
                proj_owner=pv["owner"],
                proj_owner_kind=pv["owner_kind"],
                proj_guid_law=pv["guid_law"],
                proj_guid_confidence=pv["guid_confidence"],
                sim_time=snapshot.sim_time,
                tick_count=snapshot.tick_count,
            )

        side = getattr(armor, "side", None)
        if side is None:
            side = np.zeros(n, dtype=np.uint8)  # 缺失兜底：全北约
        role = getattr(armor, "role", None)
        if role is None:
            role = np.zeros(n, dtype=np.uint8)  # 缺失兜底：全坦克
        pitch = getattr(armor, "pitch", None)
        if pitch is None:
            pitch = np.zeros(n, dtype=np.float32)
        roll = getattr(armor, "roll", None)
        if roll is None:
            roll = np.zeros(n, dtype=np.float32)

        av = _aircraft_views(snapshot)
        pv = _projectile_views(snapshot)
        fv = _formation_views(snapshot, n)
        return cls(
            armor_pos=armor.pos,
            armor_yaw=armor.yaw,
            armor_pitch=pitch,
            armor_roll=roll,
            armor_health=(
                box.hp[armor.hull_idx]  # 哨兵槽：-1 → 0，安全（part-health-design.md §2.1）
                if box is not None
                else np.zeros(n, dtype=np.float32)
            ),
            armor_active=armor.active,
            armor_confidence=np.ones(n, dtype=np.float32),   # 上帝视角
            armor_status=np.zeros(n, dtype=np.uint8),         # 全 ACTIVE
            armor_side=side,
            armor_role=role,
            armor_damaged_parts=_build_damaged_parts(armor, n),
            box_parent=box.parent if box is not None else np.zeros(0, dtype=np.int32),
            box_part=box.part if box is not None else np.zeros(0, dtype=np.int32),
            box_active=box.active if box is not None else np.zeros(0, dtype=np.bool_),
            box_armor_mm=(
                box.armor_mm if box is not None else np.zeros(0, dtype=np.float32)
            ),
            armor_order=fv["armor_order"],
            armor_formation_id=fv["armor_formation_id"],
            formation_intent=fv["formation_intent"],
            formation_order=fv["formation_order"],
            formation_stance=fv["formation_stance"],
            formation_comm=fv["formation_comm"],
            formation_member_count=fv["formation_member_count"],
            ammo_reserve=ammo_st.reserve if ammo_st is not None else None,
            ammo_capacity=ammo_st.capacity if ammo_st is not None else None,
            fuel_reserve=fuel_st.reserve if fuel_st is not None else None,
            fuel_capacity=fuel_st.capacity if fuel_st is not None else None,
            armor_speed=armor_speed,
            aircraft_pos=av["pos"],
            aircraft_active=av["active"],
            aircraft_side=av["side"],
            aircraft_vel=av["vel"],
            aircraft_health=av["health"],
            aircraft_fuel=av["fuel"],
            aircraft_aam=av["aam"],
            aircraft_agm=av["agm"],
            aircraft_state=av["state"],
            aircraft_role=av["role"],
            aircraft_formation_id=av["formation_id"],
            proj_pos=pv["pos"],
            proj_vel=pv["vel"],
            proj_active=pv["active"],
            proj_owner=pv["owner"],
            proj_owner_kind=pv["owner_kind"],
            proj_guid_law=pv["guid_law"],
            proj_guid_confidence=pv["guid_confidence"],
            sim_time=snapshot.sim_time,
            tick_count=snapshot.tick_count,
        )

    # ------------------------------------------------------------------
    # 查询接口（下游消费者唯一入口）
    # ------------------------------------------------------------------

    @classmethod
    def from_snapshot_twin(
        cls,
        snapshot: SimulationSnapshot,
        twin: Any,
    ) -> "WorldAdapter":
        """从模拟快照 + 认知孪生构造（Phase 8 战争迷雾模式，docs §4.1）。

        - 单位位置 = `twin.pos`（带误差，真值不进 adapter）；
        - 渲染状态 = `map_twin_render_status(twin.status, twin.confidence)`
          （ACTIVE/UNKNOWN/KIA/MIA 映射）；
        - 血量 = `twin.health`（报告值，可能滞后）；
        - 活跃掩码 = 孪生 active（残骸/失联单位仍渲染为 KIA/MIA 状态）。
        物理装甲数组（pos/yaw/pitch/roll/active）仍来自 armor（只读引用，
        供碰撞/选择联动；渲染位置以 twin.pos 为准）。
        """
        armor = snapshot.pools.get("armor")
        box = snapshot.pools.get("box")
        n = armor.capacity if armor is not None else 0

        ammo_mgr = snapshot.logistics.get("ammo")
        fuel_mgr = snapshot.logistics.get("fuel")
        ammo_st = ammo_mgr.state if ammo_mgr is not None else None
        fuel_st = fuel_mgr.state if fuel_mgr is not None else None

        if armor is None or twin is None:
            return cls.from_snapshot(snapshot)

        side = getattr(armor, "side", None)
        if side is None:
            side = np.zeros(n, dtype=np.uint8)
        role = getattr(armor, "role", None)
        if role is None:
            role = np.zeros(n, dtype=np.uint8)
        pitch = getattr(armor, "pitch", None)
        if pitch is None:
            pitch = np.zeros(n, dtype=np.float32)
        roll = getattr(armor, "roll", None)
        if roll is None:
            roll = np.zeros(n, dtype=np.float32)

        # 孪生视图（armor 槽位 = twin[0..n)）
        twin_confidence = np.zeros(n, dtype=np.float32)
        twin_status = np.zeros(n, dtype=np.uint8)
        twin_health = np.zeros(n, dtype=np.float32)
        if twin.capacity >= n:
            twin_confidence[:n] = twin.confidence[:n]
            twin_status[:n] = twin.status[:n]
            twin_health[:n] = twin.health[:n]

        # 渲染状态映射（向量化）
        render_status = np.array([
            int(map_twin_render_status(int(st), float(cf)))
            for st, cf in zip(twin_status, twin_confidence)
        ], dtype=np.uint8)

        av = _aircraft_views(snapshot)
        pv = _projectile_views(snapshot)
        fv = _formation_views(snapshot, n)
        return cls(
            armor_pos=twin.pos[:n] if twin.capacity >= n else armor.pos,
            armor_yaw=armor.yaw,
            armor_pitch=pitch,
            armor_roll=roll,
            armor_health=twin_health,          # 报告血量（滞后）
            armor_active=armor.active,         # 物理存活（渲染残骸/失联依赖 status）
            armor_confidence=twin_confidence,
            armor_status=render_status,
            armor_side=side,
            armor_role=role,
            armor_damaged_parts=_build_damaged_parts(armor, n),
            box_parent=box.parent if box is not None else np.zeros(0, dtype=np.int32),
            box_part=box.part if box is not None else np.zeros(0, dtype=np.int32),
            box_active=box.active if box is not None else np.zeros(0, dtype=np.bool_),
            box_armor_mm=(
                box.armor_mm if box is not None else np.zeros(0, dtype=np.float32)
            ),
            armor_order=fv["armor_order"],
            armor_formation_id=fv["armor_formation_id"],
            formation_intent=fv["formation_intent"],
            formation_order=fv["formation_order"],
            formation_stance=fv["formation_stance"],
            formation_comm=fv["formation_comm"],
            formation_member_count=fv["formation_member_count"],
            ammo_reserve=ammo_st.reserve if ammo_st is not None else None,
            ammo_capacity=ammo_st.capacity if ammo_st is not None else None,
            fuel_reserve=fuel_st.reserve if fuel_st is not None else None,
            fuel_capacity=fuel_st.capacity if fuel_st is not None else None,
            armor_speed=None,
            aircraft_pos=av["pos"],
            aircraft_active=av["active"],
            aircraft_side=av["side"],
            aircraft_vel=av["vel"],
            aircraft_health=av["health"],
            aircraft_fuel=av["fuel"],
            aircraft_aam=av["aam"],
            aircraft_agm=av["agm"],
            aircraft_state=av["state"],
            aircraft_role=av["role"],
            aircraft_formation_id=av["formation_id"],
            proj_pos=pv["pos"],
            proj_vel=pv["vel"],
            proj_active=pv["active"],
            proj_owner=pv["owner"],
            proj_owner_kind=pv["owner_kind"],
            proj_guid_law=pv["guid_law"],
            proj_guid_confidence=pv["guid_confidence"],
            sim_time=snapshot.sim_time,
            tick_count=snapshot.tick_count,
        )

    @property
    def active_mask(self) -> np.ndarray:
        """活跃单位掩码（Phase 5：等价于 armor_active）。"""
        return self.armor_active

    def unit_count(self) -> int:
        """当前活跃单位数。"""
        return int(np.count_nonzero(self.armor_active))

    def damaged_parts(self, unit_idx: int) -> list[str]:
        """返回某单位已损坏的部位名列表（空列表 = 完好）。

        Phase 5 直接来自 BoxPool 部位毁伤；Phase 8 叠加孪生置信度
        （低置信时不展示或标记「推测」）。
        """
        if (
            unit_idx < 0
            or unit_idx >= self.armor_damaged_parts.shape[0]
            or not self.armor_active[unit_idx]
        ):
            return []
        mask = self.armor_damaged_parts[unit_idx]
        return [PART_NAMES[i] for i, ok in enumerate(mask) if ok]

    def present_parts(self, unit_idx: int) -> list[str]:
        """该单位**实际装配**的部位名列表（详情面板逐项显示的依据）。

        2026-08-18 修复：此前详情面板固定遍历 5 个 PART_NAMES，未毁即显示绿 ✓，
        炮兵只装配 HULL/TURRET 也会显示「✓ 引擎 ✓ 履带 ✓ 弹药架」。
        现在按「已毁单位标记 ∪ 当前活跃 box 部位」聚合真实装配部位：
            - 未毁部位：box 槽位仍存活（active & parent==unit）→ 从 box 聚合；
            - 已毁部位：槽位已释放（parent=-1）→ 由 armor_damaged_parts 补回
              （毁伤是单位状态，见 _build_damaged_parts）；
            - 整车毁（残骸）：parts_destroyed 全置位 → 显示全毁（合理残骸语义）。

        Returns:
            部位名列表（PART_NAMES 顺序去重）；单位不可用/无部位数据 → 空列表。
        """
        if (
            unit_idx < 0
            or unit_idx >= self.armor_pos.shape[0]
            or not self.armor_active[unit_idx]
        ):
            return []
        present_idx: set[int] = set()
        # 1) 已毁部位（单位状态；槽位已释放，只能从这里追溯）
        if unit_idx < self.armor_damaged_parts.shape[0]:
            for p, ok in enumerate(self.armor_damaged_parts[unit_idx]):
                if ok and p < len(PART_NAMES):
                    present_idx.add(p)
        # 2) 当前活跃 box 的部位（未毁 → 槽位仍存活）
        if (
            self.box_parent is not None
            and len(self.box_parent)
            and self.box_active is not None
        ):
            m = (self.box_parent == unit_idx) & self.box_active
            for b in np.flatnonzero(m):
                pi = int(self.box_part[b])
                if 0 <= pi < len(PART_NAMES):
                    present_idx.add(pi)
        # 按 PART_NAMES 顺序输出（非热路径，集合去重可接受）
        return [PART_NAMES[p] for p in sorted(present_idx)]

    def part_armor_mm(self, unit_idx: int, part_name: str) -> float | None:
        """单位某部位的标称装甲厚度 (mm RHA)。

        2026-08-18：详情面板逐部位显示装甲情况。数据源 = BoxPool.armor_mm
        （部位自身厚度；0=未设置/槽位已毁释放 → None，调用方显示「—」）。
        """
        if (
            unit_idx < 0
            or self.box_parent is None
            or len(self.box_parent) == 0
            or self.box_armor_mm is None
            or part_name not in PART_NAMES
        ):
            return None
        pi = PART_NAMES.index(part_name)
        m = (self.box_parent == unit_idx) & self.box_active & (self.box_part == pi)
        idxs = np.flatnonzero(m)
        if idxs.size == 0:
            return None
        mm = float(self.box_armor_mm[idxs[0]])
        return mm if mm > 0.0 else None

    def unit_formation_status(self, unit_idx: int) -> dict[str, Any] | None:
        """单位所属编队的 AI 状态（详情面板「编队 AI」区）。

        数据源 = ArmorPool.formation_id + FormationAIPool（intent/order/stance/comm）。
        无编队 / 未装配 → None。
        """
        if self.armor_formation_id is None or unit_idx < 0:
            return None
        if unit_idx >= len(self.armor_formation_id):
            return None
        fid = int(self.armor_formation_id[unit_idx])
        if fid < 0 or self.formation_intent is None or fid >= len(self.formation_intent):
            return None
        comm = (
            float(self.formation_comm[fid])
            if self.formation_comm is not None and fid < len(self.formation_comm)
            else 1.0
        )
        members = (
            int(self.formation_member_count[fid])
            if self.formation_member_count is not None and fid < len(self.formation_member_count)
            else 0
        )
        return {
            "formation_id": fid,
            "intent": int(self.formation_intent[fid]),
            "order": (
                int(self.formation_order[fid])
                if self.formation_order is not None and fid < len(self.formation_order)
                else -1
            ),
            "stance": (
                int(self.formation_stance[fid])
                if self.formation_stance is not None and fid < len(self.formation_stance)
                else -1
            ),
            "comm": comm,
            "member_count": members,
        }

    # ------------------------------------------------------------------
    # 空中单位查询（handle = AIR_HANDLE_OFFSET + aircraft_idx；见 decode_handle）
    # ------------------------------------------------------------------

    def _aircraft_arrays_safe(self, handle: int) -> tuple[int, dict[str, np.ndarray]] | None:
        """校验 handle 是否为活跃空中单位，返回 (aircraft_idx, 视图 dict)。"""
        kind, idx = decode_handle(handle)
        if kind != 1:
            return None
        if (
            self.aircraft_active is None
            or idx < 0
            or idx >= len(self.aircraft_active)
            or not bool(self.aircraft_active[idx])
        ):
            return None
        return idx, {
            "pos": self.aircraft_pos,
            "vel": self.aircraft_vel,
            "side": self.aircraft_side,
            "health": self.aircraft_health,
            "fuel": self.aircraft_fuel,
            "aam": self.aircraft_aam,
            "agm": self.aircraft_agm,
            "state": self.aircraft_state,
            "role": self.aircraft_role,
            "formation_id": self.aircraft_formation_id,
        }

    def aircraft_name(self, handle: int) -> str:
        """空中单位展示名（阵营 × 空中 role → 型号，SideModelMapper 单一数据源）。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return "#?"
        idx, v = got
        side = int(v["side"][idx]) if v["side"] is not None else 0
        model = _SIDE_MODEL(side, 5)  # UnitRole.AIR=5（空中默认型号）
        return f"{model} #{idx:02d}"

    def aircraft_side_name(self, handle: int) -> str:
        """空中单位阵营显示名。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return "未知"
        idx, v = got
        side = int(v["side"][idx]) if v["side"] is not None else 0
        return _SIDE_NAMES(side)

    def aircraft_speed_at(self, handle: int) -> float:
        """空中单位水平速度 (m/s)。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return 0.0
        idx, v = got
        vel = v["vel"]
        if vel is None or idx >= len(vel):
            return 0.0
        return float(np.hypot(vel[idx, 0], vel[idx, 1]))

    def aircraft_alt_at(self, handle: int) -> float:
        """空中单位高度 (m)。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return 0.0
        idx, v = got
        pos = v["pos"]
        if pos is None or idx >= len(pos):
            return 0.0
        return float(pos[idx, 2])

    def aircraft_health_at(self, handle: int) -> float | None:
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return None
        idx, v = got
        h = v["health"]
        return float(h[idx]) if h is not None and idx < len(h) else None

    def aircraft_ammo_at(self, handle: int) -> tuple[int, int] | None:
        """(AAM 剩余, AGM 剩余)。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return None
        idx, v = got
        aam = v["aam"]
        agm = v["agm"]
        a = int(aam[idx]) if aam is not None and idx < len(aam) else 0
        g = int(agm[idx]) if agm is not None and idx < len(agm) else 0
        return a, g

    def aircraft_fuel_at(self, handle: int) -> float | None:
        """归一化燃料 0~1 → 百分比。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return None
        idx, v = got
        f = v["fuel"]
        return float(f[idx]) * 100.0 if f is not None and idx < len(f) else None

    def aircraft_state_at(self, handle: int) -> int | None:
        """AircraftState 枚举值。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return None
        idx, v = got
        s = v["state"]
        return int(s[idx]) if s is not None and idx < len(s) else None

    def aircraft_formation_status(self, handle: int) -> dict[str, Any] | None:
        """空中编队状态（复用 formation 池视图；空中编队在 air_formation 池，
        暂无 intent/comm 视图 → 返回 formation_id + 成员数）。"""
        got = self._aircraft_arrays_safe(handle)
        if got is None:
            return None
        idx, v = got
        fid_arr = v["formation_id"]
        if fid_arr is None or idx >= len(fid_arr):
            return None
        fid = int(fid_arr[idx])
        return {"formation_id": fid, "intent": None, "order": None,
                "stance": None, "comm": None, "member_count": None}

    def get_render_data(self) -> dict[str, Any]:
        """返回 GLWidget / HUD 需要的渲染数据。

        Phase 8 扩展：根据 confidence/status 附加不同渲染模式标记。
        """
        return {
            "pos": self.armor_pos,
            "yaw": self.armor_yaw,
            "pitch": self.armor_pitch,
            "roll": self.armor_roll,
            "health": self.armor_health,
            "active": self.armor_active,
            "side": self.armor_side,
            "damaged": self.armor_damaged_parts,
            "confidence": self.armor_confidence,   # Phase 8：战争迷雾（0~1）
            "status": self.armor_status,           # Phase 8：UnitRenderStatus（迷雾视觉）
            "count": self.unit_count(),
            "sim_time": self.sim_time,
            "tick_count": self.tick_count,
        }

    # ------------------------------------------------------------------
    # 抛射体 / 导弹可视化（渲染层唯一入口）
    # ------------------------------------------------------------------

    def projectile_markers(
        self,
        player_side: int = 0,
        confidence_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回可绘制抛射物：(pos, vel, is_missile, is_friendly)。

        规则：
            - 己方抛射物（owner 阵营 == player_side）：全绘（含子弹与导弹）。
            - 敌方导弹（guid_law >= 0 且 owner 阵营 != player_side）：
              `proj_guid_confidence >= confidence_threshold` 才绘——
              Phase 8 战争迷雾接入后仅显示「已探测」导弹；上帝视角下
              guid_confidence 恒 1.0 → 全绘。
            - 敌方无制导弹（子弹/炮弹）：不绘（太快太小，且不可探测）。

        Args:
            player_side: 玩家绝对阵营（Side；演示默认北约 0）。
            confidence_threshold: 敌方导弹可见的置信度门限。

        Returns:
            (pos (K,3) float32, vel (K,3) float32, is_missile (K,) bool,
             is_friendly (K,) bool)
        """
        empty3 = np.zeros((0, 3), dtype=np.float32)
        emptyb = np.zeros(0, dtype=bool)
        if self.proj_active is None or self.proj_pos is None:
            return empty3, empty3, emptyb, emptyb
        idx = np.flatnonzero(self.proj_active)
        n = int(idx.size)
        if n == 0:
            return empty3, empty3, emptyb, emptyb

        pos = self.proj_pos[idx]
        vel = (
            self.proj_vel[idx]
            if self.proj_vel is not None and self.proj_vel.shape[0] > 0
            else np.zeros((n, 3), dtype=np.float32)
        )
        owner = self.proj_owner[idx] if self.proj_owner is not None else np.full(n, -1, dtype=np.int32)
        owner_kind = (
            self.proj_owner_kind[idx]
            if self.proj_owner_kind is not None
            else np.zeros(n, dtype=np.int8)
        )
        guid_law = (
            self.proj_guid_law[idx]
            if self.proj_guid_law is not None
            else np.full(n, -1, dtype=np.int32)
        )
        conf = (
            self.proj_guid_confidence[idx]
            if self.proj_guid_confidence is not None
            else np.ones(n, dtype=np.float32)
        )

        # owner 阵营（owner_kind：0=装甲 1=飞机；越界/未知 → -1 视为敌方）
        is_friendly = np.zeros(n, dtype=bool)
        for k in range(n):
            ok = int(owner_kind[k])
            o = int(owner[k])
            side = -1
            if ok == 0:
                if self.armor_side is not None and 0 <= o < self.armor_side.shape[0]:
                    side = int(self.armor_side[o])
            else:
                if self.aircraft_side is not None and 0 <= o < self.aircraft_side.shape[0]:
                    side = int(self.aircraft_side[o])
            is_friendly[k] = side == int(player_side)

        is_missile = guid_law >= 0
        keep = is_friendly | (is_missile & (~is_friendly) & (conf >= confidence_threshold))
        return pos[keep], vel[keep], is_missile[keep], is_friendly[keep]

    # ------------------------------------------------------------------
    # 详情面板 / HUD 便捷查询（下游消费者唯一入口）
    # ------------------------------------------------------------------

    def unit_name(self, unit_idx: int) -> str:
        """单位展示名（side × role → 型号；Phase 8 接入 name 字段后替换）。

        2026-08-18 修复：按 role 细分型号——炮兵（ARTILLERY）显示 M109/2S19，
        不再落到阵营默认 MBT（M1A2/T-14）。armor_role 未装配（None）时按坦克默认。
        """
        if unit_idx < 0 or unit_idx >= self.armor_pos.shape[0]:
            return "#?"
        side = int(self.armor_side[unit_idx]) if unit_idx < self.armor_side.shape[0] else 0
        role = (
            int(self.armor_role[unit_idx])
            if self.armor_role is not None and unit_idx < len(self.armor_role)
            else 0
        )
        model = _SIDE_MODEL(side, role)
        return f"{model} #{unit_idx:02d}"

    def side_name(self, unit_idx: int) -> str:
        """阵营显示名（北约 / 华约 / 中立，entities.side SideNameMapper）。"""
        if unit_idx < 0 or unit_idx >= self.armor_side.shape[0]:
            return "未知"
        return _SIDE_NAMES(int(self.armor_side[unit_idx]))

    def speed_at(self, unit_idx: int) -> float:
        """单位水平速度 m/s。

        Phase 5e：渲染层位置差分估计（MainWindow 每帧喂入 EMA 平滑值）；
        未估计（armor_speed=None）或越界时回退 0.0。
        """
        if (
            self.armor_speed is None
            or unit_idx < 0
            or unit_idx >= self.armor_speed.shape[0]
        ):
            return 0.0
        return float(self.armor_speed[unit_idx])

    def _reserve_capacity(
        self,
        reserve: np.ndarray | None,
        capacity: np.ndarray | None,
        unit_idx: int,
    ) -> tuple[float, float] | None:
        if reserve is None or capacity is None or unit_idx < 0:
            return None
        if unit_idx >= reserve.shape[0]:
            return None
        return (float(reserve[unit_idx]), float(capacity[unit_idx]))

    def ammo_at(self, unit_idx: int) -> tuple[float, float] | None:
        """返回 (当前弹药, 弹药上限)；未装配后勤返回 None。"""
        return self._reserve_capacity(self.ammo_reserve, self.ammo_capacity, unit_idx)

    def fuel_at(self, unit_idx: int) -> tuple[float, float] | None:
        """返回 (当前燃料, 燃料上限)；未装配后勤返回 None。"""
        return self._reserve_capacity(self.fuel_reserve, self.fuel_capacity, unit_idx)

    @staticmethod
    def _avg_fill(
        reserve: np.ndarray | None,
        capacity: np.ndarray | None,
        mask: np.ndarray,
    ) -> float:
        """活跃单位平均存量百分比（排除容量为 0 的槽位）。"""
        if reserve is None or capacity is None or int(np.count_nonzero(mask)) == 0:
            return 0.0
        cap = capacity[mask]
        res = reserve[mask]
        valid = cap > 0
        if int(np.count_nonzero(valid)) == 0:
            return 0.0
        return float((res[valid] / cap[valid]).mean() * 100.0)

    def get_stats(self) -> dict[str, Any]:
        """级别 0（全局概况）聚合数据（docs §4.4 三级信息详细度）。"""
        active = self.armor_active
        side = self.armor_side
        # 统计口径：演示默认玩家=北约(0)，故 side 0/1 计 ally/enemy（绝对 side 的派生视图）
        combat_mask = active & (side < 2)  # 北约 + 华约（排除中立）
        return {
            "n_ally": int(np.count_nonzero(active & (side == 0))),
            "n_enemy": int(np.count_nonzero(active & (side == 1))),
            "n_neutral": int(np.count_nonzero(active & (side == 2))),
            "avg_ammo": self._avg_fill(self.ammo_reserve, self.ammo_capacity, combat_mask),
            "avg_fuel": self._avg_fill(self.fuel_reserve, self.fuel_capacity, combat_mask),
            "sim_time": self.sim_time,
            "tick_count": self.tick_count,
        }
