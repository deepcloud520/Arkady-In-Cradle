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
from Airis.lulilo.entities.side import SideNameMapper
from Airis.lulilo.simulation.state import SimulationSnapshot

logger = logging.getLogger(__name__)

# 部位名称（对应 PartType 顺序，与 armor_damaged_parts 的 5 列一一对应）
PART_NAMES = ("车体", "炮塔", "引擎", "履带", "弹药架")

# 阵营 → 单位型号（渲染层命名规则，不碰引擎；Phase 8 接入 name 字段后替换）
_SIDE_MODEL = {0: "T-90", 1: "T-72B3", 2: "载具"}
# 阵营显示名统一走 entities.side（SideNameMapper：0=北约 1=华约 2=中立）
_SIDE_NAMES = SideNameMapper()


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
    box: Any,
    armor_active: np.ndarray,
    n_armor: int,
) -> np.ndarray:
    """聚合 BoxPool：每个存活装甲单位 → 5 部位损坏掩码 (N,5) bool。

    语义：
        - 仅统计活跃单位（armor_active=True 的父单位）
        - box.parent>=0 且 box.active=False → 该部位损坏
        - box 槽位为空闲（parent=-1）或父单位已死亡，不视为部位损坏

    Args:
        box: BoxPool（可为 None，骨架期未装配）。
        armor_active: (N,) bool 存活掩码。
        n_armor: 装甲池容量。

    Returns:
        (n_armor, len(PartType)) bool 数组；True = 该部位已损坏。
    """
    damaged = np.zeros((n_armor, len(PartType)), dtype=np.bool_)
    if box is None or box.capacity == 0 or n_armor == 0:
        return damaged
    parent = box.parent
    valid = (parent >= 0) & (parent < n_armor) & armor_active[parent]
    mask = valid & (~box.active)
    if np.any(mask):
        damaged[parent[mask], box.part[mask]] = True
    return damaged


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
    # ── 后勤视图（可选；None = 未装配）──
    ammo_reserve: np.ndarray | None = None    # (N,) float32 当前弹药
    ammo_capacity: np.ndarray | None = None   # (N,) float32 弹药上限
    fuel_reserve: np.ndarray | None = None    # (N,) float32 当前燃料
    fuel_capacity: np.ndarray | None = None   # (N,) float32 燃料上限
    # ── 速度视图（可选；None = 未估计，speed_at 回退 0）──
    # Phase 5e：渲染层位置差分估计（EMA 平滑），引擎侧无 velocity 字段
    armor_speed: np.ndarray | None = None     # (N,) float32 水平速度 m/s
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
                box_parent=np.zeros(0, dtype=np.int32),
                box_part=np.zeros(0, dtype=np.int32),
                box_active=emptyb,
                sim_time=snapshot.sim_time,
                tick_count=snapshot.tick_count,
            )

        side = getattr(armor, "side", None)
        if side is None:
            side = np.zeros(n, dtype=np.uint8)  # 缺失兜底：全北约
        pitch = getattr(armor, "pitch", None)
        if pitch is None:
            pitch = np.zeros(n, dtype=np.float32)
        roll = getattr(armor, "roll", None)
        if roll is None:
            roll = np.zeros(n, dtype=np.float32)

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
            armor_damaged_parts=_build_damaged_parts(box, armor.active, n),
            box_parent=box.parent if box is not None else np.zeros(0, dtype=np.int32),
            box_part=box.part if box is not None else np.zeros(0, dtype=np.int32),
            box_active=box.active if box is not None else np.zeros(0, dtype=np.bool_),
            ammo_reserve=ammo_st.reserve if ammo_st is not None else None,
            ammo_capacity=ammo_st.capacity if ammo_st is not None else None,
            fuel_reserve=fuel_st.reserve if fuel_st is not None else None,
            fuel_capacity=fuel_st.capacity if fuel_st is not None else None,
            armor_speed=armor_speed,
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
            armor_damaged_parts=_build_damaged_parts(box, armor.active, n),
            box_parent=box.parent if box is not None else np.zeros(0, dtype=np.int32),
            box_part=box.part if box is not None else np.zeros(0, dtype=np.int32),
            box_active=box.active if box is not None else np.zeros(0, dtype=np.bool_),
            ammo_reserve=ammo_st.reserve if ammo_st is not None else None,
            ammo_capacity=ammo_st.capacity if ammo_st is not None else None,
            fuel_reserve=fuel_st.reserve if fuel_st is not None else None,
            fuel_capacity=fuel_st.capacity if fuel_st is not None else None,
            armor_speed=None,
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
    # 详情面板 / HUD 便捷查询（下游消费者唯一入口）
    # ------------------------------------------------------------------

    def unit_name(self, unit_idx: int) -> str:
        """单位展示名（Phase 5 用 side + idx 规则生成；Phase 8 接入 name 字段）。"""
        if unit_idx < 0 or unit_idx >= self.armor_pos.shape[0]:
            return "#?"
        side = int(self.armor_side[unit_idx]) if unit_idx < self.armor_side.shape[0] else 0
        model = _SIDE_MODEL.get(side, "载具")
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
