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
from Airis.lulilo.simulation.state import SimulationSnapshot

logger = logging.getLogger(__name__)

# 部位名称（对应 PartType 顺序，与 armor_damaged_parts 的 5 列一一对应）
PART_NAMES = ("车体", "炮塔", "引擎", "履带", "弹药架")

# 阵营 → 单位型号 / 名称（渲染层命名规则，不碰引擎；Phase 8 正式接入 side 后替换）
_SIDE_MODEL = {0: "T-90", 1: "T-72B3", 2: "载具"}
_SIDE_NAME = {0: "友军", 1: "敌军", 2: "中立"}


class UnitRenderStatus(IntEnum):
    """渲染用单位状态（Phase 5 全为 ACTIVE；Phase 8 扩展 UNKNOWN/KIA/MIA）。"""

    ACTIVE = 0
    UNKNOWN = 1   # Phase 8：变成问号
    KIA = 2       # Phase 8：确认击毁
    MIA = 3       # Phase 8：信号丢失


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
    armor_side: np.ndarray          # (N,)   uint8   0=友军 / 1=敌军 / 2=中立
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
    sim_time: float = 0.0
    tick_count: int = 0

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_snapshot(cls, snapshot: SimulationSnapshot) -> "WorldAdapter":
        """从模拟快照构造（Phase 5 透传模式）。

        Note:
            - 直接引用 SoA 数组，不拷贝（浅视图）
            - confidence 恒 1.0 / status 恒 ACTIVE（上帝视角）
            - armor.team：Phase 6c 正式下沉为 SoA 字段（docs/phase6-ai-design.md §1.1）；
              通过 getattr 读取，缺失时回退旧动态 side 属性，再缺失全 0（友军）
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

        side = getattr(armor, "team", None)
        if side is None:
            side = getattr(armor, "side", None)  # 兼容旧动态属性（Phase 6c 前演示）
        if side is None:
            side = np.zeros(n, dtype=np.uint8)
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
            armor_health=armor.health,
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
            sim_time=snapshot.sim_time,
            tick_count=snapshot.tick_count,
        )

    # ------------------------------------------------------------------
    # 查询接口（下游消费者唯一入口）
    # ------------------------------------------------------------------

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
        """阵营名（友军 / 敌军 / 中立）。"""
        if unit_idx < 0 or unit_idx >= self.armor_side.shape[0]:
            return "未知"
        return _SIDE_NAME.get(int(self.armor_side[unit_idx]), "中立")

    def speed_at(self, unit_idx: int) -> float:
        """单位速度 m/s（Phase 5 占位；ArmorPool 暂无 velocity 数组）。"""
        return 0.0

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
        combat_mask = active & (side < 2)  # 友军 + 敌军（排除中立）
        return {
            "n_ally": int(np.count_nonzero(active & (side == 0))),
            "n_enemy": int(np.count_nonzero(active & (side == 1))),
            "n_neutral": int(np.count_nonzero(active & (side == 2))),
            "avg_ammo": self._avg_fill(self.ammo_reserve, self.ammo_capacity, combat_mask),
            "avg_fuel": self._avg_fill(self.fuel_reserve, self.fuel_capacity, combat_mask),
            "sim_time": self.sim_time,
            "tick_count": self.tick_count,
        }
