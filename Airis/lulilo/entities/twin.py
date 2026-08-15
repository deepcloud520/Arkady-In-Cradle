"""认知孪生池（Phase 8，docs/phase8-datalink-design.md §一）。

「物理世界 Real 永远只属于引擎，一切 AI 只认 Twin」——`TwinPool` 是 AI/玩家/
渲染看到的「现实」：位置带误差、状态分档（UNKNOWN/ACTIVE/KIA/MIA）、置信度衰减。
数据链好坏（通信簇/链路质量）决定孪生与物理世界的偏差 = 战争迷雾浓度。

设计要点（docs §1.1/§1.2）：
    - **独立池而非给 ArmorPool/AircraftPool 加字段**：物理池是 Numba 热路径，
      孪生更新是 Python 1s 级遍历——写分离保证扩容/并行纪律不打架（AGENTS.md 4.3）；
      且泛化到所有单位类型（装甲/飞机/未来舰船共用）。
    - **1:1 槽映射免 dict**：`slot_of(kind, idx)` 直接算槽位——
      装甲 idx → twin[idx]，飞机 idx → twin[MAX_ARMOR + idx]。无 Python dict
      驱动逻辑（AGENTS.md 4.3 缓存纪律）。
    - **destroy 不销毁孪生**：单位销毁后孪生保留（残骸/失联状态由 TwinEngine
      提交 KIA/MIA 观测），绑定关系保持（渲染残骸）；槽位 1:1 固定无泄漏。
    - 键：unit_kind（0=armor 1=aircraft）+ unit_idx → twin_idx。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from Airis.lulilo.entities.pools import MAX_AIRCRAFT, MAX_ARMOR

# 孪生池容量（覆盖 2048 装甲 + 128 飞机 + 余量，docs §1.2）
MAX_TWIN = 4096

# 单位种类（TwinPool.unit_kind）
TWIN_KIND_ARMOR = 0
TWIN_KIND_AIRCRAFT = 1


class TwinStatus(IntEnum):
    """孪生状态（docs §1.2：概念文档「? 标记」的落地）。"""
    UNKNOWN = 0    # 未观测/超时降级
    ACTIVE = 1     # 观测中（置信度决定渲染精度）
    KIA = 2        # 确认击毁（残骸 + 红叉）
    MIA = 3        # 信号丢失（灰色轮廓闪烁）


# Numba 用 int 常量
_ST_UNKNOWN = int(TwinStatus.UNKNOWN)
_ST_ACTIVE = int(TwinStatus.ACTIVE)
_ST_KIA = int(TwinStatus.KIA)
_ST_MIA = int(TwinStatus.MIA)


class TwinPool:
    """认知孪生池（SoA 预分配，MAX_TWIN=4096）。

    数组布局（docs §1.2）：
        键：twin_idx（回指物理索引）/ unit_kind / active
        孪生状态：pos / vel / status / confidence / last_update / pos_err /
                  source / last_vel_known（dead-reckon 外推）
    """

    def __init__(self, capacity: int = MAX_TWIN) -> None:
        self.capacity = int(capacity)
        # ---- 键（1:1 槽映射：armor → [0, MAX_ARMOR)，aircraft → [MAX_ARMOR, ...)）----
        self.twin_idx = np.full(self.capacity, -1, dtype=np.int32)   # 回指物理单位索引
        self.unit_kind = np.zeros(self.capacity, dtype=np.uint8)     # 0=armor 1=aircraft
        self.active = np.zeros(self.capacity, dtype=np.bool_)
        # ---- 孪生状态（AI/玩家/渲染看到的「现实」）----
        self.pos = np.zeros((self.capacity, 3), dtype=np.float32)    # 估计位置（含误差）
        self.vel = np.zeros((self.capacity, 3), dtype=np.float32)    # 估计速度（外推用）
        self.health = np.zeros(self.capacity, dtype=np.float32)      # 报告血量（滞后/误差，docs §5.1）
        self.status = np.zeros(self.capacity, dtype=np.uint8)        # TwinStatus
        self.confidence = np.zeros(self.capacity, dtype=np.float32)  # 0~1
        self.last_update = np.zeros(self.capacity, dtype=np.float32)  # 最近观测时间
        self.pos_err = np.zeros(self.capacity, dtype=np.float32)     # 位置误差半径 (m)
        self.source = np.zeros(self.capacity, dtype=np.uint8)        # 0无/1本队/2数据链/3友军报告/4叙事
        self.last_vel_known = np.zeros((self.capacity, 3), dtype=np.float32)  # dead-reckon

    # ------------------------------------------------------------------
    # 槽映射（1:1，无 dict）
    # ------------------------------------------------------------------

    @staticmethod
    def slot_of(kind: int, idx: int) -> int:
        """单位 (kind, idx) → twin 槽位（-1 越界/非法）。"""
        if kind == TWIN_KIND_ARMOR and 0 <= idx < MAX_ARMOR:
            return idx
        if kind == TWIN_KIND_AIRCRAFT and 0 <= idx < MAX_AIRCRAFT:
            return MAX_ARMOR + idx
        return -1

    # ------------------------------------------------------------------
    # 绑定生命周期
    # ------------------------------------------------------------------

    def bind(
        self,
        kind: int,
        idx: int,
        *,
        pos: tuple[float, float, float] | np.ndarray | None = None,
        vel: tuple[float, float, float] | np.ndarray | None = None,
        init_confidence: float = 0.0,
    ) -> int:
        """绑定物理单位 → 孪生槽（docs §1.3：spawn 同步建立，初始 UNKNOWN）。

        友军单位初始由己方 C2 直接注入 ACTIVE（`TwinEngine` 首 tick 处理）。

        Returns:
            twin 槽位（-1 越界）。
        """
        s = self.slot_of(kind, idx)
        if s < 0 or s >= self.capacity:
            return -1
        self.twin_idx[s] = idx
        self.unit_kind[s] = int(kind)
        self.active[s] = True
        if pos is not None:
            self.pos[s] = pos
        if vel is not None:
            self.vel[s] = vel
        self.status[s] = _ST_UNKNOWN
        self.confidence[s] = float(init_confidence)
        self.last_update[s] = -1e9
        self.pos_err[s] = 0.0
        self.source[s] = 0
        self.last_vel_known[s] = vel if vel is not None else 0.0
        return s

    def unbind(self, kind: int, idx: int) -> None:
        """解绑（物理单位销毁）：孪生保留（残骸/失联由引擎标记状态），
        仅清 active 标记允许槽被复用？——不：1:1 固定映射，槽保留残骸渲染。
        本方法仅作显式接口（当前无调用方，供测试验证绑定生命周期）。
        """
        s = self.slot_of(kind, idx)
        if 0 <= s < self.capacity:
            # 保留数组内容（残骸），只允许状态引擎后续改写
            pass

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def is_bound(self, kind: int, idx: int) -> bool:
        s = self.slot_of(kind, idx)
        return 0 <= s < self.capacity and bool(self.active[s])

    def active_slots(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    @property
    def active_count(self) -> int:
        return int(np.sum(self.active))

    # ------------------------------------------------------------------
    # 扩容（1:1 固定映射理论上不扩容；防御性实现）
    # ------------------------------------------------------------------

    def _expand(self, factor: float = 1.25) -> None:
        new_cap = max(int(self.capacity * factor), self.capacity + 1)

        def _resize(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=old.dtype)
            slices = tuple(slice(0, min(old.shape[i], shape[i])) for i in range(old.ndim))
            new_arr[slices] = old[slices]
            return new_arr

        self.twin_idx = _resize(self.twin_idx, (new_cap,))
        self.unit_kind = _resize(self.unit_kind, (new_cap,))
        self.active = _resize(self.active, (new_cap,))
        self.pos = _resize(self.pos, (new_cap, 3))
        self.vel = _resize(self.vel, (new_cap, 3))
        self.health = _resize(self.health, (new_cap,))
        self.status = _resize(self.status, (new_cap,))
        self.confidence = _resize(self.confidence, (new_cap,))
        self.last_update = _resize(self.last_update, (new_cap,))
        self.pos_err = _resize(self.pos_err, (new_cap,))
        self.source = _resize(self.source, (new_cap,))
        self.last_vel_known = _resize(self.last_vel_known, (new_cap, 3))
        old = self.capacity
        self.twin_idx[old:] = -1
        self.capacity = new_cap


# ---------------------------------------------------------------------------
# 序列化字段清单（simulation/serialization.py 消费）
# ---------------------------------------------------------------------------

TWIN_FIELDS = (
    "twin_idx",
    "unit_kind",
    "active",
    "pos",
    "vel",
    "health",
    "status",
    "confidence",
    "last_update",
    "pos_err",
    "source",
    "last_vel_known",
)
