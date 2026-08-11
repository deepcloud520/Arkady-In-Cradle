"""弹药消耗与补给请求（AGENTS.md Phase 3：logistics/ammo.py）。

职责：
- AmmoState：每单位弹药储量 SoA（capacity / reserve / active）
- AmmoManager：消耗 / 补给 / 补给请求排队与处理
- Numba 批量消耗内核（consume_ammo_batch，prange 并行）

设计要点：
    - 弹药不足时「耗尽剩余」而非抛错——实际消耗 = min(reserve, 需求)
    - 补给请求（SupplyRequest）进入队列，由 SimulationLoop 的 Tick 间隙
      或补给车到达时处理（Phase 3 正式迭代：网络流优化）
    - 与 SoA 对齐：状态数组与 ArmorPool 共享索引语义（同长度）

性能：2048 单位批量消耗 < 0.1ms（Numba JIT 并行）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from Airis.lulilo.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Numba 内核
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def consume_ammo_batch(
    reserve: np.ndarray,     # (N,) float32  原地扣减（全长池）
    active: np.ndarray,      # (N,) bool_
    amounts: np.ndarray,     # (K,) float32  需求（每单位）
    out_consumed: np.ndarray,  # (K,) float32 实际消耗
) -> None:
    """批量弹药消耗（prange 并行，原地修改 reserve）。

    只处理前 K = amounts.shape[0] 个单位（K 可 ≤ 池长度 N）。
    弹药不足时耗尽剩余：out_consumed[i] = min(reserve[i], amounts[i])。

    ⚠️ 契约：K = len(amounts) = len(out_consumed) ≤ N = len(reserve)。
    循环上界取 K 而非 N——曾用 reserve.shape[0] 导致 i ≥ K 时越界写堆
    （2026-08-11 修复，此前在全量 pytest 下表现为堆损坏崩溃）。
    """
    n = min(amounts.shape[0], reserve.shape[0])
    for i in prange(n):
        if not active[i]:
            out_consumed[i] = 0.0
            continue
        need = amounts[i]
        if reserve[i] >= need:
            reserve[i] -= need
            out_consumed[i] = need
        else:
            out_consumed[i] = reserve[i]
            reserve[i] = 0.0


# ---------------------------------------------------------------------------
# 状态与请求
# ---------------------------------------------------------------------------


@dataclass
class SupplyRequest:
    """一次补给请求（进队列，由 loop 或补给节点处理）。"""

    target_idx: int
    amount: float
    priority: int = 0
    item: str = "ammo"


class AmmoState:
    """每单位弹药储量（SoA，与 ArmorPool 索引对齐）。

    Arrays:
        capacity (N,) float32  最大装载量
        reserve  (N,) float32  当前弹药量
        active   (N,) bool_    是否活跃（休眠单位不消耗）
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = np.zeros(capacity, dtype=np.float32)
        self.reserve = np.zeros(capacity, dtype=np.float32)
        self.active = np.zeros(capacity, dtype=np.bool_)

    def set_unit(self, idx: int, cap: float, initial: float | None = None) -> None:
        """为索引 idx 的单位装载弹药。"""
        self.capacity[idx] = cap
        self.reserve[idx] = initial if initial is not None else cap
        self.active[idx] = True

    def release(self, idx: int) -> None:
        """单位销毁时释放其弹药槽。"""
        self.active[idx] = False
        self.reserve[idx] = 0.0


class AmmoManager:
    """弹药管理器：消耗 / 补给 / 请求排队。"""

    def __init__(self, state: AmmoState | None = None, capacity: int = 2048) -> None:
        self.state = state or AmmoState(capacity)
        self._requests: list[SupplyRequest] = []

    # ------------------------------------------------------------------
    # 消耗
    # ------------------------------------------------------------------

    def consume(self, idx: int, amount: float) -> float:
        """单单位弹药消耗，返回实际消耗（不足则耗尽剩余）。"""
        if not self.state.active[idx]:
            return 0.0
        avail = self.state.reserve[idx]
        used = min(avail, amount)
        self.state.reserve[idx] = avail - used
        return float(used)

    def consume_batch(self, amounts: np.ndarray) -> np.ndarray:
        """批量消耗，返回实际消耗数组（Numba 并行）。"""
        out = np.zeros(amounts.shape[0], dtype=np.float32)
        consume_ammo_batch(self.state.reserve, self.state.active, amounts, out)
        return out

    # ------------------------------------------------------------------
    # 补给
    # ------------------------------------------------------------------

    def resupply(self, idx: int, amount: float) -> float:
        """直接补给（如补给车到达），不超容量上限。返回实际补给量。"""
        if not self.state.active[idx]:
            return 0.0
        space = self.state.capacity[idx] - self.state.reserve[idx]
        given = min(space, amount)
        self.state.reserve[idx] += given
        return float(given)

    def request_supply(self, idx: int, amount: float, priority: int = 0) -> SupplyRequest:
        """提交补给请求（入队，由 loop 间隙处理）。"""
        req = SupplyRequest(target_idx=idx, amount=amount, priority=priority, item="ammo")
        self._requests.append(req)
        return req

    def pending_requests(self) -> list[SupplyRequest]:
        return list(self._requests)

    def process_requests(self, supply_capacity: float = 1e9) -> int:
        """处理队列中的补给请求（按 priority 降序），返回实际补给次数。

        Args:
            supply_capacity: 本批可分配的补给总量（模拟补给车/基地吞吐）。

        Returns:
            成功补给的单位数。
        """
        if not self._requests:
            return 0
        self._requests.sort(key=lambda r: -r.priority)
        filled = 0
        remaining = supply_capacity
        for req in self._requests:
            if remaining <= 0:
                break
            given = self.resupply(req.target_idx, min(req.amount, remaining))
            if given > 0:
                remaining -= given
                filled += 1
        self._requests.clear()
        return filled
