"""燃料消耗与补给请求（AGENTS.md Phase 3：logistics/fuel.py）。

职责：
- FuelState：每单位燃料储量 SoA（capacity / reserve / active）
- FuelManager：消耗 / 补给 / 补给请求排队与处理
- Numba 批量消耗内核（consume_fuel_batch，prange 并行）

设计要点（与 ammo.py 同构）：
    - 燃料不足时「耗尽剩余」，实际消耗 = min(reserve, 需求)
    - 消耗速率由 SimulationLoop 按单位速度/发动机功率折算后调用
    - 补给请求（SupplyRequest）入队，由 loop 间隙处理

性能：2048 单位批量消耗 < 0.1ms（Numba JIT 并行）。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.logistics.ammo import SupplyRequest

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Numba 内核
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def consume_fuel_batch(
    reserve: np.ndarray,       # (N,) float32  原地扣减（全长池）
    active: np.ndarray,        # (N,) bool_
    amounts: np.ndarray,       # (K,) float32  需求（每单位）
    out_consumed: np.ndarray,  # (K,) float32 实际消耗
) -> None:
    """批量燃料消耗（prange 并行，原地修改 reserve）。

    只处理前 K = amounts.shape[0] 个单位（K 可 ≤ 池长度 N）。
    燃料不足时耗尽剩余：out_consumed[i] = min(reserve[i], amounts[i])。

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
# 状态与管理器
# ---------------------------------------------------------------------------


class FuelState:
    """每单位燃料储量（SoA，与 ArmorPool 索引对齐）。

    Arrays:
        capacity (N,) float32  最大油箱容量
        reserve  (N,) float32  当前燃料量
        active   (N,) bool_    是否活跃
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = np.zeros(capacity, dtype=np.float32)
        self.reserve = np.zeros(capacity, dtype=np.float32)
        self.active = np.zeros(capacity, dtype=np.bool_)

    def set_unit(self, idx: int, cap: float, initial: float | None = None) -> None:
        self.capacity[idx] = cap
        self.reserve[idx] = initial if initial is not None else cap
        self.active[idx] = True

    def release(self, idx: int) -> None:
        self.active[idx] = False
        self.reserve[idx] = 0.0


class FuelManager:
    """燃料管理器：消耗 / 补给 / 请求排队。"""

    def __init__(self, state: FuelState | None = None, capacity: int = 2048) -> None:
        self.state = state or FuelState(capacity)
        self._requests: list[SupplyRequest] = []

    # ------------------------------------------------------------------
    # 消耗
    # ------------------------------------------------------------------

    def consume(self, idx: int, amount: float) -> float:
        """单单位燃料消耗，返回实际消耗（不足则耗尽剩余）。"""
        if not self.state.active[idx]:
            return 0.0
        avail = self.state.reserve[idx]
        used = min(avail, amount)
        self.state.reserve[idx] = avail - used
        return float(used)

    def consume_batch(self, amounts: np.ndarray) -> np.ndarray:
        """批量消耗，返回实际消耗数组（Numba 并行）。"""
        out = np.zeros(amounts.shape[0], dtype=np.float32)
        consume_fuel_batch(self.state.reserve, self.state.active, amounts, out)
        return out

    # ------------------------------------------------------------------
    # 补给
    # ------------------------------------------------------------------

    def resupply(self, idx: int, amount: float) -> float:
        """直接补给，不超容量上限。返回实际补给量。"""
        if not self.state.active[idx]:
            return 0.0
        space = self.state.capacity[idx] - self.state.reserve[idx]
        given = min(space, amount)
        self.state.reserve[idx] += given
        return float(given)

    def request_supply(self, idx: int, amount: float, priority: int = 0) -> SupplyRequest:
        """提交补给请求（入队）。"""
        req = SupplyRequest(target_idx=idx, amount=amount, priority=priority, item="fuel")
        self._requests.append(req)
        return req

    def pending_requests(self) -> list[SupplyRequest]:
        return list(self._requests)

    def process_requests(self, supply_capacity: float = 1e9) -> int:
        """处理队列中的补给请求（按 priority 降序），返回实际补给次数。"""
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
