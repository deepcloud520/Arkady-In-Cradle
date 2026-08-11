"""流式随机数管理器（AGENTS.md Phase 0 规划项）。

确定性回放已废弃（AGENTS.md 4.4），但 RNGManager 仍提供：
- 可种子的多流随机数（武器散布、传感器噪声、AI 决策抖动）
- 与 NumPy 默认 RNG 隔离，避免仿真逻辑耦合全局状态
- 性能：预生成批量随机数减少单次调用开销

用法:
    from Airis.lulilo.core.rng import RNGManager

    rng = RNGManager(seed=42)
    value = rng.uniform(0.0, 1.0)          # 单个 float
    arr = rng.uniform_batch(0.0, 1.0, 100) # 批量
"""

from __future__ import annotations

import numpy as np


class RNGManager:
    """多流可种子随机数管理器。

    内部维护独立 np.random.Generator（PCG-64），不与全局 np.random 共享状态。
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.Generator(np.random.PCG64(seed))

    # ------------------------------------------------------------------
    # 基础分布
    # ------------------------------------------------------------------

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """返回 [low, high) 均匀分布 float。"""
        return float(self._rng.uniform(low, high))

    def uniform_batch(self, low: float, high: float, size: int) -> np.ndarray:
        """批量均匀分布。"""
        return self._rng.uniform(low, high, size).astype(np.float32)

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        """正态分布。"""
        return float(self._rng.normal(loc, scale))

    def normal_batch(self, loc: float, scale: float, size: int) -> np.ndarray:
        """批量正态分布。"""
        return self._rng.normal(loc, scale, size).astype(np.float32)

    def choice(self, a: int) -> int:
        """从 [0, a) 中均匀随机选一个整数。"""
        return int(self._rng.integers(0, a))

    def choice_batch(self, a: int, size: int) -> np.ndarray:
        """批量选择。"""
        return self._rng.integers(0, a, size)

    # ------------------------------------------------------------------
    # 仿真专用
    # ------------------------------------------------------------------

    def hit_probability(self, base_ph: float) -> float:
        """将基础命中概率 base_ph 转为一次 Bernoulli 试验结果 (0 或 1)。

        Args:
            base_ph: 基础命中概率 [0, 1]。

        Returns:
            1 表示命中，0 未命中。
        """
        return 1.0 if self._rng.uniform() < base_ph else 0.0

    def dispersion_offset(self, sigma: float) -> tuple[float, float]:
        """武器散布偏移（高斯散布，返回 (dx, dy)）。

        Args:
            sigma: 散布标准差（米）。

        Returns:
            (dx, dy) 世界坐标偏移量。
        """
        return (
            float(self._rng.normal(0.0, sigma)),
            float(self._rng.normal(0.0, sigma)),
        )

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def reseed(self, seed: int) -> None:
        """重新播种，重置流状态。"""
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def spawn_child(self, seed_offset: int = 0) -> RNGManager:
        """从当前状态派生一个子 RNG（用于 AI/传感器子流，互不干扰）。

        Args:
            seed_offset: 加到当前内部状态的偏移。

        Returns:
            新的 RNGManager，拥有独立的 Generator。
        """
        new_seed = self._rng.integers(0, 2**31 - 1) + seed_offset
        return RNGManager(seed=int(new_seed))
