"""core 层单元测试：logging + RNGManager。

覆盖：logger 创建、RNG 播种/多流/分布正确性。
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.core.rng import RNGManager


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_get_logger_creates_and_reuses() -> None:
    """get_logger 应创建并复用同名 logger。"""
    a = get_logger("test.core.a")
    b = get_logger("test.core.a")
    assert a is b, "同名 logger 应复用同一实例"
    assert isinstance(a, logging.Logger)


def test_get_logger_distinct_names() -> None:
    """不同名称应返回不同 logger。"""
    a = get_logger("test.core.x")
    b = get_logger("test.core.y")
    assert a is not b, "不同名称应返回不同实例"


# ---------------------------------------------------------------------------
# RNGManager
# ---------------------------------------------------------------------------


class TestRNGManager:
    """RNGManager 测试套件。"""

    def test_seed_reproducible(self) -> None:
        """相同 seed 应产生相同序列。"""
        r1 = RNGManager(seed=42)
        r2 = RNGManager(seed=42)
        for _ in range(20):
            assert r1.uniform() == pytest.approx(r2.uniform())

    def test_different_seed_diverges(self) -> None:
        """不同 seed 应产生不同序列。"""
        r1 = RNGManager(seed=1)
        r2 = RNGManager(seed=2)
        s1 = [r1.uniform() for _ in range(10)]
        s2 = [r2.uniform() for _ in range(10)]
        assert s1 != s2, "不同 seed 不应产生相同序列"

    def test_uniform_range(self) -> None:
        """uniform 应在 [low, high) 内。"""
        rng = RNGManager(seed=0)
        for _ in range(200):
            v = rng.uniform(10.0, 20.0)
            assert 10.0 <= v < 20.0

    def test_uniform_batch_shape_and_range(self) -> None:
        """批量 uniform 应返回正确形状和范围。"""
        rng = RNGManager(seed=0)
        arr = rng.uniform_batch(-1.0, 1.0, 1000)
        assert arr.shape == (1000,)
        assert arr.dtype == np.float32
        assert np.all(arr >= -1.0)
        assert np.all(arr < 1.0)

    def test_normal_approximate_stats(self) -> None:
        """正态分布应大致满足均值和标准差。"""
        rng = RNGManager(seed=0)
        arr = rng.normal_batch(5.0, 2.0, 10000)
        assert abs(float(np.mean(arr)) - 5.0) < 0.1
        assert abs(float(np.std(arr)) - 2.0) < 0.1

    def test_choice_range(self) -> None:
        """choice 应在 [0, a) 内。"""
        rng = RNGManager(seed=0)
        for _ in range(100):
            v = rng.choice(5)
            assert 0 <= v < 5

    def test_choice_batch(self) -> None:
        """批量 choice 形状和范围。"""
        rng = RNGManager(seed=0)
        arr = rng.choice_batch(3, 100)
        assert arr.shape == (100,)
        assert np.all(arr >= 0)
        assert np.all(arr < 3)

    def test_hit_probability_deterministic(self) -> None:
        """hit_probability(1.0) 应永远命中，(0.0) 永远不命中。"""
        rng = RNGManager(seed=0)
        assert rng.hit_probability(1.0) == 1.0
        assert rng.hit_probability(0.0) == 0.0

    def test_hit_probability_statistical(self) -> None:
        """hit_probability(0.5) 大量试验后频率应接近 0.5。"""
        rng = RNGManager(seed=0)
        hits = sum(rng.hit_probability(0.5) for _ in range(5000))
        assert 2300 < hits < 2700, f"p=0.5 时 5000 次命中应在 2300~2700，实际 {hits}"

    def test_dispersion_offset(self) -> None:
        """散布偏移应返回两个 float。"""
        rng = RNGManager(seed=0)
        dx, dy = rng.dispersion_offset(1.0)
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_reseed_restarts_stream(self) -> None:
        """reseed 应重置序列。"""
        rng = RNGManager(seed=42)
        s1 = [rng.uniform() for _ in range(5)]
        rng.reseed(42)
        s2 = [rng.uniform() for _ in range(5)]
        assert s1 == s2, "reseed 应产生相同序列"

    def test_spawn_child_isolated(self) -> None:
        """子 RNG 应与父 RNG 独立。"""
        parent = RNGManager(seed=0)
        child = parent.spawn_child()
        p_vals = [parent.uniform() for _ in range(10)]
        c_vals = [child.uniform() for _ in range(10)]
        assert p_vals != c_vals, "父子 RNG 序列应独立"
