"""编队推开（装甲 vs 装甲，空间哈希）单元测试。

覆盖 AGENTS.md 3.4「装甲 vs 装甲（编队推开）→ 空间哈希」：
    - 重叠装甲被推开（分离轴正确、距离减小）
    - 不重叠装甲不受影响
    - 非活跃装甲跳过
    - 编队纵队/密集队形多帧收敛（无穿透残留）
    - 与 FiringControl 集成的 tick 调用

运行：uv run python -m pytest tests/test_formation.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.combat.collision import FormationResolver, separate_armors_hash
from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import ArmorPool, BoxPool, ProjectilePool
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.simulation.loop import EventBus

import tests.test_phase3_integration as integ  # 复用 spawn_tank / make_battlefield


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def make_armor(count: int, spacing: float, overlap: float = 0.0):
    """生成一排 x 轴排列的装甲，可选重叠量（>0 则互相压入）。

    Returns:
        (ArmorPool, indices)
    """
    pool = ArmorPool(capacity=max(count, 16))
    idxs: list[int] = []
    for i in range(count):
        x = i * spacing + overlap * i  # overlap>0 时间距被压小 → 重叠
        idx = pool.spawn(x, 0.0, 0.0, half_x=1.5, half_y=3.0, half_z=1.0)
        idxs.append(idx)
    return pool, idxs


# ---------------------------------------------------------------------------
# FormationResolver 核心
# ---------------------------------------------------------------------------


class TestFormationResolver:
    def test_no_overlap_no_push(self):
        """间距足够时不产生推开。"""
        pool, _ = make_armor(4, spacing=10.0)  # half_x=1.5 → 需间距 > 3
        before = pool.pos[:, :2].copy()
        res = FormationResolver()
        pairs = res.resolve(pool.pos, pool.half, pool.active)
        assert pairs == 0
        np.testing.assert_allclose(pool.pos[:, :2], before, atol=1e-6)

    def test_overlap_gets_separated(self):
        """两辆重叠装甲沿分离轴被推开（中心距变大）。"""
        pool, idxs = make_armor(2, spacing=0.0, overlap=0.0)  # 完全重叠
        res = FormationResolver()
        pairs = res.resolve(pool.pos, pool.half, pool.active)
        assert pairs == 1
        d0 = abs(pool.pos[idxs[0], 0] - pool.pos[idxs[1], 0])
        assert d0 > 1e-3  # x 轴被推开

    def test_separation_reduces_overlap(self):
        """推开后重叠量应显著减小（各推一半 → 不再相交）。"""
        pool, idxs = make_armor(2, spacing=0.0)
        res = FormationResolver(sep_scale=0.5, push_cap=10.0)
        res.resolve(pool.pos, pool.half, pool.active)
        i, j = idxs
        half_sum = pool.half[i, 0] + pool.half[j, 0]
        dist = abs(pool.pos[i, 0] - pool.pos[j, 0])
        # 完全重叠：原距离 0，各推 half_sum/2 → 距离 = half_sum → 恰好不相交
        assert dist >= half_sum - 1e-3

    def test_inactive_skipped(self):
        """非活跃装甲不参与推开，也不被推开。"""
        pool, idxs = make_armor(3, spacing=0.0)
        pool.active[idxs[1]] = False  # 中间那辆阵亡
        before_mid = pool.pos[idxs[1]].copy()
        res = FormationResolver()
        res.resolve(pool.pos, pool.half, pool.active)
        np.testing.assert_allclose(pool.pos[idxs[1]], before_mid, atol=1e-6)

    def test_y_axis_separation(self):
        """y 轴重叠量更小（< x 重叠量）时沿 y 推开。"""
        pool = ArmorPool(capacity=8)
        # half_x=1.5, half_y=3.0
        # A(0,0) vs B(1.0, 5.5)：oxx = 3-1 = 2.0, oyy = 6-5.5 = 0.5 → oyy < oxx → 沿 y
        a = pool.spawn(0.0, 0.0, 0.0, half_x=1.5, half_y=3.0, half_z=1.0)
        b = pool.spawn(1.0, 5.5, 0.0, half_x=1.5, half_y=3.0, half_z=1.0)
        res = FormationResolver()
        res.resolve(pool.pos, pool.half, pool.active)
        assert abs(pool.pos[a, 1] - pool.pos[b, 1]) > 5.5  # y 距变大（推开）
        assert abs(pool.pos[a, 0] - pool.pos[b, 0]) <= 1.0 + 1e-3  # x 未推

    def test_dense_column_converges(self):
        """密集纵队多帧迭代后无残留穿透（空间哈希对纵队不退化）。"""
        pool, _ = make_armor(64, spacing=0.0)  # 64 辆完全堆叠
        res = FormationResolver(sep_scale=0.5, push_cap=2.0)
        # 迭代多次直到稳定（编队推开是迭代过程）
        for _ in range(200):
            pairs = res.resolve(pool.pos, pool.half, pool.active)
            if pairs == 0:
                break
        # 最终应完全分离（无重叠对）
        pairs_final = res.resolve(pool.pos, pool.half, pool.active)
        assert pairs_final == 0

    def test_push_cap_limits(self):
        """单帧推开距离受 push_cap 限制（防数值弹射；去重后每对只推一次）。"""
        pool, idxs = make_armor(2, spacing=0.0)
        res = FormationResolver(sep_scale=0.5, push_cap=0.5)
        res.resolve(pool.pos, pool.half, pool.active)
        # oxx = 3.0 → push = min(1.5, 0.5) = 0.5，各推 0.5 → 距离 = 1.0
        assert abs(pool.pos[idxs[0], 0] - pool.pos[idxs[1], 0]) <= 1.0 + 1e-3


# ---------------------------------------------------------------------------
# 与 FiringControl 集成
# ---------------------------------------------------------------------------


class TestFormationIntegrated:
    def test_fc_tick_resolves_overlap(self):
        """FiringControl.tick 自动调用编队推开。"""
        armor, boxes, projectiles, ammo, fuel, fc = integ.make_battlefield()
        # 两辆坦克完全重叠
        m1 = integ.spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        m2 = integ.spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        d_before = abs(armor.pos[m1, 0] - armor.pos[m2, 0])
        assert d_before < 1e-4
        fc.tick(1.0)  # 推进 1s，触发编队推开
        d_after = abs(armor.pos[m1, 0] - armor.pos[m2, 0])
        assert d_after > d_before + 1e-3
        # 且无穿透残留（half_x=1.5 → 需距离 ≥ 3）
        assert d_after >= armor.half[m1, 0] + armor.half[m2, 0] - 1e-2

    def test_fc_overlap_does_not_disturb_combat(self):
        """编队推开不破坏既有交战（正常间距坦克不受影响）。"""
        armor, boxes, projectiles, ammo, fuel, fc = integ.make_battlefield()
        m1 = integ.spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        m2 = integ.spawn_tank(armor, boxes, ammo, fuel, fc, 0.0, 0.0)
        # 正常间距（x 距 500），编队推开不应移动它们
        armor.pos[m2, 0] = 500.0
        before = armor.pos[:, :2].copy()
        fc.tick(1.0)
        np.testing.assert_allclose(armor.pos[:, :2], before, atol=1e-3)


# ---------------------------------------------------------------------------
# 性能冒烟（不硬断言耗时，仅确保内核可跑）
# ---------------------------------------------------------------------------


class TestFormationPerformance:
    def test_2048_armor_runs(self):
        """2048 装甲全量编队推开可执行且无越界。"""
        pool = ArmorPool(capacity=2048)
        rng = np.random.default_rng(0)
        # 稀疏战场（正常间距）+ 一小撮密集编队
        for i in range(2048):
            pool.spawn(
                float(rng.uniform(-2000, 2000)),
                float(rng.uniform(-2000, 2000)),
                0.0,
                half_x=1.5, half_y=3.0, half_z=1.0,
            )
        # 100 辆密集编队（重叠）
        for i in range(100):
            pool.spawn(100.0 + i * 0.5, 100.0, 0.0, half_x=1.5, half_y=3.0, half_z=1.0)
        res = FormationResolver()
        pairs = res.resolve(pool.pos, pool.half, pool.active)
        assert pairs >= 0  # 可执行（不抛异常）
        # 密集编队应产生推开
        assert pairs > 0
