"""抛射体/导弹可视化回归（渲染层只读，WorldAdapter.projectile_markers）。

覆盖过滤规则：
    - 己方抛射体（owner 阵营 == 玩家）：全绘（子弹 + 导弹）。
    - 敌方导弹（guid_law >= 0）：proj_guid_confidence >= 门限才绘（上帝视角恒 1.0）。
    - 敌方无制导弹（子弹/炮弹）：不绘。
    - 未装配抛射物池 → 空数组。
"""

from __future__ import annotations

import numpy as np

from Airis.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    ProjectilePool,
    ProjectileType,
)
from Airis.lulilo.simulation.state import SimulationState
from Airis.render.adapter import WorldAdapter


def _rig() -> tuple[SimulationState, int, int, ProjectilePool]:
    state = SimulationState()
    state.armor_pool = ArmorPool(capacity=8)
    state.box_pool = BoxPool(capacity=64)
    state.projectile_pool = ProjectilePool(capacity=64)
    ally = state.armor_pool.spawn(0.0, 0.0, 0.0, side=0)
    enemy = state.armor_pool.spawn(100.0, 0.0, 0.0, side=1)
    return state, ally, enemy, state.projectile_pool


class TestProjectileMarkers:
    def test_friendly_always_drawn_enemy_missile_by_confidence(self) -> None:
        state, ally, enemy, p = _rig()
        # 己方子弹（无制导）：全绘
        p.spawn(5.0, 0.0, 1.0, 100.0, 0.0, 0.0, kind=ProjectileType.BULLET, owner=ally)
        # 敌方导弹（制导，置信 1.0 >= 门限）：绘
        mi = p.spawn(50.0, 0.0, 2.0, -200.0, 0.0, 0.0, kind=ProjectileType.MISSILE, owner=enemy)
        p.guid_law[mi] = 2
        p.guid_confidence[mi] = 1.0
        # 敌方子弹（无制导）：不绘
        p.spawn(60.0, 0.0, 1.0, 300.0, 0.0, 0.0, kind=ProjectileType.BULLET, owner=enemy)

        adapter = WorldAdapter.from_snapshot(state.snapshot())
        pos, vel, is_missile, is_friendly = adapter.projectile_markers(player_side=0)

        assert pos.shape[0] == 2
        # 恰好一己方（子弹）、一敌方（导弹）
        assert int(np.sum(is_friendly)) == 1
        assert int(np.sum(~is_friendly)) == 1
        # 敌方那条是导弹，己方那条是子弹
        enemy_pos = pos[~is_friendly]
        assert enemy_pos.shape[0] == 1
        assert is_missile[~is_friendly][0]

    def test_enemy_missile_below_confidence_filtered(self) -> None:
        state, ally, enemy, p = _rig()
        mi = p.spawn(50.0, 0.0, 2.0, -200.0, 0.0, 0.0, kind=ProjectileType.MISSILE, owner=enemy)
        p.guid_law[mi] = 2
        p.guid_confidence[mi] = 0.2  # 低于门限 → 不绘（Phase 8 迷雾）
        adapter = WorldAdapter.from_snapshot(state.snapshot())
        pos, _, _, _ = adapter.projectile_markers(player_side=0, confidence_threshold=0.5)
        assert pos.shape[0] == 0

    def test_no_projectile_pool_returns_empty(self) -> None:
        state = SimulationState()
        state.armor_pool = ArmorPool(capacity=8)
        state.box_pool = BoxPool(capacity=64)
        adapter = WorldAdapter.from_snapshot(state.snapshot())
        pos, vel, is_missile, is_friendly = adapter.projectile_markers()
        assert pos.shape[0] == 0
        assert vel.shape[0] == 0
        assert is_missile.shape[0] == 0
        assert is_friendly.shape[0] == 0
