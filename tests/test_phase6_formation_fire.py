"""Phase 6c 步骤⑦ 测试：编队队形保持 + 射界规避（docs/phase6-ai-pipeline.md §二.4）。

覆盖：楔形槽位偏移 / MANEUVER 成员按槽位展开（路标分散）/ 敌方直射火力射界判定
（朝向锥 × 射程）/ 射界规避（绕出 + 事件）/ 朝向背对或超射程不触发。
"""

from __future__ import annotations

import math

import pytest

from Airia.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    ArmorPool,
    BoxPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
)
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.simulation.ai import FormationAI, UnitOrder
from Airia.lulilo.simulation.loop import EventBus, SimulationLoop

_SIMPLE_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
]


def _make_combat(seed: int = 7):
    armor = ArmorPool(capacity=32)
    boxes = BoxPool(capacity=256)
    projectiles = ProjectilePool(capacity=512)
    ammo = AmmoManager(capacity=32)
    events = EventBus()
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, events=events, rng=RNGManager(seed=seed),
    )
    return armor, boxes, projectiles, ammo, fc, events


def _spawn_unit(armor, boxes, fc, ammo, x, y, *, side=0, formation_id=-1, yaw=0.0):
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        side=side, formation_id=formation_id,
    )
    armor.yaw[idx] = yaw
    for part, hx, hy, hz, lx, ly, lz in _SIMPLE_PARTS:
        bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
        if int(part) == int(PartType.HULL):
            boxes.hp[bidx] = 100.0
            boxes.hp_max[bidx] = 100.0
            armor.bind_hull(idx, bidx)
    boxes.sync_all_from_parent(armor)
    fc.set_unit_config(idx, UnitCombatConfig(weapon_id="m256_120mm", armor_mm=300.0))
    ammo.state.set_unit(idx, 20.0, 20.0)
    return idx


# ===========================================================================
# 队形保持（楔形槽位）
# ===========================================================================


class TestFormationKeeping:
    def _make_ai(self):
        armor = ArmorPool(capacity=32)
        formations = FormationAIPool(capacity=8)
        ai = FormationAI(armor, formations)
        return armor, formations, ai

    def test_slot_zero_is_anchor(self) -> None:
        _, _, ai = self._make_ai()
        assert ai._formation_offset(0, 5) == (0.0, 0.0)
        assert ai._formation_offset(0, 1) == (0.0, 0.0)

    def test_wedge_offsets_spread(self) -> None:
        _, _, ai = self._make_ai()
        ox1, oy1 = ai._formation_offset(1, 5)
        ox2, oy2 = ai._formation_offset(2, 5)
        ox3, _ = ai._formation_offset(3, 5)
        assert ox1 == pytest.approx(ai.SLOT_SPACING_M)   # 右
        assert ox2 == pytest.approx(-ai.SLOT_SPACING_M)  # 左
        assert abs(ox3) > ai.SLOT_SPACING_M               # 第二排更远
        assert ox1 != ox2

    def test_maneuver_spreads_members(self) -> None:
        """MANEUVER：多成员按槽位展开，路标分散（非全挤 target_zone）。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        ai = FormationAI(armor, formations, fc=fc, events=events)
        fidx = formations.spawn(side=0, role=0, target_zone=(300.0, 400.0, 0.0))
        members = [
            _spawn_unit(armor, boxes, fc, ammo, i * 10.0, 0.0, side=0, formation_id=fidx)
            for i in range(3)
        ]
        for m in members:
            armor.order[m] = int(UnitOrder.MANEUVER)

        loop = SimulationLoop()
        ai.tick(loop, 1.0)

        wps = {(armor.wp[m, 0, 0], armor.wp[m, 0, 1]) for m in members}
        assert len(wps) == 3  # 三成员路标互不相同（楔形展开）


# ===========================================================================
# 射界规避
# ===========================================================================


class TestFieldOfFire:
    def _make(self):
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        ai = FormationAI(armor, formations, fc=fc, events=events)
        return armor, boxes, proj, ammo, fc, events, formations, ai

    def test_in_cone_when_facing(self) -> None:
        armor, boxes, proj, ammo, fc, events, formations, ai = self._make()
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        enemy = _spawn_unit(armor, boxes, fc, ammo, 500.0, 0.0, side=1, yaw=math.pi)
        assert ai._in_enemy_field_of_fire(me) == enemy

    def test_not_in_cone_when_facing_away(self) -> None:
        armor, boxes, proj, ammo, fc, events, formations, ai = self._make()
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        _spawn_unit(armor, boxes, fc, ammo, 500.0, 0.0, side=1, yaw=0.0)
        assert ai._in_enemy_field_of_fire(me) is None

    def test_not_threatened_out_of_range(self) -> None:
        armor, boxes, proj, ammo, fc, events, formations, ai = self._make()
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        # 敌在 5000m 外（默认射程 4000m），朝向正对我，仍不触发
        _spawn_unit(armor, boxes, fc, ammo, 5000.0, 0.0, side=1, yaw=math.pi)
        assert ai._in_enemy_field_of_fire(me) is None

    def test_avoid_field_of_fire_writes_waypoint(self) -> None:
        armor, boxes, proj, ammo, fc, events, formations, ai = self._make()
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        enemy = _spawn_unit(armor, boxes, fc, ammo, 500.0, 0.0, side=1, yaw=math.pi)
        ai._avoid_field_of_fire(me, enemy)
        assert armor.wp_count[me] > 0
        kinds = {ev.kind for ev in events.drain()}
        assert "field_of_fire_evade" in kinds

    def test_tick_applies_evasion(self) -> None:
        """端到端：敌方直射火力射界内 → tick 内触发射界规避。"""
        armor, boxes, proj, ammo, fc, events, formations, ai = self._make()
        fidx = formations.spawn(side=0, role=0, target_zone=(300.0, 400.0, 0.0))
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fidx)
        armor.order[me] = int(UnitOrder.MANEUVER)
        _spawn_unit(armor, boxes, fc, ammo, 500.0, 0.0, side=1, yaw=math.pi)

        loop = SimulationLoop()
        ai.tick(loop, 1.0)

        kinds = {ev.kind for ev in events.drain()}
        assert "field_of_fire_evade" in kinds
        assert armor.wp_count[me] > 0
