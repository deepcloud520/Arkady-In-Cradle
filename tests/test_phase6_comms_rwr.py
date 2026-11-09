"""Phase 6c 步骤⑨ 测试：通信簇 + RWR 告警 + 规避 + 诱饵（docs/phase6-ai-pipeline.md §九/§十）。

覆盖：通信簇（中央集群/独立建网/可见性/共享门控）/ RWR 告警（敌雷达/激光照射，
友方不告警）/ 规避（开 ECM + 机动 + 烟雾）/ 诱饵（部署 + 衰减 + 导弹来袭抛饵）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    EMITTER_KIND_DECOY,
    EMITTER_KIND_ILLUMINATOR,
    EMITTER_KIND_PLATFORM_RADAR,
    EMITTER_PARENT_ARMOR,
    ArmorPool,
    BoxPool,
    EmitterBand,
    EmitterPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
)
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.simulation.ai import (
    CommunicationNetwork,
    FormationAI,
    RWRDetector,
    UnitOrder,
    UnitRole,
)
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


def _spawn_unit(
    armor, boxes, fc, ammo, x, y, *, side=0, role=0, formation_id=-1
):
    idx = armor.spawn(
        x, y, 0.0,
        half_x=1.5, half_y=3.0, half_z=1.0,
        side=side, role=role, formation_id=formation_id,
    )
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
# 通信簇
# ===========================================================================


class TestCommunicationNetwork:
    def _make(self):
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        net = CommunicationNetwork(armor, formations, comm_threshold=0.5, central_side=0)
        return armor, formations, net

    def test_central_vs_hidden(self) -> None:
        _, formations, net = self._make()
        f_hi = formations.spawn(side=0)
        f_lo = formations.spawn(side=0)
        formations.comm[f_hi] = 0.9
        formations.comm[f_lo] = 0.2

        assert net.visible_formations() == [f_hi]
        assert net.hidden_formations() == [f_lo]

    def test_enemy_side_not_considered_visible(self) -> None:
        """敌方编队不进入「己方大面板可见性」判定。"""
        _, formations, net = self._make()
        formations.spawn(side=1)  # 敌方，comm=1.0
        assert net.visible_formations() == []
        assert net.hidden_formations() == []

    def test_independent_networks_grouped_by_side(self) -> None:
        _, formations, net = self._make()
        a = formations.spawn(side=0)
        b = formations.spawn(side=0)
        formations.comm[a] = 0.1
        formations.comm[b] = 0.2
        nets = net.independent_networks()
        assert len(nets) == 1
        assert sorted(nets[0]) == sorted([a, b])  # 己方失联编队（同阵营）→ 一个独立网络

    def test_share_contacts_gating(self) -> None:
        _, formations, net = self._make()
        hi = formations.spawn(side=0)
        lo = formations.spawn(side=0)
        lo2 = formations.spawn(side=1)
        formations.comm[hi] = 0.9
        formations.comm[lo] = 0.2
        formations.comm[lo2] = 0.2
        # 中央集群内可共享
        assert net.can_share_contacts(hi, hi) is True
        # 中央 ↔ 失联不可共享
        assert net.can_share_contacts(hi, lo) is False
        # 同阵营失联编队可独立建网共享；异阵营失联不可
        assert net.can_share_contacts(lo, lo) is True
        assert net.can_share_contacts(lo, lo2) is False

    def test_command_gating(self) -> None:
        _, formations, net = self._make()
        hi = formations.spawn(side=0)
        lo = formations.spawn(side=0)
        formations.comm[hi] = 0.9
        formations.comm[lo] = 0.2
        assert net.can_receive_command(hi) is True
        assert net.can_receive_command(lo) is False


# ===========================================================================
# RWR 告警
# ===========================================================================


class TestRWRDetector:
    def _make(self):
        armor, boxes, proj, ammo, fc, events = _make_combat()
        emitters = fc.emitters
        rwr = RWRDetector(armor, emitters, proj, events=events)
        return armor, boxes, proj, ammo, fc, emitters, events, rwr

    def test_enemy_radar_warns(self) -> None:
        armor, boxes, proj, ammo, fc, emitters, events, rwr = self._make()
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        enemy = _spawn_unit(armor, boxes, fc, ammo, 100.0, 0.0, side=1)
        emitters.activate(
            EMITTER_PARENT_ARMOR, enemy, 60.0, EmitterBand.RF, EMITTER_KIND_PLATFORM_RADAR
        )
        warnings = rwr.tick(1.0)
        assert len(warnings) == 1
        assert warnings[0]["unit"] == me
        assert warnings[0]["threat_kind"] == EMITTER_KIND_PLATFORM_RADAR
        kinds = {ev.kind for ev in events.drain()}
        assert "rwr_warning" in kinds

    def test_enemy_laser_warns(self) -> None:
        armor, boxes, proj, ammo, fc, emitters, events, rwr = self._make()
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        enemy = _spawn_unit(armor, boxes, fc, ammo, 50.0, 0.0, side=1)
        emitters.activate(
            EMITTER_PARENT_ARMOR, enemy, 60.0, EmitterBand.LASER, EMITTER_KIND_ILLUMINATOR
        )
        warnings = rwr.tick(1.0)
        assert len(warnings) == 1
        assert warnings[0]["threat_kind"] == EMITTER_KIND_ILLUMINATOR

    def test_friendly_emitter_no_warning(self) -> None:
        armor, boxes, proj, ammo, fc, emitters, events, rwr = self._make()
        _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        ally = _spawn_unit(armor, boxes, fc, ammo, 100.0, 0.0, side=0)
        emitters.activate(
            EMITTER_PARENT_ARMOR, ally, 60.0, EmitterBand.RF, EMITTER_KIND_PLATFORM_RADAR
        )
        assert rwr.tick(1.0) == []

    def test_no_emitter_no_warning(self) -> None:
        armor, boxes, proj, ammo, fc, emitters, events, rwr = self._make()
        _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0)
        assert rwr.tick(1.0) == []


# ===========================================================================
# 规避（FormationAI 消费 RWR 告警）
# ===========================================================================


class TestEvasion:
    def test_rwr_warning_raises_ecm_and_evades(self) -> None:
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        fidx = formations.spawn(side=0)
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fidx)
        enemy = _spawn_unit(armor, boxes, fc, ammo, 100.0, 0.0, side=1)
        fc.emitters.activate(
            EMITTER_PARENT_ARMOR, enemy, 60.0, EmitterBand.RF, EMITTER_KIND_PLATFORM_RADAR
        )

        ai = FormationAI(armor, formations, fc=fc, events=events)
        loop = SimulationLoop()
        ai.tick(loop, 1.0)

        assert float(armor.ecm[me]) > 0.0          # ECM 提升
        assert armor.wp_count[me] > 0              # 机动规避（写路标）
        kinds = {ev.kind for ev in events.drain()}
        assert "smoke" in kinds
        assert "rwr_warning" in kinds


# ===========================================================================
# 诱饵
# ===========================================================================


class TestDecoy:
    def test_deploy_decoy_and_decay(self) -> None:
        armor, boxes, proj, ammo, fc, events = _make_combat()
        emitters = fc.emitters
        idx = emitters.deploy_decoy(10.0, 20.0, 0.0, power=500.0, decay_per_s=2.0)
        assert emitters.kind[idx] == EMITTER_KIND_DECOY
        assert bool(emitters.active[idx]) is True

        # 衰减：sync_from_parent 对 DECOY 恒衰减
        emitters.sync_from_parent(armor, proj, 0.5)
        assert float(emitters.cur_power[idx]) < 500.0
        # 衰减足够久 → 失活归还
        for _ in range(10):
            emitters.sync_from_parent(armor, proj, 0.5)
        assert bool(emitters.active[idx]) is False

    def test_missile_defense_deploys_decoy(self) -> None:
        """敌制导弹（guid_law≥0）接近 → 抛诱饵 + decoy_deployed 事件。"""
        armor, boxes, proj, ammo, fc, events = _make_combat()
        formations = FormationAIPool(capacity=8)
        fidx = formations.spawn(side=0)
        me = _spawn_unit(armor, boxes, fc, ammo, 0.0, 0.0, side=0, formation_id=fidx)
        enemy = _spawn_unit(armor, boxes, fc, ammo, 500.0, 0.0, side=1)

        # 敌制导弹：guid_law≥0（用任意≥0 值模拟制导），owner=敌方
        pid = proj.spawn(200.0, 0.0, 2.0, -50.0, 0.0, 0.0, radius=0.1, dmg=10.0, kind=1, owner=enemy)
        proj.guid_law[pid] = 1

        ai = FormationAI(armor, formations, fc=fc, events=events)
        loop = SimulationLoop()
        ai.tick(loop, 1.0)

        kinds = {ev.kind for ev in events.drain()}
        assert "decoy_deployed" in kinds
        decoys = [int(i) for i in fc.emitters.active_indices() if fc.emitters.kind[i] == EMITTER_KIND_DECOY]
        assert len(decoys) == 1
