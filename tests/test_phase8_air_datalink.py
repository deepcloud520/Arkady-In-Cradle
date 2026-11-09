"""Phase 8 步骤⑧：空军数据链——A 导 B 射（relay_guidance）+ BVR 数据链发射折扣 +
协同规避（docs/phase8-datalink-design.md §十四 步骤⑧验收）。

覆盖：
    relay_guidance（中继写入 guid_target_pos + guid_comm 门控）、
    fire_air confidence 折扣（数据链共享航迹 → 散布放大）、
    AirUnitAI 数据链发射（contact_confidence 注入）、
    协同规避（RWR 锁定 → 诱饵抛热焰 + 其余成员机动脱离）。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.combat.damage_submission import DamageSubmissionPool
from Airia.lulilo.combat.engagement import FiringControl
from Airia.lulilo.combat.guidance import MissileControlRegistry
from Airia.lulilo.combat.weapons import WeaponSystem
from Airia.lulilo.core.rng import RNGManager
from Airia.lulilo.entities.pools import (
    AirFormationPool,
    AircraftPool,
    ArmorPool,
    BoxPool,
    EmitterBand,
    EmitterPool,
    ProjectilePool,
)
from Airia.lulilo.movement.aerodynamics import AircraftAero
from Airia.lulilo.movement.flight_control import FlightControl
from Airia.lulilo.simulation.air_ai import (
    AirFormationAI,
    AirUnitAI,
    AirUnitOrder,
)

from tests.test_phase7_aam import _air_registry


def _rig() -> tuple[FiringControl, AircraftPool, EmitterPool]:
    aircraft = AircraftPool(capacity=8)
    aero = AircraftAero(8)
    armor = ArmorPool(capacity=4)
    boxes = BoxPool(capacity=64)
    proj = ProjectilePool(capacity=32)
    emitters = EmitterPool(capacity=64)
    firing = FiringControl(
        armor, boxes, proj,
        weapon_system=WeaponSystem(_air_registry()),
        control=MissileControlRegistry.with_builtin(),
        emitters=emitters,
        rng=RNGManager(seed=42),
        damage_pool=DamageSubmissionPool(capacity=64),
        aircraft=aircraft,
    )
    return firing, aircraft, emitters


class TestRelayGuidance:
    """A 导 B 射（docs §7.1）。"""

    def test_relay_writes_target_and_comm(self) -> None:
        firing, aircraft, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(8000.0, 0.0, 3000.0, side=0)
        aircraft.vel[foe] = (-250.0, 0.0, 0.0)
        aircraft.aam_left[me] = 4
        rec = firing.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                              target_kind=0, target_idx=foe)
        assert rec is not None
        # 前出预警机（数据链成员）中继目标航迹
        new_pos = np.array([7000.0, 0.0, 3000.0], np.float32)
        firing.relay_guidance(rec.proj_idx, new_pos, link_q=0.8)
        p = firing.projectiles
        assert p.guid_target_pos[rec.proj_idx, 0] == pytest.approx(7000.0)
        assert p.guid_comm[rec.proj_idx] == pytest.approx(0.8)

    def test_relay_low_link_comm_gate(self) -> None:
        firing, aircraft, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(8000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        rec = firing.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                              target_kind=0, target_idx=foe)
        firing.relay_guidance(rec.proj_idx, np.array([7000.0, 0.0, 3000.0]), link_q=0.1)
        # 链路差 → guid_comm 低 → 6b 人在回路门控（comm < threshold → HOLD/MEMORY）
        assert firing.projectiles.guid_comm[rec.proj_idx] == pytest.approx(0.1)


class TestDatalinkLaunch:
    """BVR 数据链共享航迹发射（docs §7.2）。"""

    def test_shared_track_launch_spreads_sigma(self) -> None:
        firing, aircraft, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        # 本机雷达（confidence=1.0）vs 数据链共享航迹（confidence=0.4）
        r1 = firing.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                             target_kind=0, target_idx=foe, confidence=1.0)
        r2 = firing.fire_air(me, aircraft.pos[foe], ammo_id="aim120_amraam",
                             target_kind=0, target_idx=foe, confidence=0.4)
        assert r1 is not None and r2 is not None
        assert r2.sigma_m > r1.sigma_m, "低置信共享航迹 → 散布放大（命中概率折扣）"

    def test_air_unit_ai_uses_contact_confidence(self) -> None:
        firing, aircraft, emitters = _rig()
        me = aircraft.spawn(0.0, 0.0, 3000.0, side=1)
        foe = aircraft.spawn(10000.0, 0.0, 3000.0, side=0)
        aircraft.aam_left[me] = 4
        flight = FlightControl(aircraft.capacity, AircraftAero(8))
        # 数据链接触源：目标置信度 0.4（共享航迹）
        unit = AirUnitAI(aircraft, AircraftAero(8), flight, firing,
                         contact_confidence=lambda t: 0.4 if t == foe else 1.0)
        aircraft.order[me] = int(AirUnitOrder.ENGAGE_AIR)
        aircraft.target_idx[me] = foe
        unit.tick(0.1)
        assert aircraft.aam_left[me] == 3, "数据链共享航迹也应可发射（带折扣）"


class TestCoordinatedEvasion:
    """协同规避（docs §7.3）。"""

    def _rig_formation(self):
        aircraft = AircraftPool(capacity=8)
        aero = AircraftAero(8)
        formations = AirFormationPool(capacity=4)
        armor = ArmorPool(capacity=2)
        emitters = EmitterPool(capacity=64)
        flight = FlightControl(aircraft.capacity, aero)
        fa = AirFormationAI(aircraft, formations, aero, armor=armor,
                            flight=flight, emitters=emitters)
        return aircraft, formations, emitters, flight, fa

    def test_threatened_member_triggers_flare_and_break(self) -> None:
        aircraft, formations, emitters, flight, fa = self._rig_formation()
        fidx = formations.spawn(side=1)
        m1 = aircraft.spawn(0.0, 0.0, 3000.0, side=1, formation_id=fidx)
        m2 = aircraft.spawn(100.0, 0.0, 3000.0, side=1, formation_id=fidx)
        aircraft.aam_left[m1] = 0   # 弹药耗尽 → 诱饵机
        aircraft.aam_left[m2] = 2
        formations.member_count[fidx] = 2
        formations.leader_idx[fidx] = m1
        # 编队受 AAM 锁定（RWR level 3）
        aircraft.rwr_state[m1] = 3
        aircraft.rwr_state[m2] = 3
        fa.tick(1.0)
        # 诱饵（m1 弹药耗尽）抛热焰
        flares = [
            e for e in emitters.active_indices()
            if emitters.kind[e] == 5   # DECOY
        ]
        assert len(flares) >= 1, "诱饵机应抛热焰"
        assert emitters.band[flares[0]] == int(EmitterBand.IR)
        # 其余受威胁成员（m2）机动脱离（kinematic_defeat → 急转 + 下降）
        assert aircraft.phi_cmd[m2] != 0.0, "受威胁成员应机动脱离"
        assert aircraft.gamma_cmd[m2] < 0.0, "脱离机动应下降"
