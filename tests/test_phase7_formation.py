"""Phase 7 步骤⑤：AirFormationAI 编队飞行 + 意图 FSM（docs/phase7-air-pipeline.md §十四 步骤⑤验收）。

覆盖：
    编队保持（僚机追长机槽位）、队形切换（意图 → 槽位表）、CAP 巡逻、
    意图 FSM（自由开火接敌交战 / 兵力劣势 RTB / 弹药耗尽 RTB）、
    贪心目标分配（覆盖轮 + 并发轮）、STRIKE 对地分配。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.entities.pools import (
    AirFormationPool,
    AircraftPool,
    AircraftRole,
    ArmorPool,
)
from Airia.lulilo.movement.aerodynamics import AircraftAero, integrate_aircraft_motion
from Airia.lulilo.movement.flight_control import FlightControl
from Airia.lulilo.simulation.air_ai import (
    AirFormationAI,
    AirFormationIntent,
    AirUnitOrder,
    TGT_AIR,
    TGT_GROUND,
)

G = 9.81
DT = 0.1


def _rig(air_cap: int = 8, armor_cap: int = 8):
    aircraft = AircraftPool(capacity=air_cap)
    aero = AircraftAero(air_cap)
    formations = AirFormationPool(capacity=4)
    armor = ArmorPool(capacity=armor_cap)
    ai = AirFormationAI(aircraft, formations, aero, armor=armor)
    return aircraft, aero, formations, armor, ai


def _spawn_formation(aircraft, formations, aero, side: int,
                     n: int, x: float = 0.0, y: float = 0.0,
                     role: int = int(AircraftRole.FIGHTER),
                     agm: int = 0) -> tuple[int, list[int]]:
    fidx = formations.spawn(side=side)
    members = []
    for k in range(n):
        m = aircraft.spawn(x + k * 50.0, y + k * 50.0, 3000.0,
                           side=side, role=role, formation_id=fidx)
        aircraft.aam_left[m] = 6
        aircraft.agm_left[m] = agm
        aircraft.vel[m] = (250.0, 0.0, 0.0)
        aircraft.heading[m] = 0.0
        members.append(m)
    formations.member_count[fidx] = n
    formations.leader_idx[fidx] = members[0]   # 首名成员为长机
    return fidx, members


def _run_physics(aircraft, aero, flight, steps: int) -> None:
    stall = np.zeros(aircraft.capacity, np.bool_)
    for _ in range(steps):
        flight.tick(aircraft, DT)
        integrate_aircraft_motion(
            aircraft.pos, aircraft.vel, aircraft.heading, aircraft.pitch, aircraft.bank,
            aircraft.throttle, aircraft.phi_cmd, aircraft.gamma_cmd,
            aircraft.state, aircraft.dmg_factor, aircraft.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            aircraft.fuel, aero.fuel_burn,
            1.225, 8500.0, G, DT, stall,
        )
        idxs = np.flatnonzero(stall & aircraft.active)
        if idxs.size:
            aircraft.state[idxs] = 5  # CRASHING


class TestFormationFlight:
    """编队保持：僚机追长机槽位（楔形）。"""

    def test_wingman_tracks_slot(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (leader, wing) = _spawn_formation(aircraft, formations, aero, side=1, n=2)
        # 长机任务航路点：向东巡航
        aircraft.set_waypoints(leader, [(20000.0, 0.0, 3000.0)])
        aircraft.cruise_alt[leader] = 3000.0
        aircraft.cruise_spd[leader] = 250.0
        aircraft.cruise_alt[wing] = 3000.0
        aircraft.cruise_spd[wing] = 250.0
        flight = FlightControl(aircraft.capacity, aero)
        # 跑 120s：每 1s 一次 AI（编队几何），其间飞控+积分
        for _ in range(120):
            ai.tick(1.0)
            _run_physics(aircraft, aero, flight, 10)
        # 僚机应接近槽位（长机 + (-350, -220)）
        expect_x = aircraft.pos[leader, 0] - 350.0
        expect_y = aircraft.pos[leader, 1] - 220.0
        assert abs(aircraft.pos[wing, 0] - expect_x) < 250.0, \
            f"僚机 X 未达槽位: {aircraft.pos[wing, 0]:.0f} vs {expect_x:.0f}"
        assert abs(aircraft.pos[wing, 1] - expect_y) < 250.0, \
            f"僚机 Y 未达槽位: {aircraft.pos[wing, 1]:.0f} vs {expect_y:.0f}"
        # 编队整体前进
        assert aircraft.pos[leader, 0] > 5000.0, "编队应整体前进"

    def test_slot_table_switches_with_intent(self) -> None:
        """意图 STRIKE → 纵队槽位（僚机在长机正后方 -350y）。"""
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (leader, wing) = _spawn_formation(aircraft, formations, aero, side=1, n=2)
        ai.commanded_intent[fidx] = int(AirFormationIntent.STRIKE)
        ai.tick(1.0)
        assert int(formations.intent[fidx]) == int(AirFormationIntent.STRIKE)
        # 纵队槽位 (0, -350)
        assert aircraft.wp[wing, 0, 0] == pytest.approx(aircraft.pos[leader, 0])
        assert aircraft.wp[wing, 0, 1] == pytest.approx(aircraft.pos[leader, 1] - 350.0)


class TestCapPatrol:
    """CAP 巡逻：无接敌 → 意图保持 CAP，无交战订单。"""

    def test_cap_no_contact(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (leader,) = _spawn_formation(aircraft, formations, aero, side=1, n=1)
        aircraft.set_waypoints(leader, [(10000.0, 0.0, 3000.0)])
        for _ in range(10):
            ai.tick(1.0)
        assert int(formations.intent[fidx]) == int(AirFormationIntent.CAP)
        assert int(aircraft.order[leader]) == int(AirUnitOrder.HOLD)


class TestIntentFSM:
    """意图迁移（docs §5.2 迁移表）。"""

    def test_cap_engage_when_weapons_free(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (me,) = _spawn_formation(aircraft, formations, aero, side=1, n=1)
        foe = aircraft.spawn(5000.0, 0.0, 3000.0, side=0)
        ai.weapons_free[fidx] = True
        ai.tick(1.0)
        assert int(aircraft.order[me]) == int(AirUnitOrder.ENGAGE_AIR)
        assert aircraft.target_idx[me] == foe
        assert aircraft.target_kind[me] == TGT_AIR

    def test_cap_no_engage_without_weapons_free(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (me,) = _spawn_formation(aircraft, formations, aero, side=1, n=1)
        aircraft.spawn(5000.0, 0.0, 3000.0, side=0)   # 有敌机但未下令
        ai.tick(1.0)
        assert int(aircraft.order[me]) == int(AirUnitOrder.HOLD), "未下令自由开火不应交战"

    def test_cap_rtb_when_outnumbered(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (me,) = _spawn_formation(aircraft, formations, aero, side=1, n=1)
        aircraft.aam_left[me] = 0      # 弹药耗尽 → 劣势
        aircraft.spawn(3000.0, 0.0, 3000.0, side=0)
        aircraft.spawn(3100.0, 0.0, 3000.0, side=0)
        ai.weapons_free[fidx] = True
        ai.tick(1.0)
        assert int(formations.intent[fidx]) == int(AirFormationIntent.RTB)

    def test_strike_rtb_when_ammo_depleted(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (me,) = _spawn_formation(aircraft, formations, aero, side=1, n=1,
                                       role=int(AircraftRole.ATTACKER), agm=3)
        ai.commanded_intent[fidx] = int(AirFormationIntent.STRIKE)
        ai.tick(1.0)
        assert int(formations.intent[fidx]) == int(AirFormationIntent.STRIKE)
        aircraft.agm_left[me] = 0
        ai.tick(1.0)
        assert int(formations.intent[fidx]) == int(AirFormationIntent.RTB), "弹药耗尽应返航"


class TestTargetAssignment:
    """贪心两轮分配（§5.3）。"""

    def test_cover_round_each_target_one_engager(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (m1, m2) = _spawn_formation(aircraft, formations, aero, side=1, n=2)
        f1 = aircraft.spawn(4000.0, 0.0, 3000.0, side=0)
        f2 = aircraft.spawn(4000.0, 2000.0, 3000.0, side=0)
        ai.weapons_free[fidx] = True
        ai.tick(1.0)
        targets = {int(aircraft.target_idx[m]) for m in (m1, m2)}
        assert targets == {f1, f2}, f"覆盖轮应覆盖全部敌机: {targets}"

    def test_concurrent_round_extra_fighters(self) -> None:
        """3 成员 vs 1 敌机：覆盖轮 1 个，并发轮 2 个追加同一目标。"""
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (m1, m2, m3) = _spawn_formation(aircraft, formations, aero, side=1, n=3)
        foe = aircraft.spawn(4000.0, 0.0, 3000.0, side=0)
        ai.weapons_free[fidx] = True
        ai.tick(1.0)
        engagers = [m for m in (m1, m2, m3)
                    if int(aircraft.order[m]) == int(AirUnitOrder.ENGAGE_AIR)]
        assert len(engagers) == 3, f"全部成员应投入交战: {len(engagers)}"
        for m in engagers:
            assert aircraft.target_idx[m] == foe

    def test_strike_assigns_ground_target(self) -> None:
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (me,) = _spawn_formation(aircraft, formations, aero, side=1, n=1,
                                       role=int(AircraftRole.ATTACKER), agm=4)
        ai.commanded_intent[fidx] = int(AirFormationIntent.STRIKE)
        tank = armor.spawn(6000.0, 0.0, 0.0)
        armor.side[tank] = 0
        ai.tick(1.0)
        assert int(aircraft.order[me]) == int(AirUnitOrder.STRIKE_GROUND)
        assert aircraft.target_idx[me] == tank
        assert aircraft.target_kind[me] == TGT_GROUND

    def test_sead_suppresses(self) -> None:
        """SEAD 意图 + 地面目标 → SUPPRESS 订单。"""
        aircraft, aero, formations, armor, ai = _rig()
        fidx, (me,) = _spawn_formation(aircraft, formations, aero, side=1, n=1,
                                       role=int(AircraftRole.SEAD), agm=4)
        ai.commanded_intent[fidx] = int(AirFormationIntent.SEAD)
        sam = armor.spawn(6000.0, 0.0, 0.0)
        armor.side[sam] = 0
        armor.role[sam] = 3   # AA
        ai.tick(1.0)
        assert int(aircraft.order[me]) == int(AirUnitOrder.STRIKE_GROUND)
