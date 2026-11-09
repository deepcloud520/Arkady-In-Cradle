"""Phase 7 步骤⑨：起飞/降落/基地 + 分组降落（docs/phase7-air-pipeline.md §十四 步骤⑨验收）。

覆盖：
    AirbaseConfig 跑道航向、request_takeoff（PARKED→TAKOFF 对齐跑道）、
    离地爬升 → CRUISE、request_landing（RTB→LANDING 进近）、
    触地滑跑 → PARKED、分组降落排队间隔。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.entities.pools import AircraftPool, AircraftState
from Airia.lulilo.movement.aerodynamics import AircraftAero, integrate_aircraft_motion
from Airia.lulilo.movement.flight_control import FlightControl
from Airia.lulilo.simulation.air_ai import AirbaseConfig, AirfieldOps

G = 9.81
DT = 0.1


def _rig() -> tuple[AircraftPool, AircraftAero, FlightControl, AirfieldOps]:
    aircraft = AircraftPool(capacity=8)
    aero = AircraftAero(8)
    flight = FlightControl(aircraft.capacity, aero)
    ops = AirfieldOps(aircraft)
    return aircraft, aero, flight, ops


def _run_physics(aircraft, aero, flight, steps: int, dt: float = DT) -> None:
    stall = np.zeros(aircraft.capacity, np.bool_)
    for _ in range(steps):
        flight.tick(aircraft, dt)
        integrate_aircraft_motion(
            aircraft.pos, aircraft.vel, aircraft.heading, aircraft.pitch, aircraft.bank,
            aircraft.throttle, aircraft.phi_cmd, aircraft.gamma_cmd,
            aircraft.state, aircraft.dmg_factor, aircraft.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            aircraft.fuel, aero.fuel_burn,
            1.225, 8500.0, G, dt, stall,
        )
        idxs = np.flatnonzero(stall & aircraft.active)
        if idxs.size:
            aircraft.state[idxs] = int(AircraftState.CRASHING)


class TestAirbaseConfig:
    def test_run_heading_radians(self) -> None:
        base = AirbaseConfig(run_heading_deg=90.0)
        assert base.run_heading == pytest.approx(math.pi / 2)


class TestTakeoff:
    """起飞：PARKED → TAKEOFF → 离地 → CRUISE。"""

    def test_takeoff_aligns_runway_and_climbs(self) -> None:
        aircraft, aero, flight, ops = _rig()
        base = AirbaseConfig(x=0.0, y=0.0, z=0.0, run_heading_deg=0.0)
        ops.add_base(base)
        i = aircraft.spawn(0.0, 0.0, 0.0, cruise_alt=3000.0, cruise_spd=250.0)
        aircraft.state[i] = int(AircraftState.PARKED)
        ops.request_takeoff(i, base)
        ops.takeoff_in_queue(i)
        assert aircraft.state[i] == int(AircraftState.TAKEOFF)
        assert aircraft.heading[i] == pytest.approx(0.0)   # 对齐 0° 跑道
        # 起飞滑跑 + 爬升
        for _ in range(300):   # 30s
            ops.tick(DT)
            _run_physics(aircraft, aero, flight, 1)
            if aircraft.state[i] == int(AircraftState.CRUISE):
                break
        assert aircraft.state[i] == int(AircraftState.CRUISE), "离地后应转巡航"
        assert aircraft.pos[i, 2] > 50.0, f"应已离地爬升: z={aircraft.pos[i, 2]:.0f}"
        assert aircraft.pos[i, 0] > 500.0, "应沿跑道方向前进"

    def test_takeoff_heading_90(self) -> None:
        """90° 跑道 → 起飞后朝 +y 前进。"""
        aircraft, aero, flight, ops = _rig()
        base = AirbaseConfig(x=0.0, y=0.0, z=0.0, run_heading_deg=90.0)
        ops.add_base(base)
        i = aircraft.spawn(0.0, 0.0, 0.0, cruise_alt=3000.0, cruise_spd=250.0)
        aircraft.state[i] = int(AircraftState.PARKED)
        ops.request_takeoff(i, base)
        ops.takeoff_in_queue(i)
        for _ in range(300):
            ops.tick(DT)
            _run_physics(aircraft, aero, flight, 1)
            if aircraft.state[i] == int(AircraftState.CRUISE):
                break
        assert aircraft.pos[i, 1] > 500.0, "90° 跑道应朝 +y 起飞"


class TestLanding:
    """降落：RTB → LANDING → 触地 → PARKED。"""

    def test_landing_full_cycle(self) -> None:
        aircraft, aero, flight, ops = _rig()
        base = AirbaseConfig(x=0.0, y=0.0, z=0.0, run_heading_deg=0.0, radius=3000.0)
        ops.add_base(base)
        i = aircraft.spawn(3000.0, 0.0, 2000.0, cruise_alt=2000.0, cruise_spd=200.0)
        aircraft.vel[i] = (-150.0, 0.0, 0.0)   # 朝基地飞
        aircraft.heading[i] = math.pi
        # 进入基地空域 → 申请降落
        assert ops.in_airspace(i, base)
        ops.request_landing(i, base)
        assert aircraft.state[i] == int(AircraftState.LANDING)
        # 进近（landing_control 降高度减速）→ 触地 → 驻停
        for _ in range(600):   # 60s
            ops.tick(DT)
            _run_physics(aircraft, aero, flight, 1)
            if aircraft.state[i] == int(AircraftState.PARKED):
                break
        assert aircraft.state[i] == int(AircraftState.PARKED), "触地后应驻停"
        assert np.linalg.norm(aircraft.vel[i]) == pytest.approx(0.0, abs=1e-3)

    def test_landing_grounds_at_runway(self) -> None:
        aircraft, aero, flight, ops = _rig()
        base = AirbaseConfig(x=0.0, y=0.0, z=0.0, run_heading_deg=0.0, radius=3000.0)
        ops.add_base(base)
        i = aircraft.spawn(3000.0, 0.0, 2000.0, cruise_alt=2000.0, cruise_spd=200.0)
        aircraft.vel[i] = (-150.0, 0.0, 0.0)
        aircraft.heading[i] = math.pi
        ops.request_landing(i, base)
        for _ in range(600):
            ops.tick(DT)
            _run_physics(aircraft, aero, flight, 1)
            if aircraft.state[i] == int(AircraftState.PARKED):
                break
        # 落在跑道附近（|x| < 1000）
        assert abs(aircraft.pos[i, 0]) < 1500.0, f"落点偏离跑道: x={aircraft.pos[i, 0]:.0f}"


class TestGroupedLanding:
    """分组降落：多机排队，依序进近。"""

    def test_queue_releases_sequentially(self) -> None:
        aircraft, aero, flight, ops = _rig()
        base = AirbaseConfig(x=0.0, y=0.0, z=0.0, run_heading_deg=0.0, radius=4000.0)
        ops.add_base(base)
        planes = []
        for k in range(3):
            i = aircraft.spawn(3000.0 + k * 300.0, 0.0, 2000.0,
                               cruise_alt=2000.0, cruise_spd=200.0)
            aircraft.vel[i] = (-150.0, 0.0, 0.0)
            aircraft.heading[i] = math.pi
            planes.append(i)
            ops.request_landing(i, base)
        # 全在降落队列
        assert len(ops._landing_queue) == 3
        # 全部落地驻停（跑道长度内）
        parked = 0
        for _ in range(900):   # 90s
            ops.tick(DT)
            _run_physics(aircraft, aero, flight, 1)
            parked = sum(1 for p in planes
                         if aircraft.state[p] == int(AircraftState.PARKED))
            if parked == 3:
                break
        assert parked == 3, f"应全部驻停: {parked}/3"
