"""Phase 7 步骤③：飞控（docs/phase7-air-pipeline.md §十四 步骤③验收）。

覆盖：
    巡航 10min 高度/速度误差 < 1%、航路点到达推进/转向、
    lead pursuit 占位指向拦截点、kinematic defeat 规避（急转弯+下降）、
    build_air_movement_hook 端到端（失速转 CRASHING + 箱/辐射源同步）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airis.lulilo.entities.pools import (
    AircraftPool,
    AircraftState,
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)
from Airis.lulilo.movement.aerodynamics import AircraftAero
from Airis.lulilo.movement.flight_control import (
    FlightControl,
    build_air_movement_hook,
)

G = 9.81
DT = 0.1


def _make_pool(aero_cap: int = 4) -> tuple[AircraftPool, AircraftAero]:
    pool = AircraftPool(capacity=4)
    aero = AircraftAero(aero_cap)
    return pool, aero


def _run(pool: AircraftPool, aero: AircraftAero, flight: FlightControl,
         steps: int, dt: float = DT) -> None:
    stall = np.zeros(pool.capacity, np.bool_)
    for _ in range(steps):
        flight.tick(pool, dt)
        from Airis.lulilo.movement.aerodynamics import integrate_aircraft_motion

        integrate_aircraft_motion(
            pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
            pool.throttle, pool.phi_cmd, pool.gamma_cmd,
            pool.state, pool.dmg_factor, pool.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            pool.fuel, aero.fuel_burn,
            1.225, 8500.0, G, dt, stall,
        )
        idxs = np.flatnonzero(stall & pool.active)
        if idxs.size:
            pool.state[idxs] = int(AircraftState.CRASHING)


class TestCruiseControl:
    """定高定航：巡航 10min 高度/速度误差 < 1%。"""

    def test_cruise_stability_10min(self) -> None:
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        flight = FlightControl(pool.capacity, aero)
        _run(pool, aero, flight, steps=6000)  # 600s
        # 高度误差 < 1%（30m）；速度误差 < 1%（2.5m/s）
        assert abs(pool.pos[i, 2] - 3000.0) < 30.0, f"高度误差: {pool.pos[i, 2]}"
        v = np.linalg.norm(pool.vel[i])
        assert abs(v - 250.0) < 2.5, f"速度误差: {v}"
        # 油门收敛到稳定区间（非 0 非饱和）
        assert 0.05 < pool.throttle[i] < 0.6, f"油门异常: {pool.throttle[i]}"

    def test_cruise_recovers_from_altitude_error(self) -> None:
        """初始高度偏差 → 飞控拉回巡航高度。"""
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3500.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        flight = FlightControl(pool.capacity, aero)
        _run(pool, aero, flight, steps=3000)  # 300s
        assert abs(pool.pos[i, 2] - 3000.0) < 50.0, f"未拉回巡航高度: {pool.pos[i, 2]}"

    def test_cruise_speed_recovery(self) -> None:
        """初始速度偏差 → 油门自动校正。"""
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (180.0, 0.0, 0.0)   # 低于巡航速度
        flight = FlightControl(pool.capacity, aero)
        _run(pool, aero, flight, steps=2000)
        v = np.linalg.norm(pool.vel[i])
        assert abs(v - 250.0) < 10.0, f"速度未校正: {v}"


class TestWaypoint:
    """航路点：到达推进 + 转向。"""

    def test_waypoint_progression(self) -> None:
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        # 两个航路点：正东 1km，再正东 2km
        pool.set_waypoints(i, [(1000.0, 0.0, 3000.0), (3000.0, 0.0, 3000.0)])
        flight = FlightControl(pool.capacity, aero)
        _run(pool, aero, flight, steps=600)  # 60s → 应已飞过第一个点
        assert int(pool.wp_idx[i]) >= 1, f"未到达第一个航路点: wp_idx={pool.wp_idx[i]}"

    def test_waypoint_turn_toward_target(self) -> None:
        """航路点在右侧 → 滚转朝右（正 bank）。"""
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        pool.set_waypoints(i, [(1000.0, 1000.0, 3000.0)])  # 东北方向
        flight = FlightControl(pool.capacity, aero)
        flight.tick(pool, DT)
        assert pool.phi_cmd[i] > 0.0, "目标在右侧应右滚转（正 bank）"
        assert pool.phi_cmd[i] <= FlightControl.MAX_BANK + 1e-6, "滚转超过限幅"

    def test_all_waypoints_done_holds(self) -> None:
        """航路点耗尽 → 松滚转平飞（盘旋由上层接管）。"""
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        pool.set_waypoints(i, [(10.0, 0.0, 3000.0)])  # 极近 → 立即到达
        flight = FlightControl(pool.capacity, aero)
        _run(pool, aero, flight, steps=10)
        assert int(pool.wp_idx[i]) >= int(pool.wp_count[i])
        assert pool.phi_cmd[i] == pytest.approx(0.0, abs=1e-6)


class TestManeuverPrimitives:
    """机动原语（单位 AI 消费）。"""

    def test_lead_pursuit_aims_at_intercept(self) -> None:
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        flight = FlightControl(pool.capacity, aero)
        # 目标在北侧向东高速横切（本机应转向其前方拦截点 → 右偏）
        flight.lead_pursuit(pool, i, np.array([2000.0, 2000.0, 3000.0], np.float32),
                            np.array([300.0, 0.0, 0.0], np.float32))
        assert pool.phi_cmd[i] > 0.0, "横切目标应引导向拦截点（右偏）"
        assert abs(pool.phi_cmd[i]) <= FlightControl.MAX_BANK + 1e-6

    def test_lead_pursuit_tail_chase(self) -> None:
        """目标同向逃跑 → 几乎正对（小偏角）。"""
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        flight = FlightControl(pool.capacity, aero)
        flight.lead_pursuit(pool, i, np.array([1000.0, 0.0, 3000.0], np.float32),
                            np.array([250.0, 0.0, 0.0], np.float32))
        assert abs(pool.phi_cmd[i]) < 0.1, "尾追目标应小偏角"

    def test_kinematic_defeat_banks_and_dives(self) -> None:
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        flight = FlightControl(pool.capacity, aero)
        flight.kinematic_defeat(pool, i, dir_sign=-1.0)
        assert pool.phi_cmd[i] == pytest.approx(-FlightControl.DEFEAT_BANK)
        assert pool.gamma_cmd[i] == pytest.approx(FlightControl.DEFEAT_DIVE_GAMMA)
        assert pool.throttle[i] == pytest.approx(1.0)

    def test_dive_pullup_sequence(self) -> None:
        pool, aero = _make_pool()
        i = pool.spawn(0.0, 0.0, 3000.0)
        flight = FlightControl(pool.capacity, aero)
        flight.dive(pool, i)
        assert pool.gamma_cmd[i] == pytest.approx(FlightControl.DIVE_GAMMA)
        flight.pullup(pool, i)
        assert pool.gamma_cmd[i] == pytest.approx(FlightControl.PULLUP_GAMMA)


class TestAirMovementHook:
    """build_air_movement_hook 端到端（docs §3.4）。"""

    def _rig(self):
        pool, aero = _make_pool(4)
        armor = ArmorPool(capacity=2)
        boxes = BoxPool(capacity=8)
        emitters = EmitterPool(capacity=16)
        proj = ProjectilePool(capacity=4)
        return pool, aero, armor, boxes, emitters, proj

    def test_hook_moves_and_syncs(self) -> None:
        pool, aero, armor, boxes, emitters, proj = self._rig()
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.heading[i] = 0.0
        # 机身碰撞箱 + 机载雷达
        b = boxes.spawn(i, 5, 1.2, 4.5, 0.8, local_x=0.0, local_y=0.0, local_z=0.0,
                        parent_kind=1)
        e = emitters.activate_aircraft_radar(i, 70.0)
        hook = build_air_movement_hook(pool, aero, boxes, emitters, armor, proj)
        # 模拟一个 loop 假体（hook 只读 loop）
        class _Loop:
            pass

        for _ in range(100):
            hook(_Loop(), 0.1)
        assert pool.pos[i, 0] > 500.0, "飞机应前进"
        assert abs(pool.pos[i, 2] - 3000.0) < 40.0, "高度应保持"
        # 箱/辐射源跟随飞机
        assert boxes.pos[b, 0] == pytest.approx(pool.pos[i, 0], abs=0.5)
        assert emitters.pos[e, 0] == pytest.approx(pool.pos[i, 0], abs=0.5)

    def test_hook_stall_transitions_to_crashing(self) -> None:
        pool, aero, armor, boxes, emitters, proj = self._rig()
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (50.0, 0.0, 0.0)   # 低于失速速度
        hook = build_air_movement_hook(pool, aero, boxes, emitters, armor, proj)

        class _Loop:
            pass

        hook(_Loop(), 0.1)
        assert pool.state[i] == int(AircraftState.CRASHING), "失速应转 CRASHING"

    def test_hook_ignores_crashing_aircraft(self) -> None:
        pool, aero, armor, boxes, emitters, proj = self._rig()
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.state[i] = int(AircraftState.CRASHING)
        pool.vel[i] = (100.0, 0.0, 0.0)
        hook = build_air_movement_hook(pool, aero, boxes, emitters, armor, proj)

        class _Loop:
            pass

        z0 = pool.pos[i, 2]
        for _ in range(50):
            hook(_Loop(), 0.1)
        assert pool.pos[i, 2] < z0 - 50.0, "坠毁机应弹道下坠"
        assert pool.state[i] == int(AircraftState.CRASHING)
