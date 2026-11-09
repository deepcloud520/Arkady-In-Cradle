"""Phase 7 步骤②：气动内核 + 大气模型（docs/phase7-air-pipeline.md §十四 步骤②验收）。

覆盖：
    平飞巡航（高度/速度长期保持）、爬升（γ̇_cmd → 高度上升）、
    bank-to-turn 转弯（φ_cmd → 航向变化）、失速判定（V<v_min → stall_out）、
    坠毁（CRASHING → 重力弹道下落）、燃油扣减、AircraftAero 广播/扩容。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.entities.pools import AircraftPool, AircraftState
from Airia.lulilo.movement.aerodynamics import (
    AircraftAero,
    integrate_aircraft_motion,
)

G = 9.81
RHO0 = 1.225
H = 8500.0
DT = 0.1


def _default_arrays(n: int = 1):
    """构造单机默认气动/包线数组（对齐 AircraftConfig 默认战斗机基线）。"""
    return dict(
        wing_area=np.full(n, 30.0, np.float32),
        cl_max=np.full(n, 1.6, np.float32),
        cd0=np.full(n, 0.02, np.float32),
        k_ind=np.full(n, 0.05, np.float32),
        thrust_max=np.full(n, 100000.0, np.float32),
        mass=np.full(n, 15000.0, np.float32),
        max_g=np.full(n, 9.0, np.float32),
        v_min=np.full(n, 80.0, np.float32),
        v_max=np.full(n, 750.0, np.float32),
        turn_rate_max=np.full(n, np.radians(18.0), np.float32),
        climb_rate_max=np.full(n, 150.0, np.float32),
        fuel_burn=np.full(n, 1.0 / 5400.0, np.float32),
    )


def _step(arrs, pool, dt=DT, steps=1, stall_out=None):
    stall = stall_out if stall_out is not None else np.zeros(pool.capacity, np.bool_)
    for _ in range(steps):
        integrate_aircraft_motion(
            pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
            pool.throttle, pool.phi_cmd, pool.gamma_cmd,
            pool.state, pool.dmg_factor, pool.active,
            arrs["wing_area"], arrs["cl_max"], arrs["cd0"], arrs["k_ind"],
            arrs["thrust_max"], arrs["mass"], arrs["max_g"],
            arrs["v_min"], arrs["v_max"], arrs["turn_rate_max"], arrs["climb_rate_max"],
            pool.fuel, arrs["fuel_burn"],
            RHO0, H, G, dt, stall,
        )
    return stall


class TestLevelCruise:
    """平飞巡航：高度/速度长期保持（管线 §3.1 验收：巡航 10min 误差 < 1%）。"""

    def test_altitude_stable_10min(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.175   # 近似平衡油门（a_T ≈ a_D）
        arrs = _default_arrays()
        _step(arrs, pool, steps=6000)  # 600s
        # 高度漂移 < 1%（30m）；速度漂移 < 2%（5 m/s）
        assert abs(pool.pos[i, 2] - 3000.0) < 30.0, f"高度漂移过大: {pool.pos[i, 2]}"
        v = np.linalg.norm(pool.vel[i])
        assert abs(v - 250.0) < 5.0, f"速度漂移过大: {v}"

    def test_no_stall_at_cruise(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.175
        stall = _step(_default_arrays(), pool, steps=600)
        assert not stall[i], "巡航速度不应触发失速"

    def test_high_altitude_requires_more_throttle(self) -> None:
        """高空空气稀薄 → 同油门阻力小、平衡速度更高（ρ(z) 指数衰减生效）。"""
        p1 = AircraftPool(capacity=1)
        i1 = p1.spawn(0.0, 0.0, 500.0, cruise_alt=500.0, cruise_spd=250.0)
        p1.vel[i1] = (250.0, 0.0, 0.0)
        p1.throttle[i1] = 0.5
        p2 = AircraftPool(capacity=1)
        i2 = p2.spawn(0.0, 0.0, 9000.0, cruise_alt=9000.0, cruise_spd=250.0)
        p2.vel[i2] = (250.0, 0.0, 0.0)
        p2.throttle[i2] = 0.5
        arrs1 = _default_arrays()
        arrs2 = _default_arrays()
        _step(arrs1, p1, steps=300)
        _step(arrs2, p2, steps=300)
        v1 = np.linalg.norm(p1.vel[i1])
        v2 = np.linalg.norm(p2.vel[i2])
        assert v2 > v1, f"高空速度({v2:.1f})应高于低空({v1:.1f})（稀薄空气 → 阻力小 → 平衡速度高）"


class TestClimb:
    """爬升：γ̇_cmd > 0 → 高度上升。"""

    def test_climb_gains_altitude(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.6
        pool.gamma_cmd[i] = 0.05   # 2.9°/s 持续爬升指令
        _step(_default_arrays(), pool, steps=1000)  # 100s
        assert pool.pos[i, 2] > 3500.0, f"100s 爬升高度不足: {pool.pos[i, 2]}"

    def test_dive_descends(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.3
        pool.gamma_cmd[i] = -0.05  # 俯冲指令
        _step(_default_arrays(), pool, steps=500)
        assert pool.pos[i, 2] < 2800.0, f"50s 俯冲高度未降: {pool.pos[i, 2]}"


class TestTurn:
    """bank-to-turn：φ_cmd → 航向变化（升力侧向分量 → ψ̇）。"""

    def test_bank_turns_heading(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.2
        pool.phi_cmd[i] = np.radians(30.0)
        h0 = pool.heading[i]
        _step(_default_arrays(), pool, steps=100)  # 10s
        dpsi = pool.heading[i] - h0
        # 解析：a_L = g/cos30° ≈ 11.33 → ψ̇ = a_L·sin30/V ≈ 0.0227 rad/s → 10s ≈ 0.227 rad
        assert dpsi > 0.15, f"转弯不足: {dpsi:.3f} rad"
        assert dpsi < 0.30, f"转弯过大: {dpsi:.3f} rad"

    def test_bank_limited_by_turn_rate(self) -> None:
        """小 turn_rate_max → ψ̇ 被限幅（验证包线限幅生效，非自然转弯率）。

        条件：V=200, φ=85° → 自然 ψ̇ ≈ 14°/s，远超 5°/s 限幅。
        """
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (200.0, 0.0, 0.0)
        pool.throttle[i] = 0.3
        pool.phi_cmd[i] = np.radians(85.0)
        arrs = _default_arrays()
        arrs["turn_rate_max"][0] = np.radians(5.0)   # 人为调小限幅 5°/s
        h0 = pool.heading[i]
        _step(arrs, pool, steps=100)  # 10s → 自然转弯 140° 应被限到 ~50°
        dpsi = pool.heading[i] - h0
        upper = np.radians(5.0) * 10.0
        assert dpsi <= upper + 1e-2, f"转弯率未限幅: {np.degrees(dpsi):.1f}°"
        assert dpsi > upper * 0.8, f"转弯率异常小: {np.degrees(dpsi):.1f}°（限幅应饱和）"


class TestStall:
    """失速判定：V < v_min·(1+0.3·(1−dmg)) → stall_out。"""

    def test_stall_below_vmin(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (50.0, 0.0, 0.0)   # 低于失速速度 80
        stall = _step(_default_arrays(), pool, steps=1)
        assert stall[i], "低于失速速度应触发 stall_out"
        v = np.linalg.norm(pool.vel[i])
        assert v >= 80.0 - 1e-3, "失速后速度被钳到 v_min"

    def test_damaged_stall_speed_rises(self) -> None:
        """损伤（dmg_factor=0.6）→ 失速速度上升至 v_min·(1+0.3·0.4)=89.6。"""
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (85.0, 0.0, 0.0)   # 介于 80 与 89.6 之间
        pool.dmg_factor[i] = 0.6
        stall = _step(_default_arrays(), pool, steps=1)
        assert stall[i], "重伤飞机在 85 m/s 应已失速"


class TestCrash:
    """坠毁链气动部分：CRASHING → 重力弹道下落（升力≈0，推力=0）。"""

    def test_crashing_descends_ballistic(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.state[i] = int(AircraftState.CRASHING)
        pool.vel[i] = (100.0, 0.0, 0.0)
        pool.throttle[i] = 1.0          # 推力应被忽略
        _step(_default_arrays(), pool, steps=200)  # 20s
        assert pool.pos[i, 2] < 2900.0, "坠毁机应持续下坠"
        assert pool.vel[i, 2] < 0.0, "坠毁机垂直速度应为负"
        # 水平速度不被推力改变（仅空气摩擦项被忽略的惯性滑行）
        assert pool.vel[i, 0] == pytest.approx(100.0, abs=1e-3)

    def test_crashing_ignores_climb_command(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.state[i] = int(AircraftState.CRASHING)
        pool.vel[i] = (100.0, 0.0, 20.0)
        pool.gamma_cmd[i] = 0.5         # 爬升指令应被忽略
        _step(_default_arrays(), pool, steps=100)  # 10s（20 m/s 初速被重力抵消）
        assert pool.vel[i, 2] < 0.0, "坠毁机无视爬升指令"


class TestFuel:
    """燃油扣减（内核线性消耗）。"""

    def test_fuel_burn_over_time(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.175
        _step(_default_arrays(), pool, steps=6000)  # 600s
        expected = 1.0 - 600.0 / 5400.0
        assert pool.fuel[i] == pytest.approx(expected, abs=1e-3)

    def test_fuel_floor_zero(self) -> None:
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.175
        _step(_default_arrays(), pool, steps=60000)  # 6000s >> 5400s
        assert pool.fuel[i] == pytest.approx(0.0, abs=1e-6)


class TestAircraftAero:
    """AircraftAero 容器：鸭子类型广播 + 扩容回填。"""

    class _Cfg:
        wing_area_m2 = 25.0
        cl_max = 1.8
        cd0 = 0.015
        k_ind = 0.06
        thrust_max = 120000.0
        mass_kg = 18000.0
        max_g = 8.0
        v_min_m_s = 90.0
        v_max_m_s = 800.0
        turn_rate_max_deg_s = 22.0
        climb_rate_m_s = 180.0
        fuel_capacity_s = 7200.0

    def test_apply_broadcasts(self) -> None:
        aero = AircraftAero(4)
        aero.apply(2, self._Cfg())
        assert aero.wing_area[2] == pytest.approx(25.0)
        assert aero.cl_max[2] == pytest.approx(1.8)
        assert aero.thrust_max[2] == pytest.approx(120000.0)
        assert aero.mass[2] == pytest.approx(18000.0)
        assert aero.turn_rate_max[2] == pytest.approx(np.radians(22.0))
        assert aero.fuel_burn[2] == pytest.approx(1.0 / 7200.0)
        # 未应用索引保持默认
        assert aero.wing_area[0] == pytest.approx(30.0)

    def test_resize_backfills_defaults(self) -> None:
        aero = AircraftAero(2)
        aero.apply(0, self._Cfg())
        aero.resize(6)
        assert aero.capacity == 6
        assert aero.wing_area[0] == pytest.approx(25.0)   # 旧值保留
        assert aero.wing_area[5] == pytest.approx(30.0)   # 新段默认
        assert aero.turn_rate_max[5] == pytest.approx(np.radians(18.0))
        assert aero.fuel_burn[5] == pytest.approx(1.0 / 5400.0)

    def test_integrate_with_aero_arrays(self) -> None:
        """用 AircraftAero 容器数组跑内核（端到端冒烟）。"""
        pool = AircraftPool(capacity=1)
        i = pool.spawn(0.0, 0.0, 3000.0, cruise_alt=3000.0, cruise_spd=250.0)
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.175
        aero = AircraftAero(1)
        aero.apply(0, self._Cfg())
        stall = np.zeros(1, np.bool_)
        for _ in range(600):
            integrate_aircraft_motion(
                pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
                pool.throttle, pool.phi_cmd, pool.gamma_cmd,
                pool.state, pool.dmg_factor, pool.active,
                aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
                aero.thrust_max, aero.mass, aero.max_g,
                aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
                pool.fuel, aero.fuel_burn,
                RHO0, H, G, 0.1, stall,
            )
        assert not stall[i]
        v = np.linalg.norm(pool.vel[i])
        assert abs(v - 250.0) < 40.0, f"自定义机型巡航速度异常: {v}"
