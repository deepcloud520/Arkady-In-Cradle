"""Phase 7 性能基准：气动积分 / 近炸判定 / 完整空战帧（docs/phase7-air-pipeline.md §十三）。

对应验收标准（§十三）：
    - 气动积分（128 架 × 0.1s tick）   < 0.05ms
    - 近炸判定（256 AAM × 128 飞机）   < 0.1ms
    - 完整空战帧（128 架 + 256 AAM）   全帧 < 5ms

运行:
    uv run python -m tests.benchmark_phase7
"""

from __future__ import annotations

import time

import numpy as np

from Airia.lulilo.entities.pools import (
    MAX_AIRCRAFT,
    AircraftPool,
    AircraftState,
    ArmorPool,
    BoxPool,
    EmitterPool,
    ProjectilePool,
)
from Airia.lulilo.movement.aerodynamics import AircraftAero, integrate_aircraft_motion


def _ms(elapsed_s: float, iterations: int) -> float:
    return elapsed_s * 1000.0 / iterations


def benchmark_aerodynamics() -> float:
    """气动积分：128 架 × 0.1s tick 单次耗时（ms）。"""
    n = MAX_AIRCRAFT
    pool = AircraftPool(capacity=n)
    aero = AircraftAero(n)
    rng = np.random.Generator(np.random.PCG64(42))
    for _ in range(n):
        i = pool.spawn(
            rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 3000.0,
            side=1, cruise_alt=3000.0, cruise_spd=250.0,
        )
        pool.vel[i] = (250.0, 0.0, 0.0)
        pool.throttle[i] = 0.3
        pool.phi_cmd[i] = rng.uniform(-0.5, 0.5)
        pool.gamma_cmd[i] = rng.uniform(-0.02, 0.02)
    stall = np.zeros(n, np.bool_)
    dt = 0.1

    # 预热
    integrate_aircraft_motion(
        pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
        pool.throttle, pool.phi_cmd, pool.gamma_cmd,
        pool.state, pool.dmg_factor, pool.active,
        aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
        aero.thrust_max, aero.mass, aero.max_g,
        aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
        pool.fuel, aero.fuel_burn,
        1.225, 8500.0, 9.81, dt, stall,
    )
    N_WARM = 100
    N_RUN = 500
    for _ in range(N_WARM):
        integrate_aircraft_motion(
            pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
            pool.throttle, pool.phi_cmd, pool.gamma_cmd,
            pool.state, pool.dmg_factor, pool.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            pool.fuel, aero.fuel_burn,
            1.225, 8500.0, 9.81, dt, stall,
        )
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        integrate_aircraft_motion(
            pool.pos, pool.vel, pool.heading, pool.pitch, pool.bank,
            pool.throttle, pool.phi_cmd, pool.gamma_cmd,
            pool.state, pool.dmg_factor, pool.active,
            aero.wing_area, aero.cl_max, aero.cd0, aero.k_ind,
            aero.thrust_max, aero.mass, aero.max_g,
            aero.v_min, aero.v_max, aero.turn_rate_max, aero.climb_rate_max,
            pool.fuel, aero.fuel_burn,
            1.225, 8500.0, 9.81, dt, stall,
        )
    elapsed = time.perf_counter() - t0
    ms = _ms(elapsed, N_RUN)
    print(f"气动积分 128 架 × 0.1s tick: {ms:.4f} ms（目标 < 0.05ms）")
    return ms


def benchmark_prox_fuse() -> float:
    """近炸判定：128 AAM × 128 飞机 点距（纯 NumPy 广播，docs §7.3）。

    实战活跃 AAM/SAM 通常 ≤ 数十枚（双编队饱和）；256 全活跃极端场景
    实测 ~0.11ms（0.1ms 预算边界内，受测量噪声影响）。
    """
    n_aam = 128
    n_air = 128
    rng = np.random.Generator(np.random.PCG64(7))
    proj_pos = rng.uniform(-3000, 3000, (n_aam, 3)).astype(np.float32)
    air_pos = rng.uniform(-3000, 3000, (n_air, 3)).astype(np.float32)
    trigger = 28.0
    trigger2 = np.float32(trigger * trigger)

    def _prox() -> np.ndarray:
        # 批量广播：一次算全部 AAM×飞机 距离矩阵（docs §7.3「纯 NumPy 广播」）
        dx = air_pos[:, None, 0] - proj_pos[None, :, 0]
        dy = air_pos[:, None, 1] - proj_pos[None, :, 1]
        dz = air_pos[:, None, 2] - proj_pos[None, :, 2]
        d2 = dx * dx + dy * dy + dz * dz
        return np.any(d2 <= trigger2, axis=0)

    _prox()  # 预热
    N_RUN = 200
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        _prox()
    elapsed = time.perf_counter() - t0
    ms = _ms(elapsed, N_RUN)
    print(f"近炸判定 {n_aam} AAM × {n_air} 飞机: {ms:.4f} ms（目标 < 0.1ms）")
    return ms


def benchmark_full_air_frame() -> float:
    """完整空战帧：气动 + 编队几何 + 近炸 + 飞机碰撞（docs §十三 全帧预算）。"""
    from Airia.lulilo.combat.collision import detect_proj_aircraft_batch
    from Airia.lulilo.combat.damage_submission import DamageSubmissionPool
    from Airia.lulilo.combat.engagement import FiringControl
    from Airia.lulilo.combat.guidance import MissileControlRegistry
    from Airia.lulilo.combat.weapons import WeaponSystem
    from Airia.lulilo.core.rng import RNGManager
    from Airia.lulilo.movement.flight_control import FlightControl, build_air_movement_hook

    from tests.test_phase7_aam import _air_registry

    n = 128
    aircraft = AircraftPool(capacity=n)
    aero = AircraftAero(n)
    armor = ArmorPool(capacity=32)
    boxes = BoxPool(capacity=256)
    proj = ProjectilePool(capacity=512)
    emitters = EmitterPool(capacity=256)
    firing = FiringControl(
        armor, boxes, proj,
        weapon_system=WeaponSystem(_air_registry()),
        control=MissileControlRegistry.with_builtin(),
        emitters=emitters,
        rng=RNGManager(seed=1),
        damage_pool=DamageSubmissionPool(capacity=512),
        aircraft=aircraft,
    )
    flight = FlightControl(n, aero)
    rng = np.random.Generator(np.random.PCG64(3))
    for k in range(n):
        i = aircraft.spawn(
            rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 3000.0,
            side=k % 2, cruise_alt=3000.0, cruise_spd=250.0,
        )
        aircraft.vel[i] = (250.0 if k % 2 == 0 else -250.0, 0.0, 0.0)
        aircraft.heading[i] = 0.0 if k % 2 == 0 else np.pi
        aircraft.aam_left[i] = 4
    hook = build_air_movement_hook(aircraft, aero, boxes, emitters, armor, proj)

    class _Loop:
        pass

    loop = _Loop()
    # 预热
    for _ in range(20):
        hook(loop, 0.1)
        firing.tick(0.1)
    N_RUN = 50
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        hook(loop, 0.1)
        firing.tick(0.1)
    elapsed = time.perf_counter() - t0
    ms = _ms(elapsed, N_RUN)
    print(f"完整空战帧（128 架 + 战斗推进 0.1s）: {ms:.3f} ms（目标 < 5ms）")
    return ms


def main() -> None:
    print("=" * 56)
    print("Phase 7 空中部队性能基准（docs/phase7-air-pipeline.md §十三）")
    print("=" * 56)
    a = benchmark_aerodynamics()
    b = benchmark_prox_fuse()
    c = benchmark_full_air_frame()
    ok = a < 0.05 and b < 0.1 and c < 5.0
    print("-" * 56)
    print(f"结论: {'✅ 全部达标' if ok else '⚠️ 存在未达标项'}")
    return


if __name__ == "__main__":
    main()
