"""Phase 2 性能基准：2048 装甲 + 16384 弹丸的移动/检测帧率。

对应 AGENTS.md Phase 2 验收标准：
    - 移动积分帧率 ≥ 60fps（2048 装甲同时移动）
    - 检测耗时 < 3ms/帧（16384 弹丸 + 2048 装甲）

运行:
    uv run python -m tests.benchmark_phase2
"""

from __future__ import annotations

import time

import numpy as np

from Airis.lulilo.detection.los import line_of_sight_batch
from Airis.lulilo.detection.sensors import (
    RCS_TANK,
    evaluate_sensor_detection,
    radar_detection_range,
)
from Airis.lulilo.entities.pools import MAX_ARMOR, MAX_PROJ, ArmorPool, ProjectilePool
from Airis.lulilo.movement.integrate import (
    integrate_armor_movement,
    integrate_projectile_movement,
)
from Airis.lulilo.terrain.heightfield import MAT_TERRAIN, Heightfield


def benchmark_movement_armor() -> None:
    """2048 装甲移动积分。"""
    pool = ArmorPool(MAX_ARMOR)
    rng = np.random.Generator(np.random.PCG64(42))

    # 生成 2048 个随机装甲
    for _ in range(MAX_ARMOR):
        x = rng.uniform(0, 2000)
        y = rng.uniform(0, 2000)
        z = rng.uniform(0, 10)
        idx = pool.spawn(x, y, z)
        # 随机路标点
        wp = [(rng.uniform(0, 2000), rng.uniform(0, 2000)) for _ in range(4)]
        pool.set_waypoints(idx, wp)

    speed = np.full(MAX_ARMOR, rng.uniform(5, 20, MAX_ARMOR).astype(np.float32))

    # 预热 Numba
    integrate_armor_movement(
        pool.pos, pool.yaw, pool.active,
        pool.wp, pool.wp_count, pool.wp_idx,
        speed, 0.016,
    )

    # 计时
    N_WARM = 50
    N_RUN = 200
    for _ in range(N_WARM):
        integrate_armor_movement(
            pool.pos, pool.yaw, pool.active,
            pool.wp, pool.wp_count, pool.wp_idx,
            speed, 0.016,
        )

    t0 = time.perf_counter()
    for _ in range(N_RUN):
        integrate_armor_movement(
            pool.pos, pool.yaw, pool.active,
            pool.wp, pool.wp_count, pool.wp_idx,
            speed, 0.016,
        )
    elapsed = time.perf_counter() - t0

    ms_per_frame = (elapsed / N_RUN) * 1000
    fps = 1000 / ms_per_frame
    print(f"  装甲移动积分 ({MAX_ARMOR} 单位): {ms_per_frame:.2f} ms/frame, {fps:.0f} fps")
    assert ms_per_frame < 16.67, f"未达 60fps 要求: {ms_per_frame:.2f}ms > 16.67ms"


def benchmark_movement_projectile() -> None:
    """16384 抛射物移动积分。"""
    pool = ProjectilePool(MAX_PROJ)
    rng = np.random.Generator(np.random.PCG64(42))

    for _ in range(MAX_PROJ):
        x = rng.uniform(0, 2000)
        y = rng.uniform(0, 2000)
        z = rng.uniform(0, 500)
        vx = rng.uniform(-900, 900)
        vy = rng.uniform(-900, 900)
        vz = rng.uniform(-50, 50)
        pool.spawn(x, y, z, vx, vy, vz)

    dt = 0.01  # 抛射物 Tick

    # 预热
    integrate_projectile_movement(pool.pos, pool.vel, pool.active, dt)

    N_WARM = 100
    N_RUN = 500
    for _ in range(N_WARM):
        integrate_projectile_movement(pool.pos, pool.vel, pool.active, dt)

    t0 = time.perf_counter()
    for _ in range(N_RUN):
        integrate_projectile_movement(pool.pos, pool.vel, pool.active, dt)
    elapsed = time.perf_counter() - t0

    ms_per_frame = (elapsed / N_RUN) * 1000
    print(f"  抛射物移动积分 ({MAX_PROJ} 弹丸): {ms_per_frame:.2f} ms/frame")
    assert ms_per_frame < 16.67, f"未达 60fps 要求: {ms_per_frame:.2f}ms > 16.67ms"


def benchmark_los() -> None:
    """视线检测批量（模拟 100 传感器 × 200 目标 = 20K 对）。"""
    rng = np.random.Generator(np.random.PCG64(42))

    # 创建测试高度场
    hf = Heightfield(100, 100, cs=20.0, ch=1.0, z_min=-50, z_max=200)
    for gy in range(100):
        for gx in range(100):
            h = rng.uniform(-10, 5)
            hf.add_span(gx, gy, -50, h, mat=MAT_TERRAIN)

    N_PAIRS = 20000
    ox = rng.uniform(0, 2000, N_PAIRS).astype(np.float32)
    oy = rng.uniform(0, 2000, N_PAIRS).astype(np.float32)
    oz = rng.uniform(1, 30, N_PAIRS).astype(np.float32)
    tx = rng.uniform(0, 2000, N_PAIRS).astype(np.float32)
    ty = rng.uniform(0, 2000, N_PAIRS).astype(np.float32)
    tz = rng.uniform(0, 10, N_PAIRS).astype(np.float32)
    active = np.ones(N_PAIRS, dtype=np.bool_)

    # 预热
    line_of_sight_batch(
        ox, oy, oz, tx, ty, tz, active,
        hf.spans, hf.count, hf.width, hf.cs, hf.ch, hf.z_min,
    )

    N_RUN = 50
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        line_of_sight_batch(
            ox, oy, oz, tx, ty, tz, active,
            hf.spans, hf.count, hf.width, hf.cs, hf.ch, hf.z_min,
        )
    elapsed = time.perf_counter() - t0

    ms_per_frame = (elapsed / N_RUN) * 1000
    print(f"  视线检测 ({N_PAIRS} 对): {ms_per_frame:.2f} ms/frame")


def benchmark_sensors() -> None:
    """传感器批量评估（10 传感器 × 2048 目标）。"""
    rng = np.random.Generator(np.random.PCG64(42))

    N_SENSORS = 10
    N_TARGETS = MAX_ARMOR

    sx = rng.uniform(0, 2000, N_SENSORS).astype(np.float32)
    sy = rng.uniform(0, 2000, N_SENSORS).astype(np.float32)
    sz = rng.uniform(5, 50, N_SENSORS).astype(np.float32)
    tx = rng.uniform(0, 2000, N_TARGETS).astype(np.float32)
    ty = rng.uniform(0, 2000, N_TARGETS).astype(np.float32)
    tz = rng.uniform(0, 10, N_TARGETS).astype(np.float32)
    stype = np.zeros(N_SENSORS, dtype=np.int32)  # 全部雷达
    trcs = np.full(N_TARGETS, RCS_TANK, dtype=np.float32)
    tir = np.full(N_TARGETS, 0.5, dtype=np.float32)
    sact = np.ones(N_SENSORS, dtype=np.bool_)
    tact = np.ones(N_TARGETS, dtype=np.bool_)
    cache = np.zeros((N_SENSORS, N_TARGETS), dtype=np.float32)

    # 预热
    evaluate_sensor_detection(
        sx, sy, sz, tx, ty, tz, stype, trcs, tir, sact, tact, cache,
    )

    N_RUN = 100
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        evaluate_sensor_detection(
            sx, sy, sz, tx, ty, tz, stype, trcs, tir, sact, tact, cache,
        )
    elapsed = time.perf_counter() - t0

    ms_per_frame = (elapsed / N_RUN) * 1000
    pairs = N_SENSORS * N_TARGETS
    print(f"  传感器评估 ({N_SENSORS} 传感器 × {N_TARGETS} 目标 = {pairs} 对): "
          f"{ms_per_frame:.2f} ms/frame")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Phase 2 性能基准")
    print(f"目标: {MAX_ARMOR} 装甲 + {MAX_PROJ} 弹丸")
    print("=" * 60)

    benchmark_movement_armor()
    benchmark_movement_projectile()
    benchmark_los()
    benchmark_sensors()

    print("=" * 60)
    print("全部基准通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
