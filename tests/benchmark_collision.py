"""Phase 3 性能基准：空间哈希碰撞检测 + 完整开火链吞吐。

对应 AGENTS.md Phase 3 验收标准：
    - 碰撞检测耗时 < 3ms/帧（16384 弹丸 + 2048 装甲）
    - 完整交战从开火到命中 < 1 个 Tick 内完成

运行:
    uv run python -m tests.benchmark_collision
"""

from __future__ import annotations

import time

import numpy as np

from Airis.lulilo.combat.collision import CollisionDetector
from Airis.lulilo.combat.damage import ArmorType
from Airis.lulilo.combat.engagement import FiringControl, UnitCombatConfig
from Airis.lulilo.core.rng import RNGManager
from Airis.lulilo.entities.pools import (
    MAX_ARMOR,
    MAX_BOXES,
    MAX_PROJ,
    ArmorPool,
    BoxPool,
    PartType,
    ProjectilePool,
)
from Airis.lulilo.logistics.ammo import AmmoManager
from Airis.lulilo.logistics.fuel import FuelManager
from Airis.lulilo.movement.integrate import integrate_projectile_movement


# ---------------------------------------------------------------------------
# 基准 1：空间哈希 + 胶囊体碰撞（16384 弹丸 × 2048 装甲）
# ---------------------------------------------------------------------------


def benchmark_collision() -> float:
    """空间哈希碰撞检测：16384 弹丸 + 2048 装甲，返回单帧耗时 (ms)。"""
    rng = np.random.Generator(np.random.PCG64(42))

    armor = ArmorPool(MAX_ARMOR)
    for _ in range(MAX_ARMOR):
        armor.spawn(
            rng.uniform(-2000, 2000),
            rng.uniform(-2000, 2000),
            rng.uniform(0, 5),
            half_x=1.5, half_y=3.0, half_z=1.0,
        )

    proj = ProjectilePool(MAX_PROJ)
    v_scale = 300.0
    for _ in range(MAX_PROJ):
        vx = rng.uniform(-1, 1)
        vy = rng.uniform(-1, 1)
        vn = np.sqrt(vx * vx + vy * vy) + 1e-9
        proj.spawn(
            rng.uniform(-2000, 2000),
            rng.uniform(-2000, 2000),
            rng.uniform(0, 5),
            vx / vn * v_scale, vy / vn * v_scale, 0.0,
            radius=0.08,
        )

    detector = CollisionDetector(grid_size=50.0)
    detector.build_index(armor.pos, armor.half, armor.active)

    # 预热 Numba 内核
    detector.detect(
        proj.pos, proj.vel, proj.radius, proj.active,
        armor.pos, armor.half, armor.active, 0.01,
    )

    N_WARM = 3
    N_RUN = 20
    dt = 0.01  # 子弹 Tick
    for _ in range(N_WARM):
        detector.detect(
            proj.pos, proj.vel, proj.radius, proj.active,
            armor.pos, armor.half, armor.active, dt,
        )

    t0 = time.perf_counter()
    total_hits = 0
    for _ in range(N_RUN):
        hits = detector.detect(
            proj.pos, proj.vel, proj.radius, proj.active,
            armor.pos, armor.half, armor.active, dt,
        )
        total_hits += len(hits)
    elapsed_ms = (time.perf_counter() - t0) / N_RUN * 1000.0

    print(f"[1] 碰撞检测（{MAX_PROJ} 弹丸 × {MAX_ARMOR} 装甲）:")
    print(f"    平均 {elapsed_ms:.2f} ms/帧  目标 < 3ms  {'✅' if elapsed_ms < 3.0 else '❌'}")
    print(f"    平均命中 {total_hits / N_RUN:.1f} 次/帧")
    return elapsed_ms


# ---------------------------------------------------------------------------
# 基准 2：完整开火链吞吐（FiringControl.tick 一帧）
# ---------------------------------------------------------------------------


def benchmark_firing_chain() -> float:
    """开火链吞吐：多对坦克互射，tick 一帧（子弹 0.01s）的耗时。"""
    rng = np.random.Generator(np.random.PCG64(7))

    armor = ArmorPool(capacity=MAX_ARMOR)
    boxes = BoxPool(capacity=MAX_BOXES)
    projectiles = ProjectilePool(capacity=MAX_PROJ)
    ammo = AmmoManager(capacity=MAX_ARMOR)
    fuel = FuelManager(capacity=MAX_ARMOR)
    fc = FiringControl(
        armor, boxes, projectiles, ammo=ammo, fuel=fuel,
        rng=RNGManager(seed=7),
    )

    # 2048 坦克，随机配对方向（一半朝 x+，一半朝 x-），散布在战场
    n = 2048
    for i in range(n):
        x = (i % 64) * 40.0 - 1260.0
        y = (i // 64) * 40.0 - 1260.0
        yaw = 0.0 if i % 2 == 0 else np.pi
        idx = armor.spawn(x, y, 0.0, half_x=1.5, half_y=3.0, half_z=1.0, yaw=yaw)
        for part, hx, hy, hz, lx, ly, lz in [
            (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
            (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
            (PartType.ENGINE, 0.8, 1.0, 0.6, -1.2, 0.0, 0.0),
            (PartType.TRACK, 1.4, 0.6, 0.5, 0.0, 1.9, 0.0),
            (PartType.AMMO_RACK, 0.7, 0.8, 0.5, 0.5, 0.0, -0.3),
        ]:
            boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
        fc.set_unit_config(
            idx,
            UnitCombatConfig(
                weapon_id="m256_120mm",
                armor_mm=300.0,
                armor_type=int(ArmorType.RHA),
            ),
        )
        ammo.state.set_unit(idx, 100.0, 100.0)
        fuel.state.set_unit(idx, 500.0, 500.0)
    boxes.sync_all_from_parent(armor)

    # 每辆朝正前方最近的坦克开火
    fired = 0
    for i in range(n):
        if not armor.active[i]:
            continue
        target = None
        best = 1e18
        xi, yi = armor.pos[i, 0], armor.pos[i, 1]
        for j in range(n):
            if j == i or not armor.active[j]:
                continue
            # 只打同列朝向我方的前方目标（简化）
            if abs(armor.pos[j, 1] - yi) < 5.0:
                d = abs(armor.pos[j, 0] - xi)
                if 0 < d < best and ((armor.yaw[i] == 0.0 and armor.pos[j, 0] > xi)
                                     or (armor.yaw[i] != 0.0 and armor.pos[j, 0] < xi)):
                    best = d
                    target = j
        if target is not None:
            if fc.fire(i, target) is not None:
                fired += 1

    # 预热 + 计时 tick（子弹 0.01s 一帧）
    fc.tick(0.01)
    N_WARM = 20
    N_RUN = 100
    for _ in range(N_WARM):
        fc.tick(0.01)
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        fc.tick(0.01)
    elapsed_ms = (time.perf_counter() - t0) / N_RUN * 1000.0

    print(f"[2] 开火链一帧（{n} 坦克互射，{fired} 发在飞）:")
    print(f"    平均 {elapsed_ms:.2f} ms/tick(0.01s)  {'✅' if elapsed_ms < 3.0 else '❌'}")
    print(f"    单帧可承载 {(1000.0 / elapsed_ms):.0f} tick/s")
    return elapsed_ms


# ---------------------------------------------------------------------------
# 基准 3：从开火到命中延迟（Tick 数）
# ---------------------------------------------------------------------------


def benchmark_hit_latency() -> float:
    """从开火到命中的事件延迟（Tick 数），验证「< 1 个 Tick 内完成」。"""
    armor = ArmorPool(capacity=8)
    boxes = BoxPool(capacity=64)
    projectiles = ProjectilePool(capacity=256)
    ammo = AmmoManager(capacity=8)
    fuel = FuelManager(capacity=8)
    fc = FiringControl(
        armor, boxes, projectiles, ammo=ammo, fuel=fuel,
        rng=RNGManager(seed=1),
    )

    m1 = armor.spawn(0.0, 0.0, 0.0, half_x=1.5, half_y=3.0, half_z=1.0, yaw=0.0)
    t90 = armor.spawn(500.0, 0.0, 0.0, half_x=1.5, half_y=3.0, half_z=1.0, yaw=np.pi)
    for part, hx, hy, hz, lx, ly, lz in [
        (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
        (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
    ]:
        boxes.spawn(m1, int(part), hx, hy, hz, lx, ly, lz)
        boxes.spawn(t90, int(part), hx, hy, hz, lx, ly, lz)
    boxes.sync_all_from_parent(armor)
    fc.set_unit_config(m1, UnitCombatConfig(weapon_id="m256_120mm", armor_mm=300.0))
    fc.set_unit_config(t90, UnitCombatConfig(weapon_id="2a46m_125mm", armor_mm=300.0))
    ammo.state.set_unit(m1, 100.0, 100.0)
    ammo.state.set_unit(t90, 100.0, 100.0)

    fc.fire(m1, t90)
    ticks = 0
    while fc.pending_projectiles() > 0 and ticks < 200:
        fc.tick(0.01)
        ticks += 1

    flight_s = ticks * 0.01
    print(f"[3] 开火→命中延迟: {ticks} 个子弹 Tick = {flight_s:.2f}s "
          f"(500m @1555m/s 理论 {500 / 1555:.2f}s)")
    print(f"    {'✅ 远小于 1 个地面 Tick(1s)' if flight_s < 1.0 else '❌'}")
    return flight_s


def main() -> None:
    print("=" * 60)
    print("Phase 3 性能基准：碰撞检测 / 开火链 / 命中延迟")
    print("=" * 60)
    benchmark_collision()
    print("-" * 60)
    benchmark_firing_chain()
    print("-" * 60)
    benchmark_hit_latency()
    print("=" * 60)


if __name__ == "__main__":
    main()
