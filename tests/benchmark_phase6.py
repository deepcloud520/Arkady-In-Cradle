"""Phase 6c 性能基准（docs/phase6-ai-pipeline.md §十二.3）。

验收目标：
    - 编队/单位 AI 决策 < 1ms/帧（1s Tick 摊薄）
    - 伤害提交池结算 < 1ms/帧
    - 整体（AI + combat + 提交池）在 2048 装甲 + 16384 弹丸下可持续运行

运行:
    uv run python -m tests.benchmark_phase6
"""

from __future__ import annotations

import time

import numpy as np

from Airia.lulilo.combat.damage_submission import DamageKind, DamageSubmissionPool, resolve_area_damage
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
from Airia.lulilo.simulation.ai import (
    ArtilleryAI,
    CommunicationNetwork,
    FormationAI,
    MuzzleFlashLog,
    UnitRole,
)
from Airia.lulilo.simulation.loop import EventBus, SimulationLoop

_SIMPLE_PARTS = [
    (PartType.HULL, 1.5, 3.0, 1.0, 0.0, 0.0, 0.0),
    (PartType.TURRET, 1.0, 1.2, 0.6, 0.0, 0.0, 0.6),
]

N_ARMOR = 2048
N_FORMATIONS = 256
N_PROJ = 16384


def _build_world(*, ammo_amount: float = 40.0):
    """构建 2048 装甲（256 编队）+ 16384 弹丸容量的完整仿真世界。

    ammo_amount=0 → 单位无弹药（`can_fire` False），隔离「纯 AI 决策」成本
    （不含开火链：weapon_system.solve / 抛射物 spawn / 事件）。
    """
    armor = ArmorPool(capacity=N_ARMOR)
    boxes = BoxPool(capacity=N_ARMOR * 2)
    projectiles = ProjectilePool(capacity=N_PROJ)
    ammo = AmmoManager(capacity=N_ARMOR)
    events = EventBus()
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, events=events, rng=RNGManager(seed=0),
    )
    formations = FormationAIPool(capacity=N_FORMATIONS)

    per_form = N_ARMOR // N_FORMATIONS  # 8
    for fidx in range(N_FORMATIONS):
        side = fidx % 2
        role = int(UnitRole.TANK)
        formations.spawn(side=side, role=role)
        for k in range(per_form):
            idx = armor.spawn(
                float((fidx % 32) * 100.0 + k * 5.0),
                float((fidx // 32) * 100.0),
                0.0,
                half_x=1.5, half_y=3.0, half_z=1.0,
                side=side, role=role, formation_id=fidx,
            )
            for part, hx, hy, hz, lx, ly, lz in _SIMPLE_PARTS:
                bidx = boxes.spawn(idx, int(part), hx, hy, hz, lx, ly, lz)
                if int(part) == int(PartType.HULL):
                    boxes.hp[bidx] = 100.0
                    boxes.hp_max[bidx] = 100.0
                    armor.bind_hull(idx, bidx)
            fc.set_unit_config(idx, UnitCombatConfig(weapon_id="m256_120mm", armor_mm=600.0))
            ammo.state.set_unit(idx, ammo_amount, ammo_amount)
    boxes.sync_all_from_parent(armor)

    muzzle_flashes = MuzzleFlashLog()
    comm = CommunicationNetwork(armor, formations)
    artilleries = [
        ArtilleryAI(armor, formations, fc=fc, events=events, side=0, muzzle_flashes=muzzle_flashes),
        ArtilleryAI(armor, formations, fc=fc, events=events, side=1, muzzle_flashes=muzzle_flashes),
    ]
    ai = FormationAI(armor, formations, fc=fc, boxes=boxes, events=events, comm_network=comm)
    return armor, boxes, projectiles, ammo, fc, events, formations, ai, artilleries


def benchmark_ai_decision() -> float:
    """编队/单位 AI 决策（纯决策，无开火）：256 编队 × 2048 装甲，返回每帧耗时 (ms)。"""
    armor, boxes, proj, ammo, fc, events, formations, ai, artilleries = _build_world(ammo_amount=0.0)
    loop = SimulationLoop()
    # 预热（Numba 编译一次性成本不计入）
    ai.tick(loop, 1.0)
    for a in artilleries:
        a.tick(loop, 1.0)

    t0 = time.perf_counter()
    n_iter = 10
    for _ in range(n_iter):
        ai.tick(loop, 1.0)
        for a in artilleries:
            a.tick(loop, 1.0)
    elapsed = (time.perf_counter() - t0) / n_iter * 1000.0
    print(f"[1] 编队/单位 AI 决策: {elapsed:.2f} ms/帧 "
          f"({'✅ < 1ms' if elapsed < 1.0 else '⚠️ 超 1ms'})")
    return elapsed


def benchmark_damage_pool() -> float:
    """伤害提交池：4096 球形事件批量结算，返回耗时 (ms)。"""
    armor, boxes, proj, ammo, fc, events, formations, ai, artilleries = _build_world()
    pool = DamageSubmissionPool(capacity=4096)
    for k in range(4096):
        pool.submit_damage(
            DamageKind.SPHERE,
            (float((k % 64) * 50.0), float((k // 64) * 50.0), 0.0),
            radius=25.0,
            pen_mm=150.0,
        )
    t0 = time.perf_counter()
    resolve_area_damage(
        pool.kind, pool.center, pool.radius, pool.pen_mm,
        pool.direction, pool.cone_angle, pool.frag_count, pool.active,
        armor.pos, boxes.hp, armor.hull_idx, armor.active,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"[2] 伤害提交池结算 (4096 球形事件): {elapsed:.2f} ms "
          f"({'✅ < 1ms' if elapsed < 1.0 else '⚠️ 超 1ms'})")
    return elapsed


def benchmark_full_loop() -> float:
    """整体吞吐：AI + combat 持续 10 秒仿真，返回平均每地面 Tick 耗时 (ms)。"""
    armor, boxes, proj, ammo, fc, events, formations, ai, artilleries = _build_world()
    loop = SimulationLoop()
    loop.configure_default_rates()

    def ai_hook(l, dt):
        ai.tick(l, dt)
        for a in artilleries:
            a.tick(l, dt)

    loop.attach("ai", ai_hook)
    loop.attach("combat", lambda l, dt: fc.tick(dt))

    # 预热
    loop.step(1.0)
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        loop.step(1.0)
    elapsed = (time.perf_counter() - t0) / n * 1000.0
    print(f"[3] 整体吞吐 (AI + combat, 2048 装甲): {elapsed:.2f} ms/地面 Tick "
          f"({'✅' if elapsed < 1000.0 else '❌'})")
    return elapsed


def main() -> None:
    print("=" * 60)
    print("Phase 6c 性能基准：AI 决策 / 伤害提交池 / 整体吞吐")
    print("=" * 60)
    benchmark_ai_decision()
    print("-" * 60)
    benchmark_damage_pool()
    print("-" * 60)
    benchmark_full_loop()
    print("=" * 60)


if __name__ == "__main__":
    main()
