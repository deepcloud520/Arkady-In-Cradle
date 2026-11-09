"""Phase 6b 制导性能基准（docs/phase6-guidance-pipeline.md §八 性能预算）。

    [1] 制导积分内核（64 枚导弹 TRACK）：预算 < 1ms/帧
    [2] 完整开火链一帧（2048 装甲 + 64 枚导弹 + 阶段切换 + 扫描 + 碰撞）：
        端到端含制导的每帧成本

运行:
    uv run python -m tests.benchmark_guidance
"""

from __future__ import annotations

import time

import numpy as np

from Airia.lulilo.combat.damage import ArmorType
from Airia.lulilo.combat.engagement import GUIDED_MAX_ACCEL, FiringControl, UnitCombatConfig
from Airia.lulilo.combat.guidance import MissileControlRegistry
from Airia.lulilo.combat.weapons import WeaponSystem
from Airia.lulilo.entities.pools import (
    MAX_ARMOR,
    MAX_BOXES,
    MAX_PROJ,
    ArmorPool,
    BoxPool,
    EmitterBand,
    EmitterPool,
    PartType,
    ProjectilePool,
    ProjectileType,
)
from Airia.lulilo.logistics.ammo import AmmoManager
from Airia.lulilo.logistics.fuel import FuelManager
from Airia.lulilo.movement.integrate import integrate_guided_projectile_movement
from tests.test_guidance_pipeline import make_guided_registry

N_MISSILES = 64
N_ARMOR = 2048


def _spawn_missiles(projectiles: ProjectilePool, n: int) -> None:
    """n 枚 TRACK 阶段导弹（boost_coast_pn），随机散布。"""
    rng = np.random.Generator(np.random.PCG64(42))
    control = MissileControlRegistry.with_builtin()
    lid = control.index_of("boost_coast_pn")
    for i in range(n):
        projectiles.spawn(
            rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(50, 200),
            rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-30, 0),
            radius=0.2, dmg=100.0, kind=int(ProjectileType.MISSILE), owner=0,
        )
        projectiles.guid_law[i] = lid
        projectiles.guid_target_pos[i] = (
            rng.uniform(-500, 500), rng.uniform(-500, 500), 0.0,
        )
        projectiles.guid_stage[i] = 0  # TRACK
        projectiles.guid_nav_gain[i] = 3.0


def benchmark_guided_integrate() -> float:
    """制导积分内核：N_MISSILES 枚导弹一子步（0.01s）耗时。"""
    projectiles = ProjectilePool(capacity=N_MISSILES + 16)
    _spawn_missiles(projectiles, N_MISSILES)
    control = MissileControlRegistry.with_builtin()
    fn = control.law_at(control.index_of("boost_coast_pn"))
    p = projectiles

    integrate_guided_projectile_movement(
        p.pos, p.vel, p.active, p.age,
        p.guid_target_pos, p.guid_target_vel, p.guid_target_kind, p.guid_memory_pos, p.guid_stage,
        p.guid_nav_gain, p.guid_override,
        p.guid_search_fov_cos, p.guid_track_fov_cos,
        GUIDED_MAX_ACCEL, 0.01, 9.81, fn,
    )

    N_RUN = 200
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        integrate_guided_projectile_movement(
            p.pos, p.vel, p.active, p.age,
            p.guid_target_pos, p.guid_target_vel, p.guid_target_kind, p.guid_memory_pos, p.guid_stage,
            p.guid_nav_gain, p.guid_override,
            p.guid_search_fov_cos, p.guid_track_fov_cos,
            GUIDED_MAX_ACCEL, 0.01, 9.81, fn,
        )
    elapsed_ms = (time.perf_counter() - t0) / N_RUN * 1000.0
    print(f"[1] 制导积分（{N_MISSILES} 枚导弹 TRACK）:")
    print(f"    平均 {elapsed_ms:.4f} ms/帧  目标 < 1ms  {'✅' if elapsed_ms < 1.0 else '❌'}")
    return elapsed_ms


def benchmark_full_step() -> float:
    """完整开火链一帧：2048 装甲 + 64 枚导弹（制导 + 扫描 + 碰撞 + 毁伤）。"""
    rng = np.random.Generator(np.random.PCG64(7))
    armor = ArmorPool(capacity=MAX_ARMOR)
    boxes = BoxPool(capacity=MAX_BOXES)
    projectiles = ProjectilePool(capacity=MAX_PROJ)
    ammo = AmmoManager(capacity=MAX_ARMOR)
    fuel = FuelManager(capacity=MAX_ARMOR)
    fc = FiringControl(
        armor, boxes, projectiles,
        ammo=ammo, fuel=fuel,
        weapon_system=WeaponSystem(make_guided_registry()),
    )
    emitters = fc.emitters

    # 2048 坦克
    for i in range(MAX_ARMOR):
        x = (i % 64) * 40.0 - 1260.0
        y = (i // 64) * 40.0 - 1260.0
        idx = armor.spawn(x, y, 0.0, half_x=1.5, half_y=3.0, half_z=1.0)
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
                weapon_id="m256_120mm", armor_mm=300.0,
                armor_type=int(ArmorType.RHA),
            ),
        )
        ammo.state.set_unit(idx, 100.0, 100.0)
        fuel.state.set_unit(idx, 500.0, 500.0)
        # 每 32 辆一辆热（红外 emitter），模拟末敏弹可探测目标
        if i % 32 == 0:
            emitters.activate_ir(idx, 0.8, decay_per_s=0.2)
    boxes.sync_all_from_parent(armor)

    # 64 枚导弹（末敏弹 SCAN + 制导弹混合）
    _spawn_missiles(projectiles, N_MISSILES)
    control = MissileControlRegistry.with_builtin()
    sfm_lid = control.index_of("sfm_ballistic")
    for i in range(N_MISSILES):
        if i % 2 == 0:
            projectiles.guid_law[i] = sfm_lid
            projectiles.guid_stage[i] = 4  # SCAN（悬停 + 扫描）
            projectiles.pos[i, 2] = 100.0

    # 预热
    fc.tick(0.01)
    for _ in range(10):
        fc.tick(0.01)

    N_RUN = 50
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        fc.tick(0.01)
    elapsed_ms = (time.perf_counter() - t0) / N_RUN * 1000.0
    print(f"[2] 完整开火链一帧（{MAX_ARMOR} 装甲 + {N_MISSILES} 枚导弹含 SCAN）:")
    print(f"    平均 {elapsed_ms:.3f} ms/帧  目标 < 3ms  {'✅' if elapsed_ms < 3.0 else '❌'}")
    return elapsed_ms


if __name__ == "__main__":
    benchmark_guided_integrate()
    benchmark_full_step()

