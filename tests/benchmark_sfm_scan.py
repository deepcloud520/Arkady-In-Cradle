"""Phase 6b 末敏弹扫描性能基准（docs/phase6-guidance-pipeline.md §八 性能预算）。

    [1] scan_armor_detect 内核（32 SCAN 弹丸 × 2048 装甲）：预算 < 1ms/帧
        （§7.2 测算 32×2048 ≈ 6.5 万对/帧）
    [2] SfmScanner.scan 完整（按 ammo 分组 + 红外 emitter 过滤）：端到端

运行:
    uv run python -m tests.benchmark_sfm_scan
"""

from __future__ import annotations

import time

import numpy as np

from Airis.lulilo.combat.guidance import GuidanceStage
from Airis.lulilo.combat.sfm import SfmScanner, scan_armor_detect
from Airis.lulilo.entities.pools import (
    MAX_ARMOR,
    ArmorPool,
    EmitterPool,
    ProjectilePool,
    ProjectileType,
)
from Airis.lulilo.entities.pools import EmitterBand
from tests.test_guidance_pipeline import make_guided_registry

N_SFM = 32
N_ARMOR = 2048


def _make_scene() -> tuple[ArmorPool, EmitterPool, ProjectilePool]:
    """2048 装甲（RCS=50，每 4 辆一辆热）+ 32 枚 SCAN 末敏弹。"""
    rng = np.random.Generator(np.random.PCG64(42))
    armor = ArmorPool(capacity=MAX_ARMOR)
    for i in range(N_ARMOR):
        armor.spawn(
            rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), rng.uniform(0, 5),
            half_x=1.5, half_y=3.0, half_z=1.0,
        )
        armor.rcs[i] = 50.0

    emitters = EmitterPool()
    for i in range(0, N_ARMOR, 4):
        emitters.activate_ir(i, 0.8, decay_per_s=0.2)

    projectiles = ProjectilePool(capacity=64)
    reg = make_guided_registry()
    a_idx = reg.ammo_index("sfm_155")
    for i in range(N_SFM):
        projectiles.spawn(
            rng.uniform(-200, 200), rng.uniform(-200, 200), 100.0,
            0.0, 0.0, 0.0,
            radius=0.3, dmg=200.0, kind=int(ProjectileType.MISSILE),
            owner=-1, weapon=-1, ammo=a_idx,
        )
        projectiles.guid_stage[i] = int(GuidanceStage.SCAN)
        projectiles.guid_law[i] = 3  # sfm_ballistic
    return armor, emitters, projectiles


def benchmark_scan_kernel() -> float:
    """scan_armor_detect 内核：N_SFM 弹丸 × N_ARMOR 装甲。"""
    armor, emitters, projectiles = _make_scene()
    ir_idx = np.flatnonzero(
        emitters.active & (emitters.band == int(EmitterBand.IR))
    )
    px = projectiles.pos[:N_SFM, 0].copy()
    py = projectiles.pos[:N_SFM, 1].copy()
    pz = projectiles.pos[:N_SFM, 2].copy()
    out = np.full(N_SFM, -1, dtype=np.int32)

    scan_armor_detect(
        px, py, pz, 80.0,
        armor.pos[:, 0], armor.pos[:, 1], armor.rcs, armor.active,
        emitters.cur_power[ir_idx], emitters.parent_idx[ir_idx], emitters.active[ir_idx],
        5.0, 0.1, out,
    )

    N_RUN = 200
    t0 = time.perf_counter()
    for _ in range(N_RUN):
        scan_armor_detect(
            px, py, pz, 80.0,
            armor.pos[:, 0], armor.pos[:, 1], armor.rcs, armor.active,
            emitters.cur_power[ir_idx], emitters.parent_idx[ir_idx],
            emitters.active[ir_idx],
            5.0, 0.1, out,
        )
    elapsed_ms = (time.perf_counter() - t0) / N_RUN * 1000.0
    print(f"[1] scan_armor_detect 内核（{N_SFM} 弹丸 × {N_ARMOR} 装甲）:")
    print(f"    平均 {elapsed_ms:.4f} ms/帧  目标 < 1ms  {'✅' if elapsed_ms < 1.0 else '❌'}")
    return elapsed_ms


def benchmark_scanner_scan() -> float:
    """SfmScanner.scan 完整（分组 + 红外过滤 + 内核）。"""
    armor, emitters, projectiles = _make_scene()
    scanner = SfmScanner(armor, emitters)
    reg = make_guided_registry()

    def ammo_of(proj_idx: int):
        ai = int(projectiles.ammo[proj_idx])
        return reg.ammo_at(ai) if ai >= 0 else None

    scanner.scan(projectiles, ammo_of)

    N_RUN = 200
    t0 = time.perf_counter()
    total = 0
    for _ in range(N_RUN):
        hits = scanner.scan(projectiles, ammo_of)
        total += len(hits)
    elapsed_ms = (time.perf_counter() - t0) / N_RUN * 1000.0
    print(f"[2] SfmScanner.scan 完整（含分组/过滤，命中 {total / N_RUN:.1f} 次/帧）:")
    print(f"    平均 {elapsed_ms:.4f} ms/帧  目标 < 1ms  {'✅' if elapsed_ms < 1.0 else '❌'}")
    return elapsed_ms


if __name__ == "__main__":
    benchmark_scan_kernel()
    benchmark_scanner_scan()
