"""面片合并可行性探针：误差界 quadtree 合并对地面面片数的压缩率。

用途（2026-11-03 可行性研究）：在真实 fBm 地形上测量「按 ε 合并地面顶面」
能把面片数压到什么量级，以及合并本身的时间成本；为 Phase 9 大地图
（≥60km）的面片预算提供数据。

不是单元测试；运行：
    $env:PYTHONPATH="."; .venv/Scripts/python.exe tests/diag_merge_feasibility.py
"""
from __future__ import annotations

import time

import numpy as np

from Airia.lulilo.terrain.factory import TerrainFactory, TerrainFactoryConfig


# ---------------------------------------------------------------------------
# 角点高度场：与 surface.py 的平滑网格一致（cell 中心场 → 角点双线性）
# ---------------------------------------------------------------------------


def corner_z(z_map: np.ndarray) -> np.ndarray:
    """(h, w) 格中心高度 → (h+1, w+1) 角点高度（与 surface.py 同公式）。"""
    h, w = z_map.shape
    ys = np.arange(h + 1, dtype=np.float32)
    xs = np.arange(w + 1, dtype=np.float32)
    # 角点 (i,j) 位于格点 (j, i)，采样 cell 中心场需 -0.5 偏移
    x = xs - 0.5
    y = ys - 0.5
    xf = np.floor(x)
    yf = np.floor(y)
    x0 = np.clip(xf.astype(np.int32), 0, w - 2)
    y0 = np.clip(yf.astype(np.int32), 0, h - 2)
    tx = np.clip(x - xf, 0.0, 1.0)
    ty = np.clip(y - yf, 0.0, 1.0)
    z00 = z_map[y0[:, None], x0[None, :]]
    z10 = z_map[y0[:, None], x0[None, :] + 1]
    z01 = z_map[y0[:, None] + 1, x0[None, :]]
    z11 = z_map[y0[:, None] + 1, x0[None, :] + 1]
    return (
        (z00 * (1 - tx)[None, :] + z10 * tx[None, :]) * (1 - ty)[:, None]
        + (z01 * (1 - tx)[None, :] + z11 * tx[None, :]) * ty[:, None]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# 误差界 quadtree：块内角点与「四角双线性」的偏差 ≤ eps 即整块合并
# ---------------------------------------------------------------------------


def _block_error(Z: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> float:
    z00 = Z[r0, c0]
    z10 = Z[r0, c1]
    z01 = Z[r1, c0]
    z11 = Z[r1, c1]
    u = np.linspace(0.0, 1.0, c1 - c0 + 1, dtype=np.float32)
    v = np.linspace(0.0, 1.0, r1 - r0 + 1, dtype=np.float32)
    pred = (
        (z00 * (1 - u)[None, :] + z10 * u[None, :]) * (1 - v)[:, None]
        + (z01 * (1 - u)[None, :] + z11 * u[None, :]) * v[:, None]
    )
    return float(np.max(np.abs(Z[r0 : r1 + 1, c0 : c1 + 1] - pred)))


def quadtree_merge(Z: np.ndarray, eps: float) -> list[tuple[int, int, int, int]]:
    h, w = Z.shape[0] - 1, Z.shape[1] - 1
    blocks: list[tuple[int, int, int, int]] = []
    stack = [(0, 0, h, w)]
    while stack:
        r0, c0, r1, c1 = stack.pop()
        if (r1 - r0) <= 1 and (c1 - c0) <= 1:
            blocks.append((r0, c0, r1, c1))
            continue
        if _block_error(Z, r0, c0, r1, c1) <= eps:
            blocks.append((r0, c0, r1, c1))
            continue
        rm = (r0 + r1) // 2
        cm = (c0 + c1) // 2
        for a, b, c, d in (
            (r0, c0, rm, cm),
            (r0, cm, rm, c1),
            (rm, c0, r1, cm),
            (rm, cm, r1, c1),
        ):
            if a < c and b < d:
                stack.append((a, b, c, d))
    return blocks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def make_terrain(w: int, h: int, amp: float, noise: float, cs: float = 37.5):
    cfg = TerrainFactoryConfig(
        width=w, height=h, cell_size=cs, seed=42,
        amplitude=amp, noise_scale=noise, octaves=4,
        bridge_lift=2.0, bridges=[], canyon_bridge=None,
    )
    t0 = time.perf_counter()
    terrain = TerrainFactory.fbm(cfg)
    dt = time.perf_counter() - t0
    return terrain, dt


def report(name: str, terrain, build_s: float) -> None:
    hf = terrain.heightfield
    z = terrain.continuous_ground
    assert z is not None
    spans_mb = hf.spans.nbytes / 1e6
    count_mb = hf.count.nbytes / 1e6
    print(f"\n=== {name} ===")
    print(f"grid {hf.width}x{hf.height} cs={hf.cs}  z_range=({float(z.min()):.1f}, {float(z.max()):.1f})")
    print(f"heightfield build {build_s:.2f}s | spans {spans_mb:.1f} MB + count {count_mb:.1f} MB"
          f" = {(spans_mb + count_mb) * 1e6 / z.size:.1f} B/cell")
    Z = corner_z(z)
    for eps in (0.0, 0.005, 0.01, 0.05, 0.2, 1.0):
        t0 = time.perf_counter()
        blocks = quadtree_merge(Z, eps)
        dt = time.perf_counter() - t0
        n = len(blocks)
        print(f"  eps={eps:>5} m -> {n:>7} patches  ({n / z.size * 100:5.1f}% of cells)"
              f"  merge {dt:5.2f}s  ({dt / n * 1e6:6.1f} us/patch)")


def main() -> None:
    for name, (w, h, amp, noise) in {
        "flat 400x400": (400, 400, 0.0, 18.0),
        "fbm 400x400 (amp=10, scale=18)": (400, 400, 10.0, 18.0),
        "fbm 400x400 (hill amp=40, scale=8)": (400, 400, 40.0, 8.0),
        "fbm 800x800 (amp=10, scale=18)": (800, 800, 10.0, 18.0),
    }.items():
        terrain, dt = make_terrain(w, h, amp, noise)
        report(name, terrain, dt)

    scale_probe()


# ---------------------------------------------------------------------------
# 分辨率扫描：同一 15km 战场，cs 从 10m 到 200m（特征尺度固定 800m）
# ---------------------------------------------------------------------------


def scale_probe() -> None:
    print("\n\n=== 分辨率扫描：15km×15km 战场，特征尺度 800m，振幅 60m ===")
    print(f"{'cs(m)':>6} {'grid':>10} {'cells':>9} "
          f"{'eps=5cm':>9} {'eps=20cm':>9} {'eps=1m':>9} {'MB@eps20':>9} {'merge_s':>8}")
    extent = 15000.0
    for cs in (10.0, 25.0, 50.0, 100.0, 200.0):
        n = int(round(extent / cs))
        terrain, _ = make_terrain(n, n, 60.0, 800.0 / cs, cs=cs)
        z = terrain.continuous_ground
        Z = corner_z(z)
        counts = []
        t_sum = 0.0
        for eps in (0.05, 0.2, 1.0):
            t0 = time.perf_counter()
            counts.append(len(quadtree_merge(Z, eps)))
            t_sum += time.perf_counter() - t0
        cells = n * n
        mem_mb = counts[1] * 500 / 1e6  # patch 对象 + 顶点 + ndarray 头 ≈ 500B
        print(f"{cs:>6.0f} {n:>5}x{n:<4} {cells:>9} "
              f"{counts[0]:>9} {counts[1]:>9} {counts[2]:>9} "
              f"{mem_mb:>9.1f} {t_sum:>8.2f}")



if __name__ == "__main__":
    main()
