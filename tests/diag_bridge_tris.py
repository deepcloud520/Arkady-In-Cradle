"""临时诊断：分析 bridge 渲染三角形，定位顶面空缺/退化三角形。运行后删除。"""
from __future__ import annotations

from collections import Counter

import numpy as np

from tests.terrain_factory import TerrainFactory


def in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u >= -1e-6 and v >= -1e-6 and w >= -1e-6


def main() -> None:
    tm = TerrainFactory.default()
    rd = tm.to_render_data()
    bv = rd["bridge_vertices"].astype(np.float64)
    bi = rd["bridge_indices"]
    tris = bi.reshape(-1, 3)
    print(f"bridge verts={bv.shape} idx={bi.shape} tris={len(tris)}")

    zc = Counter(round(float(z), 1) for z in bv[:, 2])
    print("顶点 z 分布:", zc.most_common(8))

    use = Counter(bi.tolist())
    unused = [i for i in range(len(bv)) if use[i] == 0]
    print("未使用顶点:", len(unused))

    zi = bv[:, 2]
    bad = np.where(np.abs(zi) < 1e-6)[0]
    print("z=0 顶点:", len(bad), "其中被三角形引用:", sum(1 for i in bad if use[i] > 0))

    # 退化三角形（面积≈0）
    def area2(t: np.ndarray) -> float:
        a, b, c = bv[t[0]][:2], bv[t[1]][:2], bv[t[2]][:2]
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    areas = np.array([area2(t) for t in tris])
    print(f"退化三角形(面积<1e-4): {(areas < 1e-4).sum()}/{len(tris)}")
    print(f"小面积三角形(<10m²): {(areas < 10).sum()}/{len(tris)}")

    # 桥1 区域覆盖检测（世界格点中心，cs=2）
    samples = []
    for gx in range(100, 161):
        for gy in range(100, 105):
            samples.append((gx * 2 + 1, gy * 2 + 1))
    samples = np.array(samples, dtype=np.float64)
    uncovered = []
    for px, py in samples:
        p = np.array([px, py])
        hit = any(in_tri(p, bv[t[0]][:2], bv[t[1]][:2], bv[t[2]][:2]) for t in tris)
        if not hit:
            uncovered.append((px, py))
    print(f"桥1 格点覆盖: 未覆盖 {len(uncovered)}/{len(samples)}")
    print("  未覆盖样例:", uncovered[:30])


if __name__ == "__main__":
    main()
