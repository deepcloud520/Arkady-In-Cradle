"""临时诊断：验证桥侧面几何（顶部到顶面、底部贴合地面）。运行后删除。"""
from __future__ import annotations

from tests.terrain_factory import TerrainFactory


def main() -> None:
    tm = TerrainFactory.default()
    rd = tm.to_render_data()
    bv = rd["bridge_vertices"]
    zs = bv[:, 2]

    uniq = sorted({round(float(z), 2) for z in zs})
    print("bridge 顶点 z 唯一值:", uniq)

    # 顶面 z（patch.z=15）顶点
    top_z = max(uniq)
    n_top = sum(1 for z in zs if abs(z - top_z) < 1e-4)
    print(f"顶面高度 z={top_z}: {n_top} 顶点")

    # 侧面顶部应等于顶面 z；底部应贴合地面（< 顶面）
    # 检查是否有顶点 z 在 (0, top_z) 且为"面板底"旧值 top_z-1（不应再出现）
    mid = [z for z in zs if top_z - 1.001 < z < top_z - 0.999]
    print(f"旧『面板底』z=top_z-1 顶点残留: {len(mid)}")

    # 侧面底部最小值（应贴合地面，约地面高度）
    bot = min(zs)
    print(f"侧面最低点 z={bot:.2f} (桥区地面约 11~13)")

    # 桥1 区域覆盖再次确认
    import numpy as np
    from tests.diag_bridge_tris import in_tri  # noqa: PLC2701

    tris = rd["bridge_indices"].reshape(-1, 3)
    samples = np.array(
        [(gx * 2 + 1, gy * 2 + 1) for gx in range(100, 161) for gy in range(100, 105)],
        dtype=np.float64,
    )
    miss = sum(
        1
        for px, py in samples
        if not any(in_tri(np.array([px, py]), bv[t[0]][:2], bv[t[1]][:2], bv[t[2]][:2]) for t in tris)
    )
    print(f"桥1 顶面覆盖: 未覆盖 {miss}/{len(samples)}")


if __name__ == "__main__":
    main()
