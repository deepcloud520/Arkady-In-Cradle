"""CPU 投影验证：用 pyglm 计算 MVP，把地形顶点投影到 NDC，判断相机/矩阵/顶点数据是否正常。

用法: uv run python tests/debug_projection.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Airis.lulilo.terrain.map import TerrainMap
from Airis.render.camera import OrbitCamera


def main() -> None:
    terrain = TerrainMap().generate()
    cfg = terrain.config
    cam = OrbitCamera(
        target=(cfg.width * cfg.cell_size * 0.5, cfg.height * cfg.cell_size * 0.5, 0.0),
        distance=cfg.width * cfg.cell_size * 1.2,
    )
    mvp = cam.mvp_matrix(1.0)
    mvp_np = np.array(mvp, dtype=np.float32)
    print(f"相机位置 eye  : {tuple(round(v, 2) for v in cam.eye())}")
    print(f"mvp bytes 长度 : {len(mvp.to_bytes())} (期望 64)")

    data = terrain.to_render_data()
    verts = data["ground_vertices"][:, :3].astype(np.float32)
    n = len(verts)
    print(f"\n顶点数: {n}, 全部有限: {bool(np.isfinite(verts).all())}")
    print(
        f"顶点范围 x:[{verts[:,0].min()}, {verts[:,0].max()}] "
        f"y:[{verts[:,1].min()}, {verts[:,1].max()}] "
        f"z:[{verts[:,2].min():.1f}, {verts[:,2].max():.1f}]"
    )

    ones = np.ones((n, 1), dtype=np.float32)
    p = np.hstack([verts, ones])  # (N, 4) 行向量
    # GLM 矩阵 numpy 表示 arr[col, row] = M[col, row]，p @ arr 即 M * v
    clip = p @ mvp_np  # (N, 4) 裁剪空间
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None]
    in_front = w > 0
    visible = in_front & np.all(np.abs(ndc) <= 1.0, axis=1)

    print(f"\n相机前方 (w>0)   : {int(in_front.sum())}/{n}")
    print(f"NDC 可见          : {int(visible.sum())}/{n}")
    if in_front.any():
        print(
            f"NDC 范围 x:[{ndc[in_front,0].min():.2f}, {ndc[in_front,0].max():.2f}] "
            f"y:[{ndc[in_front,1].min():.2f}, {ndc[in_front,1].max():.2f}] "
            f"z:[{ndc[in_front,2].min():.2f}, {ndc[in_front,2].max():.2f}]"
        )


if __name__ == "__main__":
    main()
