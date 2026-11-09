"""临时诊断：检查桥顶面 patch 结构与每格桥面覆盖（定位顶面空缺）。运行后删除。"""
from __future__ import annotations

from Airia.lulilo.terrain.heightfield import MAT_BRIDGE
from Airia.lulilo.terrain.surface_mesh import SurfaceMeshBuilder
from tests.terrain_factory import TerrainFactory, TerrainFactoryConfig


def main() -> None:
    tm = TerrainFactory.default()
    hf = tm.heightfield
    mesh = SurfaceMeshBuilder(hf).build()

    bp = [p for p in mesh.patches if p.mat == MAT_BRIDGE]
    print(f"桥面 patch 数: {len(bp)}")
    for p in bp:
        nv = len(p.vertices)
        print(f"  patch#{p.patch_id} z={p.z:.2f} nv={nv} area={p.area:.0f}")
        if nv > 4:
            print(f"    顶点(>4): {p.vertices.tolist()}")

    # 桥区每格桥面覆盖检查
    cfg = TerrainFactoryConfig()  # 与 TerrainFactory.default() 相同（默认 3 座演示桥）
    for b in cfg.bridges:
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        missing = []
        for gy in range(y0, y1 + 1):
            for gx in range(x0, x1 + 1):
                col = hf._col(gx, gy)  # noqa: SLF001
                has = any(int(hf.spans[col, i, 3]) == MAT_BRIDGE for i in range(int(hf.count[col])))
                if not has:
                    missing.append((gx, gy, float(tm.continuous_ground[gy, gx])))
        print(f"桥 {b}: 桥区格子={(x1-x0+1)*(y1-y0+1)}, 缺失桥面={len(missing)}")
        for m in missing[:30]:
            print(f"    缺失 ({m[0]},{m[1]}) ground_z={m[2]:.2f}")


if __name__ == "__main__":
    main()
