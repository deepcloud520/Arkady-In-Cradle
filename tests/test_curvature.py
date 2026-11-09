"""地球曲率 / LocalFrame 测试（docs/curvature-3d-worldmesh-design.md P0）。

覆盖：
- 曲率数值表（10/40/47/60/80 km）
- `z_span` ↔ `z_wm` ↔ `msl` 换算
- 地心 / 自转向量 / ECEF 往返
- 单帧适用范围校验
- DualContourBuilder / WorldMeshBuilder 构建期压弯（frame vs 无 frame）
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Airia.lulilo.coordinates.frame import (
    MAX_SINGLE_FRAME_KM,
    R_EARTH,
    LocalFrame,
    bend_patches,
)
from Airia.lulilo.terrain.dual_contour import DualContourBuilder
from Airia.lulilo.terrain.heightfield import MAT_BRIDGE, MAT_TERRAIN, Heightfield
from Airia.lulilo.terrain.map import TerrainMap
from Airia.lulilo.terrain.world_mesh import FACE_TOP, WorldMeshBuilder


# ---------------------------------------------------------------------------
# 曲率数值
# ---------------------------------------------------------------------------


class TestDrop:
    def test_drop_table(self):
        """曲率下坠与设计文档 §0 的数值一致（R=6371km）。"""
        f = LocalFrame(x0=0.0, y0=0.0)
        cases = {
            10_000.0: 7.85,
            40_000.0: 125.6,
            47_000.0: 173.3,
            60_000.0: 282.5,
            80_000.0: 502.3,
        }
        for d, expect in cases.items():
            assert f.drop(d, 0.0) == pytest.approx(expect, rel=2e-3)

    def test_drop_uses_anchor_offset(self):
        """锚点不在原点时，按 (x−x0, y−y0) 计算。"""
        f = LocalFrame(x0=100.0, y0=50.0)
        assert f.drop(100.0, 50.0) == pytest.approx(0.0)
        assert f.drop(1100.0, 50.0) == pytest.approx(1000.0 ** 2 / (2 * R_EARTH))

    def test_wm_span_roundtrip(self):
        f = LocalFrame(x0=500.0, y0=500.0)
        z_span = 123.0
        z_wm = f.wm_z(z_span, 900.0, 700.0)
        assert z_wm < z_span
        assert f.span_z(z_wm, 900.0, 700.0) == pytest.approx(z_span)
        assert f.msl(z_wm, 900.0, 700.0) == pytest.approx(z_span)


# ---------------------------------------------------------------------------
# 地球几何
# ---------------------------------------------------------------------------


class TestEarthGeometry:
    def test_earth_center(self):
        f = LocalFrame(x0=100.0, y0=200.0, alt0=50.0)
        assert f.earth_center() == (100.0, 200.0, -R_EARTH)

    def test_omega_enu(self):
        f = LocalFrame(lat0=45.0)
        _, north, up = f.omega()
        assert north == pytest.approx(7.2921159e-5 * math.cos(math.radians(45.0)))
        assert up == pytest.approx(7.2921159e-5 * math.sin(math.radians(45.0)))

    def test_ecef_roundtrip(self):
        f = LocalFrame(lat0=49.3, lon0=-0.5, alt0=25.0)
        x, y, z = 1234.0, -567.0, 89.0
        ex, ey, ez = f.to_ecef(x, y, z)
        # ECEF 量级应为地球半径附近
        assert 6.3e6 < math.hypot(ex, ey, ez) < 6.5e6
        rx, ry, rz = f.from_ecef(ex, ey, ez)
        assert (rx, ry, rz) == pytest.approx((x, y, z), abs=1e-6)

    def test_validate_single_frame(self):
        f = LocalFrame()
        f.validate(80_000.0, 50_000.0)  # 94km 对角 → 允许
        with pytest.raises(ValueError):
            f.validate(800_000.0, 100_000.0)  # 806km → 必须分块
        assert MAX_SINGLE_FRAME_KM == 100.0


# ---------------------------------------------------------------------------
# 构建期压弯
# ---------------------------------------------------------------------------


def _flat_hf(w: int = 10, h: int = 10, cs: float = 2.0, z: float = 0.0) -> Heightfield:
    hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
    hf.set_ground_layer(np.full((h, w), z, dtype=np.float32))
    return hf


class TestBuilderBend:
    def test_dual_contour_no_frame_unchanged(self):
        mesh = DualContourBuilder(_flat_hf()).build()
        tops = [p for p in mesh.patches if p.face == FACE_TOP and p.mat == MAT_TERRAIN]
        assert tops
        for p in tops:
            assert p.vertices[:, 2] == pytest.approx(0.0)

    def test_dual_contour_bends_with_frame(self):
        """平面 z=0 + 小半径 frame → 角落 patch 下坠 100m。"""
        hf = _flat_hf()  # 20m × 20m 地图
        frame = LocalFrame(x0=10.0, y0=10.0, radius=1000.0)
        mesh = DualContourBuilder(hf, frame=frame).build()

        pid = mesh.locate_patch(1.0, 1.0)
        assert pid >= 0
        patch = mesh.patches[pid]
        # 四角 (0,0)/(2,0)/(2,2)/(0,2) 的平均下坠
        corners = [(0, 0), (2, 0), (2, 2), (0, 2)]
        expect = -sum(frame.drop(cx, cy) for cx, cy in corners) / 4.0
        assert patch.z_at(1.0, 1.0) == pytest.approx(expect, rel=1e-5, abs=1e-6)
        # 法线已重算（单位长度、带坡度）
        assert np.linalg.norm(patch.normal) == pytest.approx(1.0)
        assert patch.normal[2] == pytest.approx(1.0, abs=0.2)

    def test_structure_patch_bends(self):
        """桥顶面同样压弯（结构不是例外）。"""
        hf = _flat_hf(w=12, h=12, cs=2.0)
        for gy in range(4, 7):
            for gx in range(4, 7):
                hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)
        frame = LocalFrame(x0=12.0, y0=12.0, radius=1000.0)
        mesh = DualContourBuilder(hf, frame=frame).build()
        top = [p for p in mesh.patches if p.mat == MAT_BRIDGE and p.face == FACE_TOP][0]
        # 顶点（矩形四角）压弯；形心 z = 四角 z 的均值
        expect = 4.0 - float(
            np.mean([frame.drop(float(v[0]), float(v[1])) for v in top.vertices])
        )
        assert float(top.centroid[2]) == pytest.approx(expect, abs=1e-4)

    def test_worldmesh_builder_bends(self):
        hf = _flat_hf()
        frame = LocalFrame(x0=10.0, y0=10.0, radius=1000.0)
        mesh = WorldMeshBuilder(hf, frame=frame).build()
        tops = [p for p in mesh.patches if p.face == FACE_TOP]
        assert tops
        # 角落 patch 的顶点应低于中心 patch
        corner = min(tops, key=lambda p: float(p.centroid[0] + p.centroid[1]))
        assert float(corner.vertices[:, 2].min()) < -0.05

    def test_terrain_map_passes_frame(self):
        hf = _flat_hf()
        frame = LocalFrame(x0=10.0, y0=10.0, radius=1000.0)
        tm = TerrainMap(hf, world_mesh_builder=DualContourBuilder, frame=frame)
        pid = tm.world_mesh.locate_patch(1.0, 1.0)
        assert tm.world_mesh.patches[pid].z_at(1.0, 1.0) < -0.05

    def test_bend_patches_noop_without_frame(self):
        hf = _flat_hf()
        mesh = DualContourBuilder(hf).build()
        before = [p.vertices.copy() for p in mesh.patches]
        bend_patches(mesh.patches, None)
        for p, b in zip(mesh.patches, before):
            assert np.array_equal(p.vertices, b)
