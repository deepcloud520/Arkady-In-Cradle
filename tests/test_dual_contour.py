"""Dual Contouring 简化版（Phase B）测试。

覆盖 docs/worldmesh-3d-surface-design.md §4（2026-08-12）：
    - 顶面/底面/侧面提取（column-span 轴对齐盒）
    - 侧面一维合并（桥 4 面墙各合并为 1 面）
    - 地面平滑（B2：continuous_ground 双线性插值）
    - 法线正确（B4：±z 水平面 / ±x±y 垂直面）
    - walkable（仅顶面）
    - STRtree 定位兼容（hint_z 桥下）
    - TerrainMap 集成（world_mesh_builder=DualContourBuilder）

运行：uv run python -m pytest tests/test_dual_contour.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.terrain.dual_contour import DualContourBuilder, _bilinear_z
from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_TERRAIN,
    Heightfield,
)
from Airis.lulilo.terrain.world_mesh import (
    FACE_BOTTOM,
    FACE_NX,
    FACE_NY,
    FACE_PX,
    FACE_PY,
    FACE_TOP,
    WorldMesh,
)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def make_hf_bridge(w: int = 12, h: int = 12, cs: float = 2.0) -> Heightfield:
    """带桥地形：中央 3×3 桥，span [3,4) → 顶面 z=4，底面 z=3。"""
    hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
    hf.set_ground_layer(np.zeros((h, w), dtype=np.float32))
    for gy in range(4, 7):
        for gx in range(4, 7):
            hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)
    return hf


def bridge_patches(mesh: WorldMesh):
    """筛出桥（MAT_BRIDGE）面片。"""
    return [p for p in mesh.patches if p.mat == MAT_BRIDGE]


# ---------------------------------------------------------------------------
# 顶面/底面/侧面提取
# ---------------------------------------------------------------------------


class TestFaceExtraction:
    def test_flat_terrain(self):
        hf = Heightfield(width=10, height=10, cs=2.0, ch=1.0, z_min=-50.0, z_max=200.0)
        hf.set_ground_layer(np.zeros((10, 10), dtype=np.float32))
        mesh = DualContourBuilder(hf).build()
        # 地面顶面逐格（10×10=100）；地面不生成侧面（平滑网格 + Heightfield 兜底）
        tops = [p for p in mesh.patches if p.face == FACE_TOP]
        assert len(tops) == 100
        for t in tops:
            assert t.mat == MAT_TERRAIN
            assert t.walkable
            assert t.normal[2] == 1.0
        # 无侧面/底面（纯地面）
        assert not any(p.face >= FACE_PX for p in mesh.patches)
        assert not any(p.face == FACE_BOTTOM for p in mesh.patches)

    def test_bridge_6_faces(self):
        mesh = DualContourBuilder(make_hf_bridge()).build()
        bp = bridge_patches(mesh)
        # 顶面 1 + 底面 1 + 4 侧面（各合并为 1 面）
        faces = {p.face for p in bp}
        assert faces == {FACE_TOP, FACE_BOTTOM, FACE_PX, FACE_NX, FACE_PY, FACE_NY}
        top = [p for p in bp if p.face == FACE_TOP][0]
        bot = [p for p in bp if p.face == FACE_BOTTOM][0]
        assert top.vertices[0, 2] == pytest.approx(4.0)  # 顶面 z
        assert bot.vertices[0, 2] == pytest.approx(3.0)  # 底面 z
        assert top.walkable
        assert not bot.walkable

    def test_bridge_side_walls_merged(self):
        """桥 3×3 每面侧墙应合并为 1 个墙片（沿墙方向延伸 6m）。"""
        mesh = DualContourBuilder(make_hf_bridge()).build()
        bp = bridge_patches(mesh)
        for face in (FACE_PX, FACE_NX, FACE_PY, FACE_NY):
            walls = [p for p in bp if p.face == face]
            assert len(walls) == 1, f"face {face} 应合并为 1 墙"
            w = walls[0]
            # 沿墙方向长度 = 3 格 = 6m
            if face in (FACE_PX, FACE_NX):
                span_y = w.vertices[:, 1].max() - w.vertices[:, 1].min()
            else:
                span_y = w.vertices[:, 0].max() - w.vertices[:, 0].min()
            assert span_y == pytest.approx(6.0)
            # z 跨 [3, 4]
            assert w.vertices[:, 2].min() == pytest.approx(3.0)
            assert w.vertices[:, 2].max() == pytest.approx(4.0)
            # 法线朝外（PX=+x / NX=-x / PY=+y / NY=-y）
            n = w.normal
            assert np.linalg.norm(n) == pytest.approx(1.0)
            if face == FACE_PX:
                assert n[0] == 1.0 and n[1] == 0.0 and n[2] == 0.0
            elif face == FACE_NX:
                assert n[0] == -1.0 and n[1] == 0.0 and n[2] == 0.0
            elif face == FACE_PY:
                assert n[0] == 0.0 and n[1] == 1.0 and n[2] == 0.0
            else:  # FACE_NY
                assert n[0] == 0.0 and n[1] == -1.0 and n[2] == 0.0

    def test_bridge_ground_top_preserved(self):
        """桥下地面顶面仍存在（可走）：桥抬高 2m，桥下通道可通行。"""
        mesh = DualContourBuilder(make_hf_bridge()).build()
        ground_tops = [
            p for p in mesh.patches
            if p.mat == MAT_TERRAIN and p.face == FACE_TOP
        ]
        assert len(ground_tops) == 144  # 全部 12×12 格（含桥下）
        assert all(p.walkable for p in ground_tops)


# ---------------------------------------------------------------------------
# 地面平滑（B2）
# ---------------------------------------------------------------------------


class TestSmoothing:
    def test_bilinear_z(self):
        z_map = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        assert _bilinear_z(z_map, 2.0, 2.0, 2.0) == pytest.approx(0.0)

    def _gradient_map(self) -> tuple[Heightfield, np.ndarray]:
        """平地面 + 线性梯度连续场（验证去量化）。"""
        hf = Heightfield(width=8, height=8, cs=2.0, ch=1.0, z_min=-50.0, z_max=200.0)
        hf.set_ground_layer(np.zeros((8, 8), dtype=np.float32))
        xs = np.linspace(0.0, 4.0, 8, dtype=np.float32)
        continuous = np.broadcast_to(xs, (8, 8)).astype(np.float32)
        return hf, continuous

    def test_smooth_terrain_top(self):
        """有 continuous_ground 时地面顶面顶点 z = 双线性插值（去量化）。"""
        hf, continuous = self._gradient_map()
        mesh = DualContourBuilder(hf, continuous_ground=continuous).build()
        tops = [
            p for p in mesh.patches
            if p.mat == MAT_TERRAIN and p.face == FACE_TOP
        ]
        assert len(tops) == 64  # 逐格
        for p in tops:
            for v in p.vertices:
                z = _bilinear_z(continuous, float(v[0]), float(v[1]), 2.0)
                assert v[2] == pytest.approx(z, abs=1e-3)

    def test_smooth_relief_plateau(self):
        """非线性起伏（平台）→ 逐格地面面片，内部 z 精确（合并不丢内部）。"""
        hf = Heightfield(width=8, height=8, cs=2.0, ch=1.0, z_min=-50.0, z_max=200.0)
        hf.set_ground_layer(np.zeros((8, 8), dtype=np.float32))
        continuous = np.zeros((8, 8), dtype=np.float32)
        continuous[2:6, 2:6] = 4.0  # 中心 4×4 平台 4m
        mesh = DualContourBuilder(hf, continuous_ground=continuous).build()
        center = [
            p for p in mesh.patches
            if p.mat == MAT_TERRAIN and p.face == FACE_TOP
            and abs(p.centroid[0] - 9.0) < 1.0 and abs(p.centroid[1] - 9.0) < 1.0
        ]
        assert len(center) == 1
        # 平台内部格角点 z 应达到 4.0（非合并后丢失）
        assert center[0].vertices[:, 2].max() == pytest.approx(4.0, abs=0.1)

    def test_no_smooth_without_ground(self):
        """缺省 continuous_ground → 量化层 z（不崩）。"""
        mesh = DualContourBuilder(make_hf_bridge()).build()
        ground_tops = [
            p for p in mesh.patches
            if p.mat == MAT_TERRAIN and p.face == FACE_TOP
        ][0]
        assert ground_tops.vertices[:, 2].max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# STRtree 定位兼容 + TerrainMap 集成
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_locate_hint_z_under_bridge(self):
        mesh = DualContourBuilder(make_hf_bridge()).build()
        # 桥下 (10,10)：hint_z=0（地面）→ 地面顶面；无 hint → 桥顶
        pid = mesh.locate_patch(10.0, 10.0, hint_z=0.0)
        assert mesh.patches[pid].mat == MAT_TERRAIN
        pid_top = mesh.locate_patch(10.0, 10.0)
        assert mesh.patches[pid_top].mat == MAT_BRIDGE

    def test_terrain_map_dual_builder(self):
        from Airis.lulilo.terrain.map import TerrainMap

        tm = TerrainMap(
            make_hf_bridge(),
            world_mesh_builder=DualContourBuilder,
        )
        mesh = tm.world_mesh
        assert isinstance(mesh, WorldMesh)
        bp = bridge_patches(mesh)
        assert any(p.face == FACE_BOTTOM for p in bp)  # 完整 3D

    def test_adjacency(self):
        mesh = DualContourBuilder(make_hf_bridge()).build()
        for a, nbs in mesh.adjacency.items():
            for b in nbs:
                assert a in mesh.adjacency[b]  # 对称
