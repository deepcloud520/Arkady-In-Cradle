"""WorldMeshRenderer（Phase D）渲染测试。

覆盖 docs/worldmesh-3d-surface-design.md §7（2026-08-12）：
    - 数据源统一为 WorldMesh（DualContour 完整 3D），不再从 Heightfield 独立提取
    - 地面：平滑细分网格（连续法线 + 高度色）
    - 结构（桥）：顶面 + 底面 + 4 面侧墙完整进入渲染
    - to_render_data 接口兼容（ground/bridge 分离 + 格式）

运行：uv run python -m pytest tests/test_worldmesh_render.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.terrain.factory import TerrainFactory, TerrainFactoryConfig
from Airia.lulilo.terrain.map import TerrainMap
from Airia.lulilo.terrain.surface import WorldMeshRenderer
from Airia.lulilo.terrain.dual_contour import DualContourBuilder
from Airia.lulilo.terrain.world_mesh import FACE_BOTTOM, FACE_TOP


BRIDGE = {"x0": 20, "y0": 20, "x1": 30, "y1": 22}  # 3×11 桥（界内）


def make_terrain(w: int = 64, h: int = 64) -> TerrainMap:
    return TerrainFactory.fbm(
        TerrainFactoryConfig(width=w, height=h, bridges=[BRIDGE])
    )


class TestWorldMeshRenderer:
    def test_build_render_data(self):
        terrain = make_terrain()
        renderer = WorldMeshRenderer(
            terrain.render_world_mesh,
            terrain.heightfield,
            continuous_ground=terrain.continuous_ground,
        )
        rd = renderer.extract()
        assert rd.vertices.ndim == 2 and rd.vertices.shape[1] == 9
        assert rd.vertices.dtype == np.float32
        assert len(rd.indices) > 0 and len(rd.indices) % 3 == 0
        assert np.isfinite(rd.vertices).all()

    def test_ground_smooth_format(self):
        """地面：顶点 z 有限、法线单位化、颜色 3 通道。"""
        terrain = make_terrain()
        renderer = WorldMeshRenderer(
            terrain.render_world_mesh, terrain.heightfield,
            continuous_ground=terrain.continuous_ground,
        )
        rd = renderer.extract()
        gv = rd.vertices[: renderer.struct_offset]
        assert len(gv) > 0
        norms = np.linalg.norm(gv[:, 6:9], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)  # 单位法线
        assert gv[:, 3:6].min() >= 0.0 and gv[:, 3:6].max() <= 1.0  # 颜色范围

    def test_bridge_full_3d(self):
        """桥：顶面 + 底面 + 侧墙完整渲染（Phase B 能力进入渲染）。"""
        terrain = make_terrain()
        renderer = WorldMeshRenderer(
            terrain.render_world_mesh, terrain.heightfield,
            continuous_ground=terrain.continuous_ground,
        )
        rd = renderer.extract()
        bv = rd.vertices[renderer.struct_offset :]
        assert len(bv) > 0, "应有结构顶点"
        z_max = bv[:, 2].max()
        # 有底面（z = z_max - 1，桥面 1m 厚）
        assert np.any(np.abs(bv[:, 2] - (z_max - 1.0)) < 0.1)
        # 有侧墙（法线非 ±z）
        assert np.any(np.abs(bv[:, 6]) + np.abs(bv[:, 7]) > 0.1)


class TestToRenderDataIntegration:
    def test_ground_bridge_split(self):
        """to_render_data 接口兼容：ground/bridge 分离 + 格式。"""
        terrain = make_terrain()
        data = terrain.to_render_data()
        gv, gi = data["ground_vertices"], data["ground_indices"]
        bv, bi = data["bridge_vertices"], data["bridge_indices"]
        assert gv.shape[1] == 9 and gv.dtype == np.float32
        assert len(gi) > 0 and len(gi) % 3 == 0
        assert bv.shape[1] == 9
        assert len(bv) > 0, "桥应渲染"
        assert len(bi) % 3 == 0
        # 桥含底面（Phase B 完整 3D）
        assert np.any(np.abs(bv[:, 2] - (bv[:, 2].max() - 1.0)) < 0.1)

    def test_flat_no_bridge(self):
        """纯平地：无结构顶点，地面正常。"""
        terrain = TerrainFactory.fbm(
            TerrainFactoryConfig(width=32, height=32, bridges=[])
        )
        data = terrain.to_render_data()
        assert len(data["ground_vertices"]) > 0
        assert len(data["bridge_vertices"]) == 0


class TestRenderWorldMeshProperty:
    def test_render_world_mesh_is_dual_contour(self):
        """render_world_mesh 是完整 3D（含底面），与逻辑 world_mesh 解耦。"""
        terrain = make_terrain()
        rwm = terrain.render_world_mesh
        # 完整 3D：存在底面面片
        assert any(p.face == FACE_BOTTOM for p in rwm.patches)
        # 懒加载缓存
        assert terrain.render_world_mesh is rwm

    def test_bridge_patches_complete(self):
        """render_world_mesh 中桥面片齐全（顶/底/侧）。"""
        terrain = make_terrain()
        rwm = terrain.render_world_mesh
        bp = [p for p in rwm.patches if p.mat == 1]  # MAT_BRIDGE
        faces = {p.face for p in bp}
        assert FACE_TOP in faces
        assert FACE_BOTTOM in faces
        assert len(faces) >= 5  # 顶 + 底 + 至少 3 方向侧墙
