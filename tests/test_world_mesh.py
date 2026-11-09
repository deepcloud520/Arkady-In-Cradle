"""WorldMesh 3D 世界模型测试。

覆盖 docs/worldmesh-3d-surface-design.md Phase A（2026-08-12）：
    - WorldPatch 数据模型（3D 顶点 / 法线 / 面积 / 形心）
    - WorldMeshBuilder（从 Heightfield：平坦 / 带桥 / fBm 地形）
    - STRtree 点定位（基础 / mat 过滤 / hint_z 桥下定位）
    - ray_plane_z 竖直射线求交
    - flatten_world_mesh 展平（Numba 输入）
    - integrate_armor_movement_3d（移动 + mesh_id 局部性 + z 同步）
    - resolve_mesh_ids（STRtree fallback）

运行：uv run python -m pytest tests/test_world_mesh.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.movement.integrate import (
    _z_bilinear_flat,
    integrate_armor_movement_3d,
    resolve_mesh_ids,
)
from Airia.lulilo.terrain.dual_contour import DualContourBuilder
from Airia.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_BUILDING,
    MAT_TERRAIN,
    MAT_TUNNEL,
    Heightfield,
)
from Airia.lulilo.terrain.world_mesh import (
    WorldMesh,
    WorldMeshBuilder,
    WorldPatch,
    flatten_world_mesh,
    point_in_polygon,
)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def make_hf_flat(w: int = 10, h: int = 10, cs: float = 2.0) -> Heightfield:
    """平坦地形高度场（全部 z=0 的地面层）。"""
    hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
    hf.set_ground_layer(np.zeros((h, w), dtype=np.float32))
    return hf


def make_hf_with_bridge(w: int = 12, h: int = 12, cs: float = 2.0) -> Heightfield:
    """带桥地形：中央 3×3 桥（z=4，抬升 2m，max_climb 内可连通）。"""
    hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
    hf.set_ground_layer(np.zeros((h, w), dtype=np.float32))
    # 桥 span [3,4) → 顶面 z=4（地面 z=2，差 2 = max_climb）
    for gy in range(4, 7):
        for gx in range(4, 7):
            hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)
    return hf


# ---------------------------------------------------------------------------
# WorldPatch 数据模型
# ---------------------------------------------------------------------------


class TestWorldPatch:
    def test_3d_vertices_and_normal(self):
        verts = np.array(
            [[0, 0, 5], [4, 0, 5], [4, 4, 5], [0, 4, 5]], dtype=np.float32
        )
        p = WorldPatch(patch_id=0, vertices=verts, normal=np.array([0, 0, 1], dtype=np.float32), mat=MAT_TERRAIN)
        assert p.vertices.shape == (4, 3)
        assert p.vertices[0, 2] == 5.0  # 真 3D
        assert p.normal[2] == 1.0
        assert p.area == pytest.approx(16.0)  # 4×4
        assert p.centroid.shape == (3,)

    def test_sloped_patch_area(self):
        # 斜四边形（Newell 法线/面积正确）
        verts = np.array(
            [[0, 0, 0], [4, 0, 2], [4, 4, 2], [0, 4, 0]], dtype=np.float32
        )
        p = WorldPatch(patch_id=0, vertices=verts, normal=np.array([0, 0, 1], dtype=np.float32), mat=MAT_TERRAIN)
        # 投影面积 16，斜面积 = 16 / cos(θ)，θ=atan(0.5) → ≈17.888
        assert p.area == pytest.approx(17.8885, rel=1e-3)


# ---------------------------------------------------------------------------
# WorldMeshBuilder
# ---------------------------------------------------------------------------


class TestWorldMeshBuilder:
    def test_flat_build_all_walkable(self):
        hf = make_hf_flat(10, 10)
        mesh = WorldMeshBuilder(hf).build()
        assert len(mesh.patches) == 1  # 全平 → 单面片
        assert 0 in mesh.walkable_patches
        p = mesh.patches[0]
        assert p.vertices.shape == (4, 3)
        assert p.vertices[0, 2] == pytest.approx(0.0)
        assert p.normal[2] == 1.0

    def test_bridge_patches(self):
        hf = make_hf_with_bridge(12, 12)
        mesh = WorldMeshBuilder(hf).build()
        mats = {p.mat for p in mesh.patches}
        assert MAT_TERRAIN in mats
        assert MAT_BRIDGE in mats
        # 桥面 z=4
        bridge = [p for p in mesh.patches if p.mat == MAT_BRIDGE]
        assert len(bridge) == 1
        assert bridge[0].vertices[0, 2] == pytest.approx(4.0)
        # 桥可行走
        assert bridge[0].patch_id in mesh.walkable_patches

    def test_fbm_terrain_build(self):
        from Airia.lulilo.terrain.factory import TerrainFactory, TerrainFactoryConfig

        terrain = TerrainFactory.fbm(
            TerrainFactoryConfig(width=64, height=64)
        )
        mesh = WorldMeshBuilder(terrain.heightfield).build()
        assert len(mesh.patches) > 0
        assert len(mesh.walkable_patches) > 0
        # 所有面片 3D 顶点 + 单位法线
        for p in mesh.patches:
            assert p.vertices.shape[1] == 3
            assert np.linalg.norm(p.normal) == pytest.approx(1.0)

    def test_adjacency_symmetry(self):
        hf = make_hf_flat(10, 10, cs=2.0)
        # 制造多面片：两块 5×5 高度不同 → 无邻接；再放平桥
        hf.add_span(0, 0, 3.0, 4.0, mat=MAT_BRIDGE)  # 单格桥 z=4，与地面差 2
        mesh = WorldMeshBuilder(hf).build()
        for a, nbs in mesh.adjacency.items():
            for b in nbs:
                assert a in mesh.adjacency[b]  # 对称


# ---------------------------------------------------------------------------
# STRtree 点定位 + hint_z
# ---------------------------------------------------------------------------


class TestLocatePatch:
    def test_basic_locate(self):
        hf = make_hf_flat(10, 10, cs=2.0)
        mesh = WorldMeshBuilder(hf).build()
        # 地面中心
        pid = mesh.locate_patch(10.0, 10.0)
        assert pid >= 0
        assert mesh.patches[pid].mat == MAT_TERRAIN
        # 界外
        assert mesh.locate_patch(-100.0, -100.0) == -1

    def test_mat_filter(self):
        hf = make_hf_with_bridge(12, 12)
        mesh = WorldMeshBuilder(hf).build()
        # 桥面上方 (10,10) 是桥面中心
        pid_b = mesh.locate_patch(10.0, 10.0, mat=MAT_BRIDGE)
        assert pid_b >= 0
        assert mesh.patches[pid_b].mat == MAT_BRIDGE
        pid_t = mesh.locate_patch(10.0, 10.0, mat=MAT_TERRAIN)
        assert pid_t >= 0
        assert mesh.patches[pid_t].mat == MAT_TERRAIN

    def test_hint_z_under_bridge(self):
        """核心：桥下单位 hint_z=地面 z → 必须选中地面面片而非桥面。"""
        hf = make_hf_with_bridge(12, 12)
        mesh = WorldMeshBuilder(hf).build()
        x, y = 10.0, 10.0  # 桥下（桥面 z=4，地面 z=0）

        # 无 hint → 表面最高 = 桥面
        pid_no_hint = mesh.locate_patch(x, y)
        assert mesh.patches[pid_no_hint].mat == MAT_BRIDGE

        # hint_z=0（地面）→ 选中地面
        pid_hint = mesh.locate_patch(x, y, hint_z=0.0)
        assert mesh.patches[pid_hint].mat == MAT_TERRAIN

    def test_ray_plane_z(self):
        hf = make_hf_with_bridge(12, 12)
        mesh = WorldMeshBuilder(hf).build()
        pid_b = mesh.locate_patch(10.0, 10.0, mat=MAT_BRIDGE)
        assert mesh.ray_plane_z(pid_b, 10.0, 10.0) == pytest.approx(4.0)
        pid_t = mesh.locate_patch(10.0, 10.0, mat=MAT_TERRAIN)
        assert mesh.ray_plane_z(pid_t, 10.0, 10.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# flatten_world_mesh + Numba 移动积分
# ---------------------------------------------------------------------------


class TestFlattenAndIntegrate:
    def _setup(self):
        hf = make_hf_with_bridge(12, 12)
        mesh = WorldMeshBuilder(hf).build()
        flat = flatten_world_mesh(mesh)
        return hf, mesh, flat

    def test_flatten_roundtrip(self):
        _, mesh, (pv, vstart, vcnt, norm) = self._setup()
        assert vstart[-1] == len(pv)
        assert len(vcnt) == len(mesh.patches)
        assert norm.shape == (len(mesh.patches), 3)
        # 展平顶点与面片一致
        for i, p in enumerate(mesh.patches):
            s, c = int(vstart[i]), int(vcnt[i])
            assert np.allclose(pv[s : s + c], p.vertices, atol=1e-6)

    def test_integrate_3d_z_sync(self):
        _, mesh, flat = self._setup()
        pv, vstart, vcnt, norm = flat

        n = 2
        pos = np.array(
            [[3.0, 3.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float32
        )
        yaw = np.zeros(n, dtype=np.float32)
        active = np.ones(n, dtype=np.bool_)
        wp = np.zeros((n, 8, 2), dtype=np.float32)
        wp_count = np.zeros(n, dtype=np.uint8)
        wp_idx = np.zeros(n, dtype=np.uint8)
        speed = np.full(n, 10.0, dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        mesh_id = np.array([-1, -1], dtype=np.int32)

        # 先绑定：单位1 在地面 (3,3)，单位2 在桥面 (10,10)
        mesh_id[0] = mesh.locate_patch(3.0, 3.0, hint_z=0.0)
        mesh_id[1] = mesh.locate_patch(10.0, 10.0, hint_z=4.0)
        assert mesh_id[1] >= 0

        integrate_armor_movement_3d(
            pos, yaw, active, wp, wp_count, wp_idx, speed, vel, 1.0,
            mesh_id, pv, vstart, vcnt, norm,
        )
        # 无路标点 → 不动；z 已同步
        assert pos[0, 2] == pytest.approx(0.0)  # 地面
        assert pos[1, 2] == pytest.approx(4.0)  # 桥面
        assert int(mesh_id[0]) >= 0 and int(mesh_id[1]) >= 0

    def test_integrate_3d_cross_patch(self):
        """单位沿路标点移动穿越面片边界 → mesh_id 置 -1 → resolve 回退。"""
        _, mesh, flat = self._setup()
        pv, vstart, vcnt, norm = flat

        # 单位在地面最左格，向右移动 3 格（穿越桥覆盖区域，仍在桥下地面）
        # 地面格 (1,1) 中心 (3,3) → 移动至 (9,3)：地面面片跨越整行，
        # 但桥面(4..7)覆盖其上——地面单位应保持地面 mesh_id。
        n = 1
        pos = np.array([[3.0, 3.0, 0.0]], dtype=np.float32)
        yaw = np.zeros(n, dtype=np.float32)
        active = np.ones(n, dtype=np.bool_)
        wp = np.zeros((n, 8, 2), dtype=np.float32)
        wp_count = np.array([1], dtype=np.uint8)
        wp_idx = np.zeros(n, dtype=np.uint8)
        wp[0, 0] = (9.0, 3.0)
        speed = np.full(n, 20.0, dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        mesh_id = np.array([mesh.locate_patch(3.0, 3.0, hint_z=0.0)], dtype=np.int32)

        integrate_armor_movement_3d(
            pos, yaw, active, wp, wp_count, wp_idx, speed, vel, 1.0,
            mesh_id, pv, vstart, vcnt, norm,
        )
        # 已穿越（从地面格 → 越过桥区边缘），mesh_id 可能 -1（桥面 z=4 与地面 z=0 差 2=max_climb，地面面片不跨越桥）
        assert pos[0, 0] == pytest.approx(9.0)
        # 若 miss → resolve_mesh_ids 用 hint_z=0 找回地面面片
        resolve_mesh_ids(pos, mesh_id, mesh)
        assert int(mesh_id[0]) >= 0
        assert mesh.patches[int(mesh_id[0])].mat == MAT_TERRAIN
        assert pos[0, 2] == pytest.approx(0.0)  # z 保持地面

    def test_point_in_polygon_helper(self):
        verts = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=np.float32)
        assert point_in_polygon(2.0, 2.0, verts)
        assert not point_in_polygon(5.0, 2.0, verts)
        assert point_in_polygon(0.0, 0.0, verts)  # 边界含


# ---------------------------------------------------------------------------
# E1 采样斜面（docs/suspension-slope-design.md §4.2）：双线性 z + 梯度法线
# ---------------------------------------------------------------------------


class TestPatchSamplingE1:
    """WorldPatch.z_at / normal_at 双线性采样（悬挂触点采样的前置）。"""

    def _sloped_patch(self) -> WorldPatch:
        """双线性曲面四边形：v=[(0,0,1),(4,0,3),(4,4,5),(0,4,2)]。"""
        verts = np.array(
            [[0, 0, 1], [4, 0, 3], [4, 4, 5], [0, 4, 2]], dtype=np.float32
        )
        return WorldPatch(
            patch_id=0, vertices=verts,
            normal=np.array([0, 0, 1], dtype=np.float32), mat=MAT_TERRAIN,
        )

    def test_z_at_corners_and_center(self):
        p = self._sloped_patch()
        assert p.z_at(0, 0) == pytest.approx(1.0)
        assert p.z_at(4, 0) == pytest.approx(3.0)
        assert p.z_at(4, 4) == pytest.approx(5.0)
        assert p.z_at(0, 4) == pytest.approx(2.0)
        assert p.z_at(2, 2) == pytest.approx(2.75)  # 双线性中点 = 四角均值
        assert p.z_at(2, 0) == pytest.approx(2.0)   # 沿 x 线性（z1-z0 均匀）

    def test_z_at_planar_is_constant(self):
        verts = np.array(
            [[0, 0, 5], [4, 0, 5], [4, 4, 5], [0, 4, 5]], dtype=np.float32
        )
        p = WorldPatch(
            patch_id=0, vertices=verts,
            normal=np.array([0, 0, 1], dtype=np.float32), mat=MAT_TERRAIN,
        )
        assert p.z_at(1, 3) == pytest.approx(5.0)
        assert p.z_at(2, 2) == pytest.approx(5.0)  # 结构面片退化常量

    def test_normal_at_gradient(self):
        # 沿 x 抬升的斜面：∂z/∂x=0.5, ∂z/∂y=0 → n ∝ (-0.5, 0, 1)
        verts = np.array(
            [[0, 0, 0], [4, 0, 2], [4, 4, 2], [0, 4, 0]], dtype=np.float32
        )
        p = WorldPatch(
            patch_id=0, vertices=verts,
            normal=np.array([0, 0, 1], dtype=np.float32), mat=MAT_TERRAIN,
        )
        n = p.normal_at(2, 2)
        scale = np.sqrt(0.25 + 1.0)
        assert n[0] == pytest.approx(-0.5 / scale, abs=1e-5)
        assert n[1] == pytest.approx(0.0, abs=1e-6)
        assert n[2] == pytest.approx(1.0 / scale, abs=1e-5)
        assert np.linalg.norm(n) == pytest.approx(1.0)

    def test_normal_at_planar_is_constant(self):
        verts = np.array(
            [[0, 0, 5], [4, 0, 5], [4, 4, 5], [0, 4, 5]], dtype=np.float32
        )
        p = WorldPatch(
            patch_id=0, vertices=verts,
            normal=np.array([0, 0, 1], dtype=np.float32), mat=MAT_TERRAIN,
        )
        n = p.normal_at(2, 2)
        assert n[2] == pytest.approx(1.0)
        assert n[0] == pytest.approx(0.0, abs=1e-6)

    def test_z_bilinear_flat_matches_python(self):
        """Numba 展平版 _z_bilinear_flat 与 Python 版 z_at 逐点一致。"""
        p = self._sloped_patch()
        mesh = WorldMesh(patches=[p])
        pv, vstart, vcnt, norm = flatten_world_mesh(mesh)
        for x, y in [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (2, 0), (1.3, 2.7)]:
            assert _z_bilinear_flat(
                pv, vstart, vcnt, norm, 0, x, y
            ) == pytest.approx(p.z_at(x, y))

    def test_integrate_3d_z_continuous_on_slope(self):
        """斜坡连续场 + DualContour：单位在面片内移动 z 连续（无 v0.z 阶梯）。"""
        w = h = 8
        cs = 2.0
        hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
        hf.set_ground_layer(np.zeros((h, w), dtype=np.float32))
        # 连续场 z = 0.5*x（世界 x 方向线性坡）；量化层全 0
        xs = (np.arange(w) + 0.5) * cs
        gx, _ = np.meshgrid(xs, xs)
        continuous = (0.5 * gx).astype(np.float32)

        mesh = DualContourBuilder(hf, continuous_ground=continuous).build()
        pv, vstart, vcnt, norm = flatten_world_mesh(mesh)

        # 地面格 (1,1)（x∈[2,4]）：z 从 x=2 的 1.0 线性升到 x=4 的 2.0
        pid = mesh.locate_patch(3.0, 3.0, hint_z=1.0)
        assert pid >= 0
        assert mesh.patches[pid].mat == MAT_TERRAIN

        n = 1
        pos = np.array([[2.0, 3.0, 0.0]], dtype=np.float32)
        yaw = np.zeros(n, dtype=np.float32)
        active = np.ones(n, dtype=np.bool_)
        wp = np.zeros((n, 8, 2), dtype=np.float32)
        wp_count = np.array([1], dtype=np.uint8)
        wp_idx = np.zeros(n, dtype=np.uint8)
        wp[0, 0] = (3.0, 3.0)
        speed = np.full(n, 10.0, dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        mesh_id = np.array([pid], dtype=np.int32)

        integrate_armor_movement_3d(
            pos, yaw, active, wp, wp_count, wp_idx, speed, vel, 1.0,
            mesh_id, pv, vstart, vcnt, norm,
        )
        assert pos[0, 0] == pytest.approx(3.0)
        # 双线性：z(3,3) = 1.5（旧 ray_plane_z 恒 v0.z=1.0，阶梯）
        assert pos[0, 2] == pytest.approx(1.5, abs=1e-4)
        assert int(mesh_id[0]) == pid  # 仍在面片内


# ---------------------------------------------------------------------------
# TerrainMap.world_mesh 集成
# ---------------------------------------------------------------------------


class TestTerrainMapIntegration:
    def test_world_mesh_property(self):
        from Airia.lulilo.terrain.map import TerrainMap

        hf = make_hf_with_bridge(12, 12)
        tm = TerrainMap(hf)
        mesh = tm.world_mesh
        assert isinstance(mesh, WorldMesh)
        assert len(mesh.patches) > 0
        # 懒加载缓存
        assert tm.world_mesh is mesh
