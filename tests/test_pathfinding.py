"""漏斗算法（SSFA）单元测试（2026-08-11 适配 SurfaceMesh 统一架构）。

覆盖：
    - portal edges 提取（共享边 → 左右端点）
    - SSFA 路径平滑（走廊场景：路径收紧、含起终点）
    - 防死循环（闭合分支必然推进 + 保险丝）
    - 退化场景（无 portal → 退回形心路径简化）
    - 与 patch_astar 集成（SurfaceMesh 面片图）

运行：uv run python -m pytest tests/test_pathfinding.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.movement.nav import patch_astar
from Airis.lulilo.movement.pathfinding import (
    Pathfinder,
    _centroid_path,
    _collect_portals,
    _simplify_path,
    _triarea2,
    funnel,
)
from Airis.lulilo.terrain.surface_mesh import SurfaceMesh, SurfacePatch


# ---------------------------------------------------------------------------
# 走廊 SurfaceMesh：左室 → 窄门 → 右室（3 个面片，共享边明确）
# ---------------------------------------------------------------------------


def make_corridor_mesh() -> SurfaceMesh:
    p0 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    p1 = np.array([[10, 4], [14, 4], [14, 6], [10, 6]], dtype=np.float32)
    p2 = np.array([[14, 0], [24, 0], [24, 10], [14, 10]], dtype=np.float32)
    mesh = SurfaceMesh(
        patches=[
            SurfacePatch(patch_id=0, vertices=p0, z=0.0, mat=0, walkable=True),
            SurfacePatch(patch_id=1, vertices=p1, z=0.0, mat=0, walkable=True),
            SurfacePatch(patch_id=2, vertices=p2, z=0.0, mat=0, walkable=True),
        ],
    )
    mesh.adjacency = {0: [1], 1: [0, 2], 2: [1]}
    mesh.portal_edges = {
        0: [((10, 4), (10, 6))],
        1: [((14, 4), (14, 6)), ((10, 4), (10, 6))],
        2: [((14, 4), (14, 6))],
    }
    mesh.walkable_patches = {0, 1, 2}
    return mesh


class TestCollectPortals:
    def test_portal_edges_extracted(self):
        """走廊路径应收集到 2 条 portal（左门 + 右门）。"""
        mesh = make_corridor_mesh()
        portals = _collect_portals([0, 1, 2], mesh)
        assert len(portals) == 2
        # 每条 portal 有左右端点
        for left, right in portals:
            assert len(left) == 2 and len(right) == 2

    def test_no_portal_fallback_empty(self):
        """无 portal edges 时返回空列表（上层应退回形心路径）。"""
        mesh = make_corridor_mesh()
        mesh.portal_edges = {}
        assert _collect_portals([0, 1, 2], mesh) == []


class TestFunnel:
    def test_same_patch_direct(self):
        """起终点同一面片 → 直达两点（起点→终点）。"""
        mesh = make_corridor_mesh()
        assert funnel([0], mesh, (2.0, 2.0), (8.0, 8.0)) == [(2.0, 2.0), (8.0, 8.0)]

    def test_corridor_path_terminates(self):
        """走廊路径必然终止，含起终点，且经过窄门收紧为直线。"""
        mesh = make_corridor_mesh()
        pid = patch_astar(mesh, (2.0, 2.0), (22.0, 8.0))
        assert pid == [0, 1, 2]
        wps = funnel(pid, mesh, (2.0, 2.0), (22.0, 8.0))
        assert wps[0] == (2.0, 2.0)
        assert wps[-1] == (22.0, 8.0)
        assert len(wps) >= 2  # 至少起终点
        # 全部路标点应在走廊范围内（x 在 [0, 24]，y 在 [0, 10]）
        for wx, wy in wps:
            assert 0.0 <= wx <= 24.0
            assert 0.0 <= wy <= 10.0

    def test_no_deadlock_on_closure(self):
        """闭合分支必然推进（防死循环）：大量紧邻 portal 不卡死。"""
        mesh = make_corridor_mesh()
        # 构造 20 条相同 portal（极端退化），必须终止
        portals = [((10, 4), (10, 6))] * 20
        mesh2 = make_corridor_mesh()
        mesh2.portal_edges = {i: portals for i in range(3)}
        pid = patch_astar(mesh2, (2.0, 2.0), (22.0, 8.0))
        wps = funnel(pid, mesh2, (2.0, 2.0), (22.0, 8.0))
        assert len(wps) >= 2

    def test_pathfinder_integration(self):
        """Pathfinder.find_path 走完整管线（A* + SSFA）。"""
        mesh = make_corridor_mesh()
        pf = Pathfinder(mesh)
        wps = pf.find_path((2.0, 2.0), (22.0, 8.0))
        assert wps[0] == (2.0, 2.0)
        assert wps[-1] == (22.0, 8.0)
        assert len(wps) >= 2

    def test_no_mesh_returns_empty(self):
        """SurfaceMesh 缺失 → 空路径（不崩溃）。"""
        pf = Pathfinder(None)
        assert pf.find_path((0.0, 0.0), (1.0, 1.0)) == []


class TestHelpers:
    def test_triarea2_sign(self):
        """有向面积：逆时针为正，顺时针为负。"""
        assert _triarea2(0, 0, 1, 0, 1, 1) > 0.0   # CCW
        assert _triarea2(0, 0, 1, 0, 1, -1) < 0.0  # CW

    def test_centroid_path(self):
        """形心路径（退化备用）。"""
        mesh = make_corridor_mesh()
        pts = _centroid_path([0, 1, 2], mesh)
        assert len(pts) == 3

    def test_simplify_collinear(self):
        """简化路径：共线点被移除。"""
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        out = _simplify_path(pts, min_dist=0.5)
        assert (0.0, 0.0) == out[0]
        assert (3.0, 0.0) == out[-1]
        assert len(out) <= 2  # 共线中间点被移除
