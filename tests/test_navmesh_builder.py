"""导航网格重构（区域生长 + 三角剖分）骨架测试。

覆盖 docs/navmesh-region-growth-design.md 骨架（2026-08-12）：
    - 数据模型（WalkableConfig / WalkableCell / NavTriangle / NavGraph）
    - 可行走判定（is_top_surface / is_hanging / slope_at）
    - ① 可行走格提取（平地全可走 / 桥面独立 / 建筑覆盖地面）
    - ② 区域生长（同区域合并 / mat 隔离 / z 隔离 / 最小格数）
    - ③ 轮廓提取（shapely 多边形）
    - ④ 三角剖分（凸扇形 / 凹与带洞 → NotImplementedError）
    - ⑤ 邻接图（共享边邻接 + portal）
    - 点定位（locate_triangle / point_in_triangle）
    - NavMeshBuilder.build 端到端（凸区域地图）

运行：uv run python -m pytest tests/test_navmesh_builder.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.movement.nav import navgraph_astar
from Airis.lulilo.movement.pathfinding import GraphPathfinder, funnel_graph
from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_BUILDING,
    MAT_TERRAIN,
    Heightfield,
)
from Airis.lulilo.terrain.navmesh_builder import (
    NavGraph,
    NavMeshBuilder,
    NavTriangle,
    WalkableCell,
    WalkableConfig,
    build_navgraph,
    build_neighbor_map,
    extract_walkable_cells,
    grow_regions,
    is_convex_polygon,
    is_hanging,
    is_top_surface,
    link_cross_layer,
    locate_triangle,
    point_in_triangle,
    polygon_to_list,
    region_to_polygon,
    slope_at,
    triangulate_polygon,
    xy_overlap,
)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def make_hf(
    w: int = 10,
    h: int = 10,
    cs: float = 10.0,
    ch: float = 1.0,
    z_min: float = -10.0,
    z_max: float = 50.0,
) -> Heightfield:
    return Heightfield(w, h, cs, ch, z_min, z_max)


def make_flat(w: int = 10, h: int = 10, ground_z: float = 2.0) -> Heightfield:
    hf = make_hf(w, h)
    hf.set_ground_layer(np.full((w * h,), ground_z, dtype=np.float32))
    return hf


def _tri_area2_pts(a, b, c) -> float:
    """三点三角形面积（叉积 ×0.5，绝对值）。"""
    return 0.5 * abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    )


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


def test_walkable_config_defaults() -> None:
    cfg = WalkableConfig()
    assert cfg.max_slope_deg == 30.0
    assert cfg.max_climb == 2.0
    assert cfg.min_region_cells == 8
    assert cfg.gap_threshold == 1
    assert cfg.effective_simplify_tolerance(10.0) == 5.0
    assert cfg.effective_portal_min_length(10.0) == 5.0
    cfg2 = WalkableConfig(simplify_tolerance=1.0)
    assert cfg2.effective_simplify_tolerance(10.0) == 1.0


def test_walkable_cell_fields() -> None:
    c = WalkableCell(gx=1, gy=2, z_layer=12, z_world=2.0, mat=MAT_TERRAIN, col=42)
    assert c.gx == 1 and c.gy == 2 and c.mat == MAT_TERRAIN
    assert c.z_layer == 12 and c.z_world == 2.0 and c.col == 42


def test_nav_graph_locate_wrapper() -> None:
    t0 = NavTriangle(
        0, np.array([[0, 0], [10, 0], [10, 10]], np.float32), (20 / 3, 10 / 3), 2.0, 0
    )
    g = NavGraph(triangles=[t0])
    assert g.triangle_count == 1
    assert g.locate_triangle(1.0, 1.0) == 0


# ---------------------------------------------------------------------------
# 可行走判定
# ---------------------------------------------------------------------------


def test_is_top_surface_single_span() -> None:
    hf = make_flat()
    col = hf._col(0, 0)
    assert is_top_surface(hf, col, 0, 1) is True  # 单层 → 顶面


def test_is_top_surface_bridge_gap() -> None:
    hf = make_flat(w=4, h=1)
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_BRIDGE)  # 同列上方桥
    col = hf._col(0, 0)
    assert int(hf.count[col]) == 2
    assert is_top_surface(hf, col, 0, 1) is True  # 地面：间隙 ≥ 1
    assert is_top_surface(hf, col, 1, 1) is True  # 桥面：最后一层


def test_is_top_surface_building_covers_ground() -> None:
    hf = make_flat(w=1, h=1, ground_z=2.0)
    hf.add_span(0, 0, 2.0, 8.0, mat=MAT_BUILDING)  # 紧贴地面
    col = hf._col(0, 0)
    assert is_top_surface(hf, col, 0, 1) is False  # 地面被覆盖 → 非顶面
    assert is_top_surface(hf, col, 1, 1) is True  # 建筑顶面 → 顶面


def test_is_hanging_terrain_segment() -> None:
    hf = make_hf(1, 1)
    hf.add_span(0, 0, 0.0, 8.0, mat=MAT_TERRAIN)
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_TERRAIN)  # 悬空段（间隙 4）
    assert is_hanging(hf, 0, 0, 1, 1) is True
    assert is_hanging(hf, 0, 0, 0, 1) is False  # 地面有支撑


def test_is_hanging_artificial_never_hanging() -> None:
    hf = make_flat(w=1, h=1)
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_BRIDGE)
    assert is_hanging(hf, 0, 0, 1, 1) is False  # 人工结构自身可站立


def test_slope_at_flat_zero() -> None:
    hf = make_flat()
    assert slope_at(hf, 5, 5, 0) == pytest.approx(0.0, abs=1e-3)


def test_slope_at_steep() -> None:
    hf = make_hf(3, 1)
    hf.set_ground_layer(np.array([2.0, 14.0, 2.0], dtype=np.float32))
    # 中心 14m，两侧 2m → 梯度 1.2 → atan ≈ 50° > 30°
    assert slope_at(hf, 1, 0, 0) > 30.0


# ---------------------------------------------------------------------------
# ① 可行走格提取
# ---------------------------------------------------------------------------


def test_extract_flat_all_walkable() -> None:
    hf = make_flat(w=10, h=10)
    cells = extract_walkable_cells(hf, WalkableConfig())
    assert len(cells) == 100
    assert all(c.mat == MAT_TERRAIN for c in cells)
    assert all(c.z_world == pytest.approx(2.0) for c in cells)


def test_extract_bridge_and_ground() -> None:
    hf = make_flat(w=4, h=1)
    hf.add_span(1, 0, 12.0, 14.0, mat=MAT_BRIDGE)  # 只覆盖 (1,0)
    cells = extract_walkable_cells(hf, WalkableConfig())
    by_mat: dict[int, list[WalkableCell]] = {}
    for c in cells:
        by_mat.setdefault(c.mat, []).append(c)
    assert len(by_mat[MAT_TERRAIN]) == 4  # 地面 4 格
    assert len(by_mat[MAT_BRIDGE]) == 1  # 桥面 1 格


def test_extract_building_covers_ground() -> None:
    hf = make_flat(w=2, h=2)
    hf.add_span(0, 0, 2.0, 8.0, mat=MAT_BUILDING)
    cells = extract_walkable_cells(hf, WalkableConfig())
    mats = {(c.gx, c.gy, c.mat) for c in cells}
    assert (0, 0, MAT_TERRAIN) not in mats  # 地面被覆盖
    assert (0, 0, MAT_BUILDING) in mats  # 建筑顶面
    assert (1, 1, MAT_TERRAIN) in mats  # 未覆盖处保留


# ---------------------------------------------------------------------------
# ② 区域生长
# ---------------------------------------------------------------------------


def test_grow_regions_single_flat_region() -> None:
    hf = make_flat(w=4, h=4)
    cells = extract_walkable_cells(hf, WalkableConfig())
    regions = grow_regions(cells, max_climb=2.0, min_region_cells=1)
    assert len(regions) == 1
    assert len(regions[0]) == 16


def test_grow_regions_mat_isolation() -> None:
    hf = make_flat(w=4, h=1)
    hf.add_span(2, 0, 12.0, 14.0, mat=MAT_BRIDGE)
    cells = extract_walkable_cells(hf, WalkableConfig())
    regions = grow_regions(cells, max_climb=2.0, min_region_cells=1)
    assert len(regions) == 2  # TERRAIN 区 + BRIDGE 区


def test_grow_regions_z_isolation() -> None:
    hf = make_hf(4, 1)
    hf.set_ground_layer(np.array([2.0, 6.0, 10.0, 14.0], dtype=np.float32))
    cells = extract_walkable_cells(hf, WalkableConfig())
    regions = grow_regions(cells, max_climb=2.0, min_region_cells=1)
    assert len(regions) == 4  # 相邻差 4 > max_climb → 各自独立


def test_grow_regions_min_cells_filter() -> None:
    hf = make_hf(4, 1)
    hf.set_ground_layer(np.array([2.0, 6.0, 10.0, 14.0], dtype=np.float32))
    cells = extract_walkable_cells(hf, WalkableConfig())
    regions = grow_regions(cells, max_climb=2.0, min_region_cells=8)
    assert regions == []  # 每区域 1 格 < 8 → 全丢弃


def test_build_neighbor_map_counts() -> None:
    hf = make_flat(w=3, h=1)
    cells = extract_walkable_cells(hf, WalkableConfig())
    neighbors = build_neighbor_map(cells)
    # 端点 1 邻，中间 2 邻（同 mat 四邻域）
    assert [len(n) for n in neighbors] == [1, 2, 1]


# ---------------------------------------------------------------------------
# ③ 轮廓提取
# ---------------------------------------------------------------------------


def test_region_to_polygon_rectangle() -> None:
    hf = make_flat(w=2, h=2)
    cells = extract_walkable_cells(hf, WalkableConfig())
    poly = region_to_polygon({0, 1, 2, 3}, cells, hf)
    assert poly is not None
    assert poly.area == pytest.approx(4.0 * hf.cs * hf.cs)  # 4 格合并 20×20


def test_polygon_to_list_multi() -> None:
    from shapely.geometry import MultiPolygon, Polygon

    mp = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ]
    )
    polys = polygon_to_list(mp)
    assert len(polys) == 2


def test_polygon_to_list_none() -> None:
    assert polygon_to_list(None) == []


# ---------------------------------------------------------------------------
# ④ 三角剖分
# ---------------------------------------------------------------------------


def test_triangulate_convex_fan() -> None:
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    tris = triangulate_polygon(poly)
    assert len(tris) == 2
    # 每个三角形为 3 个世界坐标点
    for t in tris:
        assert len(t) == 3
    assert sum(_tri_area2_pts(*t) for t in tris) == pytest.approx(100.0)


def test_is_convex_polygon() -> None:
    rect = [(0, 0), (10, 0), (10, 10), (0, 10)]
    concave = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]
    assert is_convex_polygon(rect) is True
    assert is_convex_polygon(concave) is False


def test_triangulate_concave() -> None:
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)])  # 凹口 (2,2)
    tris = triangulate_polygon(poly)
    assert len(tris) == 3  # 5 顶点 → 3 三角形
    assert sum(_tri_area2_pts(*t) for t in tris) == pytest.approx(12.0)


def test_triangulate_with_hole() -> None:
    from shapely.geometry import Polygon

    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(2, 2), (8, 2), (8, 8), (2, 8)]],
    )
    tris = triangulate_polygon(poly)
    assert len(tris) >= 6  # 8 顶点 + 桥接 ≥ 6 三角形
    # 总面积 = 外环 100 - 洞 36 = 64
    assert sum(_tri_area2_pts(*t) for t in tris) == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# ⑤ 邻接图 + 点定位
# ---------------------------------------------------------------------------


def _tri(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> NavTriangle:
    v = np.array([[x0, y0], [x1, y1], [x2, y2]], dtype=np.float32)
    return NavTriangle(
        0, v, (float(v[:, 0].mean()), float(v[:, 1].mean())), 2.0, MAT_TERRAIN
    )


def test_build_navgraph_shared_edge() -> None:
    t0 = _tri(0, 0, 10, 0, 10, 10)
    t1 = _tri(0, 0, 10, 10, 0, 10)
    t0.tri_id, t1.tri_id = 0, 1
    g = build_navgraph([t0, t1], tol=0.1)
    assert g.adjacency[0] == [1]
    assert g.adjacency[1] == [0]
    assert len(g.portal_edges[0]) == 1
    assert len(g.portal_edges[1]) == 1


def test_build_navgraph_disjoint() -> None:
    t0 = _tri(0, 0, 10, 0, 10, 10)
    t1 = _tri(100, 100, 110, 100, 110, 110)
    t0.tri_id, t1.tri_id = 0, 1
    g = build_navgraph([t0, t1], tol=0.1)
    assert g.adjacency == {}


def test_locate_triangle() -> None:
    t0 = _tri(0, 0, 10, 0, 10, 10)
    t1 = _tri(0, 0, 10, 10, 0, 10)
    t0.tri_id, t1.tri_id = 0, 1
    tris = [t0, t1]
    assert locate_triangle(1.0, 1.0, tris) == 0
    assert locate_triangle(1.0, 9.0, tris) == 1


def test_point_in_triangle() -> None:
    v = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
    assert point_in_triangle(5.0, 2.0, v) is True
    assert point_in_triangle(5.0, 8.0, v) is False


# ---------------------------------------------------------------------------
# NavMeshBuilder 端到端（骨架：凸区域地图）
# ---------------------------------------------------------------------------


def test_builder_build_convex_flat() -> None:
    hf = make_flat(w=10, h=10)
    graph = NavMeshBuilder(hf).build()
    assert graph.triangle_count == 2  # 10×10 凸矩形 → 2 三角形
    assert len(graph.adjacency) == 2  # 两三角形共享对角线
    assert graph.vertical_portals == []  # 骨架：跨层未实现


def test_builder_build_bridge_cross_layer() -> None:
    hf = make_flat(w=12, h=2, ground_z=2.0)
    for gy in range(2):
        for gx in range(4, 10):
            hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)  # 桥面 z=4，与地面差 2
    graph = NavMeshBuilder(hf).build()
    # TERRAIN 区（24 格凸矩形）+ BRIDGE 区（12 格凸矩形）各 2 三角形
    assert graph.triangle_count == 4
    # 桥面与地面 z 差 2 ≤ max_climb=2 → 跨层垂直 portal 生成并注入邻接表
    assert len(graph.vertical_portals) > 0
    for a, b in graph.vertical_portals:
        assert b in graph.adjacency[a]
        assert a in graph.adjacency[b]


# ---------------------------------------------------------------------------
# ⑥ 跨层垂直 portal
# ---------------------------------------------------------------------------


def test_link_cross_layer_bridge_ground() -> None:
    hf = make_flat(w=12, h=2, ground_z=2.0)
    for gy in range(2):
        for gx in range(4, 10):
            hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)
    graph = NavMeshBuilder(hf).build()
    vp = link_cross_layer(graph.triangles, max_climb=2.0)
    assert len(vp) > 0
    # 每对固定为 (TERRAIN, BRIDGE) 顺序（pairs 定义）
    for a, b in vp:
        assert graph.triangles[a].mat == MAT_TERRAIN
        assert graph.triangles[b].mat == MAT_BRIDGE


def test_link_cross_layer_z_too_far() -> None:
    hf = make_flat(w=12, h=2, ground_z=2.0)
    for gy in range(2):
        for gx in range(4, 10):
            hf.add_span(gx, gy, 12.0, 14.0, mat=MAT_BRIDGE)  # 桥面 z=12，差 10
    graph = NavMeshBuilder(hf).build()
    vp = link_cross_layer(graph.triangles, max_climb=2.0)
    assert vp == []  # |z 差| 10 > max_climb → 无垂直 portal


def test_xy_overlap_true_false() -> None:
    t0 = NavTriangle(
        0, np.array([[0, 0], [10, 0], [10, 10]], np.float32), (20 / 3, 10 / 3), 2.0, MAT_TERRAIN
    )
    t1 = NavTriangle(
        1, np.array([[0, 0], [10, 10], [0, 10]], np.float32), (10 / 3, 20 / 3), 2.0, MAT_TERRAIN
    )
    t2 = NavTriangle(
        2, np.array([[100, 100], [110, 100], [110, 110]], np.float32), (320 / 3, 310 / 3), 2.0, MAT_TERRAIN
    )
    assert xy_overlap(t0, t1) is True  # 共享边
    assert xy_overlap(t0, t2) is False  # 分离


# ---------------------------------------------------------------------------
# NavGraph 寻路（navgraph_astar / funnel_graph / GraphPathfinder）
# ---------------------------------------------------------------------------


def test_navgraph_astar_two_triangles() -> None:
    t0 = _tri(0, 0, 10, 0, 10, 10)
    t1 = _tri(0, 0, 10, 10, 0, 10)
    t0.tri_id, t1.tri_id = 0, 1
    graph = build_navgraph([t0, t1], tol=0.1)
    path = navgraph_astar(graph, (2, 1), (1, 8))
    assert path == [0, 1]


def test_funnel_graph_two_triangles() -> None:
    t0 = _tri(0, 0, 10, 0, 10, 10)
    t1 = _tri(0, 0, 10, 10, 0, 10)
    t0.tri_id, t1.tri_id = 0, 1
    graph = build_navgraph([t0, t1], tol=0.1)
    wps = funnel_graph([0, 1], graph, (2, 1), (1, 8))
    assert wps[0] == (2, 1)
    assert wps[-1] == (1, 8)
    assert len(wps) >= 2


def test_graph_pathfinder_flat() -> None:
    hf = make_flat(w=10, h=10)
    graph = NavMeshBuilder(hf).build()
    gpf = GraphPathfinder(graph)
    wps = gpf.find_path((5, 5), (95, 95))
    assert wps and wps[0] == (5, 5)
    assert wps[-1] == (95, 95)


def test_graph_pathfinder_cross_layer() -> None:
    """地面 → 桥面：A* 通过垂直 portal 找到跨层路径。"""
    hf = make_flat(w=12, h=2, ground_z=2.0)
    for gy in range(2):
        for gx in range(4, 10):
            hf.add_span(gx, gy, 3.0, 4.0, mat=MAT_BRIDGE)
    graph = NavMeshBuilder(hf).build()
    assert len(graph.vertical_portals) > 0
    gpf = GraphPathfinder(graph)
    # start 在地面（x=5m），end 在桥面（x=60m, y=15m）
    wps = gpf.find_path((5, 5), (60, 15))
    assert wps and wps[0] == (5, 5)
    assert wps[-1] == (60, 15)


def test_builder_build_ring_with_hole() -> None:
    """环形地形端到端：外围可行走 + 内部低洼 → 带洞三角剖分 + 独立坑底区域。

    5×5 格：外围边框 z=4，内部 3×3 z=1（差 3 > max_climb → 两独立区域）。
    外环三角剖分应避开洞；坑底（z=1）独立成凸矩形 2 三角形。
    """
    hf = make_hf(5, 5)
    zs = np.full((25,), 4.0, dtype=np.float32)
    for gy in range(1, 4):
        for gx in range(1, 4):
            zs[gy * 5 + gx] = 1.0
    hf.set_ground_layer(zs)

    graph = NavMeshBuilder(hf).build()
    # 外环带洞 8 三角形（避开 30×30 洞）+ 坑底凸矩形 2 三角形
    assert graph.triangle_count == 10

    def _area(v) -> float:
        p, q, r = v
        return 0.5 * abs(
            (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        )

    # 总面积 = 5×5 格（2500 m²）；外环 = 2500 - 900 = 1600 m²
    assert sum(_area(t.vertices) for t in graph.triangles) == pytest.approx(2500.0)
    ring_area = sum(
        _area(t.vertices)
        for t in graph.triangles
        if t.z_world == pytest.approx(4.0)
    )
    assert ring_area == pytest.approx(1600.0)

    # 坑底（z=1）独立区域，与外环无垂直 portal（z 差 3 > max_climb）
    assert graph.vertical_portals == []
    assert len(graph.adjacency) == 10

    # 寻路：外环下边两点直达（不穿坑）
    gpf = GraphPathfinder(graph)
    wps = gpf.find_path((5, 5), (45, 5))
    assert wps and wps[0] == (5, 5) and wps[-1] == (45, 5)
