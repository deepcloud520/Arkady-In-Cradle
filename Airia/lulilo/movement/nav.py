"""三角形 NavGraph 寻路：A* 在三角形邻接图上搜索 + 路径平滑。

对应 docs/navmesh-region-growth-design.md §4：「A* 三角形邻接图 → SSFA funnel」。

新管线（2026-08-12 切换）：
- 导航节点 = NavTriangle（区域生长 + 三角剖分，NavMeshBuilder 产出）
- navgraph_astar：三角形邻接 A*（含跨层垂直 portal）

已废弃（兼容别名保留，供旧测试/调试脚本）：
- _patch_astar / _patch_centroid_path：SurfaceMesh 面片图 A*（GreedyMesh 旧管线）
"""
from __future__ import annotations

import heapq
import logging
from math import sqrt

import numpy as np

from Airia.lulilo.terrain.navmesh_builder import NavGraph
from Airia.lulilo.terrain.surface_mesh import SurfaceMesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A* 面片寻路（已废弃 → 用 navgraph_astar）
# ---------------------------------------------------------------------------


def _patch_astar(
    mesh: SurfaceMesh,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[int]:
    """[已废弃] 在 SurfaceMesh 面片邻接图上运行 A*，返回面片 ID 序列。

    用 navgraph_astar 替代（docs/navmesh-region-growth-design.md）。

    Args:
        mesh: SurfaceMesh（含 patches + adjacency）。
        start: 起点世界坐标 (x, y)。
        end: 终点世界坐标 (x, y)。

    Returns:
        面片 ID 序列 [start_patch, ..., end_patch]；无可达路径返回空列表。
        起点/终点在同一面片时返回 [patch_id]（单元素）。
    """
    if not mesh.patches or not mesh.walkable_patches:
        return []

    start_patch = mesh.locate_walkable(*start)
    end_patch = mesh.locate_walkable(*end)

    if start_patch < 0 or end_patch < 0:
        logger.debug("A* 起点(%d)或终点(%d)不在可行走面片上", start_patch, end_patch)
        return []

    if start_patch == end_patch:
        return [start_patch]

    # A* 主体
    g_score: dict[int, float] = {start_patch: 0.0}
    came_from: dict[int, int] = {}
    open_set: list[tuple[float, int]] = []
    heapq.heappush(
        open_set,
        (_heuristic(mesh, start_patch, end_patch), start_patch),
    )
    closed: set[int] = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == end_patch:
            break
        closed.add(current)

        for nb in mesh.adjacency.get(current, []):
            if nb in closed or nb not in mesh.walkable_patches:
                continue
            tentative = g_score[current] + _edge_cost(mesh, current, nb)
            if nb not in g_score or tentative < g_score[nb]:
                g_score[nb] = tentative
                came_from[nb] = current
                f = tentative + _heuristic(mesh, nb, end_patch)
                heapq.heappush(open_set, (f, nb))

    if end_patch not in came_from:
        logger.debug("A* 无可达路径: %d → %d", start_patch, end_patch)
        return []

    # 回溯
    path: list[int] = [end_patch]
    while path[-1] != start_patch:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def _heuristic(mesh: SurfaceMesh, a: int, b: int) -> float:
    """A* 启发式：两 patch 形心欧氏距离（admissible）。"""
    ca = mesh.patches[a].centroid
    cb = mesh.patches[b].centroid
    return sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def _edge_cost(mesh: SurfaceMesh, a: int, b: int) -> float:
    """面片间边代价：形心距离 + z 落差惩罚。

    z 落差惩罚（×3）使 A* 倾向在同一平面移动，避免频繁上下桥。
    """
    pa = mesh.patches[a]
    pb = mesh.patches[b]
    dist = sqrt(
        (pa.centroid[0] - pb.centroid[0]) ** 2
        + (pa.centroid[1] - pb.centroid[1]) ** 2
    )
    dz = abs(pa.z - pb.z)
    return dist + dz * 3.0


# ---------------------------------------------------------------------------
# 面片路径 → 形心路径（已废弃，退化备用）
# ---------------------------------------------------------------------------


def _patch_centroid_path(
    mesh: SurfaceMesh,
    patch_ids: list[int],
) -> list[tuple[float, float]]:
    """[已废弃] 面片 ID 序列 → 形心路径（不含起终点，退化后备用）。

    当 portal 信息不足时（如路径仅 1 个面片），退回此简化路径。
    """
    return [mesh.patches[pid].centroid for pid in patch_ids]


# ---------------------------------------------------------------------------
# NavGraph（三角形）A* 寻路（新管线，docs §4.1）
# ---------------------------------------------------------------------------


def navgraph_astar(
    graph: NavGraph,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[int]:
    """在三角形 NavGraph 上运行 A*，返回三角形 ID 序列。

    NavGraph 所有三角形均可行走（区域生长只产生可行走格）；邻接含跨层
    垂直 portal（退化点连接，见 navmesh_builder._inject_vertical_portal）。

    Args:
        graph: NavGraph（triangles + adjacency）。
        start/end: 起终点世界坐标。

    Returns:
        三角形 ID 序列 [start_tri, ..., end_tri]；无可达路径返回空列表。
        起终点在同一三角形时返回 [tri_id]（单元素）。
    """
    if not graph.triangles or not graph.adjacency:
        return []

    start_tri = graph.locate_triangle(*start)
    end_tri = graph.locate_triangle(*end)
    if start_tri == end_tri:
        return [start_tri]

    g_score: dict[int, float] = {start_tri: 0.0}
    came_from: dict[int, int] = {}
    open_set: list[tuple[float, int]] = []
    heapq.heappush(
        open_set,
        (_graph_heuristic(graph, start_tri, end_tri), start_tri),
    )
    closed: set[int] = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == end_tri:
            break
        closed.add(current)

        for nb in graph.adjacency.get(current, []):
            if nb in closed:
                continue
            tentative = g_score[current] + _graph_edge_cost(graph, current, nb)
            if nb not in g_score or tentative < g_score[nb]:
                g_score[nb] = tentative
                came_from[nb] = current
                f = tentative + _graph_heuristic(graph, nb, end_tri)
                heapq.heappush(open_set, (f, nb))

    if end_tri not in came_from:
        logger.debug("A* 无可达路径: %d → %d", start_tri, end_tri)
        return []

    path: list[int] = [end_tri]
    while path[-1] != start_tri:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def _graph_heuristic(graph: NavGraph, a: int, b: int) -> float:
    """A* 启发式：两三角形形心欧氏距离（admissible）。"""
    ca = graph.triangles[a].centroid
    cb = graph.triangles[b].centroid
    return sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def _graph_edge_cost(graph: NavGraph, a: int, b: int) -> float:
    """三角形间边代价：形心距离 + z 落差惩罚（×3，跨层代价高）。"""
    pa = graph.triangles[a]
    pb = graph.triangles[b]
    dist = sqrt(
        (pa.centroid[0] - pb.centroid[0]) ** 2
        + (pa.centroid[1] - pb.centroid[1]) ** 2
    )
    dz = abs(pa.z_world - pb.z_world)
    return dist + dz * 3.0


# ---------------------------------------------------------------------------
# 兼容别名（废弃约定：模块内保留，供旧调试脚本/测试）
# ---------------------------------------------------------------------------

patch_astar = _patch_astar
patch_centroid_path = _patch_centroid_path
