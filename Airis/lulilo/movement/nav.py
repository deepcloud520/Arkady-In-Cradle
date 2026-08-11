"""面片导航：A* 在 SurfaceMesh 面片邻接图上搜索 + 路径平滑。

对应 docs/surface-mesh-unification.md 第二章：「A* 面片邻接图 → 面片序列 → SSFA funnel」。

设计决策（2026-08-11）：
- 替代 navmesh/baker.py 的 A* 部分（baker 的多边形提取管线整体删除）
- SurfaceMesh 面片 = 导航节点；邻接边 = 共享 portal
- 面片可能是凹多边形 → 必须用 SSFA funnel 收紧，不能单纯形心连线
- heuristic = 两点欧氏距离（形心→形心），admissible
"""
from __future__ import annotations

import heapq
import logging
from math import sqrt

import numpy as np

from Airis.lulilo.terrain.surface_mesh import SurfaceMesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A* 面片寻路
# ---------------------------------------------------------------------------


def patch_astar(
    mesh: SurfaceMesh,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[int]:
    """在 SurfaceMesh 面片邻接图上运行 A*，返回面片 ID 序列。

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
# 面片路径 → 形心路径（退化备用）
# ---------------------------------------------------------------------------


def patch_centroid_path(
    mesh: SurfaceMesh,
    patch_ids: list[int],
) -> list[tuple[float, float]]:
    """面片 ID 序列 → 形心路径（不含起终点，退化后备用）。

    当 portal 信息不足时（如路径仅 1 个面片），退回此简化路径。
    """
    return [mesh.patches[pid].centroid for pid in patch_ids]
