"""运行时寻路服务：A* on SurfaceMesh + 漏斗算法平滑。

对应 AGENTS.md 3.3 / docs/surface-mesh-unification.md：
面片邻接图 → A* 面片序列 → SSFA funnel → 路标点。
SurfaceMesh 替代旧 NavGraph（2026-08-11 统一架构）。

依赖: terrain.surface_mesh.SurfaceMesh（面片图）、movement.nav（A*）、coordinates.grid（坐标转换）。
"""

from __future__ import annotations

import logging

import numpy as np

from Airis.lulilo.coordinates.grid import grid_to_world, world_to_grid
from Airis.lulilo.movement.nav import patch_astar, patch_centroid_path
from Airis.lulilo.terrain.surface_mesh import SurfaceMesh

logger = logging.getLogger(__name__)


class Pathfinder:
    """寻路服务：持有 SurfaceMesh 引用 + 漏斗算法。

    用法::

        pf = Pathfinder(surface_mesh, cell_size=10.0)
        waypoints = pf.find_path((0, 0), (500, 300))
        # → list[tuple[float, float]]，世界坐标路标点
    """

    def __init__(
        self, surface_mesh: SurfaceMesh | None, cell_size: float = 10.0
    ) -> None:
        self._mesh = surface_mesh
        self._cell_size = float(cell_size)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def set_mesh(self, mesh: SurfaceMesh) -> None:
        """运行时更新面片图（热交换后调用）。"""
        self._mesh = mesh

    @property
    def mesh(self) -> SurfaceMesh | None:
        return self._mesh

    def find_path(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """寻路：世界坐标 → A* 面片序列 → 漏斗平滑 → 世界坐标路标点。

        Args:
            start: 起点世界坐标 (x, y)。
            end: 终点世界坐标 (x, y)。

        Returns:
            路标点列表（世界坐标），含起点不含终点。
            无可达路径时返回空列表。
        """
        if self._mesh is None or not self._mesh.patches:
            logger.warning("SurfaceMesh 未就绪，无法寻路")
            return []

        # A* 面片路径（patch ID 序列）
        patch_ids = patch_astar(self._mesh, start, end)
        if not patch_ids:
            return []

        # 漏斗算法平滑（SSFA，消费 portal edges）
        waypoints = funnel(
            patch_ids, self._mesh, start, end,
        )
        return waypoints

    def grid_to_world_path(
        self, gx0: int, gy0: int, gx1: int, gy1: int
    ) -> list[tuple[float, float]]:
        """栅格坐标 → 世界坐标路径（便捷方法）。"""
        sx, sy = grid_to_world(gx0, gy0, self._cell_size)
        ex, ey = grid_to_world(gx1, gy1, self._cell_size)
        return self.find_path((sx, sy), (ex, ey))


# ---------------------------------------------------------------------------
# 漏斗算法（Simple Stupid Funnel Algorithm，完整实现）
# ---------------------------------------------------------------------------


def _triarea2(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    """三角形有向面积 ×2（叉积 z 分量），SSFA 方向判定核心。"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _centroid_path(
    patch_ids: list[int], mesh: SurfaceMesh
) -> list[tuple[float, float]]:
    """面片 ID 序列 → 形心路径（不含起终点，退化后备用）。"""
    pts = []
    for pid in patch_ids:
        c = mesh.patches[pid].centroid
        pts.append((float(c[0]), float(c[1])))
    return pts


def _collect_portals(
    patch_ids: list[int],
    mesh: SurfaceMesh,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """收集路径上每条相邻面片对的 portal 边，并标注左右端点。

    左右按「沿路径前进方向（形心差）」的叉积符号确定：
        cross(dir, p - mid) > 0 → 左端点；< 0 → 右端点。
    这样每条 portal 都带明确的 (left, right)，SSFA 才正确。

    Returns:
        [(left, right), ...] 每条 portal 边的左右端点（世界坐标）。
        无共享边的相邻对跳过（退化）。
    """
    portals: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for step in range(len(patch_ids) - 1):
        pid = patch_ids[step]
        nid = patch_ids[step + 1]
        edges = mesh.portal_edges.get(pid, [])
        if not edges:
            continue
        # 取第一条共享边（应恰有一条）
        p1, p2 = edges[0]
        # 路径前进方向（形心差）
        c1 = mesh.patches[pid].centroid
        c2 = mesh.patches[nid].centroid
        dx = float(c2[0] - c1[0])
        dy = float(c2[1] - c1[1])
        # 端点在路径中线上的近似点
        mx = (p1[0] + p2[0]) * 0.5
        my = (p1[1] + p2[1]) * 0.5
        s1 = dx * (p1[1] - my) - dy * (p1[0] - mx)  # cross(dir, p1 - mid)
        s2 = dx * (p2[1] - my) - dy * (p2[0] - mx)
        if s1 > s2:
            left, right = p1, p2
        else:
            left, right = p2, p1
        portals.append((left, right))
    return portals


def funnel(
    patch_ids: list[int],
    mesh: SurfaceMesh,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    """完整 SSFA：把 A* 面片路径平滑为紧凑路标点。

    实现保证**绝对终止**：
    - 每条 portal 边只消费一次（for 循环，非 while），主推进严格单调
    - 闭合（漏斗卡住）时记录对侧边界为路标点并重置 apex，但**立即推进**
    - 另有 `max_iter` 保险丝兜底（防意外）

    Args:
        patch_ids: A* 返回的面片 ID 序列（patch_astar）。
        mesh: SurfaceMesh（含 portal_edges / patches.centroid）。
        start/end: 起终点世界坐标。

    Returns:
        路标点列表（世界坐标，含起点不含终点）。
    """
    if len(patch_ids) <= 1:
        # 起点终点在同一面片：直达（两点直线）
        return [start] if start == end else [start, end]

    portals = _collect_portals(patch_ids, mesh)
    if not portals:
        # 退化：无 portal 信息，退回简化（形心路径）
        return _simplify_path([start] + _centroid_path(patch_ids, mesh) + [end])

    waypoints: list[tuple[float, float]] = [start]

    apex = start
    left = start
    right = start

    n = len(portals)
    # 保险丝：即使逻辑异常也绝不死循环（n 条 portal × 有限次重置）
    max_iter = n * 4 + 8
    iterations = 0

    # 标记闭合后是否已重置（同一 portal 闭合一次后不再重复闭合）
    closed_left = False
    closed_right = False

    portal_idx = 0
    while portal_idx < n:
        iterations += 1
        if iterations > max_iter:
            logger.warning("SSFA 迭代超限，中止（%d portals）", n)
            break

        lx, ly = portals[portal_idx][0]
        rx, ry = portals[portal_idx][1]

        # ---- 左候选：仅在未闭合过时处理 ----
        if not closed_left and _triarea2(
            apex[0], apex[1], right[0], right[1], lx, ly
        ) >= 0.0:
            if _triarea2(apex[0], apex[1], left[0], left[1], lx, ly) < 0.0:
                # 左候选在 left 更外侧 → 收紧 left
                left = (lx, ly)
            else:
                # 左候选越过 right → 闭合：记录 right 为路标点
                if right != waypoints[-1]:
                    waypoints.append(right)
                apex = right
                left = right
                closed_left = True  # 本 portal 左候选已消费，不再重试

        # ---- 右候选：仅在未闭合过时处理 ----
        if not closed_right and _triarea2(
            apex[0], apex[1], left[0], left[1], rx, ry
        ) <= 0.0:
            if _triarea2(apex[0], apex[1], right[0], right[1], rx, ry) > 0.0:
                # 右候选在 right 更外侧 → 收紧 right
                right = (rx, ry)
            else:
                # 右候选越过 left → 闭合：记录 left 为路标点
                if left != waypoints[-1]:
                    waypoints.append(left)
                apex = left
                right = left
                closed_right = True  # 本 portal 右候选已消费

        portal_idx += 1  # 必然推进 → 绝对终止

    # 终点
    if end != waypoints[-1]:
        waypoints.append(end)

    logger.debug("漏斗平滑：%d 面片 → %d 路标点", len(patch_ids), len(waypoints))
    return waypoints


def _simplify_path(
    path: list[tuple[float, float]],
    min_dist: float = 5.0,
) -> list[tuple[float, float]]:
    """简化路径：移除过近的连续点 + 共线点精简。

    Args:
        path: 输入路径点序列。
        min_dist: 最小间距（世界单位），小于此距离的连续点合并。

    Returns:
        精简后的路标点列表。
    """
    if len(path) <= 2:
        return list(path)

    pts = np.array(path, dtype=np.float32)

    # 1. 合并过近点
    keep = [0]
    for i in range(1, len(pts) - 1):  # 保留终点
        d = np.linalg.norm(pts[i] - pts[keep[-1]])
        if d >= min_dist:
            keep.append(i)
    keep.append(len(pts) - 1)

    # 2. 共线点精简：三点共线（夹角余弦 ≈ 1）时移除中间点
    if len(keep) > 2:
        simplified = [keep[0]]
        for i in range(1, len(keep) - 1):
            a = pts[simplified[-1]]
            b = pts[keep[i]]
            c = pts[keep[i + 1]]
            ab = b - a
            bc = c - b
            len_ab = np.linalg.norm(ab)
            len_bc = np.linalg.norm(bc)
            if len_ab < 1e-6 or len_bc < 1e-6:
                continue
            cos_angle = np.dot(ab, bc) / (len_ab * len_bc)
            # 如果三折点几乎共线（cos > 0.99），跳过中间点
            if cos_angle < 0.99:
                simplified.append(keep[i])
        simplified.append(keep[-1])
        keep = simplified

    return [(float(pts[i, 0]), float(pts[i, 1])) for i in keep]
