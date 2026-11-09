"""高度场 → 导航图（NavGraph）烘焙。

算法思路参考 recastnavigation (c) 2009 Mikko Mononen，zlib License：
    https://github.com/recastnavigation/recastnavigation

对应 AGENTS.md 3.3：运行时 A* 的多边形导航图，由加载期烘焙生成。
设计决策（2026-08-10）：加载期全量 bake；运行时由 tiles 做局部 bake_tile。

实现管线：
  高度场 → 可行走掩码（坡度+爬升约束）→ 连通域标记 → 轮廓提取 → 多边形简化 → 邻接图
"""
from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from shapely.geometry import Polygon, Point

from Airia.lulilo.terrain.heightfield import NO_SURFACE, Heightfield

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NavGraph 数据结构
# ---------------------------------------------------------------------------


@dataclass
class _NavGraph:
    """导航图：多边形节点 + 邻接表（AGENTS.md 3.3）。

    Attributes:
        polys: 多边形列表，每个多边形为 (V, 2) float32 [x, y] 顶点环（首尾闭合）。
        adjacency: 邻接表 {poly_id: [neighbor_id, ...]}。
        centroids: (N, 2) float32 多边形形心，A* 启发式用。
        portal_edges: {poly_id: [(p1, p2), ...]} 该多边形与相邻多边形的共享边
            （门户边，SSFA 漏斗用）。p1/p2 为 (x, y) 世界坐标。
    """

    polys: list[np.ndarray] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    centroids: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    portal_edges: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = (
        field(default_factory=dict)
    )

    def __post_init__(self) -> None:
        if len(self.polys) and len(self.centroids) == 0:
            self._compute_centroids()

    def _compute_centroids(self) -> None:
        self.centroids = np.zeros((len(self.polys), 2), dtype=np.float32)
        for i, p in enumerate(self.polys):
            self.centroids[i] = p.mean(axis=0)

    # ------------------------------------------------------------------
    # A* 寻路
    # ------------------------------------------------------------------

    def astar(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """在多边形导航图上运行 A*。

        返回从 start 到 end 的多边形形心序列（含起终点）。
        若无可达路径，返回空列表。
        """
        if not self.polys:
            return []

        start_poly = self._locate_point(start)
        end_poly = self._locate_point(end)
        if start_poly < 0 or end_poly < 0:
            return []
        if start_poly == end_poly:
            return [start, end]

        # A* on polygon graph
        n = len(self.polys)
        g_score = {start_poly: 0.0}
        came_from: dict[int, int] = {}
        open_set: list[tuple[float, int]] = []
        heapq.heappush(open_set, (self._heuristic(start_poly, end_poly), start_poly))
        closed: set[int] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == end_poly:
                break
            closed.add(current)

            for nb in self.adjacency.get(current, []):
                if nb in closed:
                    continue
                tentative = g_score[current] + self._edge_cost(current, nb)
                if nb not in g_score or tentative < g_score[nb]:
                    g_score[nb] = tentative
                    came_from[nb] = current
                    f = tentative + self._heuristic(nb, end_poly)
                    heapq.heappush(open_set, (f, nb))

        if end_poly not in came_from:
            return []

        # 回溯路径
        path_indices: list[int] = [end_poly]
        while path_indices[-1] != start_poly:
            path_indices.append(came_from[path_indices[-1]])
        path_indices.reverse()

        # 多边形形心 + 起终点
        waypoints: list[tuple[float, float]] = [start]
        for pi in path_indices:
            c = self.centroids[pi]
            waypoints.append((float(c[0]), float(c[1])))
        waypoints.append(end)
        return waypoints

    def astar_poly_ids(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[int]:
        """A* 返回路径经过的多边形 ID 序列（SSFA 漏斗用）。

        Returns:
            多边形 ID 序列 [start_poly, ..., end_poly]；无可达路径返回空列表。
            起点/终点在同一多边形时返回 [poly]（单元素）。
        """
        if not self.polys:
            return []

        start_poly = self._locate_point(start)
        end_poly = self._locate_point(end)
        if start_poly < 0 or end_poly < 0:
            return []
        if start_poly == end_poly:
            return [start_poly]

        # A* on polygon graph（与 astar 同逻辑，仅返回 ID 序列）
        n = len(self.polys)
        g_score = {start_poly: 0.0}
        came_from: dict[int, int] = {}
        open_set: list[tuple[float, int]] = []
        heapq.heappush(open_set, (self._heuristic(start_poly, end_poly), start_poly))
        closed: set[int] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == end_poly:
                break
            closed.add(current)

            for nb in self.adjacency.get(current, []):
                if nb in closed:
                    continue
                tentative = g_score[current] + self._edge_cost(current, nb)
                if nb not in g_score or tentative < g_score[nb]:
                    g_score[nb] = tentative
                    came_from[nb] = current
                    f = tentative + self._heuristic(nb, end_poly)
                    heapq.heappush(open_set, (f, nb))

        if end_poly not in came_from:
            return []

        path_indices: list[int] = [end_poly]
        while path_indices[-1] != start_poly:
            path_indices.append(came_from[path_indices[-1]])
        path_indices.reverse()
        return path_indices

    def _locate_point(self, pt: tuple[float, float]) -> int:
        """找到包含 pt 的多边形 ID，若无返回 -1。"""
        p = Point(pt)
        for i, poly in enumerate(self.polys):
            if Polygon(poly).contains(p) or Polygon(poly).exterior.distance(p) < 1e-4:
                return i
        # 最近多边形
        best, best_dist = -1, float("inf")
        for i, c in enumerate(self.centroids):
            d = (c[0] - pt[0]) ** 2 + (c[1] - pt[1]) ** 2
            if d < best_dist:
                best, best_dist = i, d
        return best

    def _heuristic(self, a: int, b: int) -> float:
        ca = self.centroids[a]
        cb = self.centroids[b]
        return sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)

    def _edge_cost(self, a: int, b: int) -> float:
        return self._heuristic(a, b)


# ---------------------------------------------------------------------------
# Baker
# ---------------------------------------------------------------------------


class _Baker:
    """从 Heightfield 烘焙导航图（compact→region→contour→polygon）。"""

    def __init__(self, hf: Heightfield, config: dict | None = None) -> None:
        self.hf = hf
        cfg = config or {}
        self.max_slope_deg: float = float(cfg.get("max_slope_deg", 30.0))
        self.max_climb: float = float(cfg.get("max_climb", 2.0))
        self.min_region_size: int = int(cfg.get("min_region_size", 8))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def bake(self) -> NavGraph:
        """加载期全量烘焙。

        管线：可行走掩码 → 连通域 → 轮廓 → 多边形 → 邻接图。
        """
        w, h = self.hf.width, self.hf.height

        # 1. 构建可行走掩码（ground layer）
        walkable = self._build_walkable_mask()

        # 2. 连通域标记（BFS，含爬升约束）
        labels, num_regions = self._label_regions(walkable)

        # 3. 每区域提取轮廓 → 多边形
        polys = self._extract_region_polygons(labels, num_regions)

        # 4. 构建邻接表 + portal edges
        adjacency, portal_edges = self._build_adjacency(polys)

        logger.info("bake 完成：%d 个区域 → %d 个多边形", num_regions, len(polys))
        return NavGraph(polys=polys, adjacency=adjacency, portal_edges=portal_edges)

    def bake_tile(self, gx0: int, gy0: int, gx1: int, gy1: int) -> NavGraph:
        """分块烘焙（局部重建用）。

        只处理 [gx0, gx1) × [gy0, gy1) 范围内的列。
        """
        w = gx1 - gx0
        h = gy1 - gy0
        if w <= 0 or h <= 0:
            return NavGraph()

        walkable = np.zeros((h, w), dtype=np.bool_)
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                col = self.hf._col(gx, gy)  # noqa: SLF001
                if self.hf.count[col] > 0:
                    layer_top = int(self.hf.spans[col, 0, 1])  # _ZMAX
                    z = self.hf._to_z(layer_top)  # noqa: SLF001
                    if z > NO_SURFACE + 1.0:
                        walkable[gy - gy0, gx - gx0] = True

        # tile 内连通域（复用 BFS + 爬升约束）
        labels, num = self._label_regions(walkable)
        polys = self._extract_region_polygons(labels, num, offset_x=gx0, offset_y=gy0)
        adjacency, portal_edges = self._build_adjacency(polys)

        return NavGraph(polys=polys, adjacency=adjacency, portal_edges=portal_edges)

    # ------------------------------------------------------------------
    # 内部管线
    # ------------------------------------------------------------------

    def _build_walkable_mask(self) -> np.ndarray:
        """构建可行走掩码（ground layer，bool 二维数组）。

        一列可行走当且仅当 ground_z 有效（有地面 span）。
        """
        w, h = self.hf.width, self.hf.height
        walkable = np.zeros((h, w), dtype=np.bool_)
        for gy in range(h):
            for gx in range(w):
                col = self.hf._col(gx, gy)  # noqa: SLF001
                if self.hf.count[col] > 0:
                    layer_top = int(self.hf.spans[col, 0, 1])  # _ZMAX
                    z = self.hf._to_z(layer_top)  # noqa: SLF001
                    if z > NO_SURFACE + 1.0:
                        walkable[gy, gx] = True
        return walkable

    def _label_regions(
        self, walkable: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """连通域标记（BFS + 爬升约束）。

        相邻可行走格子在高度差 < max_climb 时属于同一区域。
        """
        h, w = walkable.shape
        labels = np.zeros((h, w), dtype=np.int32)
        region_id = 0

        for sy in range(h):
            for sx in range(w):
                if not walkable[sy, sx] or labels[sy, sx] != 0:
                    continue
                region_id += 1
                q: deque[tuple[int, int]] = deque()
                q.append((sy, sx))
                labels[sy, sx] = region_id
                base_z = self._ground_z_at(sx, sy)

                while q:
                    cy, cx = q.popleft()
                    for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if not walkable[ny, nx] or labels[ny, nx] != 0:
                            continue
                        nz = self._ground_z_at(nx, ny)
                        if abs(nz - base_z) > self.max_climb:
                            continue
                        labels[ny, nx] = region_id
                        q.append((ny, nx))

        return labels, region_id

    def _ground_z_at(self, gx: int, gy: int) -> float:
        """读取 (gx, gy) 处地面世界高度。"""
        col = self.hf._col(gx, gy)  # noqa: SLF001
        if self.hf.count[col] > 0:
            layer_top = int(self.hf.spans[col, 0, 1])  # _ZMAX
            return self.hf._to_z(layer_top)  # noqa: SLF001
        return NO_SURFACE

    def _extract_region_polygons(
        self, labels: np.ndarray, num_regions: int,
        offset_x: int = 0, offset_y: int = 0,
    ) -> list[np.ndarray]:
        """从连通域标签提取每个区域的多边形轮廓。

        使用 shapely unary_union 合并格方块 → 简化 → 外环顶点。
        返回世界坐标多边形顶点列表（每个为 (V,2) float32）。
        """
        from shapely.geometry import box
        from shapely.ops import unary_union

        cs = self.hf.cs
        h, w = labels.shape
        polys: list[np.ndarray] = []

        for rid in range(1, num_regions + 1):
            ys, xs = np.where(labels == rid)
            if len(ys) < self.min_region_size:
                continue

            # 按格构建方块并合并
            cells = [
                box(
                    (x + offset_x) * cs,
                    (y + offset_y) * cs,
                    (x + offset_x + 1) * cs,
                    (y + offset_y + 1) * cs,
                )
                for y, x in zip(ys, xs)
            ]
            merged = unary_union(cells)
            if merged.is_empty:
                continue

            simplified = merged.simplify(cs * 0.5, preserve_topology=True)
            if simplified.is_empty or simplified.area < (cs * cs):
                continue

            # 处理 MultiPolygon：取每个子多边形外环
            geoms = (
                simplified.geoms if hasattr(simplified, "geoms") else [simplified]
            )
            for geom in geoms:
                if geom.is_empty or geom.area < (cs * cs):
                    continue
                exterior = np.array(geom.exterior.coords, dtype=np.float32)
                if len(exterior) >= 3:
                    polys.append(exterior)

        return polys

    def _build_adjacency(
        self, polys: list[np.ndarray]
    ) -> tuple[dict[int, list[int]], dict[int, list[tuple[tuple[float, float], tuple[float, float]]]]]:
        """基于多边形之间的共享边界构建邻接表 + portal edges。

        两多边形相邻当且仅当它们的交集是一条线段（共享边）。
        共享边即 portal edge（SSFA 漏斗的核心输入）。

        Returns:
            (adjacency, portal_edges):
                adjacency: {poly_id: [neighbor_id, ...]}
                portal_edges: {poly_id: [(p1, p2), ...]} p1/p2 为 (x,y)。
        """
        n = len(polys)
        adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
        portal_edges: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = (
            {i: [] for i in range(n)}
        )
        cs = self.hf.cs
        threshold = cs * 1.1  # 略大于 cell_size，容差

        shape_polys = [Polygon(p) for p in polys]

        for i in range(n):
            for j in range(i + 1, n):
                # 快速剔除：用形心距离
                ci = np.mean(polys[i], axis=0)
                cj = np.mean(polys[j], axis=0)
                if np.linalg.norm(ci - cj) > cs * 20:
                    continue

                dist = shape_polys[i].distance(shape_polys[j])
                if dist >= threshold:
                    continue

                # 求交集 → 共享边（LineString）才是 portal
                inter = shape_polys[i].intersection(shape_polys[j])
                if inter.is_empty or inter.geom_type != "LineString":
                    continue
                if inter.length < cs * 0.5:
                    continue
                coords = list(inter.coords)
                p1 = (float(coords[0][0]), float(coords[0][1]))
                p2 = (float(coords[-1][0]), float(coords[-1][1]))

                adjacency[i].append(j)
                adjacency[j].append(i)
                portal_edges[i].append((p1, p2))
                portal_edges[j].append((p2, p1))  # 反向端点，保持「自己 → 邻居」方向一致

        return adjacency, portal_edges


# ---------------------------------------------------------------------------
# ⚠️ 废弃兼容别名（2026-08-11）
# _NavGraph / _Baker 已被 SurfaceMesh + GreedyMesh 统一架构替代：
#   - 导航图   → terrain.surface_mesh.SurfaceMesh
#   - A* 寻路  → movement.nav.patch_astar
# 此处别名仅供旧测试/调试脚本兼容，新代码一律禁止使用。
# ---------------------------------------------------------------------------
NavGraph = _NavGraph
Baker = _Baker
