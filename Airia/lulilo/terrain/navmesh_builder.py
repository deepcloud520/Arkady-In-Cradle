"""导航网格重构：按可走性区域生长 + 三角剖分（骨架）。

对应 docs/navmesh-region-growth-design.md 完整管线：

    ① 可行走格点提取  →  ② BFS 区域生长  →  ③ shapely 轮廓
    →  ④ Ear Clipping 三角剖分  →  ⑤ 三角形邻接图 + portal 边  →  ⑥ 跨层垂直 portal

状态：**管线完整（2026-08-12）**。
- ① 可行走判定 + 提取、② BFS 区域生长（mat 硬约束）、③ shapely 轮廓、
  ④ 三角剖分（凸无洞扇形 + 凹/带洞 **shapely 约束 Delaunay**）、⑤ 共享边邻接
  + portal、⑥ 跨层垂直 portal（空间哈希 + 退化点 portal）全部实现
- 集成：movement/nav.py（navgraph_astar）+ movement/pathfinding.py（GraphPathfinder）
- 替代 surface_mesh.py 的 GreedyMesh 方案（docs/surface-mesh-unification.md 已归档）
- ⚠️ ④ 手写 ear clipping + 桥接已废弃（2026-08-12：fBm 多洞/点接触洞拓扑不可靠），
  改走 shapely CDT；旧代码保留供参考/回退

性能纪律：本模块全部在加载期执行一次，运行时只读；不引入 Numba 热路径。
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from Airia.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_BUILDING,
    MAT_TERRAIN,
    MAT_TUNNEL,
    MAX_SPANS,
    _MAT,
    _ZMAX,
    _ZMIN,
    Heightfield,
)

if TYPE_CHECKING:  # 避免 shapely 成为硬依赖（仅轮廓提取需要）
    from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)


# ===========================================================================
# 数据模型
# ===========================================================================


@dataclass
class WalkableConfig:
    """可行走格提取 / 区域生长 / 三角剖分参数（docs 附录 A 推荐默认值）。

    Attributes:
        max_slope_deg: 最大可行走坡度（度）。
        max_climb: 区域生长最大 z 差（m），相邻格 z 差超过则不合并。
        min_region_cells: 最小区域格数，孤立小区域被丢弃。
        gap_threshold: 顶面检测的间隙阈值（层数），上方 span 紧贴则非顶面。
        simplify_tolerance: 轮廓简化容差（m），None → cs × 0.5。
        portal_min_length: portal 边最小长度（m），None → cs × 0.5。
    """

    max_slope_deg: float = 30.0
    max_climb: float = 2.0
    min_region_cells: int = 8
    gap_threshold: int = 1
    simplify_tolerance: float | None = None
    portal_min_length: float | None = None

    def effective_simplify_tolerance(self, cs: float) -> float:
        """解析 simplify_tolerance：None → cs × 0.5。"""
        return cs * 0.5 if self.simplify_tolerance is None else float(self.simplify_tolerance)

    def effective_portal_min_length(self, cs: float) -> float:
        """解析 portal_min_length：None → cs × 0.5。"""
        return cs * 0.5 if self.portal_min_length is None else float(self.portal_min_length)


@dataclass
class WalkableCell:
    """单个可行走格点（docs §3.1.1）。

    同一 (gx, gy) 列上可能存在多个 WalkableCell——例如地面 (z=2m, TERRAIN)
    与上方桥面 (z=12m, BRIDGE)，各自独立参与区域生长。

    Attributes:
        gx, gy: 格坐标。
        z_layer: span 顶面层索引（uint16 编码，heightfield 层单位）。
        z_world: 世界高度 (m)。
        mat: 材质（MAT_TERRAIN / MAT_BRIDGE / MAT_BUILDING / MAT_TUNNEL）。
        col: 列索引 = gx + gy * width（冗余加速）。
    """

    gx: int
    gy: int
    z_layer: int
    z_world: float
    mat: int
    col: int


@dataclass
class NavTriangle:
    """三角形导航节点（docs §3.5.1）。

    三角形天然凸 → 面片内两点直线可达，是导航节点的理想单元。

    Attributes:
        tri_id: 唯一 ID（在 NavGraph 内索引）。
        vertices: (3, 2) float32 世界坐标 (x, y)。
        centroid: 形心 (x, y) 世界坐标。
        z_world: 代表高度 (m)，取区域格点均值。
        mat: 材质。
    """

    tri_id: int
    vertices: np.ndarray  # (3, 2) float32
    centroid: tuple[float, float]
    z_world: float
    mat: int


@dataclass
class NavGraph:
    """三角形导航图（docs §3.5.1 + §3.6）。

    Attributes:
        triangles: 所有三角形（按 tri_id 索引）。
        adjacency: 邻接表 {tri_id: [neighbor_id, ...]}，含跨层垂直 portal。
        portal_edges: {tri_id: [(p1, p2), ...]} 门户边（世界坐标，退化点 portal
            用 (p, p) 表示，见 docs §3.6.3）。
        vertical_portals: 跨层垂直 portal 列表 [(tri_a, tri_b), ...]，
            由 link_cross_layer 填充，并同时注入 adjacency。
    """

    triangles: list[NavTriangle] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    portal_edges: dict[
        int, list[tuple[tuple[float, float], tuple[float, float]]]
    ] = field(default_factory=dict)
    vertical_portals: list[tuple[int, int]] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        """三角形节点数。"""
        return len(self.triangles)

    def locate_triangle(self, x: float, y: float) -> int:
        """点定位：返回包含 (x, y) 的三角形 ID（模块级 locate_triangle 封装）。"""
        return locate_triangle(x, y, self.triangles)


# ===========================================================================
# ① 可行走格点提取（docs §3.1）
# ===========================================================================


def extract_walkable_cells(
    hf: Heightfield, cfg: WalkableConfig
) -> list[WalkableCell]:
    """遍历所有列的所有 span 顶面，提取可行走格点（numpy 向量化）。

    筛除条件（docs §3.1.2）：① 非暴露表面（上方有紧贴 span 覆盖）
    ② 坡度超标（max_slope_deg）③ 悬空（TERRAIN 段下方无支撑）。

    2026-08-12 性能优化：旧实现为 Python 双重循环（W×H×spans + 每格四邻域
    坡度），改为 numpy 向量化——顶面/悬空按 spans 数组批量判定；坡度用
    「每格每 mat 顶面 z 网格」四邻域差分（np.roll + nan 边界）。
    400×400 提取 679ms → ~60ms；输出与旧版逐格一致（测试已断言）。

    Args:
        hf: 高度场。
        cfg: 提取参数。

    Returns:
        可行走格点列表。同一列多个 span 顶面各自产出一个 cell。
    """
    spans, count = hf.spans, hf.count
    w, h = hf.width, hf.height
    n_cols = w * h
    gap = cfg.gap_threshold
    max_slope = cfg.max_slope_deg
    cols = np.arange(n_cols)

    # ① 顶面掩码（每列每 span）
    span_idx = np.arange(MAX_SPANS)[None, :]
    is_valid = count[:, None] > span_idx
    is_last = count[:, None] == (span_idx + 1)
    gap_ok = np.ones((n_cols, MAX_SPANS), dtype=bool)
    gap_ok[:, :-1] = (spans[:, 1:, _ZMIN] - spans[:, :-1, _ZMAX]) >= gap
    is_top = is_valid & (is_last | gap_ok)

    # ② 悬空掩码（TERRAIN si>0 且下方有间隙）
    mat_si = spans[:, :, _MAT].astype(np.int64)
    hanging = np.zeros((n_cols, MAX_SPANS), dtype=bool)
    for si in range(1, MAX_SPANS):
        hang = (mat_si[:, si] == MAT_TERRAIN) & (
            spans[:, si, _ZMIN].astype(np.int64)
            - spans[:, si - 1, _ZMAX].astype(np.int64)
        ) > gap
        hanging[:, si] = hang

    # ③ 坡度：每格每 mat 顶面 z 网格 → 四邻域最大 |dz|/cs
    z_world = hf.z_min + spans[:, :, _ZMAX] * hf.ch
    z_top = np.full((n_cols, 4), np.nan)
    with np.errstate(all="ignore"):
        for m in range(4):
            z_top[:, m] = np.where(mat_si == m, z_world, np.nan).max(axis=1)
    z_grid = z_top.reshape(h, w, 4)

    def _shift(z: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """四邻域移位（越界置 nan）。"""
        out = np.roll(z, (dy, dx), axis=(0, 1))
        if dy > 0:
            out[:dy, :, :] = np.nan
        elif dy < 0:
            out[dy:, :, :] = np.nan
        if dx > 0:
            out[:, :dx, :] = np.nan
        elif dx < 0:
            out[:, dx:, :] = np.nan
        return out

    with np.errstate(all="ignore"):
        diffs = [
            np.degrees(
                np.arctan(np.abs(z_grid - _shift(z_grid, dy, dx)) / hf.cs)
            )
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ]
        # where+max 替代 nanmax（全 nan 列 → -1 → 0，规避 All-NaN warning）
        stack = np.stack(diffs, axis=-1)
        maxd = np.where(np.isfinite(stack), stack, -1.0).max(axis=-1)
        grad_deg = np.where(maxd < 0.0, 0.0, maxd).reshape(n_cols, 4)
    slope_ok = grad_deg <= max_slope

    # 坡度按 span 的 mat 映射
    slope_mask = np.zeros((n_cols, MAX_SPANS), dtype=bool)
    for si in range(MAX_SPANS):
        slope_mask[:, si] = slope_ok[cols, mat_si[:, si]]
    candidate = is_top & ~hanging & slope_mask

    ys, si = np.nonzero(candidate)
    if ys.size == 0:
        return []
    gx = ys % w
    gy = ys // w
    z_layer = spans[ys, si, _ZMAX]
    mat_out = spans[ys, si, _MAT]
    zw = hf.z_min + z_layer * hf.ch
    return [
        WalkableCell(
            gx=int(gx[i]),
            gy=int(gy[i]),
            z_layer=int(z_layer[i]),
            z_world=float(zw[i]),
            mat=int(mat_out[i]),
            col=int(ys[i]),
        )
        for i in range(len(ys))
    ]


def is_top_surface(
    hf: Heightfield, col: int, si: int, gap_threshold: int = 1
) -> bool:
    """si 段是否为暴露顶面：上方无紧贴 span 覆盖（docs §3.1.3）。

    地面 si=0 + 上方 12m 桥 → 间隙大 → 地面和桥面都是顶面 ✅
    地面 si=0 + 上方紧贴建筑底层 → 间隙小 → 地面被覆盖，不可走 ✅
    """
    n = int(hf.count[col])
    if si == n - 1:
        return True  # 最后一层 → 一定是顶面
    next_zmin = int(hf.spans[col, si + 1, _ZMIN])
    this_zmax = int(hf.spans[col, si, _ZMAX])
    return (next_zmin - this_zmax) >= gap_threshold


def find_neighbor_z(
    hf: Heightfield, n_col: int, mat: int, z_center: float
) -> float | None:
    """在邻居列中查找与 z_center 最接近的同 mat span 顶面高度。

    Returns:
        邻居同 mat 顶面世界高度；无同 mat span 时返回 None。
    """
    best: float | None = None
    best_d = float("inf")
    for i in range(int(hf.count[n_col])):
        if int(hf.spans[n_col, i, _MAT]) != mat:
            continue
        z = hf._to_z(int(hf.spans[n_col, i, _ZMAX]))
        d = abs(z - z_center)
        if d < best_d:
            best_d = d
            best = z
    return best


def slope_at(hf: Heightfield, gx: int, gy: int, si: int) -> float:
    """局部坡度（度）：四邻域同 mat 顶面的最大高度梯度（docs §3.1.4）。"""
    col = hf._col(gx, gy)
    z_center = hf._to_z(int(hf.spans[col, si, _ZMAX]))
    mat = int(hf.spans[col, si, _MAT])

    max_grad = 0.0
    for nx, ny in ((gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)):
        if not (0 <= nx < hf.width and 0 <= ny < hf.height):
            continue
        n_z = find_neighbor_z(hf, hf._col(nx, ny), mat, z_center)
        if n_z is None:
            continue
        grad = abs(z_center - n_z) / hf.cs
        if grad > max_grad:
            max_grad = grad
    return math.degrees(math.atan(max_grad))


def is_hanging(
    hf: Heightfield, gx: int, gy: int, si: int, gap_threshold: int = 1
) -> bool:
    """面片是否悬空（docs §3.1.5）。

    人工结构（BRIDGE/BUILDING/TUNNEL）自身可站立 → 不悬空；
    TERRAIN 需检查 si>0 时是否紧贴下方 span。
    """
    col = hf._col(gx, gy)
    mat = int(hf.spans[col, si, _MAT])
    if mat != MAT_TERRAIN:
        return False
    if si == 0:
        return False  # 地面层 → 有支撑
    this_zmin = int(hf.spans[col, si, _ZMIN])
    prev_zmax = int(hf.spans[col, si - 1, _ZMAX])
    return (this_zmin - prev_zmax) > gap_threshold


# ===========================================================================
# ② 区域生长（docs §3.2）
# ===========================================================================


def grow_regions(
    cells: list[WalkableCell],
    max_climb: float,
    min_region_cells: int = 8,
) -> list[set[int]]:
    """BFS 区域生长：相邻格「逐对 z 比较 + mat 硬约束」合并。

    合并条件（docs §3.2.1）：
        - mat 相同（不同材质绝不合并）
        - |z_world 差| ≤ max_climb（逐对相对比较，非与种子绝对比较）

    Args:
        cells: 可行走格点列表（extract_walkable_cells 输出）。
        max_climb: 最大 z 差（m）。
        min_region_cells: 最小区域格数，小于此值的区域丢弃。

    Returns:
        区域列表，每个区域为 cell 索引集合（list[set[int]]）。
    """
    neighbors = build_neighbor_map(cells)
    visited: set[int] = set()
    regions: list[set[int]] = []

    for seed_idx in range(len(cells)):
        if seed_idx in visited:
            continue
        region: set[int] = set()
        queue: deque[int] = deque([seed_idx])
        visited.add(seed_idx)

        while queue:
            cur_idx = queue.popleft()
            region.add(cur_idx)
            cur = cells[cur_idx]
            for nb_idx in neighbors[cur_idx]:
                if nb_idx in visited:
                    continue
                nb = cells[nb_idx]
                if nb.mat != cur.mat:  # mat 硬约束
                    continue
                if abs(nb.z_world - cur.z_world) > max_climb:
                    continue
                visited.add(nb_idx)
                queue.append(nb_idx)

        if len(region) >= min_region_cells:
            regions.append(region)
    return regions


def build_neighbor_map(cells: list[WalkableCell]) -> list[list[int]]:
    """构建四邻域邻接表（docs §3.2.2）。

    以 (gx, gy, mat) 索引：同一格可能存在多个不同 mat 的 cell，仅同 mat 邻接。
    2026-08-12 性能优化：dict 逐个查询 → numpy cell_grid 批量邻居索引。
    """
    n = len(cells)
    if n == 0:
        return []
    gx = np.fromiter((c.gx for c in cells), dtype=np.int64, count=n)
    gy = np.fromiter((c.gy for c in cells), dtype=np.int64, count=n)
    mat = np.fromiter((c.mat for c in cells), dtype=np.int64, count=n)
    h = int(gy.max()) + 1
    w = int(gx.max()) + 1

    # (gy, gx, mat) → cell 索引（numpy 高级索引批量填充）
    cell_grid = np.full((h, w, 4), -1, dtype=np.int64)
    cell_grid[gy, gx, mat] = np.arange(n)

    neighbors: list[list[int]] = [[] for _ in range(n)]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny = gy + dy
        nx = gx + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nb = np.full(n, -1, dtype=np.int64)
        nb[valid] = cell_grid[ny[valid], nx[valid], mat[valid]]
        # 逐 cell 收集（仅有效邻居）
        for i in range(n):
            if nb[i] >= 0:
                neighbors[i].append(int(nb[i]))
    return neighbors


# ===========================================================================
# ③ 轮廓提取（docs §3.3）
# ===========================================================================


def region_to_polygon(
    cell_idxs: set[int] | list[int],
    cells: list[WalkableCell],
    hf: Heightfield,
    simplify_tolerance: float | None = None,
) -> "Polygon | MultiPolygon | None":
    """区域 → shapely 多边形（Marching Squares 轮廓追踪 + simplify）。

    2026-08-12 性能优化（第二次）：旧 unary_union 逐格 box 合并 O(面积)
    （400×400 需秒级），且「只对边界格建 box」的优化会产生伪洞（外圈 box
    并集是回字形，内部格无 box 填充）——改用 numpy **Marching Squares 轮廓
    追踪**（O(周长)）：提取每格暴露边 → 连环 → 外环 + 洞，毫秒级且正确。
    点接触的洞（仅共享角）在网格语义下边界相连 → 合并为单洞（对导航是
    更合理的障碍合并）。

    Args:
        cell_idxs: 区域内 cell 索引集合。
        cells: 全部可行走格点。
        hf: 高度场（取 cs 与容差）。
        simplify_tolerance: 简化容差（m），None → cs × 0.5。

    Returns:
        合并后的简单多边形（可能带洞）；区域为空返回 None。
    """
    cs = float(hf.cs)
    tol = cs * 0.5 if simplify_tolerance is None else float(simplify_tolerance)

    # 区域格掩码（numpy 高级索引一次填充）
    idx_list = list(cell_idxs)
    if not idx_list:
        return None
    gx_arr = np.fromiter(
        (cells[i].gx for i in idx_list), dtype=np.intp, count=len(idx_list)
    )
    gy_arr = np.fromiter(
        (cells[i].gy for i in idx_list), dtype=np.intp, count=len(idx_list)
    )
    mask = np.zeros((hf.height, hf.width), dtype=bool)
    mask[gy_arr, gx_arr] = True

    poly = _mask_to_polygon(mask, cs)
    if poly is None:
        return None
    poly = poly.simplify(tol, preserve_topology=True)
    if not poly.is_valid:
        # 点接触洞（仅共享角）在 Marching Squares 下边界相连 → 自相交环；
        # buffer(0) 修复为有效多边形（把接触洞正确拆开）。
        poly = poly.buffer(0)
    return poly


def _mask_to_polygon(
    mask: np.ndarray, cs: float
) -> "Polygon | None":
    """二值 mask → shapely Polygon（Marching Squares 轮廓追踪，含洞）。

    流程：① 提取每格暴露边（四邻域不在区域内则暴露，numpy 批量）；
    ② 边端点邻接表连环（O(周长)）；③ 面积最大环 = 外环，其余环 = 洞。

    Args:
        mask: (h, w) bool 可行走格掩码（区域 4 邻域连通，单外环）。
        cs: 单格世界尺寸（m/格）。

    Returns:
        简单多边形（可能带洞）；空掩码返回 None。
    """
    h, w = mask.shape
    mp = np.pad(mask, 1, constant_values=False)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    pi, pj = ys + 1, xs + 1  # pad 坐标（上方/下方/左方/右方用 pad 边界判暴露）

    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # 顶边（上方无区域）
    sel = ~mp[pi - 1, pj]
    if sel.any():
        edges.extend(
            ((int(xs[k]), int(ys[k])), (int(xs[k]) + 1, int(ys[k])))
            for k in np.nonzero(sel)[0]
        )
    # 底边
    sel = ~mp[pi + 1, pj]
    if sel.any():
        edges.extend(
            ((int(xs[k]), int(ys[k]) + 1), (int(xs[k]) + 1, int(ys[k]) + 1))
            for k in np.nonzero(sel)[0]
        )
    # 左边
    sel = ~mp[pi, pj - 1]
    if sel.any():
        edges.extend(
            ((int(xs[k]), int(ys[k])), (int(xs[k]), int(ys[k]) + 1))
            for k in np.nonzero(sel)[0]
        )
    # 右边
    sel = ~mp[pi, pj + 1]
    if sel.any():
        edges.extend(
            ((int(xs[k]) + 1, int(ys[k])), (int(xs[k]) + 1, int(ys[k]) + 1))
            for k in np.nonzero(sel)[0]
        )
    if not edges:
        return None

    # 端点邻接表 → 连环
    adj: dict[tuple[int, int], list[int]] = {}
    for eid, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append(eid)
        adj.setdefault(b, []).append(eid)
    visited = [False] * len(edges)
    rings: list[list[tuple[int, int]]] = []
    for eid in range(len(edges)):
        if visited[eid]:
            continue
        ring: list[tuple[int, int]] = []
        cur_pt = edges[eid][0]
        cur_eid = eid
        while True:
            visited[cur_eid] = True
            a, b = edges[cur_eid]
            nxt_pt = b if a == cur_pt else a
            ring.append(cur_pt)
            cur_pt = nxt_pt
            cands = [e for e in adj[cur_pt] if not visited[e]]
            if not cands:
                break
            cur_eid = cands[0]
        if len(ring) >= 3:
            rings.append(ring)
    if not rings:
        return None

    def _ring_area2(r: list[tuple[int, int]]) -> float:
        s = 0.0
        n = len(r)
        for i in range(n):
            x1, y1 = r[i]
            x2, y2 = r[(i + 1) % n]
            s += x1 * y2 - y1 * x2
        return abs(s) * 0.5

    # 面积最大环 = 外环；其余环 = 洞
    rings.sort(key=_ring_area2, reverse=True)
    shell = rings[0]
    holes = [_scale(r, cs) for r in rings[1:]]
    shell_pts = _scale(shell, cs)
    from shapely.geometry import Polygon

    return Polygon(shell_pts, holes=holes)


def _scale(
    ring: list[tuple[int, int]], cs: float
) -> list[tuple[float, float]]:
    """格点环 → 世界坐标环。"""
    return [(cs * x, cs * y) for x, y in ring]


def polygon_to_list(poly: "Polygon | MultiPolygon | None") -> list["Polygon"]:
    """统一接口：Polygon → [Polygon]；MultiPolygon → 拆为列表（docs §3.3 处理）。"""
    if poly is None:
        return []
    if poly.geom_type == "MultiPolygon":
        return [g for g in poly.geoms if not g.is_empty]
    if poly.geom_type == "Polygon":
        return [poly] if not poly.is_empty else []
    return []


# ===========================================================================
# ④ 三角剖分（docs §3.4）
#    凸无洞 → 扇形快速路径；凹/带洞 → shapely 约束 Delaunay（CDT）。
#    手写 ear clipping + 桥接（下方 _ear_clip/_bridge_hole/_diagonal 等）已废弃
#    （2026-08-12：对 fBm 多洞/点接触洞拓扑不可靠），保留供参考/回退。
# ===========================================================================


def triangulate_polygon(
    polygon: "Polygon",
) -> list[
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
]:
    """多边形三角剖分（docs §3.4）：凸无洞 → 扇形；凹/带洞 → shapely CDT。

    2026-08-12 修复：fBm 真实地形的区域轮廓带**大量洞**（内部山峰/坑洼），
    且洞可**点接触**（相邻山峰对角相碰，共享一个顶点）。手写 ear clipping
    + 桥接（Recast mergeContours）对复杂多洞拓扑不可靠（桥接失败/三角化卡死）
    → 凹/带洞路径改走 shapely **约束 Delaunay 三角剖分**
    （constrained_delaunay_triangles）：洞边界为约束边，三角形天然不覆盖洞、
    全凸、处理任意拓扑（多洞/点接触/凹洞）。

    Args:
        polygon: shapely Polygon。

    Returns:
        三角形列表，每个三角形为 3 个世界坐标点 ((x, y), (x, y), (x, y))。
        零面积退化三角形已过滤。
    """
    outer = list(polygon.exterior.coords)[:-1]  # 首尾去重
    holes = [list(h.coords)[:-1] for h in polygon.interiors]

    # 凸快速路径（无洞）：扇形三角化（顶点序保持 shapely 外环 CCW）
    if not holes and is_convex_polygon(outer):
        n = len(outer)
        return [
            (tuple(outer[0]), tuple(outer[i + 1]), tuple(outer[i + 2]))
            for i in range(n - 2)
        ]

    # 凹/带洞：shapely 约束 Delaunay（健壮，处理多洞/点接触洞/凹洞）
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import GeometryCollection

    result: list[
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ] = []
    tri_geoms = constrained_delaunay_triangles(polygon)
    if isinstance(tri_geoms, GeometryCollection):
        geoms = tri_geoms.geoms
    else:
        geoms = [tri_geoms]
    for t in geoms:
        coords = list(t.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        a = (float(coords[0][0]), float(coords[0][1]))
        b = (float(coords[1][0]), float(coords[1][1]))
        c = (float(coords[2][0]), float(coords[2][1]))
        if _triangle_area2(a, b, c) < 1e-9:
            continue
        result.append((a, b, c))
    return result


def is_convex_polygon(verts: list[tuple[float, float]]) -> bool:
    """多边形是否严格凸（所有顶点叉积同号）。"""
    n = len(verts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        cx, cy = verts[(i + 2) % n]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross != 0.0:
            s = 1.0 if cross > 0.0 else -1.0
            if sign == 0.0:
                sign = s
            elif s != sign:
                return False
    return True


# 凹多边形 Ear Clipping（docs §3.4）—— 已废弃（2026-08-12）：
# fBm 多洞/点接触洞拓扑下桥接与 ear clipping 不可靠，已被 shapely 约束
# Delaunay（triangulate_polygon 内）替代。以下为 Recast RecastMesh.cpp 的
# diagonal / inCone / diagonalie 判定移植，保留供参考/回退。

_EPS = 1e-9


# ---- 基础几何判定（RecastMesh.cpp：area2/left/leftOn/collinear/intersectProp/between/intersect/vequal）----


def _area2(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """叉积 (b-a) × (c-a) 的 z 分量（Recast area2）。"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _left(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    return _area2(a, b, c) < 0.0


def _left_on(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    return _area2(a, b, c) <= _EPS


def _collinear(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    return abs(_area2(a, b, c)) <= _EPS


def _intersect_prop(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """ab 与 cd 是否 proper 相交（交叉穿越，排除共线/端点）。"""
    if (
        _collinear(a, b, c)
        or _collinear(a, b, d)
        or _collinear(c, d, a)
        or _collinear(c, d, b)
    ):
        return False
    return (_left(a, b, c) != _left(a, b, d)) and (
        _left(c, d, a) != _left(c, d, b)
    )


def _between(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """(a,b,c) 共线且 c 在闭线段 ab 上（Recast between）。"""
    if not _collinear(a, b, c):
        return False
    if abs(a[0] - b[0]) > _EPS:
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
    return min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """ab 与 cd 是否相交（proper 或 improper，Recast intersect）。"""
    if _intersect_prop(a, b, c, d):
        return True
    return (
        _between(a, b, c)
        or _between(a, b, d)
        or _between(c, d, a)
        or _between(c, d, b)
    )


def _vequal(
    a: tuple[float, float], b: tuple[float, float]
) -> bool:
    return abs(a[0] - b[0]) <= _EPS and abs(a[1] - b[1]) <= _EPS


# ---- 对角线判定（RecastMesh.cpp：diagonalie/inCone/diagonal + loose 变体）----


def _diagonalie(
    i: int,
    j: int,
    indices: list[int],
    vertices: list[tuple[float, float]],
) -> bool:
    """对角线 (i,j)（位置索引）不穿过任何边（跳过 i/j 邻边与共享端点边）。"""
    d0 = vertices[indices[i]]
    d1 = vertices[indices[j]]
    n = len(indices)
    for k in range(n):
        k1 = (k + 1) % n
        if k == i or k1 == i or k == j or k1 == j:
            continue
        p0 = vertices[indices[k]]
        p1 = vertices[indices[k1]]
        if _vequal(d0, p0) or _vequal(d1, p0) or _vequal(d0, p1) or _vequal(d1, p1):
            continue
        if _intersect(d0, d1, p0, p1):
            return False
    return True


def _in_cone(
    i: int,
    j: int,
    indices: list[int],
    vertices: list[tuple[float, float]],
) -> bool:
    """对角线 (i,j) 在 i 端点锥形内（凸/reflex 顶点分别处理，Recast inCone）。"""
    n = len(indices)
    pi = vertices[indices[i]]
    pj = vertices[indices[j]]
    pi1 = vertices[indices[(i + 1) % n]]
    pin1 = vertices[indices[(i - 1) % n]]
    if _left_on(pin1, pi, pi1):  # pi 是凸顶点（Recast CW 约定）
        return _left(pi, pj, pin1) and _left(pj, pi, pi1)
    return not (_left_on(pi, pj, pi1) and _left_on(pj, pi, pin1))


def _diagonal(
    i: int,
    j: int,
    indices: list[int],
    vertices: list[tuple[float, float]],
) -> bool:
    """对角线 (i,j) 是否是多边形内对角线（Recast diagonal）。"""
    return _in_cone(i, j, indices, vertices) and _diagonalie(
        i, j, indices, vertices
    )


def _diagonalie_loose(
    i: int,
    j: int,
    indices: list[int],
    vertices: list[tuple[float, float]],
) -> bool:
    """loose：仅排除 proper 相交（容忍重叠边，退化恢复用）。"""
    d0 = vertices[indices[i]]
    d1 = vertices[indices[j]]
    n = len(indices)
    for k in range(n):
        k1 = (k + 1) % n
        if k == i or k1 == i or k == j or k1 == j:
            continue
        p0 = vertices[indices[k]]
        p1 = vertices[indices[k1]]
        if _vequal(d0, p0) or _vequal(d1, p0) or _vequal(d0, p1) or _vequal(d1, p1):
            continue
        if _intersect_prop(d0, d1, p0, p1):
            return False
    return True


def _in_cone_loose(
    i: int,
    j: int,
    indices: list[int],
    vertices: list[tuple[float, float]],
) -> bool:
    n = len(indices)
    pi = vertices[indices[i]]
    pj = vertices[indices[j]]
    pi1 = vertices[indices[(i + 1) % n]]
    pin1 = vertices[indices[(i - 1) % n]]
    if _left_on(pin1, pi, pi1):
        return _left_on(pi, pj, pin1) and _left_on(pj, pi, pi1)
    return not (_left_on(pi, pj, pi1) and _left_on(pj, pi, pin1))


def _diagonal_loose(
    i: int,
    j: int,
    indices: list[int],
    vertices: list[tuple[float, float]],
) -> bool:
    return _in_cone_loose(i, j, indices, vertices) and _diagonalie_loose(
        i, j, indices, vertices
    )


def _triangle_area2(
    p: tuple[float, float],
    q: tuple[float, float],
    r: tuple[float, float],
) -> float:
    """三角形有向面积 ×2（绝对值）。"""
    return abs(
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    )


# ---- ear clip 主体（RecastMesh.cpp：triangulate，最短耳优先 + loose 恢复）----


def _ear_clip(
    vertices: list[tuple[float, float]],
) -> list[tuple[int, int, int]]:
    """移植 Recast triangulate：对角线下三角剖分，最短耳优先裁剪。

    indices 存原始顶点序号；diagonal 判定 = inCone + diagonalie。
    桥接重复顶点（Recast mergeContours 产物）由 vequal 跳过容忍。

    Raises:
        RuntimeError: 多边形自相交/退化，无法三角剖分（不应发生，保险丝）。
    """
    n = len(vertices)
    if n < 3:
        return []
    indices = list(range(n))
    removable = [False] * n

    for i in range(n):
        i1 = (i + 1) % n
        i2 = (i1 + 1) % n
        if _diagonal(i, i2, indices, vertices):
            removable[i1] = True

    tris: list[tuple[int, int, int]] = []
    guard = 0
    max_guard = n * n + 16

    while n > 3:
        guard += 1
        if guard > max_guard:
            raise RuntimeError(
                f"ear clipping 未收敛（顶点 {len(vertices)}，剩余 {n}）"
            )

        # 选最短对角线的 ear（Recast 策略，减少退化）
        min_len = -1.0
        mini = -1
        for i in range(n):
            i1 = (i + 1) % n
            if removable[i1]:
                p0 = vertices[indices[i]]
                p2 = vertices[indices[(i1 + 1) % n]]
                ln = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
                if min_len < 0.0 or ln < min_len:
                    min_len = ln
                    mini = i

        if mini == -1:
            # loose 恢复（Recast：重叠段轮廓，放松 inCone/diagonalie）
            min_len = -1.0
            mini = -1
            for i in range(n):
                i1 = (i + 1) % n
                i2 = (i1 + 1) % n
                if _diagonal_loose(i, i2, indices, vertices):
                    p0 = vertices[indices[i]]
                    p2 = vertices[indices[(i2 + 1) % n]]
                    ln = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
                    if min_len < 0.0 or ln < min_len:
                        min_len = ln
                        mini = i
            if mini == -1:
                raise RuntimeError(
                    f"多边形自相交/退化，无法三角剖分（剩余 {n}）"
                )

        i = mini
        i1 = (i + 1) % n
        i2 = (i1 + 1) % n
        tris.append((indices[i], indices[i1], indices[i2]))

        # 移除 indices[i1]（左移）
        n -= 1
        for k in range(i1, n):
            indices[k] = indices[k + 1]
            removable[k] = removable[k + 1]
        if i1 >= n:
            i1 = 0
        i = (i1 - 1) % n
        # 更新相邻位置的 ear 标志（Recast：diagonal(prev(i),i1) / diagonal(i,next(i1))）
        removable[i] = _diagonal((i - 1) % n, i1, indices, vertices)
        removable[i1] = _diagonal(i, (i1 + 1) % n, indices, vertices)

    if n == 3:
        tris.append((indices[0], indices[1], indices[2]))
    return tris


# ---- 洞桥接（Recast mergeContours 思路，navi.md §空洞合并）----


def _polygon_signed_area(
    pts: list[tuple[float, float]],
) -> float:
    """多边形有向面积（shoelace）：>0 逆时针 / <0 顺时针。"""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - y1 * x2
    return 0.5 * s


def _leftmost_index(
    hole: list[tuple[float, float]],
) -> int:
    """洞最左顶点（min x，平局取 min y——Recast findLeftMostVertex）。"""
    return min(range(len(hole)), key=lambda i: (hole[i][0], hole[i][1]))


def _bridge_hole(
    outer: list[tuple[float, float]],
    hole: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """把洞缝合到外环上（docs §3.4.3，Recast mergeContours 同款序列）。

    方向约定：外环 CW、洞 CCW（调用方翻转）。流程：
    ① 找洞最左顶点 H；② 外环上找 V→H 最短不相交对角线（Recast intersect
    判定，含 improper）；③ 缝合 outer[:v_idx+1] + hole[h_idx:] + hole[:h_idx+1]
    + outer[v_idx:]（V 与 H 各出现两次——Recast 同款，ear clip 的 vequal 容忍）。
    """
    h_idx = _leftmost_index(hole)
    h_pt = hole[h_idx]
    v_idx = _find_visible_vertex(outer, h_pt, hole)
    result = outer[: v_idx + 1]
    result.extend(hole[h_idx:] + hole[: h_idx + 1])
    result.extend(outer[v_idx:])
    return result


def _find_visible_vertex(
    outer: list[tuple[float, float]],
    h_pt: tuple[float, float],
    hole: list[tuple[float, float]],
) -> int:
    """外环上对洞最左点 H 可见的顶点 V（V→H 不穿越任何边界 + 最短）。"""
    hx, hy = h_pt
    best: int | None = None
    best_d2 = float("inf")
    for vi, v in enumerate(outer):
        if _segment_clear(v, h_pt, outer, hole):
            d2 = (v[0] - hx) ** 2 + (v[1] - hy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = vi
    if best is None:
        raise ValueError("找不到可见桥接顶点（洞与外环相交或退化）")
    return best


def _segment_clear(
    p: tuple[float, float],
    q: tuple[float, float],
    outer: list[tuple[float, float]],
    hole: list[tuple[float, float]],
) -> bool:
    """线段 pq 不穿越任何边界（跳过与 p/q 共享端点的边）。

    用 _vequal 判定端点共享（而非按邻边位置索引）——缝合后外环可能含
    重复顶点（前一个洞的桥），且多洞可**点接触**（共享顶点，如相邻山峰
    对角相碰），此时 q 可能已存在于 outer 中；共享端点的边必须全部跳过。
    """
    n_out = len(outer)
    for k in range(n_out):
        a, b = outer[k], outer[(k + 1) % n_out]
        if _vequal(p, a) or _vequal(p, b) or _vequal(q, a) or _vequal(q, b):
            continue
        if _intersect(p, q, a, b):
            return False
    n_hole = len(hole)
    for k in range(n_hole):
        a, b = hole[k], hole[(k + 1) % n_hole]
        if _vequal(q, a) or _vequal(q, b):
            continue
        if _intersect(p, q, a, b):
            return False
    return True


# ===========================================================================
# ⑤ 邻接图 + portal 边（docs §3.5）
# ===========================================================================


def build_navgraph(
    triangles: list[NavTriangle],
    tol: float = 0.1,
) -> NavGraph:
    """构建三角形邻接图 + portal 边（docs §3.5.2）。

    两三角形邻接 ⇔ 共享一条边（顶点量化后相等，容差 tol）。

    Args:
        triangles: 三角形列表（tri_id 按列表序）。
        tol: 顶点量化容差（m），默认 0.1 ≈ cs × 0.1 量级。

    Returns:
        NavGraph（vertical_portals 初始为空，由 link_cross_layer 填充）。
    """
    edge_to_tri: dict[tuple, list[int]] = {}
    for tid, tri in enumerate(triangles):
        verts = tri.vertices
        for i in range(3):
            a = (float(verts[i][0]), float(verts[i][1]))
            b = (float(verts[(i + 1) % 3][0]), float(verts[(i + 1) % 3][1]))
            key = canonical_edge(a, b, tol)
            edge_to_tri.setdefault(key, []).append(tid)

    adjacency: dict[int, list[int]] = {}
    portal_edges: dict[
        int, list[tuple[tuple[float, float], tuple[float, float]]]
    ] = {}
    for e, tids in edge_to_tri.items():
        if len(tids) == 2:
            a, b = tids[0], tids[1]
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
            portal_edges.setdefault(a, []).append(e)
            portal_edges.setdefault(b, []).append((e[1], e[0]))  # 反向

    return NavGraph(
        triangles=triangles,
        adjacency=adjacency,
        portal_edges=portal_edges,
        vertical_portals=[],
    )


def canonical_edge(
    a: tuple[float, float],
    b: tuple[float, float],
    tol: float = 0.1,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """规范化边 key：端点排序 + 量化到 tol 网格（整数 key，浮点安全）。

    tol 为量化网格尺寸；两端点交换后 key 不变。
    """
    qa = (int(round(a[0] / tol)), int(round(a[1] / tol)))
    qb = (int(round(b[0] / tol)), int(round(b[1] / tol)))
    return (qa, qb) if qa <= qb else (qb, qa)


def locate_triangle(
    x: float, y: float, triangles: list[NavTriangle]
) -> int:
    """点定位：返回包含 (x, y) 的三角形 ID（docs §3.5.3）。

    精确判定（2D 同侧法）失败时退回最近形心三角形。
    """
    for tri in triangles:
        if point_in_triangle(x, y, tri.vertices):
            return tri.tri_id
    return min(
        triangles,
        key=lambda t: (t.centroid[0] - x) ** 2 + (t.centroid[1] - y) ** 2,
    ).tri_id


def point_in_triangle(
    px: float, py: float, verts: np.ndarray
) -> bool:
    """2D 点在三角形内（同侧法，含边界）。"""
    (x0, y0), (x1, y1), (x2, y2) = verts

    def _cross(
        ax: float, ay: float, bx: float, by: float, cx: float, cy: float
    ) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    d1 = _cross(px, py, x0, y0, x1, y1)
    d2 = _cross(px, py, x1, y1, x2, y2)
    d3 = _cross(px, py, x2, y2, x0, y0)
    has_neg = (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
    has_pos = (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)
    return not (has_neg and has_pos)


# ===========================================================================
# ⑥ 跨层链接（docs §3.6）
# ===========================================================================


def link_cross_layer(
    triangles: list[NavTriangle],
    max_climb: float,
) -> list[tuple[int, int]]:
    """跨 mat 垂直 portal 检测（docs §3.6.2）。

    判据：不同 mat、XY 投影重叠、|z 差| ≤ max_climb → 垂直 portal（点连接）。
    用 bbox 空间哈希加速（cell_size = 50.0），避免全对检测。

    Returns:
        垂直 portal 列表 [(tri_a, tri_b), ...]（tri_a ∈ 低层 mat，tri_b ∈ 高层 mat）。
    """
    pairs = [
        (MAT_TERRAIN, MAT_BRIDGE),
        (MAT_TERRAIN, MAT_BUILDING),
        (MAT_BRIDGE, MAT_TUNNEL),
    ]
    by_mat: dict[int, list[NavTriangle]] = {}
    for t in triangles:
        by_mat.setdefault(t.mat, []).append(t)

    vertical: list[tuple[int, int]] = []
    for mat_a, mat_b in pairs:
        tris_a = by_mat.get(mat_a, [])
        tris_b = by_mat.get(mat_b, [])
        if not tris_a or not tris_b:
            continue
        buckets = _build_bbox_hash(tris_a)
        for tb in tris_b:
            x0, y0, x1, y1 = _tri_bbox(tb)
            seen: set[int] = set()  # 防多桶重复
            for ta in _hash_query(buckets, x0, y0, x1, y1):
                if ta.tri_id in seen:
                    continue
                seen.add(ta.tri_id)
                if ta.tri_id == tb.tri_id:
                    continue
                if abs(ta.z_world - tb.z_world) > max_climb:
                    continue
                if not xy_overlap(ta, tb):
                    continue
                vertical.append((ta.tri_id, tb.tri_id))
    return vertical


def xy_overlap(ta: NavTriangle, tb: NavTriangle) -> bool:
    """两三角形 XY 投影是否相交（docs §3.6.2 判据之一）。

    任一顶点在对方内部（含边界），或任一边相交（含端点/共线）。
    """
    va, vb = ta.vertices, tb.vertices
    for p in va:
        if point_in_triangle(p[0], p[1], vb):
            return True
    for p in vb:
        if point_in_triangle(p[0], p[1], va):
            return True
    for i in range(3):
        a1, a2 = va[i], va[(i + 1) % 3]
        for j in range(3):
            b1, b2 = vb[j], vb[(j + 1) % 3]
            if _segments_intersect_inclusive(a1, a2, b1, b2):
                return True
    return False


def _segments_intersect_inclusive(
    p1: np.ndarray | tuple[float, float],
    p2: np.ndarray | tuple[float, float],
    p3: np.ndarray | tuple[float, float],
    p4: np.ndarray | tuple[float, float],
) -> bool:
    """两线段是否相交（含端点接触与共线重叠，Recast intersect 语义）。"""
    a1 = (float(p1[0]), float(p1[1]))
    a2 = (float(p2[0]), float(p2[1]))
    b1 = (float(p3[0]), float(p3[1]))
    b2 = (float(p4[0]), float(p4[1]))
    return _intersect(a1, a2, b1, b2)


def _tri_bbox(t: NavTriangle) -> tuple[float, float, float, float]:
    """三角形 bbox (x0, y0, x1, y1)。"""
    xs = t.vertices[:, 0]
    ys = t.vertices[:, 1]
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    )


def _build_bbox_hash(
    tris: list[NavTriangle], cell_size: float = 50.0
) -> dict[tuple[int, int], list[NavTriangle]]:
    """按三角形 bbox 覆盖格分桶（无状态，整数格 key）。"""
    buckets: dict[tuple[int, int], list[NavTriangle]] = {}
    for t in tris:
        x0, y0, x1, y1 = _tri_bbox(t)
        for cx in range(int(x0 // cell_size), int(x1 // cell_size) + 1):
            for cy in range(int(y0 // cell_size), int(y1 // cell_size) + 1):
                buckets.setdefault((cx, cy), []).append(t)
    return buckets


def _hash_query(
    buckets: dict[tuple[int, int], list[NavTriangle]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    cell_size: float = 50.0,
) -> list[NavTriangle]:
    """查询 bbox 覆盖格内的候选三角形。"""
    out: list[NavTriangle] = []
    for cx in range(int(x0 // cell_size), int(x1 // cell_size) + 1):
        for cy in range(int(y0 // cell_size), int(y1 // cell_size) + 1):
            out.extend(buckets.get((cx, cy), []))
    return out


def _inject_vertical_portal(graph: NavGraph, a: int, b: int) -> None:
    """垂直 portal 注入邻接表 + 退化点 portal 边（docs §3.6.3）。

    退化点 portal = ((pt, pt)，funnel 沿其收紧时强制经过该点。
    """
    ta, tb = graph.triangles[a], graph.triangles[b]
    graph.adjacency.setdefault(a, []).append(b)
    graph.adjacency.setdefault(b, []).append(a)
    pt = (
        (ta.centroid[0] + tb.centroid[0]) * 0.5,
        (ta.centroid[1] + tb.centroid[1]) * 0.5,
    )
    graph.portal_edges.setdefault(a, []).append((pt, pt))
    graph.portal_edges.setdefault(b, []).append((pt, pt))


# ===========================================================================
# 管线编排
# ===========================================================================


class NavMeshBuilder:
    """从 Heightfield 构建三角形 NavGraph（加载期执行一次）。

    管线编排（docs 第二节）：extract_walkable_cells → grow_regions →
    region_to_polygon → triangulate_polygon → build_navgraph → link_cross_layer。
    各步为模块级纯函数，可独立测试；本类只做编排与参数解析。
    """

    def __init__(
        self,
        hf: Heightfield,
        cfg: WalkableConfig | None = None,
    ) -> None:
        self.hf = hf
        self.cfg = cfg if cfg is not None else WalkableConfig()
        self.cs = float(hf.cs)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def build(self) -> NavGraph:
        """全量构建 NavGraph（6 步管线）。

        骨架降级策略：跨层垂直 portal 与凹/带洞三角剖分未实现时跳过并
        记录 warning，保证凸区域地图端到端可用；待实现后自动生效。
        """
        # ① 可行走格提取
        cells = extract_walkable_cells(self.hf, self.cfg)
        logger.debug("① 可行走格点: %d", len(cells))

        # ② 区域生长
        regions = grow_regions(
            cells, self.cfg.max_climb, self.cfg.min_region_cells
        )
        logger.debug("② 区域: %d", len(regions))

        # ③④ 区域 → 多边形 → 三角形
        triangles, skipped = self._triangulate_regions(cells, regions)
        logger.debug("④ 三角形: %d（跳过 %d 个区域）", len(triangles), skipped)

        # ⑤ 邻接图 + portal 边
        graph = build_navgraph(triangles, tol=max(self.cs * 0.1, 1e-3))

        # ⑥ 跨层垂直 portal（注入邻接表 + 退化点 portal 边）
        graph.vertical_portals = link_cross_layer(
            triangles, self.cfg.max_climb
        )
        for a, b in graph.vertical_portals:
            _inject_vertical_portal(graph, a, b)
        if graph.vertical_portals:
            logger.debug("⑥ 垂直 portal: %d", len(graph.vertical_portals))

        if skipped:
            logger.warning("%d 个区域因三角剖分失败被跳过", skipped)
        return graph

    # ------------------------------------------------------------------
    # 内部编排
    # ------------------------------------------------------------------

    def _triangulate_regions(
        self,
        cells: list[WalkableCell],
        regions: list[set[int]],
    ) -> tuple[list[NavTriangle], int]:
        """③ 区域 → shapely 多边形；④ 三角剖分 → NavTriangle 列表。

        三角形 z_world 取区域格点均值；mat 取区域首个 cell 材质。
        凹/带洞多边形（ear clipping 未实现）→ 跳过该区域并计数。

        Returns:
            (triangles, skipped_region_count)。
        """
        triangles: list[NavTriangle] = []
        skipped = 0
        tol = self.cfg.effective_simplify_tolerance(self.cs)

        for region in regions:
            region_cells = [cells[i] for i in region]
            mat = region_cells[0].mat
            z_avg = float(np.mean([c.z_world for c in region_cells]))

            poly = region_to_polygon(region, cells, self.hf, tol)
            try:
                for p in polygon_to_list(poly):
                    for tri in triangulate_polygon(p):
                        verts = np.array(tri, dtype=np.float32)
                        cx = float(verts[:, 0].mean())
                        cy = float(verts[:, 1].mean())
                        triangles.append(
                            NavTriangle(
                                tri_id=len(triangles),
                                vertices=verts,
                                centroid=(cx, cy),
                                z_world=z_avg,
                                mat=mat,
                            )
                        )
            except (NotImplementedError, ValueError, RuntimeError) as exc:
                skipped += 1
                logger.warning(
                    "区域三角剖分跳过（%s）：mat=%d 格数=%d",
                    exc,
                    mat,
                    len(region),
                )
        return triangles, skipped


__all__ = [
    "WalkableConfig",
    "WalkableCell",
    "NavTriangle",
    "NavGraph",
    "NavMeshBuilder",
    "extract_walkable_cells",
    "is_top_surface",
    "is_hanging",
    "slope_at",
    "find_neighbor_z",
    "grow_regions",
    "build_neighbor_map",
    "region_to_polygon",
    "polygon_to_list",
    "triangulate_polygon",
    "is_convex_polygon",
    "build_navgraph",
    "canonical_edge",
    "locate_triangle",
    "point_in_triangle",
    "link_cross_layer",
    "xy_overlap",
    # 材质常量 re-export（方便消费方引用，避免重复导入 heightfield）
    "MAT_TERRAIN",
    "MAT_BRIDGE",
    "MAT_BUILDING",
    "MAT_TUNNEL",
]
