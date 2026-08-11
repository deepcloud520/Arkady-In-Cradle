"""表面网格统一架构：GreedyMesh 面片提取 + 邻接图（三位一体：渲染/导航/碰撞）。

对应 docs/surface-mesh-unification.md 第二章：从 Heightfield 所有 span 顶面
按 (world_z, mat) 分组做 2D greedy mesh 合并，生成四边形面片 SurfacePatch。
面片本身即导航节点（凸四边形或任意多边形），共享边即 portal。

设计决策（2026-08-11）：
- GreedyMesh 价值在导航/统一，不在顶点数（结构层仅占 <1%）
- 地面层不合并（每个格不同 z），结构层（桥/建筑）大幅合并
- 面片可能是凹多边形（L/T 形建筑）→ 导航需 SSFA funnel 而非形心连线
- 提取在加载期执行，运行时查询面片图
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np

from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_BUILDING,
    MAT_TERRAIN,
    MAT_TUNNEL,
    MAX_SPANS,
    NO_SURFACE,
    _AREA,
    _MAT,
    _ZMAX,
    _ZMIN,
    Heightfield,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SurfacePatch —— 单个表面面片
# ---------------------------------------------------------------------------


@dataclass
class SurfacePatch:
    """单个表面面片（GreedyMesh 合并后的矩形/多边形）。

    Attributes:
        patch_id: 唯一 ID（在 SurfaceMesh 内索引）。
        vertices: (V, 2) float32 世界坐标 (x, y)，逆时针闭环（首尾不重复）。
        z: 世界高度（同平面所有顶点 z 相同）。
        mat: 材质（MAT_TERRAIN / MAT_BRIDGE / MAT_BUILDING / MAT_TUNNEL）。
        walkable: 是否可行走。
        area: 面积 (m²)。
        centroid: 形心 (x, y) 世界坐标。
    """

    patch_id: int
    vertices: np.ndarray  # (V, 2) float32
    z: float
    mat: int
    walkable: bool = False
    area: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.vertices) >= 3:
            self.area = float(_polygon_area(self.vertices))
            self.centroid = (
                float(self.vertices[:, 0].mean()),
                float(self.vertices[:, 1].mean()),
            )

    @property
    def normal(self) -> tuple[float, float, float]:
        """顶面法线（始终向上，水平面片）。"""
        return (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# SurfaceMesh —— 面片集合 + 邻接图
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMesh:
    """三位一体表面网格：所有 SurfacePatch + 邻接图 + portal 边。

    面片 = 导航节点；共享边 = portal；邻接图 = A* 搜索空间。

    Attributes:
        patches: 所有面片列表（按 patch_id 索引）。
        adjacency: 邻接表 {patch_id: [neighbor_patch_id, ...]}。
        portal_edges: {patch_id: [(p1, p2), ...]} 门户边（世界坐标）。
        walkable_patches: 可行走面片 ID 集合（缓存）。
    """

    patches: list[SurfacePatch] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    portal_edges: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = (
        field(default_factory=dict)
    )
    walkable_patches: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.patches and not self.adjacency:
            self._build_adjacency()

    # ------------------------------------------------------------------
    # 邻接图构建
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> None:
        """构建面片邻接图：共享边 + z 落差 < max_climb。

        两 patch 邻接条件：
        1. 存在共享边（两个顶点都在对方边界上）
        2. z 落差 < max_climb（默认 2m）

        实现：**扫描线算法**（O(n log n)）。
        面片为轴对齐矩形 → 共享边 = 同一直线上的重叠边。
        - 垂直边 (x, y0, y1) 按 x 分组，组内检查 y 区间重叠
        - 水平边 (y, x0, x1) 按 y 分组，组内检查 x 区间重叠
        避免对全部面片对做 O(n²) 检测（地面每格一个面片，n 可达数千）。
        """
        self.adjacency = {}
        self.portal_edges = {}
        self.walkable_patches = {p.patch_id for p in self.patches if p.walkable}

        max_climb = 2.0
        eps = 1e-4

        # 收集矩形边界（轴对齐矩形）
        v_edges: list[tuple[float, float, float, int]] = []  # (x, y0, y1, pid)
        h_edges: list[tuple[float, float, float, int]] = []  # (y, x0, x1, pid)
        for p in self.patches:
            xs = p.vertices[:, 0]
            ys = p.vertices[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            v_edges.append((x0, y0, y1, p.patch_id))
            v_edges.append((x1, y0, y1, p.patch_id))
            h_edges.append((y0, x0, x1, p.patch_id))
            h_edges.append((y1, x0, x1, p.patch_id))

        def _register(
            a: int,
            b: int,
            p1: tuple[float, float],
            p2: tuple[float, float],
        ) -> None:
            pa = self.patches[a]
            pb = self.patches[b]
            if abs(pa.z - pb.z) > max_climb:
                return
            self.adjacency.setdefault(a, []).append(b)
            self.adjacency.setdefault(b, []).append(a)
            self.portal_edges.setdefault(a, []).append((p1, p2))
            self.portal_edges.setdefault(b, []).append((p1, p2))

        def _scan(
            edges: list[tuple[float, float, float, int]],
            is_vertical: bool,
        ) -> None:
            """扫描同线边：区间重叠 → 共享边。

            is_vertical: True 处理垂直边 (x 常数, y 区间)；False 水平边。
            """
            edges.sort(key=lambda e: (e[0], e[1], e[2]))
            i = 0
            n = len(edges)
            while i < n:
                coord = edges[i][0]
                j = i
                while j < n and edges[j][0] == coord:
                    j += 1
                grp = edges[i:j]
                # 组内按区间起点排序，检查两两重叠（贪心：区间重叠且非同一 patch）
                for a in range(len(grp)):
                    for b in range(a + 1, len(grp)):
                        ea, eb = grp[a], grp[b]
                        # 区间重叠判断（有公共长度）
                        if eb[1] >= ea[2] - eps or eb[2] <= ea[1] + eps:
                            continue
                        if ea[3] == eb[3]:
                            continue  # 同一面片的两条边
                        lo = max(ea[1], eb[1])
                        hi = min(ea[2], eb[2])
                        if is_vertical:
                            _register(ea[3], eb[3], (coord, lo), (coord, hi))
                        else:
                            _register(ea[3], eb[3], (lo, coord), (hi, coord))
                i = j

        _scan(v_edges, is_vertical=True)
        _scan(h_edges, is_vertical=False)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def locate_patch(self, x: float, y: float) -> int:
        """点定位：返回包含 (x,y) 的面片 ID，不在任何面片内返回 -1。

        使用质心距离粗筛 + 射线法精确判定。
        """
        best = -1
        best_dist = float("inf")
        # 粗筛：找最近的形心
        for p in self.patches:
            cx, cy = p.centroid
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 < best_dist:
                best_dist = d2
                best = p.patch_id
        # 精确判定：点在多边形内
        if best >= 0 and _point_in_polygon(x, y, self.patches[best].vertices):
            return best
        # 粗筛失败 → 遍历所有
        for p in self.patches:
            if _point_in_polygon(x, y, p.vertices):
                return p.patch_id
        return -1

    def locate_walkable(self, x: float, y: float) -> int:
        """定位可行走面片（仅在 walkable 中搜索）。"""
        pid = self.locate_patch(x, y)
        if pid >= 0 and self.patches[pid].walkable:
            return pid
        return -1


# ---------------------------------------------------------------------------
# GreedyMesh 构建器
# ---------------------------------------------------------------------------


class SurfaceMeshBuilder:
    """从 Heightfield 构建 SurfaceMesh（加载期执行）。"""

    def __init__(
        self,
        hf: Heightfield,
        max_slope_deg: float = 30.0,
        max_climb: float = 2.0,
        min_walkable_area: float = 4.0,
    ) -> None:
        self.hf = hf
        self.max_slope_deg = float(max_slope_deg)
        self.max_climb = float(max_climb)
        self.min_walkable_area = float(min_walkable_area)
        self.cs = float(hf.cs)
        self.ch = float(hf.ch)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def build(self) -> SurfaceMesh:
        """全量构建 SurfaceMesh。

        管线：
        1. 扫描所有列的所有 span → 按 (layer_z, mat) 分组
        2. 每组做 2D greedy mesh 合并 → 四边形面片
        3. 可走性筛选
        4. 构建邻接图 + portal 边
        """
        # 1. 扫描分组
        groups = self._scan_groups()

        # 2. GreedyMesh → 面片
        patches = self._build_patches(groups)

        # 3. 可走性筛选
        self._filter_walkable(patches)

        # 4. 构建 SurfaceMesh（邻接图在 __post_init__ 中构建）
        mesh = SurfaceMesh(patches=patches)

        # 5. portal 富化（去重）
        self._enrich_portals(mesh)

        logger.info(
            "SurfaceMesh 构建完成: %d 面片, %d 可行走, %d 邻接边",
            len(mesh.patches),
            len(mesh.walkable_patches),
            sum(len(v) for v in mesh.adjacency.values()) // 2,
        )
        return mesh

    # ------------------------------------------------------------------
    # 扫描分组：按 (layer_z, mat) 分组
    # ------------------------------------------------------------------

    def _scan_groups(self) -> dict[tuple[int, int], np.ndarray]:
        """扫描所有列的所有 span 顶面，按 (layer_z, mat) 分组。

        Returns:
            {(layer_z, mat): bool_mask (h, w)} 每组一个布尔掩码。
        """
        w, h = self.hf.width, self.hf.height
        groups: dict[tuple[int, int], np.ndarray] = {}

        for gy in range(h):
            for gx in range(w):
                col = self.hf._col(gx, gy)  # noqa: SLF001
                n = int(self.hf.count[col])
                for li in range(n):
                    z_layer = int(self.hf.spans[col, li, _ZMAX])
                    mat = int(self.hf.spans[col, li, _MAT])
                    key = (z_layer, mat)
                    if key not in groups:
                        groups[key] = np.zeros((h, w), dtype=np.bool_)
                    groups[key][gy, gx] = True
        return groups

    # ------------------------------------------------------------------
    # GreedyMesh 合并
    # ------------------------------------------------------------------

    def _build_patches(
        self, groups: dict[tuple[int, int], np.ndarray]
    ) -> list[SurfacePatch]:
        """对每组做 2D greedy mesh 合并 → 四边形面片列表。"""
        patches: list[SurfacePatch] = []
        pid = 0

        for (layer_z, mat), mask in groups.items():
            z_world = self.hf._to_z(layer_z)  # noqa: SLF001
            h, w = mask.shape
            visited = np.zeros((h, w), dtype=np.bool_)

            for gy in range(h):
                for gx in range(w):
                    if not mask[gy, gx] or visited[gy, gx]:
                        continue

                    # 向右扩展
                    ex = gx
                    while ex + 1 < w and mask[gy, ex + 1] and not visited[gy, ex + 1]:
                        ex += 1

                    # 向下扩展
                    ey = gy
                    while ey + 1 < h:
                        row_ok = True
                        for rx in range(gx, ex + 1):
                            if not mask[ey + 1, rx] or visited[ey + 1, rx]:
                                row_ok = False
                                break
                        if not row_ok:
                            break
                        ey += 1

                    # 标记已访问
                    visited[gy : ey + 1, gx : ex + 1] = True

                    # 构建四边形顶点（世界坐标）
                    x0, y0 = gx * self.cs, gy * self.cs
                    x1, y1 = (ex + 1) * self.cs, (ey + 1) * self.cs
                    verts = np.array(
                        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        dtype=np.float32,
                    )

                    patch = SurfacePatch(
                        patch_id=pid,
                        vertices=verts,
                        z=z_world,
                        mat=int(mat),
                    )
                    patches.append(patch)
                    pid += 1

        return patches

    # ------------------------------------------------------------------
    # 可走性筛选
    # ------------------------------------------------------------------

    def _filter_walkable(self, patches: list[SurfacePatch]) -> None:
        """对每个面片判定可走性。

        条件：
        1. 面积 ≥ min_walkable_area
        2. 坡度 < max_slope_deg（水平面片坡度=0，满足）
        3. 非悬空（面片下方有支撑）—— **仅 TERRAIN 面片检查**

        人工结构（BRIDGE/BUILDING/TUNNEL）本身可站立，直接可行走
        （设计文档 §1.1 问题 #3：桥面导航必须可达）。
        """
        for p in patches:
            if p.area < self.min_walkable_area:
                continue
            # 人工结构面片：本身是「可站立的表面」，不做悬空判定
            if p.mat != MAT_TERRAIN:
                p.walkable = True
                continue
            # TERRAIN 面片：检查下方支撑
            if self._is_hanging(p):
                continue
            p.walkable = True

    def _is_hanging(self, patch: SurfacePatch) -> bool:
        """检查面片是否悬空（下方无支撑）。

        面片覆盖的所有格中，若有任一格的 span 顶面 = 本面片 z，
        且该 span 下方紧邻另一个 span（或地面）→ 有支撑，不悬空。
        否则悬空。

        简化实现：检查面片 z 对应的 layer_z 是否是所在列的最顶段 —
        如果是，且该段下面是空隙（无紧邻下一段），则悬空。

        ⚠️ 浮点坑（2026-08-11）：patch.vertices 是 float32，`x - 1e-6` 在
        numpy 2.x NEP 50 下仍为 float32，1e-6 小于 ULP 被吃掉 → 边界格
        多算一列 → 误判悬空。必须先 `float()` 提升为 float64。
        """
        cs = self.cs
        ch = self.ch
        # 先提升为 float64（避免 float32 精度吃掉 1e-6 容差）
        x0 = float(patch.vertices[:, 0].min())
        x1 = float(patch.vertices[:, 0].max())
        y0 = float(patch.vertices[:, 1].min())
        y1 = float(patch.vertices[:, 1].max())
        gx0 = max(0, int(x0 / cs))
        gx1 = min(self.hf.width - 1, int((x1 - 1e-6) / cs))
        gy0 = max(0, int(y0 / cs))
        gy1 = min(self.hf.height - 1, int((y1 - 1e-6) / cs))

        layer_z = int((patch.z - self.hf.z_min) / ch)

        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                col = self.hf._col(gx, gy)  # noqa: SLF001
                n = int(self.hf.count[col])
                found = False
                for li in range(n):
                    if int(self.hf.spans[col, li, _ZMAX]) == layer_z:
                        # 找到该面片对应的 span → 检查是否是地面（li==0）
                        # 或下方是否紧邻另一个 span
                        if li == 0:
                            found = True  # 地面 → 有支撑
                        elif (
                            int(self.hf.spans[col, li - 1, _ZMAX])
                            >= int(self.hf.spans[col, li, _ZMIN]) - 1
                        ):
                            found = True  # 下方紧邻 → 有支撑
                        break
                if not found:
                    return True  # 悬空
        return False

    # ------------------------------------------------------------------
    # Portal 富化
    # ------------------------------------------------------------------

    def _enrich_portals(self, mesh: SurfaceMesh) -> None:
        """为邻接图补充 portal 边（共享边已在 _build_adjacency 中生成）。

        这一步确保 portal_edges 中每条边都是「沿行进方向」的左右端点，
        与现有 SSFA funnel 接口兼容。
        """
        # _build_adjacency 中已填充 portal_edges，此处做后处理（如去重）
        for pid in list(mesh.portal_edges.keys()):
            # 去重（同一 portal 边可能被双向添加）
            seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
            unique: list[tuple[tuple[float, float], tuple[float, float]]] = []
            for p1, p2 in mesh.portal_edges[pid]:
                key = (p1, p2) if p1 <= p2 else (p2, p1)
                if key not in seen:
                    seen.add(key)
                    unique.append((p1, p2))
            mesh.portal_edges[pid] = unique


# ---------------------------------------------------------------------------
# Numba 辅助内核
# ---------------------------------------------------------------------------


def _polygon_area(vertices: np.ndarray) -> float:
    """多边形有向面积（Shoelace 公式），>0 为逆时针。"""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _find_shared_edge(
    verts_a: np.ndarray, verts_b: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """查找两个多边形的共享边。

    共享边定义：两个端点都在对方边界上（±eps 容差）。
    返回 (p1, p2) 世界坐标，无共享边返回 None。
    """
    eps = 1e-3
    na, nb = len(verts_a), len(verts_b)

    def _on_boundary(pt: np.ndarray, verts: np.ndarray) -> bool:
        """点是否在多边形边界上（任意边 ±eps）。"""
        nv = len(verts)
        for i in range(nv):
            a = verts[i]
            b = verts[(i + 1) % nv]
            # 点到线段距离
            ab = b - a
            ap = pt - a
            t = np.dot(ap, ab) / max(1e-9, np.dot(ab, ab))
            t = np.clip(t, 0.0, 1.0)
            closest = a + t * ab
            if np.linalg.norm(pt - closest) < eps:
                return True
        return False

    # 遍历 A 的所有边，检查是否为 B 的边
    for i in range(na):
        p1 = verts_a[i]
        p2 = verts_a[(i + 1) % na]
        if _on_boundary(p1, verts_b) and _on_boundary(p2, verts_b):
            # 确认这条边在 B 上也存在（顺序可能相反）
            for j in range(nb):
                q1 = verts_b[j]
                q2 = verts_b[(j + 1) % nb]
                d1 = np.linalg.norm(p1 - q1) + np.linalg.norm(p2 - q2)
                d2 = np.linalg.norm(p1 - q2) + np.linalg.norm(p2 - q1)
                if min(d1, d2) < eps * 2:
                    return (
                        (float(p1[0]), float(p1[1])),
                        (float(p2[0]), float(p2[1])),
                    )

    return None


def _point_in_polygon(x: float, y: float, vertices: np.ndarray) -> bool:
    """射线法：点是否在多边形内部（含边界）。"""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i, 0]), float(vertices[i, 1])
        xj, yj = float(vertices[j, 0]), float(vertices[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
