"""WorldMesh 3D 世界模型：WorldPatch 面片集合 + STRtree 空间索引 + 邻接图。

对应 docs/worldmesh-3d-surface-design.md：从 Heightfield 提取 3D 表面面片，
作为物理/碰撞精算/z 采样的世界模型（区别于 Heightfield 的碰撞粗筛）。

与旧 SurfaceMesh（surface_mesh.py）的核心差别：
- vertices 从 (V,2)+z 升级为 (V,3) 真 3D 坐标
- normal 从隐式 (0,0,1) 升级为显式 (3,) 任意朝向
- 点定位从 O(n) 形心粗筛升级为 STRtree O(log N)
- 新增 ray_plane_z 竖直射线求交（斜面/桥下单位定位）

2026-08-12 设计决策（与用户讨论）：
1. 移动热路径 mesh_id 追踪**不需要邻接表**——利用「单位移动连续」的物理事实：
   先查当前面片（Numba point_in_poly），99%+ 的 tick 直接命中；
   miss 才回退 Python 侧 STRtree locate_patch(x, y, hint_z=z)。
2. 邻接表（adjacency/portal_edges）仅供 A* 寻路 + SSFA 漏斗使用（纯 Python）。
3. 生产规模（4000×4000 ~ 8000×10000）下，不建全量 Grid Hash（内存浪费），
   STRtree 内存 O(面片数) 且与格数无关。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np

from Airis.lulilo.terrain.heightfield import (
    MAT_TERRAIN,
    _MAT,
    _ZMAX,
    _ZMIN,
    Heightfield,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 面朝向枚举（Phase B Dual Contouring 使用）
# ---------------------------------------------------------------------------

FACE_TOP = 0      # 顶面（水平，法线 +z）
FACE_BOTTOM = 1   # 底面（水平，法线 -z）
FACE_PX = 2       # 侧面（垂直，法线 +x）
FACE_NX = 3       # 侧面（垂直，法线 -x）
FACE_PY = 4       # 侧面（垂直，法线 +y）
FACE_NY = 5       # 侧面（垂直，法线 -y）

# 面朝向 → 单位法线
_FACE_NORMALS: dict[int, tuple[float, float, float]] = {
    FACE_TOP: (0.0, 0.0, 1.0),
    FACE_BOTTOM: (0.0, 0.0, -1.0),
    FACE_PX: (1.0, 0.0, 0.0),
    FACE_NX: (-1.0, 0.0, 0.0),
    FACE_PY: (0.0, 1.0, 0.0),
    FACE_NY: (0.0, -1.0, 0.0),
}

# ---------------------------------------------------------------------------
# WorldPatch —— 3D 世界面片（世界模型的原子单元）
# ---------------------------------------------------------------------------


@dataclass
class WorldPatch:
    """3D 世界面片 —— 世界模型的原子单元。

    Attributes:
        patch_id: 唯一 ID（在 WorldMesh 内索引）。
        vertices: (V, 3) float32  世界 3D 坐标 (x, y, z)，逆时针闭环（首尾不重复）。
        normal: (3,) float32  面片法线（单位向量，指向可站立表面外侧）。
        mat: 材质（MAT_TERRAIN / MAT_BRIDGE / MAT_BUILDING / MAT_TUNNEL）。
        face: 面朝向（FACE_TOP / FACE_BOTTOM / FACE_PX...，Phase B）。
        walkable: 是否可行走（由坡度 + 悬空 + 材质判定）。
        area: 面积 (m²)。
        centroid: (3,) float32  形心世界坐标。
    """

    patch_id: int
    vertices: np.ndarray        # (V, 3) float32
    normal: np.ndarray          # (3,) float32
    mat: int
    face: int = FACE_TOP
    walkable: bool = False
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self) -> None:
        if len(self.vertices) >= 3:
            self.area = float(_polygon_area_3d(self.vertices))
            self.centroid = self.vertices.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# WorldMesh —— 面片集合 + STRtree 空间索引 + 邻接图
# ---------------------------------------------------------------------------


@dataclass
class WorldMesh:
    """3D 世界面片集合 + 空间索引 + 邻接图。

    Attributes:
        patches: 所有面片（按 patch_id 索引）。
        adjacency: {patch_id: [neighbor_id]}  共享边邻接（供 A*/SSFA）。
        portal_edges: {patch_id: [(p1_3d, p2_3d)]}  门户边（3D 世界坐标）。
        walkable_patches: {patch_id}  可行走面片缓存。
        _polygons: [shapely.Polygon, ...]  面片 2D 轮廓（与 patches 同序）。
        _tree: STRtree 空间索引（懒构建）。
    """

    patches: list[WorldPatch]
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    portal_edges: dict[
        int, list[tuple[np.ndarray, np.ndarray]]
    ] = field(default_factory=dict)
    walkable_patches: set[int] = field(default_factory=set)

    # 空间索引（内部，懒构建）
    _polygons: Any | None = field(default=None, repr=False)
    _tree: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.walkable_patches = {p.patch_id for p in self.patches if p.walkable}

    # ------------------------------------------------------------------
    # 空间索引（STRtree）
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """构建 STRtree 空间索引（首次查询时懒加载）。

        用面片 XY 投影的 2D 轮廓建索引（点定位是 2D 查询 + 3D z 裁决）。
        """
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree

        self._polygons = [
            Polygon(p.vertices[:, :2]) for p in self.patches
        ]
        self._tree = STRtree(self._polygons)

    # ------------------------------------------------------------------
    # 点定位
    # ------------------------------------------------------------------

    def locate_patch(
        self,
        x: float,
        y: float,
        mat: int | None = None,
        hint_z: float | None = None,
    ) -> int:
        """点定位：返回覆盖 (x,y) 的面片 ID。O(log N) 粗筛 + 精确判定。

        Args:
            x, y: 世界坐标。
            mat: 材质过滤（可选）。
            hint_z: 期望高度（可选）。当同 (x,y) 有多个面片时，
                比较各面片在 (x,y) 处的 ray_plane_z 值与 hint_z 的
                接近程度，返回最接近的面片（而非表面最高的）。
                用于桥下单位定位等场景。

        Returns:
            面片 ID，未找到返回 -1。
        """
        if self._tree is None:
            self._build_index()

        from shapely.geometry import Point

        pt = Point(x, y)
        candidates: list[int] = list(self._tree.query(pt))

        best_pid = -1
        # 无 hint → 取表面最高（越大越好，用 -inf 让首个候选入选）；
        # 有 hint → 取与 hint_z 最接近（越小越好，用 +inf）。
        best_dist = float("-inf") if hint_z is None else float("inf")
        for pid in candidates:
            p = self.patches[pid]
            if mat is not None and p.mat != mat:
                continue
            # covers 含边界（逐格地面面片的共享边/角点必须命中）
            if not self._polygons[pid].covers(pt):
                continue
            z = self._ray_plane_z_impl(p, x, y)
            if hint_z is not None:
                d = abs(z - hint_z)
                if d < best_dist:
                    best_dist = d
                    best_pid = pid
            elif z > best_dist:  # 无 hint → 取表面最高的面片
                best_dist = z
                best_pid = pid
        return best_pid

    def locate_walkable(
        self, x: float, y: float, hint_z: float | None = None
    ) -> int:
        """定位可行走面片（仅在 walkable 中搜索）。"""
        pid = self.locate_patch(x, y, hint_z=hint_z)
        if pid >= 0 and self.patches[pid].walkable:
            return pid
        return -1

    # ------------------------------------------------------------------
    # z 采样（竖直射线求交）
    # ------------------------------------------------------------------

    def ray_plane_z(self, patch_id: int, x: float, y: float) -> float:
        """竖直射线 (x,y,±∞) 与面片的交点 z。

        ray(t) = (x, y, t), plane: n·(p - v0) = 0
        → t = (n·v0 - n.x·x - n.y·y) / n.z
        """
        return self._ray_plane_z_impl(self.patches[patch_id], x, y)

    @staticmethod
    def _ray_plane_z_impl(patch: WorldPatch, x: float, y: float) -> float:
        n = patch.normal
        v0 = patch.vertices[0]
        if abs(n[2]) < 1e-9:
            return -1e9  # 竖直面，不可站立
        return (float(np.dot(n, v0)) - n[0] * x - n[1] * y) / n[2]


# ---------------------------------------------------------------------------
# WorldMeshBuilder —— 从 Heightfield 构建 WorldMesh（加载期执行）
# ---------------------------------------------------------------------------


class WorldMeshBuilder:
    """从 Heightfield 构建 WorldMesh（加载期执行一次）。

    管线（与 SurfaceMeshBuilder 同结构，输出 3D 面片）：
    1. 扫描所有列的所有 span 顶面 → 按 (layer_z, mat) 分组
    2. 每组做 2D greedy mesh 合并 → WorldPatch（3D 顶点 + 法线）
    3. 可走性筛选
    4. 构建邻接图 + portal 边（3D）

    Phase A 简化：面片均为轴对齐水平面（normal=(0,0,1)）。
    斜面/任意朝向由 Phase B（Dual Contouring）引入。
    """

    def __init__(
        self,
        hf: Heightfield,
        max_slope_deg: float = 30.0,
        max_climb: float = 2.0,
        min_walkable_area: float = 4.0,
        continuous_ground: np.ndarray | None = None,
    ) -> None:
        self.hf = hf
        self.max_slope_deg = float(max_slope_deg)
        self.max_climb = float(max_climb)
        self.min_walkable_area = float(min_walkable_area)
        self.cs = float(hf.cs)
        self.ch = float(hf.ch)
        # 统一接口（Phase B DualContourBuilder 使用）；Phase A 顶面不做平滑
        self.continuous_ground = continuous_ground

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def build(self) -> WorldMesh:
        """全量构建 WorldMesh。

        管线：
        1. 扫描所有列的所有 span → 按 (layer_z, mat) 分组
        2. 每组做 2D greedy mesh 合并 → 3D 面片
        3. 可走性筛选
        4. 构建邻接图 + 3D portal 边
        """
        # 1. 扫描分组
        groups = self._scan_groups()

        # 2. GreedyMesh → 3D 面片
        patches = self._build_patches_3d(groups)

        # 3. 可走性筛选
        self._filter_walkable(patches)

        # 4. 构建 WorldMesh（邻接图在此构建）
        mesh = WorldMesh(patches=patches)
        self._build_adjacency_3d(mesh)

        logger.info(
            "WorldMesh 构建完成: %d 面片, %d 可行走, %d 邻接边",
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
        spans, count = self.hf.spans, self.hf.count

        for gy in range(h):
            for gx in range(w):
                col = gy * w + gx
                n = int(count[col])
                for li in range(n):
                    z_layer = int(spans[col, li, _ZMAX])
                    mat = int(spans[col, li, _MAT])
                    key = (z_layer, mat)
                    if key not in groups:
                        groups[key] = np.zeros((h, w), dtype=np.bool_)
                    groups[key][gy, gx] = True
        return groups

    # ------------------------------------------------------------------
    # GreedyMesh 合并 → 3D 面片
    # ------------------------------------------------------------------

    def _build_patches_3d(
        self, groups: dict[tuple[int, int], np.ndarray]
    ) -> list[WorldPatch]:
        """对每组做 2D greedy mesh 合并 → 3D 世界面片列表。"""
        patches: list[WorldPatch] = []
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

                    # 构建 3D 四边形顶点（世界坐标）
                    x0, y0 = gx * self.cs, gy * self.cs
                    x1, y1 = (ex + 1) * self.cs, (ey + 1) * self.cs
                    verts = np.array(
                        [
                            [x0, y0, z_world],
                            [x1, y0, z_world],
                            [x1, y1, z_world],
                            [x0, y1, z_world],
                        ],
                        dtype=np.float32,
                    )

                    patch = WorldPatch(
                        patch_id=pid,
                        vertices=verts,
                        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                        mat=int(mat),
                    )
                    patches.append(patch)
                    pid += 1

        return patches

    # ------------------------------------------------------------------
    # 可走性筛选
    # ------------------------------------------------------------------

    def _filter_walkable(self, patches: list[WorldPatch]) -> None:
        """对每个面片判定可走性。

        条件：
        1. 面积 ≥ min_walkable_area
        2. 坡度 < max_slope_deg（水平面片坡度=0，满足）
        3. 非悬空（面片下方有支撑）—— **仅 TERRAIN 面片检查**

        人工结构（BRIDGE/BUILDING/TUNNEL）本身可站立，直接可行走。
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

    def _is_hanging(self, patch: WorldPatch) -> bool:
        """检查面片是否悬空（下方无支撑）。

        面片覆盖的所有格中，若有任一格的 span 顶面 = 本面片 z，
        且该 span 下方紧邻另一个 span（或地面）→ 有支撑，不悬空。
        否则悬空。

        ⚠️ 浮点坑（2026-08-11）：vertices 是 float32，`x - 1e-6` 在
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

        layer_z = int((patch.vertices[0, 2] - self.hf.z_min) / ch)

        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                col = gy * self.hf.width + gx
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
    # 邻接图 + 3D portal 边
    # ------------------------------------------------------------------

    def _build_adjacency_3d(self, mesh: WorldMesh) -> None:
        """构建面片邻接图：共享边 + z 落差 < max_climb（委托模块级函数）。"""
        build_adjacency(mesh, self.max_climb)


# ---------------------------------------------------------------------------
# 邻接图构建（模块级，供 WorldMeshBuilder / DualContourBuilder 共用）
# ---------------------------------------------------------------------------


def build_adjacency(mesh: WorldMesh, max_climb: float) -> None:
    """构建面片邻接图：共享边 + z 落差 < max_climb。

    两 patch 邻接条件：
    1. 存在共享边（两个顶点都在对方边界上）
    2. z 落差 < max_climb（默认 2m）

    实现：**扫描线算法**（O(n log n)）。面片为轴对齐矩形 → 共享边 =
    同一直线上的重叠边。
    - 垂直边 (x, y0, y1) 按 x 分组，组内检查 y 区间重叠
    - 水平边 (y, x0, x1) 按 y 分组，组内检查 x 区间重叠

    portal 边为 3D 坐标（含 z）。只对水平面片（顶/底面）有意义；
    侧面（垂直面）的 z 取 vertices[0,2] 不参与（Phase C 寻路只筛 walkable）。
    """
    mesh.adjacency = {}
    mesh.portal_edges = {}

    eps = 1e-4

    # 收集矩形边界（轴对齐矩形）
    v_edges: list[tuple[float, float, float, int]] = []  # (x, y0, y1, pid)
    h_edges: list[tuple[float, float, float, int]] = []  # (y, x0, x1, pid)
    for p in mesh.patches:
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
        pa = mesh.patches[a]
        pb = mesh.patches[b]
        if abs(pa.vertices[0, 2] - pb.vertices[0, 2]) > max_climb:
            return
        z = float(pa.vertices[0, 2])
        p1_3d = (float(p1[0]), float(p1[1]), z)
        p2_3d = (float(p2[0]), float(p2[1]), z)
        mesh.adjacency.setdefault(a, []).append(b)
        mesh.adjacency.setdefault(b, []).append(a)
        mesh.portal_edges.setdefault(a, []).append((p1_3d, p2_3d))
        mesh.portal_edges.setdefault(b, []).append((p1_3d, p2_3d))

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


# ---------------------------------------------------------------------------
# Numba 展平辅助（移动热路径用）
# ---------------------------------------------------------------------------


def flatten_world_mesh(
    mesh: WorldMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把 WorldMesh 展平为 Numba 友好数组（每 tick 移动积分/同步用）。

    Returns:
        patch_v:      (total_v, 3) float32  所有面片顶点拼接
        patch_vstart: (M+1,) int32          面片顶点起始偏移（含末尾哨兵）
        patch_vcnt:   (M,) int32            每面片顶点数
        patch_norm:   (M, 3) float32        每面片法线
    """
    m = len(mesh.patches)
    total_v = sum(len(p.vertices) for p in mesh.patches)
    patch_v = np.zeros((total_v, 3), dtype=np.float32)
    patch_vstart = np.zeros(m + 1, dtype=np.int32)
    patch_vcnt = np.zeros(m, dtype=np.int32)
    patch_norm = np.zeros((m, 3), dtype=np.float32)

    off = 0
    for i, p in enumerate(mesh.patches):
        patch_vstart[i] = off
        patch_vcnt[i] = len(p.vertices)
        patch_v[off : off + len(p.vertices)] = p.vertices
        patch_norm[i] = p.normal
        off += len(p.vertices)
    patch_vstart[m] = off
    return patch_v, patch_vstart, patch_vcnt, patch_norm


# ---------------------------------------------------------------------------
# 通用几何工具
# ---------------------------------------------------------------------------


def _polygon_area_3d(vertices: np.ndarray) -> float:
    """3D 多边形面积（Newell's method，适用于任意朝向平面多边形）。

    面积 = 0.5 * |n|（n 为未归一化法线）。
    """
    n = len(vertices)
    nx = ny = nz = 0.0
    for i in range(n):
        x1, y1, z1 = vertices[i]
        x2, y2, z2 = vertices[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return float(0.5 * sqrt(nx * nx + ny * ny + nz * nz))


def _polygon_area_2d(vertices: np.ndarray) -> float:
    """多边形有向面积（Shoelace 公式），>0 为逆时针。"""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(
        0.5 * abs(float(np.dot(x, np.roll(y, 1))) - float(np.dot(y, np.roll(x, 1))))
    )


def point_in_polygon(x: float, y: float, vertices: np.ndarray) -> bool:
    """射线法：点是否在多边形内部（含边界）。vertices 可为 (V,2) 或 (V,3)。"""
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
