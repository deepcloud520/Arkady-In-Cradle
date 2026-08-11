"""高度场 → 渲染网格 表面提取（平滑连续网格 + GreedyMesh 结构面片）。

地面使用平滑连续三角网格：顶点落在格点、z 取四邻格中心高度的
双线性插值 → 相邻格自然形成斜坡，无体素台阶感（2026-08-10 方案 A）。
结构层（桥/建筑/隧道顶）使用 GreedyMesh 合并面片（2026-08-11 统一架构），
替代逐格四边形 → 减少 drawcall 并免费提供导航面片。

设计决策（2026-08-11）：
- 地面层保持平滑连续网格（GreedyMesh 无法优化曲面）
- 结构层从 SurfaceMesh 面片生成渲染数据（替代逐 span 四边形）
- SurfaceMesh 同时作为导航输入（三位一体：渲染/导航/碰撞）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numba import njit

from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_BUILDING,
    MAT_TERRAIN,
    MAT_TUNNEL,
    NO_SURFACE,
    _ZMAX,
    _ZMIN,
    Heightfield,
)
from Airis.lulilo.terrain.surface_mesh import SurfaceMesh, SurfaceMeshBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 颜色映射（加载期，纯 Python）
# ---------------------------------------------------------------------------

# 材质 → 基础色 (r, g, b)
_MAT_COLOR: dict[int, tuple[float, float, float]] = {
    MAT_TERRAIN: (0.25, 0.45, 0.18),   # 草绿（实际地面由 _terrain_color 按高度着色）
    MAT_BRIDGE: (0.50, 0.50, 0.55),    # 钢灰
    MAT_BUILDING: (0.65, 0.60, 0.55),  # 混凝土
    MAT_TUNNEL: (0.30, 0.28, 0.25),    # 暗岩
}


# 地面平滑网格细分度（每格 SUBDIV×SUBDIV 子格）：细分让子 quad 更接近平面，
# 消除「每格 2 个三角形折痕」产生的规律斜纹（2026-08-10）
SUBDIV = 2

# 地形色标：低处深墨绿 → 草绿 → 亮绿 → 顶部浅绿/土黄（共享端点 → 连续）
_TERRAIN_STOPS: list[tuple[float, tuple[float, float, float]]] = [
    (0.00, (0.10, 0.22, 0.14)),  # 深墨绿（低处）
    (0.35, (0.20, 0.40, 0.18)),  # 草绿
    (0.70, (0.32, 0.50, 0.26)),  # 亮绿
    (1.00, (0.45, 0.55, 0.32)),  # 顶部浅绿/土黄
]


def _terrain_color(z: float, z_min: float, z_max: float) -> tuple[float, float, float]:
    """地形颜色（标量）：纯高度连续渐变（深墨绿→草绿→亮绿→浅绿/土黄）。"""
    z_range = max(1e-6, z_max - z_min)
    t = min(1.0, max(0.0, (z - z_min) / z_range))
    for i in range(len(_TERRAIN_STOPS) - 1):
        t0, c0 = _TERRAIN_STOPS[i]
        t1, c1 = _TERRAIN_STOPS[i + 1]
        if t <= t1:
            s = (t - t0) / max(1e-6, t1 - t0)
            return tuple(c0[j] * (1 - s) + c1[j] * s for j in range(3))
    return _TERRAIN_STOPS[-1][1]


def _terrain_color_vec(t: np.ndarray) -> np.ndarray:
    """地形颜色（向量化，t ∈ [0,1] 数组）→ (N,3) float32。"""
    ts = np.array([s[0] for s in _TERRAIN_STOPS], dtype=np.float32)
    cols = np.array([s[1] for s in _TERRAIN_STOPS], dtype=np.float32)
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
    t0 = ts[idx]
    t1 = ts[idx + 1]
    w = ((t - t0) / np.maximum(1e-6, t1 - t0))[:, None]
    return (cols[idx] * (1.0 - w) + cols[idx + 1] * w).astype(np.float32)


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------


@dataclass
class RenderData:
    """表面提取输出：顶点/索引/每顶点颜色+法线（与现有 to_render_data 同构）。"""

    vertices: np.ndarray  # (N, 9) float32 [x, y, z, r, g, b, nx, ny, nz]
    indices: np.ndarray   # (M,) uint32


# ---------------------------------------------------------------------------
# SurfaceExtractor
# ---------------------------------------------------------------------------


class SurfaceExtractor:
    """从 Heightfield 提取渲染网格（平滑地面 + GreedyMesh 结构面片 + 侧面）。

    2026-08-11 重构：结构层改用 SurfaceMesh（GreedyMesh 合并面片），
    替代逐格四边形。SurfaceMesh 通过 property 暴露供导航使用。
    """

    def __init__(self, hf: Heightfield, ground_z: np.ndarray | None = None) -> None:
        """hf: 高度场；ground_z: 可选连续地面高度场 (h, w)（fBm 原始值）。

        传入连续值后，渲染网格顶点高度不再受 Heightfield 1m 量化层限制，
        消除层状「体素感」（2026-08-10）；缺省时回退 hf 量化层。
        """
        self.hf = hf
        self._ground_z = ground_z
        self._ground_z_map: np.ndarray | None = None
        self._ground_mat_map: np.ndarray | None = None
        self._ground_z_min: float = 0.0
        self._ground_z_max: float = 0.0
        # SurfaceMesh 懒加载（结构提取 + 导航共用）
        self._surface_mesh: SurfaceMesh | None = None
        # 结构顶点起始偏移（extract 时记录，供上层分离 ground/结构）
        self.struct_offset: int = 0

    @property
    def surface_mesh(self) -> SurfaceMesh:
        """获取 SurfaceMesh（懒构建），供导航层复用。"""
        if self._surface_mesh is None:
            self._surface_mesh = SurfaceMeshBuilder(self.hf).build()
        return self._surface_mesh

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract(self) -> RenderData:
        """加载期全量表面提取：平滑地面 + 结构顶面 + 结构侧面。"""
        self._build_ground_maps()
        all_verts: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []
        offset = 0

        # 1. 地面层 —— 平滑连续网格
        gv, gi = self._extract_ground_smooth()
        if len(gv):
            all_verts.append(gv)
            all_idx.append(gi)
            offset += len(gv)

        # 记录结构起始偏移（供上层按顶点分离 ground/结构）
        self.struct_offset = offset

        # 2. 结构层 —— GreedyMesh 面片顶面
        sv, si = self._extract_structures()
        if len(sv):
            all_verts.append(sv)
            all_idx.append(si + offset)
            offset += len(sv)

        # 3. 悬崖/结构侧面
        cliff_v, cliff_i = self._extract_cliff_sides()
        if len(cliff_v):
            all_verts.append(cliff_v)
            all_idx.append(cliff_i + offset)
            offset += len(cliff_v)

        if not all_verts:
            return RenderData(
                vertices=np.zeros((0, 9), dtype=np.float32),
                indices=np.zeros((0,), dtype=np.uint32),
            )

        return RenderData(
            vertices=np.concatenate(all_verts),
            indices=np.concatenate(all_idx),
        )

    def extract_tile(self, gx0: int, gy0: int, gx1: int, gy1: int) -> RenderData:
        """运行时局部重提取（tile 脏重建用）。

        只处理 [gx0, gx1) × [gy0, gy1) 范围内的列，返回局部 RenderData。
        侧面需额外包含 tile 边界外的邻居列，以正确生成边界侧面。
        """
        self._build_ground_maps()
        all_verts: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []
        offset = 0

        gv, gi = self._extract_ground_smooth_region(gx0, gy0, gx1, gy1)
        if len(gv):
            all_verts.append(gv)
            all_idx.append(gi)
            offset += len(gv)

        sv, si = self._extract_structures_region(gx0, gy0, gx1, gy1)
        if len(sv):
            all_verts.append(sv)
            all_idx.append(si + offset)
            offset += len(sv)

        cv, ci = self._extract_cliff_sides_region(gx0, gy0, gx1, gy1)
        if len(cv):
            all_verts.append(cv)
            all_idx.append(ci + offset)
            offset += len(cv)

        if not all_verts:
            return RenderData(
                vertices=np.zeros((0, 9), dtype=np.float32),
                indices=np.zeros((0,), dtype=np.uint32),
            )
        return RenderData(
            vertices=np.concatenate(all_verts),
            indices=np.concatenate(all_idx),
        )

    # ------------------------------------------------------------------
    # 地面高度/材质缓存
    # ------------------------------------------------------------------

    def _build_ground_maps(self) -> None:
        """构建 ground_z + mat 二维图（加载期/重建时调用）。

        ground_z_map 优先用外部**连续**高度场（渲染精细、无 1m 量化层带），
        缺省回退 hf 量化层。同时记录地面实际高度范围供颜色归一化。
        """
        w, h = self.hf.width, self.hf.height
        if self._ground_z is not None and self._ground_z.shape == (h, w):
            z_map = np.asarray(self._ground_z, dtype=np.float32)
        else:
            z_map = np.full((h, w), NO_SURFACE, dtype=np.float32)
            for gy in range(h):
                for gx in range(w):
                    col = self.hf._col(gx, gy)  # noqa: SLF001
                    if self.hf.count[col] > 0:
                        layer_top = int(self.hf.spans[col, 0, _ZMAX])
                        z_map[gy, gx] = self.hf._to_z(layer_top)  # noqa: SLF001
        self._ground_z_map = z_map
        self._ground_mat_map = np.zeros((h, w), dtype=np.uint8)
        for gy in range(h):
            for gx in range(w):
                col = self.hf._col(gx, gy)  # noqa: SLF001
                if self.hf.count[col] > 0:
                    self._ground_mat_map[gy, gx] = int(self.hf.spans[col, 0, 3])  # _MAT
        valid = z_map[z_map > NO_SURFACE + 1.0]
        if valid.size:
            self._ground_z_min = float(valid.min())
            self._ground_z_max = float(valid.max())
        else:
            self._ground_z_min = self.hf.z_min
            self._ground_z_max = self.hf.z_max

    # ------------------------------------------------------------------
    # 平滑连续网格 —— 地面层（2026-08-10 方案 A，替代 greedy meshing）
    # ------------------------------------------------------------------

    def _extract_ground_smooth(self) -> tuple[np.ndarray, np.ndarray]:
        """全量平滑连续网格提取地面顶面。"""
        return self._extract_ground_smooth_region(
            0, 0, self.hf.width, self.hf.height
        )

    def _extract_ground_smooth_region(
        self, gx0: int, gy0: int, gx1: int, gy1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """[gx0, gx1) × [gy0, gy1) 范围平滑连续网格（每格细分 SUBDIV×SUBDIV）。

        顶点高度用**双线性插值**自格中心高度场生成（相邻格共享边界顶点 →
        水密无裂缝），法线用高度场中心差分（跨格连续 → Phong 光照平滑）。
        细分让每个子 quad 更接近平面，消除「每格 2 个三角形折痕」产生的
        规律斜纹（2026-08-10）。

        顶点格式 (N, 9) = [x, y, z, r, g, b, nx, ny, nz]，全 numpy 向量化。
        """
        z_map = self._ground_z_map
        assert z_map is not None
        map_h, map_w = z_map.shape
        cs = self.hf.cs
        z_min = self._ground_z_min
        z_max = self._ground_z_max
        s = SUBDIV

        vw = (gx1 - gx0) * s + 1  # 细分顶点列数
        vh = (gy1 - gy0) * s + 1  # 细分顶点行数

        # 细分格点坐标（格点浮点坐标，跨 tile 边界时保持绝对位置）
        gxf = gx0 + np.arange(vw, dtype=np.float32) / s
        gyf = gy0 + np.arange(vh, dtype=np.float32) / s

        # 顶点高度：双线性插值自格中心场 z_map（坐标偏移 -0.5）
        x = gxf - 0.5
        y = gyf - 0.5
        xf = np.floor(x)
        yf = np.floor(y)
        x0 = np.clip(xf.astype(np.int32), 0, map_w - 2)
        y0 = np.clip(yf.astype(np.int32), 0, map_h - 2)
        tx = np.clip(x - xf, 0.0, 1.0)
        ty = np.clip(y - yf, 0.0, 1.0)
        z00 = z_map[y0[:, None], x0[None, :]]
        z10 = z_map[y0[:, None], x0[None, :] + 1]
        z01 = z_map[y0[:, None] + 1, x0[None, :]]
        z11 = z_map[y0[:, None] + 1, x0[None, :] + 1]
        zz = (
            (z00 * (1.0 - tx)[None, :] + z10 * tx[None, :]) * (1.0 - ty)[:, None]
            + (z01 * (1.0 - tx)[None, :] + z11 * tx[None, :]) * ty[:, None]
        )
        zz = np.asarray(zz, dtype=np.float32)
        # 防护：NO_SURFACE 列经插值会带出巨大负值 → 钳制到有效范围
        zz = np.clip(zz, self.hf.z_min, self.hf.z_max)

        # 顶点法线：中心差分（细分网格上，间距 cs/s）→ 跨格连续
        step = cs / s
        dz_dx = np.zeros_like(zz)
        dz_dy = np.zeros_like(zz)
        dz_dx[:, 1:-1] = (zz[:, 2:] - zz[:, :-2]) / (2.0 * step)
        dz_dx[:, 0] = (zz[:, 1] - zz[:, 0]) / step
        dz_dx[:, -1] = (zz[:, -1] - zz[:, -2]) / step
        dz_dy[1:-1, :] = (zz[2:, :] - zz[:-2, :]) / (2.0 * step)
        dz_dy[0, :] = (zz[1, :] - zz[0, :]) / step
        dz_dy[-1, :] = (zz[-1, :] - zz[-2, :]) / step
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(zz)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        nx = nx * inv
        ny = ny * inv
        nz = nz * inv

        # 顶点数组 (N, 9)
        gxx, gyy = np.meshgrid(gxf * cs, gyf * cs)
        verts = np.zeros((vh * vw, 9), dtype=np.float32)
        verts[:, 0] = gxx.ravel()
        verts[:, 1] = gyy.ravel()
        verts[:, 2] = zz.ravel()
        t = np.clip((zz - z_min) / max(1e-6, z_max - z_min), 0.0, 1.0)
        verts[:, 3:6] = _terrain_color_vec(t.ravel())
        verts[:, 6] = nx.ravel()
        verts[:, 7] = ny.ravel()
        verts[:, 8] = nz.ravel()

        # 索引：每子格 2 个三角形（向量化）
        nq = (vh - 1) * (vw - 1)
        gi = np.arange(nq, dtype=np.uint32)
        row = gi // (vw - 1)
        col = gi % (vw - 1)
        i00 = row * vw + col
        i10 = i00 + 1
        i01 = i00 + vw
        i11 = i01 + 1
        idx = np.empty((nq, 6), dtype=np.uint32)
        idx[:, 0] = i00
        idx[:, 1] = i10
        idx[:, 2] = i01
        idx[:, 3] = i10
        idx[:, 4] = i11
        idx[:, 5] = i01
        indices = idx.ravel()

        return verts, indices

    # ------------------------------------------------------------------
    # 结构层（桥/建筑/隧道顶）—— GreedyMesh 面片（2026-08-11 统一架构）
    # ------------------------------------------------------------------

    def _extract_structures(self) -> tuple[np.ndarray, np.ndarray]:
        return self._extract_structures_region(
            0, 0, self.hf.width, self.hf.height
        )

    def _extract_structures_region(
        self, gx0: int, gy0: int, gx1: int, gy1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """从 SurfaceMesh 面片提取结构层渲染数据（替代逐 span 四边形）。

        只提取非 TERRAIN 面片（桥/建筑/隧道顶），每面片生成 2 个三角形。
        GreedyMesh 已合并同平面同材质面片 → 大幅减少三角形数。
        """
        mesh = self.surface_mesh
        cs = self.hf.cs

        verts_list: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        offset = 0

        # 筛选范围内的非 TERRAIN 面片
        x0_world = gx0 * cs
        x1_world = gx1 * cs
        y0_world = gy0 * cs
        y1_world = gy1 * cs

        for patch in mesh.patches:
            if patch.mat == MAT_TERRAIN:
                continue
            cx, cy = patch.centroid
            if not (x0_world <= cx <= x1_world and y0_world <= cy <= y1_world):
                continue

            # 每个面片 → 1 个四边形（2 个三角形）
            cr, cg_, cb = _MAT_COLOR.get(patch.mat, (0.5, 0.5, 0.5))
            z = patch.z

            # 面片顶点（世界坐标）→ 渲染顶点
            pv = np.zeros((4, 9), dtype=np.float32)
            for vi in range(min(4, len(patch.vertices))):
                pv[vi, 0] = patch.vertices[vi, 0]  # x
                pv[vi, 1] = patch.vertices[vi, 1]  # y
                pv[vi, 2] = z
                pv[vi, 3] = cr
                pv[vi, 4] = cg_
                pv[vi, 5] = cb
                pv[vi, 6] = 0.0  # nx（顶面法线向上）
                pv[vi, 7] = 0.0  # ny
                pv[vi, 8] = 1.0  # nz

            verts_list.append(pv)
            # 2 个三角形：0-1-2, 1-3-2
            base = offset
            idx_list.append(
                np.array([base, base + 1, base + 2, base + 1, base + 3, base + 2],
                         dtype=np.uint32)
            )
            offset += 4

        if not verts_list:
            return (
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
        return np.concatenate(verts_list), np.concatenate(idx_list)

    # ------------------------------------------------------------------
    # 侧面（悬崖 + 结构侧边）
    # ------------------------------------------------------------------

    def _extract_cliff_sides(self) -> tuple[np.ndarray, np.ndarray]:
        return self._extract_cliff_sides_region(
            0, 0, self.hf.width, self.hf.height
        )

    def _extract_cliff_sides_region(
        self, gx0: int, gy0: int, gx1: int, gy1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """结构层悬空侧面（桥/建筑/隧道侧壁）。

        地形高差侧面已由平滑连续网格自然呈现（斜坡），不再生成垂直台阶
        （2026-08-10 方案 A，消除体素感）。
        """
        z_map = self._ground_z_map
        assert z_map is not None
        cs = self.hf.cs

        verts: list[list[float]] = []
        indices: list[int] = []

        # 结构悬空侧面：span 底面高于邻居地面时补侧面
        self._add_structure_sides(verts, indices, gx0, gy0, gx1, gy1, cs, z_map)

        if not verts:
            return (
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
        return np.asarray(verts, dtype=np.float32), np.asarray(indices, dtype=np.uint32)

    def _add_structure_sides(
        self,
        verts: list[list[float]],
        indices: list[int],
        gx0: int,
        gy0: int,
        gx1: int,
        gy1: int,
        cs: float,
        z_map: np.ndarray,
    ) -> None:
        """结构悬空侧面：基于 SurfaceMesh 面片边界检测悬空。

        对每个非 TERRAIN 面片，检查其四边是否悬空（邻居地面低于面片底面），
        悬空则补垂直侧面。面片底面 = z - 1m（默认 1m 厚结构板）。
        """
        mesh = self.surface_mesh
        h = self.hf.height
        w_cols = self.hf.width

        x0_world = gx0 * cs
        x1_world = gx1 * cs
        y0_world = gy0 * cs
        y1_world = gy1 * cs

        for patch in mesh.patches:
            if patch.mat == MAT_TERRAIN:
                continue
            cx, cy = patch.centroid
            if not (x0_world <= cx <= x1_world and y0_world <= cy <= y1_world):
                continue

            z_bot = patch.z - 1.0  # 默认 1m 厚结构板
            z_top = patch.z
            mat = patch.mat
            cr, cg_, cb = _MAT_COLOR.get(mat, (0.5, 0.5, 0.5))

            # 面片边界世界坐标
            v = patch.vertices
            nv = len(v)

            for vi in range(nv):
                p1 = v[vi]
                p2 = v[(vi + 1) % nv]
                # 判断边的方向（水平还是垂直）
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])

                if dx < 1e-6:
                    # 垂直边（沿 Y）：检查左右邻居
                    edge_x = p1[0]
                    y_lo = min(p1[1], p2[1])
                    y_hi = max(p1[1], p2[1])
                    gx_edge = int(edge_x / cs)
                    if gx_edge <= 0 or gx_edge >= w_cols:
                        continue
                    # 检查边两侧的地面高度
                    gy_mid = int((y_lo + y_hi) * 0.5 / cs)
                    gy_mid = max(0, min(h - 1, gy_mid))
                    nz_left = float(z_map[gy_mid, gx_edge - 1]) if gx_edge > 0 else self.hf.z_min
                    nz_right = float(z_map[gy_mid, gx_edge]) if gx_edge < w_cols else self.hf.z_min
                    # 若任一侧地面低于底面 → 悬空，生成侧面
                    if nz_left < z_bot - 1e-4:
                        snx = -1.0
                        base = len(verts)
                        verts.append([edge_x, y_lo, max(nz_left, self.hf.z_min), cr, cg_, cb, snx, 0.0, 0.0])
                        verts.append([edge_x, y_hi, max(nz_left, self.hf.z_min), cr, cg_, cb, snx, 0.0, 0.0])
                        verts.append([edge_x, y_lo, z_bot, cr, cg_, cb, snx, 0.0, 0.0])
                        verts.append([edge_x, y_hi, z_bot, cr, cg_, cb, snx, 0.0, 0.0])
                        indices.extend([base, base + 1, base + 2, base + 1, base + 3, base + 2])
                    if nz_right < z_bot - 1e-4:
                        snx = 1.0
                        base = len(verts)
                        verts.append([edge_x, y_lo, max(nz_right, self.hf.z_min), cr, cg_, cb, snx, 0.0, 0.0])
                        verts.append([edge_x, y_hi, max(nz_right, self.hf.z_min), cr, cg_, cb, snx, 0.0, 0.0])
                        verts.append([edge_x, y_lo, z_bot, cr, cg_, cb, snx, 0.0, 0.0])
                        verts.append([edge_x, y_hi, z_bot, cr, cg_, cb, snx, 0.0, 0.0])
                        indices.extend([base, base + 1, base + 2, base + 1, base + 3, base + 2])
                elif dy < 1e-6:
                    # 水平边（沿 X）：检查上下邻居
                    edge_y = p1[1]
                    x_lo = min(p1[0], p2[0])
                    x_hi = max(p1[0], p2[0])
                    gy_edge = int(edge_y / cs)
                    if gy_edge <= 0 or gy_edge >= h:
                        continue
                    gx_mid = int((x_lo + x_hi) * 0.5 / cs)
                    gx_mid = max(0, min(w_cols - 1, gx_mid))
                    nz_top = float(z_map[gy_edge - 1, gx_mid]) if gy_edge > 0 else self.hf.z_min
                    nz_bot = float(z_map[gy_edge, gx_mid]) if gy_edge < h else self.hf.z_min
                    if nz_top < z_bot - 1e-4:
                        sny = -1.0
                        base = len(verts)
                        verts.append([x_lo, edge_y, max(nz_top, self.hf.z_min), cr, cg_, cb, 0.0, sny, 0.0])
                        verts.append([x_hi, edge_y, max(nz_top, self.hf.z_min), cr, cg_, cb, 0.0, sny, 0.0])
                        verts.append([x_lo, edge_y, z_bot, cr, cg_, cb, 0.0, sny, 0.0])
                        verts.append([x_hi, edge_y, z_bot, cr, cg_, cb, 0.0, sny, 0.0])
                        indices.extend([base, base + 1, base + 2, base + 1, base + 3, base + 2])
                    if nz_bot < z_bot - 1e-4:
                        sny = 1.0
                        base = len(verts)
                        verts.append([x_lo, edge_y, max(nz_bot, self.hf.z_min), cr, cg_, cb, 0.0, sny, 0.0])
                        verts.append([x_hi, edge_y, max(nz_bot, self.hf.z_min), cr, cg_, cb, 0.0, sny, 0.0])
                        verts.append([x_lo, edge_y, z_bot, cr, cg_, cb, 0.0, sny, 0.0])
                        verts.append([x_hi, edge_y, z_bot, cr, cg_, cb, 0.0, sny, 0.0])
                        indices.extend([base, base + 1, base + 2, base + 1, base + 3, base + 2])


