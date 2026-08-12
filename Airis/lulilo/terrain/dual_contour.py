"""Dual Contouring 简化版：column-span 体素 → 完整 3D WorldPatch（顶面+底面+侧面）。

对应 docs/worldmesh-3d-surface-design.md §4（Phase B）：
从 Heightfield 的 column-span 提取 3D 表面面片，作为 WorldMesh 的世界模型。

设计（2026-08-12）：
- column-span 是**轴对齐盒** → 暴露面恒为轴对齐（水平顶/底面 + 垂直侧面）
  → **不需要解 QEF**——顶点直接定位在盒边，保留锐利特征（桥边/建筑棱角）。
  完整 QEF/Hermite 仅在需要非轴对齐斜面（弹坑/坡道）时引入（后续迭代）。
- 顶面/底面：水平面片（normal ±z），2D GreedyMesh 合并
- 侧面：垂直面片（normal ±x/±y），沿墙方向 + z 层带一维合并
- 平滑（B2）：MAT_TERRAIN 顶面顶点 z 用 continuous_ground 双线性插值
  替代量化层 z（消除 1m 台阶感）
- 法线（B4）：按面朝向直接给出（Phase B 不引入斜面法线）

与 Phase A WorldMeshBuilder 的关系：
- WorldMeshBuilder：仅顶面（2.5D，Phase A）
- DualContourBuilder：顶面 + 底面 + 侧面（完整 3D，Phase B）
  两者输出同为 WorldMesh，可互切换（TerrainMap.world_mesh 走本实现）。
"""

from __future__ import annotations

import logging

import numpy as np

from Airis.lulilo.terrain.heightfield import (
    MAT_TERRAIN,
    MAX_SPANS,
    _MAT,
    _ZMAX,
    _ZMIN,
    Heightfield,
)
from Airis.lulilo.terrain.world_mesh import (
    FACE_BOTTOM,
    FACE_NX,
    FACE_NY,
    FACE_PX,
    FACE_PY,
    FACE_TOP,
    _FACE_NORMALS,
    WorldMesh,
    WorldPatch,
    build_adjacency,
)

logger = logging.getLogger(__name__)


class DualContourBuilder:
    """从 Heightfield 提取完整 3D WorldMesh（顶面+底面+侧面，加载期执行）。

    Args:
        hf: 高度场（column-span）。
        continuous_ground: 可选连续地面高度场 (h, w)。提供时 MAT_TERRAIN
            顶面顶点 z 用其双线性插值（平滑去量化）；缺省回退量化层 z。
        max_climb: 邻接最大 z 落差（m）。
        min_walkable_area: 可行走最小面积 (m²)。
        gap: 暴露面判定间隙阈值（层）。
    """

    def __init__(
        self,
        hf: Heightfield,
        continuous_ground: np.ndarray | None = None,
        max_climb: float = 2.0,
        min_walkable_area: float = 4.0,
        gap: int = 1,
        include_ground: bool = True,
        build_adjacency: bool = True,
    ) -> None:
        self.hf = hf
        self.continuous_ground = continuous_ground
        self.max_climb = float(max_climb)
        self.min_walkable_area = float(min_walkable_area)
        self.gap = int(gap)
        self.cs = float(hf.cs)
        self.ch = float(hf.ch)
        self.w = int(hf.width)
        self.h = int(hf.height)
        # 渲染用 WorldMesh：跳过逐格地面（地面由 continuous_ground 提供，
        # 65536 个地面 patch 是纯负担）+ 跳过邻接（渲染不寻路）
        self.include_ground = include_ground
        self.build_adjacency = build_adjacency

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def build(self) -> WorldMesh:
        """全量构建 WorldMesh（完整 3D 表面）。

        管线：
        1. 暴露面判定（numpy 向量化：顶/底/侧）
        2. 顶面/底面 2D GreedyMesh → 水平面片
        3. 侧面一维合并 → 垂直面片
        4. 地面平滑（MAT_TERRAIN 顶面 z ← continuous_ground 双线性）
        5. walkable 筛选（仅顶面）
        6. 邻接图（仅水平面片，扫描线）
        """
        top_mask, bottom_mask, side_masks = self._expose_masks()

        # 2. 顶面/底面
        patches = self._build_horizontal(top_mask, bottom_mask)

        # 3. 侧面
        patches.extend(self._build_sides(side_masks))

        # 4. 地面平滑
        if self.continuous_ground is not None:
            self._smooth_ground(patches)

        # 5. walkable 筛选
        self._mark_walkable(patches)

        # 6. 邻接图（渲染用跳过）
        mesh = WorldMesh(patches=patches)
        if self.build_adjacency:
            build_adjacency(mesh, self.max_climb)

        logger.info(
            "DualContour WorldMesh: %d 面片（顶 %d / 底 %d / 侧 %d）, %d 可行走, %d 邻接边",
            len(mesh.patches),
            sum(1 for p in mesh.patches if p.face == FACE_TOP),
            sum(1 for p in mesh.patches if p.face == FACE_BOTTOM),
            sum(1 for p in mesh.patches if p.face >= FACE_PX),
            len(mesh.walkable_patches),
            sum(len(v) for v in mesh.adjacency.values()) // 2,
        )
        return mesh

    # ------------------------------------------------------------------
    # ① 暴露面判定（numpy 向量化）
    # ------------------------------------------------------------------

    def _expose_masks(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        """判定所有列所有 span 的暴露面。

        Returns:
            top_mask:    (n_cols, MAX_SPANS) bool  顶面暴露
            bottom_mask: (n_cols, MAX_SPANS) bool  底面暴露
            side_masks:  {face_dir: (n_cols, MAX_SPANS) bool}  各方向侧面暴露
        """
        spans, count = self.hf.spans, self.hf.count
        n_cols = self.w * self.h
        gap = self.gap

        span_idx = np.arange(MAX_SPANS)[None, :]
        is_valid = count[:, None] > span_idx

        # ---- 顶面：最顶层 或 上方有间隙 ----
        is_last = count[:, None] == (span_idx + 1)
        gap_ok = np.ones((n_cols, MAX_SPANS), dtype=bool)
        gap_ok[:, :-1] = (spans[:, 1:, _ZMIN] - spans[:, :-1, _ZMAX]) >= gap
        top_mask = is_valid & (is_last | gap_ok)

        # ---- 底面：si>0 且 下方有间隙 ----
        bottom_mask = np.zeros((n_cols, MAX_SPANS), dtype=bool)
        if MAX_SPANS > 1:
            bottom_mask[:, 1:] = (
                spans[:, 1:, _ZMIN] - spans[:, :-1, _ZMAX]
            ) >= gap
        bottom_mask &= is_valid

        # ---- 侧面：邻居列在 [z0,z1) 区间无覆盖 ----
        # 🔥 2026-08-12 修复：邻居 **TERRAIN 地面不遮挡结构侧面**。
        # 桥墩立在峡谷底，邻居地面（斜坡/桥台）只要高于桥墩底，
        # 地面 span 与桥墩 z 区间重叠 → 误判覆盖 → 桥墩某方向侧面整个缺失
        # （看起来桥墩"短了一截"）。地面是地面，不参与结构侧面遮挡。
        gx_all = np.arange(n_cols) % self.w
        gy_all = np.arange(n_cols) // self.w
        side_masks: dict[int, np.ndarray] = {}
        for face, (dx, dy) in (
            (FACE_PX, (1, 0)),
            (FACE_NX, (-1, 0)),
            (FACE_PY, (0, 1)),
            (FACE_NY, (0, -1)),
        ):
            nx = gx_all + dx
            ny = gy_all + dy
            valid = (nx >= 0) & (nx < self.w) & (ny >= 0) & (ny < self.h)
            ncol = np.where(valid, ny * self.w + nx, -1)

            nb_spans = spans[ncol]  # 越界 -1 → 最后一行（被 nb_valid 屏蔽）
            nb_valid = np.where(valid, count[ncol], 0)[:, None] > span_idx
            nb_is_struct = nb_spans[:, :, _MAT] != MAT_TERRAIN

            this_z0 = spans[:, :, _ZMIN].astype(np.int32)
            this_z1 = spans[:, :, _ZMAX].astype(np.int32)
            nb_z0 = nb_spans[:, :, _ZMIN].astype(np.int32)
            nb_z1 = nb_spans[:, :, _ZMAX].astype(np.int32)

            covered = np.zeros((n_cols, MAX_SPANS), dtype=bool)
            for sj in range(MAX_SPANS):
                ov = (nb_z0[:, sj : sj + 1] < this_z1) & (
                    nb_z1[:, sj : sj + 1] > this_z0
                )
                ov &= nb_valid[:, sj : sj + 1]
                ov &= nb_is_struct[:, sj : sj + 1]  # 仅结构遮挡结构
                covered |= ov
            side_masks[face] = is_valid & ~covered

        return top_mask, bottom_mask, side_masks

    # ------------------------------------------------------------------
    # ② 顶面/底面 → 水平面片
    # ------------------------------------------------------------------

    def _build_horizontal(
        self,
        top_mask: np.ndarray,
        bottom_mask: np.ndarray,
    ) -> list[WorldPatch]:
        """顶面/底面 → 水平面片。

        地面顶面（MAT_TERRAIN）：**逐格**四边形 —— 连续场有内部起伏，
        GreedyMesh 合并会丢失内部顶点导致平滑失效（2026-08-12 设计决策，
        docs §4.2「每格最多 1~3 个面片」）；顶点 z 由 B2 平滑阶段用
        continuous_ground 双线性插值。
        结构顶面/底面（BRIDGE/BUILDING/TUNNEL）：**GreedyMesh 合并** ——
        结构是平面，合并大幅减面且不丢信息。
        """
        patches: list[WorldPatch] = []
        w, h, cs = self.w, self.h, self.cs
        spans = self.hf.spans

        # ---- 结构顶面/底面：GreedyMesh 合并 ----
        for face in (FACE_TOP, FACE_BOTTOM):
            mask = top_mask if face == FACE_TOP else bottom_mask
            ys, si = np.nonzero(mask)
            if ys.size == 0:
                continue
            # 只处理非 TERRAIN（地面走逐格路径）
            struct = [
                k for k in range(len(ys))
                if int(spans[int(ys[k]), int(si[k]), _MAT]) != MAT_TERRAIN
            ]
            if not struct:
                continue
            ys_s = ys[struct]
            si_s = si[struct]
            # 按 (mat, z_layer) 分组
            groups: dict[tuple[int, int], np.ndarray] = {}
            z_layers = (
                spans[ys_s, si_s, _ZMAX] if face == FACE_TOP
                else spans[ys_s, si_s, _ZMIN]
            )
            for k in range(len(ys_s)):
                key = (int(spans[int(ys_s[k]), int(si_s[k]), _MAT]), int(z_layers[k]))
                if key not in groups:
                    groups[key] = np.zeros((h, w), dtype=np.bool_)
                groups[key][int(ys_s[k]) // w, int(ys_s[k]) % w] = True

            for (mat, z_layer), gmask in groups.items():
                z_world = self.hf._to_z(z_layer)  # noqa: SLF001
                self._emit_merged(patches, gmask, mat, face, z_world)

        # ---- 地面顶面：逐格（include_ground=False 时跳过——渲染地面走 continuous_ground）----
        if self.include_ground:
            ys, si = np.nonzero(top_mask)
            ground = [
                k for k in range(len(ys))
                if int(spans[int(ys[k]), int(si[k]), _MAT]) == MAT_TERRAIN
            ]
            for k in ground:
                col = int(ys[k])
                gx, gy = col % w, col // w
                z_layer = int(spans[col, int(si[k]), _ZMAX])
                z_world = self.hf._to_z(z_layer)  # noqa: SLF001
                x0, y0 = gx * cs, gy * cs
                x1, y1 = (gx + 1) * cs, (gy + 1) * cs
                verts = np.array(
                    [
                        [x0, y0, z_world],
                        [x1, y0, z_world],
                        [x1, y1, z_world],
                        [x0, y1, z_world],
                    ],
                    dtype=np.float32,
                )
                patches.append(
                    WorldPatch(
                        patch_id=0,  # 占位
                        vertices=verts,
                        normal=np.array(_FACE_NORMALS[FACE_TOP], dtype=np.float32),
                        mat=MAT_TERRAIN,
                        face=FACE_TOP,
                    )
                )

        for i, p in enumerate(patches):
            p.patch_id = i
        return patches

    def _emit_merged(
        self,
        patches: list[WorldPatch],
        gmask: np.ndarray,
        mat: int,
        face: int,
        z_world: float,
    ) -> None:
        """对布尔掩码做 2D GreedyMesh 合并，输出同平面水平面片。"""
        w, h, cs = self.w, self.h, self.cs
        for gy in range(h):
            for gx in range(w):
                if not gmask[gy, gx]:
                    continue
                ex = gx
                while ex + 1 < w and gmask[gy, ex + 1]:
                    ex += 1
                ey = gy
                while ey + 1 < h and np.all(gmask[ey + 1, gx : ex + 1]):
                    ey += 1
                gmask[gy : ey + 1, gx : ex + 1] = False

                x0, y0 = gx * cs, gy * cs
                x1, y1 = (ex + 1) * cs, (ey + 1) * cs
                verts = np.array(
                    [
                        [x0, y0, z_world],
                        [x1, y0, z_world],
                        [x1, y1, z_world],
                        [x0, y1, z_world],
                    ],
                    dtype=np.float32,
                )
                patches.append(
                    WorldPatch(
                        patch_id=0,  # 占位
                        vertices=verts,
                        normal=np.array(_FACE_NORMALS[face], dtype=np.float32),
                        mat=mat,
                        face=face,
                    )
                )

    # ------------------------------------------------------------------
    # ③ 侧面 → 垂直面片（一维合并）
    # ------------------------------------------------------------------

    def _build_sides(
        self, side_masks: dict[int, np.ndarray]
    ) -> list[WorldPatch]:
        """各方向侧面 → 垂直面片。

        只生成**结构**（BRIDGE/BUILDING/TUNNEL）侧面——地面（MAT_TERRAIN）
        的侧面不生成：地面 span 从世界底延伸到地表，地图边界会产出
        52m 巨墙；地面由平滑连续网格渲染 + Heightfield 碰撞兜底
        （2026-08-12 设计决策）。

        对每个方向：
        - PX/NX（x 常数平面）：按 (gx, z0, z1) 分组，沿 gy 一维合并
        - PY/NY（y 常数平面）：按 (gy, z0, z1) 分组，沿 gx 一维合并
        墙片顶点 = 墙轴 × z 轴的矩形（法线按方向）。
        """
        patches: list[WorldPatch] = []
        w, h, cs = self.w, self.h, self.cs
        spans = self.hf.spans

        for face, mask in side_masks.items():
            ys, si = np.nonzero(mask)
            if ys.size == 0:
                continue

            is_x_plane = face in (FACE_PX, FACE_NX)
            # 收集 (固定坐标, z0, z1) → 合并轴坐标列表；跳过地面
            groups: dict[tuple[int, int, int], list[int]] = {}
            for k in range(len(ys)):
                col = int(ys[k])
                if int(spans[col, int(si[k]), _MAT]) == MAT_TERRAIN:
                    continue  # 地面不生成侧面
                gx, gy = col % w, col // w
                fixed = gx if is_x_plane else gy
                merge_axis = gy if is_x_plane else gx
                z0 = int(spans[col, int(si[k]), _ZMIN])
                z1 = int(spans[col, int(si[k]), _ZMAX])
                groups.setdefault((fixed, z0, z1), []).append(merge_axis)

            for (fixed, z0, z1), axis_list in groups.items():
                z0_w = self.hf._to_z(z0)  # noqa: SLF001
                z1_w = self.hf._to_z(z1)  # noqa: SLF001
                # 反查材质：用首个单元所在列在 [z0,z1) 区间匹配 span
                a0 = axis_list[0]
                col_ref = a0 * w + fixed if is_x_plane else fixed * w + a0
                mat = 0
                for li in range(int(self.hf.count[col_ref])):
                    if (
                        int(spans[col_ref, li, _ZMIN]) == z0
                        and int(spans[col_ref, li, _ZMAX]) == z1
                    ):
                        mat = int(spans[col_ref, li, _MAT])
                        break

                # 沿合并轴排序 → 连续段合并
                axis_list.sort()
                start = axis_list[0]
                prev = axis_list[0]
                for a in axis_list[1:] + [None]:
                    if a is not None and a == prev + 1:
                        prev = a
                        continue
                    # 生成墙片 [start..prev+1]
                    if is_x_plane:
                        fx = fixed * cs if face == FACE_NX else (fixed + 1) * cs
                        y0 = start * cs
                        y1 = (prev + 1) * cs
                        verts = np.array(
                            [
                                [fx, y0, z0_w],
                                [fx, y1, z0_w],
                                [fx, y1, z1_w],
                                [fx, y0, z1_w],
                            ],
                            dtype=np.float32,
                        )
                    else:
                        fy = fixed * cs if face == FACE_NY else (fixed + 1) * cs
                        x0 = start * cs
                        x1 = (prev + 1) * cs
                        verts = np.array(
                            [
                                [x0, fy, z0_w],
                                [x1, fy, z0_w],
                                [x1, fy, z1_w],
                                [x0, fy, z1_w],
                            ],
                            dtype=np.float32,
                        )
                    patches.append(
                        WorldPatch(
                            patch_id=0,  # 占位
                            vertices=verts,
                            normal=np.array(
                                _FACE_NORMALS[face], dtype=np.float32
                            ),
                            mat=mat,
                            face=face,
                        )
                    )
                    if a is not None:
                        start = a
                        prev = a

        for i, p in enumerate(patches):
            p.patch_id = i
        return patches

    # ------------------------------------------------------------------
    # ④ 地面平滑（B2）
    # ------------------------------------------------------------------

    def _smooth_ground(self, patches: list[WorldPatch]) -> None:
        """MAT_TERRAIN 顶面顶点 z ← continuous_ground 双线性插值。"""
        z_map = np.asarray(self.continuous_ground, dtype=np.float32)
        if z_map.shape != (self.h, self.w):
            logger.warning(
                "continuous_ground 形状 %s ≠ 期望 (%d,%d)，跳过平滑",
                z_map.shape, self.h, self.w,
            )
            return
        cs = self.cs
        for p in patches:
            if p.mat != MAT_TERRAIN or p.face != FACE_TOP:
                continue
            for vi in range(len(p.vertices)):
                x, y = float(p.vertices[vi, 0]), float(p.vertices[vi, 1])
                p.vertices[vi, 2] = _bilinear_z(z_map, x, y, cs)

    # ------------------------------------------------------------------
    # ⑤ walkable 筛选（仅顶面）
    # ------------------------------------------------------------------

    def _mark_walkable(self, patches: list[WorldPatch]) -> None:
        """可行走判定：仅顶面可走。

        Dual Contour 的顶面是「暴露表面」（上方无紧贴覆盖），天然排除
        被覆盖的层；TERRAIN 顶面即地面层（si=0，暴露表面），人工结构
        （BRIDGE/BUILDING/TUNNEL）顶面本身可站立。底面/侧面不可走。
        """
        for p in patches:
            if p.face != FACE_TOP:
                p.walkable = False
                continue
            if p.area < self.min_walkable_area:
                p.walkable = False
                continue
            p.walkable = True  # 暴露顶面（地面或结构顶）→ 可走


# ---------------------------------------------------------------------------
# 双线性插值（格中心场 → 任意世界坐标）
# ---------------------------------------------------------------------------


def _bilinear_z(
    z_map: np.ndarray,
    x: float,
    y: float,
    cs: float,
) -> float:
    """在 (x, y) 处对格中心场 z_map 做双线性插值。

    格中心坐标 = (gx+0.5, gy+0.5)*cs → 插值坐标 = (x/cs - 0.5, y/cs - 0.5)。
    与 surface.py 的平滑网格插值一致（跨格连续、水密）。
    """
    map_h, map_w = z_map.shape
    xf = x / cs - 0.5
    yf = y / cs - 0.5
    x0 = int(np.clip(np.floor(xf), 0, map_w - 2))
    y0 = int(np.clip(np.floor(yf), 0, map_h - 2))
    tx = float(np.clip(xf - x0, 0.0, 1.0))
    ty = float(np.clip(yf - y0, 0.0, 1.0))
    z00 = float(z_map[y0, x0])
    z10 = float(z_map[y0, x0 + 1])
    z01 = float(z_map[y0 + 1, x0])
    z11 = float(z_map[y0 + 1, x0 + 1])
    return (
        (z00 * (1.0 - tx) + z10 * tx) * (1.0 - ty)
        + (z01 * (1.0 - tx) + z11 * tx) * ty
    )
