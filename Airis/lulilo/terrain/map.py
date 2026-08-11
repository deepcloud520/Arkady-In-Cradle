"""2.5D 栅格地形：TerrainConfig + TerrainMap（Heightfield 逻辑 + 连续地表渲染场）。

对应 AGENTS.md 3.2 / 3.6 节。依赖方向：coordinates → terrain → ...（本模块仅依赖
engine.coordinates 与同包内的 heightmap / heightfield / surface）。

存储（2026-08-10 清理后）：
- heightfield：列式 span 高度场 —— 逻辑/碰撞/寻路唯一数据源
- continuous_ground：连续地表高度场 —— 仅渲染去量化（fBm 原始值，隧道格覆盖）
不再保留 heightmap / layer_map / layer_alt / entry_mask 旧双层存储。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_TERRAIN,
    MAT_TUNNEL,
    Heightfield,
)
from Airis.lulilo.terrain.heightmap import generate_fbm
from Airis.lulilo.terrain.surface import SurfaceExtractor
from Airis.lulilo.terrain.surface_mesh import SurfaceMesh, SurfaceMeshBuilder

logger = logging.getLogger(__name__)


class TerrainConfig(BaseModel):
    """地形配置（Pydantic 配置类，可序列化为 YAML）。"""

    width: int = Field(default=400, ge=2, description="栅格宽（格）")
    height: int = Field(default=400, ge=2, description="栅格高（格）")
    cell_size: float = Field(default=2.0, gt=0.0, description="单格世界尺寸（m/格）")
    seed: int = Field(default=42, description="随机种子")
    amplitude: float = Field(default=10.0, ge=0.0, description="高度图振幅（m）")
    noise_scale: float = Field(default=18.0, gt=0.0, description="噪声频率尺度（格）")
    octaves: int = Field(default=4, ge=1, le=8, description="fBm 八度数")
    max_slope_deg: float = Field(default=30.0, gt=0.0, lt=90.0, description="可行走最大坡度（度）")
    bridge_lift: float = Field(default=2.0, gt=0.0, description="桥面相对桥区地面最大高度的抬升量（m）")
    tunnel_alt: float = Field(default=-6.0, description="隧道底绝对海拔（m）")
    bridges: list[dict[str, Any]] = Field(default_factory=list, description="桥矩形列表（栅格坐标）")
    tunnels: list[dict[str, Any]] = Field(default_factory=list, description="隧道矩形列表（栅格坐标）")


class TerrainMap:
    """2.5D 栅格地形：高度图 + 图层掩码 + 双层逻辑海拔。"""

    def __init__(self, config: TerrainConfig | None = None) -> None:
        self.config = config if config is not None else TerrainConfig()
        cfg = self.config

        # === Heightfield 存储（逻辑/碰撞/寻路唯一数据源） ===
        self.heightfield = Heightfield(
            width=cfg.width,
            height=cfg.height,
            cs=cfg.cell_size,
            ch=1.0,
            z_min=-50.0,
            z_max=200.0,
        )

        # 连续地面高度场（渲染层去量化用，fBm 原始值；隧道格覆盖为隧道底）
        self.continuous_ground: np.ndarray | None = None
        self._z_min: float = 0.0
        self._z_max: float = 0.0
        # SurfaceMesh 懒加载（导航/渲染共用，generate 后首次访问时构建）
        self._surface_mesh: SurfaceMesh | None = None

    @property
    def surface_mesh(self) -> SurfaceMesh:
        """获取 SurfaceMesh（GreedyMesh 面片图），供导航层复用。

        generate() 后首次访问时从 Heightfield 构建并缓存。
        """
        if self._surface_mesh is None:
            self._surface_mesh = SurfaceMeshBuilder(self.heightfield).build()
        return self._surface_mesh

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def generate(self) -> "TerrainMap":
        """生成完整地形：fBm 连续地表 → Heightfield spans（地面/桥面/隧道）。"""
        cfg = self.config
        w, h = cfg.width, cfg.height

        # 1. 连续地表（fBm 生成，渲染去量化用；逻辑层仍走 Heightfield 量化 span）
        ground = generate_fbm(
            w, h, cfg.amplitude, cfg.noise_scale, cfg.octaves, cfg.seed
        )
        self.continuous_ground = ground.copy()

        # 2. 桥/隧道矩形 → 布尔掩码（局部量，不存储）
        bridge_mask = self._expand_rects(cfg.bridges, w, h)
        tunnel_mask = self._expand_rects(cfg.tunnels, w, h)

        # 2.5 隧道格地表覆盖为隧道底（渲染层看到隧道入口）
        self.continuous_ground[tunnel_mask] = cfg.tunnel_alt

        # 3. Heightfield：地面层（所有列）
        for gy in range(h):
            for gx in range(w):
                z_ground = (
                    cfg.tunnel_alt if tunnel_mask[gy, gx] else float(ground[gy, gx])
                )
                self.heightfield.add_span(gx, gy, -50.0, z_ground, mat=MAT_TERRAIN)

        # 4. Heightfield：桥面层（桥面顶 = 桥区地面 max + 抬升 → 贴近地面，桥头自然连通）
        #    2026-08-11：桥不再固定绝对海拔（曾 bridge_alt=14 导致桥面 vs 地面落差
        #    > max_climb → 桥面导航不可达），改为相对桥区最高地面抬升 bridge_lift。
        if bridge_mask.any():
            z_bridge_max_ground = float(ground[bridge_mask].max())
            z_bridge_top = z_bridge_max_ground + cfg.bridge_lift
            z_bridge_bot = z_bridge_top - 1.0  # 1m 厚桥面板
            for gy, gx in np.argwhere(bridge_mask):
                z_ground = float(ground[int(gy), int(gx)])
                # 仅当桥底高于地面时才添加悬空桥面（保留桥下通道）
                if z_bridge_bot > z_ground + 0.5:
                    self.heightfield.add_span(
                        int(gx), int(gy), z_bridge_bot, z_bridge_top, mat=MAT_BRIDGE
                    )

        # 5. Heightfield：隧道（双层：隧道底 + 隧道顶，留 2m 空隙供单位通行）
        for gy, gx in np.argwhere(tunnel_mask):
            z_tun = cfg.tunnel_alt
            # 隧道顶 span（留 2m 间隙：底段 [-50, z_tun)、顶段 [z_tun+2, z_tun+4)）
            self.heightfield.add_span(
                int(gx), int(gy), z_tun + 2.0, z_tun + 4.0, mat=MAT_TUNNEL
            )

        self._z_min = float(ground.min())
        self._z_max = float(ground.max())
        logger.info(
            "地形生成完成: %dx%d 格, 高度范围 [%.1f, %.1f] m, "
            "%d 桥面列 / %d 隧道列",
            w, h, self._z_min, self._z_max,
            int(bridge_mask.sum()), int(tunnel_mask.sum()),
        )
        return self

    @staticmethod
    def _expand_rects(
        rects: list[dict[str, Any]], w: int, h: int
    ) -> np.ndarray:
        """矩形列表（dict: x0/y0/x1/y1 栅格坐标）→ (h, w) 布尔掩码（局部量）。"""
        mask = np.zeros((h, w), dtype=np.bool_)
        for r in rects:
            x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            mask[y0:y1 + 1, x0:x1 + 1] = True
        return mask

    # ------------------------------------------------------------------
    # 渲染数据
    # ------------------------------------------------------------------
    def to_render_data(self) -> dict[str, np.ndarray]:
        """生成渲染网格（使用 Heightfield + SurfaceExtractor）。

        返回（渲染层消费）:
            ground_vertices (N, 9) float32 [x, y, z, r, g, b, nx, ny, nz]
            ground_indices  (M,) uint32
            bridge_vertices (N, 9) float32
            bridge_indices  (M,) uint32
        """
        extractor = SurfaceExtractor(
            self.heightfield,
            ground_z=self.continuous_ground,
        )

        # 全量提取（合并网格）
        rd = extractor.extract()

        if len(rd.vertices) == 0:
            return {
                "ground_vertices": np.zeros((0, 9), dtype=np.float32),
                "ground_indices": np.zeros((0,), dtype=np.uint32),
                "bridge_vertices": np.zeros((0, 9), dtype=np.float32),
                "bridge_indices": np.zeros((0,), dtype=np.uint32),
            }

        # 分离地面与结构：按结构顶点起始偏移（SurfaceExtractor 记录）
        # 2026-08-11：取代旧的 bridge_alt 高度阈值 —— 桥面现贴近地面，
        # 高度阈值不可靠；改按「顶点是否属于结构面片」判定。
        verts = rd.vertices
        idx = rd.indices.reshape(-1, 3)
        struct_start = extractor.struct_offset
        is_struct_v = np.zeros(len(verts), dtype=bool)
        is_struct_v[struct_start:] = True
        is_bridge_tri = np.all(is_struct_v[idx], axis=1)
        ground_tris = idx[~is_bridge_tri]
        bridge_tris = idx[is_bridge_tri]

        # 地面：去重顶点
        gv, gi = self._compact_mesh(verts, ground_tris)
        bv, bi = self._compact_mesh(verts, bridge_tris)

        return {
            "ground_vertices": gv,
            "ground_indices": gi,
            "bridge_vertices": bv,
            "bridge_indices": bi,
        }

    @staticmethod
    def _compact_mesh(
        verts: np.ndarray, tri_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """从大顶点池中提取子集的紧凑表示（去重未使用顶点）。"""
        if len(tri_indices) == 0:
            return (
                np.zeros((0, 9), dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
            )
        used = np.unique(tri_indices.ravel())
        old_to_new = {int(old): new for new, old in enumerate(used)}
        compact_verts = verts[used]
        compact_idx = np.array(
            [old_to_new[int(i)] for i in tri_indices.ravel()], dtype=np.uint32
        )
        return compact_verts, compact_idx
