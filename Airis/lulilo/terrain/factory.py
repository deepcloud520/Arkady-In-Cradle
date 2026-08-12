"""程序化地形生成工厂：开发/测试阶段的测试场景提供者。

2026-08-12：战役地形为人工编辑（不走 fBm），但项目仍处于测试阶段 ——
UI 设计/调试需要可选测试场景，故生产侧保留 TerrainFactory 供 UI 与测试脚本使用。
TerrainMap 已纯化为注入式容器（Airis/lulilo/terrain/map.py），本模块负责
「TerrainFactoryConfig → 已建好的 TerrainMap」。

场景：
    from Airis.lulilo.terrain.factory import (
        TerrainFactory,
        TerrainFactoryConfig,
    )

    terrain = TerrainFactory.default()                      # 400×400 + 3 座演示桥
    terrain = TerrainFactory.flat(100, 100, z=2.0)          # 确定性平地
    terrain = TerrainFactory.fbm(TerrainFactoryConfig(...)) # 自定义 fBm + 桥/隧

tests/terrain_factory.py 为兼容 re-export 层（并留测试专用复杂地形的扩展口）。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_BUILDING,
    MAT_TUNNEL,
    Heightfield,
)
from Airis.lulilo.terrain.heightmap import generate_fbm
from Airis.lulilo.terrain.map import TerrainMap


class CanyonBridgeConfig(BaseModel):
    """峡谷长桥配置：山体被切开（峡谷）+ 桥墩支撑（现实山间桥梁）。

    桥区 [x0,y0,x1,y1]（栅格坐标），桥面横跨全区、锚定两端**桥台**（未挖）
    最高点 + bridge_lift。中间 canyon_frac 长度挖深 canyon_depth 成峡谷
    （山体被切开）；峡谷段每隔 pier_interval 格立 pier_width 宽桥墩
    （从谷底到桥底）。

    2026-08-12 新增：贴合生产环境的现实地形（长桥 + 峡谷 + 桥墩）。
    """

    x0: int
    y0: int
    x1: int
    y1: int
    canyon_frac: float = Field(
        default=0.6, gt=0.0, lt=1.0, description="峡谷占桥长比例（中间段）"
    )
    canyon_depth: float = Field(
        default=14.0, gt=0.0, description="峡谷挖深（m）"
    )
    pier_interval: int = Field(
        default=16, ge=2, description="桥墩间隔（格）"
    )
    pier_width: int = Field(
        default=2, ge=1, le=6, description="桥墩截面宽（格）"
    )


class TerrainFactoryConfig(BaseModel):
    """TerrainFactory 的生成配置（程序化测试场景参数）。

    2026-08-12 命名：战役地形为人工编辑后不存在独立「地形配置」，此配置只
    描述工厂如何生成测试场景，故命名为 TerrainFactoryConfig。
    """

    width: int = Field(default=400, ge=2, description="栅格宽（格）")
    height: int = Field(default=400, ge=2, description="栅格高（格）")
    cell_size: float = Field(default=2.0, gt=0.0, description="单格世界尺寸（m/格）")
    seed: int = Field(default=42, description="随机种子")
    amplitude: float = Field(default=10.0, ge=0.0, description="高度图振幅（m）")
    noise_scale: float = Field(default=18.0, gt=0.0, description="噪声频率尺度（格）")
    octaves: int = Field(default=4, ge=1, le=8, description="fBm 八度数")
    max_slope_deg: float = Field(
        default=30.0, gt=0.0, lt=90.0, description="可行走最大坡度（度）"
    )
    bridge_lift: float = Field(
        default=2.0, gt=0.0, description="桥面相对桥区地面最大高度的抬升量（m）"
    )
    tunnel_alt: float = Field(default=-6.0, description="隧道底绝对海拔（m）")
    bridges: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"x0": 100, "y0": 100, "x1": 160, "y1": 104},
            {"x0": 240, "y0": 300, "x1": 244, "y1": 360},
            {"x0": 60, "y0": 220, "x1": 120, "y1": 224},
        ],
        description="桥矩形列表（栅格坐标）；默认内置 3 座演示桥（方便查看）",
    )
    tunnels: list[dict[str, Any]] = Field(
        default_factory=list, description="隧道矩形列表（栅格坐标）"
    )
    canyon_bridge: CanyonBridgeConfig | None = Field(
        default=None, description="峡谷长桥（可选）：山体切开 + 桥墩支撑"
    )


class TerrainFactory:
    """程序化地形构造器：确定性、可控的测试场景（供 UI 测试场景与测试脚本）。"""

    @staticmethod
    def flat(w: int = 64, h: int = 64, z: float = 0.0, cs: float = 2.0) -> TerrainMap:
        """纯平地（确定性；适合移动/寻路/渲染的基线测试）。"""
        hf = Heightfield(width=w, height=h, cs=cs, ch=1.0, z_min=-50.0, z_max=200.0)
        hf.set_ground_layer(np.full((h, w), z, dtype=np.float32))
        return TerrainMap(hf)

    @staticmethod
    def fbm(cfg: TerrainFactoryConfig | None = None) -> TerrainMap:
        """fBm 随机起伏 + 矩形桥/隧（原 TerrainMap.generate() 全逻辑）。"""
        cfg = cfg if cfg is not None else TerrainFactoryConfig()
        w, h = cfg.width, cfg.height
        hf = Heightfield(
            width=w, height=h, cs=cfg.cell_size, ch=1.0, z_min=-50.0, z_max=200.0
        )

        # 1. 连续地表（fBm） + 桥/隧道矩形布尔掩码
        ground = generate_fbm(
            w, h, cfg.amplitude, cfg.noise_scale, cfg.octaves, cfg.seed
        )
        bridge_mask = _expand_rects(cfg.bridges, w, h)
        tunnel_mask = _expand_rects(cfg.tunnels, w, h)

        # 1.5 峡谷长桥：先挖峡谷（山体切开，原地修改 ground）
        canyon_geo = None
        if cfg.canyon_bridge is not None:
            canyon_geo = _carve_canyon(
                ground, cfg.canyon_bridge, cfg.bridge_lift
            )

        # 2. 连续地表场（渲染去量化用）；隧道格覆盖为隧道底
        continuous = ground.copy()
        continuous[tunnel_mask] = cfg.tunnel_alt

        # 3. Heightfield 地面层（向量化批量写入；隧道格地表覆盖为隧道底）
        ground_layer = np.where(tunnel_mask, cfg.tunnel_alt, ground).astype(np.float32)
        hf.set_ground_layer(ground_layer)

        # 4. 桥面层（相对桥区最高地面抬升 bridge_lift → 贴近地面，桥头自然连通）
        if bridge_mask.any():
            z_bridge_max_ground = float(ground[bridge_mask].max())
            z_bridge_top = z_bridge_max_ground + cfg.bridge_lift
            z_bridge_bot = z_bridge_top - 1.0  # 1m 厚桥面板
            for gy, gx in np.argwhere(bridge_mask):
                z_ground = float(ground[int(gy), int(gx)])
                # 仅当桥底高于地面时才添加悬空桥面（保留桥下通道）
                if z_bridge_bot > z_ground + 0.5:
                    hf.add_span(
                        int(gx), int(gy), z_bridge_bot, z_bridge_top, mat=MAT_BRIDGE
                    )

        # 4.5 峡谷长桥：架桥面板（横跨全区）+ 桥墩（峡谷段支撑）
        if canyon_geo is not None:
            _add_canyon_bridge_spans(hf, ground, cfg.canyon_bridge, canyon_geo)

        # 5. 隧道（双层：隧道底 + 隧道顶，留 2m 空隙供单位通行）
        for gy, gx in np.argwhere(tunnel_mask):
            z_tun = cfg.tunnel_alt
            hf.add_span(int(gx), int(gy), z_tun + 2.0, z_tun + 4.0, mat=MAT_TUNNEL)

        return TerrainMap(hf, continuous_ground=continuous)

    @staticmethod
    def default() -> TerrainMap:
        """默认演示地形（400×400 + 3 座演示桥，等价旧 TerrainMap().generate()）。"""
        return TerrainFactory.fbm(TerrainFactoryConfig())

    @staticmethod
    def default_config() -> TerrainFactoryConfig:
        """与 default() 对应的默认生成配置（供脚本读取桥/隧矩形等参数）。"""
        return TerrainFactoryConfig()


def _expand_rects(rects: list[dict[str, Any]], w: int, h: int) -> np.ndarray:
    """矩形列表（dict: x0/y0/x1/y1 栅格坐标）→ (h, w) 布尔掩码。"""
    mask = np.zeros((h, w), dtype=np.bool_)
    for r in rects:
        x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


# ---------------------------------------------------------------------------
# 峡谷长桥（山体切开 + 桥墩）—— 2026-08-12
# ---------------------------------------------------------------------------
# 现实山间桥梁：桥面横跨被切开的山体（峡谷），桥墩立在谷底支撑。
# 实现：
#   1. _carve_canyon：桥区中间 canyon_frac 长度挖深 canyon_depth（山体切开），
#      两端保留原高度作桥台；桥面锚定 = 桥台最高点 + bridge_lift
#   2. _add_canyon_bridge_spans：桥面板横跨全区 + 峡谷段每隔 pier_interval 格
#      立 pier_width 宽桥墩（[谷底, 桥底]，DualContour 自动提取桥墩侧面）


def _carve_canyon(
    ground: np.ndarray,
    cb: CanyonBridgeConfig,
    bridge_lift: float,
) -> tuple[float, float, np.ndarray, bool, int, int, int, int, int, int]:
    """挖峡谷（原地修改 ground），返回架桥所需的几何参数。

    Returns:
        (z_top, z_bot, canyon_mask, horizontal,
         x0, y0, x1, y1, c_lo, c_hi)
        z_top: 桥面顶世界高度；z_bot: 桥底（面板厚 1m）；
        canyon_mask: 峡谷区域掩码 (h, w)；
        horizontal: 长轴是否横向（否则纵向）；
        (x0,y0,x1,y1): 桥区裁剪后范围；[c_lo, c_hi]: 峡谷段坐标范围（长轴）。
    """
    h, w = ground.shape
    x0, y0 = max(0, cb.x0), max(0, cb.y0)
    x1, y1 = min(w - 1, cb.x1), min(h - 1, cb.y1)
    if x1 <= x0 or y1 <= y0:
        return -1.0, -1.0, np.zeros((h, w), dtype=bool), True, x0, y0, x1, y1, x0, x0

    horizontal = (x1 - x0) >= (y1 - y0)

    # 峡谷段范围（长轴中间 canyon_frac）
    if horizontal:
        total = x1 - x0 + 1
        canyon = max(1, int(round(total * cb.canyon_frac)))
        side = (total - canyon) // 2
        c_lo, c_hi = x0 + side, x0 + side + canyon - 1
        abut_l = ground[y0 : y1 + 1, x0:c_lo]
        abut_r = ground[y0 : y1 + 1, c_hi + 1 : x1 + 1]
    else:
        total = y1 - y0 + 1
        canyon = max(1, int(round(total * cb.canyon_frac)))
        side = (total - canyon) // 2
        c_lo, c_hi = y0 + side, y0 + side + canyon - 1
        abut_l = ground[y0:c_lo, x0 : x1 + 1]
        abut_r = ground[c_hi + 1 : y1 + 1, x0 : x1 + 1]

    # 桥面锚定：两端桥台（未挖）最高点 + lift
    z_max = -1e9
    for a in (abut_l, abut_r):
        if a.size:
            z_max = max(z_max, float(np.max(a)))
    z_top = z_max + bridge_lift
    z_bot = z_top - 1.0

    # 挖峡谷（山体切开）
    canyon_mask = np.zeros((h, w), dtype=bool)
    if horizontal:
        canyon_mask[y0 : y1 + 1, c_lo : c_hi + 1] = True
    else:
        canyon_mask[c_lo : c_hi + 1, x0 : x1 + 1] = True
    ground[canyon_mask] -= cb.canyon_depth

    return (
        z_top, z_bot, canyon_mask, horizontal,
        x0, y0, x1, y1, c_lo, c_hi,
    )


def _add_canyon_bridge_spans(
    hf: Heightfield,
    ground: np.ndarray,
    cb: CanyonBridgeConfig,
    geo: tuple,
) -> None:
    """架桥面板（横跨全区）+ 桥墩（峡谷段每隔 pier_interval 格）。"""
    (
        z_top, z_bot, _canyon_mask, horizontal,
        x0, y0, x1, y1, c_lo, c_hi,
    ) = geo
    pw = cb.pier_width

    # 桥面板：整个桥区（含桥台段，桥台段贴近地面自然衔接）
    for gy in range(y0, y1 + 1):
        for gx in range(x0, x1 + 1):
            z_ground = float(ground[gy, gx])
            if z_bot > z_ground + 0.5:
                hf.add_span(gx, gy, z_bot, z_top, mat=MAT_BRIDGE)

    # 桥墩：峡谷段，沿长轴每隔 pier_interval 格立一个 pier_width 宽柱。
    # 🔥 材质必须用 MAT_BUILDING（混凝土桥墩）：若用 MAT_BRIDGE 会与桥面板
    # [z_bot,z_top] 相邻合并成实心堤坝（add_span 同 mat 相邻合并）——桥变坝！
    if horizontal:
        pier_x = c_lo
        while pier_x <= c_hi:
            gx_lo = pier_x
            gx_hi = min(c_hi, pier_x + pw - 1)
            gy_lo = y0 + max(0, (y1 - y0 + 1 - pw) // 2)
            gy_hi = min(y1, gy_lo + pw - 1)
            for gy in range(gy_lo, gy_hi + 1):
                for gx in range(gx_lo, gx_hi + 1):
                    z_ground = float(ground[gy, gx])
                    if z_bot > z_ground + 0.5:
                        hf.add_span(
                            gx, gy, z_ground, z_bot, mat=MAT_BUILDING
                        )
            pier_x += cb.pier_interval
    else:
        pier_y = c_lo
        while pier_y <= c_hi:
            gy_lo = pier_y
            gy_hi = min(c_hi, pier_y + pw - 1)
            gx_lo = x0 + max(0, (x1 - x0 + 1 - pw) // 2)
            gx_hi = min(x1, gx_lo + pw - 1)
            for gy in range(gy_lo, gy_hi + 1):
                for gx in range(gx_lo, gx_hi + 1):
                    z_ground = float(ground[gy, gx])
                    if z_bot > z_ground + 0.5:
                        hf.add_span(
                            gx, gy, z_ground, z_bot, mat=MAT_BUILDING
                        )
            pier_y += cb.pier_interval
