"""Phase 2 地形系统单元测试（Heightfield 时代，2026-08-10 清理旧接口）。"""
from __future__ import annotations

import numpy as np
import pytest

from Airia.lulilo.terrain import TerrainMap
from Airia.lulilo.coordinates.grid import grid_to_world, world_to_grid
from tests.terrain_factory import TerrainFactory, TerrainFactoryConfig

# 固定小地图，含一座桥和一段隧道，便于精确断言
BRIDGE = {"x0": 40, "y0": 20, "x1": 60, "y1": 22}
TUNNEL = {"x0": 30, "y0": 50, "x1": 70, "y1": 54}
CFG = TerrainFactoryConfig(
    width=100,
    height=100,
    cell_size=2.0,
    seed=7,
    amplitude=8.0,
    noise_scale=20.0,
    octaves=3,
    bridge_lift=2.0,
    tunnel_alt=-6.0,
    bridges=[BRIDGE],
    tunnels=[TUNNEL],
)


@pytest.fixture()
def terrain() -> TerrainMap:
    return TerrainFactory.fbm(CFG)


def test_generate_shapes(terrain: TerrainMap) -> None:
    hf = terrain.heightfield
    assert terrain.continuous_ground is not None
    w, h = CFG.width, CFG.height
    assert terrain.continuous_ground.shape == (h, w)
    assert hf.width == w and hf.height == h
    assert hf.spans.shape == (w * h, 8, 4)
    assert hf.count.shape == (w * h,)


def test_bridge_tunnel_spans(terrain: TerrainMap) -> None:
    """桥/隧道区域在 Heightfield 中生成对应 span（surface_z 验证）。"""
    hf = terrain.heightfield
    # 桥中心格：地面 + 桥面两个 span
    # 桥面顶 = 桥区地面 max + bridge_lift（贴近地面 → 桥头自然连通）
    cg = terrain.continuous_ground
    bx0, by0, bx1, by1 = 40, 20, 60, 22
    bridge_zone_max = float(np.max(cg[by0 : by1 + 1, bx0 : bx1 + 1]))
    expected_top = bridge_zone_max + CFG.bridge_lift
    assert hf.span_count(50, 21) >= 2
    assert hf.surface_z(50, 21, 1) == pytest.approx(expected_top, abs=1.0)
    assert hf.surface_z(50, 21, 1) > hf.surface_z(50, 21, 0)
    # 隧道格：底段 [-50, z_tun) + 2m 空隙 + 顶段 [z_tun+2, z_tun+4)，不合并
    assert hf.span_count(50, 52) == 2
    assert hf.ground_z(50, 52) == pytest.approx(CFG.tunnel_alt)
    # 空隙应空心（单位可通行）
    assert not hf.solid_at(50, 52, CFG.tunnel_alt + 1.0)
    # 隧道顶应实心
    assert hf.solid_at(50, 52, CFG.tunnel_alt + 3.0)
    # 普通区域只有地面一个 span，无第二层段 → NO_SURFACE(负哨兵)
    assert hf.span_count(5, 5) == 1
    assert hf.surface_z(5, 5, 1) < 0.0


def test_continuous_ground(terrain: TerrainMap) -> None:
    """连续地表场：形状正确，隧道格被覆盖为隧道底，其余保留 fBm 连续值。"""
    cg = terrain.continuous_ground
    assert cg is not None
    w, h = CFG.width, CFG.height
    assert cg.shape == (h, w)
    # 隧道格覆盖为隧道底（TUNNEL = [50:55, 30:71]）
    assert cg[52, 50] == pytest.approx(CFG.tunnel_alt)
    # 非隧道格保留 fBm 连续值
    assert np.isfinite(cg).all()
    assert cg[5, 5] == pytest.approx(terrain.heightfield.ground_z(5, 5), abs=1.0)


def test_bridge_tunnel_ground_z_consistency(terrain: TerrainMap) -> None:
    """ground_z 与 surface_z(layer=0) 语义一致（地面站位高度）。"""
    hf = terrain.heightfield
    for gx, gy in [(50, 21), (50, 52), (5, 5)]:
        assert hf.ground_z(gx, gy) == pytest.approx(hf.surface_z(gx, gy, 0))


def test_world_grid_roundtrip(terrain: TerrainMap) -> None:
    cs = CFG.cell_size
    # 世界坐标 → 栅格 → 栅格中心世界坐标，应落在同格
    for wx, wy in [(13.0, 27.0), (0.0, 0.0), (199.0 * cs, 199.0 * cs)]:
        gx, gy = world_to_grid(wx, wy, cs)
        cx, cy = grid_to_world(gx, gy, cs)
        assert world_to_grid(cx, cy, cs) == (gx, gy)


def test_to_render_data(terrain: TerrainMap) -> None:
    data = terrain.to_render_data()
    gv = data["ground_vertices"]
    gi = data["ground_indices"]
    # 顶点格式为 (N, 9) float32 [x, y, z, r, g, b, nx, ny, nz]
    assert gv.ndim == 2 and gv.shape[1] == 9
    assert gv.dtype == np.float32
    assert len(gv) > 0, "应有地面顶点"
    # 索引格式为 (M,) uint32，M 为 3 的倍数
    assert gi.ndim == 1
    assert gi.dtype == np.uint32
    assert len(gi) > 0 and len(gi) % 3 == 0, "索引应为三角形列表"
    # 桥面网格格式正确（可能为空）
    assert data["bridge_vertices"].ndim == 2
    assert data["bridge_vertices"].shape[1] == 9
    assert len(data["bridge_vertices"]) > 0, "应有桥面顶点"
