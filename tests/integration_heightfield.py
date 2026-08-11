"""HeightField 系统端到端集成测试。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Airis.lulilo.terrain.map import TerrainMap, TerrainConfig
from Airis.lulilo.navmesh.baker import Baker
from Airis.lulilo.navmesh.tiles import TileManager

cfg = TerrainConfig(
    width=40, height=40, cell_size=2.0,
    bridges=[{"x0": 10, "y0": 15, "x1": 30, "y1": 16}],
)
tm = TerrainMap(cfg)
tm.generate()
hf = tm.heightfield

# 1. Heightfield
print("=== 1. Heightfield ===")
gc = sum(1 for gy in range(40) for gx in range(40) if hf.span_count(gx, gy) >= 1)
bc = sum(1 for gy in range(40) for gx in range(40) if hf.span_count(gx, gy) >= 2)
print(f"  Ground: {gc}/1600, Bridge: {bc}")
assert gc == 1600
assert bc >= 20

# 2. to_render_data
print()
print("=== 2. to_render_data ===")
rd = tm.to_render_data()
print(f"  Ground: {rd['ground_vertices'].shape[0]}v, {len(rd['ground_indices'])//3}t")
print(f"  Bridge: {rd['bridge_vertices'].shape[0]}v, {len(rd['bridge_indices'])//3}t")
assert rd["ground_vertices"].shape[0] > 0
assert rd["bridge_vertices"].shape[0] > 0

# 3. Baker + A*
print()
print("=== 3. Baker + A* ===")
baker = Baker(hf, {"max_slope_deg": 30.0, "max_climb": 2.0})
graph = baker.bake()
print(f"  Polys: {len(graph.polys)}")
path = graph.astar((5.0, 5.0), (70.0, 70.0))
print(f"  A*: {len(path)} waypoints")
assert len(graph.polys) >= 1
assert len(path) >= 2

# 4. TileManager dirty + rebuild
print()
print("=== 4. TileManager ===")
tile_mgr = TileManager(hf, tile_size=16)
tile_mgr.full_bake()
ng1 = tile_mgr.nav_graph()
print(f"  Before: {len(ng1.polys)} polys")

# 炸掉 (15, 15) 处的桥面
hf.remove_span(15, 15, 12.0, 14.0)
tile_mgr.mark_dirty(14, 14, 17, 17)
tile_mgr.rebuild_dirty(async_=False)
ng2 = tile_mgr.nav_graph()
print(f"  After:  {len(ng2.polys)} polys")
path2 = ng2.astar((5.0, 5.0), (70.0, 70.0))
print(f"  A*: {len(path2)} waypoints")
assert ng1 is not None and ng2 is not None
assert len(path2) >= 2

# 5. solid_at 桥下穿透（用未破坏的桥柱 (20, 15)）
print()
print("=== 5. solid_at bridge underpass ===")
assert not hf.solid_at(20, 15, 7.0), "桥下应为空心"
assert hf.solid_at(20, 15, 13.5), "桥面板内应为实心"
assert hf.solid_at(20, 15, 1.0), "地面应为实心"
print("  OK")

# 6. remove_span 破坏验证（检查 (15, 15) 已被破坏）
print()
print("=== 6. remove_span ===")
assert not hf.solid_at(15, 15, 13.0), "炸后桥面位置应为空心"
assert hf.solid_at(16, 15, 13.0), "邻居桥面应仍实心"
print("  OK")

print()
print("ALL INTEGRATION TESTS PASSED!")
