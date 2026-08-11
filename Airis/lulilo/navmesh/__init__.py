"""engine.navmesh 层（2026-08-11 已废弃）。

旧 NavGraph 多边形烘焙管线已被 SurfaceMesh + GreedyMesh 统一架构替代。
baker.py / tiles.py 仅保留供旧测试兼容，生产代码应使用：
- terrain.surface_mesh.SurfaceMesh（导航图）
- movement.nav.patch_astar（A* 寻路）
- movement.pathfinding.Pathfinder（寻路服务）

⚠️ 废弃符号以 `_` 前缀标记（_Baker/_NavGraph/_Tile/_TileManager），
本包不导出无前缀旧名（旧名仅在 baker.py/tiles.py 内作兼容别名）。
"""

from Airis.lulilo.navmesh.baker import _Baker, _NavGraph
from Airis.lulilo.navmesh.tiles import _Tile, _TileManager

__all__ = [
    "_Baker",
    "_NavGraph",
    "_Tile",
    "_TileManager",
]
