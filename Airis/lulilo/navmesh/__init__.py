"""engine.navmesh 层（2026-08-11 已废弃）。

旧 NavGraph 多边形烘焙管线已被统一架构替代。废弃约定：文件加 `_` 前缀
（_baker.py / _tiles.py），只删引用不删文件。生产代码应使用：
- terrain.navmesh_builder.NavMeshBuilder / NavGraph（区域生长 + 三角剖分新管线）
- movement.nav.navgraph_astar（A* 寻路）
- movement.pathfinding.GraphPathfinder（寻路服务）

⚠️ 废弃符号以 `_` 前缀标记（_Baker/_NavGraph/_Tile/_TileManager），
本包不导出无前缀旧名（旧名仅在 _baker.py/_tiles.py 内作兼容别名）。
"""

from Airis.lulilo.navmesh._baker import _Baker, _NavGraph
from Airis.lulilo.navmesh._tiles import _Tile, _TileManager

__all__ = [
    "_Baker",
    "_NavGraph",
    "_Tile",
    "_TileManager",
]
