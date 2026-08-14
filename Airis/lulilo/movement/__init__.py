"""engine.movement 层：移动积分 + 运行时寻路（AGENTS.md 3.3 / Phase 2）。

2026-08-12 切换管线：默认寻路 = 三角形 NavGraph（GraphPathfinder + navgraph_astar）。
旧 SurfaceMesh 面片寻路（_Pathfinder/_patch_astar/_funnel）已废弃（`_` 前缀），
仅导出 `_` 版（旧名从包级导入会 ImportError；模块内保留兼容别名）。
"""

from Airis.lulilo.movement.integrate import (
    build_armor_movement_hook,
    compute_speed,
    integrate_armor_movement,
    integrate_armor_movement_3d,
    integrate_projectile_movement,
    integrate_single_armor,
    resolve_contact_mesh_ids,
    resolve_mesh_ids,
    solve_suspension,
)
from Airis.lulilo.movement.nav import (
    _patch_astar,
    _patch_centroid_path,
    navgraph_astar,
)
from Airis.lulilo.movement.pathfinding import (
    GraphPathfinder,
    _Pathfinder,
    _funnel,
    funnel_graph,
)

__all__ = [
    "GraphPathfinder",
    "funnel_graph",
    "navgraph_astar",
    "compute_speed",
    "integrate_armor_movement",
    "integrate_armor_movement_3d",
    "integrate_projectile_movement",
    "integrate_single_armor",
    "solve_suspension",
    "resolve_mesh_ids",
    "resolve_contact_mesh_ids",
    "build_armor_movement_hook",
    "_Pathfinder",
    "_funnel",
    "_patch_astar",
    "_patch_centroid_path",
]
