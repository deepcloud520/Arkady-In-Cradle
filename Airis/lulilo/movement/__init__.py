"""engine.movement 层：移动积分 + 运行时寻路（AGENTS.md 3.3 / Phase 2）。

2026-08-11 统一架构：Pathfinder 改用 SurfaceMesh + A* 面片图，替代旧 NavGraph。
"""

from Airis.lulilo.movement.integrate import (
    compute_speed,
    integrate_armor_movement,
    integrate_projectile_movement,
    integrate_single_armor,
)
from Airis.lulilo.movement.nav import patch_astar, patch_centroid_path
from Airis.lulilo.movement.pathfinding import Pathfinder, funnel

__all__ = [
    "Pathfinder",
    "compute_speed",
    "funnel",
    "integrate_armor_movement",
    "integrate_projectile_movement",
    "integrate_single_armor",
    "patch_astar",
    "patch_centroid_path",
]
