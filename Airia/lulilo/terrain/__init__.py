"""engine.terrain 层：地形主类 + fBm 生成器 + 表面网格（AGENTS.md 3.2）。

对外统一 re-export 常用符号，便于 `from Airia.lulilo.terrain import TerrainMap`。
2026-08-11：新增 SurfaceMesh / SurfaceMeshBuilder / SurfacePatch 导出（统一架构）。
2026-08-12：新增 navmesh_builder 导出（区域生长 + 三角剖分新管线骨架，docs/navmesh-region-growth-design.md）。
2026-08-12：TerrainMap 纯化为注入式容器；程序化生成迁至 terrain/factory.py
（TerrainFactory + TerrainFactoryConfig，供 UI 测试场景与测试脚本使用）——
战役地形为人工编辑，generate 纯属测试用途。
"""
from Airia.lulilo.terrain.dual_contour import DualContourBuilder
from Airia.lulilo.terrain.factory import (
    CanyonBridgeConfig,
    TerrainFactory,
    TerrainFactoryConfig,
)
from Airia.lulilo.terrain.heightmap import generate_fbm
from Airia.lulilo.terrain.map import TerrainMap
from Airia.lulilo.terrain.navmesh_builder import (
    NavGraph,
    NavMeshBuilder,
    NavTriangle,
    WalkableCell,
    WalkableConfig,
)
from Airia.lulilo.terrain.surface import WorldMeshRenderer
from Airia.lulilo.terrain.surface_mesh import (
    SurfaceMesh,
    SurfaceMeshBuilder,
    SurfacePatch,
)
from Airia.lulilo.terrain.world_mesh import (
    WorldMesh,
    WorldMeshBuilder,
    WorldPatch,
)

__all__ = [
    "TerrainMap",
    "TerrainFactory",
    "TerrainFactoryConfig",
    "CanyonBridgeConfig",
    "generate_fbm",
    "SurfaceMesh",
    "SurfaceMeshBuilder",
    "SurfacePatch",
    "WorldMesh",
    "WorldMeshBuilder",
    "WorldPatch",
    "DualContourBuilder",
    "WorldMeshRenderer",
    "NavMeshBuilder",
    "NavGraph",
    "NavTriangle",
    "WalkableCell",
    "WalkableConfig",
]
