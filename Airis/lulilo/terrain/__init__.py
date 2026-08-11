"""engine.terrain 层：地形主类 + fBm 生成器 + 表面网格（AGENTS.md 3.2）。

对外统一 re-export 常用符号，便于 `from Airis.lulilo.terrain import TerrainMap`。
2026-08-11：新增 SurfaceMesh / SurfaceMeshBuilder / SurfacePatch 导出（统一架构）。
"""
from Airis.lulilo.terrain.heightmap import generate_fbm
from Airis.lulilo.terrain.map import TerrainConfig, TerrainMap
from Airis.lulilo.terrain.surface_mesh import (
    SurfaceMesh,
    SurfaceMeshBuilder,
    SurfacePatch,
)

__all__ = [
    "TerrainConfig",
    "TerrainMap",
    "generate_fbm",
    "SurfaceMesh",
    "SurfaceMeshBuilder",
    "SurfacePatch",
]
