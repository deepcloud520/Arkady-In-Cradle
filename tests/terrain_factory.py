"""测试地形工厂：re-export 生产侧 TerrainFactory + 测试专用扩展入口。

2026-08-12：生产侧已保留 TerrainFactory（Airia/lulilo/terrain/factory.py），
供 UI 开发期选择测试场景；本模块 re-export 之，保持既有测试脚本 import 兼容。
测试专用复杂地形（悬崖/迷宫等）可在此追加（以 _ 前缀命名区分）。
"""
from __future__ import annotations

from Airia.lulilo.terrain.factory import (
    TerrainFactory,
    TerrainFactoryConfig,
)

__all__ = ["TerrainFactoryConfig", "TerrainFactory"]
