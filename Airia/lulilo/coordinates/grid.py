"""栅格坐标转换：世界坐标 ↔ 栅格坐标（engine.coordinates 层，被 terrain 依赖）。

依赖方向：core → coordinates → terrain → ...（本模块只依赖 numpy，属于最底层）。
"""
from __future__ import annotations

import numpy as np


def world_to_grid(x: float, y: float, cell_size: float) -> tuple[int, int]:
    """世界坐标 → 栅格索引（向下取整）。

    Args:
        x: 世界 X 坐标。
        y: 世界 Y 坐标。
        cell_size: 单格世界尺寸（m/格）。

    Returns:
        栅格索引 (gx, gy)。
    """
    gx = int(np.floor(x / cell_size))
    gy = int(np.floor(y / cell_size))
    return gx, gy


def grid_to_world(gx: int, gy: int, cell_size: float) -> tuple[float, float]:
    """栅格中心的世界坐标。

    Args:
        gx: 栅格 X 索引。
        gy: 栅格 Y 索引。
        cell_size: 单格世界尺寸。

    Returns:
        世界坐标 (x, y)，位于格子中心。
    """
    return (gx + 0.5) * cell_size, (gy + 0.5) * cell_size


def grid_corner_to_world(gx: int, gy: int, cell_size: float) -> tuple[float, float]:
    """栅格左上角（网格顶点）的世界坐标，用于渲染网格顶点采样。

    Args:
        gx: 栅格 X 索引。
        gy: 栅格 Y 索引。
        cell_size: 单格世界尺寸。

    Returns:
        世界坐标 (x, y)，位于格子左上角。
    """
    return gx * cell_size, gy * cell_size
