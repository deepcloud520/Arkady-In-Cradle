"""Heightfield（列式 span 高度场）单元测试。

覆盖：addSpan 合并、地面/桥悬空、solid_at 桥下穿透、remove_span 破坏、列满裁剪。
"""
from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.terrain.heightfield import (
    MAT_BRIDGE,
    MAT_TERRAIN,
    MAX_SPANS,
    Heightfield,
)


@pytest.fixture
def hf() -> Heightfield:
    # 2x2 格，cs=2m，ch=1m，垂直范围 [-10, 50]
    return Heightfield(width=2, height=2, cs=2.0, ch=1.0, z_min=-10.0, z_max=50.0)


def test_add_span_merges_overlap(hf: Heightfield) -> None:
    """重叠段应合并为一段（复刻 rcAddSpan 取并集）。"""
    hf.add_span(0, 0, 0.0, 5.0)
    hf.add_span(0, 0, 3.0, 8.0)
    assert hf.span_count(0, 0) == 1
    assert hf.surface_z(0, 0, 0) == pytest.approx(8.0)  # 合并后顶面为 8


def test_add_span_adjacent_not_merged(hf: Heightfield) -> None:
    """不重叠段应各自保留（可表示桥/隧道多层结构）。"""
    hf.add_span(0, 0, 0.0, 2.0, mat=MAT_TERRAIN)      # 地面
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_BRIDGE)    # 高位桥面（悬空）
    assert hf.span_count(0, 0) == 2
    assert hf.surface_z(0, 0, 0) == pytest.approx(2.0)   # 地面顶面
    assert hf.surface_z(0, 0, 1) == pytest.approx(14.0)  # 桥面顶面


def test_bridge_underpass_not_solid(hf: Heightfield) -> None:
    """桥下悬空：桥面板与地面之间的空隙应为空心（单位可穿过）。"""
    hf.add_span(0, 0, 0.0, 2.0, mat=MAT_TERRAIN)
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_BRIDGE)
    assert not hf.solid_at(0, 0, 7.0)   # 桥下空腔
    assert hf.solid_at(0, 0, 13.0)      # 桥面板内
    assert hf.solid_at(0, 0, 1.0)       # 地面段内


def test_remove_span_carves_hole(hf: Heightfield) -> None:
    """破坏：挖掉桥面中段后应留下残余桥面段。"""
    hf.add_span(0, 0, 0.0, 2.0, mat=MAT_TERRAIN)
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_BRIDGE)
    hf.remove_span(0, 0, 12.0, 13.0)   # 炸掉桥面 [12, 13)
    assert hf.span_count(0, 0) == 2     # 地面段 + 残余桥面段
    assert not hf.solid_at(0, 0, 12.5)  # 炸口为空心
    assert hf.solid_at(0, 0, 13.5)      # 残余桥面仍实心
    # 残余桥面段为 [13, 14)，顶面世界 z = 14
    assert hf.surface_z(0, 0, 1) == pytest.approx(14.0)


def test_remove_span_fully_destroys_bridge(hf: Heightfield) -> None:
    """整段炸毁桥面：桥面段应消失，只剩地面。"""
    hf.add_span(0, 0, 0.0, 2.0, mat=MAT_TERRAIN)
    hf.add_span(0, 0, 12.0, 14.0, mat=MAT_BRIDGE)
    hf.remove_span(0, 0, 12.0, 14.0)
    assert hf.span_count(0, 0) == 1
    assert hf.surface_z(0, 0, 1) == -1e9  # 桥面层不存在
    assert not hf.solid_at(0, 0, 13.0)


def test_column_capacity_clipped(hf: Heightfield) -> None:
    """超过 MAX_SPANS 时应裁剪，保留最低层（地面+桥优先）。"""
    for k in range(MAX_SPANS + 4):
        z0 = float(k * 4)
        hf.add_span(0, 0, z0, z0 + 2.0)
    assert hf.span_count(0, 0) == MAX_SPANS
    assert hf.surface_z(0, 0, 0) == pytest.approx(2.0)  # 最低段保留


def test_zero_size_span_ignored(hf: Heightfield) -> None:
    """零尺寸段应被忽略（对应 rcAddSpan early return）。"""
    hf.add_span(0, 0, 0.0, 0.0)
    assert hf.span_count(0, 0) == 0
    assert hf.ground_z(0, 0) == -1e9


def test_tunnel_gap_not_merged(hf: Heightfield) -> None:
    """隧道场景：地面段 + 留空隙 + 隧道顶段，不应合并为单段。"""
    # 模拟隧道：底段 [-50, -6)、空隙 [-6, -4)、顶段 [-4, -2)
    hf.add_span(0, 0, -50.0, -6.0, mat=MAT_TERRAIN)
    hf.add_span(0, 0, -4.0, -2.0, mat=3)  # MAT_TUNNEL
    assert hf.span_count(0, 0) == 2
    assert hf.ground_z(0, 0) == pytest.approx(-6.0)
    assert hf.surface_z(0, 0, 1) == pytest.approx(-2.0)  # 隧道顶
    # 空隙应空心（单位可通行）
    assert not hf.solid_at(0, 0, -5.0)
    # 隧道顶应实心
    assert hf.solid_at(0, 0, -3.0)


def test_solid_at_outside_range(hf: Heightfield) -> None:
    """无实心段列：任何高度都应判为空心。"""
    assert not hf.solid_at(1, 1, 0.0)
    assert hf.ground_z(1, 1) == -1e9
