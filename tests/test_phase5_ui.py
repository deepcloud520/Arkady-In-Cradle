"""Phase 5b/5c 无头测试：picking 投影 / 拾取 + WorldAdapter 统计聚合。

GUI 相关（HUDOverlay 渲染 / 面板显示）需 OpenGL 上下文，不在无头测试覆盖；
冒烟验证见 tests/smoke_gui.py。
"""
from __future__ import annotations

import glm
import numpy as np
import pytest

from Airis.render.adapter import WorldAdapter
from Airis.render.picking import pick_nearest, project_world


def _make_adapter() -> WorldAdapter:
    """构造一个最小 WorldAdapter（3 单位：友军/敌军/中立 + 后勤）。"""
    n = 8
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[0] = (0.0, 0.0, 0.0)   # 友军 @ 原点
    pos[1] = (2.0, 0.0, 0.0)   # 敌军 @ x=2
    pos[2] = (0.0, 2.0, 0.0)   # 中立 @ y=2
    active = np.array([True, True, True, False, False, False, False, False])
    side = np.array([0, 1, 2, 0, 0, 0, 0, 0], dtype=np.uint8)
    ammo_reserve = np.array([36.0, 20.0, 40.0, 0, 0, 0, 0, 0], dtype=np.float32)
    ammo_capacity = np.array([40.0, 40.0, 40.0, 1, 1, 1, 1, 1], dtype=np.float32)
    fuel_reserve = np.array([90.0, 60.0, 100.0, 0, 0, 0, 0, 0], dtype=np.float32)
    fuel_capacity = np.array([100.0, 100.0, 100.0, 1, 1, 1, 1, 1], dtype=np.float32)
    return WorldAdapter(
        armor_pos=pos,
        armor_yaw=np.zeros(n, dtype=np.float32),
        armor_pitch=np.zeros(n, dtype=np.float32),
        armor_roll=np.zeros(n, dtype=np.float32),
        armor_health=np.full(n, 100.0, dtype=np.float32),
        armor_active=active,
        armor_confidence=np.ones(n, dtype=np.float32),
        armor_status=np.zeros(n, dtype=np.uint8),
        armor_side=side,
        armor_damaged_parts=np.zeros((n, 5), dtype=np.bool_),
        box_parent=np.zeros(0, dtype=np.int32),
        box_part=np.zeros(0, dtype=np.int32),
        box_active=np.zeros(0, dtype=np.bool_),
        ammo_reserve=ammo_reserve,
        ammo_capacity=ammo_capacity,
        fuel_reserve=fuel_reserve,
        fuel_capacity=fuel_capacity,
        sim_time=123.0,
        tick_count=7380,
    )


# ---------------------------------------------------------------------------
# project_world
# ---------------------------------------------------------------------------


def test_project_identity_center():
    mvp = glm.mat4(1.0)
    world = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    screen, ndc_z, clip_w, visible = project_world(world, mvp, 800, 600)
    assert screen.shape == (1, 2)
    assert screen[0, 0] == pytest.approx(400.0)
    assert screen[0, 1] == pytest.approx(300.0)
    assert ndc_z[0] == pytest.approx(0.0)
    assert clip_w[0] == pytest.approx(1.0)
    assert bool(visible[0])


def test_project_identity_corner():
    # 世界 (1,0,0) → NDC (1,0,0) → 屏幕最右中心
    mvp = glm.mat4(1.0)
    world = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    screen, _z, _w, visible = project_world(world, mvp, 800, 600)
    assert screen[0, 0] == pytest.approx(800.0)
    assert screen[0, 1] == pytest.approx(300.0)
    assert bool(visible[0])


def test_project_empty():
    mvp = glm.mat4(1.0)
    screen, ndc_z, clip_w, visible = project_world(
        np.zeros((0, 3), dtype=np.float32), mvp, 800, 600
    )
    assert screen.shape == (0, 2)
    assert visible.shape == (0,)


def test_project_matches_glm():
    """真实相机 MVP：project_world 必须与 glm 逐点一致（覆盖 to_list 转置 bug）。"""
    from Airis.render.camera import OrbitCamera

    cam = OrbitCamera(target=(256.0, 256.0, 0.0), distance=200.0, yaw=45.0, pitch=40.0)
    mvp = cam.mvp_matrix(1.6)
    world = np.array(
        [[238.0, 238.0, -8.0], [60.0, 60.0, 0.0], [102.0, 102.0, 3.0]],
        dtype=np.float32,
    )
    screen, ndc_z, clip_w, visible = project_world(world, mvp, 1280, 800)
    for k, (x, y, z) in enumerate(world):
        r = mvp * glm.vec4(x, y, z, 1.0)
        wgt = r.w
        ndc = glm.vec3(r.x / wgt, r.y / wgt, r.z / wgt)
        ex = (ndc.x * 0.5 + 0.5) * 1280
        ey = (1.0 - (ndc.y * 0.5 + 0.5)) * 800
        assert screen[k, 0] == pytest.approx(ex, abs=0.01)
        assert screen[k, 1] == pytest.approx(ey, abs=0.01)
        assert clip_w[k] == pytest.approx(wgt, abs=0.01)
        assert bool(visible[k]) == (wgt > 0)


def test_project_distinct_positions():
    """不同世界位置必须投影到不同屏幕点（回归：曾全部挤到同一位置）。"""
    from Airis.render.camera import OrbitCamera

    cam = OrbitCamera(target=(256.0, 256.0, 0.0), distance=200.0, yaw=45.0, pitch=40.0)
    mvp = cam.mvp_matrix(1.6)
    world = np.array(
        [[238.0, 238.0, -8.0], [196.0, 196.0, -6.0], [102.0, 102.0, 3.0]],
        dtype=np.float32,
    )
    screen, _z, _w, visible = project_world(world, mvp, 1280, 800)
    d01 = np.hypot(screen[0, 0] - screen[1, 0], screen[0, 1] - screen[1, 1])
    d02 = np.hypot(screen[0, 0] - screen[2, 0], screen[0, 1] - screen[2, 1])
    assert d01 > 5.0
    assert d02 > 5.0


# ---------------------------------------------------------------------------
# pick_nearest
# ---------------------------------------------------------------------------


def test_pick_hits_nearest():
    adapter = _make_adapter()
    mvp = glm.mat4(1.0)  # identity：世界 x 范围 [-1,1] 在屏幕 [0,800] 内
    # 单位 0 @ 原点 → 屏幕中心 (400,300)；单位 1 @ x=2 → NDC x=2 → 屏幕外（不可见）
    idx = pick_nearest(400, 300, adapter, mvp, 800, 600, max_px=30.0)
    assert idx == 0


def test_pick_blank_returns_minus_one():
    adapter = _make_adapter()
    mvp = glm.mat4(1.0)
    # 屏幕左上角 (0,0)：距离原点 > max_px
    idx = pick_nearest(5, 5, adapter, mvp, 800, 600, max_px=30.0)
    assert idx == -1


def test_pick_no_adapter():
    mvp = glm.mat4(1.0)
    assert pick_nearest(400, 300, None, mvp, 800, 600) == -1


# ---------------------------------------------------------------------------
# WorldAdapter 统计
# ---------------------------------------------------------------------------


def test_adapter_get_stats():
    adapter = _make_adapter()
    s = adapter.get_stats()
    assert s["n_ally"] == 1
    assert s["n_enemy"] == 1
    assert s["n_neutral"] == 1
    # 弹药均值（仅友军+敌军，排除中立）：(36/40 + 20/40)/2 = 0.7
    assert s["avg_ammo"] == pytest.approx(70.0)
    # 燃料均值：(90/100 + 60/100)/2 = 0.75
    assert s["avg_fuel"] == pytest.approx(75.0)
    assert s["sim_time"] == pytest.approx(123.0)


def test_adapter_unit_name_and_side():
    adapter = _make_adapter()
    assert adapter.unit_name(0) == "T-90 #00"
    assert adapter.unit_name(1) == "T-72B3 #01"
    assert adapter.unit_name(2) == "载具 #02"
    assert adapter.side_name(0) == "友军"
    assert adapter.side_name(1) == "敌军"


def test_adapter_ammo_fuel_at():
    adapter = _make_adapter()
    assert adapter.ammo_at(0) == pytest.approx((36.0, 40.0))
    assert adapter.fuel_at(0) == pytest.approx((90.0, 100.0))
    assert adapter.ammo_at(99) is None  # 越界
