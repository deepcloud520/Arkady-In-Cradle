"""Phase 5b/5c 无头测试：picking 投影 / 拾取 + WorldAdapter 统计聚合。

GUI 相关（HUDOverlay 渲染 / 面板显示）需 OpenGL 上下文，不在无头测试覆盖；
冒烟验证见 tests/smoke_gui.py。
"""
from __future__ import annotations

import glm
import numpy as np
import pytest

from Airia.render.adapter import WorldAdapter
from Airia.render.picking import pick_nearest, project_world


def _make_adapter(
    role: np.ndarray | None = None,
    aircraft: dict | None = None,
) -> WorldAdapter:
    """构造一个最小 WorldAdapter（3 单位：北约/华约/中立 + 后勤）。

    role: 可选 UnitRole 数组（None → 全 0=坦克，型号走阵营默认）。
    aircraft: 可选空中视图 dict（pos/active/side/...；None → 未装配）。
    """
    n = 8
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[0] = (0.0, 0.0, 0.0)   # 北约 @ 原点
    pos[1] = (2.0, 0.0, 0.0)   # 华约 @ x=2
    pos[2] = (0.0, 2.0, 0.0)   # 中立 @ y=2
    active = np.array([True, True, True, False, False, False, False, False])
    side = np.array([0, 1, 2, 0, 0, 0, 0, 0], dtype=np.uint8)
    if role is None:
        role = np.zeros(n, dtype=np.uint8)
    ammo_reserve = np.array([36.0, 20.0, 40.0, 0, 0, 0, 0, 0], dtype=np.float32)
    ammo_capacity = np.array([40.0, 40.0, 40.0, 1, 1, 1, 1, 1], dtype=np.float32)
    fuel_reserve = np.array([90.0, 60.0, 100.0, 0, 0, 0, 0, 0], dtype=np.float32)
    fuel_capacity = np.array([100.0, 100.0, 100.0, 1, 1, 1, 1, 1], dtype=np.float32)

    ac = aircraft or {}
    k = ac.get("k", 0)
    apos = ac.get("pos", np.zeros((k, 3), dtype=np.float32))
    aactive = ac.get("active", np.zeros(k, dtype=np.bool_))
    aside = ac.get("side", np.zeros(k, dtype=np.uint8))
    avel = ac.get("vel", np.zeros((k, 3), dtype=np.float32))
    ahealth = ac.get("health", np.full(k, 100.0, dtype=np.float32))
    afuel = ac.get("fuel", np.zeros(k, dtype=np.float32))
    aaam = ac.get("aam", np.zeros(k, dtype=np.uint8))
    aagm = ac.get("agm", np.zeros(k, dtype=np.uint8))
    astate = ac.get("state", np.zeros(k, dtype=np.uint8))
    arole = ac.get("role", np.zeros(k, dtype=np.uint8))
    afid = ac.get("formation_id", np.full(k, -1, dtype=np.int32))
    box_armor_mm = ac.get("box_armor_mm", np.zeros(0, dtype=np.float32))
    box_parent = ac.get("box_parent", np.zeros(0, dtype=np.int32))
    box_part = ac.get("box_part", np.zeros(0, dtype=np.int32))
    box_active = ac.get("box_active", np.zeros(0, dtype=np.bool_))

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
        armor_role=role,
        armor_damaged_parts=np.zeros((n, 5), dtype=np.bool_),
        box_parent=box_parent,
        box_part=box_part,
        box_active=box_active,
        box_armor_mm=box_armor_mm,
        ammo_reserve=ammo_reserve,
        ammo_capacity=ammo_capacity,
        fuel_reserve=fuel_reserve,
        fuel_capacity=fuel_capacity,
        aircraft_pos=apos,
        aircraft_active=aactive,
        aircraft_side=aside,
        aircraft_vel=avel,
        aircraft_health=ahealth,
        aircraft_fuel=afuel,
        aircraft_aam=aaam,
        aircraft_agm=aagm,
        aircraft_state=astate,
        aircraft_role=arole,
        aircraft_formation_id=afid,
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
    from Airia.render.camera import OrbitCamera

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
    from Airia.render.camera import OrbitCamera

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
    assert adapter.unit_name(0) == "M1A2 #00"
    assert adapter.unit_name(1) == "T-14 #01"
    assert adapter.unit_name(2) == "载具 #02"
    assert adapter.side_name(0) == "北约"
    assert adapter.side_name(1) == "华约"


def test_side_model_mapper_defaults():
    """型号命名唯一数据源（entities.side.SIDE_MODEL）：GUI 与叙事层共用。"""
    from Airia.lulilo.entities.side import SIDE_MODEL, Side, SideModelMapper, side_model

    assert SideModelMapper()(int(Side.NATO)) == "M1A2"
    assert SideModelMapper()(int(Side.PACT)) == "T-14"
    assert SideModelMapper()(int(Side.NEUTRAL)) == "载具"
    assert SideModelMapper()(99) == "载具"  # 未知阵营兜底
    # 便捷函数与默认表一致
    assert side_model(int(Side.NATO)) == SIDE_MODEL[Side.NATO]
    # 自定义表注入
    assert SideModelMapper({1: "T-80BVM"})(1) == "T-80BVM"
    assert SideModelMapper({1: "T-80BVM"})(0) == "M1A2"  # 未覆盖项仍用默认


def test_side_model_mapper_role_models():
    """按 role 细分型号（2026-08-18 回归）：炮兵 ≠ MBT，role 覆盖优先于阵营默认。"""
    from Airia.lulilo.entities.side import SIDE_ROLE_MODEL, Side, SideModelMapper, side_model

    # 炮兵（role=1）：北约 M109 / 华约 2S19
    assert SideModelMapper()(int(Side.NATO), 1) == "M109"
    assert SideModelMapper()(int(Side.PACT), 1) == "2S19"
    # 坦克（role=0）/ 缺省：阵营默认型号不变
    assert SideModelMapper()(int(Side.NATO), 0) == "M1A2"
    assert SideModelMapper()(int(Side.PACT), 0) == "T-14"
    assert SideModelMapper()(int(Side.NATO)) == "M1A2"
    # 便捷函数与映射表一致
    assert side_model(int(Side.NATO), 1) == SIDE_ROLE_MODEL[(int(Side.NATO), 1)]
    # 自定义 role 覆盖表注入
    mapper = SideModelMapper(role_models={(int(Side.PACT), 1): "2S19M"})
    assert mapper(int(Side.PACT), 1) == "2S19M"
    assert mapper(int(Side.NATO), 1) == "M109"  # 未覆盖项仍用默认表


def test_adapter_unit_name_artillery_role():
    """adapter 型号命名按 role 细分（2026-08-18 回归）：炮兵显示自行火炮而非 MBT。"""
    role = np.zeros(8, dtype=np.uint8)
    role[0] = 1  # 北约炮兵
    role[1] = 1  # 华约炮兵
    adapter = _make_adapter(role=role)
    assert adapter.unit_name(0) == "M109 #00"
    assert adapter.unit_name(1) == "2S19 #01"
    # 坦克 role 不受影响
    assert adapter.unit_name(2) == "载具 #02"
    # 未装配 role（None）→ 坦克默认（向后兼容）
    assert _make_adapter().unit_name(0) == "M1A2 #00"


def test_adapter_present_parts_only_assembled():
    """详情面板只显示实际装配部位（2026-08-18 回归）。

    炮兵只装配 HULL/TURRET → present 不含引擎/履带/弹药架（此前固定 5 部位
    全显示绿 ✓）；已毁部位（槽位已释放）由单位标记 armor_damaged_parts 补回。
    """
    adapter = _make_adapter()
    # 模拟：单位 0/1 各装配 HULL+TURRET 两个 box（炮兵式装配）
    adapter.box_parent = np.array([0, 0, 1, 1], dtype=np.int32)
    adapter.box_part = np.array([0, 1, 0, 1], dtype=np.int32)  # HULL=0, TURRET=1
    adapter.box_active = np.array([True, True, True, True], dtype=np.bool_)
    # 单位 0 的 TRACK（PartType=3）已毁：槽位释放（不在 box 里），单位标记补回
    adapter.armor_damaged_parts[0, 3] = True
    assert adapter.present_parts(0) == ["车体", "炮塔", "履带"]  # 已毁补回，无引擎/弹药架
    assert adapter.present_parts(1) == ["车体", "炮塔"]           # 完好炮兵
    # inactive 单位 / 越界 → 空
    assert adapter.present_parts(4) == []
    assert adapter.present_parts(99) == []
    # 未装配 box（box_parent 为空数组）→ 仅单位标记
    bare = _make_adapter()
    bare.armor_damaged_parts[0, 0] = True
    assert bare.present_parts(0) == ["车体"]


def test_build_damaged_parts_reads_armor_flag():
    """部位毁伤数据源 = ArmorPool.parts_destroyed。

    回归（2026-08-18）：旧聚合逻辑靠 box.parent/active 推断已毁部位，
    但引擎毁伤即释放槽位（active=False, parent=-1），已毁部位永远显示
    不出——「一直扣血但部位不坏」。现在渲染层直接读单位级标记。
    """
    from Airia.lulilo.entities.pools import ArmorPool, BoxPool, PartType
    from Airia.render.adapter import _build_damaged_parts

    armor = ArmorPool(capacity=8)
    idx = armor.spawn(0.0, 0.0, 0.0)
    armor.parts_destroyed[idx, int(PartType.TURRET)] = True
    # 模拟引擎毁伤后的 box 槽位（active=False + parent=-1，已回 free_stack）
    boxes = BoxPool(capacity=8)
    bidx = boxes.spawn(idx, int(PartType.TURRET), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    boxes.active[bidx] = False
    boxes.parent[bidx] = -1

    damaged = _build_damaged_parts(armor, armor.capacity)
    assert bool(damaged[idx, int(PartType.TURRET)])
    assert not damaged[idx, int(PartType.HULL)]
    # 槽位推断时代这行会失败：parent=-1 → 无法关联到单位
    assert damaged.shape == (armor.capacity, 5)


def test_adapter_ammo_fuel_at():
    adapter = _make_adapter()
    assert adapter.ammo_at(0) == pytest.approx((36.0, 40.0))
    assert adapter.fuel_at(0) == pytest.approx((90.0, 100.0))
    assert adapter.ammo_at(99) is None  # 越界


# ---------------------------------------------------------------------------
# 2026-08-18：双池拾取（地面装甲 + 空中单位）+ 空中详情 / 部件装甲 / 编队 AI
# ---------------------------------------------------------------------------


def _aircraft_view(n: int = 4, side: int = 0):
    """构造 4 机空中视图（北约 side=0；首机 active，其余 inactive）。"""
    return {
        "k": n,
        "pos": np.array([[0.0, 0.0, 3000.0], [100.0, 0.0, 3000.0],
                         [200.0, 0.0, 3000.0], [300.0, 0.0, 3000.0]], dtype=np.float32),
        "active": np.array([True, False, False, False], dtype=np.bool_),
        "side": np.full(n, side, dtype=np.uint8),
        "vel": np.array([[250.0, 0.0, 0.0]] * n, dtype=np.float32),
        "health": np.full(n, 100.0, dtype=np.float32),
        "fuel": np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
        "aam": np.array([4, 4, 4, 4], dtype=np.uint8),
        "agm": np.array([0, 0, 0, 0], dtype=np.uint8),
        "state": np.array([1, 1, 1, 1], dtype=np.uint8),   # CRUISE
        "role": np.zeros(n, dtype=np.uint8),               # FIGHTER
        "formation_id": np.array([3, 3, 3, 3], dtype=np.int32),
    }


def test_pick_aircraft_handle():
    """双池拾取（2026-08-18 回归）：飞机可选中（此前只查 armor，飞机点不到）。

    飞机 @ (0,0,3000) 与地面单位 @ (0,0,0) 同屏位置——identity MVP 下
    z 只影响深度不影响屏幕 x/y，两者投影到同一点；飞机距相机更近？
    为确定性验证，把飞机放在地面单位正上方，选同一像素 → 应取屏幕更近者；
    本用例直接验证 aircraft handle 编码与解码往返 + 拾取优先性。
    """
    from Airia.render.adapter import AIR_HANDLE_OFFSET, decode_handle, is_air_handle
    from Airia.render.picking import pick_nearest

    # 编码/解码往返
    h = AIR_HANDLE_OFFSET + 2
    assert is_air_handle(h)
    assert decode_handle(h) == (1, 2)
    assert decode_handle(7) == (0, 7)
    assert decode_handle(-1) == (-1, -1)

    # 地面单位 @ (0,0,0)（屏幕中心）；飞机 inactive 不参与
    adapter = _make_adapter()
    mvp = glm.mat4(1.0)
    assert pick_nearest(400, 300, adapter, mvp, 800, 600) == 0  # 装甲语义不变

    # 飞机 active @ 同屏点（identity MVP 下 z 须在 [-1,1] 才可见，取 z=0.5）
    av = _aircraft_view()
    av["pos"][0] = (0.0, 0.0, 0.5)  # 飞机 0 与装甲 0 同屏幕位置、可见
    adapter2 = _make_adapter(aircraft=av)
    handle = pick_nearest(400, 300, adapter2, mvp, 800, 600)
    assert handle == 0  # 同点地面优先（严格小于才覆盖）
    # 把地面单位移开 → 同点只剩飞机 → 返回飞机 handle
    adapter3 = _make_adapter(aircraft=av)
    adapter3.armor_pos[0] = (10.0, 0.0, 0.0)  # 地面挪出命中半径
    handle3 = pick_nearest(400, 300, adapter3, mvp, 800, 600)
    assert is_air_handle(handle3) and decode_handle(handle3) == (1, 0)


def test_adapter_aircraft_queries():
    """空中单位详情查询（型号/阵营/速度/高度/挂载/燃料/状态/编队）。"""
    from Airia.render.adapter import AIR_HANDLE_OFFSET

    av = _aircraft_view(side=0)
    adapter = _make_adapter(aircraft=av)
    h = AIR_HANDLE_OFFSET + 0
    assert adapter.aircraft_name(h) == "F-16C #00"
    assert adapter.aircraft_side_name(h) == "北约"
    assert adapter.aircraft_speed_at(h) == pytest.approx(250.0)
    assert adapter.aircraft_alt_at(h) == pytest.approx(3000.0)
    assert adapter.aircraft_health_at(h) == pytest.approx(100.0)
    assert adapter.aircraft_ammo_at(h) == (4, 0)
    assert adapter.aircraft_fuel_at(h) == pytest.approx(80.0)
    assert adapter.aircraft_state_at(h) == 1  # CRUISE
    assert adapter.aircraft_formation_status(h)["formation_id"] == 3
    # 华约飞机型号
    av2 = _aircraft_view(side=1)
    assert _make_adapter(aircraft=av2).aircraft_name(h) == "Su-27 #00"
    # 无效 handle / inactive → 兜底
    assert adapter.aircraft_name(AIR_HANDLE_OFFSET + 1) == "#?"  # inactive
    assert adapter.aircraft_name(5) == "#?"                      # 非空中 handle


def test_adapter_part_armor_mm():
    """每部位标称装甲厚度（详情面板显示，2026-08-18）。"""
    adapter = _make_adapter(
        aircraft={
            "box_parent": np.array([0, 0], dtype=np.int32),
            "box_part": np.array([0, 2], dtype=np.int32),       # HULL, ENGINE
            "box_active": np.array([True, True], dtype=np.bool_),
            "box_armor_mm": np.array([350.0, 150.0], dtype=np.float32),
        }
    )
    assert adapter.part_armor_mm(0, "车体") == pytest.approx(350.0)
    assert adapter.part_armor_mm(0, "引擎") == pytest.approx(150.0)
    # 未装配部位 / 已毁（槽位释放）→ None
    assert adapter.part_armor_mm(0, "履带") is None
    assert adapter.part_armor_mm(1, "车体") is None
    assert adapter.part_armor_mm(0, "不存在") is None


def test_adapter_unit_formation_status():
    """编队 AI 状态（详情面板「编队 AI」区，2026-08-18）。"""
    from Airia.lulilo.entities.pools import ArmorPool
    from Airia.lulilo.simulation.state import SimulationState

    st = SimulationState()
    armor = ArmorPool(capacity=8)
    st.armor_pool = armor
    idx = armor.spawn(0.0, 0.0, 0.0)
    armor.formation_id[idx] = 2
    # 手填编队池视图（构造 WorldAdapter 直填）
    adapter = _make_adapter()
    adapter.armor_formation_id = armor.formation_id
    adapter.formation_intent = np.array([0, 1, 1], dtype=np.int8)      # 2 号编队=进攻
    adapter.formation_order = np.array([0, 0, 2], dtype=np.int8)       # MANEUVER
    adapter.formation_stance = np.array([0, 0, 0], dtype=np.int8)
    adapter.formation_comm = np.array([1.0, 1.0, 0.85], dtype=np.float32)
    adapter.formation_member_count = np.array([8, 8, 4], dtype=np.int32)
    fs = adapter.unit_formation_status(idx)
    assert fs is not None
    assert fs["formation_id"] == 2
    assert fs["intent"] == 1           # ADVANCE
    assert fs["order"] == 2            # MANEUVER
    assert fs["comm"] == pytest.approx(0.85)
    assert fs["member_count"] == 4
    # 无编队 → None
    assert adapter.unit_formation_status(1) is None
    # 未装配 formation 视图 → None
    assert _make_adapter().unit_formation_status(idx) is None
