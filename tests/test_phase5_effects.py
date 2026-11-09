"""Phase 5e 无头测试：特效缓冲（EffectLayer）/ 快捷键映射（ControlMap）/
速度估计（WorldAdapter.armor_speed）/ 字体图集（FontAtlas）。

GUI 相关（TextRenderer 渲染 / HUD 特效绘制）需 OpenGL 上下文，
不在无头测试覆盖；冒烟验证见 tests/smoke_gui.py。
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from Airia.render.adapter import WorldAdapter
from Airia.render.effects import EffectLayer
from Airia.ui.controls import ControlMap


# ---------------------------------------------------------------------------
# EffectLayer（弹道线 / 命中标记事件缓冲）
# ---------------------------------------------------------------------------


def test_effect_ingest_fire_and_hit():
    fx = EffectLayer()
    fx.on_events(
        [
            {"kind": "fire", "data": {"shooter": 1, "target": 2}},
            {"kind": "hit", "data": {"shooter": 1, "target": 2, "part": 0}},
            {"kind": "destroyed", "data": {"armor_idx": 5}},
            {"kind": "ricochet", "data": {"target": 3}},
            {"kind": "command", "data": {"type": "HOLD"}},   # 非特效事件应忽略
        ],
        now=10.0,
    )
    assert fx.tracer_count() == 1
    tr = fx.tracer_events[0]
    assert tr.shooter == 1 and tr.target == 2
    # hit / destroyed / ricochet 都算命中标记（3 条）
    assert fx.hit_count() == 3


def test_effect_ignores_bad_events():
    fx = EffectLayer()
    fx.on_events(
        [
            {"kind": "fire", "data": {"shooter": -1, "target": 2}},  # 无效射手
            {"kind": "fire", "data": {"target": 2}},                 # 缺射手
            {"kind": "hit", "data": {"target": -1}},
            {"kind": "unknown_kind", "data": {"a": 1}},
        ],
        now=0.0,
    )
    assert fx.tracer_count() == 0
    assert fx.hit_count() == 0


def test_effect_cull_by_lifetime():
    fx = EffectLayer(fire_lifetime=0.4, hit_lifetime=0.6)
    fx.on_events(
        [
            {"kind": "fire", "data": {"shooter": 1, "target": 2}},
            {"kind": "hit", "data": {"target": 2}},
        ],
        now=0.0,
    )
    fx.cull(now=0.2)   # 都在存活期
    assert fx.tracer_count() == 1
    assert fx.hit_count() == 1
    fx.cull(now=0.5)   # 弹道线过期，命中标记存活
    assert fx.tracer_count() == 0
    assert fx.hit_count() == 1
    fx.cull(now=0.7)   # 命中标记过期
    assert fx.hit_count() == 0


def test_effect_cap_drops_oldest():
    fx = EffectLayer(max_tracers=2, max_hits=2)
    fire_evts = [{"kind": "fire", "data": {"shooter": 1, "target": i}} for i in range(5)]
    fx.on_events(fire_evts, now=0.0)
    assert fx.tracer_count() == 2
    targets = sorted(t.target for t in fx.tracer_events)
    assert targets == [3, 4]   # 丢最旧（0/1/2 被挤掉）


def test_effect_clear():
    fx = EffectLayer()
    fx.on_events([{"kind": "fire", "data": {"shooter": 1, "target": 2}}], now=0.0)
    assert fx.tracer_count() == 1
    fx.clear()
    assert fx.tracer_count() == 0
    assert fx.hit_count() == 0


# ---------------------------------------------------------------------------
# ControlMap（键盘 → 动作名映射）
# ---------------------------------------------------------------------------


class _FakeKey:
    """最小 QKeyEvent 替身（只暴露 key/modifiers）。"""

    def __init__(self, key, mods=Qt.KeyboardModifier.NoModifier):
        self._key = key
        self._mods = mods

    def key(self):
        return self._key

    def modifiers(self):
        return self._mods


def _k(key, mods=Qt.KeyboardModifier.NoModifier):
    return _FakeKey(key, mods)


def test_control_basic_keys():
    assert ControlMap.key_action(_k(Qt.Key.Key_Space)) == ControlMap.TOGGLE_PAUSE
    assert ControlMap.key_action(_k(Qt.Key.Key_Return)) == ControlMap.ADVANCE_DIALOGUE
    assert ControlMap.key_action(_k(Qt.Key.Key_Enter)) == ControlMap.ADVANCE_DIALOGUE
    assert ControlMap.key_action(_k(Qt.Key.Key_S)) == ControlMap.SKIP_DIALOGUE
    assert ControlMap.key_action(_k(Qt.Key.Key_Escape)) == ControlMap.CANCEL_SELECTION
    assert ControlMap.key_action(_k(Qt.Key.Key_F1)) == ControlMap.SHOW_HELP


def test_control_option_digits():
    assert ControlMap.key_action(_k(Qt.Key.Key_1)) == "choose_option:0"
    assert ControlMap.key_action(_k(Qt.Key.Key_5)) == "choose_option:4"
    assert ControlMap.key_action(_k(Qt.Key.Key_9)) == "choose_option:8"
    assert ControlMap.key_action(_k(Qt.Key.Key_0)) is None


def test_control_theme_hotkeys():
    ctrl = Qt.KeyboardModifier.ControlModifier
    assert ControlMap.key_action(_k(Qt.Key.Key_1, ctrl)) == "theme:default"
    assert ControlMap.key_action(_k(Qt.Key.Key_2, ctrl)) == "theme:constructivism"
    assert ControlMap.key_action(_k(Qt.Key.Key_3, ctrl)) == "theme:anime"


def test_control_modified_keys_not_mapped():
    ctrl = Qt.KeyboardModifier.ControlModifier
    shift = Qt.KeyboardModifier.ShiftModifier
    alt = Qt.KeyboardModifier.AltModifier
    assert ControlMap.key_action(_k(Qt.Key.Key_S, ctrl)) is None      # Ctrl+S
    assert ControlMap.key_action(_k(Qt.Key.Key_Space, shift)) is None  # Shift+Space
    assert ControlMap.key_action(_k(Qt.Key.Key_1, alt)) is None        # Alt+1
    assert ControlMap.key_action(_k(Qt.Key.Key_A)) is None             # 普通字母
    assert ControlMap.key_action(_k(Qt.Key.Key_F2)) is None            # 其它功能键


# ---------------------------------------------------------------------------
# WorldAdapter 速度（armor_speed 注入 / 回退）
# ---------------------------------------------------------------------------


def _make_adapter() -> WorldAdapter:
    n = 8
    return WorldAdapter(
        armor_pos=np.zeros((n, 3), dtype=np.float32),
        armor_yaw=np.zeros(n, dtype=np.float32),
        armor_pitch=np.zeros(n, dtype=np.float32),
        armor_roll=np.zeros(n, dtype=np.float32),
        armor_health=np.full(n, 100.0, dtype=np.float32),
        armor_active=np.zeros(n, dtype=np.bool_),
        armor_confidence=np.ones(n, dtype=np.float32),
        armor_status=np.zeros(n, dtype=np.uint8),
        armor_side=np.zeros(n, dtype=np.uint8),
        armor_damaged_parts=np.zeros((n, 5), dtype=np.bool_),
        box_parent=np.zeros(0, dtype=np.int32),
        box_part=np.zeros(0, dtype=np.int32),
        box_active=np.zeros(0, dtype=np.bool_),
        sim_time=0.0,
        tick_count=0,
    )


def test_speed_at_with_array():
    a = _make_adapter()
    a.armor_speed = np.array([5.0, 8.0, 3.5, 0, 0, 0, 0, 0], dtype=np.float32)
    assert a.speed_at(0) == pytest.approx(5.0)
    assert a.speed_at(1) == pytest.approx(8.0)
    assert a.speed_at(2) == pytest.approx(3.5)
    assert a.speed_at(99) == 0.0      # 越界回退
    assert a.speed_at(-1) == 0.0


def test_speed_at_none_fallback():
    a = _make_adapter()   # armor_speed=None（未估计）
    assert a.speed_at(0) == 0.0


def test_from_snapshot_armor_speed():
    from Airia.lulilo.entities.pools import ArmorPool
    from Airia.lulilo.simulation.state import SimulationState

    state = SimulationState()
    armor = ArmorPool(capacity=8)
    state.armor_pool = armor
    for i in range(3):
        armor.spawn(
            x=float(i), y=0.0, z=0.0,
            half_x=1.0, half_y=2.0, half_z=1.0, yaw=0.0,
        )
    speed = np.array([5.0, 6.0, 7.0, 0, 0, 0, 0, 0], dtype=np.float32)
    a = WorldAdapter.from_snapshot(state.snapshot(), armor_speed=speed)
    assert a.unit_count() == 3
    assert a.speed_at(0) == pytest.approx(5.0)
    assert a.speed_at(2) == pytest.approx(7.0)
    # 未注入 speed 的适配器回退 0
    b = WorldAdapter.from_snapshot(state.snapshot())
    assert b.speed_at(0) == 0.0


# ---------------------------------------------------------------------------
# FontAtlas（PIL 图集构建，CPU 侧无 GL）
# ---------------------------------------------------------------------------


def test_font_atlas_basic():
    from Airia.render.hud.text import FontAtlas

    atlas = FontAtlas(size=32)
    if not atlas.available:
        pytest.skip("无可用系统字体，跳过图集测试")
    assert atlas.has("A")
    assert atlas.has("车")          # 中文部位名
    assert atlas.has("⚠")
    assert atlas.has(" ")
    assert atlas.atlas_width > 0 and atlas.atlas_height > 0
    g = atlas.glyph("A")
    assert g is not None
    assert g.advance > 0
    assert g.w > 0 and g.h > 0
    assert g.u1 > g.u0 and g.v1 > g.v0
    # 字符集覆盖：ASCII 可打印 + 附加字符
    assert len(atlas.charset()) >= 95 + 16


def test_font_atlas_missing_glyph():
    from Airia.render.hud.text import FontAtlas

    atlas = FontAtlas(size=32)
    if not atlas.available:
        pytest.skip("无可用系统字体，跳过图集测试")
    assert not atlas.has("\U0001F4A5")   # emoji 一般不在图集
    assert atlas.glyph("\U0001F4A5") is None
