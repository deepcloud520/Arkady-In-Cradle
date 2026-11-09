"""C4I 浮层界面（2026-11-03 重构）无头测试。

覆盖**纯逻辑**部分（不需要 OpenGL 上下文）：
    - 响应式尺寸 ``ui/metrics.py``：各分辨率/断点/DPI 行为
    - 时间栏 Selected 三态文本
    - 调色板语义色（日志 kind 着色）与深色面板 alpha 对比度修正
    - 通话弹窗 STT 打字机（``TypewriterModel``）与状态接线
    - 缩略图路径解析 / MANIFEST / 回退
    - 驾驶舱浮层几何（地图铺满 + 侧栏右侧 + 日志底部 + 抽屉断点）
    - 相机远裁剪面按地图对角线放大

GUI 相关（真实 GL 渲染）不在无头测试覆盖，冒烟见 ``tests/smoke_c4i_shell.py``。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QWidget

from Airia.render.camera import far_for_extent
from Airia.render.hud.log_widget import LogWidget
from Airia.ui import palette as pal
from Airia.ui.cockpit import Cockpit
from Airia.ui.comms_popup import CommsPopup, TypewriterModel
from Airia.ui.metrics import for_viewport
from Airia.ui.dock import DockPanel
from Airia.ui.message_panel import MessageLogPanel
from Airia.ui.timebar import TimeBar, TimeBarState, format_selected
from Airia.ui.unit_thumb import UnitThumbnail, load_manifest, thumb_path


@pytest.fixture(scope="module")
def app() -> QApplication:
    inst = QApplication.instance()
    if inst is None:
        inst = QApplication([])
    return inst  # type: ignore[return-value]


# ----------------------------------------------------------------------
# 响应式尺寸
# ----------------------------------------------------------------------

def test_metrics_1080p() -> None:
    m = for_viewport(1920, 1080)
    assert m.mode == "full"
    assert m.k == pytest.approx(1.2)
    assert m.sidebar_w == 240          # 200 × 1.2
    assert m.timebar_h == 65           # 54 × 1.2
    assert m.bottombar_h == 115        # 96 × 1.2
    assert m.thumb == 115
    assert m.timebar_blocks == ("地点", "日期", "ZULU", "剩余", "高度")


def test_metrics_4k_is_clamped() -> None:
    m = for_viewport(3840, 2160)
    assert m.k == pytest.approx(2.0)        # 2160/900 = 2.4 → 钳到 2.0
    assert m.sidebar_w == 400
    assert m.thumb == 192


def test_metrics_ultrawide_uses_height_only() -> None:
    m = for_viewport(3440, 1440)
    assert m.k == pytest.approx(1.6)
    assert m.sidebar_w == 320                # 不是按宽度 3440 放大


def test_metrics_breakpoints() -> None:
    assert for_viewport(1600, 900).mode == "full"
    assert for_viewport(1400, 900).mode == "compact"
    assert for_viewport(1000, 800).mode == "tight"
    drawer = for_viewport(800, 600)
    assert drawer.mode == "drawer"
    assert drawer.sidebar_w == 0
    assert drawer.timebar_blocks == ()


def test_metrics_short_window_shrinks_bars() -> None:
    short = for_viewport(1366, 768)     # 常见笔记本
    tall = for_viewport(1366, 1000)
    assert short.timebar_h < tall.timebar_h
    assert short.bottombar_h < tall.bottombar_h


def test_metrics_user_scale_and_bounds() -> None:
    assert for_viewport(1920, 1080, 1.3).k > for_viewport(1920, 1080, 1.0).k
    # 极小窗口：k 不低于 0.75，且侧栏不超过窗口 22%
    tiny = for_viewport(640, 480)
    assert tiny.k >= 0.75
    assert tiny.sidebar_w <= int(round(0.22 * 640)) + 1


# ----------------------------------------------------------------------
# 时间栏 Selected 三态
# ----------------------------------------------------------------------

def test_format_selected_three_states() -> None:
    assert format_selected("none") == "Selected: 无"
    assert format_selected("single", "M1A2 #03", "第 7 装甲连 / ALPHA") == (
        "1x M1A2 #03 · 第 7 装甲连 / ALPHA"
    )
    assert format_selected("multi", count=3, formation="第 7 装甲连") == (
        "3 units · 第 7 装甲连"
    )
    assert format_selected("multi", count=3, multi_formation=True) == (
        "3 units · 多编队"
    )
    # 空中单位无编队 → 只显示名字
    assert format_selected("single", "J-20 #01") == "1x J-20 #01"


def test_format_cmo_line_matches_reference() -> None:
    """时间栏 = CMO 那条半透明黑底白字：Time / Local / Zulu / 剩余 / 高度。"""
    from Airia.ui.timebar import format_cmo_line

    line = format_cmo_line("07:45:44", "05:45:44", 23 * 3600 + 59 * 60, 6818)
    assert line.startswith("Time: 07:45:44")
    assert "Local: 07:45:44" in line
    assert "Zulu: 05:45:44" in line
    assert "23 hr 59 min to go" in line
    assert "Camera Alt: 6818 m" in line
    assert "  ·  " in line
    assert "Time Scale" not in line


def test_closed_message_panel_stays_closed(app: QApplication) -> None:
    """Bug 修复：暂停/重排时不能把玩家关掉的窗口又 show 回来。"""
    c = _make_cockpit()
    c.resize(1920, 1080)
    c._apply_geometry()
    c.message_panel.open(expanded=True)
    c.message_panel.header.close_btn.click()
    assert c.message_panel.isHidden()
    # 任何重排（含 resize）都不应擅自弹回
    c._apply_geometry()
    c.resize(1800, 1000)
    c._apply_geometry()
    assert c.message_panel.isHidden()
    # 明确要求（L 键 / 菜单）时才重新打开
    c.toggle_message_panel()
    assert not c.message_panel.isHidden()


def test_timebar_blocks_respect_metrics(app: QApplication) -> None:
    tb = TimeBar()
    tb.set_metrics(for_viewport(1000, 800))       # tight：只留 ZULU
    tb.set_state(
        TimeBarState(clock="07:45:44", blocks=(("ZULU", "05:45:44"),))
    )
    assert tb.height() == for_viewport(1000, 800).timebar_h


# ----------------------------------------------------------------------
# 调色板 / 日志着色
# ----------------------------------------------------------------------

def test_dark_panel_alpha_meets_contrast_fix() -> None:
    # 2.2 决策：深色主题面板 alpha 由 158 → 184（≈72%），保证亮地形上小字可读
    assert pal.DARK.panel[3] == 184
    assert pal.LIGHT.panel[3] == 148          # 浅色卡片叠亮底本来就安全，保持 58%


def test_log_kind_colors_are_semantic() -> None:
    d, h, m = (
        pal.log_kind_color(pal.DARK, "destroyed"),
        pal.log_kind_color(pal.DARK, "hit"),
        pal.log_kind_color(pal.DARK, "detection"),
    )
    assert d == pal.DARK.color("pact")
    assert h == pal.DARK.color("warn")
    assert m == pal.DARK.color("nato")
    assert pal.log_kind_color(pal.DARK, "unknown_kind") == pal.DARK.color("text")


def test_qss_contains_palette_colors() -> None:
    q = pal.qss(pal.DARK)
    assert pal.DARK.bg in q
    assert pal.DARK.text in q
    # 控件质感：按钮/标题条用渐变面，而不是平涂矩形
    assert "qlineargradient" in q
    assert "QScrollBar::handle" in q
    assert "QPushButton:pressed" in q


def test_qpalette_has_native_roles() -> None:
    from PySide6.QtGui import QPalette

    qp = pal.DARK.qpalette()
    assert qp.color(QPalette.ColorRole.Window).name() == pal.DARK.bg
    assert qp.color(QPalette.ColorRole.Base).name() == pal.DARK.bar_bg
    # 禁用态文字变淡（原生控件需要）
    assert (
        qp.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name()
        == pal.DARK.text_dim
    )


def test_log_widget_system_and_events(app: QApplication) -> None:
    log = LogWidget()
    log.set_palette(pal.DARK)
    log.append_system("战术终端已就绪")
    log.append_events([{"time": 1.0, "kind": "destroyed", "data": {"target": 3}}])
    texts = [log._list.item(i).text() for i in range(log._list.count())]
    assert texts[0].startswith("[SYS]")
    assert "击毁" in texts[1] and "3" in texts[1]


# ----------------------------------------------------------------------
# 通话弹窗（STT）
# ----------------------------------------------------------------------

def test_typewriter_progression() -> None:
    tw = TypewriterModel("敌军装甲连正在渡河", chars_per_sec=10.0)
    assert tw.shown == ""
    assert not tw.finished
    tw.advance(0.5)
    assert tw.revealed == 5          # 10 字/秒 × 0.5s
    tw.advance(10.0)
    assert tw.finished and tw.shown == tw.text


def test_typewriter_reveal_all_is_idempotent() -> None:
    tw = TypewriterModel("abc", chars_per_sec=1.0)
    assert tw.reveal_all() == "abc"
    assert tw.advance(1.0) == "abc"
    assert tw.finished


def _dialogue_state(text: str = "敌军坦克连正在渡河，立即前出阻截。") -> SimpleNamespace:
    return SimpleNamespace(
        active_dialogue=SimpleNamespace(
            dialogue_id="d1",
            speaker="第 7 装甲连 / ALPHA",
            text=text,
            options=["明白", "请求火力支援"],
            history=["[07:45] 通讯建立"],
            is_finished=False,
        )
    )


def test_comms_popup_stt_then_options(app: QApplication) -> None:
    popup = CommsPopup(chars_per_sec=100.0)
    popup.set_state(_dialogue_state())
    assert popup.isVisible()
    assert popup._typing
    # 文本未吐完时选项不出现（避免剧透）
    assert popup._options_layout.count() == 0
    for _ in range(20):
        popup._on_tick()
    assert not popup._typing
    assert popup._body.text() == popup._tw.text
    assert popup._options_layout.count() == 2
    popup.set_state(SimpleNamespace(active_dialogue=None))
    assert not popup.isVisible()


def test_comms_popup_without_avatar_shows_sound_only(app: QApplication) -> None:
    popup = CommsPopup(chars_per_sec=100.0)
    popup.set_state(_dialogue_state())
    assert popup._avatar.has_avatar is False     # 无头像 → Sound only
    assert popup._avatar.isVisible()


# ----------------------------------------------------------------------
# 缩略图（无版权图片 + 回退）
# ----------------------------------------------------------------------

def test_thumb_path_manifest_and_scan() -> None:
    # 不用 pytest 的 tmp_path：本机 pytest 临时根不可写；用工作区 tests/output 子目录
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parent / "output" / "_thumb_test"
    shutil.rmtree(root, ignore_errors=True)
    try:
        thumbs = root / "assets" / "units" / "thumbs"
        thumbs.mkdir(parents=True)
        (thumbs / "t14.png").write_bytes(b"")
        (thumbs / "custom.png").write_bytes(b"")
        (thumbs / "MANIFEST.toml").write_text(
            '[m1a2]\nfile = "custom.png"\nlicense = "CC0"\n', encoding="utf-8"
        )

        assert thumb_path("t14", root=root).name == "t14.png"
        assert thumb_path("m1a2", root=root).name == "custom.png"
        assert thumb_path("unknown_type", root=root) is None
        assert load_manifest(root)["m1a2"]["license"] == "CC0"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unit_thumbnail_falls_back_to_placeholder(app: QApplication) -> None:
    thumb = UnitThumbnail("no_such_unit_xyz", "M1A2", 0, size=96)
    assert thumb.has_image is False
    assert not thumb.grab().isNull()          # 占位也能画出来
    thumb.set_size(144)
    assert thumb.width() == 144


# ----------------------------------------------------------------------
# 驾驶舱浮层几何
# ----------------------------------------------------------------------

def _make_cockpit() -> Cockpit:
    return Cockpit(
        map_widget=QWidget(),
        side_widget=QWidget(),
        message_panel=MessageLogPanel(LogWidget()),
        comms=CommsPopup(),
        timebar=TimeBar(),
    )


def test_cockpit_full_layout(app: QApplication) -> None:
    c = _make_cockpit()
    c.resize(1920, 1080)
    c._apply_geometry()
    m = c.metrics
    assert c.map.geometry().getRect() == (0, 0, 1920, 1080)      # 地图铺满
    assert c.side.geometry().x() == 1920 - m.sidebar_w           # 侧栏贴右
    assert c.side.height() == 1080                               # 无底栏，侧栏占满高
    assert c.timebar.width() == 1920 - m.sidebar_w
    # MESSAGE LOG 是面板：默认折叠（只留标题栏），贴左下
    assert isinstance(c.message_panel, DockPanel)
    assert c.message_panel.is_collapsed
    assert c.message_panel.height() == c.message_panel.header.height()
    assert c.message_panel.geometry().x() == m.pad * 2
    assert c.message_panel.geometry().bottom() <= 1080


def test_message_log_is_widget_panel(app: QApplication) -> None:
    """用户定稿：message 必须是控件（可打开 / 折叠 / 关闭）。"""
    log = LogWidget()
    panel = MessageLogPanel(log)
    assert panel.is_collapsed                       # 默认折叠，不挡战场
    panel.push_system("战术终端已就绪")
    assert panel.header.title.text().startswith("MESSAGE LOG · 1")
    panel.toggle_collapsed()
    assert not panel.is_collapsed
    assert panel.header.title.text() == "MESSAGE LOG"   # 展开后清零未读
    panel.header.close_btn.click()
    assert panel.isHidden()                         # 可关闭


def test_dock_panel_drag_is_clamped(app: QApplication) -> None:
    host = QWidget()
    host.resize(800, 600)
    panel = DockPanel("TEST", QWidget(), host)
    panel.setGeometry(10, 10, 200, 120)
    panel.begin_drag(panel.pos())
    panel.drag_to(panel.pos() + QPoint(5000, 5000))
    assert panel.x() <= 800 - panel.width()
    assert panel.y() <= 600 - panel.height()
    assert panel.moved_by_user


def test_cockpit_drawer_hides_sidebar(app: QApplication) -> None:
    c = _make_cockpit()
    c.resize(820, 600)
    c._apply_geometry()
    assert c.metrics.mode == "drawer"
    assert c.side.isHidden()
    # 手动唤出仍可用（抽屉浮出，不挤压地图）
    c.toggle_sidebar(True)
    assert not c.side.isHidden()
    assert c.side.width() > 0
    assert c.timebar.width() == 820          # drawer 是浮层，不给时间栏让位


def test_cockpit_user_scale_changes_sidebar(app: QApplication) -> None:
    c = _make_cockpit()
    c.resize(1920, 1080)
    base = c.metrics.sidebar_w
    c.set_user_scale(1.3)
    assert c.metrics.sidebar_w > base


# ----------------------------------------------------------------------
# 相机远裁剪面
# ----------------------------------------------------------------------

def test_far_covers_15km_map() -> None:
    far = far_for_extent(15000, 15000)
    # 相机在 9500m 处看地图远角 ≈ 9500 + 10600 → 必须远超 12000
    assert far >= 15000 * 1.41 * 3
    assert far_for_extent(1000, 1000) == 12000.0     # 小地图保持旧下限
