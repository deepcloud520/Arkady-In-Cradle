"""C4I 界面调色板（light / dark）+ 全局 QSS（2026-11-03 新增）。

对应 docs/ui-cmo-redesign.md §3.6：
    - **现实军事 C4I** 语言：低饱和、方角、无渐变、小字号、高密度
    - 两套主题并存：``light``（指挥所工作站）/ ``dark``（车载战术终端）

与 GL 地形 shader 主题（``assets/shaders/*``，ThemeManager 管理）**是两根轴**：
    - 本模块 = Qt 面板调色板（侧栏 / 时间栏 / 日志 / 弹窗）
    - ThemeManager = 3D 地形光照风格
两者在「视图」菜单里分开命名，互不覆盖。

设计约束：
    - 纯数据 + QSS 字符串，不依赖 QApplication（可无头单测）
    - 语义色固定（阵营/警告/选中），**不允许**在控件里写死十六进制色值
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class Palette:
    """一套界面配色（数值与原型 ``prototype/ui_c4i_demo.py`` 的 THEMES 对齐）。"""

    name: str
    bg: str

    # 半透明面板（(r, g, b, a)）：只给有内容的地方铺底，空白透地图
    panel: tuple[int, int, int, int]
    panel2: tuple[int, int, int, int]
    header: tuple[int, int, int, int]
    log: tuple[int, int, int, int]

    header_text: str
    border: str
    border_dark: str
    text: str
    text_dim: str
    accent: str
    accent_text: str
    select: str

    # 语义色
    nato: str
    pact: str
    neutral: str
    unknown: str
    warn: str
    ok: str

    # 地图 / 图片底
    map_bg: str
    bar_bg: str

    # 时间栏（浮层压在三维地图上；浅色主题必须换深色字 + 浅色阴影，
    # 否则白字压浅底 → 对比度不足，2026-11-03 修复）
    timebar_bg: tuple[int, int, int, int] = (18, 22, 26, 72)
    timebar_text: str = "#ffffff"
    timebar_dim: str = "#c6d1dc"
    timebar_shadow: tuple[int, int, int, int] = (0, 0, 0, 150)

    def color(self, key: str) -> QColor:
        """按属性名取 QColor（semantic / text / border 等均可用）。"""
        value = getattr(self, key)
        if isinstance(value, tuple):
            return rgba(value)
        return QColor(value)

    def css(self, key: str) -> str:
        """按属性名取 CSS 颜色串（半透明面板返回 ``rgba(...)``）。"""
        value = getattr(self, key)
        if isinstance(value, tuple):
            r, g, b, a = value
            return f"rgba({r}, {g}, {b}, {round(a / 255.0, 3)})"
        return str(value)

    def qpalette(self) -> QPalette:
        """Qt 调色板（原生控件质感；见 ``_build_qpalette``）。"""
        return _build_qpalette(self)


def rgba(value: tuple[int, int, int, int]) -> QColor:
    """``(r, g, b, a)`` → QColor。"""
    return QColor(int(value[0]), int(value[1]), int(value[2]), int(value[3]))


LIGHT = Palette(
    name="light",
    bg="#d6d4cf",
    panel=(243, 241, 236, 148),
    panel2=(226, 223, 214, 178),
    header=(74, 90, 106, 225),
    log=(238, 237, 231, 148),
    header_text="#ffffff",
    border="#c9c5bb",
    border_dark="#a9a59b",
    text="#2b2b2b",
    text_dim="#6d6d68",
    accent="#4a5a6a",
    accent_text="#ffffff",
    select="#a85f18",
    nato="#24507a",
    pact="#8a2b2b",
    neutral="#33602c",
    unknown="#75561a",
    warn="#8a5a10",
    ok="#2c5e2c",
    map_bg="#b6bcb2",
    bar_bg="#ffffff",
    timebar_bg=(243, 241, 236, 170),
    timebar_text="#2b2b2b",
    timebar_dim="#5c5c58",
    timebar_shadow=(255, 255, 255, 150),
)

DARK = Palette(
    name="dark",
    bg="#2a2c2e",
    # 2.2 对比度修正（2026-11-03）：深色主题暗卡叠明亮地形（沙漠/雪地）时，
    # 62% 的混合底约 (111,108,99)，#d8d8d8 小字对比度仅 ~3.3:1 < AA 4.5:1。
    # 提高到 72%（≈184/255）后混合底约 (94,91,84)，对比度 ~4.6:1，达标。
    panel=(54, 58, 62, 184),
    panel2=(66, 71, 76, 200),
    header=(60, 74, 86, 238),
    log=(48, 51, 54, 184),
    header_text="#e8e8e8",
    border="#4a4e52",
    border_dark="#232629",
    text="#d8d8d8",
    text_dim="#979b9f",
    accent="#7fa0b8",
    accent_text="#ffffff",
    select="#e0a050",
    nato="#6f93b5",
    pact="#b06a6a",
    neutral="#6f8f6f",
    unknown="#c0a050",
    warn="#c09040",
    ok="#6f9a6f",
    map_bg="#4e544e",
    bar_bg="#2a2c2e",
)

PALETTES: dict[str, Palette] = {"light": LIGHT, "dark": DARK}
DEFAULT_THEME = "dark"

_current: Palette = PALETTES[DEFAULT_THEME]


def current() -> Palette:
    """当前界面配色。"""
    return _current


def get(name: str) -> Palette:
    """按名称取配色（未知 → 当前配色）。"""
    return PALETTES.get(str(name), _current)


def set_current(name: str) -> Palette:
    """切换当前界面配色并返回之（不触发重绘，由调用方统一刷新）。"""
    global _current
    _current = PALETTES.get(str(name), _current)
    return _current


# ----------------------------------------------------------------------
# 语义映射（控件不得写死颜色）
# ----------------------------------------------------------------------

# 阵营 SoA 值（entities/side.py Side）→ 语义键
_SIDE_KEY = {0: "nato", 1: "pact", 2: "neutral"}


def side_color(palette: Palette, side: int) -> QColor:
    """阵营 → 颜色（未知阵营走 ``unknown``）。"""
    return palette.color(_SIDE_KEY.get(int(side), "unknown"))


# 事件 kind → 语义键（docs §5.2 消息日志着色）
_LOG_KIND_KEY = {
    "destroyed": "pact",
    "kill": "pact",
    "hit": "warn",
    "ricochet": "warn",
    "damage": "warn",
    "fire": "select",
    "miss": "text_dim",
    "detection": "nato",
    "detected": "nato",
    "command": "ok",
    "order": "ok",
    "system": "accent",
    "tutorial": "accent",
}


def log_kind_color(palette: Palette, kind: str) -> QColor:
    """事件类型 → 日志文字颜色（未知 kind 用正文色）。"""
    return palette.color(_LOG_KIND_KEY.get(str(kind), "text"))


# ----------------------------------------------------------------------
# 颜色工具（控件质感：渐变 / 明暗层次）
# ----------------------------------------------------------------------

def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """线性混色（t=0 → a，t=1 → b）。"""
    t = max(0.0, min(1.0, float(t)))
    return QColor(
        int(round(a.red() * (1 - t) + b.red() * t)),
        int(round(a.green() * (1 - t) + b.green() * t)),
        int(round(a.blue() * (1 - t) + b.blue() * t)),
        int(round(a.alpha() * (1 - t) + b.alpha() * t)),
    )


def _shade(color: QColor, factor: float) -> QColor:
    """整体提亮/压暗（factor > 1 变亮）。"""
    return QColor(
        max(0, min(255, int(round(color.red() * factor)))),
        max(0, min(255, int(round(color.green() * factor)))),
        max(0, min(255, int(round(color.blue() * factor)))),
        color.alpha(),
    )


def _hex(color: QColor) -> str:
    return color.name()


def _build_qpalette(self: Palette) -> QPalette:
    """Qt 调色板（让原生控件画出真正的控件质感，而不是平涂矩形）。"""
    p = QPalette()
    window = QColor(self.bg)
    base = QColor(self.bar_bg)
    button = _mix(rgba(self.panel2), window, 0.35)
    text = QColor(self.text)

    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, _shade(base, 1.04))
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, button)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Highlight, QColor(self.accent))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.ToolTipBase, base)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Mid, QColor(self.border))
    p.setColor(QPalette.ColorRole.Midlight, _shade(button, 1.15))
    p.setColor(QPalette.ColorRole.Dark, QColor(self.border_dark))
    p.setColor(QPalette.ColorRole.Light, _shade(button, 1.28))
    p.setColor(QPalette.ColorRole.Shadow, _shade(QColor(self.border_dark), 0.8))

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(
            QPalette.ColorGroup.Disabled, role, QColor(self.text_dim)
        )
    return p


# ----------------------------------------------------------------------
# 全局 QSS（控件质感：渐变面 + 立体边 + 原生滚动条）
# ----------------------------------------------------------------------

def qss(palette: Palette) -> str:
    """整窗样式表。

    2026-11-03 重做：上一版把 `QWidget { background: transparent }` 全局铺开，
    结果所有控件都被剥成了「平涂矩形」（用户原话：像 pygame 的 draw.rect）。
    现在只覆盖**外观**：渐变按钮面、1px 立体边、有槽的滚动条、菜单/列表的
    原生选中态；不破坏控件的绘制。
    """
    p = palette
    win = QColor(p.bg)
    base = QColor(p.bar_bg)
    border = QColor(p.border)
    border_dark = QColor(p.border_dark)
    text = QColor(p.text)
    text_dim = QColor(p.text_dim)

    btn = _mix(rgba(p.panel2), win, 0.30)
    btn_hi = _shade(btn, 1.18)
    btn_press = _shade(btn, 0.88)
    field = _mix(base, win, 0.25)
    field_hi = _shade(field, 1.05)

    header = rgba(p.header)
    header_hi = _shade(header, 1.30)
    header_lo = _shade(header, 0.92)

    return f"""
QWidget {{
    color: {_hex(text)};
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 11px;
}}
QMainWindow, QDialog, QWidget#cockpitRoot {{
    background: {_hex(win)};
}}
QLabel {{ background: transparent; color: {_hex(text)}; font-size: 11px; }}
QLabel#panelTitle {{
    color: {_hex(QColor(p.header_text))};
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    background: transparent;
}}

/* ── 按钮：渐变面 + 立体边（真控件手感）── */
QPushButton, QToolButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(btn_hi)}, stop:0.55 {_hex(btn)}, stop:1 {_hex(_shade(btn, 0.96))});
    border: 1px solid {_hex(border_dark)};
    border-radius: 2px;
    padding: 3px 9px;
    color: {_hex(text)};
    min-height: 16px;
}}
QPushButton:hover, QToolButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(_shade(btn_hi, 1.10))}, stop:1 {_hex(_shade(btn, 1.06))});
    border-color: {_hex(_shade(border_dark, 1.35))};
}}
QPushButton:pressed, QToolButton:pressed {{
    background: {_hex(btn_press)};
    border-color: {_hex(border_dark)};
}}
QPushButton:disabled, QToolButton:disabled {{
    background: {_hex(_mix(btn, win, 0.5))};
    color: {_hex(text_dim)};
    border-color: {_hex(_mix(border, win, 0.4))};
}}
QPushButton:checked, QToolButton:checked {{
    background: {_hex(_mix(QColor(p.accent), win, 0.25))};
    border-color: {_hex(QColor(p.accent))};
}}

/* ── 文本/列表：凹陷输入面 ── */
QTextBrowser, QPlainTextEdit, QTextEdit, QListWidget, QLineEdit, QComboBox {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(_shade(field, 0.97))}, stop:1 {_hex(field_hi)});
    color: {_hex(text)};
    border: 1px solid {_hex(border_dark)};
    border-radius: 1px;
    padding: 2px;
    selection-background-color: {_hex(QColor(p.accent))};
    selection-color: #ffffff;
}}
QListWidget::item {{ padding: 2px 4px; }}
QListWidget::item:selected {{
    background: {_hex(QColor(p.accent))};
    color: #ffffff;
}}
QListWidget::item:hover {{ background: {_hex(_mix(btn, win, 0.2))}; }}

QScrollArea {{ border: none; background: transparent; }}

/* ── 滚动条：有槽 + 渐变滑块 + 箭头按钮 ── */
QScrollBar:vertical {{
    background: {_hex(_mix(win, border, 0.35))};
    width: 12px;
    margin: 0px;
    border-left: 1px solid {_hex(border_dark)};
}}
QScrollBar::handle:vertical {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {_hex(_shade(btn, 0.92))}, stop:0.5 {_hex(btn_hi)}, stop:1 {_hex(btn)});
    border: 1px solid {_hex(border_dark)};
    border-radius: 2px;
    min-height: 24px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover {{ background: {_hex(_shade(btn_hi, 1.08))}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 12px;
    background: {_hex(btn)};
    border: 1px solid {_hex(border_dark)};
    subcontrol-origin: margin;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {_hex(_mix(win, border, 0.25))};
}}
QScrollBar:horizontal {{
    background: {_hex(_mix(win, border, 0.35))};
    height: 12px;
    border-top: 1px solid {_hex(border_dark)};
}}
QScrollBar::handle:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(btn_hi)}, stop:1 {_hex(btn)});
    border: 1px solid {_hex(border_dark)};
    border-radius: 2px;
    min-width: 24px;
    margin: 1px;
}}

/* ── 菜单栏 / 菜单 ── */
QMenuBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(_shade(btn, 1.10))}, stop:1 {_hex(btn)});
    color: {_hex(text)};
    border-bottom: 1px solid {_hex(border_dark)};
}}
QMenuBar::item {{ padding: 3px 10px; background: transparent; }}
QMenuBar::item:selected {{
    background: {_hex(QColor(p.accent))};
    color: {_hex(QColor(p.accent_text))};
}}
QMenu {{
    background: {_hex(base)};
    color: {_hex(text)};
    border: 1px solid {_hex(border_dark)};
    padding: 2px;
}}
QMenu::item {{ padding: 3px 22px 3px 18px; }}
QMenu::item:selected {{
    background: {_hex(QColor(p.accent))};
    color: {_hex(QColor(p.accent_text))};
}}

QStatusBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(btn)}, stop:1 {_hex(_shade(btn, 0.94))});
    color: {_hex(text_dim)};
    border-top: 1px solid {_hex(border_dark)};
}}
QToolTip {{
    background: {_hex(base)}; color: {_hex(text)};
    border: 1px solid {_hex(border_dark)}; padding: 3px;
}}

/* ── 进度条 / 分组框（后续 MSR、传感器用）── */
QProgressBar {{
    background: {_hex(_shade(field, 0.94))};
    border: 1px solid {_hex(border_dark)};
    border-radius: 1px;
    text-align: center;
    color: {_hex(text)};
    height: 12px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_hex(_shade(QColor(p.ok), 1.25))}, stop:1 {_hex(QColor(p.ok))});
}}
QGroupBox {{
    border: 1px solid {_hex(border_dark)};
    border-radius: 2px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {_hex(text)};
    background: {_hex(base)};
}}
QCheckBox, QRadioButton {{ spacing: 5px; }}
"""


def panel_header_qss(palette: Palette) -> str:
    """面板标题栏（渐变 + 立体边）——供 DockPanel / 通话窗复用。"""
    p = palette
    header = rgba(p.header)
    hi = _shade(header, 1.30)
    lo = _shade(header, 0.90)
    return (
        "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {_hex(hi)}, stop:1 {_hex(lo)});"
        f" border: 1px solid {_hex(QColor(p.border_dark))};"
        f" border-bottom: 1px solid {_hex(_shade(QColor(p.border_dark), 0.85))};"
    )


