"""AIC UI 原型 · 现实 C4I 风格（半透明浮动面板 / 浅色 + 深色）。

独立演示程序，**不接入主项目**（不 import Airia.*，不改任何现有文件）。

设计要点（对应 docs/ui-cmo-redesign.md）：
    1. 地图铺满，所有面板以**半透明浮层**叠加其上（真实 C4I / CMO 的形态）
    2. 顶部时间栏：`Time: ... - Zulu ... - Local ... - 剩余 ... - 相机高度 ...`
    3. 折叠分组侧栏（窄，200px）
    4. 装备工程图只保留轮廓 + 尺寸标注（不放 120mm/CREW 这类与战术无关的信息）
    5. 低饱和度战术配色，避免鲜艳色块

运行:
    .venv\\Scripts\\python.exe prototype/ui_c4i_demo.py
    .venv\\Scripts\\python.exe prototype/ui_c4i_demo.py --theme dark

离屏截图:
    .venv\\Scripts\\python.exe prototype/ui_c4i_demo.py --theme light --shot prototype/output/c4i_light.png
"""
from __future__ import annotations

import math
import os
import sys

if "--shot" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ======================================================================
# 主题（低饱和战术配色；面板半透明）
# ======================================================================

class C:
    pass


THEMES: dict[str, dict] = {
    # ── 浅色：现代指挥所（冷灰米白，不是 Windows 95 那种黄灰）──
    "light": {
        "BG": "#d6d4cf",
        "PANEL_RGBA": (243, 241, 236, 148),
        "PANEL_2_RGBA": (226, 223, 214, 178),
        "PANEL_2": "#e2dfd6",
        "HEADER_RGBA": (74, 90, 106, 225),
        "HEADER_TEXT": "#ffffff",
        "BORDER": "#c9c5bb",
        "BORDER_DARK": "#a9a59b",
        "TEXT": "#2b2b2b",
        "TEXT_DIM": "#6d6d68",
        "ACCENT": "#4a5a6a",
        "ACCENT_TEXT": "#ffffff",
        "SELECT": "#a85f18",
        "NATO": "#24507a",
        "PACT": "#8a2b2b",
        "NEUTRAL": "#33602c",
        "UNKNOWN": "#75561a",
        "WARN": "#8a5a10",
        "OK": "#2c5e2c",
        "MAP_BG": "#b6bcb2",
        "MAP_GRID": "#9ba196",
        "MAP_CONTOUR": "#aab0a6",
        "BLUEPRINT_BG": "#f9f8f3",
        "BLUEPRINT_GRID": "#e2ded2",
        "BLUEPRINT_LINE": "#4a5a6a",
        "LOG_RGBA": (238, 237, 231, 148),
        "BAR_BG": "#ffffff",
    },
    # ── 深色：车内战术终端 ──
    "dark": {
        "BG": "#2a2c2e",
        "PANEL_RGBA": (54, 58, 62, 158),
        "PANEL_2_RGBA": (66, 71, 76, 186),
        "PANEL_2": "#42474c",
        "HEADER_RGBA": (60, 74, 86, 238),
        "HEADER_TEXT": "#e8e8e8",
        "BORDER": "#4a4e52",
        "BORDER_DARK": "#232629",
        "TEXT": "#d8d8d8",
        "TEXT_DIM": "#979b9f",
        "ACCENT": "#7fa0b8",
        "ACCENT_TEXT": "#ffffff",
        "SELECT": "#e0a050",
        "NATO": "#6f93b5",
        "PACT": "#b06a6a",
        "NEUTRAL": "#6f8f6f",
        "UNKNOWN": "#c0a050",
        "WARN": "#c09040",
        "OK": "#6f9a6f",
        "MAP_BG": "#4e544e",
        "MAP_GRID": "#5e645c",
        "MAP_CONTOUR": "#575d55",
        "BLUEPRINT_BG": "#34383c",
        "BLUEPRINT_GRID": "#3e4348",
        "BLUEPRINT_LINE": "#b0bcc8",
        "LOG_RGBA": (48, 51, 54, 158),
        "BAR_BG": "#2a2c2e",
    },
}


def apply_theme(name: str) -> None:
    for k, v in THEMES[name].items():
        setattr(C, k, v)


SIDE_KEY = {"北约": "NATO", "华约": "PACT", "中立": "NEUTRAL", "未知": "UNKNOWN"}


def side_color(side: str) -> QColor:
    return QColor(getattr(C, SIDE_KEY.get(side, "UNKNOWN")))


def rgba(t: tuple[int, int, int, int]) -> QColor:
    return QColor(t[0], t[1], t[2], t[3])


# ======================================================================
# 假数据
# ======================================================================

UNIT = {
    "name": "M1A2 #03",
    "side": "北约",
    "role": "主战坦克",
    "status": "战斗中",
    "readiness": "PMC",
    "readiness_cn": "部分可用",
    "course": 32,
    "speed": 12.0,
    "systems": [
        ("机动", "受限", "左履带断裂", "WARN"),
        ("火力", "可用", "", "OK"),
        ("防护", "可用", "弹药架受损", "OK"),
        ("通信", "可用", "", "OK"),
        ("三防", "正常", "", "OK"),
    ],
    "ammo": (38, 40),
    "fuel": 0.82,
    "radar": "主动",
    "formation": "第 7 装甲连 / ALPHA",
    "comm": 0.87,
    "intent": "进攻",
    "order": "机动",
    "events": [
        "[12.3s] 命中 T-14 #07（未击穿）",
        "[11.8s] 引擎受损（起火风险）",
        "[10.4s] 履带被毁（无法机动）",
    ],
}

BP = {
    "total_len": 9.8,
    "hull_len": 7.9,
    "hull_h": 1.45,
    "turret_len": 3.4,
    "turret_h": 1.05,
    "wheel_d": 0.72,
    "wheels": 6,
}

SIDE_W = 200
TIME_H = 54
BOTTOM_H = 96


# ======================================================================
# 装备工程图（只保留轮廓与尺寸标注，无无关数据）
# ======================================================================

def draw_blueprint(painter: QPainter, rect: QRectF, unit: dict) -> None:
    painter.save()
    painter.setClipRect(rect)
    painter.fillRect(rect, QColor(C.BLUEPRINT_BG))

    ink = QColor(C.BLUEPRINT_LINE)
    grid = QColor(C.BLUEPRINT_GRID)

    painter.setPen(QPen(grid, 1))
    gx = rect.left() - (rect.left() % 16)
    while gx < rect.right():
        painter.drawLine(QPointF(gx, rect.top()), QPointF(gx, rect.bottom()))
        gx += 16
    gy = rect.top() - (rect.top() % 16)
    while gy < rect.bottom():
        painter.drawLine(QPointF(rect.left(), gy), QPointF(rect.right(), gy))
        gy += 16

    pad_l, pad_r = 10.0, 10.0
    top_label = rect.top() + 16.0
    bot_data = rect.bottom() - 8.0
    draw_w = rect.width() - pad_l - pad_r
    draw_h = bot_data - top_label

    s = draw_w / BP["total_len"]
    hull_l = BP["hull_len"] * s
    hull_h = BP["hull_h"] * s
    tur_l = BP["turret_len"] * s
    tur_h = BP["turret_h"] * s
    wheel_r = BP["wheel_d"] * s / 2

    total_h = hull_h + tur_h + wheel_r * 2
    ground_y = top_label + (draw_h - total_h) / 2 + total_h
    hull_top = ground_y - wheel_r * 2 - hull_h
    hull_left = rect.left() + pad_l
    hull_bottom = hull_top + hull_h

    line = QPen(ink, 1.0)
    line.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    outline = QPen(ink, 1.7)
    outline.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    faint = QPen(QColor(ink.red(), ink.green(), ink.blue(), 100), 0.8)

    # 履带
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    track = QRectF(hull_left + 2, hull_bottom - wheel_r * 0.9,
                   hull_l - 4, wheel_r * 2.9)
    painter.drawRoundedRect(track, wheel_r * 0.9, wheel_r * 0.9)

    # 负重轮
    painter.setPen(line)
    n = BP["wheels"]
    for i in range(n):
        cx = track.left() + track.width() * (i + 0.5) / n
        painter.drawEllipse(QPointF(cx, track.center().y()),
                            wheel_r * 0.85, wheel_r * 0.85)

    # 车体
    body = QPainterPath()
    body.moveTo(hull_left, hull_top + hull_h * 0.15)
    body.lineTo(hull_left + hull_l * 0.06, hull_top)
    body.lineTo(hull_left + hull_l * 0.92, hull_top)
    body.lineTo(hull_left + hull_l, hull_top + hull_h * 0.45)
    body.lineTo(hull_left + hull_l * 0.97, hull_bottom)
    body.lineTo(hull_left + hull_l * 0.02, hull_bottom)
    body.closeSubpath()
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    painter.drawPath(body)

    # 发动机格栅
    painter.setPen(faint)
    for i in range(3):
        yy = hull_top + hull_h * (0.32 + i * 0.16)
        painter.drawLine(QPointF(hull_left + hull_l * 0.07, yy),
                         QPointF(hull_left + hull_l * 0.22, yy))

    # 炮塔
    tur_left = hull_left + hull_l * 0.30
    tur_top = hull_top - tur_h
    turret = QPainterPath()
    turret.moveTo(tur_left, hull_top + 1)
    turret.lineTo(tur_left + tur_l * 0.10, tur_top + tur_h * 0.25)
    turret.lineTo(tur_left + tur_l * 0.55, tur_top)
    turret.lineTo(tur_left + tur_l * 0.88, tur_top + tur_h * 0.12)
    turret.lineTo(tur_left + tur_l, tur_top + tur_h * 0.75)
    turret.lineTo(tur_left + tur_l * 0.96, hull_top + 1)
    turret.closeSubpath()
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    painter.drawPath(turret)

    # 炮管
    gun_y = tur_top + tur_h * 0.55
    gun_w = max(2.0, tur_h * 0.20)
    gun_start = tur_left + tur_l * 0.96
    gun_end = hull_left + BP["total_len"] * s
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    painter.drawRect(QRectF(gun_start, gun_y - gun_w / 2, gun_end - gun_start, gun_w))
    painter.drawRect(QRectF(gun_end - gun_w * 2.2, gun_y - gun_w * 1.05,
                            gun_w * 2.2, gun_w * 2.1))

    # 尺寸线（只留一条总长，信息收敛）
    dim_y = max(tur_top - 6, top_label + 2)
    painter.setPen(QPen(QColor(ink.red(), ink.green(), ink.blue(), 190), 0.8))
    painter.drawLine(QPointF(hull_left, dim_y), QPointF(gun_end, dim_y))
    for xx in (hull_left, gun_end):
        painter.drawLine(QPointF(xx, dim_y - 2.5), QPointF(xx, dim_y + 2.5))

    f = painter.font()
    f.setFamily("Consolas")
    f.setPointSizeF(5.8)
    painter.setFont(f)
    painter.setPen(ink)
    mid_x = (hull_left + gun_end) / 2
    txt = QRectF(mid_x - 20, dim_y - 5, 40, 10)
    painter.fillRect(txt, QColor(C.BLUEPRINT_BG))
    painter.drawText(txt, Qt.AlignmentFlag.AlignCenter, "9.8 m")

    # 阵营小色块（角落，不占版面）
    painter.fillRect(QRectF(rect.left() + 5, rect.top() + 5, 7, 7),
                     side_color(unit["side"]))

    painter.restore()


class BlueprintThumb(QLabel):
    def __init__(self, unit: dict, w: int = 186, h: int = 92, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(w, h)
        pm = QPixmap(w, h)
        pm.fill(QColor(C.BLUEPRINT_BG))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        draw_blueprint(p, QRectF(0, 0, w, h), unit)
        p.end()
        self.setPixmap(pm)


# ======================================================================
# NATO APP-6 简化军标
# ======================================================================

def draw_unit_symbol(p: QPainter, x: float, y: float, yaw: float,
                     col: QColor, kind: str) -> None:
    w, h = 22.0, 14.0

    rad = math.radians(yaw)
    dx, dy = math.sin(rad), -math.cos(rad)
    tip = QPointF(x + dx * 17, y + dy * 17)
    perp = QPointF(-dy, dx)
    b1 = QPointF(x + dx * 10 + perp.x() * 3.4, y + dy * 10 + perp.y() * 3.4)
    b2 = QPointF(x + dx * 10 - perp.x() * 3.4, y + dy * 10 - perp.y() * 3.4)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(col)
    p.drawPolygon(QPolygonF([tip, b1, b2]))

    box = QRectF(x - w / 2, y - h / 2, w, h)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(col, 1.5))
    p.drawRect(box)

    p.setPen(QPen(col, 1.1))
    if kind == "armor":
        p.drawEllipse(QRectF(x - 6.5, y - 4.0, 13, 8))
    elif kind == "artillery":
        p.setBrush(col)
        p.drawEllipse(QPointF(x, y), 3.0, 3.0)
        p.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "infantry":
        p.drawLine(QPointF(x - 6, y - 6), QPointF(x + 6, y + 6))
        p.drawLine(QPointF(x + 6, y - 6), QPointF(x - 6, y + 6))
    else:
        p.drawText(QRectF(x - w / 2, y - h / 2, w, h),
                   Qt.AlignmentFlag.AlignCenter, "?")


# ======================================================================
# 战场（铺满，作为所有浮层的背景）
# ======================================================================

class Battlefield(QWidget):
    UNITS = [
        (0.26, 0.44, 32, "北约", "armor", "M1A2 #03", True),
        (0.18, 0.60, 318, "北约", "armor", "M1A2 #05", False),
        (0.30, 0.31, 18, "北约", "artillery", "M109 #02", False),
        (0.34, 0.76, 75, "北约", "armor", "M1A2 #07", False),
        (0.15, 0.84, 12, "北约", "infantry", "步 1 班", False),
        (0.60, 0.26, 155, "华约", "armor", "T-14 #07", False),
        (0.70, 0.40, 200, "华约", "armor", "T-14 #09", False),
        (0.54, 0.54, 128, "华约", "artillery", "2S19 #01", False),
        (0.78, 0.64, 240, "华约", "armor", "T-14 #11", False),
        (0.85, 0.82, 300, "华约", "armor", "T-14 #13", False),
        (0.48, 0.16, 190, "华约", "armor", "T-14 #04", False),
        (0.42, 0.90, 40, "中立", "infantry", "卡车队", False),
        (0.64, 0.74, 0, "未知", "unknown", "接触 ?", False),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.safe_bottom = 0     # 底部浮层高度（比例尺/坐标避让）

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        hb = h - self.safe_bottom

        p.fillRect(self.rect(), QColor(C.MAP_BG))

        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(5):
            p.setPen(QPen(QColor(C.MAP_CONTOUR), 1))
            p.drawEllipse(QRectF(0.22 * w - i * 34, 0.34 * h - i * 22,
                                 320 + i * 66, 250 + i * 46))
        for i in range(4):
            p.setPen(QPen(QColor(C.MAP_CONTOUR), 1))
            p.drawEllipse(QRectF(0.74 * w - i * 26, 0.72 * h - i * 16,
                                 170 + i * 46, 120 + i * 32))

        # 网格 + 分格标注
        step = 96
        f = p.font()
        f.setFamily("Consolas")
        f.setPointSizeF(6.5)
        p.setFont(f)
        for i, x in enumerate(range(0, w, step)):
            p.setPen(QPen(QColor(C.MAP_GRID), 1))
            p.drawLine(x, 0, x, h)
            p.setPen(QColor(C.MAP_GRID).darker(112))
            p.drawText(QPointF(x + 3, 12), f"{i * 2}")
        for i, y in enumerate(range(0, h, step)):
            p.setPen(QPen(QColor(C.MAP_GRID), 1))
            p.drawLine(0, y, w, y)
            p.setPen(QColor(C.MAP_GRID).darker(112))
            p.drawText(QPointF(3, y + 12), f"{chr(65 + i)}")

        # 射程圈
        sel = next((u for u in self.UNITS if u[6]), None)
        if sel is not None:
            sx, sy = sel[0] * w, sel[1] * h
            p.setPen(QPen(QColor(C.SELECT), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = min(w, h) * 0.185
            p.drawEllipse(QPointF(sx, sy), r, r)
            p.setPen(QColor(C.SELECT))
            p.drawText(QRectF(sx + r - 40, sy - 12, 46, 12),
                       Qt.AlignmentFlag.AlignLeft, "4000 m")

        # 军标
        for fx, fy, yaw, side, kind, label, selected in self.UNITS:
            x, y = fx * w, fy * h
            col = side_color(side)
            if selected:
                p.setPen(QPen(QColor(C.SELECT), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(x - 17, y - 14, 34, 28))
            draw_unit_symbol(p, x, y, yaw, col, kind)
            if selected:
                f2 = p.font()
                f2.setPointSizeF(8.0)
                p.setFont(f2)
                p.setPen(QColor(C.SELECT))
                p.drawText(QPointF(x + 21, y - 3), label)
                f2.setPointSizeF(7.0)
                p.setFont(f2)
                p.setPen(QColor(C.TEXT))
                p.drawText(QPointF(x + 21, y + 9), f"{yaw:03d}°   12 m/s")

        # 指北针
        nx, ny = w - 30, 78
        p.setPen(QPen(QColor(C.TEXT), 1.3))
        p.drawLine(QPointF(nx, ny + 14), QPointF(nx, ny - 12))
        p.setBrush(QColor(C.TEXT))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([QPointF(nx, ny - 18), QPointF(nx - 3.5, ny - 11),
                                 QPointF(nx + 3.5, ny - 11)]))
        f2 = p.font()
        f2.setPointSizeF(7.0)
        p.setFont(f2)
        p.setPen(QColor(C.TEXT))
        p.drawText(QRectF(nx - 10, ny - 34, 20, 12),
                   Qt.AlignmentFlag.AlignCenter, "N")

        # 比例尺
        bar_px = 130
        bx, by = w - 46 - bar_px, hb - 24
        p.setPen(QPen(QColor(C.TEXT), 1))
        p.drawLine(QPointF(bx, by), QPointF(bx + bar_px, by))
        for xx in (bx, bx + bar_px // 2, bx + bar_px):
            p.drawLine(QPointF(xx, by - 4), QPointF(xx, by + 4))
        f2.setPointSizeF(7.0)
        p.setFont(f2)
        p.drawText(QRectF(bx - 24, by + 5, bar_px + 48, 12),
                   Qt.AlignmentFlag.AlignCenter,
                   "0            250            500 m")

        # 光标坐标
        p.drawText(QRectF(0, hb - 24, w, 14), Qt.AlignmentFlag.AlignCenter,
                   "X 1243 m    Y 881 m    标高 4 m")
        p.end()


# ======================================================================
# 浮层面板基类
# ======================================================================

class OverlayPanel(QWidget):
    """半透明浮层面板基类（子类决定在哪里铺底）。"""

    FILL_KEY = "PANEL_RGBA"

    def paintEvent(self, _event) -> None:
        pass


class Card(QWidget):
    """有内容才铺底的小卡片（没内容的地方保持透明）。"""

    def __init__(self, rgba_key: str = "PANEL_RGBA", parent=None) -> None:
        super().__init__(parent)
        self._key = rgba_key

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), rgba(getattr(C, self._key)))
        p.end()


class SectionHeader(QWidget):
    """分组标题栏（自绘三角）。"""

    def __init__(self, parent, title: str, expanded: bool, summary: str = "") -> None:
        super().__init__(parent)
        self._title = title
        self._summary = summary
        self._expanded = expanded
        self.setFixedHeight(21)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_summary(self, text: str) -> None:
        self._summary = text
        self.update()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), rgba(C.PANEL_2_RGBA))
        p.setPen(QPen(QColor(C.BORDER_DARK), 1))
        p.drawLine(QPointF(0, self.height() - 0.5),
                   QPointF(self.width(), self.height() - 0.5))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(C.TEXT_DIM))
        cx, cy = 9.0, self.height() / 2
        if self._expanded:
            p.drawPolygon(QPolygonF([QPointF(cx - 3.5, cy - 2),
                                     QPointF(cx + 3.5, cy - 2),
                                     QPointF(cx, cy + 2.5)]))
        else:
            p.drawPolygon(QPolygonF([QPointF(cx - 2, cy - 3.5),
                                     QPointF(cx - 2, cy + 3.5),
                                     QPointF(cx + 2.5, cy)]))

        f = p.font()
        f.setPointSizeF(7.5)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(C.TEXT))
        p.drawText(QRectF(18, 0, self.width() - 60, self.height()),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self._title)
        if self._summary:
            f.setBold(False)
            f.setPointSizeF(7.0)
            p.setFont(f)
            p.setPen(QColor(C.TEXT_DIM))
            p.drawText(QRectF(0, 0, self.width() - 5, self.height()),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self._summary)
        p.end()


# ======================================================================
# 侧栏
# ======================================================================

class UnitStatusPanel(OverlayPanel):
    def __init__(self, unit: dict, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDE_W)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        # 标题条（底色在 paintEvent 里画）
        head = QWidget(self)
        head.setFixedHeight(18)
        head.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(8, 0, 6, 0)
        t = QLabel("UNIT STATUS", head)
        t.setStyleSheet(
            f"color:{C.TEXT}; font-size:9px; font-weight:700;"
            "letter-spacing:1px; background:transparent;"
        )
        hl.addWidget(t)
        hl.addStretch(1)
        v.addWidget(head)

        # 卡片：工程图 + 名称
        card = Card("PANEL_RGBA", self)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(4, 4, 4, 5)
        cv.setSpacing(2)

        thumb = BlueprintThumb(unit, SIDE_W - 12, 92, card)
        cv.addWidget(thumb, 0, Qt.AlignmentFlag.AlignHCenter)

        name = QLabel(unit["name"], card)
        name.setStyleSheet(
            f"color:{C.TEXT}; font-size:12px; font-weight:700;"
            "background:transparent;"
        )
        cv.addWidget(name)

        sub = QLabel(f"{unit['side']} / {unit['role']} / {unit['status']}", card)
        sub.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:9px; background:transparent;"
        )
        cv.addWidget(sub)
        v.addWidget(card)

        # 分组
        self._sections: list[tuple[SectionHeader, QWidget]] = []
        v.addWidget(self._mk_basic(unit))
        v.addWidget(self._mk_status(unit))
        v.addWidget(self._mk_maneuver(unit))
        v.addWidget(self._mk_folded("武器与弹药", f"主炮 {unit['ammo'][0]}/{unit['ammo'][1]}",
                                    [("主炮 120mm", f"{unit['ammo'][0]}/{unit['ammo'][1]} 发"),
                                     ("并列机枪", "1200 发"),
                                     ("高射机枪", "900 发")]))
        v.addWidget(self._mk_folded("传感器 / EMCON", f"雷达 {unit['radar']}",
                                    [("火控雷达", unit["radar"]),
                                     ("热像仪", "工作"),
                                     ("激光告警", "待机")]))
        v.addWidget(self._mk_folded("编队", f"{unit['intent']} · 通信 {unit['comm'] * 100:.0f}%",
                                    [("意图", unit["intent"]),
                                     ("命令", unit["order"]),
                                     ("隶属", unit["formation"])]))
        v.addStretch(1)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(QRectF(0, 0, self.width(), 18), rgba(C.PANEL_2_RGBA))
        p.setPen(QPen(QColor(C.BORDER_DARK), 1))
        p.drawLine(QPointF(0, 17.5), QPointF(self.width(), 17.5))
        p.end()

    # ── 分组构造 ──

    def _section(self, title: str, summary: str, expanded: bool):
        holder = QWidget(self)
        holder.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        hdr = SectionHeader(holder, title, expanded, summary)
        body = Card("PANEL_RGBA", holder)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(8, 4, 8, 6)
        bl.setSpacing(2)
        body.setVisible(expanded)

        def _toggle(_e=None):
            expanded_new = not body.isVisible()
            body.setVisible(expanded_new)
            hdr.set_expanded(expanded_new)

        hdr.mousePressEvent = _toggle  # type: ignore[method-assign]
        lay.addWidget(hdr)
        lay.addWidget(body)
        return holder, bl

    def _kv2(self, bl, k1, v1, k2, v2, c1=None, c2=None):
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        for key, val, col in ((k1, v1, c1), (k2, v2, c2)):
            cell = QWidget(row)
            cell.setStyleSheet("background:transparent;")
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(3)
            a = QLabel(key, cell)
            a.setFixedWidth(24)
            a.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:9px; background:transparent;")
            b = QLabel(val, cell)
            b.setStyleSheet(
                f"color:{col or C.TEXT}; font-size:9px; font-weight:700;"
                "background:transparent;"
            )
            cl.addWidget(a)
            cl.addWidget(b)
            cl.addStretch(1)
            hl.addWidget(cell, 1)
        bl.addWidget(row)

    def _mk_basic(self, unit: dict) -> QWidget:
        holder, bl = self._section("基本状态", "", True)
        self._kv2(bl, "阵营", "北约", "航向", f"{unit['course']}°", C.NATO)
        self._kv2(bl, "速度", f"{unit['speed']:.0f} m/s", "标高", "4 m")
        self._kv2(bl, "坐标", "1243/881", "隶属", "ALPHA")
        return holder

    def _mk_status(self, unit: dict) -> QWidget:
        holder, bl = self._section("状态", f"{unit['readiness']} · {unit['readiness_cn']}", True)

        row = QWidget()
        row.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 2)
        rl.setSpacing(5)
        rk = QLabel("战备", row)
        rk.setFixedWidth(24)
        rk.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:9px; background:transparent;")
        rv = QLabel(unit["readiness"], row)
        rv.setStyleSheet(
            f"color:{C.WARN}; background:{C.PANEL_2}; font-size:9px;"
            f"font-weight:700; border:1px solid {C.BORDER_DARK}; padding:0 3px;"
        )
        rn = QLabel(unit["readiness_cn"], row)
        rn.setStyleSheet(f"color:{C.WARN}; font-size:9px; background:transparent;")
        rl.addWidget(rk)
        rl.addWidget(rv)
        rl.addWidget(rn)
        rl.addStretch(1)
        bl.addWidget(row)

        for name, state, note, ckey in unit["systems"]:
            r = QWidget()
            r.setStyleSheet("background:transparent;")
            rl2 = QHBoxLayout(r)
            rl2.setContentsMargins(0, 0, 0, 0)
            rl2.setSpacing(5)
            n = QLabel(name, r)
            n.setFixedWidth(26)
            n.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:9px; background:transparent;")
            s = QLabel(state, r)
            s.setFixedWidth(30)
            s.setStyleSheet(
                f"color:{getattr(C, ckey)}; font-size:9px; font-weight:700;"
                "background:transparent;"
            )
            note_l = QLabel(note, r)
            note_l.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:8px; background:transparent;")
            rl2.addWidget(n)
            rl2.addWidget(s)
            rl2.addWidget(note_l, 1)
            bl.addWidget(r)
        return holder

    def _mk_maneuver(self, unit: dict) -> QWidget:
        holder, bl = self._section("机动", "巡航", True)
        grid = QWidget()
        grid.setStyleSheet("background:transparent;")
        gl = QHBoxLayout(grid)
        gl.setContentsMargins(0, 1, 0, 1)
        gl.setSpacing(2)
        for txt, active in (("停止", False), ("慢速", False),
                            ("巡航", True), ("全速", False), ("最大", False)):
            b = QPushButton(txt, grid)
            b.setFixedHeight(19)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(_btn_qss(active))
            gl.addWidget(b)
        bl.addWidget(grid)
        return holder

    def _mk_folded(self, title: str, summary: str, pairs: list[tuple[str, str]]) -> QWidget:
        holder, bl = self._section(title, summary, False)
        for k, v in pairs:
            r = QWidget()
            r.setStyleSheet("background:transparent;")
            rl = QHBoxLayout(r)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            a = QLabel(k, r)
            a.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:9px; background:transparent;")
            b = QLabel(v, r)
            b.setStyleSheet(f"color:{C.TEXT}; font-size:9px; background:transparent;")
            b.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            rl.addWidget(a)
            rl.addStretch(1)
            rl.addWidget(b)
            bl.addWidget(r)
        return holder


def _btn_qss(active: bool) -> str:
    if active:
        return (
            f"QPushButton{{background:{C.ACCENT}; color:{C.ACCENT_TEXT};"
            f"border:1px solid {C.BORDER_DARK}; font-size:9px; font-weight:700;}}"
        )
    return (
        f"QPushButton{{background:{C.PANEL_2}; color:{C.TEXT};"
        f"border:1px solid {C.BORDER_DARK}; font-size:9px;}}"
        f"QPushButton:hover{{background:{C.BORDER};}}"
    )


# ======================================================================
# 顶部时间栏（CMO 样式：单行信息条 + Selected 行）
# ======================================================================

class TimeBar(QWidget):
    """顶部时间栏：一个主时间大字 + 若干信息分块 + Selected 行。"""

    BLOCKS = [
        ("地点", "东欧 / 第聂伯河下游"),
        ("日期", "2060-03-14 周四"),
        ("ZULU", "05:45:44"),
        ("剩余", "23 时 59 分"),
        ("高度", "6818 m"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(TIME_H)

    def _shadow(self, p: QPainter, rect: QRectF, text: str,
                color: QColor, align=Qt.AlignmentFlag.AlignLeft) -> None:
        """带阴影的文字（底更透时保证可读）。"""
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(rect.translated(1, 1), align, text)
        p.setPen(color)
        p.drawText(rect, align, text)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)

        # ── 先量内容宽度：没字的地方不铺底 ──
        fm_v = QFontMetrics(QFont("Consolas", 0))
        f_v = QFont("Consolas")
        f_v.setPointSizeF(9.5)
        f_v.setBold(True)
        fm_v = QFontMetrics(f_v)
        f_l = QFont("Microsoft YaHei")
        f_l.setPointSizeF(7.5)
        fm_l = QFontMetrics(f_l)

        x_end = 160.0
        for label, value in self.BLOCKS:
            x_end += max(fm_l.horizontalAdvance(label),
                         fm_v.horizontalAdvance(value)) + 21
        content_w = min(x_end + 6, float(self.width()))

        # ── 仅文字区域一层淡底 ──
        p.fillRect(QRectF(0, 0, content_w, self.height()),
                   QColor(18, 22, 26, 72))

        # ── 主时间大字 ──
        f = QFont("Consolas")
        f.setPointSizeF(20.0)
        f.setBold(True)
        p.setFont(f)
        self._shadow(p, QRectF(10, 3, 148, 28), "07:45:44",
                     QColor(255, 255, 255, 248),
                     Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        f = QFont("Microsoft YaHei")
        f.setPointSizeF(7.5)
        p.setFont(f)
        self._shadow(p, QRectF(12, 30, 140, 12), "当地时间",
                     QColor(205, 214, 224, 215),
                     Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # ── 信息分块 ──
        x = 160.0
        for label, value in self.BLOCKS:
            f.setPointSizeF(7.5)
            f.setBold(False)
            p.setFont(f)
            lw = p.fontMetrics().horizontalAdvance(label)
            bw = max(lw, fm_v.horizontalAdvance(value))
            self._shadow(p, QRectF(x, 9, bw, 13), label,
                         QColor(198, 209, 220, 220))

            f.setPointSizeF(9.5)
            f.setBold(True)
            p.setFont(f)
            self._shadow(p, QRectF(x, 22, bw, 16), value,
                         QColor(248, 250, 252, 248))

            x += bw + 13
            p.setPen(QPen(QColor(255, 255, 255, 55), 1))
            p.drawLine(QPointF(x - 6, 12), QPointF(x - 6, 40))
            x += 8

        # ── 第二行：Selected ──
        f = QFont("Microsoft YaHei")
        f.setPointSizeF(8.0)
        p.setFont(f)
        self._shadow(p, QRectF(160, 39, self.width() - 172, 12),
                     "Selected:  1x M1A2 #03    ·    第 7 装甲连 / ALPHA",
                     QColor(214, 222, 230, 230))

        p.end()


# ======================================================================
# 底部：接触列表 + 消息日志
# ======================================================================

class BottomBar(OverlayPanel):
    FILL_KEY = "LOG_RGBA"

    COLS = [("ID", 46), ("类型", 64), ("方位", 54), ("距离", 68),
            ("速度", 60), ("威胁", 42)]
    ROWS = [
        ("C-01", "T-14",  "145°", "2100 m", "18 m/s", "高"),
        ("C-02", "T-14",  "162°", "2400 m", " 0 m/s", "中"),
        ("C-03", "2S19",  "128°", "3200 m", "12 m/s", "高"),
        ("C-04", "T-14",  "171°", "3700 m", "22 m/s", "低"),
        ("C-05", "未知",  " 98°", "4100 m", "  —   ", "低"),
    ]
    THREAT_KEY = {"高": "PACT", "中": "WARN", "低": "OK"}
    LOG = [
        ("destroyed", "12.9s  华约 T-14 #07 被击毁（弹药架殉爆）"),
        ("hit",       "12.3s  北约 M1A2 #03 命中 T-14 #07（未击穿）"),
        ("fire",      "12.3s  北约 M1A2 #03 主炮开火 120 mm APFSDS"),
        ("detection", "11.8s  北约 M1A2 #03 雷达发现 2 个接触"),
        ("command",   "10.1s  玩家指令：第 7 装甲连 转移至 (1380, 940)"),
    ]
    LOG_KEY = {"destroyed": "PACT", "hit": "WARN", "fire": "TEXT",
               "detection": "NATO", "command": "OK"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(BOTTOM_H)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        split_x = 470
        gap = 6

        # 两块独立卡片（中间透出地图）
        p.fillRect(QRectF(0, 0, split_x, h), rgba(C.LOG_RGBA))
        p.fillRect(QRectF(split_x + gap, 0, w - split_x - gap, h),
                   rgba(C.LOG_RGBA))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # ── 接触列表 ──
        f = p.font()
        f.setPointSizeF(7.0)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(C.TEXT_DIM))
        p.drawText(QRectF(8, 1, 200, 12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   "CONTACTS  5")

        f.setBold(False)
        f.setPointSizeF(7.0)
        p.setFont(f)
        x = 8
        for name, wdt in self.COLS:
            p.setPen(QColor(C.TEXT_DIM))
            p.drawText(QRectF(x, 12, wdt, 11), Qt.AlignmentFlag.AlignLeft, name)
            x += wdt
        p.setPen(QPen(QColor(C.BORDER), 1))
        p.drawLine(QPointF(6, 24), QPointF(split_x - 6, 24))

        f.setFamily("Consolas")
        f.setPointSizeF(8.0)
        p.setFont(f)
        for i, row in enumerate(self.ROWS):
            ry = 26 + i * 13
            if i % 2 == 1:
                p.fillRect(QRectF(4, ry - 1, split_x - 10, 13),
                           QColor(0, 0, 0, 12))
            x = 8
            for (name, wdt), val in zip(self.COLS, row):
                if name == "威胁":
                    f.setBold(True)
                    p.setFont(f)
                    p.setPen(QColor(getattr(C, self.THREAT_KEY.get(val, "TEXT"))))
                else:
                    f.setBold(False)
                    p.setFont(f)
                    p.setPen(QColor(C.TEXT))
                p.drawText(QRectF(x, ry, wdt, 12), Qt.AlignmentFlag.AlignLeft, val)
                x += wdt

        # ── 消息日志（右卡片）──
        lx = split_x + gap + 8
        f.setBold(True)
        f.setPointSizeF(7.0)
        p.setFont(f)
        p.setPen(QColor(C.TEXT_DIM))
        p.drawText(QRectF(lx, 1, 220, 12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   "MESSAGE LOG")
        f.setBold(False)
        f.setFamily("Microsoft YaHei")
        f.setPointSizeF(8.0)
        p.setFont(f)
        for i, (kind, text) in enumerate(self.LOG):
            p.setPen(QColor(getattr(C, self.LOG_KEY.get(kind, "TEXT"))))
            p.drawText(QRectF(lx, 13 + i * 15, w - lx - 10, 12),
                       Qt.AlignmentFlag.AlignLeft, text)
        p.end()


# ======================================================================
# 驾驶舱（地图铺满 + 浮层布局）
# ======================================================================

class Cockpit(QWidget):
    def __init__(self, theme: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")

        self.map = Battlefield(self)
        self.timebar = TimeBar(self)
        self.side = UnitStatusPanel(UNIT, self)
        self.bottom = BottomBar(self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        self.map.safe_bottom = BOTTOM_H
        self.map.setGeometry(0, 0, w, h)
        self.timebar.setGeometry(0, 0, w - SIDE_W, TIME_H)
        self.side.setGeometry(w - SIDE_W, 0, SIDE_W, h - BOTTOM_H)
        self.bottom.setGeometry(0, h - BOTTOM_H, w, BOTTOM_H)
        self.map.lower()
        self.timebar.raise_()
        self.side.raise_()
        self.bottom.raise_()


class DemoWindow(QMainWindow):
    def __init__(self, theme: str = "light") -> None:
        super().__init__()
        self._theme = theme
        self.setWindowTitle("Arkady in Cradle — C4I 界面原型")
        self.resize(1440, 900)
        self.setStyleSheet(f"background:{C.BG}; color:{C.TEXT};")

        root = QWidget(self)
        self.setCentralWidget(root)
        vl = QVBoxLayout(root)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        vl.addWidget(self._build_menubar(root))
        vl.addWidget(self._build_actionbar(root))
        vl.addWidget(Cockpit(theme, root), 1)
        vl.addWidget(self._build_statusbar(root))

    def _build_menubar(self, parent) -> QWidget:
        bar = QWidget(parent)
        bar.setFixedHeight(22)
        bar.setStyleSheet(
            f"background:{C.PANEL_2}; border-bottom:1px solid {C.BORDER_DARK};"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(6, 0, 6, 0)
        hl.setSpacing(14)
        for txt in ("文件", "地图", "单位指令", "接触", "任务", "参考点",
                    "视图", "帮助"):
            l = QLabel(txt, bar)
            l.setStyleSheet(f"color:{C.TEXT}; font-size:11px; background:transparent;")
            hl.addWidget(l)
        hl.addStretch(1)
        l = QLabel("C4I 原型 / v0.2", bar)
        l.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:10px; background:transparent;"
        )
        hl.addWidget(l)
        return bar

    def _build_actionbar(self, parent) -> QWidget:
        bar = QWidget(parent)
        bar.setFixedHeight(26)
        bar.setStyleSheet(
            f"background:{C.PANEL_2}; border-bottom:1px solid {C.BORDER_DARK};"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(6, 2, 6, 2)
        hl.setSpacing(3)
        items = [
            ("自动交战", False), ("手动交战", True), ("规划航线", True),
            ("油门/高度", True), ("编队", True), ("弹药", True),
            ("武器", True), ("传感器", True), ("系统损伤", True),
            ("条令", True),
        ]
        for txt, enabled in items:
            b = QToolButton(bar)
            b.setText(txt)
            b.setFixedHeight(20)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if enabled:
                b.setStyleSheet(_btn_qss(False) + "QToolButton{padding:0 7px;}")
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:{C.PANEL_2}; color:{C.BORDER};"
                    f"border:1px solid {C.BORDER}; padding:0 7px; font-size:10px;}}"
                )
                b.setEnabled(False)
            hl.addWidget(b)
        hl.addStretch(1)
        return bar

    def _build_statusbar(self, parent) -> QWidget:
        bar = QWidget(parent)
        bar.setFixedHeight(20)
        bar.setStyleSheet(
            f"background:{C.PANEL_2}; border-top:1px solid {C.BORDER_DARK};"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(8, 0, 8, 0)
        hl.setSpacing(0)
        cells = [
            ("就绪", 80), ("单位 13", 100), ("FPS 60", 90),
            (f"主题 {self._theme}", 110), ("相机 d=1200", 140),
        ]
        for txt, wdt in cells:
            l = QLabel(txt, bar)
            l.setFixedWidth(wdt)
            l.setStyleSheet(
                f"color:{C.TEXT}; font-size:10px; background:transparent;"
                f"border-right:1px solid {C.BORDER}; padding-left:6px;"
            )
            hl.addWidget(l)
        hl.addStretch(1)
        return bar


def _load_fonts(app: QApplication) -> None:
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"):
        if os.path.exists(path):
            fid = QFontDatabase.addApplicationFont(path)
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                app.setFont(QFont(fams[0], 9))
                return


def main() -> int:
    theme = "light"
    if "--theme" in sys.argv:
        i = sys.argv.index("--theme")
        if i + 1 < len(sys.argv):
            theme = sys.argv[i + 1]
    apply_theme(theme)

    app = QApplication(sys.argv)
    _load_fonts(app)

    win = DemoWindow(theme)
    win.show()

    if "--shot" in sys.argv:
        i = sys.argv.index("--shot")
        out = sys.argv[i + 1] if i + 1 < len(sys.argv) else f"prototype/output/c4i_{theme}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        app.processEvents()
        pm = win.grab()
        pm.save(out)
        print(f"saved {out}  {pm.width()}x{pm.height()}")
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
