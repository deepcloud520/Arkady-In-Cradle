"""AIC UI 原型：CMO 风格侧栏 + 军用蓝图缩略图。

独立演示程序，**不接入主项目**（不 import Airia.*，不改任何现有文件）。
目的是在动正式代码之前，把「CMO 感」的视觉效果先做出来看一眼。

运行（正常窗口）:
    .venv\\Scripts\\python.exe prototype/ui_cmo_demo.py

离屏截图（无窗口，直接出 PNG）:
    .venv\\Scripts\\python.exe prototype/ui_cmo_demo.py --shot prototype/output/ui_demo.png

对应设计文档: docs/ui-cmo-redesign.md
    §3   A 线：折叠分组侧栏
    §3.5 军用技术手册/蓝图缩略图（宽幅侧视图）
    §4.2 朝向楔
    §4.5 俯视冲突：模型自动/手动三态
    §5.1 时间压缩条
    §5.2 消息日志着色
"""
from __future__ import annotations

import math
import os
import sys

# 离屏渲染必须在导入 Qt 之前设置
if "--shot" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ======================================================================
# 配色（与 docs/ui-cmo-redesign.md §3.5.1 一致）
# ======================================================================

class C:
    BG = "#0a0a0f"
    PANEL = "#11141c"
    PANEL_2 = "#161b26"
    BORDER = "#232a38"
    TEXT = "#d8dce2"
    TEXT_DIM = "#8b93a3"
    CYAN = "#9fd8e8"
    GRID = "#1c2733"
    NATO = "#4dd97a"
    PACT = "#e63329"
    NEUTRAL = "#a6a6a6"
    WARN = "#e0a040"
    ACCENT = "#e8c85a"
    BLUEPRINT_BG = "#0d1117"


SIDE_COLOR = {"北约": C.NATO, "华约": C.PACT, "中立": C.NEUTRAL}


# ======================================================================
# 假数据（原型专用；正式实现读 WorldAdapter）
# ======================================================================

UNIT = {
    "name": "M1A2 #03",
    "side": "北约",
    "role": "主战坦克",
    "status": "战斗中",
    "course": 32,
    "speed": 12.0,
    "pos": (1243, 881, 4),
    "health": 62.0,
    "parts": [
        ("车体", 600, True),
        ("炮塔", 400, True),
        ("引擎", 0, False),
        ("履带", 0, False),
        ("弹药架", 250, True),
    ],
    "ammo": (38, 40),
    "fuel": 0.82,
    "radar": "主动",
    "formation": "第 7 装甲连 · Alpha",
    "comm": 0.87,
    "intent": "进攻",
    "order": "机动",
    "events": [
        "[12.3s] 命中 T-14 #07（未击穿）",
        "[11.8s] 引擎受损（起火风险）",
        "[10.4s] 履带被毁（无法机动）",
    ],
}

# 蓝图尺寸（米）——正式实现从 data/units/*.yaml 读
BP = {
    "total_len": 9.8,
    "hull_len": 7.9,
    "hull_h": 1.45,
    "turret_len": 3.4,
    "turret_h": 1.05,
    "wheel_d": 0.72,
    "wheels": 6,
}


# ======================================================================
# 蓝图缩略图（军用技术手册风格 · 宽幅侧视图）
# ======================================================================

def draw_blueprint(painter: QPainter, rect: QRectF, unit: dict) -> None:
    """在 rect 内绘制「军用技术手册」风格的装备侧视图（宽幅）。

    Args:
        painter: 已开启抗锯齿的画笔。
        rect: 目标矩形（建议 190×110 左右，宽幅）。
        unit: 单位数据（取 side 决定阵营条颜色）。
    """
    painter.save()
    painter.setClipRect(rect)
    painter.fillRect(rect, QColor(C.BLUEPRINT_BG))

    # ── 网格：16px 细 / 64px 粗 ──
    painter.setPen(QPen(QColor(C.GRID), 1))
    gx = rect.left() - (rect.left() % 16)
    while gx < rect.right():
        painter.drawLine(QPointF(gx, rect.top()), QPointF(gx, rect.bottom()))
        gx += 16
    gy = rect.top() - (rect.top() % 16)
    while gy < rect.bottom():
        painter.drawLine(QPointF(rect.left(), gy), QPointF(rect.right(), gy))
        gy += 16

    painter.setPen(QPen(QColor("#243040"), 1))
    gx = rect.left() - (rect.left() % 64)
    while gx < rect.right():
        painter.drawLine(QPointF(gx, rect.top()), QPointF(gx, rect.bottom()))
        gx += 64
    gy = rect.top() - (rect.top() % 64)
    while gy < rect.bottom():
        painter.drawLine(QPointF(rect.left(), gy), QPointF(rect.right(), gy))
        gy += 64

    # ── 绘图布局 ──
    pad_l, pad_r = 8.0, 10.0
    top_label = rect.top() + 14.0        # 角标下方
    bot_data = rect.bottom() - 15.0      # 数据栏上方
    draw_w = rect.width() - pad_l - pad_r
    draw_h = bot_data - top_label

    s = draw_w / BP["total_len"]         # px / m
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

    line = QPen(QColor(C.CYAN), 1.2)
    line.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    outline = QPen(QColor(C.CYAN), 2.0)
    outline.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    faint = QPen(QColor(C.CYAN).darker(150), 1.0)

    # ── 履带（车体下方一条带）──
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    track = QRectF(
        hull_left + 2, hull_bottom - wheel_r * 0.9,
        hull_l - 4, wheel_r * 2.9,
    )
    painter.drawRoundedRect(track, wheel_r * 0.9, wheel_r * 0.9)

    # 负重轮
    painter.setPen(line)
    n = BP["wheels"]
    for i in range(n):
        cx = track.left() + track.width() * (i + 0.5) / n
        painter.drawEllipse(
            QPointF(cx, track.center().y()), wheel_r * 0.85, wheel_r * 0.85
        )

    # ── 车体 ──
    body = QPainterPath()
    # 前装甲斜角（右侧为车头）
    body.moveTo(hull_left, hull_top + hull_h * 0.15)
    body.lineTo(hull_left + hull_l * 0.06, hull_top)
    body.lineTo(hull_left + hull_l * 0.92, hull_top)
    body.lineTo(hull_left + hull_l, hull_top + hull_h * 0.45)
    body.lineTo(hull_left + hull_l * 0.97, hull_bottom)
    body.lineTo(hull_left + hull_l * 0.02, hull_bottom)
    body.lineTo(hull_left, hull_top + hull_h * 0.15)
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    painter.drawPath(body)

    # 发动机格栅（车尾，左侧）
    painter.setPen(faint)
    for i in range(4):
        yy = hull_top + hull_h * (0.28 + i * 0.14)
        painter.drawLine(
            QPointF(hull_left + hull_l * 0.06, yy),
            QPointF(hull_left + hull_l * 0.22, yy),
        )

    # ── 炮塔（车体中后部）──
    tur_left = hull_left + hull_l * 0.30
    tur_top = hull_top - tur_h
    turret = QPainterPath()
    turret.moveTo(tur_left, hull_top + 1)
    turret.lineTo(tur_left + tur_l * 0.10, tur_top + tur_h * 0.25)
    turret.lineTo(tur_left + tur_l * 0.55, tur_top)
    turret.lineTo(tur_left + tur_l * 0.88, tur_top + tur_h * 0.12)
    turret.lineTo(tur_left + tur_l, tur_top + tur_h * 0.75)
    turret.lineTo(tur_left + tur_l * 0.96, hull_top + 1)
    turret.lineTo(tur_left, hull_top + 1)
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    painter.drawPath(turret)

    # 车长塔
    painter.setPen(line)
    painter.drawLine(
        QPointF(tur_left + tur_l * 0.30, tur_top + tur_h * 0.25),
        QPointF(tur_left + tur_l * 0.30, tur_top),
    )
    painter.drawLine(
        QPointF(tur_left + tur_l * 0.42, tur_top),
        QPointF(tur_left + tur_l * 0.42, tur_top + tur_h * 0.18),
    )
    painter.drawLine(
        QPointF(tur_left + tur_l * 0.30, tur_top),
        QPointF(tur_left + tur_l * 0.42, tur_top),
    )

    # ── 炮管（炮塔前方水平伸出到总长）──
    gun_y = tur_top + tur_h * 0.55
    gun_w = max(2.0, tur_h * 0.20)
    gun_start = tur_left + tur_l * 0.96
    gun_end = hull_left + BP["total_len"] * s
    painter.setPen(outline)
    painter.setBrush(QColor(C.BLUEPRINT_BG))
    painter.drawRect(QRectF(gun_start, gun_y - gun_w / 2, gun_end - gun_start, gun_w))
    # 炮口制退器
    painter.drawRect(
        QRectF(gun_end - gun_w * 2.4, gun_y - gun_w * 1.1, gun_w * 2.4, gun_w * 2.2)
    )

    # ── 总长尺寸线（车体上方）──
    dim_y = max(tur_top - 7, top_label + 4)
    painter.setPen(QPen(QColor(C.CYAN).darker(125), 1))
    painter.drawLine(QPointF(hull_left, dim_y), QPointF(gun_end, dim_y))
    for xx in (hull_left, gun_end):
        painter.drawLine(QPointF(xx, dim_y - 3), QPointF(xx, dim_y + 3))
    painter.drawLine(QPointF(hull_left + 2, dim_y + 3), QPointF(hull_left, hull_top))
    painter.drawLine(QPointF(gun_end, dim_y + 3), QPointF(gun_end, gun_y - gun_w))

    f = painter.font()
    f.setFamily("Consolas")
    f.setPointSizeF(6.2)
    painter.setFont(f)
    painter.setPen(QColor(C.CYAN))
    painter.drawText(
        QRectF(hull_left + 40, dim_y - 1, gun_end - hull_left - 80, 10),
        Qt.AlignmentFlag.AlignCenter,
        "9.8 m",
    )

    # ── 角标（右上，避开左侧尺寸标注）──
    f.setFamily("Consolas")
    f.setPointSizeF(6.5)
    painter.setFont(f)
    painter.setPen(QColor(C.TEXT_DIM))
    painter.drawText(
        QRectF(rect.left(), rect.top() + 2, rect.width() - 6, 11),
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        "AIC-DB",
    )

    # ── 阵营竖条 ──
    painter.fillRect(
        QRectF(rect.left() + 5, rect.bottom() - 13, 3.0, 10),
        QColor(SIDE_COLOR.get(unit["side"], C.NEUTRAL)),
    )

    # ── 数据栏（单行）──
    f.setPointSizeF(6.5)
    painter.setFont(f)
    painter.setPen(QColor(C.TEXT_DIM))
    painter.drawText(
        QRectF(rect.left() + 12, rect.bottom() - 14, rect.width() - 16, 12),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        "M1A2 ABRAMS   ·   120 mm   ·   CREW 4",
    )

    painter.restore()


class BlueprintThumb(QLabel):
    """蓝图缩略图控件（宽幅，定尺渲染）。"""

    def __init__(self, unit: dict, w: int = 190, h: int = 110, parent=None) -> None:
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
# 折叠分组（CMO 侧栏的 accordion）
# ======================================================================

class CollapsibleSection(QWidget):
    """可折叠分组：标题栏（三角 + 标题 + 右侧摘要）+ 内容区。"""

    def __init__(self, title: str, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QWidget(self)
        self._header.setFixedHeight(26)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(
            f"background:{C.PANEL_2}; border-top:1px solid {C.BORDER};"
        )
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(8, 0, 8, 0)
        hl.setSpacing(6)

        self._arrow = _Triangle(self._header, expanded, C.CYAN)

        self._title = QLabel(title, self._header)
        self._title.setStyleSheet(
            f"color:{C.TEXT}; font-size:12px; font-weight:600; background:transparent;"
        )

        self._summary = QLabel("", self._header)
        self._summary.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:11px; background:transparent;"
        )
        self._summary.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        hl.addWidget(self._arrow)
        hl.addWidget(self._title)
        hl.addStretch(1)
        hl.addWidget(self._summary)

        self._body = QWidget(self)
        self._body.setStyleSheet(f"background:{C.PANEL};")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(10, 8, 10, 9)
        self._body_layout.setSpacing(5)

        outer.addWidget(self._header)
        outer.addWidget(self._body)
        self._body.setVisible(expanded)

        self._header.mousePressEvent = self._on_header_click  # type: ignore[method-assign]

    def _on_header_click(self, _event) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._arrow.set_expanded(expanded)

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def add_row(self, text: str, dim: bool = False) -> QLabel:
        lbl = QLabel(text, self._body)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color:{C.TEXT_DIM if dim else C.TEXT}; font-size:12px;"
            "background:transparent;"
        )
        self._body_layout.addWidget(lbl)
        return lbl

    def add_kv(self, key: str, value: str, color: str | None = None) -> None:
        row = QWidget(self._body)
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        k = QLabel(key, row)
        k.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:12px; background:transparent;")

        v = QLabel(value, row)
        v.setStyleSheet(
            f"color:{color or C.TEXT}; font-size:12px; font-weight:600;"
            "background:transparent;"
        )
        v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        hl.addWidget(k)
        hl.addStretch(1)
        hl.addWidget(v)
        self._body_layout.addWidget(row)

    def add_bar(self, label: str, ratio: float, color: str, tail: str = "") -> None:
        row = QWidget(self._body)
        row.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        k = QLabel(label, row)
        k.setFixedWidth(38)
        k.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:12px; background:transparent;")

        bar = RatioBar(row, ratio, color)

        t = QLabel(tail, row)
        t.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:11px; background:transparent;")
        t.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        hl.addWidget(k)
        hl.addWidget(bar, 1)
        hl.addWidget(t)
        self._body_layout.addWidget(row)


class _Triangle(QWidget):
    """自绘折叠三角（避免字符字体缺字变方框）。"""

    def __init__(self, parent, expanded: bool, color: str) -> None:
        super().__init__(parent)
        self._expanded = expanded
        self._color = color
        self.setFixedSize(10, 10)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._color))
        w, h = self.width(), self.height()
        if self._expanded:
            p.drawPolygon(QPolygonF([
                QPointF(1, 3), QPointF(w - 1, 3), QPointF(w / 2, h - 2)
            ]))
        else:
            p.drawPolygon(QPolygonF([
                QPointF(3, 1), QPointF(3, h - 1), QPointF(w - 2, h / 2)
            ]))
        p.end()


class RatioBar(QWidget):
    """极简比例条（细，CMO 风格）。"""

    def __init__(self, parent, ratio: float, color: str) -> None:
        super().__init__(parent)
        self._ratio = max(0.0, min(1.0, ratio))
        self._color = color
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect())
        p.fillRect(r, QColor("#0b0e14"))
        p.setPen(QPen(QColor(C.BORDER), 1))
        p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
        if self._ratio > 0:
            p.fillRect(
                QRectF(1, 1, (r.width() - 2) * self._ratio, r.height() - 2),
                QColor(self._color),
            )
        p.end()


# ======================================================================
# 战场占位（QPainter 画的示意场景，不接 3D 引擎）
# ======================================================================

class BattlefieldPlaceholder(QWidget):
    """示意战场：地形底 + 朝向楔单位 + 射程圈 + 比例尺 + 光标坐标。"""

    # (fx, fy, yaw_deg, side, label, selected) —— 相对坐标，铺满视野
    UNITS = [
        (0.28, 0.40, 32, "北约", "M1A2 #03", True),
        (0.21, 0.55, 318, "北约", "M1A2 #05", False),
        (0.33, 0.28, 18, "北约", "M109 #02", False),
        (0.37, 0.72, 75, "北约", "M1A2 #07", False),
        (0.18, 0.80, 12, "北约", "M1A2 #09", False),
        (0.62, 0.24, 155, "华约", "T-14 #07", False),
        (0.71, 0.38, 200, "华约", "T-14 #09", False),
        (0.56, 0.50, 128, "华约", "2S19 #01", False),
        (0.79, 0.62, 240, "华约", "T-14 #11", False),
        (0.86, 0.80, 300, "华约", "T-14 #13", False),
        (0.50, 0.14, 190, "华约", "T-14 #04", False),
        (0.44, 0.86, 40, "中立", "卡车队", False),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(600, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # ── 地形底 ──
        g = QLinearGradient(0, 0, w, h)
        g.setColorAt(0.0, QColor("#1a241d"))
        g.setColorAt(0.45, QColor("#141c17"))
        g.setColorAt(1.0, QColor("#0d1411"))
        p.fillRect(self.rect(), QBrush(g))

        p.setPen(Qt.PenStyle.NoPen)
        blobs = [
            (0.16, 0.14, 210, QColor(38, 62, 44, 110)),
            (0.78, 0.22, 180, QColor(32, 54, 40, 100)),
            (0.52, 0.86, 260, QColor(42, 48, 34, 90)),
            (0.90, 0.74, 150, QColor(24, 40, 32, 100)),
            (0.34, 0.60, 130, QColor(46, 60, 38, 70)),
        ]
        for fx, fy, rad, col in blobs:
            p.setBrush(col)
            p.drawEllipse(QPointF(fx * w, fy * h), rad, rad * 0.72)

        # 等高线
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(8):
            p.setPen(QPen(QColor(56, 78, 64, 34), 1))
            p.drawEllipse(QRectF(
                0.28 * w - i * 30, 0.40 * h - i * 20,
                260 + i * 60, 200 + i * 42,
            ))

        # 网格（很淡，避免棋盘感）
        p.setPen(QPen(QColor(130, 170, 158, 7), 1))
        for x in range(0, w, 64):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, 64):
            p.drawLine(0, y, w, y)

        # ── 射程圈（选中单位）──
        sel = next((u for u in self.UNITS if u[5]), None)
        if sel is not None:
            sx, sy = sel[0] * w, sel[1] * h
            p.setPen(QPen(QColor(232, 200, 90, 130), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = min(w, h) * 0.19
            p.drawEllipse(QPointF(sx, sy), r, r)

        # ── 单位 ──
        for fx, fy, yaw, side, label, selected in self.UNITS:
            x, y = fx * w, fy * h
            col = QColor(SIDE_COLOR[side])
            rad = math.radians(yaw)
            dx, dy = math.sin(rad), -math.cos(rad)
            size = 12.0
            tip = QPointF(x + dx * size, y + dy * size)
            perp = QPointF(-dy, dx)
            base = QPointF(x - dx * size * 0.55, y - dy * size * 0.55)
            left = QPointF(base.x() + perp.x() * size * 0.70,
                           base.y() + perp.y() * size * 0.70)
            right = QPointF(base.x() - perp.x() * size * 0.70,
                            base.y() - perp.y() * size * 0.70)

            if selected:
                p.setPen(QPen(QColor(C.ACCENT), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(x, y), size * 1.8, size * 1.8)

            p.setPen(QPen(col.lighter(160), 1.4))
            p.setBrush(col)
            p.drawPolygon(QPolygonF([tip, left, right]))

            if selected:
                f = p.font()
                f.setPointSizeF(8.5)
                p.setFont(f)
                p.setPen(QColor(C.ACCENT))
                p.drawText(QPointF(x + 26, y - 4), label)
                f.setPointSizeF(7.2)
                p.setFont(f)
                p.setPen(QColor(C.TEXT_DIM))
                p.drawText(QPointF(x + 26, y + 9), f"{yaw:03d}°  12 m/s")

        # ── 比例尺 ──
        bar_px = 130
        bx, by = w - 46 - bar_px, h - 30
        p.setPen(QPen(QColor(C.TEXT_DIM), 1))
        p.drawLine(QPointF(bx, by), QPointF(bx + bar_px, by))
        for xx in (bx, bx + bar_px // 2, bx + bar_px):
            p.drawLine(QPointF(xx, by - 4), QPointF(xx, by + 4))
        f = p.font()
        f.setPointSizeF(7.0)
        p.setFont(f)
        p.drawText(
            QRectF(bx - 20, by + 5, bar_px + 40, 12),
            Qt.AlignmentFlag.AlignCenter,
            "0            250            500 m",
        )

        # ── 光标坐标 ──
        p.setPen(QColor(C.TEXT_DIM))
        p.drawText(
            QRectF(0, h - 30, w, 14),
            Qt.AlignmentFlag.AlignCenter,
            "X 1243 m    Y 881 m    地形 4 m    ·    比例 1:4 000",
        )

        # ── 相机信息 ──
        p.drawText(
            QRectF(w - 230, 10, 218, 14),
            Qt.AlignmentFlag.AlignRight,
            "CAM d=1200m   yaw 38°   pitch 42°",
        )
        p.end()


# ======================================================================
# 侧栏
# ======================================================================

class UnitStatusPanel(QWidget):
    """CMO 风格 UNIT STATUS 侧栏（折叠分组堆叠）。"""

    def __init__(self, unit: dict, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(384)
        self.setStyleSheet(f"background:{C.PANEL};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("UNIT STATUS", self)
        title.setFixedHeight(24)
        title.setStyleSheet(
            f"background:{C.PANEL_2}; color:{C.TEXT}; font-size:11px;"
            f"font-weight:700; letter-spacing:1px; padding-left:10px;"
            f"border-bottom:1px solid {C.BORDER};"
        )
        outer.addWidget(title)

        # ── 头部：缩略图（宽幅）+ 名称 ──
        head = QWidget(self)
        head.setStyleSheet(f"background:{C.PANEL}; border-bottom:1px solid {C.BORDER};")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(10, 10, 10, 10)
        hl.setSpacing(10)

        thumb = BlueprintThumb(unit, 190, 110, head)
        thumb.setStyleSheet(f"border:1px solid {C.BORDER};")

        right = QVBoxLayout()
        right.setSpacing(3)
        name = QLabel(unit["name"], head)
        name.setStyleSheet(
            f"color:{C.TEXT}; font-size:14px; font-weight:700; background:transparent;"
        )
        sub = QLabel(f"{unit['side']} · {unit['role']}", head)
        sub.setStyleSheet(f"color:{C.TEXT_DIM}; font-size:11px; background:transparent;")
        badge = QLabel(f"● {unit['status']}", head)
        badge.setStyleSheet(
            f"color:{SIDE_COLOR[unit['side']]}; font-size:11px;"
            "font-weight:600; background:transparent;"
        )
        sel = QLabel("[ 已选中 ]", head)
        sel.setStyleSheet(f"color:{C.ACCENT}; font-size:11px; background:transparent;")

        right.addWidget(name)
        right.addWidget(sub)
        right.addWidget(badge)
        right.addWidget(sel)
        right.addStretch(1)

        hl.addWidget(thumb, 0, Qt.AlignmentFlag.AlignTop)
        hl.addLayout(right, 1)
        outer.addWidget(head)

        # ── 折叠分组 ──
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background:{C.PANEL}; border:none;")

        body = QWidget()
        body.setStyleSheet(f"background:{C.PANEL};")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)

        # 1) 基本状态
        s1 = CollapsibleSection("基本状态", True)
        s1.add_kv("阵营", "北约", C.NATO)
        s1.add_kv("航向", f"{unit['course']}°")
        s1.add_kv("速度", f"{unit['speed']:.0f} m/s")
        s1.add_kv("位置", "1243, 881, 4")
        s1.add_kv("所属", unit["formation"])
        bl.addWidget(s1)

        # 2) 损伤与系统
        s2 = CollapsibleSection("损伤与系统", True)
        s2.set_summary(f"{unit['health']:.0f}% · 2 处损坏")
        s2.add_bar("结构", unit["health"] / 100.0,
                   C.WARN if unit["health"] < 70 else C.NATO,
                   f"{unit['health']:.0f}%")
        for pname, mm, ok in unit["parts"]:
            if ok:
                s2.add_bar(pname, 1.0, C.NATO, f"{mm} mm   完好")
            else:
                s2.add_bar(pname, 0.0, C.PACT, "—   损毁")
        bl.addWidget(s2)

        # 3) 机动（读写成对）
        s3 = CollapsibleSection("机动", True)
        s3.set_summary("巡航")
        grid = QWidget(s3._body)
        grid.setStyleSheet("background:transparent;")
        gl = QHBoxLayout(grid)
        gl.setContentsMargins(0, 2, 0, 2)
        gl.setSpacing(5)
        for txt, active in (("停止", False), ("慢速", False),
                            ("巡航", True), ("全速", False), ("最大", False)):
            b = QPushButton(txt, grid)
            b.setFixedHeight(26)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if active:
                b.setStyleSheet(
                    f"QPushButton{{background:#243040; color:{C.CYAN};"
                    f"border:1px solid {C.CYAN}; font-size:11px; font-weight:600;}}"
                )
            else:
                b.setStyleSheet(
                    f"QPushButton{{background:#1a2030; color:{C.TEXT_DIM};"
                    f"border:1px solid {C.BORDER}; font-size:11px;}}"
                    f"QPushButton:hover{{background:#222a3a; color:{C.TEXT};}}"
                )
            gl.addWidget(b)
        s3.body_layout().addWidget(grid)
        bl.addWidget(s3)

        # 4) 武器与弹药
        s4 = CollapsibleSection("武器与弹药", False)
        s4.set_summary(f"主炮 {unit['ammo'][0]}/{unit['ammo'][1]}")
        s4.add_kv("主炮 120mm", f"{unit['ammo'][0]}/{unit['ammo'][1]} 发")
        s4.add_kv("并列机枪 7.62", "1200 发")
        s4.add_kv("最大射程", "4000 m")
        bl.addWidget(s4)

        # 5) 传感器 / EMCON
        s5 = CollapsibleSection("传感器 / EMCON", False)
        s5.set_summary(f"雷达 {unit['radar']}")
        s5.add_kv("雷达", unit["radar"], C.CYAN)
        s5.add_kv("热像仪", "工作")
        s5.add_kv("激光告警", "待机", C.TEXT_DIM)
        bl.addWidget(s5)

        # 6) 编队 AI
        s6 = CollapsibleSection("编队 AI", False)
        s6.set_summary(f"{unit['intent']} · 通信 {unit['comm'] * 100:.0f}%")
        s6.add_kv("意图", unit["intent"])
        s6.add_kv("命令", unit["order"])
        s6.add_bar("通信", unit["comm"], C.CYAN, f"{unit['comm'] * 100:.0f}%")
        bl.addWidget(s6)

        # 7) 最近事件
        s7 = CollapsibleSection("最近事件", False)
        s7.set_summary(f"{len(unit['events'])} 条")
        for ev in unit["events"]:
            s7.add_row(ev, dim=True)
        bl.addWidget(s7)

        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)


# ======================================================================
# 消息日志 + 时间压缩
# ======================================================================

class MessageLog(QWidget):
    """底部消息日志（按 kind 着色）。"""

    EVENTS = [
        ("destroyed", "12.9s  华约 T-14 #07 被击毁（弹药架殉爆）"),
        ("hit",       "12.3s  北约 M1A2 #03 命中 T-14 #07（未击穿）"),
        ("fire",      "12.3s  北约 M1A2 #03 主炮开火 120 mm APFSDS"),
        ("detection", "11.8s  北约 M1A2 #03 雷达发现 2 个接触"),
        ("command",   "10.1s  玩家指令：第 7 装甲连 转移至 (1380, 940)"),
    ]
    COLOR = {
        "destroyed": C.PACT,
        "hit": C.WARN,
        "fire": "#d8c060",
        "detection": "#5aa8d8",
        "command": C.NATO,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(96)
        self.setStyleSheet(f"background:#0b0e14; border-top:1px solid {C.BORDER};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(1)

        head = QLabel("MESSAGE LOG", self)
        head.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:10px; font-weight:700;"
            "letter-spacing:1px; background:transparent;"
        )
        lay.addWidget(head)

        for kind, text in self.EVENTS:
            lbl = QLabel(text, self)
            lbl.setStyleSheet(
                f"color:{self.COLOR.get(kind, C.TEXT)}; font-size:11px;"
                "background:transparent;"
            )
            lay.addWidget(lbl)


class TimeCompressionBar(QWidget):
    """时间压缩条（暂停 / 1x / 2x / 5x / 15x）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setStyleSheet(
            f"background:{C.PANEL_2}; border-top:1px solid {C.BORDER};"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 10, 3)
        lay.setSpacing(4)

        lbl = QLabel("TIME", self)
        lbl.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:10px; letter-spacing:1px;"
            "background:transparent;"
        )
        lay.addWidget(lbl)

        for txt, active in (("II", False), ("1x", False), ("2x", False),
                            ("5x", True), ("15x", False)):
            b = QToolButton(self)
            b.setText(txt)
            b.setFixedSize(40, 22)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if active:
                b.setStyleSheet(
                    f"QToolButton{{background:#243040; color:{C.CYAN};"
                    f"border:1px solid {C.CYAN}; font-size:11px; font-weight:700;}}"
                )
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:#1a2030; color:{C.TEXT_DIM};"
                    f"border:1px solid {C.BORDER}; font-size:11px;}}"
                    f"QToolButton:hover{{color:{C.TEXT};}}"
                )
            lay.addWidget(b)

        lay.addStretch(1)

        t = QLabel("T+ 00:12:34   ·   Zulu 07:45:44   ·   2060-03-14", self)
        t.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:11px; background:transparent;"
        )
        lay.addWidget(t)


# ======================================================================
# 主窗口
# ======================================================================

class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AIC UI 原型 · CMO 风格")
        self.resize(1440, 900)
        self.setStyleSheet(f"background:{C.BG}; color:{C.TEXT};")

        root = QWidget(self)
        self.setCentralWidget(root)
        vl = QVBoxLayout(root)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        vl.addWidget(self._build_menubar(root))
        vl.addWidget(self._build_actionbar(root))

        mid = QWidget(root)
        ml = QHBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(1)
        ml.addWidget(BattlefieldPlaceholder(mid), 1)
        ml.addWidget(UnitStatusPanel(UNIT, mid), 0)
        vl.addWidget(mid, 1)

        vl.addWidget(MessageLog(root))
        vl.addWidget(TimeCompressionBar(root))

    def _build_menubar(self, parent) -> QWidget:
        bar = QWidget(parent)
        bar.setFixedHeight(26)
        bar.setStyleSheet(
            f"background:#0e1118; border-bottom:1px solid {C.BORDER};"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(10, 0, 10, 0)
        hl.setSpacing(16)
        for txt in ("文件", "地图设置", "快速跳转", "单位指令", "接触",
                    "任务", "参考点", "编辑器", "帮助"):
            l = QLabel(txt, bar)
            l.setStyleSheet(
                f"color:{C.TEXT_DIM}; font-size:12px; background:transparent;"
            )
            hl.addWidget(l)
        hl.addStretch(1)
        l = QLabel("Arkady in Cradle  ·  UI 原型", bar)
        l.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:11px; background:transparent;"
        )
        hl.addWidget(l)
        return bar

    def _build_actionbar(self, parent) -> QWidget:
        bar = QWidget(parent)
        bar.setFixedHeight(34)
        bar.setStyleSheet(
            f"background:{C.PANEL_2}; border-bottom:1px solid {C.BORDER};"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(8, 3, 8, 3)
        hl.setSpacing(4)

        items = [
            ("自动交战", False), ("手动交战", False), ("规划航线", True),
            ("油门/高度", True), ("编队", True), ("弹药", True),
            ("武器", True), ("传感器", True), ("系统与损伤", True),
            ("条令", False), ("任务", False),
        ]
        for txt, enabled in items:
            b = QToolButton(bar)
            b.setText(txt)
            b.setFixedHeight(26)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if enabled:
                b.setStyleSheet(
                    f"QToolButton{{background:#1c2431; color:{C.TEXT};"
                    f"border:1px solid {C.BORDER}; padding:0 10px;"
                    f"font-size:11px;}}"
                    f"QToolButton:hover{{background:#26303f; border-color:{C.CYAN};}}"
                )
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:#141922; color:#4a5262;"
                    f"border:1px solid #1a1f29; padding:0 10px; font-size:11px;}}"
                )
                b.setEnabled(False)
            hl.addWidget(b)

        hl.addStretch(1)

        mode = QLabel("模型：自动", bar)
        mode.setStyleSheet(
            f"color:{C.CYAN}; font-size:11px; background:#1a2030;"
            f"border:1px solid {C.BORDER}; padding:3px 10px;"
        )
        hl.addWidget(mode)
        return bar


def _load_fonts(app: QApplication) -> None:
    """离屏模式下 Qt 找不到字体目录，显式加载系统中文字体。"""
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if os.path.exists(path):
            fid = QFontDatabase.addApplicationFont(path)
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                app.setFont(QFont(fams[0], 10))
                return


def main() -> int:
    app = QApplication(sys.argv)
    _load_fonts(app)

    win = DemoWindow()
    win.show()

    if "--shot" in sys.argv:
        i = sys.argv.index("--shot")
        out = sys.argv[i + 1] if i + 1 < len(sys.argv) else "prototype/output/ui_demo.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        app.processEvents()
        pm = win.grab()
        pm.save(out)
        print(f"saved {out}  {pm.width()}x{pm.height()}")
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
