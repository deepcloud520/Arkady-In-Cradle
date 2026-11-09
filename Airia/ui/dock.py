"""可停靠面板（CMO 风格控件，2026-11-03）。

对应用户定稿：「做成有明显控件风格的」「message 也不是控件，不能打开」
「很多窗口也不能拖动」。

``DockPanel`` = 标题栏（折叠按钮 + 标题 + 关闭按钮，可拖动）+ 内容区：

    ┌ MESSAGE LOG ────────────── [−][×] ┐   ← 标题栏：拖动 / 折叠 / 关闭
    │ （内容：任意 QWidget）             │
    └───────────────────────────────────┘

全部由 QFrame / QLabel / QToolButton + QSS 构成（无自绘），
拖动只改本面板几何（限制在父容器内），不改布局树。
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from Airia.ui import palette as pal
from Airia.ui.metrics import Metrics, for_viewport


class _PanelHeader(QFrame):
    """面板标题栏（拖动句柄 + 折叠/关闭按钮）。"""

    collapse_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelHeader")
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self._dragging = False
        self._moved_by_user = False
        self._palette = pal.current()
        self._metrics: Metrics = for_viewport(1440, 900)

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 4, 0)
        row.setSpacing(4)

        self.collapse_btn = QToolButton(self)
        self.collapse_btn.setObjectName("panelBtn")
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.setAutoRaise(False)
        self.collapse_btn.setFixedSize(16, 14)
        self.collapse_btn.clicked.connect(self.collapse_clicked)
        row.addWidget(self.collapse_btn)

        self.title = QLabel(title, self)
        self.title.setObjectName("panelTitleLabel")
        row.addWidget(self.title, 1)

        self.close_btn = QToolButton(self)
        self.close_btn.setObjectName("panelBtn")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setAutoRaise(False)
        self.close_btn.setFixedSize(16, 14)
        self.close_btn.setIcon(QIcon())
        self.close_btn.setText("x")
        self.close_btn.clicked.connect(self.close_clicked)
        row.addWidget(self.close_btn)

        self.set_collapsed_icon(False)

    def set_collapsed_icon(self, collapsed: bool) -> None:
        """折叠按钮用字符 − / +（系统图标在深色标题条上看不清，改用高对比文字）。"""
        self.collapse_btn.setIcon(QIcon())
        self.collapse_btn.setText("+" if collapsed else "–")
        self.collapse_btn.setToolTip("展开" if collapsed else "折叠")

    # ---- 外观：斜切标签标题条（参考 EntityWindow 的 title 造型）----

    def apply_style(self, palette: pal.Palette, metrics: Metrics) -> None:
        self._palette = palette
        self._metrics = metrics
        self.title.setStyleSheet(
            f"color: {palette.accent_text};"
            f" font-size: {max(8, int(round(9 * metrics.k)))}px;"
            " font-weight: 700; letter-spacing: 1px; background: transparent;"
        )
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        try:
            self._paint_title(p)
        finally:
            p.end()

    def _paint_title(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        pl = self._palette
        header = pal.rgba(pl.header)

        # 标题条底：上亮下暗的渐变（控件感）
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, header.lighter(130))
        grad.setColorAt(1.0, header.darker(108))
        p.fillRect(self.rect(), grad)

        # 斜切标签：把标题文字包进一块 accent 色梯形（EntityWindow 的 title）
        # 宽度按**文字实际宽度**算（QLabel 在布局里会被拉伸占满，不能用 width()）
        label_x = self.title.mapTo(self, QPoint(0, 0)).x()
        text_w = self.title.fontMetrics().horizontalAdvance(self.title.text())
        pad = max(8, int(round(10 * self._metrics.k)))
        tab_w = label_x + text_w + pad
        slant = max(4, int(round(h * 0.55)))
        tab = QPolygonF([
            QPointF(0, 0),
            QPointF(tab_w - slant, 0),
            QPointF(tab_w, h),
            QPointF(0, h),
        ])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pl.accent))
        p.drawPolygon(tab)

        # 底部一道深色分隔线（与面板主体描边衔接）
        p.setPen(QPen(QColor(pl.border_dark), 1))
        p.drawLine(QPointF(0, h - 0.5), QPointF(w, h - 0.5))

    # ---- 拖动（在父容器内移动整个面板）----

    @property
    def moved_by_user(self) -> bool:
        return self._moved_by_user

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            panel = self.parentWidget()
            if panel is not None and hasattr(panel, "begin_drag"):
                panel.begin_drag(event.globalPosition().toPoint())
            self._dragging = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        panel = self.parentWidget()
        if self._dragging and panel is not None and hasattr(panel, "drag_to"):
            panel.drag_to(event.globalPosition().toPoint())
            self._moved_by_user = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DockPanel(QFrame):
    """带标题栏的面板：可拖动 / 折叠 / 关闭。"""

    closed = Signal(object)
    collapsed_changed = Signal(bool)

    def __init__(
        self,
        title: str,
        body: QWidget,
        parent: QWidget | None = None,
        collapsible: bool = True,
        closable: bool = True,
        start_collapsed: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dockPanel")
        self._palette = pal.current()
        self._metrics: Metrics = for_viewport(1440, 900)
        # 浮层阴影：面板像"窗口"而不是贴图色块（控件级效果，非自绘）
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(14)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)
        self._collapsed = False
        self._drag_origin: QPoint | None = None
        self._drag_start_pos: QPoint | None = None
        self._expanded_height: int = 0
        self._moved_by_user = False
        self._user_closed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = _PanelHeader(title, self)
        self.header.collapse_clicked.connect(self.toggle_collapsed)
        self.header.close_clicked.connect(self._on_close)
        if not collapsible:
            self.header.collapse_btn.hide()
        if not closable:
            self.header.close_btn.hide()
        root.addWidget(self.header)

        self.body = QFrame(self)
        self.body.setObjectName("panelBody")
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body.setParent(self.body)
        body_layout.addWidget(body)
        root.addWidget(self.body, 1)

        self._body_widget = body
        self.set_palette(self._palette)
        self.set_metrics(self._metrics)
        if start_collapsed:
            self.set_collapsed(True)

    # ---- 状态 ----

    @property
    def body_widget(self) -> QWidget:
        return self._body_widget

    @property
    def moved_by_user(self) -> bool:
        return self._moved_by_user or self.header.moved_by_user

    @property
    def user_closed(self) -> bool:
        """玩家点过 x：布局重排不得再擅自 show 回来（否则"暂停时关不掉"）。"""
        return self._user_closed

    def open(self, expanded: bool = True) -> None:
        """重新打开（菜单 / 快捷键调用）。"""
        self._user_closed = False
        self.show()
        self.set_collapsed(not expanded)

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_title(self, title: str) -> None:
        self.header.title.setText(title)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        if collapsed:
            self._expanded_height = self.height()
            self.body.hide()
            self.setFixedHeight(self.header.height())
        else:
            self.body.show()
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            if self._expanded_height > 0:
                self.resize(self.width(), self._expanded_height)
        self._collapsed = collapsed
        self.header.set_collapsed_icon(collapsed)
        self.collapsed_changed.emit(collapsed)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _on_close(self) -> None:
        self._user_closed = True
        self.hide()
        self.closed.emit(self)

    # ---- 拖动 ----

    def begin_drag(self, global_pos: QPoint) -> None:
        self._drag_origin = global_pos
        self._drag_start_pos = self.pos()

    def drag_to(self, global_pos: QPoint) -> None:
        if self._drag_origin is None or self._drag_start_pos is None:
            return
        delta = global_pos - self._drag_origin
        pos = self._drag_start_pos + delta
        parent = self.parentWidget()
        if parent is not None:
            max_x = max(0, parent.width() - self.width())
            max_y = max(0, parent.height() - self.height())
            pos.setX(min(max(0, pos.x()), max_x))
            pos.setY(min(max(0, pos.y()), max_y))
        self.move(pos)
        self._moved_by_user = True

    # ---- 样式 ----

    def set_metrics(self, metrics: Metrics) -> None:
        self._metrics = metrics
        k = metrics.k
        self.header.setFixedHeight(max(16, int(round(20 * k))))
        size = max(12, int(round(16 * k)))
        for btn in (self.header.collapse_btn, self.header.close_btn):
            btn.setFixedSize(size, max(10, int(round(14 * k))))
            btn.setIconSize(QSize(max(8, int(round(10 * k))), max(8, int(round(10 * k)))))
        # 折叠态：高度必须跟随标题栏（否则 set_metrics 后会露出空白/裁切）
        if self._collapsed:
            self.setFixedHeight(self.header.height())
        self._apply_style()

    def set_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self._apply_style()

    def _apply_style(self) -> None:
        p = self._palette
        k = self._metrics.k
        # 标题条自身由 _PanelHeader.paintEvent 画（渐变 + 斜切标签），
        # 这里只保留容器/主体的描边：主体用标签同色 accent 描边（同 EntityWindow）。
        self.header.apply_style(p, self._metrics)
        self.setStyleSheet(
            f"""
#dockPanel {{ background: transparent; border: none; }}
#panelHeader {{ background: transparent; border: none; }}
#panelTitleLabel {{ background: transparent; }}
#panelBtn {{
    padding: 0px;
    min-height: 0px;
    border-radius: 1px;
    color: {p.accent_text};
    font-weight: 700;
    font-size: {max(9, int(round(10 * k)))}px;
}}
#panelBody {{
    background: {p.css("panel")};
    border: 1px solid {p.accent};
    border-top: none;
}}
"""
        )
