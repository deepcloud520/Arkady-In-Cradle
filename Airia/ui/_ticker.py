"""事件提示条（瞬时浮层，取代常驻底部日志，2026-11-03）。

用户定稿：「不需要底部常驻栏，沉浸感很重要；除战场绘制外一律用控件」。

因此完整日志收进**按需打开的窗口**（`ui/windows.py` + `LogWidget`），
屏幕上只保留这条**瞬时**提示：有新事件时淡入几行，静置数秒后淡出，不占版面。

实现全部是控件（QFrame + QLabel + QGraphicsOpacityEffect + QPropertyAnimation），
没有 QPainter 自绘。
"""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from Airia.ui import palette as pal
from Airia.ui.metrics import Metrics, for_viewport


class MessageTicker(QFrame):
    """瞬时消息条：最近 N 条事件，淡入 → 停留 → 淡出。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        max_lines: int = 3,
        hold_ms: int = 6000,
    ) -> None:
        super().__init__(parent)
        self._palette = pal.current()
        self._metrics: Metrics = for_viewport(1440, 900)
        self._max_lines = int(max_lines)

        self.setObjectName("messageTicker")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # 不拦截点击：提示条只是视觉层，鼠标操作要能穿过它落到战场上
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(1)
        self._layout = layout

        self._labels: list[QLabel] = []
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)

        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(700)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade.finished.connect(self._on_faded)

        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(int(hold_ms))
        self._hold.timeout.connect(self._start_fade)

        self._apply_style()
        self.hide()

    # ------------------------------------------------------------------

    def set_metrics(self, metrics: Metrics) -> None:
        if metrics == self._metrics:
            return
        self._metrics = metrics
        self._apply_style()

    def set_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self._apply_style()
        self._restyle_labels()

    # ------------------------------------------------------------------

    def push(self, text: str, kind: str) -> None:
        """追加一条提示（新事件到达时调用）。"""
        color = pal.log_kind_color(self._palette, kind)
        label = QLabel(str(text), self)
        label.setWordWrap(False)
        label.setStyleSheet(
            f"color: {color.name()}; background: transparent;"
            f" font-size: {self._font_px()}px;"
        )
        self._layout.addWidget(label)
        self._labels.append(label)
        while len(self._labels) > self._max_lines:
            old = self._labels.pop(0)
            self._layout.removeWidget(old)
            old.setParent(None)
            old.deleteLater()

        self._fade.stop()
        self._effect.setOpacity(1.0)
        self.show()
        self.raise_()
        self.adjustSize()
        self._hold.start()

    def clear(self) -> None:
        for lab in self._labels:
            self._layout.removeWidget(lab)
            lab.setParent(None)
            lab.deleteLater()
        self._labels.clear()
        self.hide()

    # ------------------------------------------------------------------

    def _font_px(self) -> int:
        return max(8, int(round(self._metrics.font_sm)))

    def _apply_style(self) -> None:
        p = self._palette
        m = self._metrics
        pad = max(4, int(round(6 * m.k)))
        self.setStyleSheet(
            f"""
#messageTicker {{
    background: {p.css("log")};
    border: none;
    border-left: 2px solid {p.css("border_dark")};
}}
"""
        )
        self._layout.setContentsMargins(pad + 4, pad, pad, pad)
        self._restyle_labels()

    def _restyle_labels(self) -> None:
        for lab in self._labels:
            color = lab.palette().color(lab.foregroundRole())
            lab.setStyleSheet(
                f"color: {color.name()}; background: transparent;"
                f" font-size: {self._font_px()}px;"
            )

    def _start_fade(self) -> None:
        if not self._labels:
            return
        self._fade.stop()
        self._fade.start()

    def _on_faded(self) -> None:
        # 淡出后不销毁内容：下次 push 直接复用（避免频繁建删控件）
        self.hide()
