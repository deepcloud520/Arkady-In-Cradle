"""驾驶舱容器：地图铺满 + 可停靠面板（2026-11-03，CMO 控件风格）。

对应用户定稿：
    - 「做成有明显控件风格的」→ 面板有标题栏 / 折叠按钮 / 关闭按钮（`ui/dock.py`）
    - 「message 也不是控件，不能打开」→ MESSAGE LOG 是真正的面板（`ui/message_panel.py`）
    - 「很多窗口也不能拖动」→ 面板标题栏可拖动（在容器内自由摆放，带边界钳制）
    - 「除了战场绘制，所有 UI 使用控件」→ 本文件无 QPainter 自绘
    - 可缩放窗口：resize 节流 33ms 后重算 ``Metrics``，**已被玩家拖动的面板不再强行归位**
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QTimer, Qt
from PySide6.QtWidgets import QWidget

from Airia.ui import palette as pal
from Airia.ui.dock import DockPanel
from Airia.ui.metrics import Metrics, for_viewport


class Cockpit(QWidget):
    """地图铺满的驾驶舱（面板容器）。"""

    RESIZE_THROTTLE_MS = 33

    def __init__(
        self,
        map_widget: QWidget,
        side_widget: QWidget | None = None,
        message_panel: DockPanel | None = None,
        comms: QWidget | None = None,
        timebar: QWidget | None = None,
        parent: QWidget | None = None,
        user_scale: float = 1.0,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cockpitRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._palette = pal.current()
        self._user_scale = float(user_scale)
        self._metrics = for_viewport(1440, 900, self._user_scale)
        self._sidebar_override: bool | None = None

        self.map = map_widget
        self.map.setParent(self)

        self.timebar = timebar
        if self.timebar is not None:
            self.timebar.setParent(self)

        # 右侧 UNIT STATUS：真面板（可折叠、可拖动，不可关闭）
        self.side: DockPanel | None = None
        if side_widget is not None:
            self.side = DockPanel(
                "UNIT STATUS", side_widget, self,
                collapsible=True, closable=False,
            )

        self.message_panel = message_panel
        if self.message_panel is not None:
            self.message_panel.setParent(self)

        self.comms = comms
        if self.comms is not None:
            self.comms.setParent(self)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(self.RESIZE_THROTTLE_MS)
        self._resize_timer.timeout.connect(self._apply_geometry)

        self.set_palette(self._palette)
        self._apply_geometry()

    # ---- 注入 ----

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    def set_user_scale(self, scale: float) -> None:
        self._user_scale = float(scale)
        self._recompute_metrics()
        self._apply_geometry()

    def set_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self.setStyleSheet(f"#cockpitRoot {{ background: {palette.bg}; }}")
        for panel in (self.side, self.message_panel):
            if panel is not None:
                panel.set_palette(palette)
        if self.timebar is not None and hasattr(self.timebar, "apply_palette"):
            self.timebar.apply_palette(palette)
        if self.comms is not None and hasattr(self.comms, "apply_palette"):
            self.comms.apply_palette(palette)

    def toggle_sidebar(self, visible: bool | None = None) -> bool:
        current = self._sidebar_visible()
        self._sidebar_override = (not current) if visible is None else bool(visible)
        self._apply_geometry()
        return self._sidebar_visible()

    def toggle_message_panel(self) -> None:
        """展开/折叠消息日志（``L`` 键与菜单调用）；关掉后再按 L 重新打开。"""
        panel = self.message_panel
        if panel is None:
            return
        if panel.isHidden():
            panel.open(expanded=True)
        else:
            panel.toggle_collapsed()
        self._apply_geometry()

    # ---- 布局 ----

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._recompute_metrics()
        self._apply_geometry()
        self._resize_timer.start()

    def _recompute_metrics(self) -> None:
        self._metrics = for_viewport(self.width(), self.height(), self._user_scale)

    def _apply_geometry(self) -> None:
        self._recompute_metrics()
        m = self._metrics
        w, h = max(1, self.width()), max(1, self.height())

        # 地图铺满（面板只是叠在上面）
        self.map.setGeometry(QRect(0, 0, w, h))
        self.map.lower()

        if self.timebar is not None:
            side_w = self._reserved_side_width()
            self.timebar.setGeometry(QRect(0, 0, max(1, w - side_w), m.timebar_h))
            if hasattr(self.timebar, "set_metrics"):
                self.timebar.set_metrics(m)

        # 右侧 UNIT STATUS（玩家拖过就尊重其位置）
        if self.side is not None:
            show = self._sidebar_visible()
            self.side.setVisible(show)
            if show:
                self.side.set_metrics(m)
                if not self.side.moved_by_user:
                    side_w = self._sidebar_width()
                    self.side.setGeometry(QRect(w - side_w, 0, side_w, h))

        # 左下 MESSAGE LOG（折叠时只有标题栏高度；玩家 x 关掉后不再自动弹回）
        if self.message_panel is not None:
            panel = self.message_panel
            panel.set_metrics(m)
            if not panel.isHidden() and not panel.moved_by_user:
                pad = m.pad * 2
                width = min(max(320, int(round(520 * m.k))), int(w * 0.55))
                height = (
                    panel.header.height() if panel.is_collapsed
                    else min(int(round(240 * m.k)), int(h * 0.36))
                )
                panel.setGeometry(QRect(pad, h - height - pad, width, height))

        # 左侧实时通话（玩家拖过就不动）
        if self.comms is not None:
            pad = m.pad * 2
            if not getattr(self.comms, "moved_by_user", False):
                self.comms.setGeometry(
                    QRect(
                        pad,
                        m.timebar_h + pad,
                        self.comms.width(),
                        max(100, self.comms.sizeHint().height()),
                    )
                )

        for widget in (self.timebar, self.side, self.message_panel, self.comms):
            if widget is not None and not widget.isHidden():
                widget.raise_()

    # ---- 断点辅助 ----

    def _sidebar_width(self) -> int:
        m = self._metrics
        if m.sidebar_w > 0:
            return m.sidebar_w
        if m.mode == "drawer":
            return max(180, min(int(round(0.34 * self.width())), int(round(300 * m.k))))
        return 0

    def _sidebar_visible(self) -> bool:
        auto = self._metrics.sidebar_w > 0
        if self._sidebar_override is None:
            return auto
        return bool(self._sidebar_override)

    def _reserved_side_width(self) -> int:
        if self._metrics.mode == "drawer":
            return 0
        return self._sidebar_width() if self._sidebar_visible() else 0
