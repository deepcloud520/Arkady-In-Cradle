"""中央弹出窗口（简报 / 档案，2026-11-03 新增）。

对应 docs/ui-cmo-redesign.md §3.6.2 的布局收敛 + 用户 2026-11-03 定稿：
    - **战斗简报 = 中央独立窗口**（不再占用右侧栏 Tab）
    - 档案同理，两者复用同一个 ``CenterWindow`` 基类
    - 非模态：推演继续，窗口不阻塞主循环（玩家可边看边打）

尺寸按 ``Metrics`` 缩放，并且**不超过父窗口**（小窗口下自动收缩）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from Airia.ui import palette as pal
from Airia.ui.archive_panel import ArchivePanel
from Airia.ui.briefing_panel import BriefingPanel
from Airia.ui.metrics import Metrics, for_viewport


class CenterWindow(QDialog):
    """居中于主窗的非模态弹窗基类。"""

    def __init__(
        self,
        title: str,
        content: QWidget,
        parent: QWidget | None = None,
        base_size: tuple[int, int] = (720, 560),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self._base_size = base_size
        self._content = content
        self._palette = pal.current()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)

        self._resize_for_parent()
        self.apply_palette(self._palette)

    # ---- 尺寸 ----

    def _resize_for_parent(self) -> None:
        parent = self.parentWidget()
        pw = parent.width() if parent is not None else 1600
        ph = parent.height() if parent is not None else 900
        m: Metrics = for_viewport(pw, ph)
        w = min(int(self._base_size[0] * m.k), int(pw * 0.92))
        h = min(int(self._base_size[1] * m.k), int(ph * 0.88))
        self.resize(max(360, w), max(240, h))

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        geo = parent.frameGeometry()
        self.move(
            geo.center().x() - self.width() // 2,
            geo.center().y() - self.height() // 2,
        )

    def showEvent(self, event) -> None:  # noqa: N802
        self._resize_for_parent()
        self._center_on_parent()
        super().showEvent(event)

    # ---- 配色 ----

    def apply_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self.setStyleSheet(
            f"QDialog {{ background: {palette.bg}; color: {palette.text}; }}"
        )
        if hasattr(self._content, "apply_palette"):
            self._content.apply_palette(palette)


class BriefingWindow(CenterWindow):
    """战斗简报（中央独立窗口）。"""

    def __init__(
        self,
        generate_cb=None,
        parent: QWidget | None = None,
    ) -> None:
        self.panel = BriefingPanel(generate_cb=generate_cb)
        super().__init__("战斗简报", self.panel, parent, base_size=(760, 580))

    def refresh(self) -> None:
        self.panel.generate()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()


class ArchiveWindow(CenterWindow):
    """战斗档案（中央独立窗口）。"""

    def __init__(
        self,
        timeline_source=None,
        parent: QWidget | None = None,
    ) -> None:
        self.panel = ArchivePanel(timeline_source=timeline_source)
        super().__init__("战斗档案", self.panel, parent, base_size=(680, 620))

    def refresh(self) -> None:
        self.panel.refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()


class LogWindow(CenterWindow):
    """完整消息日志（独立窗口，备用入口）。

    注意：主界面现在把日志做成**可折叠面板**（``ui/message_panel.py``），
    按 ``L`` 展开；本窗口保留给"想要一个常驻独立窗口"的场景。
    """

    def __init__(
        self,
        log_widget: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        self.panel = log_widget
        super().__init__("消息日志", self.panel, parent, base_size=(760, 460))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.panel.show()
