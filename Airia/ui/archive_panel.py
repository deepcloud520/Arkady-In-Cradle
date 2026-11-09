"""档案库面板（SidePanel Tab2，Phase 5b 骨架）。

对应 docs/phase5-ui-pipeline-design.md §4.1：
    - 骨架：展示 CombatEventRecorder 的事件时间线
    - 完整档案（战役记录 / 单位服役档案 / 事件检索）留待后续迭代
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from Airia.ui import palette as pal


class ArchivePanel(QWidget):
    """档案库骨架：事件时间线列表。"""

    def __init__(
        self,
        timeline_source: Callable[[], list[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._timeline_source = timeline_source

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._title = QLabel("战斗档案")
        top.addWidget(self._title)
        top.addStretch(1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        top.addWidget(refresh)
        layout.addLayout(top)

        self._list = QListWidget(self)
        layout.addWidget(self._list)

        self._empty = QLabel("（暂无记录）")
        layout.addWidget(self._empty)

        self.apply_palette(pal.current())
        self.refresh()

    def apply_palette(self, palette: pal.Palette) -> None:
        """C4I 配色（随界面主题切换）。"""
        self._title.setStyleSheet(f"font-weight: 700; color: {palette.text};")
        self._empty.setStyleSheet(f"color: {palette.text_dim};")
        self._list.setStyleSheet(
            f"QListWidget {{ background: {palette.bar_bg}; color: {palette.text};"
            f" border: 1px solid {palette.border}; }}"
        )

    def set_timeline_source(self, fn: Callable[[], list[str]]) -> None:
        self._timeline_source = fn

    def refresh(self) -> None:
        if self._timeline_source is None:
            return
        self._list.clear()
        lines = self._timeline_source()
        self._list.addItems(lines)
        self._empty.setVisible(not lines)
