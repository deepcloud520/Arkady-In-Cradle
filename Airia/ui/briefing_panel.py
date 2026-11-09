"""战斗简报面板（SidePanel Tab3，Phase 5b 骨架）。

对应 docs/phase5-ui-pipeline-design.md §4.1：
    - 消费 CombatEventRecorder.generate_briefing() → CombatBriefing.to_text()
    - 骨架：手动「生成简报」按钮触发；任务结束自动生成留待后续
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from Airia.narrative.recorder import CombatBriefing
from Airia.ui import palette as pal


class BriefingPanel(QWidget):
    """战斗简报面板骨架。"""

    def __init__(
        self,
        generate_cb: Callable[[], CombatBriefing | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._generate_cb = generate_cb

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._title = QLabel("战斗简报")
        top.addWidget(self._title)
        top.addStretch(1)
        self._gen_btn = QPushButton("生成简报")
        self._gen_btn.clicked.connect(self.generate)
        top.addWidget(self._gen_btn)
        layout.addLayout(top)

        self._view = QTextBrowser(self)
        layout.addWidget(self._view)

        self.apply_palette(pal.current())
        self._set_placeholder()

    def apply_palette(self, palette: pal.Palette) -> None:
        """C4I 配色（浅色/深色主题切换时由窗口调用）。"""
        self._title.setStyleSheet(
            f"font-weight: 700; color: {palette.text};"
        )
        self._view.setStyleSheet(
            f"QTextBrowser {{ background: {palette.bar_bg}; color: {palette.text};"
            f" border: 1px solid {palette.border}; font-size: 12px; }}"
        )

    def set_generate_cb(self, fn: Callable[[], CombatBriefing | None]) -> None:
        self._generate_cb = fn

    def generate(self) -> None:
        if self._generate_cb is None:
            return
        briefing = self._generate_cb()
        if briefing is None:
            self._view.setPlainText("（未配置事件记录器，无法生成简报）")
            return
        self._view.setPlainText(briefing.to_text())

    def _set_placeholder(self) -> None:
        self._view.setPlainText("（任务结束后点击「生成简报」查看战果统计）")
