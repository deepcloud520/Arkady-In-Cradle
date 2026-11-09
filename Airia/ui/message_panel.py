"""消息日志面板（可折叠 / 可拖动 / 可关闭，2026-11-03）。

用户定稿：「message 也不是控件，不能打开」——所以日志现在是**真正的面板**：

    ┌ MESSAGE LOG · 3 ────────── [−][×] ┐   ← 标题栏随时可点开/拖动
    │  [12.3s] 命中 …                    │
    └────────────────────────────────────┘

默认**折叠**（只留标题栏，不挡战场）；新事件在折叠态累积为 ``· N`` 未读数，
点标题栏展开后清零。数据仍来自 ``LogWidget``（按 kind 着色 + 系统消息）。
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget

from Airia.ui.dock import DockPanel


class MessageLogPanel(DockPanel):
    """MESSAGE LOG 面板（折叠态显示未读数）。"""

    BASE_TITLE = "MESSAGE LOG"

    def __init__(self, log_widget: QWidget, parent: QWidget | None = None) -> None:
        self._log = log_widget
        self._unread = 0
        super().__init__(
            self.BASE_TITLE, log_widget, parent, start_collapsed=True
        )
        # 面板自带标题栏 → 隐藏 LogWidget 的内部标题，避免双标题
        if hasattr(log_widget, "set_title_visible"):
            log_widget.set_title_visible(False)
        self.collapsed_changed.connect(self._on_collapsed_changed)

    @property
    def log_widget(self) -> QWidget:
        return self._log

    # ---- 数据入口（MainWindow 调用）----

    def push_events(self, events: list[dict]) -> None:
        if not events:
            return
        self._log.append_events(events)
        self._bump(len(events))

    def push_system(self, text: str) -> None:
        self._log.append_system(text)
        self._bump(1)

    # ---- 内部 ----

    def _bump(self, count: int) -> None:
        if not self.is_collapsed:
            return
        self._unread += int(count)
        self.set_title(f"{self.BASE_TITLE} · {self._unread}")

    def _on_collapsed_changed(self, collapsed: bool) -> None:
        if not collapsed:
            self._unread = 0
            self.set_title(self.BASE_TITLE)
