"""Qt 透明滚动日志（Phase 5c 新增，render/hud 子项）。

对应 docs/phase5-ui-pipeline-design.md §4.3 / §4.1：
    - 透明 QWidget overlay 在 GLWidget 左下角，显示事件摘要
    - 数据来源：NarrativeState.recent_events（叙事层已过滤/摘要）
    - 限制最大行数（200），超出丢弃旧行（docs §8 风险缓解）

与 HUDOverlay 的分工：HUDOverlay 画世界空间标记（moderngl），
LogWidget 画屏幕空间文本（Qt widget，透明背景叠加在 GL 之上）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QVBoxLayout, QWidget

_MAX_LINES = 200

# 事件类型 → 摘要前缀（无文本渲染负担，直接格式化字符串）
_KIND_LABEL = {
    "fire": "开火",
    "hit": "命中",
    "destroyed": "击毁",
    "miss": "脱靶",
    "detection": "探测",
    "command": "指令",
}


class LogWidget(QWidget):
    """透明滚动日志（叠加在 GLWidget 左下角）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("hudLogWidget")
        self.setStyleSheet(
            "#hudLogWidget { background: rgba(10,10,16,0.55);"
            " border: 1px solid rgba(255,255,255,0.10); border-radius: 4px; }"
            " QListWidget { background: transparent; border: none;"
            " color: #c8d0d8; font-size: 11px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._list = QListWidget(self)
        self._list.setWordWrap(True)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._list)

    def append_events(self, events: list[dict]) -> None:
        """追加事件摘要（最旧在下，滚动到底部）。"""
        for ev in events:
            kind = str(ev.get("kind", ""))
            data = ev.get("data") or {}
            label = _KIND_LABEL.get(kind, kind)
            t = float(ev.get("time", 0.0))
            line = f"[{t:5.1f}s] {label}"
            # 附加关键字段（有则显示，无则不显示）
            extra = self._fmt(data)
            if extra:
                line += f" {extra}"
            self._list.addItem(line)
        self._trim()

    def clear(self) -> None:
        self._list.clear()

    def _fmt(self, data: dict) -> str:
        parts: list[str] = []
        for key in ("target", "shooter", "armor_idx", "source"):
            if key in data and data[key] is not None:
                parts.append(str(data[key]))
        return " ".join(parts)

    def _trim(self) -> None:
        while self._list.count() > _MAX_LINES:
            self._list.takeItem(0)
        self._list.scrollToBottom()

    def layout_overlay(self, parent_w: int, parent_h: int) -> None:
        """由宿主（GLWidget resizeEvent）调用，定位到左下角。"""
        w = min(420, max(200, parent_w - 20))
        h = min(170, max(80, parent_h // 4))
        self.setGeometry(10, parent_h - h - 10, w, h)
