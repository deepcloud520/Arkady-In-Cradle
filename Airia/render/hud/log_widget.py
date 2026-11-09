"""消息日志（C4I Message Log，2026-11-03 重写）。

对应 docs/ui-cmo-redesign.md §5.2 + 用户 2026-11-03 定稿：
    - **系统对话 / 教程**走本组件（``append_system``），与战斗事件共用一条时间轴
    - 战斗事件按 ``kind`` 分级着色（击毁=红 / 命中=橙 / 开火=琥珀 / 探测=蓝 / 指令=绿）
    - 实时通话（人物对话）**不进这里**，走左侧 ``CommsPopup``

性能：只追加条目（``QListWidgetItem``），不做每帧全量重建；上限 200 行。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from Airia.render import palette as pal

_MAX_LINES = 200

# 事件类型 → 摘要前缀（无文本渲染负担，直接格式化字符串）
_KIND_LABEL = {
    "fire": "开火",
    "hit": "命中",
    "destroyed": "击毁",
    "miss": "脱靶",
    "detection": "探测",
    "command": "指令",
    "command_hold": "指令",
    "command_redeploy": "指令",
    "system": "系统",
    "tutorial": "教程",
}


class LogWidget(QWidget):
    """半透明消息日志卡片（叠加在驾驶舱底部）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = pal.current()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题条（CMO 的 MESSAGE LOG 语义；与 UNIT STATUS 同款）
        self._title = QLabel("MESSAGE LOG", self)
        self._title.setObjectName("logTitle")
        self._title.setFixedHeight(16)
        layout.addWidget(self._title)

        self._list = QListWidget(self)
        self._list.setObjectName("messageLog")
        self._list.setWordWrap(True)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        layout.addWidget(self._list)

        self._apply_palette()

    # ---- 配色 / 字号 ----

    def set_title_visible(self, visible: bool) -> None:
        """内嵌到面板（``MessageLogPanel``）时隐藏自带标题，避免双标题。"""
        self._title.setVisible(bool(visible))

    def set_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self._apply_palette()

    def set_metrics(self, metrics) -> None:
        """按 UI metrics 调整字号。

        参数是 ``Airia.ui.metrics.Metrics``（鸭子类型，不 import：render 层
        不得反向依赖 ui 层，见 AGENTS.md 4.1 依赖方向）。
        """
        self._apply_palette(metrics)

    def _apply_palette(self, metrics=None) -> None:
        p = self._palette
        size = 11
        if metrics is not None:
            size = max(8, int(round(getattr(metrics, "font_sm", 11))))
        self.setStyleSheet(
            f"""
#messageLog {{
    background: {p.css("log")};
    border: 1px solid {p.border_dark};
    border-radius: 0px;
    color: {p.text};
    font-size: {size}px;
    padding: 2px 4px;
}}
#messageLog::item {{ padding: 1px 2px; }}
#logTitle {{
    background: {p.css("header")};
    color: {p.header_text};
    font-size: {max(7, size - 2)}px;
    font-weight: 700;
    letter-spacing: 1px;
    padding-left: 6px;
}}
"""
        )

    # ---- 写入 ----

    def append_events(self, events: list[dict]) -> None:
        """追加战斗事件（按 kind 着色）。"""
        for ev in events:
            kind = str(ev.get("kind", ""))
            data = ev.get("data") or {}
            label = _KIND_LABEL.get(kind, kind)
            t = float(ev.get("time", 0.0))
            line = f"[{t:5.1f}s] {label}"
            extra = self._fmt(data)
            if extra:
                line += f" {extra}"
            self._add(line, kind)

    def append_system(self, text: str) -> None:
        """追加系统 / 教程消息（CMO 式系统对话，不阻塞推演）。"""
        self._add(f"[SYS] {text}", "system")

    def _add(self, line: str, kind: str) -> None:
        item = QListWidgetItem(line)
        item.setForeground(QColor(pal.log_kind_color(self._palette, kind)))
        self._list.addItem(item)
        self._trim()

    def clear(self) -> None:
        self._list.clear()

    # ---- 内部 ----

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

    # ---- 兼容旧接口（旧版把它当 GLWidget 子控件定位）----

    def layout_overlay(self, parent_w: int, parent_h: int) -> None:
        w = min(420, max(200, parent_w - 20))
        h = min(170, max(80, parent_h // 4))
        self.setGeometry(10, parent_h - h - 10, w, h)
