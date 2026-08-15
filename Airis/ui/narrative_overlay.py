"""对话面板（SidePanel Tab1，Phase 5b 新增）。

对应 docs/phase5-ui-pipeline-design.md §4.2：
    - 显示 speaker / 对话文本 / 选项按钮 / 对话历史
    - 选项点击 → option_selected(dialogue_id, index) 信号 → MainWindow
      调用 NarrativeEngine.on_player_choice()
    - 无对话时隐藏选项区，显示「待命中...」占位
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from Airis.narrative.engine import NarrativeState

_MAX_HISTORY_LINES = 200


class NarrativeOverlay(QWidget):
    """对话面板：消费 NarrativeState.active_dialogue。"""

    option_selected = Signal(str, int)  # (dialogue_id, option_index)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dialogue_id: str | None = None
        self._build_ui()
        self._set_idle()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._title = QLabel("📻 前线通讯")
        self._title.setObjectName("dlgTitle")
        self._title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #e8b060;"
        )
        layout.addWidget(self._title)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setStyleSheet("color: #e0e0e6; font-size: 13px;")
        layout.addWidget(self._body, 0)

        # 选项按钮区（动态生成）
        self._options_layout = QVBoxLayout()
        self._options_layout.setSpacing(6)
        layout.addLayout(self._options_layout)

        # 历史（滚动）
        history_label = QLabel("─── 历史 ───")
        history_label.setStyleSheet("color: #88888f; font-size: 11px;")
        layout.addWidget(history_label)

        self._history = QLabel("")
        self._history.setWordWrap(True)
        self._history.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._history.setStyleSheet("color: #a8adb5; font-size: 11px;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._history)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        layout.addWidget(scroll, 1)

    def update_state(self, state: NarrativeState | None) -> None:
        """刷新对话面板（docs §2.1 数据流第 4 步）。"""
        d = state.active_dialogue if state is not None else None
        if d is None or d.is_finished:
            self._set_idle()
            return

        self._dialogue_id = d.dialogue_id
        self._title.setText(f"📻 {d.speaker or '前线通讯'}")
        self._body.setText(d.text)

        # 重建选项按钮
        self._clear_options()
        for i, opt in enumerate(d.options):
            btn = QPushButton(opt)
            btn.setStyleSheet(
                "text-align: left; padding: 6px 8px;"
                " background: #1c1c26; color: #d8dce2; border: 1px solid #34343f;"
            )
            btn.clicked.connect(lambda _=False, idx=i: self.option_selected.emit(
                self._dialogue_id or "", idx))
            self._options_layout.addWidget(btn)
        self._options_layout.addStretch(1)

        # 历史
        self._history.setText("\n".join(d.history[-_MAX_HISTORY_LINES:]))

    def _clear_options(self) -> None:
        while self._options_layout.count():
            item = self._options_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _set_idle(self) -> None:
        """无活跃对话时的占位显示。

        Phase 5e：**不**清空对话历史——对话结束后 / 切换 Tab 期间历史保留，
        供玩家回顾（「对话历史跨 Tab 保留」）；新对话激活时 update_state
        会用新对话的历史整体替换。
        """
        self._dialogue_id = None
        self._title.setText("📻 前线通讯")
        self._body.setText("待命中...")
        self._clear_options()
        idle = QLabel("（当前无活跃对话）")
        idle.setStyleSheet("color: #66666e; font-size: 12px;")
        self._options_layout.addWidget(idle)
