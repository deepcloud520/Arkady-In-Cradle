"""键盘 → 叙事/命令队列动作映射（AGENTS.md Phase 5：ui/controls.py）。

对应 docs/phase5-ui-pipeline-design.md §1.1 / 验收「所有鼠标/键盘事件正确响应」：
    - ControlMap：QKeyEvent → 动作名（纯逻辑，可无头单测）
    - MainWindow 在 GLWidget 上装事件过滤器，命中动作后分发给处理槽：
        对话选项 → NarrativeEngine.on_player_choice → CommandQueue（命令入队）
        推进/跳过对话、手动暂停、主题切换等

快捷键表（F1 可查）：
    空格        手动暂停 / 恢复（与对话暂停独立）
    回车        推进当前对话（text 节点自动前进）
    1~9         选择对话选项（等价点击选项按钮 → Command 入队）
    S           跳过当前对话
    Esc         取消选中单位（GLWidget 原有处理，过滤不拦截）
    Ctrl+1/2/3  切换主题 default / constructivism / anime
    F1          状态栏显示快捷键帮助
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent


class ControlMap:
    """按键 → 动作名映射表（纯数据，可测试）。"""

    # 动作名常量
    TOGGLE_PAUSE = "toggle_pause"
    ADVANCE_DIALOGUE = "advance_dialogue"
    SKIP_DIALOGUE = "skip_dialogue"
    CANCEL_SELECTION = "cancel_selection"
    SHOW_HELP = "show_help"
    THEME_PREFIX = "theme:"            # theme:default / theme:constructivism / theme:anime
    OPTION_PREFIX = "choose_option:"   # choose_option:0 ~ choose_option:8

    # 主题快捷键（Ctrl+数字，与选项数字键无冲突）
    THEME_HOTKEYS: dict[int, str] = {
        Qt.Key.Key_1: "theme:default",
        Qt.Key.Key_2: "theme:constructivism",
        Qt.Key.Key_3: "theme:anime",
    }

    @classmethod
    def key_action(cls, event: QKeyEvent) -> str | None:
        """把按键事件映射为动作名；未映射返回 None。

        Args:
            event: QKeyEvent（仅处理 KeyPress）。
        """
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        meta = bool(mods & Qt.KeyboardModifier.MetaModifier)

        # Ctrl+数字：主题切换（仅 Ctrl，无其它修饰键）
        if ctrl and not (shift or alt or meta):
            tid = cls.THEME_HOTKEYS.get(key)
            if tid is not None:
                return tid
            return None

        # 其它带修饰键的组合不拦截（留给 Qt 默认行为）
        if ctrl or shift or alt or meta:
            return None

        if key == Qt.Key.Key_Space:
            return cls.TOGGLE_PAUSE
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return cls.ADVANCE_DIALOGUE
        if key == Qt.Key.Key_S:
            return cls.SKIP_DIALOGUE
        if key == Qt.Key.Key_Escape:
            return cls.CANCEL_SELECTION
        if key == Qt.Key.Key_F1:
            return cls.SHOW_HELP
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            return f"{cls.OPTION_PREFIX}{key - Qt.Key.Key_1}"
        return None

    # ------------------------------------------------------------------
    # 帮助文本（状态栏展示）
    # ------------------------------------------------------------------

    HELP_TEXT = (
        "快捷键：空格=暂停/恢复　回车=推进对话　1-9=选择选项　"
        "S=跳过对话　Esc=取消选中　Ctrl+1/2/3=主题　F1=帮助"
    )
