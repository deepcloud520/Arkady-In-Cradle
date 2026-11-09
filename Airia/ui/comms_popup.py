"""实时通话弹窗（左侧浮层，STT 逐字显现，2026-11-03 定稿口径）。

对应用户 2026-11-03 定稿：
    - 实时通话 = **左侧弹出式窗口**（不是侧栏 Tab）
    - 显示说话者头像；**无头像时显示 ``Sound only``**（不静默留白）
    - 文本**按秒推进、每次显现一定字符数**，模拟语音转写（STT）效果
    - 选项在文本吐完后才出现（避免剧透）；点击正文可「跳过打字机」

数据来源：``NarrativeState.active_dialogue``（与旧 ``NarrativeOverlay`` 同源，
``option_selected(dialogue_id, index)`` 信号签名保持不变，MainWindow 无需改逻辑）。
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from Airia.ui import palette as pal
from Airia.ui.metrics import Metrics, for_viewport
from Airia.ui.unit_thumb import avatars_dir

_AVATAR_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_MAX_HISTORY = 4       # 弹窗内保留的最近对话行数（完整历史见 message log）


# ----------------------------------------------------------------------
# 打字机模型（纯逻辑，可无头单测）
# ----------------------------------------------------------------------

class TypewriterModel:
    """STT 逐字显现模型：按「字符/秒」推进，不依赖 Qt。

    用法::

        tw = TypewriterModel("敌军装甲连正在渡河", chars_per_sec=24)
        while not tw.finished:
            label.setText(tw.advance(0.05))
        label.setText(tw.reveal_all())
    """

    def __init__(self, text: str, chars_per_sec: float = 26.0) -> None:
        self._text = str(text)
        self.chars_per_sec = max(float(chars_per_sec), 1.0)
        self._elapsed = 0.0
        self._revealed = 0

    # ---- 属性 ----

    @property
    def text(self) -> str:
        return self._text

    @property
    def revealed(self) -> int:
        """已显现字符数。"""
        return self._revealed

    @property
    def finished(self) -> bool:
        return self._revealed >= len(self._text)

    @property
    def shown(self) -> str:
        """当前应显示的文本片段。"""
        return self._text[: self._revealed]

    # ---- 推进 ----

    def advance(self, dt: float) -> str:
        """前进 ``dt`` 秒，返回新的可见文本。"""
        if self.finished:
            return self._text
        self._elapsed += max(float(dt), 0.0)
        self._revealed = min(
            len(self._text), int(self._elapsed * self.chars_per_sec)
        )
        return self._text[: self._revealed]

    def reveal_all(self) -> str:
        """立即全部显现（点击跳过）。"""
        self._revealed = len(self._text)
        self._elapsed = len(self._text) / self.chars_per_sec
        return self._text


# ----------------------------------------------------------------------
# 头像（无图 → Sound only）
# ----------------------------------------------------------------------

def _slug(text: str) -> str:
    """说话者名 → 文件名安全串（中英文均保留）。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(text or "").strip())
    return s.strip("_") or "unknown"


def avatar_path(speaker: str, root: Path | None = None) -> Path | None:
    """解析说话者头像文件（无 → None）。"""
    slug = _slug(speaker)
    directory = avatars_dir(root)
    for ext in _AVATAR_EXTENSIONS:
        p = directory / f"{slug}{ext}"
        if p.is_file():
            return p
    return None


class AvatarBox(QLabel):
    """64×64 头像框（**控件实现**）；无头像时显示 ``Sound only``。"""

    def __init__(self, size: int = 64, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = int(size)
        self._pixmap: QPixmap | None = None
        self._speaker = ""
        self.setObjectName("avatarBox")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(self._size, self._size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Sound only")

    def set_speaker(self, speaker: str, root: Path | None = None) -> None:
        self._speaker = str(speaker or "")
        path = avatar_path(self._speaker, root) if self._speaker else None
        pm = QPixmap(str(path)) if path is not None else QPixmap()
        if pm.isNull():
            self._pixmap = None
            self.setText("Sound only")
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        target = int(round(self._size * dpr))
        scaled = pm.scaled(
            target,
            target,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self._pixmap = scaled
        self.setText("")
        self.setPixmap(scaled)

    @property
    def has_avatar(self) -> bool:
        return self._pixmap is not None

    def set_size(self, size: int) -> None:
        self._size = max(32, int(size))
        self.setFixedSize(self._size, self._size)


# ----------------------------------------------------------------------
# 通话弹窗
# ----------------------------------------------------------------------

class CommsPopup(QWidget):
    """左侧实时通话浮层（头像 + STT 正文 + 选项）。"""

    option_selected = Signal(str, int)  # (dialogue_id, option_index)

    def __init__(
        self,
        parent: QWidget | None = None,
        width: int = 356,
        chars_per_sec: float = 26.0,
    ) -> None:
        super().__init__(parent)
        self._dialogue_id: str | None = None
        self._chars_per_sec = float(chars_per_sec)
        self._tw: TypewriterModel | None = None
        self._options: list[str] = []
        self._palette = pal.current()
        self._metrics: Metrics = for_viewport(1440, 900)
        self._typing = False
        self._drag_offset = None
        self._moved_by_user = False
        self._muted_id: str | None = None

        self.setFixedWidth(int(width))
        self.setObjectName("commsPopup")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(60)      # 约 16.7 次/秒
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.hide()

    # ---- UI ----

    def _build_ui(self) -> None:
        # 窗口阴影（与 DockPanel 一致，控件级效果）
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(14)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 标题条
        self._header = QWidget(self)
        self._header.setObjectName("commsHeader")
        self._header.setFixedHeight(20)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(8, 0, 6, 0)
        self._title = QLabel("实时通话")
        self._title.setObjectName("panelTitle")
        hl.addWidget(self._title)
        hl.addStretch(1)
        self._tag = QLabel("COMMS")
        self._tag.setObjectName("commsTag")
        hl.addWidget(self._tag)
        # 控件风格：像其它面板一样带关闭按钮（关掉只是隐藏，剧情继续）
        self._close_btn = QToolButton(self._header)
        self._close_btn.setObjectName("panelBtn")
        self._close_btn.setFixedSize(16, 14)
        self._close_btn.setAutoRaise(False)
        self._close_btn.setIcon(QIcon())
        self._close_btn.setText("x")
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.clicked.connect(self._on_close_clicked)
        hl.addWidget(self._close_btn)
        self._header.setCursor(Qt.CursorShape.SizeAllCursor)
        root.addWidget(self._header)

        # 正文卡片
        self._card = QWidget(self)
        self._card.setObjectName("commsCard")
        cl = QVBoxLayout(self._card)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._avatar = AvatarBox(64, self._card)
        top.addWidget(self._avatar, 0, Qt.AlignmentFlag.AlignTop)
        info = QVBoxLayout()
        info.setSpacing(2)
        self._speaker = QLabel("—")
        self._speaker.setObjectName("commsSpeaker")
        self._speaker.setWordWrap(True)
        info.addWidget(self._speaker)
        self._stt = QLabel("")
        self._stt.setObjectName("commsStt")
        info.addWidget(self._stt)
        info.addStretch(1)
        top.addLayout(info, 1)
        cl.addLayout(top)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._body.setObjectName("commsBody")
        self._body.setMinimumHeight(56)
        cl.addWidget(self._body, 0)

        self._options_layout = QVBoxLayout()
        self._options_layout.setSpacing(3)
        cl.addLayout(self._options_layout)

        self._history = QLabel("")
        self._history.setWordWrap(True)
        self._history.setObjectName("commsHistory")
        cl.addWidget(self._history)

        root.addWidget(self._card, 1)
        self._apply_palette()

    # ---- 调色板 ----

    def apply_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self._apply_palette()

    def set_metrics(self, metrics: Metrics) -> None:
        """尺寸随窗口缩放（宽度 + 字号）；窗口 resize 时由 Cockpit 调用。"""
        self._metrics = metrics
        self.setFixedWidth(max(220, metrics.popup_w))
        self._apply_palette()
        self.adjustSize()

    def _apply_palette(self) -> None:
        p = self._palette
        m = self._metrics
        f_xs = max(7, int(round(m.font_xs)))
        f_sm = max(8, int(round(m.font_sm)))
        f_md = max(9, int(round(m.font_md)))
        f_lg = max(10, int(round(m.font_lg)))
        self.setStyleSheet(
            f"""
#commsHeader {{ {pal.panel_header_qss(p)} }}
#commsHeader QLabel#panelTitle {{ color: {p.header_text}; font-size: {f_sm}px; }}
#commsTag {{ color: {p.header_text}; font-size: {f_xs}px; letter-spacing: 1px; }}
#commsHeader #panelBtn {{
    color: {p.text};
    font-weight: 700;
    font-size: {f_sm}px;
    padding: 0px;
    min-height: 0px;
}}
#commsCard {{
    background: {p.css("panel")};
    border: 1px solid {p.border_dark};
    border-top: none;
}}
#commsSpeaker {{ color: {p.select}; font-size: {f_md}px; font-weight: 700; }}
#commsStt {{ color: {p.text_dim}; font-size: {f_xs}px; letter-spacing: 1px; }}
#commsBody {{ color: {p.text}; font-size: {f_lg}px; }}
#commsHistory {{ color: {p.text_dim}; font-size: {f_sm}px; }}
#avatarBox {{
    background: rgba(18, 22, 26, 0.78);
    border: 1px solid {p.border_dark};
    color: {p.text_dim};
    font-size: {f_xs}px;
}}
QPushButton#commsOption {{
    text-align: left; padding: 4px 6px;
    background: {p.css("panel2")}; color: {p.text};
    border: 1px solid {p.border_dark}; font-size: {f_sm}px;
}}
QPushButton#commsOption:hover {{ background: {p.border}; }}
"""
        )
        self._avatar.setStyleSheet(
            f"#avatarBox {{ background: rgba(18, 22, 26, 0.78);"
            f" border: 1px solid {p.border_dark}; color: {p.text_dim};"
            f" font-size: {f_xs}px; }}"
        )

    # ---- 数据注入（保持旧 NarrativeOverlay.update_state 语义）----

    def set_state(self, state) -> None:
        """刷新通话弹窗（NarrativeState 鸭子类型：需要 ``active_dialogue``）。"""
        d = getattr(state, "active_dialogue", None) if state is not None else None
        if d is None or getattr(d, "is_finished", False):
            self._close_call()
            return

        new_id = str(getattr(d, "dialogue_id", "") or "")
        text = str(getattr(d, "text", "") or "")
        speaker = str(getattr(d, "speaker", "") or "前线通讯")

        # 玩家手动关掉本次通话后，本次对话不再自动弹回（新对话才重新弹出）
        if self._muted_id is not None and new_id != self._muted_id:
            self._muted_id = None
        if new_id == self._muted_id:
            return

        changed = new_id != self._dialogue_id or (
            self._tw is not None and self._tw.text != text
        )
        self._dialogue_id = new_id
        self._options = [str(o) for o in (getattr(d, "options", None) or [])]

        if changed:
            self._start_line(speaker, text)
        self._history.setText("\n".join(
            list(getattr(d, "history", None) or [])[-_MAX_HISTORY:]
        ))
        self._history.setVisible(bool(self._history.text()))
        self.show()
        self.raise_()
        self._request_relayout()

    # 兼容旧调用名（MainWindow 历史上调 update_state）
    update_state = set_state

    def _start_line(self, speaker: str, text: str) -> None:
        self._speaker.setText(speaker)
        self._avatar.set_speaker(speaker)
        self._tw = TypewriterModel(text, self._chars_per_sec)
        self._typing = True
        self._body.setText("")
        self._clear_options()
        self._stt.setText("· 语音转写中…")

    def _close_call(self) -> None:
        self._dialogue_id = None
        self._tw = None
        self._typing = False
        self._options = []
        self._clear_options()
        self.hide()

    def _on_close_clicked(self) -> None:
        """玩家手动关闭本次通话（本次对话不再自动弹回）。"""
        self._muted_id = self._dialogue_id
        self._close_call()

    # ---- 打字机推进 ----

    def _on_tick(self) -> None:
        if not self.isVisible():
            return
        if self._tw is None:
            return
        if self._typing:
            dt = self._timer.interval() / 1000.0
            self._body.setText(self._tw.advance(dt) + "|")
            if self._tw.finished:
                self._finish_typing()

    def _finish_typing(self) -> None:
        if self._tw is None:
            return
        self._typing = False
        self._body.setText(self._tw.text)
        self._stt.setText("· 转写完成")
        self._build_options()

    def _build_options(self) -> None:
        self._clear_options()
        for i, opt in enumerate(self._options):
            btn = QPushButton(opt)
            btn.setObjectName("commsOption")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(
                lambda _=False, idx=i: self.option_selected.emit(
                    self._dialogue_id or "", idx
                )
            )
            self._options_layout.addWidget(btn)
        self.adjustSize()
        self._request_relayout()

    def _request_relayout(self) -> None:
        """内容高度变化后请宿主重排（Cockpit 负责几何）。"""
        self.adjustSize()
        parent = self.parentWidget()
        apply_geometry = getattr(parent, "_apply_geometry", None)
        if callable(apply_geometry):
            apply_geometry()

    def _clear_options(self) -> None:
        while self._options_layout.count():
            item = self._options_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # ---- 交互：标题栏拖动 / 点击正文跳过打字机 ----

    @property
    def moved_by_user(self) -> bool:
        return self._moved_by_user

    def mousePressEvent(self, event) -> None:  # noqa: N802
        in_header = event.position().y() <= self._header.height() + 2
        if event.button() == Qt.MouseButton.LeftButton and in_header:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        if self._tw is not None and self._typing:
            self._tw.reveal_all()
            self._finish_typing()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None:
            pos = event.globalPosition().toPoint() - self._drag_offset
            parent = self.parentWidget()
            if parent is not None:
                pos.setX(min(max(0, pos.x()), max(0, parent.width() - self.width())))
                pos.setY(min(max(0, pos.y()), max(0, parent.height() - self.height())))
            self.move(pos)
            self._moved_by_user = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None:
            self._drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

