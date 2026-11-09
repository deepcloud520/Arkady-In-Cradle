"""顶部时间栏（CMO 风格：一条半透明黑底白字，2026-11-04 定稿）。

用户口径（第三次收敛）：
    - **不要网页前端风格的卡片/分块/渐变**；
    - 就是 CMO 那样「一块半透明黑底 + 白字」，一行信息 + 一行 Selected：

        Time: 07:45:44 · Local: 07:45:44 · Zulu: 05:45:44 · 23 hr 59 min to go · Camera Alt: 6818 m
        Selected: 1x LCM-6

仍是控件实现（QFrame + QLabel + QSS），没有 QPainter 自绘。
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from Airia.ui import palette as pal
from Airia.ui.metrics import Metrics, for_viewport


@dataclass(frozen=True)
class TimeBarState:
    """时间栏数据快照（相等则跳过重建）。"""

    clock: str = "00:00:00"
    clock_note: str = "当地时间"
    blocks: tuple[tuple[str, str], ...] = ()
    selected: str = ""
    selected_kind: str = "none"     # none | single | multi
    line: str = ""                  # 整行文本（CMO 风格，优先）


def format_selected(
    kind: str,
    name: str = "",
    formation: str = "",
    count: int = 1,
    multi_formation: bool = False,
) -> str:
    """Selected 行文本（§2.7 三态）。"""
    if kind == "single" and name:
        tail = f" · {formation}" if formation else ""
        return f"1x {name}{tail}"
    if kind == "multi" and count > 0:
        if multi_formation or not formation:
            return f"{count} units · 多编队"
        return f"{count} units · {formation}"
    return "Selected: 无"


def format_cmo_line(
    local: str,
    zulu: str,
    remain_s: float,
    alt_m: float,
    time_scale: str | None = None,
) -> str:
    """CMO 风格的一行信息（与截图同构）。

    ``Time / Local / Zulu / 剩余 / Camera Alt``，用 `` · `` 分隔。
    """
    hours = int(max(0.0, remain_s) // 3600)
    minutes = int((max(0.0, remain_s) % 3600) // 60)
    parts = [
        f"Time: {local}",
        f"Local: {local}",
        f"Zulu: {zulu}",
        f"{hours} hr {minutes} min to go",
        f"Camera Alt: {alt_m:.0f} m",
    ]
    if time_scale:
        parts.append(f"Time Scale: {time_scale}")
    return "  ·  ".join(parts)


class TimeBar(QWidget):
    """左上角信息条（单行 + Selected 行，半透明黑底白字）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._metrics: Metrics = for_viewport(1440, 900)
        self._palette = pal.current()
        self._state = TimeBarState()

        self.setFixedHeight(self._metrics.timebar_h)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._card = QFrame(self)
        self._card.setObjectName("timebarCard")
        outer.addWidget(self._card, 0, Qt.AlignmentFlag.AlignLeft)
        outer.addStretch(1)

        layout = QVBoxLayout(self._card)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.setSpacing(0)

        self._line = QLabel("", self._card)
        self._line.setObjectName("tbLine")
        self._selected = QLabel("", self._card)
        self._selected.setObjectName("tbSel")
        layout.addWidget(self._line)
        layout.addWidget(self._selected)
        self._selected.hide()

        self._apply_style()
        self._rebuild()

    # ------------------------------------------------------------------

    def set_metrics(self, metrics: Metrics) -> None:
        if metrics == self._metrics:
            return
        self._metrics = metrics
        self.setFixedHeight(metrics.timebar_h)
        self._apply_style()
        self._fit_card()

    def apply_palette(self, palette: pal.Palette) -> None:
        self._palette = palette
        self._apply_style()

    def set_state(self, state: TimeBarState) -> None:
        if state == self._state:
            return
        self._state = state
        self._rebuild()

    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        k = self._metrics.k
        line_px = max(18, int(round(13 * k)))
        sel_px = max(15, int(round(11 * k)))
        # CMO 味道：纯半透明黑底 + 白字，无边框、无渐变、无圆角
        self._card.setStyleSheet(
            f"""
#timebarCard {{
    background: rgba(0, 0, 0, 140);
    border: none;
}}
#tbLine {{
    color: #ffffff;
    font-size: {line_px}px;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    background: transparent;
}}
#tbSel {{
    color: #d8dde3;
    font-size: {sel_px}px;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    background: transparent;
}}
"""
        )
        lay = self._card.layout()
        if lay is not None:
            pad_x = max(6, int(round(10 * k)))
            lay.setContentsMargins(pad_x, 2, pad_x, 2)

    def _rebuild(self) -> None:
        st = self._state
        if st.line:
            self._line.setText(st.line)
        else:
            # 兼容旧调用：用 clock + blocks 拼一行
            cols = "  ·  ".join(f"{key}: {value}" for key, value in st.blocks)
            self._line.setText(
                f"Time: {st.clock}" + (f"  ·  {cols}" if cols else "")
            )

        has_selected = st.selected_kind != "none" and bool(st.selected)
        self._selected.setVisible(has_selected)
        if has_selected:
            self._selected.setText(st.selected)

        self._fit_card()

    def _fit_card(self) -> None:
        self._card.adjustSize()
        width = min(max(1, self._card.sizeHint().width()), max(1, self.width()))
        self._card.setGeometry(0, 0, width, self.height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_card()
