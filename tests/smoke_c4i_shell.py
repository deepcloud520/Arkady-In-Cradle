"""C4I 驾驶舱冒烟：离屏渲染一帧浮层布局（2026-11-03 新增）。

不需要 OpenGL / 模拟引擎——只验证**布局与配色**（地图铺满 + 时间栏 + 侧栏卡片 +
底部消息日志 + 左侧通话弹窗），产物写入 ``tests/output/``（AGENTS.md 4.5）。

运行::

    .venv\\Scripts\\python.exe tests\\smoke_c4i_shell.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 允许直接 `python tests/smoke_c4i_shell.py`（把仓库根加进 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from Airia.render.hud.log_widget import LogWidget
from Airia.ui import palette as pal
from Airia.ui.cockpit import Cockpit
from Airia.ui.comms_popup import CommsPopup
from Airia.ui.fonts import load_ui_fonts
from Airia.ui.message_panel import MessageLogPanel
from Airia.ui.metrics import for_viewport
from Airia.ui.timebar import TimeBar, TimeBarState, format_cmo_line

from _output import OUTPUT_DIR  # noqa: E402  (tests/ 目录在 sys.path 中)


class PlaceholderMap(QWidget):
    """假地图（网格 + 等高线感），仅用于布局预览。"""

    def __init__(self, bg: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg = QColor(bg)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)
        grid = self._bg.darker(112)
        p.setPen(QPen(grid, 1))
        step = 96
        for x in range(0, self.width(), step):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            p.drawLine(0, y, self.width(), y)
        # 等高线感（几个同心椭圆）
        p.setPen(QPen(self._bg.darker(118), 1))
        cx, cy = self.width() * 0.38, self.height() * 0.42
        for i in range(5):
            p.drawEllipse(
                QRectF(cx - 160 - i * 40, cy - 120 - i * 30,
                       320 + i * 80, 240 + i * 60)
            )
        p.end()


class SideSample(QWidget):
    """侧栏内容样例（正式版由 DetailPanel 填充）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        p = pal.current()
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(4)
        for text, color, size, bold in (
            ("M1A2 #03", p.text, 13, True),
            ("北约 / 主战坦克 / 战斗中", p.text_dim, 9, False),
            ("MSR  PMC · 部分可用", p.warn, 10, True),
            ("机动  受限 · 左履带断裂", p.warn, 9, False),
            ("火力  可用", p.ok, 9, False),
            ("防护  可用 · 弹药架受损", p.ok, 9, False),
            ("弹药  38 / 40", p.text, 9, False),
            ("编队  进攻 · 通信 87%", p.text, 9, False),
        ):
            lab = QLabel(text, self)
            f = QFont()
            f.setPixelSize(size)
            f.setBold(bold)
            lab.setFont(f)
            lab.setStyleSheet(f"color: {color}; background: transparent;")
            v.addWidget(lab)
        v.addStretch(1)


def build(theme: str, width: int = 1440, height: int = 900) -> QWidget:
    p = pal.set_current(theme)
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(pal.qss(p))

    metrics = for_viewport(width, height)
    log = LogWidget()
    message_panel = MessageLogPanel(log)
    cockpit = Cockpit(
        map_widget=PlaceholderMap(p.map_bg),
        side_widget=SideSample(),
        message_panel=message_panel,
        comms=CommsPopup(),
        timebar=TimeBar(),
    )
    cockpit.resize(width, height)
    cockpit._apply_geometry()

    cockpit.timebar.set_metrics(metrics)
    cockpit.timebar.set_state(
        TimeBarState(
            clock="07:45:44",
            line=format_cmo_line(
                "07:45:44", "05:45:44", 23 * 3600 + 59 * 60, 6818
            ),
            selected="Selected: 1x M1A2 #03",
            selected_kind="single",
        )
    )

    message_panel.set_palette(p)
    message_panel.set_metrics(metrics)
    message_panel.push_system("战术终端已就绪：右侧 UNIT STATUS，L 打开日志。")
    message_panel.push_events([
        {"time": 12.9, "kind": "destroyed", "data": {"target": 7}},
        {"time": 12.3, "kind": "hit", "data": {"shooter": 3, "target": 7}},
        {"time": 12.3, "kind": "fire", "data": {"shooter": 3}},
    ])
    message_panel.set_collapsed(False)
    cockpit._apply_geometry()

    comms = cockpit.comms
    comms.set_state(
        type("S", (), {
            "active_dialogue": type("D", (), {
                "dialogue_id": "demo",
                "speaker": "第 7 装甲连 / ALPHA",
                "text": "敌军装甲连正在渡河，你部立即前出阻截。",
                "options": ["明白", "请求火力支援"],
                "history": [],
                "is_finished": False,
            })(),
        })()
    )
    comms._tw.reveal_all()
    comms._finish_typing()
    cockpit._apply_geometry()
    return cockpit


def main() -> int:
    # AGENTS.md 4.5：产物统一落 tests/output/。这里只清本脚本自己的旧产物
    # （整目录清空在部分环境会因 .pytest-basetemp 的 ACL 拒绝而失败）。
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("ui_c4i_shell_*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    app = QApplication([])
    load_ui_fonts(app)

    saved = []
    for theme in ("dark", "light"):
        widget = build(theme)
        app.processEvents()
        pm = widget.grab()
        path = out / f"ui_c4i_shell_{theme}.png"
        pm.save(str(path))
        saved.append((path, pm.width(), pm.height()))

    for path, w, h in saved:
        print(f"saved {path}  {w}x{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
