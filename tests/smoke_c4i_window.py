"""主窗口冒烟：真实 MainWindow（含 GL + 模拟层）截图（2026-11-03 新增）。

验证「4 Tab → C4I 浮层」接线是否真的跑得起来：菜单、时间栏、侧栏卡片、
底部消息日志、通话弹窗、地形渲染。产物落 ``tests/output/``。

运行::

    .venv\\Scripts\\python.exe tests\\smoke_c4i_window.py
    .venv\\Scripts\\python.exe tests\\smoke_c4i_window.py --offscreen   # 无 GL 布局验证
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

OFFSCREEN = "--offscreen" in sys.argv
if OFFSCREEN:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from Airia.ui.fonts import load_ui_fonts
from Airia.ui.main_window import MainWindow

from _output import OUTPUT_DIR  # noqa: E402


def main() -> int:
    app = QApplication([])
    load_ui_fonts(app)

    win = MainWindow()
    win.resize(1440, 900)
    win.show()

    # 让 GL 初始化 + 模拟层跑若干帧
    deadline = time.time() + 3.0
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "ui_c4i_window_offscreen.png" if OFFSCREEN else "ui_c4i_window.png"
    path = OUTPUT_DIR / name
    win.grab().save(str(path))
    print(f"saved {path}  {win.width()}x{win.height()}")

    # 打印关键接线状态，便于无 GUI 环境断言
    m = win.cockpit.metrics
    print(
        f"layout={m.mode} k={m.k:.2f} sidebar={m.sidebar_w} "
        f"timebar={m.timebar_h} bottom={m.bottombar_h} "
        f"units={win.adapter.unit_count() if win.adapter else 0}"
    )
    win.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
