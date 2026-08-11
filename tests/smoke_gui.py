"""GUI 冒烟测试：创建主窗口并渲染一帧后自动退出。

用法：uv run python tests/smoke_gui.py
"""
from __future__ import annotations

import logging
import os
import sys

# 确保项目根在导入路径中（脚本以文件方式运行时 sys.path[0] 指向 tests/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from Airis.ui.main_window import MainWindow
from _output import clean_output_dir, output_path


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    frames = {"count": 0}

    def on_frame() -> None:
        """在真实帧交换后抓取，确保拿到已渲染的帧。"""
        frames["count"] += 1
        if frames["count"] >= 5:
            img = window.gl_widget.grabFramebuffer()
            out = output_path("terrain_preview.png")
            # QImage.save 只接受 str，不接受 Path（PySide6 签名限制）
            img.save(str(out))
            print(f"GUI_SMOKE_OK -> {out} (frames={frames['count']})")
            app.quit()

    def timeout() -> None:
        print("GUI_TIMEOUT: 未收到帧交换，可能无显示环境")
        app.quit()

    window.gl_widget.frameSwapped.connect(on_frame)
    QTimer.singleShot(5000, timeout)
    app.exec()
    print("GUI_EXIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
