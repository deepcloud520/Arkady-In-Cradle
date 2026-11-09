"""对照实验：GLWidget 独立运行（不放入 MainWindow），判断黑屏来源。

用法: uv run python tests/debug_terrain_gui.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from Airia.render.gl_widget import GLWidget
from _output import clean_output_dir, output_path


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/
    app = QApplication(sys.argv)
    w = GLWidget()
    w.resize(640, 480)
    w.show()

    def finish() -> None:
        img = w.grabFramebuffer()
        out = output_path("debug_terrain.png")
        img.save(out)
        # 客观统计
        arr = np.array(Image.open(out).convert("RGB"))
        nonblack = int((arr.sum(axis=2) > 30).sum())
        print(f"[done] {out} shape={arr.shape} mean={arr.mean():.2f} nonblack={nonblack}")
        app.quit()

    QTimer.singleShot(1200, finish)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
