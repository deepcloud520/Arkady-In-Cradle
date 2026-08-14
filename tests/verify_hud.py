"""HUD 标记渲染验证（Phase 5c 冒烟）：抓帧后统计阵营圆点像素。

用法：uv run python tests/verify_hud.py
验证逻辑（参考 2026-08-13 单位标记像素统计方法）：
    - 友军绿圆 (0.30,0.85,0.45) → 8 位约 (76,217,115)
    - 敌军红圆 (0.92,0.20,0.16) → 8 位约 (235,51,41)
    - 帧中应同时存在两种颜色的像素（标记成功渲染）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from Airis.ui.main_window import MainWindow
from _output import clean_output_dir, output_path

# 阵营标记目标色（0~255，含抗锯齿过渡容忍）
ALLY = np.array([76, 217, 115], dtype=np.int64)
ENEMY = np.array([235, 51, 41], dtype=np.int64)
_TOL = 40


def _count_near(img: np.ndarray, target: np.ndarray, tol: int) -> int:
    """统计与目标色曼哈顿距离 < tol 的像素数。"""
    d = np.abs(img.astype(np.int64) - target).sum(axis=2)
    return int((d < tol * 3).sum())


def main() -> int:
    clean_output_dir()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    frames = {"count": 0}

    def on_frame() -> None:
        frames["count"] += 1
        if frames["count"] >= 5:
            img = window.gl_widget.grabFramebuffer()
            out = output_path("hud_preview.png")
            img.save(str(out))

            img = img.convertToFormat(QImage.Format.Format_RGB888)
            w, h = img.width(), img.height()
            bpl = img.bytesPerLine()
            buf = img.bits()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)[:, : w * 3]
            rgb = arr.reshape(h, w, 3).astype(np.int64)
            n_ally = _count_near(rgb, ALLY, _TOL)
            n_enemy = _count_near(rgb, ENEMY, _TOL)
            print(f"HUD_PIXELS ally={n_ally} enemy={n_enemy} -> {out}")
            ok = n_ally > 0 and n_enemy > 0
            print("HUD_VERIFY_" + ("OK" if ok else "FAIL"))
            app.quit()

    def timeout() -> None:
        print("GUI_TIMEOUT: 未收到帧交换")
        app.quit()

    window.gl_widget.frameSwapped.connect(on_frame)
    QTimer.singleShot(8000, timeout)
    app.exec()
    print("GUI_EXIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
