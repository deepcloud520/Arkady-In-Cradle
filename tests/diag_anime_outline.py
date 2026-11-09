"""诊断：anime 描边阈值扫描。

用法：uv run python tests/diag_anime_outline.py

对 anime 主题扫描 u_outline_threshold / u_depth_edge_scale / u_normal_edge_scale，
统计「接近 u_outline_color 的像素比例」，判断边缘检测响应与最佳参数区间。
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from Airia.render.gl_widget import GLWidget
from _output import clean_output_dir

_BG_LUM = 25.0
_OUTLINE_RGB = np.array([31, 26, 41], dtype=np.float32)
_OUTLINE_DIST = 45.0


def _img_to_np(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.bits(), dtype=np.uint8, count=h * bpl).reshape(h, bpl)
    return arr[:, : w * 3].reshape(h, w, 3).copy()


def _outline_ratio(img: np.ndarray) -> tuple[float, int]:
    rgb = img.astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    terrain = lum >= _BG_LUM
    n = int(terrain.sum())
    if n == 0:
        return 0.0, 0
    dist = np.linalg.norm(rgb - _OUTLINE_RGB, axis=2)
    return float((terrain & (dist < _OUTLINE_DIST)).sum()) / n, n


def main() -> int:
    clean_output_dir()
    app = QApplication(sys.argv)
    gl = GLWidget()
    gl.resize(960, 640)
    gl.show()
    app.processEvents()

    gl.makeCurrent()
    try:
        gl.theme_manager.switch("anime")
    finally:
        gl.doneCurrent()
    prog = gl.theme_manager.current.program

    def grab() -> float:
        gl.update()
        app.processEvents()
        img = gl.grabFramebuffer()
        return _outline_ratio(_img_to_np(img))[0]

    # 1) 阈值扫描（depth_scale=1.0 / normal_scale=1.5 固定）
    print("== u_outline_threshold 扫描（depth=1.0 / normal=1.5）==")
    for t in [0.02, 0.04, 0.06, 0.10]:
        gl.makeCurrent()
        prog["u_outline_threshold"].value = t
        gl.doneCurrent()
        r = grab()
        print(f"  threshold={t:5.3f}  outline_ratio={r*100:6.2f}%")

    # 2) 消融截图（threshold=0.06 固定）：
    #    A=仅深度  B=仅法线  C=双通道  D=无描边
    combos = [
        ("depth_only", 1.0, 0.0),
        ("normal_only", 0.0, 1.5),
        ("both", 1.0, 1.5),
        ("none", 0.0, 0.0),
    ]
    print("== 消融截图（threshold=0.06）==")
    for name, ds, ns in combos:
        gl.makeCurrent()
        prog["u_outline_threshold"].value = 0.06
        prog["u_depth_edge_scale"].value = ds
        prog["u_normal_edge_scale"].value = ns
        gl.doneCurrent()
        r = grab()
        gl.update()
        app.processEvents()
        img = gl.grabFramebuffer()
        out = f"tests/output/anime_ablate_{name}.png"
        img.save(out)
        print(f"  [{name}] ratio={r*100:6.2f}%  saved -> {out}")

    print("DIAG_ANIME_OUTLINE_DONE")
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
