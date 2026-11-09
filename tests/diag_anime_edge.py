"""诊断：anime 边缘值可视化分布（u_debug_edge 调试通道）。

用法：uv run python tests/diag_anime_edge.py

用 u_debug_edge 把 d_edge / n_edge / edge 输出为灰度（show*10 → [0,1]），
分析灰度分布分位数 → 得到各通道边缘值的实际量级，指导阈值/权重标定。
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


def _img_to_np(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.bits(), dtype=np.uint8, count=h * bpl).reshape(h, bpl)
    return arr[:, : w * 3].reshape(h, w, 3).copy()


def _analyze(name: str, img: np.ndarray) -> None:
    rgb = img.astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    terrain = lum >= _BG_LUM
    g = lum[terrain] / 255.0          # debug 灰度（≈ clamp(edge*10,0,1)）
    if g.size == 0:
        print(f"[{name}] no terrain")
        return
    # show = edge*10 → edge = show/10 = g/10
    q = np.percentile(g, [50, 75, 90, 95, 99, 99.9, 100])
    frac_gt_06 = float((g > 0.6).mean())          # edge > 0.06
    frac_gt_03 = float((g > 0.3).mean())          # edge > 0.03
    frac_gt_10 = float((g > 1.0).mean())          # edge > 0.10
    print(
        f"[{name}] n={g.size} p50={q[0]:.3f} p75={q[1]:.3f} p90={q[2]:.3f} "
        f"p95={q[3]:.3f} p99={q[4]:.3f} p999={q[5]:.3f} max={q[6]:.3f} "
        f"| >0.06:{frac_gt_06*100:5.2f}% >0.03:{frac_gt_03*100:5.2f}% >0.10:{frac_gt_10*100:5.2f}%"
    )


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

    # 统一 baseline：threshold 0.06 / depth 1.0 / normal 1.5
    gl.makeCurrent()
    prog["u_outline_threshold"].value = 0.06
    prog["u_depth_edge_scale"].value = 1.0
    prog["u_normal_edge_scale"].value = 1.5
    gl.doneCurrent()

    for mode, name in [(1, "d_edge(NDC z)"), (2, "n_edge(normal)"), (3, "edge=max")]:
        gl.makeCurrent()
        prog["u_debug_edge"].value = float(mode)
        gl.doneCurrent()
        gl.update()
        app.processEvents()
        img = gl.grabFramebuffer()
        _analyze(name, _img_to_np(img))
        gl.grabFramebuffer().save(f"tests/output/anime_edge_mode{mode}.png")

    print("DIAG_ANIME_EDGE_DONE")
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
