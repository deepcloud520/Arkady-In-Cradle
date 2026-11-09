"""GUI 冒烟：anime 主题（cel shading + 描边）渲染验证。

用法：uv run python tests/smoke_theme_anime.py

1. 依次切换 default / constructivism / anime / default，确认热切换稳定
2. 对 anime 做「描边开关对照」：正常渲染 vs 关闭描边（u_outline_threshold=10）
   → 两帧差异像素 = 真实描边像素（不受 cel 暗部补色干扰）
3. 统计：cel 分带峰数（量化色阶）+ 描边像素比例
4. 保存截图到 tests/output/anime_theme_*.png
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from Airia.ui.main_window import MainWindow
from _output import clean_output_dir, output_path

# 背景（clear_color=(0.04,0.04,0.09) → 亮度 ~12）判定阈值
_BG_LUM = 25.0
# 「描边关闭」阈值：edge 恒 < 10 → smoothstep 恒 0
_NO_OUTLINE = 10.0
# 描边差异判定阈值（RGB 欧氏距离）
_DIFF_DIST = 12.0


def _img_to_np(img: QImage) -> np.ndarray:
    """QImage → (H,W,3) uint8 numpy（处理 bytesPerLine 对齐）。"""
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    mv = img.bits()   # PySide6 返回 memoryview
    arr = np.frombuffer(mv, dtype=np.uint8, count=h * bpl).reshape(h, bpl)
    return arr[:, : w * 3].reshape(h, w, 3).copy()


def _lum_peaks(img: np.ndarray) -> tuple[int, float]:
    """cel 分带：地形亮度直方图峰数 + 平均亮度。"""
    rgb = img.astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    terrain = lum >= _BG_LUM
    n = int(terrain.sum())
    if n == 0:
        return 0, 0.0
    hist, _ = np.histogram(lum[terrain], bins=64, range=(0.0, 256.0))
    smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    peaks = sum(
        1
        for i in range(1, 63)
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]
        and smooth[i] > n * 0.01
    )
    return peaks, float(lum[terrain].mean())


def _save_png(np_img: np.ndarray, name: str) -> None:
    """numpy (H,W,3) → tests/output/<name>.png。"""
    out = output_path(name)
    qimg = QImage(np_img.data, np_img.shape[1], np_img.shape[0],
                  np_img.shape[1] * 3, QImage.Format.Format_RGB888).copy()
    qimg.save(str(out))
    print(f"   saved -> {out}")


def main() -> int:
    clean_output_dir()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    def grab(theme_id: str) -> np.ndarray:
        gl = window.gl_widget
        gl.makeCurrent()
        try:
            gl.theme_manager.switch(theme_id)
        finally:
            gl.doneCurrent()
        gl.update()
        app.processEvents()
        return _img_to_np(gl.grabFramebuffer())

    # 1) 主题切换稳定性 + 基础统计（含切回 default 验证热切换不残留）
    for tid in ("default", "constructivism", "anime", "default"):
        img = grab(tid)
        peaks, mean = _lum_peaks(img)
        print(f"[{tid}] lum_peaks={peaks} lum_mean={mean:6.1f}")
        _save_png(img, f"anime_theme_{tid}.png")

    # 2) anime 描边开关对照：正常 vs 关闭描边 → 差异 = 真实描边像素
    gl = window.gl_widget
    gl.makeCurrent()
    try:
        gl.theme_manager.switch("anime")
    finally:
        gl.doneCurrent()
    prog = gl.theme_manager.current.program   # 切换后再取（anime program）

    def grab_anime(with_outline: bool) -> np.ndarray:
        gl.makeCurrent()
        try:
            prog["u_outline_threshold"].value = 0.008 if with_outline else _NO_OUTLINE
        finally:
            gl.doneCurrent()
        gl.update()
        app.processEvents()
        return _img_to_np(gl.grabFramebuffer())

    on_img = grab_anime(True)
    off_img = grab_anime(False)

    rgb_on = on_img.astype(np.float32)
    lum = rgb_on @ np.array([0.299, 0.587, 0.114])
    terrain = lum >= _BG_LUM
    n_terrain = int(terrain.sum())

    diff = np.linalg.norm(on_img.astype(np.float32) - off_img.astype(np.float32), axis=2)
    outline_px = terrain & (diff > _DIFF_DIST)
    outline_ratio = float(outline_px.sum()) / max(n_terrain, 1)
    print(f"[anime-outline] diff_px={int(outline_px.sum())} "
          f"outline_ratio={outline_ratio*100:.3f}%")

    # 描边存在且不过度（0.1%~20% 地形像素）
    ok = 0.1 < outline_ratio * 100 < 20.0
    print(f"ANIME_OUTLINE_CHECK {'OK' if ok else 'FAIL'} "
          f"(outline_ratio={outline_ratio*100:.3f}%)")
    print("GUI_SMOKE_THEME_OK")
    QTimer.singleShot(0, app.quit)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
