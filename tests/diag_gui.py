"""诊断脚本：输出 GL 上下文 / FBO / 异常细节。

用法：uv run python tests/diag_gui.py
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from Airis.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    gl = window.gl_widget

    def dump() -> None:
        print("DIAG: 窗口尺寸", gl.width(), gl.height())
        print("DIAG: defaultFramebufferObject =", gl.defaultFramebufferObject())
        print("DIAG: context 是否有效 =", gl.context().isValid() if gl.context() else "N/A")
        print("DIAG: ctx moderngl =", gl.ctx)
        print("DIAG: program =", gl.program)
        print("DIAG: vao_ground =", gl.vao_ground)

        # 像素级验证：makeCurrent 后直接读回当前 FBO 内容
        gl.makeCurrent()
        fbo = gl.ctx.detect_framebuffer()
        print("DIAG: detect_framebuffer() =", fbo)
        if fbo is not None:
            data = fbo.read(components=3, alignment=1)
            arr = memoryview(data)
            n = len(arr)
            nz = sum(1 for i in range(0, min(n, 4000000), 3) if arr[i] != 0)
            print(f"DIAG: fbo.read() 采样非零像素 = {nz}")
        app.quit()

    QTimer.singleShot(1200, dump)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
