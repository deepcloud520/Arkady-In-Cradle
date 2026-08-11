"""深度诊断：对比 moderngl 检测的 FBO 与 Qt 实际使用的 FBO。

在 paintGL 里打印：
- moderngl detect_framebuffer() 的 GL 对象 id
- Qt QOpenGLWidget.defaultFramebufferObject() 的 id
- 原始 GL_FRAMEBUFFER_BINDING

自动截图 + 统计后退出。
用法: uv run python tests/debug_fbo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl
import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

from _output import clean_output_dir, output_path

fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)

VERT = """
#version 330 core
in vec2 in_vert;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_color = in_color;
}
"""
FRAG = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


class W(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = None
        self.prog = None
        self.vao = None
        self.painted = 0
        self.resize(480, 480)

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        verts = np.array(
            [-0.6, -0.6, 1, 0, 0, 0.6, -0.6, 0, 1, 0, 0.0, 0.6, 0, 0, 1],
            dtype=np.float32,
        )
        vbo = self.ctx.buffer(verts.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "2f 3f", "in_vert", "in_color")])
        self.ctx.clear_color = (0.12, 0.12, 0.25, 1.0)

    def paintGL(self) -> None:
        self.painted += 1
        if self.painted == 1:
            self._diag()

        # 尝试方法 A：detect_framebuffer 绑定
        fb = self.ctx.detect_framebuffer()
        if fb is not None:
            self.ctx.fbo = fb
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLES)
        if self.painted == 1:
            self._diag()

    def _diag(self) -> None:
        try:
            fb = self.ctx.detect_framebuffer()
            fbid = getattr(fb, "glo", "?")
            qt_id = self.defaultFramebufferObject()
            print(f"[diag] moderngl detect fbo.glo = {fbid}")
            print(f"[diag] Qt defaultFramebufferObject = {qt_id}")
            print(f"[diag] screen.glo = {getattr(self.ctx.screen, 'glo', '?')}")
            print(f"[diag] window size = {self.width()}x{self.height()}")
            print(f"[diag] ctx.fbo.glo = {getattr(self.ctx.fbo, 'glo', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[diag] 异常: {exc}")


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/
    app = QApplication(sys.argv)
    w = W()
    w.show()

    def finish() -> None:
        img = w.grabFramebuffer()
        out = output_path("fbo_diag.png")
        img.save(out)
        arr = np.array(Image.open(out).convert("RGB"))
        nonblack = int((arr.sum(axis=2) > 30).sum())
        print(
            f"[done] {out} mean={arr.mean():.2f} nonblack={nonblack}/{arr.shape[0] * arr.shape[1]}"
        )
        app.quit()

    QTimer.singleShot(800, finish)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
