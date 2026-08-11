"""自动验证：paintGL 里用 ctx.screen.use() 渲染三角形（moderngl 官方 Qt 集成方式）。

自动保存 PNG + 打印像素统计后自动退出，无需人工确认。
用法: uv run python tests/debug_screen_use.py
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
        self.resize(480, 480)

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        print(f"[init] {self.ctx.info.get('GL_VERSION')} / {self.ctx.info.get('GL_RENDERER')}")
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        verts = np.array(
            [-0.6, -0.6, 1, 0, 0, 0.6, -0.6, 0, 1, 0, 0.0, 0.6, 0, 0, 1],
            dtype=np.float32,
        )
        vbo = self.ctx.buffer(verts.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "2f 3f", "in_vert", "in_color")])
        self.ctx.clear_color = (0.12, 0.12, 0.25, 1.0)

    def paintGL(self) -> None:
        # 关键：绑定默认屏幕缓冲（moderngl 官方 Qt 集成方式）
        self.ctx.screen.use()
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLES)


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/
    app = QApplication(sys.argv)
    w = W()
    w.show()

    def finish() -> None:
        img = w.grabFramebuffer()
        out = output_path("screen_use.png")
        img.save(out)
        arr = np.array(Image.open(out).convert("RGB"))
        nonblack = int((arr.sum(axis=2) > 30).sum())
        red = int(((arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)).sum())
        print(
            f"[done] {out} mean={arr.mean():.2f} nonblack={nonblack}/{arr.shape[0] * arr.shape[1]} red={red}"
        )
        app.quit()

    QTimer.singleShot(1000, finish)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
