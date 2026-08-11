"""极简 PySide6 + ModernGL 三角形测试（隔离 Qt FBO 问题的最小复现）。

用法: uv run python tests/minimal_triangle.py
输出: tests/output/minimal_triangle.png + 每 20 帧打印非黑像素统计
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

from _output import clean_output_dir, output_path

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


class TriangleWidget(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = None
        self.prog = None
        self.vao = None
        self.frames = 0

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        print(f"[init] renderer : {self.ctx.info.get('GL_RENDERER')}")
        print(f"[init] detect   : {self.ctx.detect_framebuffer()}")
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        verts = np.array(
            [
                -0.6, -0.6, 1.0, 0.0, 0.0,  # 红
                 0.6, -0.6, 0.0, 1.0, 0.0,  # 绿
                 0.0,  0.6, 0.0, 0.0, 1.0,  # 蓝
            ],
            dtype=np.float32,
        )
        vbo = self.ctx.buffer(verts.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(vbo, "2f 3f", "in_vert", "in_color")]
        )
        self.ctx.clear_color = (0.1, 0.1, 0.2, 1.0)

    def paintGL(self) -> None:
        self.makeCurrent()
        fbo = self.ctx.detect_framebuffer()
        if fbo is not None:
            self.ctx.fbo = fbo
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLES)
        self.frames += 1

        if self.frames % 20 == 0:
            buf = self.ctx.fbo.read(components=3)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                self.height(), self.width(), 3
            )
            nonblack = int((arr.sum(axis=2) > 30).sum())
            total = arr.shape[0] * arr.shape[1]
            print(f"[frame {self.frames}] 非黑像素: {nonblack}/{total}")


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/
    app = QApplication(sys.argv)
    w = TriangleWidget()
    w.resize(480, 480)
    w.show()

    def finish() -> None:
        img = w.grabFramebuffer()
        out = output_path("minimal_triangle.png")
        img.save(out)
        print(f"[done] saved {out}")
        app.quit()

    QTimer.singleShot(1200, finish)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
