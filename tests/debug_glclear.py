"""终极隔离测试：用原始 OpenGL glClear 清成红色（绕过 moderngl）。

- 第 1 帧：原始 glClear 亮红 → 检查 grabFramebuffer 是否变红
- 第 2 帧：moderngl 渲染三角形 → 检查是否有内容

自动截图 + 统计后退出。
用法: uv run python tests/debug_glclear.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl
import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QOpenGLContext, QOpenGLFunctions, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

from _output import clean_output_dir, output_path

GL_COLOR_BUFFER_BIT = 0x00004000

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

    def paintGL(self) -> None:
        self.painted += 1
        fns = QOpenGLFunctions(QOpenGLContext.currentContext())
        if self.painted == 1:
            # 方法1：原始 glClear 亮红（完全不碰 moderngl 的 fbo 状态）
            fns.glClearColor(1.0, 0.0, 0.0, 1.0)
            fns.glClear(GL_COLOR_BUFFER_BIT)
            print(f"[frame {self.painted}] 原始 glClear 红色")
        else:
            # 方法2：moderngl 渲染（先绑定 Qt fbo）
            qt_fbo = self.defaultFramebufferObject()
            fb = self.ctx.detect_framebuffer()
            if fb is not None:
                self.ctx.fbo = fb
            self.ctx.clear()
            self.vao.render(moderngl.TRIANGLES)
            print(
                f"[frame {self.painted}] moderngl 渲染 (qt_fbo={qt_fbo}, "
                f"ctx.fbo.glo={getattr(self.ctx.fbo, 'glo', '?')})"
            )


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/
    app = QApplication(sys.argv)
    w = W()
    w.show()

    def finish() -> None:
        img = w.grabFramebuffer()
        out = output_path("glclear_diag.png")
        img.save(out)
        arr = np.array(Image.open(out).convert("RGB"))
        red = int(((arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)).sum())
        nonblack = int((arr.sum(axis=2) > 30).sum())
        print(
            f"[done] {out} mean={arr.mean():.2f} nonblack={nonblack}/{arr.shape[0] * arr.shape[1]} red={red}"
        )
        app.quit()

    QTimer.singleShot(800, finish)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
