"""稳定保持的三角形验证窗口（供人工确认渲染是否正常）。

用法: uv run python tests/triangle_persist.py
- 窗口保持打开，直到你关闭它（不会自动退出）
- 每 30 帧打印当前 FBO 读回的像素统计（客观数据，可看终端）
- 按键 S 保存截图到 tests/output/triangle_persist.png
- 按键 Q / Esc 退出

说明：显式请求 OpenGL 3.3 Core Profile + 深度缓冲（moderngl 的要求）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QOpenGLContext, QOpenGLFunctions, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

from _output import clean_output_dir, output_path

# 在创建 QApplication 之前，请求 OpenGL 3.3 Core Profile + 深度缓冲
fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)

GL_FRAMEBUFFER = 0x8D40
GL_COLOR_BUFFER_BIT = 0x00004000

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


class TriWidget(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = None
        self.prog = None
        self.vao = None
        self.frames = 0
        self.setWindowTitle("三角形验证  [S]=保存截图  [Q/Esc]=退出")
        self.resize(480, 480)

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        info = self.ctx.info
        print(f"[init] version : {info.get('GL_VERSION')}")
        print(f"[init] renderer: {info.get('GL_RENDERER')}")
        print(f"[init] fbo     : {self.ctx.detect_framebuffer()}")
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
        self.ctx.clear_color = (0.12, 0.12, 0.25, 1.0)

    def paintGL(self) -> None:
        self.makeCurrent()
        fns = QOpenGLFunctions(QOpenGLContext.currentContext())
        qt_fbo = self.defaultFramebufferObject()
        # 关键修复：原始 API 强制绑定 Qt FBO（moderngl 的 ctx.fbo 在此环境下不可靠）
        fns.glBindFramebuffer(GL_FRAMEBUFFER, qt_fbo)
        fns.glViewport(0, 0, self.width(), self.height())
        fns.glClearColor(0.12, 0.12, 0.25, 1.0)
        fns.glClear(GL_COLOR_BUFFER_BIT)
        self.vao.render(moderngl.TRIANGLES)
        self.frames += 1
        if self.frames % 30 == 0:
            self._report()

    def _report(self) -> None:
        """读回当前 FBO 并打印像素统计（客观判断三角形是否存在）。"""
        try:
            fbo = self.ctx.detect_framebuffer()
            w, h = self.width(), self.height()
            buf = fbo.read(viewport=(0, 0, w, h), components=3, alignment=1)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            nonblack = int((arr.sum(axis=2) > 30).sum())
            print(
                f"[frame {self.frames}] fbo={fbo} size={fbo.size} "
                f"nonblack={nonblack}/{w * h}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[frame {self.frames}] read 失败: {exc}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            QApplication.instance().quit()
        elif event.key() == Qt.Key.Key_S:
            img = self.grabFramebuffer()
            out = output_path("triangle_persist.png")
            img.save(out)
            print(f"[save] 已保存 {out}")


def main() -> int:
    clean_output_dir()  # AGENTS.md 4.5：运行前清空 tests/output/，保证按 S 保存时有目录
    app = QApplication(sys.argv)
    w = TriWidget()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
