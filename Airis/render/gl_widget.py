"""QOpenGLWidget 渲染视图：2.5D 地形网格（Phase 2）。

- 使用 ctx.detect_framebuffer() 绑定 Qt 管理的 FBO（AGENTS.md 3.6 关键修复）
- 轨道相机交互：左键拖拽旋转、滚轮缩放
"""
from __future__ import annotations

import logging
import os

import moderngl
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from Airis.lulilo.terrain.map import TerrainMap
from Airis.render.camera import OrbitCamera

logger = logging.getLogger(__name__)

VERTEX_SHADER = """
#version 330 core
in vec3 in_vert;
in vec3 in_color;
in vec3 in_normal;
uniform mat4 mvp;
out vec3 v_color;
out vec3 v_normal;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_color = in_color;
    v_normal = in_normal;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in vec3 v_normal;
out vec4 frag_color;
uniform vec3 u_light_dir;    // 世界空间方向光（指向光源）
uniform vec3 u_light_color;  // 光源颜色
uniform float u_ambient;     // 环境光强度
void main() {
    vec3 n = normalize(v_normal);
    // 双面：法线朝下时翻转，避免陡坡/背面全黑
    if (n.z < 0.0) {
        n = -n;
    }
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    vec3 color = v_color * (u_ambient + (1.0 - u_ambient) * diff * u_light_color);
    frag_color = vec4(color, 1.0);
}
"""


class GLWidget(QOpenGLWidget):
    def __init__(self, terrain: TerrainMap | None = None):
        super().__init__()
        self.terrain = terrain if terrain is not None else TerrainMap().generate()
        cfg = self.terrain.config
        self.camera = OrbitCamera(
            target=(
                cfg.width * cfg.cell_size * 0.5,
                cfg.height * cfg.cell_size * 0.5,
                0.0,
            ),
            distance=cfg.width * cfg.cell_size * 1.2,
        )
        self._last_pos: tuple[float, float] | None = None

        self.ctx = None
        self.program = None
        self.vao_ground = None
        self.vao_bridge = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def initializeGL(self):
        try:
            self.ctx = moderngl.create_context()
            info = self.ctx.info
            logger.info("OpenGL %s / %s / %s", info.get("GL_VERSION"), info.get("GL_VENDOR"), info.get("GL_RENDERER"))
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear_color = (0.04, 0.04, 0.09, 1.0)

            self.program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER,
            )
            # 世界空间方向光（模拟日光）：从上方偏东南照射，与环境光配合
            self.program["u_light_dir"].value = (0.45, -0.60, 0.65)
            self.program["u_light_color"].value = (1.0, 0.97, 0.90)
            self.program["u_ambient"].value = 0.42
            self._build_meshes()
        except Exception:
            logger.exception("initializeGL 失败")

    def _build_meshes(self):
        data = self.terrain.to_render_data()
        gv = data["ground_vertices"].astype(np.float32)
        gi = data["ground_indices"].astype(np.uint32)
        vbo = self.ctx.buffer(gv.tobytes())
        ibo = self.ctx.buffer(gi.tobytes())
        self.vao_ground = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f 3f", "in_vert", "in_color", "in_normal")],
            index_buffer=ibo,
        )
        bv = data["bridge_vertices"].astype(np.float32)
        bi = data["bridge_indices"].astype(np.uint32)
        if len(bv):
            vbo_b = self.ctx.buffer(bv.tobytes())
            ibo_b = self.ctx.buffer(bi.tobytes())
            self.vao_bridge = self.ctx.vertex_array(
                self.program,
                [(vbo_b, "3f 3f 3f", "in_vert", "in_color", "in_normal")],
                index_buffer=ibo_b,
            )
        else:
            self.vao_bridge = None

    def paintGL(self):
        if self.ctx is None or self.program is None or self.vao_ground is None:
            return
        try:
            self.makeCurrent()
            w = max(1, self.width())
            h = max(1, self.height())
            self.ctx.viewport = (0, 0, w, h)

            # 关键修复（AGENTS.md 3.6 陷阱）：moderngl 的 Python 层 ctx.fbo 与 C 层
            # mglo.fbo / 实际 GL 绑定不同步（create_context 时缓存），直接 ctx.clear()
            # 会清到错误的缓冲。必须用 fb.use()（moderngl 官方绑定方法，同步三层状态）
            # 显式绑定 Qt 当前 FBO，之后 ctx.clear() / vao.render() 才落到 Qt 帧缓冲。
            fb = self.ctx.detect_framebuffer()
            fb.use()
            self.ctx.clear()  # 清颜色 + 深度

            mvp = self.camera.mvp_matrix(w / h)
            # GLM 矩阵为 column-major，与 OpenGL uniform 布局一致，直接写入
            self.program["mvp"].write(mvp.to_bytes())

            self.vao_ground.render(moderngl.TRIANGLES)
            if self.vao_bridge is not None:
                self.vao_bridge.render(moderngl.TRIANGLES)

            if os.environ.get("AIRC_DUMP_FRAME") == "1":
                self._dump_frame()
        except Exception:
            logger.exception("paintGL 失败")

    def _dump_frame(self) -> None:
        """调试用：渲染后从当前 FBO 读回并保存（环境变量 AIRC_DUMP_FRAME=1 启用）。"""
        fbo = self.ctx.detect_framebuffer()
        if fbo is None:
            logger.warning("dump_frame: detect_framebuffer() 返回 None")
            return
        w, h = self.width(), self.height()
        try:
            data = fbo.read(viewport=(0, 0, w, h), components=3, alignment=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dump_frame: read 失败: %s", exc)
            return
        arr = np.frombuffer(data, dtype=np.uint8)
        logger.info(
            "dump_frame: 非零字节=%d/%d mean=%.1f",
            int(np.count_nonzero(arr)),
            arr.size,
            float(arr.mean()),
        )
        try:
            from PIL import Image

            img = arr.reshape(h, w, 3)
            # 统一输出到 tests/output/（AGENTS.md 4.5 规则）
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "tests",
                "output",
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "frame_internal.png")
            Image.fromarray(img).save(out_path)
            logger.info("dump_frame: 已保存 %s", out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dump_frame: 保存失败: %s", exc)

    def resizeGL(self, width, height):
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, width, height)

    # ---- 相机交互 ----
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # 只保存坐标（浮点），不要保存事件对象：Qt 事件用完即销毁
            p = event.position()
            self._last_pos = (p.x(), p.y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_pos = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_pos is None:
            return
        p = event.position()
        dx = p.x() - self._last_pos[0]
        dy = p.y() - self._last_pos[1]
        self.camera.orbit(dx, -dy)
        self._last_pos = (p.x(), p.y())
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.camera.zoom(1.1 if event.angleDelta().y() > 0 else 0.9)
        self.update()
