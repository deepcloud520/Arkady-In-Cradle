"""交互式 Z-fighting 诊断查看器（类似 main.py，供人肉找角度看条纹）。

用法: uv run python tests/diag_viewer.py

诊断修改（仅本脚本的渲染/地形层，不碰生产代码）：
- 背景 Clear Color = 亮红 #FF0000（默认开启）：裂缝 → 红背景透出；Z冲突 → 明暗条纹
- 地形/桥顶点颜色强制纯白（默认开启）：排除颜色干扰，专注几何重叠
- 环境光压低 u_ambient=0.15：背光面更暗，明暗对比更强

按键：
  1 / 2  只渲染桥 / 地面+桥（切换后需 1~2s 重建网格）
  3 / 4  生产颜色（钢灰桥+绿地面） / 纯白诊断色
  R      红背景 / 深蓝背景 切换
  F      打印当前相机参数（方便反馈条纹角度）
  Esc    退出

鼠标：左键拖拽旋转、滚轮缩放。状态栏实时显示相机参数。
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

import moderngl
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow

from Airis.lulilo.terrain.map import TerrainMap
from Airis.render.camera import OrbitCamera
from tests.terrain_factory import TerrainFactory

# 与生产 gl_widget.py 相同的着色器（诊断不引入新变量）
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
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform float u_ambient;
void main() {
    vec3 n = normalize(v_normal);
    // 双面：按三角形朝向翻转法线，背面也正确受光（与生产 gl_widget.py 同步）
    if (!gl_FrontFacing) {
        n = -n;
    }
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    vec3 color = v_color * (u_ambient + (1.0 - u_ambient) * diff * u_light_color);
    frag_color = vec4(color, 1.0);
}
"""

# 第一座演示桥（main.py 默认对准）：世界 x∈[200,320] y∈[200,208]，中心 (260,204,0)
BRIDGE_CENTER = (260.0, 204.0, 0.0)


class DiagViewer(QOpenGLWidget):
    """诊断渲染视图：红背景 + 白色网格 + 低环境光，支持按键切换模式。"""

    def __init__(self, terrain: TerrainMap) -> None:
        super().__init__()
        self.terrain = terrain
        # 缓存原始（生产颜色）与白色副本，避免切换时反复重新提取网格
        self._rd = terrain.to_render_data()
        self._rd_white: dict[str, np.ndarray] = {}
        for k, v in self._rd.items():
            w = v.copy()
            if len(w) and k in ("ground_vertices", "bridge_vertices"):
                w[:, 3:6] = 1.0
            self._rd_white[k] = w

        self.ctx = None
        self.program = None
        self.vao_ground = None
        self.vao_bridge = None
        self._last_pos: tuple[float, float] | None = None

        # 诊断开关
        self.diag_red_bg = True
        self.diag_white = True
        self.bridge_only = False

        # 相机：斜上方对准桥1（main.py 默认视角）
        self.camera = OrbitCamera(
            target=BRIDGE_CENTER, distance=120.0, yaw=45.0, pitch=30.0
        )
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    # ------------------------------------------------------------------
    # GL 生命周期
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context(require=460)
        info = self.ctx.info
        print(
            f"OpenGL {info.get('GL_VERSION')} / {info.get('GL_RENDERER')}"
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        # 打印实际深度位数（确认 Qt 是否给了 32 位深度缓冲）
        print(f"Depth bits = {self.context().format().depthBufferSize()}")
        self._apply_clear()

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.program["u_light_dir"].value = (0.45, -0.60, 0.65)
        self.program["u_light_color"].value = (1.0, 0.97, 0.90)
        # 压低环境光：背光重叠面更暗，明暗条纹更清晰
        self.program["u_ambient"].value = 0.15
        self._build_meshes()

    def _apply_clear(self) -> None:
        if self.ctx is None:
            return
        if self.diag_red_bg:
            self.ctx.clear_color = (1.0, 0.0, 0.0, 1.0)   # 亮红（诊断）
        else:
            self.ctx.clear_color = (0.04, 0.04, 0.09, 1.0)  # 生产深蓝

    def _build_meshes(self) -> None:
        """用当前诊断开关构建地面/桥 VAO（白色或生产色，桥或全量）。"""
        data = self._rd_white if self.diag_white else self._rd
        if not self.bridge_only:
            gv = data["ground_vertices"].astype(np.float32)
            gi = data["ground_indices"].astype(np.uint32)
            if len(gv):
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

    def _rebuild(self) -> None:
        """切换模式后重建网格（释放旧 VAO）。"""
        if self.ctx is None or self.program is None:
            return
        for vao in (self.vao_ground, self.vao_bridge):
            if vao is not None:
                vao.release()
        self.vao_ground = None
        self.vao_bridge = None
        self._build_meshes()
        self.update()

    def paintGL(self) -> None:
        if self.ctx is None or self.program is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        self.ctx.viewport = (0, 0, w, h)
        # 生产同款：detect_framebuffer + fb.use()（AGENTS.md 3.6 关键修复）
        fb = self.ctx.detect_framebuffer()
        fb.use()
        # 直接传参 clear（不依赖 ctx.clear_color 状态）：亮红诊断 / 深蓝生产
        if self.diag_red_bg:
            self.ctx.clear(1.0, 0.0, 0.0, 1.0)       # 亮红 #FF0000
        else:
            self.ctx.clear(0.04, 0.04, 0.09, 1.0)    # 生产深蓝
        mvp = self.camera.mvp_matrix(w / h)
        self.program["mvp"].write(mvp.to_bytes())
        if self.vao_ground is not None:
            self.vao_ground.render(moderngl.TRIANGLES)
        if self.vao_bridge is not None:
            self.vao_bridge.render(moderngl.TRIANGLES)

    def resizeGL(self, width: int, height: int) -> None:
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, width, height)

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
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
        self.camera.orbit(-dx, dy)
        self._last_pos = (p.x(), p.y())
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.camera.zoom(0.9 if event.angleDelta().y() > 0 else 1.1)
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_1:      # 只渲染桥
            self.bridge_only = True
            self._rebuild()
            self._print_mode()
        elif key == Qt.Key.Key_2:    # 地面+桥
            self.bridge_only = False
            self._rebuild()
            self._print_mode()
        elif key == Qt.Key.Key_3:    # 生产颜色
            self.diag_white = False
            self._rebuild()
            self._print_mode()
        elif key == Qt.Key.Key_4:    # 纯白诊断色
            self.diag_white = True
            self._rebuild()
            self._print_mode()
        elif key == Qt.Key.Key_R:
            self.diag_red_bg = not self.diag_red_bg
            self._apply_clear()
            self._print_mode()
        elif key == Qt.Key.Key_F:    # 打印相机参数
            self._print_cam()
        elif key == Qt.Key.Key_Escape:
            QApplication.instance().quit()
        else:
            super().keyPressEvent(event)

    def _print_cam(self) -> None:
        cam = self.camera
        e = cam.eye()
        print(
            f"CAM distance={cam.distance:.1f} yaw={cam.yaw:.1f} pitch={cam.pitch:.1f} "
            f"target=({cam.target.x:.1f},{cam.target.y:.1f},{cam.target.z:.1f}) "
            f"eye=({e.x:.1f},{e.y:.1f},{e.z:.1f})"
        )

    def _print_mode(self) -> None:
        print(
            f"MODE bridge_only={self.bridge_only} white={self.diag_white} "
            f"red_bg={self.diag_red_bg}"
        )


class ViewerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Z-fighting 诊断查看器 (1/2=桥或全量, 3/4=颜色, R=红背景, F=相机, Esc=退出)")
        self.setGeometry(100, 100, 800, 600)
        self.gl = DiagViewer(TerrainFactory.default())
        self.setCentralWidget(self.gl)
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._update_cam_info)
        self._cam_timer.start(100)

    def _update_cam_info(self) -> None:
        cam = self.gl.camera
        t = cam.target
        e = cam.eye()
        self.statusBar().showMessage(
            f"bridge_only={int(self.gl.bridge_only)} white={int(self.gl.diag_white)} "
            f"red_bg={int(self.gl.diag_red_bg)}  |  "
            f"distance={cam.distance:.1f} yaw={cam.yaw:.1f} pitch={cam.pitch:.1f} "
            f"target=({t.x:.1f},{t.y:.1f},{t.z:.1f}) "
            f"eye=({e.x:.1f},{e.y:.1f},{e.z:.1f})"
        )


def main() -> int:
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    # 2026-08-11：请求 32 位深度缓冲
    fmt.setDepthBufferSize(32)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = ViewerWindow()
    window.show()
    print(
        "按键：1=只桥 2=全量 3=生产色 4=白色 R=红背景切换 F=打印相机参数 Esc=退出"
    )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
