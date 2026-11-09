"""Z-fighting 诊断：亮红背景 + 纯白网格 + 多角度抓帧 + 帧间抖动检测。

用法: uv run python tests/diag_zfight_red.py

诊断逻辑（对齐用户停损指令）：
- 背景 Clear Color = 亮红 #FF0000，地形/桥材质强制纯白 → 隔离网格自身重叠
- 若条纹区域「透出红色」→ 网格裂缝（T-Junction，背景漏出）
- 若条纹区域为「白/灰明暗交替」→ Z 值冲突（两个近共面片深度竞争）
- 同一角度连抓 3 帧，检测帧间抖动像素（Z-fighting 的抖动特征）

产物统一写入 tests/output/（AGENTS.md 4.5）。
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

import moderngl
import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

from Airia.lulilo.terrain.map import TerrainMap
from Airia.render.camera import OrbitCamera
from _output import clean_output_dir, output_path
from tests.terrain_factory import TerrainFactory

# 与生产 gl_widget.py 相同的着色器（隔离网格几何，不引入新变量）
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
    if (n.z < 0.0) {
        n = -n;
    }
    float diff = max(dot(n, normalize(u_light_dir)), 0.0);
    vec3 color = v_color * (u_ambient + (1.0 - u_ambient) * diff * u_light_color);
    frag_color = vec4(color, 1.0);
}
"""

# 第一座演示桥（main.py 默认对准）：栅格 x∈[100,160] y∈[100,104]，世界 x∈[200,320] y∈[200,208]
BRIDGE_CENTER = (260.0, 204.0, 0.0)


class DiagWidget(QOpenGLWidget):
    """诊断渲染控件：红背景 + 白色网格，可只渲染桥（排除地面干扰）。"""

    def __init__(self, terrain: TerrainMap, bridge_only: bool = False) -> None:
        super().__init__()
        self.terrain = terrain
        self.bridge_only = bridge_only
        self.ctx = None
        self.program = None
        self.vao_ground = None
        self.vao_bridge = None
        self.camera = OrbitCamera(
            target=BRIDGE_CENTER, distance=80.0, yaw=0.0, pitch=10.0
        )
        self.resize(800, 600)

    def initializeGL(self) -> None:
        # 与生产一致：OpenGL 4.6 Core + 深度测试
        self.ctx = moderngl.create_context(require=460)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # 诊断：亮红背景（#FF0000）
        self.ctx.clear_color = (1.0, 0.0, 0.0, 1.0)

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.program["u_light_dir"].value = (0.45, -0.60, 0.65)
        self.program["u_light_color"].value = (1.0, 0.97, 0.90)
        # 压低环境光：让「背光面 vs 朝光面」的明暗对比更强，条纹更清晰
        self.program["u_ambient"].value = 0.15
        self._build_meshes()

    def _build_meshes(self) -> None:
        data = self.terrain.to_render_data()
        # 诊断：所有顶点颜色强制纯白（裂缝→红背景透出；重叠→明暗条纹）
        for key in ("ground_vertices", "bridge_vertices"):
            if len(data[key]):
                data[key][:, 3:6] = 1.0

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

    def set_cam(self, yaw: float, pitch: float, distance: float) -> None:
        self.camera.yaw = yaw
        self.camera.pitch = pitch
        self.camera.distance = distance
        self.camera.target = BRIDGE_CENTER

    def paintGL(self) -> None:
        if self.ctx is None or self.program is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        self.ctx.viewport = (0, 0, w, h)
        # 生产同款：detect_framebuffer + fb.use()（AGENTS.md 3.6 关键修复）
        fb = self.ctx.detect_framebuffer()
        fb.use()
        self.ctx.clear()
        mvp = self.camera.mvp_matrix(w / h)
        self.program["mvp"].write(mvp.to_bytes())
        if self.vao_ground is not None:
            self.vao_ground.render(moderngl.TRIANGLES)
        if self.vao_bridge is not None:
            self.vao_bridge.render(moderngl.TRIANGLES)


def _frame_stats(arr: np.ndarray, tag: str) -> None:
    """数值统计：红（裂缝）/ 亮（正常面）/ 暗（背光重叠面）像素占比。"""
    h, w = arr.shape[:2]
    n = h * w
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    red = (r > 200) & (g < 80) & (b < 80)
    bright = arr.mean(axis=2) > 200
    dark = arr.mean(axis=2) < 120
    print(
        f"[{tag}] 红={red.sum() / n * 100:5.2f}%  "
        f"亮白={bright.sum() / n * 100:5.2f}%  暗={dark.sum() / n * 100:5.2f}%  "
        f"(红=裂缝/背景漏出, 暗=背光重叠面)"
    )


def _jitter_report(frames: list[np.ndarray], tag: str) -> None:
    """相邻帧差：检测 Z-fighting 抖动像素及其颜色构成。"""
    if len(frames) < 2:
        return
    diffs = []
    for a, b in zip(frames, frames[1:]):
        d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
        diffs.append(d)
    jitter = np.maximum.reduce(diffs) > 10
    n = jitter.sum()
    if n == 0:
        print(f"[{tag}] 帧间抖动像素 = 0（该角度深度稳定）")
        return
    jr = frames[1][jitter]
    red_part = ((jr[:, 0] > 200) & (jr[:, 1] < 80) & (jr[:, 2] < 80)).mean() * 100
    dark_part = (jr.mean(axis=1) < 120).mean() * 100
    print(
        f"[{tag}] 抖动像素 = {n} 个；其中 红(裂缝)={red_part:.1f}%  暗(背光面)={dark_part:.1f}%  "
        f"→ {'裂缝' if red_part > 30 else 'Z值冲突(近共面重叠)'}"
    )


# 角度: (标签, yaw, pitch, distance) —— 相机对准桥1 (260,204,0)
ANGLES = [
    ("short_side", 0.0, 10.0, 60.0),     # 短边正视（竖直条纹主视角）
    ("iso_45", 45.0, 30.0, 200.0),       # 斜上方（main.py 默认视角）
    ("long_side", -90.0, 10.0, 60.0),    # 长边正视
    ("far_top", 45.0, 60.0, 400.0),      # 高空远景
]


def run_mode(app: QApplication, mode: str, bridge_only: bool) -> None:
    """跑完一个模式的全部角度（内部驱动事件循环，完成后返回）。"""
    w = DiagWidget(TerrainFactory.default(), bridge_only=bridge_only)
    w.show()
    w.set_cam(*ANGLES[0][1:])

    state = {"ai": 0, "frame": 0, "caps": []}

    def on_swap() -> None:
        st = state
        if st["ai"] >= len(ANGLES):
            app.quit()
            return
        st["frame"] += 1
        if st["frame"] < 6:  # 前 6 帧让管线稳定
            return
        if st["frame"] == 6:
            st["caps"] = []
        img = w.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
        arr = np.frombuffer(img.bits(), dtype=np.uint8).reshape(
            img.height(), img.width(), 3
        ).copy()
        st["caps"].append(arr)
        if st["frame"] >= 8:  # 6 稳定 + 3 抓帧
            tag = ANGLES[st["ai"]][0]
            name = f"diag_{mode}_{tag}"
            Image.fromarray(st["caps"][0]).save(str(output_path(name + "_f0.png")))
            _frame_stats(st["caps"][0], name)
            _jitter_report(st["caps"], name)
            st["ai"] += 1
            st["frame"] = 0
            if st["ai"] < len(ANGLES):
                w.set_cam(*ANGLES[st["ai"]][1:])
                w.update()
            else:
                app.quit()

    w.frameSwapped.connect(on_swap)
    QTimer.singleShot(8000, app.quit)  # 保险丝：单模式最多 8s
    app.exec()


def main() -> int:
    clean_output_dir()
    app = QApplication(sys.argv)

    # 模式：full=地面+桥全白（完整场景），bridge_only=只桥（排除地面）
    for mode, bridge_only in (("full", False), ("bridge_only", True)):
        print(f"===== 模式 {mode}（bridge_only={bridge_only}） =====")
        run_mode(app, mode, bridge_only)

    print("DIAG_DONE -> tests/output/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
