"""QOpenGLWidget 渲染视图：2.5D 战场（Phase 2 地形 + Phase 5 管线骨架）。

- 使用 ctx.detect_framebuffer() 绑定 Qt 管理的 FBO（AGENTS.md 3.6 关键修复）
- 轨道相机交互：左键拖拽旋转、滚轮缩放
- Phase 5：
    * 着色器外提：内联 VERTEX/FRAGMENT_SHADER → assets/shaders/<theme>/
    * 主题系统：ThemeManager 加载/热切换（默认 default 主题）
    * 渲染数据源：通过 WorldAdapter 读取单位数据（渲染层永不直读 Real SoA）
"""
from __future__ import annotations

import logging
import os

import moderngl
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from Airis.lulilo.terrain.map import TerrainMap
from Airis.render.adapter import WorldAdapter
from Airis.render.camera import OrbitCamera
from Airis.render.hud.overlay import HUDOverlay
from Airis.render.picking import Picker, project_world
from Airis.render.theme import Theme, ThemeManager

logger = logging.getLogger(__name__)


def _default_terrain() -> TerrainMap:
    """默认演示地形：fBm 起伏 + 峡谷长桥 + 2 座普通桥（WorldMesh Phase B/D 展示）。

    可见效果：
    - 平滑地面（continuous_ground 去量化，无 1m 台阶）
    - **峡谷长桥**：山体被切开（峡谷）+ 桥墩支撑（现实山间桥梁）
    - 普通桥的完整 3D（顶面 + 侧面 + 底面）
    尺寸 256×256 控制 DualContour 构建在演示可接受范围。
    """
    from Airis.lulilo.terrain.factory import (
        CanyonBridgeConfig,
        TerrainFactory,
        TerrainFactoryConfig,
    )

    cfg = TerrainFactoryConfig(
        width=256,
        height=256,
        cell_size=2.0,
        seed=42,
        amplitude=10.0,
        noise_scale=18.0,
        octaves=4,
        bridge_lift=2.0,
        bridges=[
            {"x0": 60, "y0": 60, "x1": 120, "y1": 63},    # 普通横桥
            {"x0": 200, "y0": 60, "x1": 203, "y1": 120},  # 普通纵桥
        ],
        canyon_bridge=CanyonBridgeConfig(
            x0=40, y0=128, x1=215, y1=131,   # 长桥 176 格（352m），横跨山体
            canyon_frac=0.3,                 # 中间 60% 挖成峡谷
            canyon_depth=14.0,               # 峡谷挖深 14m
            pier_interval=16,                # 桥墩间隔 16 格（32m）
            pier_width=2,                    # 桥墩截面 2×2 格（4×4m）
        ),
    )
    return TerrainFactory.fbm(cfg)

# 单位标记着色器已外提到 HUDOverlay（Phase 5c，屏幕空间正交绘制，固定像素圆点）。
# 旧内联 3D 实例化 quad 标记（Phase 5a 骨架）已移除，见 hud/overlay.py。


class GLWidget(QOpenGLWidget):
    # 拾取信号（Phase 5c）：索引为装甲池索引，-1 = 空白/取消
    unit_hovered = Signal(int)
    unit_selected = Signal(int)

    def __init__(
        self,
        terrain: TerrainMap | None = None,
        theme_manager: ThemeManager | None = None,
    ):
        """初始化。

        Args:
            terrain: 地形（默认演示地形）。
            theme_manager: 外部注入的主题管理器（可选）。未注入时在
                initializeGL 内自建（需 OpenGL 上下文就绪后创建）。
        """
        super().__init__()
        self.terrain = terrain if terrain is not None else _default_terrain()
        hf = self.terrain.heightfield
        self.camera = OrbitCamera(
            target=(hf.width * hf.cs * 0.5, hf.height * hf.cs * 0.5, 0.0),
            distance=hf.width * hf.cs * 1.2,
        )
        self._last_pos: tuple[float, float] | None = None
        self._press_pos: tuple[float, float] | None = None
        self._dragging = False
        self._hover_idx = -1
        self._selected_idx = -1
        self._picker = Picker(max_px=30.0)
        self._drag_threshold = 4.0

        self.theme_manager: ThemeManager | None = theme_manager
        self._adapter: WorldAdapter | None = None
        self._mvp_bytes: bytes | None = None
        self._mvp = None                     # glm.mat4（拾取复用最近一帧 MVP）

        self.ctx = None
        self.program = None
        self.hud: HUDOverlay | None = None
        self.vao_ground = None
        self.vao_bridge = None
        # 缓存 VBO/IBO 引用（主题切换重建 VAO 时复用 buffer）
        self._ground_vbo = None
        self._ground_ibo = None
        self._bridge_vbo = None
        self._bridge_ibo = None

        # 叠加在 GL 之上的 Qt 子 widget（如 LogWidget），resizeEvent 重新定位
        self.overlay_widgets: list = []

        # 无按键时也接收 mouseMove（hover 拾取）+ 键盘焦点（Esc 取消选择）
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    # ---- Phase 5：渲染数据源注入 ----

    def set_adapter(self, adapter: WorldAdapter | None) -> None:
        """注入 WorldAdapter（渲染数据源抽象）。None = 纯地形模式。"""
        self._adapter = adapter

    @property
    def adapter(self) -> WorldAdapter | None:
        return self._adapter

    def initializeGL(self):
        try:
            # require=460：显式请求 OpenGL 4.6 Core（配合 main.py 的 QSurfaceFormat）
            self.ctx = moderngl.create_context(require=460)
            info = self.ctx.info
            logger.info("OpenGL %s / %s / %s", info.get("GL_VERSION"), info.get("GL_VENDOR"), info.get("GL_RENDERER"))
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.clear_color = (0.04, 0.04, 0.09, 1.0)

            # 主题系统：自建（未注入时）→ 加载默认主题 → 注册热切换回调
            if self.theme_manager is None:
                self.theme_manager = ThemeManager(self.ctx)
            self.theme_manager.on_switch(self._on_theme_switched)
            if self.theme_manager.current is None:
                self.theme_manager.switch("default")
            self.program = self.theme_manager.current.program

            self._build_meshes()
            self.hud = HUDOverlay(self.ctx)
            self.theme_manager.apply_uniforms()
        except Exception:
            logger.exception("initializeGL 失败")

    def _on_theme_switched(self, theme: Theme) -> None:
        """主题热切换回调：重绑地形 VAO + 写入默认 uniforms。

        注意（2026-08-13 修复）：回调可能从菜单槽（GL context 非 current）
        触发，任何 moderngl 调用（重建 VAO / 写 uniform）都必须包在
        ``makeCurrent()`` 内，否则在错误/无上下文下创建的对象会导致
        地形渲染变黑（静默失败）。
        """
        self.program = theme.program
        if self._ground_vbo is not None:   # 首次切换时 VAO 尚未构建，跳过
            self.makeCurrent()
            try:
                self._rebuild_terrain_vaos()
                if self.theme_manager is not None:
                    self.theme_manager.apply_uniforms(theme)
            finally:
                self.doneCurrent()
        self.update()

    def _build_meshes(self):
        """创建地形网格 buffer 并绑定 VAO（buffer 缓存供主题切换重建）。"""
        data = self.terrain.to_render_data()
        gv = data["ground_vertices"].astype(np.float32)
        gi = data["ground_indices"].astype(np.uint32)
        self._ground_vbo = self.ctx.buffer(gv.tobytes())
        self._ground_ibo = self.ctx.buffer(gi.tobytes())

        bv = data["bridge_vertices"].astype(np.float32)
        bi = data["bridge_indices"].astype(np.uint32)
        if len(bv):
            self._bridge_vbo = self.ctx.buffer(bv.tobytes())
            self._bridge_ibo = self.ctx.buffer(bi.tobytes())
        else:
            self._bridge_vbo = None
            self._bridge_ibo = None

        self._rebuild_terrain_vaos()

    def _rebuild_terrain_vaos(self):
        """用当前 program 重建地形 VAO（主题切换后调用）。"""
        self.vao_ground = self.ctx.vertex_array(
            self.program,
            [(self._ground_vbo, "3f 3f 3f", "in_vert", "in_color", "in_normal")],
            index_buffer=self._ground_ibo,
        )
        if self._bridge_vbo is not None:
            self.vao_bridge = self.ctx.vertex_array(
                self.program,
                [(self._bridge_vbo, "3f 3f 3f", "in_vert", "in_color", "in_normal")],
                index_buffer=self._bridge_ibo,
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
            mvp_bytes = mvp.to_bytes()
            self.program["mvp"].write(mvp_bytes)
            self._mvp_bytes = mvp_bytes
            self._mvp = mvp

            self.vao_ground.render(moderngl.TRIANGLES)
            if self.vao_bridge is not None:
                self.vao_bridge.render(moderngl.TRIANGLES)

            # Phase 5c：HUD 单位标记（读 WorldAdapter，屏幕空间正交绘制）
            self._render_hud(w, h)

            if os.environ.get("AIRC_DUMP_FRAME") == "1":
                self._dump_frame()
        except Exception:
            logger.exception("paintGL 失败")

    def _render_hud(self, viewport_w: int, viewport_h: int) -> None:
        """Phase 5c：HUD 单位标记（屏幕空间正交绘制，读 WorldAdapter）。

        3D 透视 pass 之后调用：把单位世界坐标投影到屏幕像素 → HUDOverlay
        在 NDC 空间画固定像素圆点。选中单位额外叠加高亮环。

        HUD 语义 = 永远在最上层：绘制期间关闭深度测试（低角度观察时
        近处地形深度 < 标记深度，GL_LESS 会把标记剔除 → 圆点消失）。
        """
        if self.hud is None or self._adapter is None or self._mvp is None:
            return
        active = self._adapter.armor_active
        n_active = int(np.count_nonzero(active))
        if n_active == 0:
            return
        self.ctx.disable(moderngl.DEPTH_TEST)
        try:
            self._render_hud_inner(viewport_w, viewport_h, active)
        finally:
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_hud_inner(
        self, viewport_w: int, viewport_h: int, active: np.ndarray
    ) -> None:
        idx = np.flatnonzero(active)
        pos = self._adapter.armor_pos[idx].copy()
        pos[:, 2] += 0.5   # 标记抬升到单位上方，避免与地形/单位遮挡
        screen, _ndc_z, _clip_w, visible = project_world(
            pos, self._mvp, viewport_w, viewport_h
        )
        if int(np.count_nonzero(visible)) == 0:
            return

        side = self._adapter.armor_side[idx]
        colors = np.array(
            [HUDOverlay.side_color(int(k)) for k in side], dtype=np.float32
        )

        # 1) 选中高亮（大圆，黄色，先画在下层）
        if self._selected_idx >= 0:
            hit = np.where(idx == self._selected_idx)[0]
            if hit.size and visible[hit[0]]:
                j = int(hit[0])
                self.hud.render(
                    screen[j:j + 1],
                    np.array([HUDOverlay.highlight_color()], dtype=np.float32),
                    viewport_w, viewport_h, pixel_radius=13.0,
                )

        # 2) 状态圆点（阵营色）
        self.hud.render(
            screen[visible], colors[visible], viewport_w, viewport_h, pixel_radius=6.0
        )

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
        self._reposition_overlays(width, height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlays(self.width(), self.height())

    def _reposition_overlays(self, w: int, h: int) -> None:
        """重定位叠加在 GL 之上的 Qt 子 widget（如 LogWidget）。"""
        for ow in self.overlay_widgets:
            if hasattr(ow, "layout_overlay"):
                ow.layout_overlay(w, h)

    # ---- 拾取辅助 ----

    def _pick_at(self, x: float, y: float) -> int:
        """用最近一帧 MVP 拾取屏幕坐标附近的单位。"""
        if self._mvp is None:
            return -1
        return self._picker.pick(
            x, y, self._adapter, self._mvp, self.width(), self.height()
        )

    # ---- 相机交互 + 拾取 ----
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # 只保存坐标（浮点），不要保存事件对象：Qt 事件用完即销毁
            p = event.position()
            self._press_pos = (p.x(), p.y())
            self._last_pos = (p.x(), p.y())
            self._dragging = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        p = event.position()
        x, y = p.x(), p.y()
        if self._last_pos is not None:
            # 左键按住：位移超过阈值判为拖拽（旋转相机），否则等待判定单击
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            if abs(dx) + abs(dy) > self._drag_threshold:
                self._dragging = True
            if self._dragging:
                self.camera.orbit(-dx, dy)
                self._last_pos = (x, y)
                self.update()
        else:
            # 无按键移动：hover 检测
            hov = self._pick_at(x, y)
            if hov != self._hover_idx:
                self._hover_idx = hov
                self.unit_hovered.emit(hov)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            was_drag = self._dragging
            p = event.position()
            self._last_pos = None
            self._press_pos = None
            self._dragging = False
            if not was_drag:
                # 单击 → 选中（点空白则 sel=-1，取消选择）
                sel = self._pick_at(p.x(), p.y())
                self._selected_idx = sel
                self.unit_selected.emit(sel)
                self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            if self._selected_idx != -1:
                self._selected_idx = -1
                self.unit_selected.emit(-1)
                self.update()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.camera.zoom(0.9 if event.angleDelta().y() > 0 else 1.1)
        self.update()
