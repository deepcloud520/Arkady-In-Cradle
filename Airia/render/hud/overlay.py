"""基于 moderngl 的 HUD 叠加层（Phase 5c 新增，5e 扩展）。

对应 docs/phase5-ui-pipeline-design.md §4.3：
    - 世界空间元素（单位状态标记 / 选中高亮 / 命中标记）在 3D 透视之后，
      用正交（NDC）坐标绘制固定像素大小的 2D 标记
    - 数据来源只读 WorldAdapter（渲染层永不直读 Real SoA）

Phase 5c：单位状态圆点 + 选中高亮环（实心大圆）。
Phase 5e 扩展：
    - 圆环模式（u_mode=1）：命中标记 / 区域警示（软边圆环）
    - 线段渲染（render_lines）：弹道线 / 命中标记十字
    - 文字渲染（TextRenderer，见 hud/text.py）：单位文字标签 / 损坏 ⚠ 摘要
"""
from __future__ import annotations

import logging

import moderngl
import numpy as np

from Airia.render.hud.text import FontAtlas, TextRenderer

logger = logging.getLogger(__name__)

# 最大实例数（与 ArmorPool 容量对齐，缓冲预分配）
_MAX_INSTANCES = 2048

# 阵营圆点颜色（绝对 side：0=北约 1=华约 2=中立）
_SIDE_COLOR = {
    0: (0.30, 0.85, 0.45),
    1: (0.92, 0.20, 0.16),
    2: (0.65, 0.65, 0.65),
}
# 选中高亮色（黄色描边）
_HIGHLIGHT_COLOR = (1.0, 0.85, 0.30)

_HUD_VERT = """
#version 330 core
in vec2 in_quad;      // 单位方形角点（[-1,1]，圆心在原点）
in vec2 in_center;    // 实例：圆心 NDC
in float in_radius;   // 实例：NDC 半径
in vec3 in_color;     // 实例：颜色
out vec2 v_quad;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_center + in_quad * in_radius, 0.0, 1.0);
    v_quad = in_quad;
    v_color = in_color;
}
"""

_HUD_FRAG = """
#version 330 core
in vec2 v_quad;
in vec3 v_color;
uniform int u_mode;   // 0=实心圆（默认） / 1=圆环（软边带）
out vec4 frag_color;
void main() {
    float r = length(v_quad);
    float alpha;
    if (u_mode == 1) {
        // 圆环：r∈[0.62, 1.1] 软边带（命中标记/警示环）
        alpha = smoothstep(0.55, 0.72, r) * (1.0 - smoothstep(0.92, 1.08, r));
    } else {
        alpha = 1.0 - smoothstep(0.78, 1.0, r);
    }
    if (alpha <= 0.001) {
        discard;
    }
    frag_color = vec4(v_color, alpha);
}
"""

# 线段渲染（弹道线 / 命中十字）：实例化 quad，按端点方向拉伸 + 法线宽度
_LINE_VERT = """
#version 330 core
in vec2 in_quad;      // 单位方形角点（[-1,1]）
in vec2 in_p1;        // 实例：端点 1 NDC
in vec2 in_p2;        // 实例：端点 2 NDC
in float in_half_w;   // 实例：NDC 半宽
in vec4 in_color;     // 实例：RGBA
out vec4 v_color;
void main() {
    vec2 d = in_p2 - in_p1;
    float len = max(length(d), 1e-6);
    vec2 n = vec2(-d.y, d.x) / len;
    vec2 along = mix(in_p1, in_p2, in_quad.x * 0.5 + 0.5);
    gl_Position = vec4(along + n * in_quad.y * in_half_w, 0.0, 1.0);
    v_color = in_color;
}
"""

_LINE_FRAG = """
#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
"""

# 线段最大实例数
_MAX_LINES = 1024


class HUDOverlay:
    """屏幕空间标记渲染器（单位状态圆点 + 圆环 + 线段 + 文字，Phase 5c/5e）。"""

    def __init__(
        self,
        ctx: moderngl.Context,
        atlas: FontAtlas | None = None,
    ) -> None:
        """初始化。

        Args:
            ctx: OpenGL 上下文。
            atlas: 可选字体图集（提供且可用时启用文字渲染，Phase 5e）。
        """
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=_HUD_VERT,
            fragment_shader=_HUD_FRAG,
        )
        quad = np.array(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float32,
        )
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._quad_ibo = ctx.buffer(idx.tobytes())
        # 实例 buffer：每实例 (cx, cy, radius, r, g, b) = 6 float
        # 关键：format 必须加 `/i`（divisor=1），否则 moderngl 当作 per-vertex
        self._inst_vbo = ctx.buffer(b"\x00" * (6 * 4 * _MAX_INSTANCES))
        self.vao = ctx.vertex_array(
            self.program,
            [
                (self._quad_vbo, "2f", "in_quad"),
                (self._inst_vbo, "2f 1f 3f /i", "in_center", "in_radius", "in_color"),
            ],
            index_buffer=self._quad_ibo,
        )

        # Phase 5e：线段渲染（弹道线 / 命中十字）
        self._line_program = ctx.program(
            vertex_shader=_LINE_VERT,
            fragment_shader=_LINE_FRAG,
        )
        self._line_inst_vbo = ctx.buffer(b"\x00" * (9 * 4 * _MAX_LINES))
        self.line_vao = ctx.vertex_array(
            self._line_program,
            [
                (self._quad_vbo, "2f", "in_quad"),
                (self._line_inst_vbo, "2f 2f 1f 4f /i",
                 "in_p1", "in_p2", "in_half_w", "in_color"),
            ],
            index_buffer=self._quad_ibo,
        )

        # Phase 5e：文字渲染（图集可用时启用）
        self.text: TextRenderer | None = None
        if atlas is not None and atlas.available:
            self.text = TextRenderer(ctx, atlas)

    def render(
        self,
        screen: np.ndarray,
        colors: np.ndarray,
        viewport_w: int,
        viewport_h: int,
        pixel_radius: float = 6.0,
    ) -> None:
        """绘制 N 个实心圆形标记。

        Args:
            screen: (N,2) float32 屏幕像素坐标（左上角原点，y 向下）。
            colors: (N,3) float32 RGB 颜色（0~1）。
            viewport_w / viewport_h: 视口像素尺寸。
            pixel_radius: 标记像素半径（近似，按视口高度换算 NDC 半径）。
        """
        self._render_marks(
            screen, colors, viewport_w, viewport_h, pixel_radius, mode=0
        )

    def render_ring(
        self,
        screen: np.ndarray,
        colors: np.ndarray,
        viewport_w: int,
        viewport_h: int,
        pixel_radius: float = 20.0,
    ) -> None:
        """绘制 N 个软边圆环（命中标记 / 警示环，Phase 5e）。"""
        self._render_marks(
            screen, colors, viewport_w, viewport_h, pixel_radius, mode=1
        )

    def _render_marks(
        self,
        screen: np.ndarray,
        colors: np.ndarray,
        viewport_w: int,
        viewport_h: int,
        pixel_radius: float,
        mode: int,
    ) -> None:
        n = int(screen.shape[0])
        if n == 0:
            return
        if n > _MAX_INSTANCES:
            logger.warning("HUD 实例数超容量: %d > %d", n, _MAX_INSTANCES)
            n = _MAX_INSTANCES

        vw = max(1, int(viewport_w))
        vh = max(1, int(viewport_h))
        # 屏幕像素 → NDC（圆心）
        ndc_x = (screen[:n, 0].astype(np.float32) * 2.0 / vw) - 1.0
        ndc_y = 1.0 - (screen[:n, 1].astype(np.float32) * 2.0 / vh)
        # 像素半径 → NDC 半径（以视口高度为基准，近似固定像素大小）
        radius_ndc = float(pixel_radius) * 2.0 / vh

        inst = np.empty((n, 6), dtype=np.float32)
        inst[:, 0] = ndc_x
        inst[:, 1] = ndc_y
        inst[:, 2] = radius_ndc
        inst[:, 3:6] = colors[:n].astype(np.float32)
        self._inst_vbo.write(inst.tobytes())
        self.program["u_mode"].value = mode
        self.vao.render(moderngl.TRIANGLES, instances=n)

    def render_lines(
        self,
        segments: np.ndarray,
        colors: np.ndarray,
        viewport_w: int,
        viewport_h: int,
        width_px: float = 2.0,
    ) -> None:
        """绘制 N 条线段（Phase 5e：弹道线 / 命中十字）。

        Args:
            segments: (N,4) float32 屏幕像素端点 [x1,y1,x2,y2]。
            colors: (N,4) float32 RGBA 颜色（0~1）。
            viewport_w / viewport_h: 视口像素尺寸。
            width_px: 线段像素宽（近似，按视口高度换算 NDC）。
        """
        n = int(segments.shape[0])
        if n == 0:
            return
        if n > _MAX_LINES:
            logger.warning("线段实例数超容量: %d > %d", n, _MAX_LINES)
            n = _MAX_LINES
        vw = max(1, int(viewport_w))
        vh = max(1, int(viewport_h))

        s = segments[:n].astype(np.float32)
        inst = np.empty((n, 9), dtype=np.float32)
        inst[:, 0] = s[:, 0] * 2.0 / vw - 1.0
        inst[:, 1] = 1.0 - s[:, 1] * 2.0 / vh
        inst[:, 2] = s[:, 2] * 2.0 / vw - 1.0
        inst[:, 3] = 1.0 - s[:, 3] * 2.0 / vh
        inst[:, 4] = float(width_px) / vh       # 半宽（NDC y 单位）
        inst[:, 5:9] = colors[:n].astype(np.float32)
        self._line_inst_vbo.write(inst.tobytes())
        self.line_vao.render(moderngl.TRIANGLES, instances=n)

    @staticmethod
    def side_color(side: int) -> tuple[float, float, float]:
        return _SIDE_COLOR.get(int(side), _SIDE_COLOR[2])

    @staticmethod
    def highlight_color() -> tuple[float, float, float]:
        return _HIGHLIGHT_COLOR
