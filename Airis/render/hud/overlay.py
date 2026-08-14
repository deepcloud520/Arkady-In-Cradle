"""基于 moderngl 的 HUD 叠加层（Phase 5c 新增）。

对应 docs/phase5-ui-pipeline-design.md §4.3：
    - 世界空间元素（单位状态标记 / 选中高亮 / 命中标记）在 3D 透视之后，
      用正交（NDC）坐标绘制固定像素大小的 2D 标记
    - 数据来源只读 WorldAdapter（渲染层永不直读 Real SoA）

Phase 5c 范围：单位状态圆点 + 选中高亮环。文字标签 / 弹道线 / 命中标记
留待后续迭代（文字需字体纹理，命中标记需事件源）。
"""
from __future__ import annotations

import logging

import moderngl
import numpy as np

logger = logging.getLogger(__name__)

# 最大实例数（与 ArmorPool 容量对齐，缓冲预分配）
_MAX_INSTANCES = 2048

# 阵营圆点颜色（0=友军 / 1=敌军 / 2=中立）
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
out vec4 frag_color;
void main() {
    float r = length(v_quad);
    float alpha = 1.0 - smoothstep(0.78, 1.0, r);
    if (alpha <= 0.001) {
        discard;
    }
    frag_color = vec4(v_color, alpha);
}
"""


class HUDOverlay:
    """屏幕空间圆形标记渲染器（单位状态标记 + 选中高亮）。"""

    def __init__(self, ctx: moderngl.Context) -> None:
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

    def render(
        self,
        screen: np.ndarray,
        colors: np.ndarray,
        viewport_w: int,
        viewport_h: int,
        pixel_radius: float = 6.0,
    ) -> None:
        """绘制 N 个圆形标记。

        Args:
            screen: (N,2) float32 屏幕像素坐标（左上角原点，y 向下）。
            colors: (N,3) float32 RGB 颜色（0~1）。
            viewport_w / viewport_h: 视口像素尺寸。
            pixel_radius: 标记像素半径（近似，按视口高度换算 NDC 半径）。
        """
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
        self.vao.render(moderngl.TRIANGLES, instances=n)

    @staticmethod
    def side_color(side: int) -> tuple[float, float, float]:
        return _SIDE_COLOR.get(int(side), _SIDE_COLOR[2])

    @staticmethod
    def highlight_color() -> tuple[float, float, float]:
        return _HIGHLIGHT_COLOR
