"""HUD 文字渲染：字体图集（FontAtlas）+ 屏幕空间文本绘制（Phase 5e 新增）。

对应 docs/phase5-ui-pipeline-design.md §4.3「HUD 文字标签」：
    - 文字需要字体纹理：PIL 把字符渲染进单张纹理图集（atlas）
    - TextRenderer 用实例化 quad 绘制：每字符一个实例
      （NDC 中心 / NDC 尺寸 / 归一化 UV / 颜色），屏幕空间正交绘制
    - 图集字符集 = ASCII 可打印 + 常用中文（部位名 / 阵营名 / 状态符号）

字体查找：Windows 优先（微软雅黑 / 黑体 / 宋体），跨平台回退
（Noto Sans CJK / DejaVu Sans / PingFang）。找不到字体时
``FontAtlas.available=False``，渲染层自动跳过文字（不崩溃）。

数据流：FontAtlas 是 CPU 侧纯数据（可无头单测）；TextRenderer 持有
GL 资源（program / 纹理 / VAO），只在 OpenGL 上下文中构造。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 默认附加字符：部位名 + 阵营名 + 状态符号（ASCII 可打印字符另行内置）
_DEFAULT_EXTRA_CHARS = "友敌军中立" "车炮塔引擎履带弹药架" "⚠"

# 候选字体路径（按优先级；Windows 自带 CJK 字体）
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑（CJK）
    "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑粗体
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    "C:/Windows/Fonts/arial.ttf",      # ASCII 回退
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]


def find_font_file() -> str | None:
    """返回第一个可用的字体文件路径（找不到返回 None）。"""
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


@dataclass(frozen=True)
class Glyph:
    """图集中的一个字符（纹理像素坐标 + 排版度量）。

    排版约定：文本行以「行顶左上角」为原点 (x, y)；
    字符墨水框左上角 = (x + pen + dx, y + dy)，宽高 (w, h)。
    """

    u0: float          # 纹理像素坐标（墨水框左上）
    v0: float
    u1: float          # 墨水框右下
    v1: float
    w: int             # 墨水框像素宽
    h: int             # 墨水框像素高
    dx: float          # 相对笔位置的 x 偏移（bearing）
    dy: float          # 相对行顶的 y 偏移
    advance: float     # 前进宽度（像素，font size 单位）


@dataclass(frozen=True)
class TextItem:
    """一条要绘制的文本（屏幕像素坐标，左上角原点，y 向下）。"""

    x: float
    y: float
    text: str
    color: tuple[float, float, float] = (0.90, 0.90, 0.93)
    size_px: float = 12.0


class FontAtlas:
    """PIL 渲染的字符图集（CPU 侧纯数据；纹理对象由 TextRenderer 在 GL 侧创建）。"""

    def __init__(
        self,
        font_path: str | None = None,
        size: int = 32,
        extra_chars: str = _DEFAULT_EXTRA_CHARS,
        cell_pad: int = 2,
    ) -> None:
        self.size = int(size)
        self.font_path = font_path or find_font_file()
        self.available = False
        self._glyphs: dict[str, Glyph] = {}
        self._atlas_w = 0
        self._atlas_h = 0
        self.image: Image.Image | None = None

        if self.font_path is None:
            logger.warning("未找到可用字体，HUD 文字标签将不可用")
            return
        try:
            self._font = ImageFont.truetype(self.font_path, self.size)
        except OSError as exc:  # noqa: BLE001
            logger.warning("字体加载失败 %s: %s", self.font_path, exc)
            return
        self.available = True
        self._cell_pad = int(cell_pad)
        self._build(extra_chars)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def atlas_width(self) -> int:
        return self._atlas_w

    @property
    def atlas_height(self) -> int:
        return self._atlas_h

    def has(self, ch: str) -> bool:
        return ch in self._glyphs

    def glyph(self, ch: str) -> Glyph | None:
        return self._glyphs.get(ch)

    def charset(self) -> list[str]:
        return list(self._glyphs)

    # ------------------------------------------------------------------
    # 图集构建
    # ------------------------------------------------------------------

    def _build(self, extra_chars: str) -> None:
        """渲染字符集 → 网格打包进图集（L 灰度，白字黑底）。"""
        chars: list[str] = []
        seen: set[str] = set()
        for code in range(32, 127):          # ASCII 可打印（含空格）
            ch = chr(code)
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
        for ch in extra_chars:               # 附加中文/符号（去重）
            if ch and ch not in seen:
                seen.add(ch)
                chars.append(ch)

        cell = self.size + 2 * self._cell_pad
        cols = max(1, int(math.ceil(math.sqrt(len(chars)))))
        rows = max(1, int(math.ceil(len(chars) / cols)))
        self._atlas_w = cols * cell
        self._atlas_h = rows * cell
        atlas = Image.new("L", (self._atlas_w, self._atlas_h), 0)
        draw = ImageDraw.Draw(atlas)

        for i, ch in enumerate(chars):
            col, row = i % cols, i // cols
            ox = col * cell + self._cell_pad
            oy = row * cell + self._cell_pad
            left, top, right, bottom = self._font.getbbox(ch)
            draw.text((ox, oy), ch, font=self._font, fill=255)
            advance = float(self._font.getlength(ch))
            # 墨水框（钳制到图集边界；空格等空白字符 w/h 可能为 0）
            u0 = max(0, ox + left)
            v0 = max(0, oy + top)
            u1 = min(self._atlas_w, ox + right)
            v1 = min(self._atlas_h, oy + bottom)
            self._glyphs[ch] = Glyph(
                u0=float(u0), v0=float(v0), u1=float(u1), v1=float(v1),
                w=max(0, u1 - u0), h=max(0, v1 - v0),
                dx=float(left), dy=float(top), advance=advance,
            )
        self.image = atlas
        logger.info(
            "字体图集就绪：%s（%d 字符，%dx%d，字号 %d）",
            Path(self.font_path).name, len(chars), self._atlas_w, self._atlas_h, self.size,
        )


# 最大字符实例数（缓冲预分配；16 单位 × 20 字 × 多标签远小于此）
_MAX_TEXT_INSTANCES = 4096

_TEXT_VERT = """
#version 330 core
in vec2 in_quad;      // 单位方形角点（[-1,1]）
in vec2 in_center;    // 实例：字符中心 NDC
in vec2 in_size;      // 实例：字符 NDC 尺寸
in vec4 in_uv;        // 实例：UV (u0,v0,u1,v1) 归一化
in vec3 in_color;     // 实例：颜色
out vec2 v_uv;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_center + in_quad * in_size, 0.0, 1.0);
    v_uv = mix(in_uv.xy, in_uv.zw, in_quad * 0.5 + 0.5);
    v_color = in_color;
}
"""

_TEXT_FRAG = """
#version 330 core
in vec2 v_uv;
in vec3 v_color;
uniform sampler2D u_atlas;
out vec4 frag_color;
void main() {
    float a = texture(u_atlas, v_uv).r;
    if (a < 0.02) {
        discard;
    }
    frag_color = vec4(v_color, a);
}
"""


class TextRenderer:
    """实例化字符 quad 绘制（Phase 5e；需 OpenGL 上下文）。"""

    def __init__(self, ctx: moderngl.Context, atlas: FontAtlas) -> None:
        if not atlas.available:
            raise ValueError("FontAtlas 不可用，无法创建 TextRenderer")
        self.ctx = ctx
        self.atlas = atlas
        self.program = ctx.program(vertex_shader=_TEXT_VERT, fragment_shader=_TEXT_FRAG)

        quad = np.array(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float32,
        )
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._quad_ibo = ctx.buffer(idx.tobytes())
        # 实例 buffer：每字符 (cx, cy, sx, sy, u0, v0, u1, v1, r, g, b) = 11 float
        self._inst_vbo = ctx.buffer(b"\x00" * (11 * 4 * _MAX_TEXT_INSTANCES))
        self.vao = ctx.vertex_array(
            self.program,
            [
                (self._quad_vbo, "2f", "in_quad"),
                (self._inst_vbo, "2f 2f 4f 3f /i",
                 "in_center", "in_size", "in_uv", "in_color"),
            ],
            index_buffer=self._quad_ibo,
        )

        # 图集纹理（L 灰度，红通道作 alpha）
        data = np.asarray(atlas.image, dtype=np.uint8)
        self.texture = ctx.texture(
            (atlas.atlas_width, atlas.atlas_height), 1, data.tobytes(), dtype="f1"
        )
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.use(0)
        self.program["u_atlas"].value = 0

    def render(
        self,
        items: list[TextItem],
        viewport_w: int,
        viewport_h: int,
    ) -> None:
        """绘制多条文本（屏幕像素坐标，左上角原点，y 向下）。

        Args:
            items: 文本条目列表。
            viewport_w / viewport_h: 视口像素尺寸。
        """
        inst = self._layout(items, int(viewport_w), int(viewport_h))
        n = inst.shape[0]
        if n == 0:
            return
        if n > _MAX_TEXT_INSTANCES:
            logger.warning("文字实例超容量: %d > %d", n, _MAX_TEXT_INSTANCES)
            inst = inst[:_MAX_TEXT_INSTANCES]
            n = _MAX_TEXT_INSTANCES
        self._inst_vbo.write(inst.tobytes())
        self.vao.render(moderngl.TRIANGLES, instances=n)

    # ------------------------------------------------------------------
    # 排版
    # ------------------------------------------------------------------

    def _layout(self, items: list[TextItem], vw: int, vh: int) -> np.ndarray:
        """文本条目 → 每字符实例数组 (N, 11)。"""
        aw, ah = self.atlas.atlas_width, self.atlas.atlas_height
        scale_base = float(self.atlas.size)
        rows: list[tuple[float, float, float, float,
                         float, float, float, float,
                         float, float, float]] = []
        for it in items:
            pen_x = float(it.x)
            top_y = float(it.y)
            scale = float(it.size_px) / scale_base
            for ch in it.text:
                g = self.atlas.glyph(ch)
                if g is None:
                    if ch == " ":
                        sp = self.atlas.glyph(" ")
                        pen_x += (sp.advance if sp else 0.0) * scale
                    continue
                gw = g.w * scale
                gh = g.h * scale
                if gw <= 0 or gh <= 0:
                    pen_x += g.advance * scale
                    continue
                gx = pen_x + g.dx * scale
                gy = top_y + g.dy * scale
                cx = gx + gw / 2.0
                cy = gy + gh / 2.0
                ndc_x = cx * 2.0 / vw - 1.0
                ndc_y = 1.0 - cy * 2.0 / vh
                sz_x = gw * 2.0 / vw
                sz_y = gh * 2.0 / vh
                u0 = g.u0 / aw
                v0 = g.v0 / ah
                u1 = g.u1 / aw
                v1 = g.v1 / ah
                r, gg, b = (float(c) for c in it.color)
                rows.append((ndc_x, ndc_y, sz_x, sz_y, u0, v0, u1, v1, r, gg, b))
                pen_x += g.advance * scale
        if not rows:
            return np.zeros((0, 11), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
