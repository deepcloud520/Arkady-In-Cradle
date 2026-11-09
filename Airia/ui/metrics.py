"""响应式尺寸（UI metrics）：所有界面尺寸的唯一来源（2026-11-03 新增）。

背景：本作是**可缩放的窗口**（用户可拖边缘、最大化铺满），不是独占全屏。
因此界面不能写死像素——文档 §3.6 那批 ``200 / 54 / 96 / 96 / 22×14`` 全部是
**1440×900 基准分辨率**上的收敛值，必须按窗口尺寸连续换算。

缩放模型（三条轴，互不叠加）：

    s = clamp(viewport_h / 900, 0.80, 2.00)   ← 只按高度，超宽屏不撑大 UI
    k = clamp(s × 用户偏好, 0.75, 2.40)        ← 用户偏好 0.85 / 1.0 / 1.15 / 1.3
    侧栏 = min(200 × k, 0.22 × viewport_w)     ← 超宽屏收窄，把空间还给地图

⚠️ 只按**逻辑像素**计算：Qt6 的 High-DPI 缩放已由 ``devicePixelRatio`` 处理，
这里**绝不再乘 DPR**，否则 125% 屏会双重放大。

宽度断点（响应式）：

    ≥1600  full     侧栏 200k，时间栏 5 块全显
    1200-1599 compact  侧栏 200k，时间栏隐藏「剩余/高度」
    900-1199  tight    侧栏收窄到 0.22W，时间栏只留 ZULU
    <900     drawer    侧栏抽屉化（默认隐藏），时间栏只留大字时钟

高度断点：h < 700（如 1366×768）→ 时间栏 54→46、底栏 96→72。
"""
from __future__ import annotations

from dataclasses import dataclass

# 设计基准分辨率（原型验证分辨率，16:10）
BASE_W = 1440
BASE_H = 900

# 时间栏信息分块（顺序即显示顺序）
BLOCKS_ALL: tuple[str, ...] = ("地点", "日期", "ZULU", "剩余", "高度")

# 用户偏好可选档位
UI_SCALES: tuple[float, ...] = (0.85, 1.0, 1.15, 1.3)


@dataclass(frozen=True)
class Metrics:
    """一帧的界面尺寸快照（frozen：resize 时整体替换，不就地改）。"""

    width: int = BASE_W
    height: int = BASE_H
    k: float = 1.0
    mode: str = "full"           # full | compact | tight | drawer

    sidebar_w: int = 200
    timebar_h: int = 54
    bottombar_h: int = 96
    header_h: int = 18
    pad: int = 6
    row_h: int = 15

    thumb: int = 96
    popup_w: int = 356

    font_xs: float = 8.0
    font_sm: float = 10.0
    font_md: float = 12.0
    font_lg: float = 14.0
    font_clock: float = 26.0

    symbol_w: int = 22
    symbol_h: int = 14
    select_w: int = 34
    select_h: int = 28

    timebar_blocks: tuple[str, ...] = BLOCKS_ALL

    # ---- 便捷量 ----

    @property
    def sidebar_visible(self) -> bool:
        return self.sidebar_w > 0

    @property
    def map_w(self) -> int:
        return max(1, self.width - self.sidebar_w)

    @property
    def map_h(self) -> int:
        return max(1, self.height - self.bottombar_h)

    def px(self, value: float) -> int:
        """基准像素 → 当前窗口像素（至少 1）。"""
        return max(1, int(round(float(value) * self.k)))


def for_viewport(
    width: int,
    height: int,
    user_scale: float = 1.0,
) -> Metrics:
    """按窗口逻辑像素生成尺寸快照。

    Args:
        width / height: 窗口客户区逻辑像素。
        user_scale: 用户偏好乘子（见 ``UI_SCALES``）。

    Returns:
        Metrics（frozen）。
    """
    w = max(1, int(width))
    h = max(1, int(height))

    s = min(2.0, max(0.8, h / float(BASE_H)))
    k = min(2.4, max(0.75, s * float(user_scale)))

    short = h < 700
    timebar_h = int(round((46 if short else 54) * k))
    bottombar_h = int(round((72 if short else 96) * k))

    if w >= 1600:
        mode = "full"
        sidebar_w = int(round(200 * k))
        thumb = int(round(96 * k))
        blocks = BLOCKS_ALL
    elif w >= 1200:
        mode = "compact"
        sidebar_w = max(int(round(140 * k)), int(round(200 * k)))
        thumb = int(round(96 * k))
        blocks = ("地点", "日期", "ZULU")
    elif w >= 900:
        mode = "tight"
        sidebar_w = int(round(0.22 * w))
        thumb = int(round(72 * k))
        blocks = ("ZULU",)
    else:
        mode = "drawer"
        sidebar_w = 0
        thumb = int(round(64 * k))
        blocks = ()

    # 侧栏不得超过窗口的 22%（超宽屏/极窄窗都靠这条兜底）
    if sidebar_w > 0:
        sidebar_w = min(sidebar_w, int(round(0.22 * w)))

    return Metrics(
        width=w,
        height=h,
        k=k,
        mode=mode,
        sidebar_w=sidebar_w,
        timebar_h=timebar_h,
        bottombar_h=bottombar_h,
        header_h=int(round(18 * k)),
        pad=int(round(6 * k)),
        row_h=int(round(15 * k)),
        thumb=thumb,
        popup_w=int(round(min(356 * k, 0.30 * w))),
        font_xs=round(8.0 * k, 1),
        font_sm=round(10.0 * k, 1),
        font_md=round(12.0 * k, 1),
        font_lg=round(14.0 * k, 1),
        font_clock=round(26.0 * k, 1),
        symbol_w=int(round(22 * k)),
        symbol_h=int(round(14 * k)),
        select_w=int(round(34 * k)),
        select_h=int(round(28 * k)),
        timebar_blocks=blocks,
    )
