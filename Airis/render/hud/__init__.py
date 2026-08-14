"""render.hud 子项：基于 moderngl 的 HUD / UI 覆盖层（官方实现）。

战场界面上叠加的 UI 由两部分组成（Phase 5c）：
    - HUDOverlay：moderngl 绘制的世界空间标记（单位状态圆点 / 选中高亮）
    - LogWidget ：Qt 透明滚动日志（屏幕空间文本，叠加在 GLWidget 上）
"""
from Airis.render.hud.log_widget import LogWidget
from Airis.render.hud.overlay import HUDOverlay

__all__ = [
    "HUDOverlay",
    "LogWidget",
]
