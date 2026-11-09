"""C4I 界面调色板（GUI 外观兼容层）。

真正的实现在 ``Airia/render/palette.py``——之所以下沉到 render 层，是因为
``render/hud/log_widget.py``（消息日志）也需要语义色，而依赖方向规定
``ui → render``，render 不能反向 import ui（否则形成循环导入）。

本模块只做**转发**，保持 ``from Airia.ui import palette as pal`` 的既有写法可用。
"""
from __future__ import annotations

from Airia.render.palette import (  # noqa: F401
    DEFAULT_THEME,
    DARK,
    LIGHT,
    PALETTES,
    Palette,
    current,
    get,
    log_kind_color,
    panel_header_qss,
    qss,
    rgba,
    set_current,
    side_color,
)

__all__ = [
    "DARK",
    "DEFAULT_THEME",
    "LIGHT",
    "PALETTES",
    "Palette",
    "current",
    "get",
    "log_kind_color",
    "qss",
    "rgba",
    "set_current",
    "side_color",
]
