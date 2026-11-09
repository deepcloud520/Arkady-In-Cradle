"""Airia.ui：Qt 交互上层（AGENTS.md 4.1）。

承载 render（GLWidget）+ 叙事/信息面板，组装成完整主窗口。
依赖方向：ui → render / narrative（不反向）。

2026-11-03：布局由「QSplitter 分栏 + 4 Tab」改为「地图铺满 + C4I 浮层」。

模块：
    metrics.py          ：响应式尺寸唯一来源（窗口可缩放）
    palette.py          ：C4I 调色板（转发 render.palette，避免 render→ui 反向依赖）
    dock.py             ：可停靠面板（标题栏 + 折叠/关闭 + 拖动）
    cockpit.py          ：驾驶舱容器（地图铺满 + 面板定位）
    timebar.py          ：顶部时间栏（QLabel/QFrame 组合，非自绘）
    message_panel.py    ：MESSAGE LOG 面板（可折叠/拖动/关闭，折叠态显示未读数）
    comms_popup.py      ：左侧实时通话弹窗（头像 / Sound only + STT 逐字，可拖动）
    unit_thumb.py       ：装备缩略图（QLabel + 无版权图片 + MANIFEST + 回退）
    windows.py          ：中央弹出窗口（战斗简报 / 战斗档案 / 消息日志备用窗）
    _ticker.py          ：已废弃（被 message_panel 取代）
    detail_panel.py     ：单位详情（三级信息详细度）
    narrative_overlay.py：旧对话面板（已由 comms_popup 取代，保留兼容）
    archive_panel.py    ：档案内容（由 windows.ArchiveWindow 承载）
    briefing_panel.py   ：简报内容（由 windows.BriefingWindow 承载）

注意：本包 **惰性导出**（PEP 562）。``import Airia.ui`` 不会连带导入
``main_window``——否则 render 层导入 ``ui.palette`` 时会触发循环导入。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅类型检查期，运行时不导入
    from Airia.ui.archive_panel import ArchivePanel
    from Airia.ui.briefing_panel import BriefingPanel
    from Airia.ui.comms_popup import CommsPopup, TypewriterModel
    from Airia.ui.detail_panel import DetailPanel
    from Airia.ui.main_window import MainWindow
    from Airia.ui.narrative_overlay import NarrativeOverlay

_EXPORTS = {
    "ArchivePanel": "Airia.ui.archive_panel",
    "BriefingPanel": "Airia.ui.briefing_panel",
    "CommsPopup": "Airia.ui.comms_popup",
    "TypewriterModel": "Airia.ui.comms_popup",
    "DetailPanel": "Airia.ui.detail_panel",
    "MainWindow": "Airia.ui.main_window",
    "NarrativeOverlay": "Airia.ui.narrative_overlay",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value          # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
