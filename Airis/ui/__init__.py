"""Airis.ui：Qt 交互上层（AGENTS.md 4.1）。

承载 render（GLWidget）+ 叙事面板（对话/档案/简报/详情），
组装成完整主窗口。依赖方向：ui → render / narrative（不反向）。

模块：
    main_window.py      ：MainWindow（QSplitter 布局 + 菜单 + 主循环）
    narrative_overlay.py：对话面板（SidePanel Tab1）
    archive_panel.py    ：档案库骨架（SidePanel Tab2）
    briefing_panel.py   ：战斗简报骨架（SidePanel Tab3）
    detail_panel.py     ：单位详情（SidePanel Tab4，三级信息详细度）
"""
from Airis.ui.archive_panel import ArchivePanel
from Airis.ui.briefing_panel import BriefingPanel
from Airis.ui.detail_panel import DetailPanel
from Airis.ui.main_window import MainWindow
from Airis.ui.narrative_overlay import NarrativeOverlay

__all__ = [
    "ArchivePanel",
    "BriefingPanel",
    "DetailPanel",
    "MainWindow",
    "NarrativeOverlay",
]
