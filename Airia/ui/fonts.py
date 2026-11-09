"""界面字体加载（2026-11-03 新增）。

背景：``QT_QPA_PLATFORM=offscreen`` 下 Qt 找不到系统字体（``QFontDatabase``
报 "Cannot find font directory"），所有中文都会变成豆腐块——冒烟截图因此
全是方框。Windows 正常窗口模式能拿到系统字体，但显式加载更稳妥，也顺便
解决了字体回退链（``Consolas`` 无中文，数字/中文混排必须回退到雅黑）。

用法（QApplication 创建之后）::

    from Airia.ui.fonts import load_ui_fonts
    load_ui_fonts(app)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

# 候选字体文件（按优先级）：中文正文 → 中文粗体 → 等宽（数字/时间码）
_WINDOWS_FONTS = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
)

# 兜底：系统里可能已注册的家族名（不加载文件也能用）
_FALLBACK_FAMILIES = (
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Segoe UI",
    "DejaVu Sans",
)

APP_FONT_PX = 11


def load_ui_fonts(app: QApplication | None = None) -> list[str]:
    """加载界面字体并设置应用默认字体，返回已注册的家族名列表。"""
    families: list[str] = []

    for path in _WINDOWS_FONTS:
        if not os.path.exists(path):
            continue
        fid = QFontDatabase.addApplicationFont(str(path))
        if fid >= 0:
            families.extend(QFontDatabase.applicationFontFamilies(fid))

    if not families:
        available = set(QFontDatabase.families())
        families = [f for f in _FALLBACK_FAMILIES if f in available]

    if not families:
        logger.warning("未找到可用界面字体，中文可能显示为方块")
        return []

    # 优先中文字体（雅黑/思源），等宽只用于时间码
    cjk = next(
        (f for f in families if any(k in f for k in ("YaHei", "CJK", "Han", "PingFang"))),
        families[0],
    )
    if app is not None:
        font = QFont(cjk)
        font.setPixelSize(APP_FONT_PX)
        app.setFont(font)
    return families


def ui_font_families() -> list[str]:
    """给控件的 children 用的回退链（避免 Consolas 缺中文）。"""
    return ["Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", "sans-serif"]
