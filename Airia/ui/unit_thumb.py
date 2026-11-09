"""装备缩略图（无版权图片，2026-11-03）。

对应 docs/ui-cmo-redesign.md §3.5 的**修订**：
    - 原方案（YAML → 蓝图线框图）改为**直接使用无版权装备图片**（测试阶段口径）
    - 图片来源与授权登记在 ``assets/units/thumbs/MANIFEST.toml``（source/license/credit）
    - 回退链：MANIFEST 图片 → 同目录同名文件（png/jpg/jpeg/webp）→ 明确占位

实现口径（用户 2026-11-03）：**用控件，不自绘** → 本模块是 ``QLabel`` 子类，
有图显示 pixmap，无图显示「型号 + NO IMAGE」文本（QSS 控制样式）。

设计约束：
    - **纯 CPU + Qt 图片解码，零 OpenGL**（避开 QOpenGLWidget 嵌套 context 雷区）
    - 运行时只做盘查 + QPixmap 缓存，不生成、不重采样原图
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QWidget

logger = logging.getLogger(__name__)

# 项目根（Airia/ui/unit_thumb.py → parents[2] = 仓库根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_MANIFEST_NAME = "MANIFEST.toml"

# 阵营 → 竖条颜色（与 palette 语义色一致，QLabel 用 border-left 表达）
_SIDE_COLORS = {0: "#6f93b5", 1: "#b06a6a", 2: "#6f8f6f"}
_SIDE_COLOR_UNKNOWN = "#c0a050"

# 缓存：unit_id → QPixmap | None（None = 已查过、无图）
_PIXMAP_CACHE: dict[str, QPixmap | None] = {}
_MANIFEST_CACHE: dict[str, dict[str, dict]] = {}


def thumbs_dir(root: Path | None = None) -> Path:
    """缩略图目录。"""
    return (Path(root) if root is not None else PROJECT_ROOT) / "assets" / "units" / "thumbs"


def avatars_dir(root: Path | None = None) -> Path:
    """通话头像目录（``Sound only`` 回退见 comms_popup）。"""
    return (Path(root) if root is not None else PROJECT_ROOT) / "assets" / "avatars"


def load_manifest(root: Path | None = None) -> dict[str, dict]:
    """读取 ``MANIFEST.toml``（缺文件/格式错 → 空字典，不抛）。

    结构::

        [m1a2]
        file = "m1a2.png"
        license = "CC0 / public domain"
        source = "https://..."
    """
    base = str(Path(root) if root is not None else PROJECT_ROOT)
    if base in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[base]

    path = thumbs_dir(root) / _MANIFEST_NAME
    data: dict[str, dict] = {}
    if path.is_file():
        try:
            import tomllib

            with path.open("rb") as fh:
                raw = tomllib.load(fh)
            data = {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
        except Exception:  # noqa: BLE001 —— 配置坏了不该拖垮 UI
            logger.warning("缩略图 MANIFEST 解析失败：%s", path, exc_info=True)
            data = {}
    _MANIFEST_CACHE[base] = data
    return data


def clear_cache() -> None:
    """清空图片与 MANIFEST 缓存（测试/热重载用）。"""
    _PIXMAP_CACHE.clear()
    _MANIFEST_CACHE.clear()


def _find_file(unit_id: str, directory: Path) -> Path | None:
    for ext in _EXTENSIONS:
        p = directory / f"{unit_id}{ext}"
        if p.is_file():
            return p
    return None


def thumb_path(unit_id: str, root: Path | None = None) -> Path | None:
    """解析单位缩略图文件路径（无 → None）。

    优先 MANIFEST 指定的 ``file``（相对 thumbs/），其次同名文件扫描。
    """
    if not unit_id:
        return None
    directory = thumbs_dir(root)
    entry = load_manifest(root).get(str(unit_id))
    if entry:
        fname = entry.get("file")
        if fname:
            p = directory / str(fname)
            if p.is_file():
                return p
            logger.warning("MANIFEST 指向的缩略图不存在：%s", p)
    return _find_file(str(unit_id), directory)


def load_pixmap(unit_id: str, root: Path | None = None) -> QPixmap | None:
    """加载并缓存缩略图（无图 → None）。"""
    key = f"{Path(root) if root is not None else PROJECT_ROOT}::{unit_id}"
    if key in _PIXMAP_CACHE:
        return _PIXMAP_CACHE[key]
    path = thumb_path(unit_id, root)
    pm = QPixmap(str(path)) if path is not None else QPixmap()
    if path is not None and pm.isNull():
        logger.warning("缩略图解码失败：%s", path)
        pm = QPixmap()
    result = pm if not pm.isNull() else None
    _PIXMAP_CACHE[key] = result
    return result


class UnitThumbnail(QLabel):
    """96×96 装备缩略图（QLabel：有图显示图片，无图显示明确占位）。"""

    def __init__(
        self,
        unit_id: str = "",
        model: str = "",
        side: int = 0,
        size: int = 96,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._unit_id = str(unit_id)
        self._model = str(model)
        self._side = int(side)
        self._size = int(size)
        self._pixmap: QPixmap | None = None

        self.setObjectName("unitThumb")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(self._size, self._size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._reload()
        self._apply_style()

    # ---- 公开 API ----

    def set_unit(self, unit_id: str, model: str, side: int) -> None:
        """切换单位（同一单位不重载图片，避免每帧 IO）。"""
        changed = (
            str(unit_id) != self._unit_id
            or str(model) != self._model
            or int(side) != self._side
        )
        if not changed:
            return
        self._unit_id = str(unit_id)
        self._model = str(model)
        self._side = int(side)
        self._reload()
        self._apply_style()

    def set_size(self, size: int) -> None:
        """尺寸随 UI metrics 缩放（窗口 resize 时调用）。"""
        size = max(24, int(size))
        if size == self._size:
            return
        self._size = size
        self.setFixedSize(size, size)
        self._reload()
        self._apply_style()

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None

    # ---- 内部 ----

    def _reload(self) -> None:
        self._pixmap = load_pixmap(self._unit_id) if self._unit_id else None
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            self.setText(f"{self._model or self._unit_id or '—'}\nNO IMAGE")
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        target = int(round(self._size * dpr))
        scaled = self._pixmap.scaled(
            target,
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self.setText("")
        self.setPixmap(scaled)

    def _apply_style(self) -> None:
        color = _SIDE_COLORS.get(self._side, _SIDE_COLOR_UNKNOWN)
        size = max(7, int(round(self._size * 0.09)))
        self.setStyleSheet(
            f"""
QLabel#unitThumb {{
    background: #20242a;
    color: #8a929a;
    border: none;
    border-left: 3px solid {color};
    font-size: {size}px;
}}
"""
        )
