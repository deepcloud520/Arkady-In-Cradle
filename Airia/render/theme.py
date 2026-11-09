"""着色器主题系统（Phase 5 新增）。

对应 docs/phase5-ui-pipeline-design.md §3：
    - 目录结构：assets/shaders/<theme_id>/{terrain.vert, terrain.frag, theme.toml}
    - ThemeManager：discover / load / switch（热切换）
    - 热切换流程：编译新 program → 通知消费者重绑 VAO → 释放旧 program

接口约束（重要）：
    所有主题的 terrain.vert 必须声明相同的 attributes
    （in_vert / in_color / in_normal）与 uniform（mvp + theme.toml 的
    [uniforms]），否则地形 VAO 无法跨主题复用重建。
"""
from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import moderngl

logger = logging.getLogger(__name__)

# assets/shaders/ 根目录（相对本文件 ../../assets/shaders）
SHADER_ROOT = Path(__file__).resolve().parents[2] / "assets" / "shaders"


@dataclass
class Theme:
    """一个已加载的着色器主题。"""

    id: str
    name: str
    program: moderngl.Program             # 已编译的着色器程序
    uniforms: dict[str, Any] = field(default_factory=dict)   # theme.toml [uniforms]
    metadata: dict[str, str] = field(default_factory=dict)   # theme.toml [theme]
    root: Path | None = None


class ThemeManager:
    """着色器主题管理器：加载 / 热切换 / 通知消费者。

    用法:
        manager = ThemeManager(ctx)            # ctx 需在 OpenGL 上下文就绪后创建
        manager.switch("default")              # 加载并设为当前
        manager.on_switch(self._on_theme)      # 注册切换回调（GLWidget 重绑 VAO）
        manager.switch("constructivism")       # 运行时热切换
    """

    def __init__(self, ctx: moderngl.Context, root: Path | None = None) -> None:
        self.ctx = ctx
        self.root = root or SHADER_ROOT
        self._themes: dict[str, Theme] = {}
        self._current: Theme | None = None
        self._on_switch: list[Callable[[Theme], None]] = []

    # ------------------------------------------------------------------
    # 发现 / 加载
    # ------------------------------------------------------------------

    def discover(self) -> list[str]:
        """扫描 assets/shaders/ 返回可用主题 id 列表（含 theme.toml 的目录）。"""
        if not self.root.is_dir():
            logger.warning("主题目录不存在: %s", self.root)
            return []
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "theme.toml").is_file()
        )

    def load(self, theme_id: str) -> Theme:
        """加载一个主题（编译着色器 + 读取元数据 + 默认 uniforms）。"""
        theme_dir = self.root / theme_id
        if not (theme_dir / "theme.toml").is_file():
            raise FileNotFoundError(f"主题不存在: {theme_id} @ {theme_dir}")

        with open(theme_dir / "theme.toml", "rb") as f:
            meta = tomllib.load(f)

        program = self.ctx.program(
            vertex_shader=(theme_dir / "terrain.vert").read_text(encoding="utf-8"),
            fragment_shader=(theme_dir / "terrain.frag").read_text(encoding="utf-8"),
        )
        theme = Theme(
            id=theme_id,
            name=meta.get("theme", {}).get("name", theme_id),
            program=program,
            uniforms=dict(meta.get("uniforms", {})),
            metadata=dict(meta.get("theme", {})),
            root=theme_dir,
        )
        self._themes[theme_id] = theme
        logger.info("主题加载: %s (%s)", theme_id, theme.name)
        return theme

    def ensure(self, theme_id: str) -> Theme:
        """返回已加载主题；未加载则先加载。"""
        return self._themes.get(theme_id) or self.load(theme_id)

    # ------------------------------------------------------------------
    # 切换
    # ------------------------------------------------------------------

    def switch(self, theme_id: str) -> Theme:
        """热切换到指定主题。

        流程（docs §3.4）：
            编译新 program → 通知消费者重绑 VAO（回调先执行）→ 释放旧 program。
        先回调后释放，避免消费者重绑时引用已释放的 program。

        注意（2026-08-13 修复）：释放旧 program 后必须将其移出缓存
        （``self._themes``），否则切回该主题时 ``ensure()`` 会复用已释放的
        InvalidObject program（重建 VAO 抛 ``TypeError: argument 1 must be
        mgl.Program, not InvalidObject``）。移出缓存后下次切换会重新编译。
        """
        theme = self.ensure(theme_id)
        old = self._current
        self._current = theme
        for cb in self._on_switch:
            cb(theme)
        if old is not None and old is not theme:
            old.program.release()
            # 已释放的 program 不可复用：移出缓存，下次切换时重新编译
            self._themes.pop(old.id, None)
            logger.info("主题切换: %s → %s（旧 program 已释放并移出缓存）", old.id, theme_id)
        return theme

    def apply_uniforms(self, theme: Theme | None = None) -> None:
        """把主题默认 uniforms 写入 program（theme.toml [uniforms]）。"""
        theme = theme or self._current
        if theme is None:
            return
        for name, value in theme.uniforms.items():
            if name not in theme.program:
                continue
            if isinstance(value, list):
                value = tuple(value)
            theme.program[name].value = value

    # ------------------------------------------------------------------
    # 查询 / 回调
    # ------------------------------------------------------------------

    @property
    def current(self) -> Theme | None:
        return self._current

    @property
    def theme_ids(self) -> list[str]:
        return list(self._themes)

    def on_switch(self, callback: Callable[[Theme], None]) -> None:
        """注册切换回调（如 GLWidget 重绑 VAO）。"""
        self._on_switch.append(callback)
