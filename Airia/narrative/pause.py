"""推演暂停策略（2026-11-03 定稿）。

规则（用户口径）：

    推演的暂停**只能由玩家或教程触发**；
    真正战役（``real_campaign`` flag）**禁用暂停**。

实现要点：
    - 暂停来源分开记账（manual / tutorial / narrative），便于 UI 分别提示；
    - ``locked=True``（战役进行中）时**所有来源**都被拒绝，且立即清空既有暂停
      ——包括剧情对话暂停：战役里对话不应该打断推演（叙事层不阻塞模拟的原则）；
    - 本模块是纯逻辑（无 Qt / 无引擎依赖），可无头单测。

为什么不放在 UI：暂停是**玩法规则**，将来无头回放 / 自动化测试也要遵守，
所以放在 narrative 层，UI 只是调用方。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 真正战役标志位（由剧情/关卡设置）
FLAG_REAL_CAMPAIGN = "real_campaign"


@dataclass
class PauseController:
    """暂停来源与战役锁。"""

    manual: bool = False      # 玩家（空格 / 按钮）
    tutorial: bool = False    # 教程步骤
    narrative: bool = False   # 剧情对话（pause_sim）
    locked: bool = False      # 真正战役：禁止一切暂停

    # ---- 查询 ----

    @property
    def paused(self) -> bool:
        """当前是否应暂停（战役锁优先）。"""
        if self.locked:
            return False
        return self.manual or self.tutorial or self.narrative

    @property
    def can_pause(self) -> bool:
        """是否允许触发暂停。"""
        return not self.locked

    @property
    def active_sources(self) -> tuple[str, ...]:
        """当前生效的暂停来源（UI 提示用）。"""
        if self.locked:
            return ()
        names = (("manual", self.manual), ("tutorial", self.tutorial),
                 ("narrative", self.narrative))
        return tuple(name for name, on in names if on)

    # ---- 修改 ----

    def set_locked(self, locked: bool) -> None:
        """切换战役锁；上锁时清空所有暂停来源。"""
        self.locked = bool(locked)
        if self.locked:
            self.manual = False
            self.tutorial = False
            self.narrative = False

    def request_manual(self, on: bool | None = None) -> bool:
        """玩家请求暂停/恢复（None = 切换）。返回是否被接受。"""
        if self.locked:
            return False
        self.manual = (not self.manual) if on is None else bool(on)
        return True

    def request_tutorial(self, on: bool = True) -> bool:
        """教程请求暂停。返回是否被接受。"""
        if self.locked:
            return False
        self.tutorial = bool(on)
        return True

    def set_narrative(self, on: bool) -> None:
        """剧情对话暂停（战役锁下被忽略）。"""
        self.narrative = bool(on) and not self.locked

    def clear(self) -> None:
        """清空所有暂停来源（不改锁状态）。"""
        self.manual = False
        self.tutorial = False
        self.narrative = False
