"""HUD 战场特效事件缓冲（Phase 5e 新增）。

对应 docs/phase5-ui-pipeline-design.md §4.3：
    - 弹道线（fire 事件：射手 → 目标，短暂显示并渐隐）
    - 命中标记（hit / ricochet / destroyed 事件：目标位置圆环 + 十字，渐隐）

设计：
    - EffectLayer 是**纯逻辑无状态缓冲**（不持有 GL 资源，可无头单测）：
      on_events() 接收事件列表 → 记录 (start_time, 相关单位索引)
      cull() 按墙钟时间（monotonic）老化丢弃过期特效
    - 渲染由 GLWidget 消费查询结果（tracer_events / hit_marks），
      用 WorldAdapter 在渲染时刻解析单位世界位置（不缓存旧位置）
    - 事件来源：MainWindow 每帧 drain EventBus 后喂入
      （与 LogWidget 同一份事件列表，不新增数据流）

单位索引约定：-1 = 无（跳过）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 事件种类常量（与 EventBus / LogWidget 标签一致）
_KIND_FIRE = "fire"
_HIT_KINDS = ("hit", "ricochet", "destroyed")


@dataclass(frozen=True)
class TracerEvent:
    """一条弹道线：射手 → 目标。"""

    shooter: int
    target: int
    start: float          # 墙钟时间（monotonic）
    lifetime: float


@dataclass(frozen=True)
class HitMark:
    """一个命中标记：目标单位位置。"""

    target: int
    start: float
    lifetime: float


class EffectLayer:
    """弹道线 / 命中标记事件缓冲（Phase 5e）。

    Args:
        fire_lifetime: 弹道线存活时长（秒）。
        hit_lifetime: 命中标记存活时长（秒）。
        max_tracers / max_hits: 缓冲上限（超出丢最旧，防刷屏）。
    """

    def __init__(
        self,
        fire_lifetime: float = 0.40,
        hit_lifetime: float = 0.60,
        max_tracers: int = 64,
        max_hits: int = 64,
    ) -> None:
        self.fire_lifetime = float(fire_lifetime)
        self.hit_lifetime = float(hit_lifetime)
        self.max_tracers = int(max_tracers)
        self.max_hits = int(max_hits)
        self._tracers: list[TracerEvent] = []
        self._hits: list[HitMark] = []

    # ------------------------------------------------------------------
    # 事件摄入
    # ------------------------------------------------------------------

    def on_events(self, events: list[dict], now: float) -> None:
        """摄入一帧事件列表（每帧 drain 后调用一次）。

        Args:
            events: [{kind, data, ...}]（与 LogWidget 同一份）。
            now: 当前墙钟时间（time.monotonic()）。
        """
        for ev in events:
            kind = ev.get("kind", "")
            data = ev.get("data") or {}
            if kind == _KIND_FIRE:
                shooter = int(data.get("shooter", -1))
                target = int(data.get("target", -1))
                if shooter >= 0 and target >= 0:
                    self._tracers.append(
                        TracerEvent(shooter, target, now, self.fire_lifetime)
                    )
            elif kind in _HIT_KINDS:
                target = data.get("target", data.get("armor_idx", -1))
                if target is None:
                    continue
                target = int(target)
                if target >= 0:
                    self._hits.append(
                        HitMark(target, now, self.hit_lifetime)
                    )
        self._cap()

    def cull(self, now: float) -> None:
        """丢弃已过期的特效（每帧渲染前调用）。"""
        self._tracers = [t for t in self._tracers if now - t.start <= t.lifetime]
        self._hits = [h for h in self._hits if now - h.start <= h.lifetime]

    def clear(self) -> None:
        self._tracers.clear()
        self._hits.clear()

    # ------------------------------------------------------------------
    # 查询（渲染消费）
    # ------------------------------------------------------------------

    @property
    def tracer_events(self) -> list[TracerEvent]:
        return list(self._tracers)

    @property
    def hit_marks(self) -> list[HitMark]:
        return list(self._hits)

    def tracer_count(self) -> int:
        return len(self._tracers)

    def hit_count(self) -> int:
        return len(self._hits)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _cap(self) -> None:
        """超出上限丢最旧（防大量事件刷屏导致每帧遍历膨胀）。"""
        if len(self._tracers) > self.max_tracers:
            del self._tracers[: len(self._tracers) - self.max_tracers]
        if len(self._hits) > self.max_hits:
            del self._hits[: len(self._hits) - self.max_hits]
