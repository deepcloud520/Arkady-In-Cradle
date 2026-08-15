"""Tick 主循环 + 事件队列（AGENTS.md Phase 3：simulation/loop.py）。

对应 AGENTS.md 3.5「仿真引擎（无头模式）」：
    - 外循环：固定 Tick（地面 1s / 空中 0.1s / 子弹 0.01s），推进连续状态
    - 内循环：事件驱动（命中、摧毁、通信中断）

SimulationLoop 职责：
    1. 消费 CommandQueue 中的玩家指令（AGENTS.md 3.7）
    2. 按外循环 Tick 调度各子系统（movement → detection → combat → logistics）
    3. 收集内循环事件（命中/摧毁/补给）入 EventBus，供上层消费
    4. sim_paused 标志：扩容 / NavGraph 热交换只在暂停间隙执行（AGENTS.md 3.1）

骨架期：子系统以可选参数注入；未装配时 step() 可空转（推进时钟 + 消费命令）。
多分辨率 Tick 的正式调度（各层独立累加器）在 Phase 3 迭代中接入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from Airis.lulilo.core.logging import get_logger
from Airis.lulilo.simulation.commands import Command, CommandQueue
from Airis.lulilo.simulation.state import SimulationState

logger = get_logger(__name__)

# 多分辨率 Tick（AGENTS.md 3.5 外循环）
TICK_GROUND = 1.0    # 地面单位
TICK_AIR = 0.1       # 空中单位
TICK_BULLET = 0.01   # 子弹/高速抛射物


@dataclass(frozen=True)
class SimulationEvent:
    """一次内循环事件（命中/摧毁/补给/通信…），供叙事层消费。"""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """事件队列（内循环产物，上层 drain 消费）。"""

    def __init__(self) -> None:
        self._events: list[SimulationEvent] = []

    def emit(self, kind: str, data: dict[str, Any] | None = None) -> None:
        self._events.append(SimulationEvent(kind=kind, data=data or {}))

    def drain(self) -> list[SimulationEvent]:
        events, self._events = self._events, []
        return events

    def drain_filter(self, kinds: set[str]) -> list[SimulationEvent]:
        """定向取事件：只取指定 kind 的事件，其余保留在队列（Phase 8 多消费者）。

        炮兵 AI 订阅 support_request 时用（不吞掉 command 等其他事件）。
        """
        matched = [e for e in self._events if e.kind in kinds]
        self._events = [e for e in self._events if e.kind not in kinds]
        return matched

    def __len__(self) -> int:
        return len(self._events)


# 子系统 hook 类型：step(loop, dt) -> 可调用（正式迭代实现）
Hook = Callable[["SimulationLoop", float], None]


class SimulationLoop:
    """无头仿真主循环。

    用法:
        loop = SimulationLoop(state=SimulationState())
        loop.attach("movement", move_hook)
        loop.step(0.1)            # 推进一个逻辑帧
        loop.run_headless(60.0, 1.0)  # 无头跑 60s
        events = loop.events.drain()  # 消费事件
    """

    def __init__(self, state: SimulationState | None = None) -> None:
        self.state = state or SimulationState()
        self.commands = CommandQueue()
        self.events = EventBus()
        self.sim_paused = False          # 暂停间隙：扩容/烘焙热交换（AGENTS.md 3.1）
        self._hooks: dict[str, Hook] = {}
        # Phase 6c 步骤⑤（docs/phase6-ai-pipeline.md §六.2）：指令路由回调。
        # 装配层（scenario ⑩）注入：STRIKE → 炮兵 AI 直连（loop 已先 _drain_commands，
        # AI 不从本队列读玩家指令）；返回 True 表示已消费（不再发通用 command 事件）。
        self.command_handler: Callable[[Command], bool] | None = None
        # 多分辨率 Tick 调度（AGENTS.md 3.5 外循环）：name → 分频周期(秒)
        # 未配置的子系统每帧执行（兼容骨架行为）
        self._rate_intervals: dict[str, float] = {}
        self._rate_accum: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 多分辨率 Tick 调度
    # ------------------------------------------------------------------

    def set_rate(self, name: str, interval: float) -> None:
        """为子系统设置分频 Tick 周期（秒）。

        Args:
            name: 子系统名（movement / detection / combat / logistics …）。
            interval: 分频周期 (s)；>0 启用分频（累加器累计，到期执行一次），
                ≤0 表示每帧执行（默认，兼容骨架）。
        """
        if interval > 0.0:
            self._rate_intervals[name] = float(interval)
            self._rate_accum.setdefault(name, 0.0)
        else:
            self._rate_intervals.pop(name, None)
            self._rate_accum.pop(name, None)

    def rate_of(self, name: str) -> float:
        """返回子系统当前分频周期；未配置返回 0.0（每帧执行）。"""
        return float(self._rate_intervals.get(name, 0.0))

    def configure_default_rates(self) -> None:
        """配置 AGENTS.md 3.5 默认多分辨率 Tick：
        地面 1s / 空中 0.1s / 子弹 0.01s。

        - movement / detection / ai / logistics：地面 Tick（1s）
        - combat：子弹 Tick（0.01s，内部再细分抛射物积分）
        """
        self.set_rate("movement", TICK_GROUND)
        self.set_rate("air_movement", TICK_AIR)     # Phase 7：气动积分 + 飞控（0.1s）
        self.set_rate("detection", TICK_GROUND)
        self.set_rate("ai", TICK_GROUND)          # Phase 6c 编队/单位 AI（docs/phase6-ai-pipeline.md §二.5）
        self.set_rate("air_ai", TICK_GROUND)      # Phase 7：空中编队/单位 AI（1s 秒级慢决策）
        self.set_rate("combat", TICK_BULLET)
        self.set_rate("logistics", TICK_GROUND)


    def attach(self, name: str, hook: Hook) -> None:
        """挂载一个子系统 hook（movement / detection / combat / logistics …）。"""
        if name in self._hooks:
            logger.warning("子系统 %s 重复挂载，将覆盖旧实现", name)
        self._hooks[name] = hook

    def detach(self, name: str) -> None:
        """卸载一个子系统 hook。"""
        self._hooks.pop(name, None)
        self.set_rate(name, 0.0)  # 一并清除其分频配置

    def has(self, name: str) -> bool:
        """查询子系统是否已挂载。"""
        return name in self._hooks

    # ------------------------------------------------------------------
    # 指令执行（AGENTS.md 3.7：Command 消费）
    # ------------------------------------------------------------------

    def _apply_command(self, cmd: Command) -> None:
        """执行一条玩家指令（AGENTS.md 3.7）。

        - command_handler（装配层注入）优先：STRIKE → 炮兵 AI 直连（Phase 6c 步骤⑤）；
          返回 True 表示已消费，不再发通用 command 事件。
        - 其余指令仅记录事件（正式迭代分派到子系统）。
        """
        if self.command_handler is not None:
            try:
                if self.command_handler(cmd):
                    return
            except Exception:
                logger.exception("command_handler 处理指令失败: %s", cmd)
        self.events.emit("command", {"type": cmd.type.name, "target": cmd.target_id})

    def _drain_commands(self) -> None:
        for cmd in self.commands.drain():
            self._apply_command(cmd)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """推进一个逻辑帧（外循环 Tick）。

        骨架调度顺序（AGENTS.md 依赖链）：
            命令消费 → movement → detection → combat → logistics
        各 hook 按挂载顺序执行；未挂载的子系统跳过。

        Args:
            dt: 本帧时长 (s)。
        """
        if self.sim_paused:
            # 暂停间隙：只消费命令，不推进物理状态（扩容/烘焙窗口）
            self._drain_commands()
            return

        self._drain_commands()

        # 按依赖链顺序调度（挂载即执行，未挂载跳过）
        # 多分辨率：配置了分频周期的子系统按累加器到期执行；否则每帧执行
        # Phase 6c：ai（编队决策 + 单位执行）插在 detection 之后（读接触）、
        # combat 之前（开火即时 spawn，弹丸本 tick 在 combat 推进）
        # Phase 7：air_movement（飞→探测）在 detection 之前；air_ai（决策→开火）
        # 在 combat 之前——保证「飞→探测→AI→开火→弹丸推进」因果序（docs §1.2）
        for name in ("movement", "air_movement", "detection", "ai", "air_ai", "combat", "logistics"):
            hook = self._hooks.get(name)
            if hook is None:
                continue
            interval = self._rate_intervals.get(name)
            if interval is None:
                hook(self, dt)          # 兼容骨架：每帧执行
                continue
            acc = self._rate_accum[name] + dt
            if acc >= interval:
                hook(self, acc)         # 传累计时长作为子 dt（子系统内细分）
                self._rate_accum[name] = 0.0
            else:
                self._rate_accum[name] = acc

        self.state.sim_time += dt
        self.state.tick_count += 1

    def run_headless(self, duration: float, dt: float = TICK_GROUND) -> int:
        """无头运行 duration 秒，返回执行的 Tick 数。"""
        t = 0.0
        ticks = 0
        while t < duration:
            self.step(dt)
            t += dt
            ticks += 1
        logger.info("无头运行完成: %.1fs / %d ticks", duration, ticks)
        return ticks
