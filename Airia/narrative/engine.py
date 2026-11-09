"""叙事层中央调度器（AGENTS.md Phase 4：narrative/engine.py）。

对应 docs/phase4-narrative-design.md §3.1：
    - NarrativeEngine：每帧 tick(snapshot, events) → NarrativeState
    - 集成 DialogueEngine / TriggerSystem / CommandFactory / CombatEventRecorder
    - 管理全局 flags；处理玩家选择；生成战斗简报

数据流（docs §5.1）：
    [模拟层] EventBus.drain() → events
    [叙事层] engine.tick(snapshot, events)
        ├── recorder.on_events(events, snapshot)   # 事件记录
        ├── trigger_system.evaluate(...)           # 触发器
        ├── dialogue_engine.activate(...)          # 对话
        └── return NarrativeState                   # 渲染层消费
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.simulation.commands import Command, CommandQueue
from Airia.lulilo.simulation.loop import SimulationEvent
from Airia.lulilo.simulation.state import SimulationSnapshot

from Airia.narrative.commands import CommandFactory
from Airia.narrative.dialogue import DialogueEngine, DialogueState
from Airia.narrative.pause import FLAG_REAL_CAMPAIGN, PauseController
from Airia.narrative.recorder import (
    CombatBriefing,
    CombatEventRecorder,
    UnitStatsReport,
)
from Airia.narrative.triggers import TriggerAction, TriggerSystem

logger = get_logger(__name__)


@dataclass
class NarrativeState:
    """渲染层消费的叙事状态（docs §3.1 NarrativeState）。"""

    active_dialogue: DialogueState | None = None   # 当前活跃对话（None = 无对话）
    pending_commands: list[Command] = field(default_factory=list)  # 待推入队列的指令
    flags: set[str] = field(default_factory=set)  # 全局标志位
    paused: bool = False                           # 模拟是否应暂停（等待玩家选择）
    recent_events: list[dict] = field(default_factory=list)  # 事件摘要（HUD 滚动日志）


class NarrativeEngine:
    """叙事层中央调度器。

    用法（docs §5.1）:
        engine = NarrativeEngine(command_queue, recorder=recorder)
        state = engine.tick(snapshot, events)   # 每帧
        engine.on_player_choice("first_blood", 1)  # 玩家点击选项
        briefing = engine.generate_briefing()     # 任务结束

    NOTE（AGENTS.md 3.7）：本引擎永不持有 SimulationState 引用，
    只读 Snapshot + EventBus，只通过 CommandQueue 下发意图。
    """

    def __init__(
        self,
        command_queue: CommandQueue,
        recorder: CombatEventRecorder | None = None,
        loop: Any | None = None,
    ) -> None:
        """初始化。

        Args:
            command_queue: 模拟层 CommandQueue（玩家指令注入）。
            recorder: 战斗事件记录器（可选；未提供则无事件记录/简报）。
            loop: 可选的 SimulationLoop 引用。提供后，对话激活/结束时
                自动同步 ``loop.sim_paused``（docs §零：叙事层可设置暂停）。
        """
        self.command_queue = command_queue
        self.dialogue_engine = DialogueEngine()
        self.trigger_system = TriggerSystem()
        self.recorder = recorder
        self.loop = loop

        self._flags: set[str] = set()
        self._active_dialogue: DialogueState | None = None
        self._active_tree_id: str | None = None
        # 暂停策略（玩家/教程可暂停；真正战役禁止暂停，见 narrative/pause.py）
        self._pause = PauseController()
        self._last_snapshot: SimulationSnapshot | None = None
        self._recent_events: list[dict] = []
        self._sim_started = False
        self._sim_time = 0.0
        # 单位上下文（触发器 side 过滤用）
        self._unit_names: dict[int, str] = {}
        self._side_map: dict[int, str] = {}

        # 玩家选择后的待执行动作（本帧 advance 收集）
        self._pending_actions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 数据注入
    # ------------------------------------------------------------------

    def set_unit_context(
        self,
        unit_registry: dict[int, str] | None = None,
        side_map: dict[int, str] | None = None,
    ) -> None:
        """注入单位名称/阵营映射（recorder + 触发器 side 过滤使用）。

        Args:
            unit_registry: armor_idx → 单位名。
            side_map: armor_idx → "ally"/"enemy"/"neutral"。
        """
        if unit_registry:
            self._unit_names.update(unit_registry)
        if side_map:
            self._side_map.update(side_map)
        if self.recorder is not None:
            if unit_registry:
                self.recorder.unit_registry.update(unit_registry)
            if side_map:
                self.recorder.side_map.update(side_map)

    def set_flag(self, name: str) -> None:
        """设置一个全局标志位（``real_campaign`` 会锁定/解锁暂停）。"""
        self._flags.add(name)
        if name == FLAG_REAL_CAMPAIGN:
            self._pause.set_locked(True)
            self._sync_pause()

    def clear_flag(self, name: str) -> None:
        self._flags.discard(name)
        if name == FLAG_REAL_CAMPAIGN:
            self._pause.set_locked(False)
            self._sync_pause()

    def has_flag(self, name: str) -> bool:
        return name in self._flags

    # ------------------------------------------------------------------
    # 主 Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        snapshot: SimulationSnapshot,
        events: list[SimulationEvent],
    ) -> NarrativeState:
        """每帧调用（docs §5.1）。

        Args:
            snapshot: 模拟层只读快照。
            events: EventBus.drain() 返回的本帧事件。

        Returns:
            NarrativeState：当前叙事状态（渲染层据此决定显示什么）。
        """
        self._last_snapshot = snapshot
        self._sim_time = snapshot.sim_time

        # 1. 事件记录（不阻塞，O(1) 写缓冲区）
        if self.recorder is not None:
            self.recorder.on_events(events, snapshot)

        # 2. 事件摘要（HUD 滚动日志，最近 20 条）
        for ev in events[-20:]:
            self._recent_events.append(
                {"time": snapshot.sim_time, "kind": ev.kind, "data": ev.data}
            )
        self._recent_events = self._recent_events[-20:]

        # 3. 首帧触发开场对话
        if not self._sim_started:
            self._sim_started = True
            self._dispatch_trigger_actions(
                self.trigger_system.evaluate(
                    events=[SimulationEvent(kind="__tick0__")],
                    snapshot=snapshot,
                    flags=self._flags,
                    sim_time=snapshot.sim_time,
                    side_map=self._side_map,
                )
            )

        # 4. 触发器求值（已有活跃对话时不打断，避免对话被新触发覆盖）
        if self._active_dialogue is None:
            fired = self.trigger_system.evaluate(
                events=events,
                snapshot=snapshot,
                flags=self._flags,
                sim_time=snapshot.sim_time,
                side_map=self._side_map,
            )
            self._dispatch_trigger_actions(fired)

        # 5. 返回渲染状态
        return self._build_state()

    # ------------------------------------------------------------------
    # 触发器 → 对话激活
    # ------------------------------------------------------------------

    def _dispatch_trigger_actions(self, actions: list[TriggerAction]) -> None:
        """执行触发器结果（按 priority 降序，首个可激活的对话优先）。"""
        for act in actions:
            if not act.dialogue:
                continue
            if self._active_dialogue is not None:
                continue  # 已有活跃对话，跳过
            if not self.dialogue_engine.has(act.dialogue):
                logger.warning("触发器引用未加载的对话树: %s", act.dialogue)
                continue
            self._activate_dialogue(act.dialogue, pause_sim=act.pause_sim)
            return  # 一次只激活一个对话

    def _activate_dialogue(self, dialogue_id: str, pause_sim: bool = True) -> None:
        """激活一棵对话树并暂停模拟。

        eval_ctx 随激活一并注入：入口节点若为 choice，其条件选项
        （docs §2.3）需要 flags/state 才能正确过滤。
        """
        self._active_tree_id = dialogue_id
        self._active_dialogue = self.dialogue_engine.activate(
            dialogue_id, self._eval_ctx()
        )
        self._pause.set_narrative(bool(pause_sim))
        self._sync_pause()
        logger.info("对话激活: %s (pause=%s)", dialogue_id, pause_sim)

    # ------------------------------------------------------------------
    # 玩家交互
    # ------------------------------------------------------------------

    def on_player_choice(self, dialogue_id: str, option_index: int) -> NarrativeState:
        """玩家选择了某个选项。

        Args:
            dialogue_id: 对话树 id。
            option_index: 选项索引。

        Returns:
            新的 NarrativeState（对话可能已结束）。
        """
        if self._active_dialogue is None or self._active_tree_id != dialogue_id:
            logger.warning("无活跃对话或 id 不匹配: %s", dialogue_id)
            return self._build_state()

        # 推进对话（统一走 advance_dialogue，更新缓存）
        state = self.advance_dialogue(option_index)

        # 选项 action → Command → 入队
        cmds = CommandFactory.from_actions(self._pending_actions)
        for cmd in cmds:
            self.command_queue.push(cmd)

        return state

    def advance_dialogue(self, option_index: int | None = None) -> NarrativeState:
        """推进当前对话到下一节点（docs §3.2）。

        - text/action/branch 节点自动前进：option_index=None
        - choice 节点需玩家选择：option_index=选项索引

        Args:
            option_index: choice 节点选项索引；None = 自动前进。

        Returns:
            新的 NarrativeState（对话结束则 active_dialogue=None）。
        """
        if self._active_dialogue is None:
            return self._build_state()

        self._pending_actions.clear()
        ctx = self._eval_ctx()
        new_state = self.dialogue_engine.advance(option_index, ctx)

        if new_state.is_finished:
            self._active_dialogue = None
            self._active_tree_id = None
            self._pause.set_narrative(False)
            self._sync_pause()
            logger.info("对话结束，模拟恢复")
        else:
            self._active_dialogue = new_state

        return self._build_state()

    def skip_active_dialogue(self) -> NarrativeState:
        """跳过当前对话（直接结束，不产生指令）。"""
        self._active_dialogue = None
        self._active_tree_id = None
        self._pause.set_narrative(False)
        self._sync_pause()
        return self._build_state()

    # ------------------------------------------------------------------
    # 简报
    # ------------------------------------------------------------------

    def generate_briefing(self, subjective_notes: str = "") -> CombatBriefing | None:
        """任务结束：生成战斗简报（需配置 recorder）。"""
        if self.recorder is None:
            logger.warning("未配置 recorder，无法生成简报")
            return None
        return self.recorder.generate_briefing(subjective_notes=subjective_notes)

    def unit_stats(self, armor_idx: int) -> UnitStatsReport | None:
        if self.recorder is None:
            return None
        return self.recorder.unit_stats(armor_idx)

    # ------------------------------------------------------------------
    # 查询 / 状态
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """模拟是否应暂停（受暂停策略约束：战役中恒为 False）。"""
        return self._pause.paused

    @property
    def pause(self) -> PauseController:
        """暂停策略（UI 读取来源 / 判定是否允许）。"""
        return self._pause

    @property
    def pause_locked(self) -> bool:
        """真正战役进行中（禁止暂停）。"""
        return self._pause.locked

    # ---- 暂停请求（玩家 / 教程）----

    def request_manual_pause(self, on: bool | None = None) -> bool:
        """玩家请求暂停/恢复；返回是否被接受（战役中被拒绝）。"""
        accepted = self._pause.request_manual(on)
        self._sync_pause()
        return accepted

    def request_tutorial_pause(self, on: bool = True) -> bool:
        """教程请求暂停；返回是否被接受（战役中被拒绝）。"""
        accepted = self._pause.request_tutorial(on)
        self._sync_pause()
        return accepted

    def set_campaign_mode(self, active: bool) -> None:
        """切换真正战役模式（开=锁暂停，关=解锁）。"""
        if active:
            self.set_flag(FLAG_REAL_CAMPAIGN)
        else:
            self.clear_flag(FLAG_REAL_CAMPAIGN)
        if self.loop is not None:
            self.loop.sim_paused = self._pause.paused

    @property
    def has_active_dialogue(self) -> bool:
        return self._active_dialogue is not None

    @property
    def flags(self) -> set[str]:
        return set(self._flags)

    def _sync_pause(self) -> None:
        """把 is_paused 同步到 SimulationLoop.sim_paused（docs §零）。"""
        if self.loop is not None:
            self.loop.sim_paused = self._pause.paused

    def _build_state(self) -> NarrativeState:
        return NarrativeState(
            active_dialogue=self._active_dialogue,
            pending_commands=[],
            flags=set(self._flags),
            paused=self._pause.paused,
            recent_events=list(self._recent_events),
        )

    def _eval_ctx(self) -> dict[str, Any]:
        """对话条件求值上下文。"""
        return {
            "flags": self._flags,
            "state": self._last_snapshot,
            "event": self._recent_events[-1] if self._recent_events else None,
            "sim_time": self._sim_time,
            "target_unit": None,
            "pending_actions": self._pending_actions,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def load_dialogues(self, dir_path: Path | str) -> int:
        """加载目录下全部对话树。"""
        return self.dialogue_engine.load_all(dir_path)

    def load_triggers(self, dir_path: Path | str) -> int:
        """加载目录下全部触发器 YAML。"""
        d = Path(dir_path)
        count = 0
        for p in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            count += self.trigger_system.load(p)
        return count

    def reset(self) -> None:
        """重置叙事状态（新任务开始）。"""
        self._flags.clear()
        self._active_dialogue = None
        self._active_tree_id = None
        self._pause.set_narrative(False)
        self._sync_pause()
        self._recent_events.clear()
        self._sim_started = False
        self.trigger_system.reset()
        if self.recorder is not None:
            self.recorder.reset()

