"""Phase 4c 集成测试：完整 Tick 循环 + 示例对话树 + 事件记录 + 简报生成。

覆盖 docs/phase4-narrative-design.md §5.1 完整数据流：
    SimulationLoop.step() → EventBus → NarrativeEngine.tick()
        ├── recorder.on_events()       # 事件记录 → SQLite
        ├── trigger_system.evaluate()  # 触发器
        ├── dialogue_engine.activate() # 对话激活（pause_sim）
        └── on_player_choice()         # 玩家选择 → Command → CommandQueue
    → 模拟层下一 Tick 消费指令

使用 story/dialogues/ + story/triggers/ 的示例 YAML（Phase 4c 交付物）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Airia.lulilo.simulation.commands import CommandQueue, CommandType
from Airia.lulilo.simulation.loop import SimulationEvent, SimulationLoop
from Airia.lulilo.simulation.state import SimulationState

from Airia.narrative.engine import NarrativeEngine
from Airia.narrative.recorder import CombatEventRecorder

# 示例数据目录（项目根 story/）
STORY_DIR = Path(__file__).resolve().parents[1] / "story"
DIALOGUES_DIR = STORY_DIR / "dialogues"
TRIGGERS_DIR = STORY_DIR / "triggers"


@pytest.fixture
def story_files_exist() -> None:
    """前置：示例 YAML 存在（Phase 4c 交付物）。"""
    assert (DIALOGUES_DIR / "prologue.yaml").exists()
    assert (DIALOGUES_DIR / "first_blood.yaml").exists()
    assert (DIALOGUES_DIR / "time_warning.yaml").exists()
    assert (TRIGGERS_DIR / "chapter1.yaml").exists()


def make_world() -> tuple[SimulationLoop, NarrativeEngine, CombatEventRecorder]:
    """构建完整世界：无头模拟 + 叙事层 + 事件记录（内存 SQLite）。"""
    state = SimulationState()
    loop = SimulationLoop(state=state)

    recorder = CombatEventRecorder(
        db_path=":memory:",
        unit_registry={0: "M1A2 #0", 1: "T-90 #1", 2: "T-90 #2"},
        side_map={0: "ally", 1: "enemy", 2: "enemy"},
        mission_id="integ_001",
    )
    # 接线：engine.is_paused → loop.sim_paused（docs §零）
    engine = NarrativeEngine(command_queue=loop.commands, recorder=recorder, loop=loop)
    # 注入单位上下文（recorder 名称 + 触发器 side 过滤）
    engine.set_unit_context(
        unit_registry={0: "M1A2 #0", 1: "T-90 #1", 2: "T-90 #2"},
        side_map={0: "ally", 1: "enemy", 2: "enemy"},
    )
    return loop, engine, recorder


def pump_events(loop: SimulationLoop, engine: NarrativeEngine, *events) -> None:
    """把事件注入 EventBus 并驱动叙事层一个 Tick（模拟 combat 埋点）。"""
    for ev in events:
        loop.events.emit(ev.kind, ev.data)
    drained = loop.events.drain()
    snap = loop.state.snapshot()
    engine.tick(snap, drained)


class TestIntegratedFlow:
    def test_full_flow_with_example_data(self, story_files_exist: None) -> None:
        """端到端：开场对话 → 玩家选择 → 指令 → 战斗 → 首次击毁对话 → 简报。"""
        loop, engine, recorder = make_world()
        assert engine.load_dialogues(DIALOGUES_DIR) == 3
        assert engine.load_triggers(TRIGGERS_DIR) == 3

        # ---- 第 1 帧：首帧触发开场对话（on_first_tick → prologue_001）----
        loop.step(1.0)
        pump_events(loop, engine)

        assert engine.has_active_dialogue
        assert engine.is_paused
        state = engine._build_state()
        assert state.active_dialogue is not None
        assert state.active_dialogue.speaker == "前线通讯"

        # 推进到 choice 节点（统一接口）
        engine.advance_dialogue(None)
        choice = engine._build_state().active_dialogue
        assert choice is not None
        # 无 has_scout_report → 条件选项「等待更多侦察情报」被引擎过滤（docs §2.3）
        assert len(choice.options) == 2  # 固守 / 出击
        assert choice.option_indices == [0, 1]

        # ---- 玩家选择「固守」→ Command 入队 ----
        engine.on_player_choice("prologue_001", 0)
        drained = loop.commands.drain()
        assert len(drained) == 1
        assert drained[0].type == CommandType.HOLD
        assert drained[0].params["formation"] == "defensive_line"
        assert engine.has_flag("defensive_posture")

        # 对话应推进到 deploy_defensive（text，未结束）
        assert engine.has_active_dialogue
        assert engine.is_paused

        # 再 advance → 对话结束，恢复模拟
        engine.on_player_choice("prologue_001", 0)  # text 节点 advance（忽略索引）
        assert not engine.has_active_dialogue
        assert not engine.is_paused

        # ---- 战斗事件：开火 / 命中 / 首次击毁（T-90 #1）----
        loop.step(1.0)  # 推进时钟到 2.0s
        snap = loop.state.snapshot()
        engine.tick(
            snap,
            [
                SimulationEvent("fire", {"shooter": 0, "target": 1, "weapon": "m256", "ammo": "m829a3"}),
                SimulationEvent("hit", {"shooter": 0, "target": 1, "part": 1, "penetrated": True}),
                SimulationEvent("destroyed", {"armor_idx": 1, "cause": "health"}),
            ],
        )

        # 首次击毁 → first_blood 对话激活 + 暂停
        assert engine.has_active_dialogue
        assert engine.is_paused
        assert engine._active_tree_id == "first_blood"

        # 推进到 choice
        engine.advance_dialogue(None)
        # 玩家选择「重组」（index 1）→ HOLD 指令
        engine.on_player_choice("first_blood", 1)
        drained = loop.commands.drain()
        assert len(drained) == 1
        assert drained[0].type == CommandType.HOLD
        assert drained[0].params["formation"] == "consolidate"

        # 对话推进到 regroup（text）→ 结束
        engine.on_player_choice("first_blood", 0)
        assert not engine.has_active_dialogue
        assert not engine.is_paused

        # ---- 事件记录：SQLite 已写入全部事件 ----
        events = recorder.timeline()
        types = [e.event_type for e in events]
        assert "fire" in types and "hit" in types and "destroyed" in types
        # 击杀归属：destroyed 的 source 应回填为 M1A2（最近命中者）
        destroyed = [e for e in events if e.event_type == "destroyed"]
        assert destroyed[0].source_id == 0

        # ---- 简报生成 ----
        briefing = recorder.generate_briefing()
        assert briefing.enemies_destroyed == 1
        assert briefing.total_shots_fired >= 1
        assert briefing.outcome == "stalemate"  # 1/2 敌未消灭
        assert any("首次击毁" in h for h in briefing.highlights)
        assert "战斗简报" in briefing.to_text()

        recorder.close()

    def test_conditional_option_visible_with_scenario_flag(
        self, story_files_exist: None
    ) -> None:
        """场景标志位 → 条件选项可见（docs §2.3；2026-09-11 起引擎侧过滤）。

        `main_window._load_story` 对含无人机侦察的场景设置 `has_scout_report`；
        此处复现该装配顺序：先 set_flag，再驱动首帧触发器。
        """
        loop, engine, recorder = make_world()
        engine.load_dialogues(DIALOGUES_DIR)
        engine.load_triggers(TRIGGERS_DIR)
        engine.set_flag("has_scout_report")  # 场景初始情报（与 prologue 文案同源）

        loop.step(1.0)
        pump_events(loop, engine)
        engine.advance_dialogue(None)
        choice = engine._build_state().active_dialogue
        assert choice is not None
        assert len(choice.options) == 3
        assert choice.option_indices == [0, 1, 2]

        # 选「等待更多侦察情报」（可见索引 2 = 原始索引 2）→ 无指令 + cautious_commander
        engine.on_player_choice("prologue_001", 2)
        assert len(loop.commands.drain()) == 0
        assert engine.has_flag("cautious_commander")

        recorder.close()

    def test_command_consumed_by_simulation(self, story_files_exist: None) -> None:
        """验证玩家指令确实被模拟层消费（产生 command 事件）。"""
        loop, engine, recorder = make_world()
        engine.load_dialogues(DIALOGUES_DIR)
        engine.load_triggers(TRIGGERS_DIR)

        # 首帧 → 开场对话 → 玩家选择（HOLD 入队）
        loop.step(1.0)
        pump_events(loop, engine)
        engine.advance_dialogue(None)
        engine.on_player_choice("prologue_001", 0)
        assert len(loop.commands) == 1  # HOLD 已入队待消费

        # 模拟层下一 Tick：消费 HOLD 指令 → emit "command" 事件
        # （对话未结束 → sim_paused=True → 只消费指令不推进物理）
        assert loop.sim_paused is True
        loop.step(1.0)
        drained = loop.events.drain()
        command_events = [e for e in drained if e.kind == "command"]
        assert len(command_events) == 1
        assert command_events[0].data["type"] == "HOLD"
        assert command_events[0].data["target"] == "7th_armor"
        # 暂停期间时钟不推进
        assert loop.state.sim_time == 1.0

        recorder.close()

    def test_skip_dialogue(self, story_files_exist: None) -> None:
        """跳过对话：直接结束且不产生指令。"""
        loop, engine, recorder = make_world()
        engine.load_dialogues(DIALOGUES_DIR)
        engine.load_triggers(TRIGGERS_DIR)

        loop.step(1.0)
        pump_events(loop, engine)
        assert engine.has_active_dialogue

        engine.skip_active_dialogue()
        assert not engine.has_active_dialogue
        assert not engine.is_paused
        assert len(loop.commands.drain()) == 0  # 跳过不产生指令

        recorder.close()

    def test_reset_for_new_mission(self, story_files_exist: None) -> None:
        """任务重置：flags/对话/事件全部清空，可重新开始。"""
        loop, engine, recorder = make_world()
        engine.load_dialogues(DIALOGUES_DIR)
        engine.load_triggers(TRIGGERS_DIR)

        loop.step(1.0)
        pump_events(loop, engine)
        engine.advance_dialogue(None)
        engine.on_player_choice("prologue_001", 0)
        assert engine.has_flag("defensive_posture")

        engine.reset()
        assert not engine.flags
        assert not engine.has_active_dialogue
        assert not engine.is_paused
        assert len(recorder.all_events()) == 0  # SQLite 已清空

        recorder.close()
