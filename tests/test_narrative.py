"""Phase 4 叙事层管线单元测试（AGENTS.md 4.5：单元测试不产生文件产物）。

覆盖：
    - dialogue.py：加载/激活/推进/分支/条件选项
    - triggers.py：ConditionEvaluator 安全求值 + 触发器匹配/优先级/once/cooldown
    - commands.py：CommandFactory 转换
    - recorder.py：事件写入/查询/简报生成/单位统计（SQLite 用 :memory:）
    - engine.py：tick 流程/触发器激活对话/玩家选择/指令入队
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from Airia.lulilo.simulation.commands import Command, CommandQueue, CommandType
from Airia.lulilo.simulation.loop import SimulationEvent
from Airia.lulilo.simulation.state import SimulationSnapshot

from Airia.narrative.commands import CommandFactory
from Airia.narrative.dialogue import DialogueEngine, DialogueTreeConfig
from Airia.narrative.engine import NarrativeEngine
from Airia.narrative.recorder import (
    BriefingGenerator,
    CombatEventRecorder,
    CombatEventRecord,
    RingBuffer,
)
from Airia.narrative.triggers import (
    ConditionEvaluator,
    TriggerAction,
    TriggerCondition,
    TriggerRule,
    TriggerSystem,
)


# ---------------------------------------------------------------------------
# 测试数据（YAML 片段）
# ---------------------------------------------------------------------------

PROLOGUE_YAML = """
dialogue_id: "prologue_001"
title: "序章"
nodes:
  - id: start
    type: text
    speaker: "系统"
    text: "抵达集结点。"
    next: choice_deploy
  - id: choice_deploy
    type: choice
    speaker: "指挥官"
    text: "如何部署？"
    options:
      - text: "固守"
        next: hold_node
        action: { type: HOLD, target_id: "company", params: { formation: "line" } }
      - text: "出击"
        next: attack_node
        action: { type: REDEPLOY, target_id: "company", params: { waypoint: [1, 2] } }
      - text: "等情报"
        next: hold_node
        condition: "flags.contains('has_scout')"
  - id: hold_node
    type: text
    speaker: "副官"
    text: "明白，固守。"
    next: null
    set_flags: ["defensive"]
  - id: attack_node
    type: text
    speaker: "副官"
    text: "楔形推进！"
    next: null
    set_flags: ["offensive"]
"""

BRANCH_YAML = """
dialogue_id: "branch_001"
title: "分支测试"
nodes:
  - id: entry
    type: branch
    conditions: ["flags.contains('scout')"]
    next_true: scouted
    next_false: blind
  - id: scouted
    type: text
    speaker: "A"
    text: "有侦察。"
    next: null
  - id: blind
    type: text
    speaker: "A"
    text: "无侦察。"
    next: null
"""

TRIGGER_YAML = """
triggers:
  - trigger_id: "first_blood"
    description: "首次击毁"
    condition:
      type: event_match
      event_kind: "destroyed"
      filter: { side: "enemy" }
      count: 1
    activate:
      dialogue: "prologue_001"
      priority: 5
      once: true
      pause_sim: true
  - trigger_id: "half_health"
    description: "血量低"
    condition:
      type: state_check
      check: "sim_time_greater_than"
      threshold: 100.0
    activate:
      dialogue: "branch_001"
      priority: 3
      once: false
      cooldown: 10.0
"""


def _tree_from_yaml(text: str) -> DialogueTreeConfig:
    """YAML 字符串 → DialogueTreeConfig（免临时文件，沙箱环境可跑）。"""
    return DialogueTreeConfig.model_validate(yaml.safe_load(text))


# ---------------------------------------------------------------------------
# dialogue.py
# ---------------------------------------------------------------------------


class TestDialogueEngine:
    def test_load_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "prologue.yaml"
        f.write_text(PROLOGUE_YAML, encoding="utf-8")
        eng = DialogueEngine()
        tree = eng.load(f)
        assert isinstance(tree, DialogueTreeConfig)
        assert tree.dialogue_id == "prologue_001"
        assert len(tree.nodes) == 4
        assert eng.has("prologue_001")

    def test_activate_text_then_choice(self) -> None:
        eng = DialogueEngine()
        eng.register(_tree_from_yaml(PROLOGUE_YAML))

        st = eng.activate("prologue_001")
        assert st.speaker == "系统"
        assert not st.is_finished

        # 自动前进到 choice 节点（无 flags → 条件选项「等情报」不可见）
        st2 = eng.advance(None)
        assert st2.speaker == "指挥官"
        assert st2.options == ["固守", "出击"]
        assert st2.option_indices == [0, 1]

        # 有 has_scout 标志 → 第三个条件选项可见
        eng.activate("prologue_001")
        st3 = eng.advance(None, {"flags": {"has_scout"}})
        assert st3.options == ["固守", "出击", "等情报"]
        assert st3.option_indices == [0, 1, 2]

    def test_choice_advance_with_action(self) -> None:
        eng = DialogueEngine()
        eng.register(_tree_from_yaml(PROLOGUE_YAML))
        eng.activate("prologue_001")
        eng.advance(None)  # → choice

        actions: list[dict] = []
        ctx = {"pending_actions": actions, "flags": set()}
        st = eng.advance(0, ctx)  # 选择「固守」
        assert st.speaker == "副官"
        assert len(actions) == 1
        assert actions[0]["type"] == "HOLD"

        # 再 advance → 结束
        st2 = eng.advance(None, ctx)
        assert st2.is_finished

    def test_branch_node(self) -> None:
        eng = DialogueEngine()
        eng.register(_tree_from_yaml(BRANCH_YAML))

        # 无 scout 标志 → blind
        eng.activate("branch_001")
        st = eng.advance(None, {"flags": set()})
        assert st.text == "无侦察。"

        # 有 scout 标志 → scouted
        eng.activate("branch_001")
        st2 = eng.advance(None, {"flags": {"scout"}})
        assert st2.text == "有侦察。"

    def test_condition_option_visibility(self) -> None:
        """条件选项：只有满足条件才显示（docs §2.3）。

        2026-09-11 起由**引擎**过滤（原骨架期把过滤推给 UI，但 UI 拿不到
        eval_ctx/flags，实际等于不过滤 —— 见 docs/story-architecture-review.md §三.2）。

        用 `DialogueEngine.register()` 内存构造：免临时文件（沙箱可跑）。
        """
        eng = DialogueEngine()
        eng.register(_tree_from_yaml(PROLOGUE_YAML))

        # 无 flags → 条件选项隐藏
        eng.activate("prologue_001")
        st = eng.advance(None)
        assert "等情报" not in st.options
        assert st.options == ["固守", "出击"]
        assert st.option_indices == [0, 1]

        # 有 flags → 条件选项显示
        eng.activate("prologue_001")
        st2 = eng.advance(None, {"flags": {"has_scout"}})
        assert "等情报" in st2.options
        assert st2.option_indices == [0, 1, 2]

    def test_visible_index_maps_to_original_option(self) -> None:
        """回归：隐藏中间选项后，可见索引必须映射回原始索引。

        docs/架构评审报告.md §十 建议 10 明确警告：过滤后不保留映射会「选错分支」。
        """
        eng = DialogueEngine()
        eng.register(_tree_from_yaml("""
dialogue_id: "mapping"
nodes:
  - id: start
    type: choice
    speaker: "指挥官"
    text: "选？"
    options:
      - text: "A"
        next: node_a
      - text: "B(条件)"
        next: node_b
        condition: "flags.contains('unlock_b')"
      - text: "C"
        next: node_c
  - id: node_a
    type: text
    text: "走了 A"
  - id: node_b
    type: text
    text: "走了 B"
  - id: node_c
    type: text
    text: "走了 C"
"""))

        # 无 unlock_b → 可见列表 = [A, C]
        st = eng.activate("mapping", {"flags": set()})
        assert st.options == ["A", "C"]
        assert st.option_indices == [0, 2]

        # 选「可见索引 1」应到 C（原始索引 2）；若丢掉映射会错走 B
        st2 = eng.advance(1, {"flags": set()})
        assert st2.text == "走了 C"

    def test_invalid_next_reference(self, tmp_path: Path) -> None:
        yaml = """
dialogue_id: "bad"
nodes:
  - id: a
    type: text
    text: "hi"
    next: "missing"
"""
        f = tmp_path / "bad.yaml"
        f.write_text(yaml, encoding="utf-8")
        eng = DialogueEngine()
        eng.load(f)
        eng.activate("bad")
        with pytest.raises(ValueError):
            eng.advance(None)

    def test_duplicate_node_id_rejected(self, tmp_path: Path) -> None:
        yaml = """
dialogue_id: "dup"
nodes:
  - id: a
    type: text
    text: "1"
  - id: a
    type: text
    text: "2"
"""
        f = tmp_path / "dup.yaml"
        f.write_text(yaml, encoding="utf-8")
        eng = DialogueEngine()
        eng.load(f)
        with pytest.raises(ValueError):
            eng.activate("dup")  # node_map() 检测重复


# ---------------------------------------------------------------------------
# triggers.py
# ---------------------------------------------------------------------------


class TestConditionEvaluator:
    def test_flag_contains(self) -> None:
        assert ConditionEvaluator.evaluate("flags.contains('scout')", {"flags": {"scout"}})
        assert not ConditionEvaluator.evaluate("flags.contains('scout')", {"flags": set()})

    def test_comparison(self) -> None:
        assert ConditionEvaluator.evaluate("sim_time > 100", {"sim_time": 150})
        assert not ConditionEvaluator.evaluate("sim_time > 100", {"sim_time": 50})

    def test_composite(self) -> None:
        ctx = {"flags": {"a"}, "sim_time": 200}
        assert ConditionEvaluator.evaluate(
            "flags.contains('a') and sim_time > 100", ctx
        )
        assert not ConditionEvaluator.evaluate(
            "flags.contains('b') or sim_time < 100", ctx
        )

    def test_empty_is_true(self) -> None:
        assert ConditionEvaluator.evaluate("", {})

    def test_rejects_arbitrary_names(self) -> None:
        # 非白名单名称 → False（不抛异常）
        assert not ConditionEvaluator.evaluate("__import__('os')", {})
        assert not ConditionEvaluator.evaluate("evil_fn()", {})

    def test_rejects_attribute_chain(self) -> None:
        assert not ConditionEvaluator.evaluate("flags.clear()", {"flags": set()})

    def test_invalid_expr_returns_false(self) -> None:
        assert not ConditionEvaluator.evaluate("1 +", {})


class TestTriggerSystem:
    def test_load_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "t.yaml"
        f.write_text(TRIGGER_YAML, encoding="utf-8")
        ts = TriggerSystem()
        assert ts.load(f) == 2

    def test_event_match_once(self) -> None:
        ts = TriggerSystem()
        ts.add_rule(
            TriggerRule(
                trigger_id="first_blood",
                condition=TriggerCondition(
                    type="event_match", event_kind="destroyed", count=1
                ),
                activate=TriggerAction(dialogue="d1", priority=5, once=True),
            )
        )
        ev = [SimulationEvent("destroyed", {"armor_idx": 3, "cause": "hit"})]
        fired = ts.evaluate(ev)
        assert len(fired) == 1
        assert fired[0].dialogue == "d1"

        # once：第二次不再触发
        assert ts.evaluate(ev) == []

    def test_priority_sort(self) -> None:
        ts = TriggerSystem()
        ts.add_rule(
            TriggerRule(
                trigger_id="low",
                condition=TriggerCondition(type="event_match", event_kind="hit"),
                activate=TriggerAction(dialogue="low", priority=1),
            )
        )
        ts.add_rule(
            TriggerRule(
                trigger_id="high",
                condition=TriggerCondition(type="event_match", event_kind="hit"),
                activate=TriggerAction(dialogue="high", priority=9),
            )
        )
        fired = ts.evaluate([SimulationEvent("hit", {"target": 1})])
        assert [a.dialogue for a in fired] == ["high", "low"]

    def test_cooldown(self) -> None:
        ts = TriggerSystem()
        ts.add_rule(
            TriggerRule(
                trigger_id="cd",
                condition=TriggerCondition(type="event_match", event_kind="hit"),
                activate=TriggerAction(dialogue="d", priority=1, once=False, cooldown=10.0),
            )
        )
        assert ts.evaluate([SimulationEvent("hit", {})], sim_time=0.0)
        # 冷却期内
        assert not ts.evaluate([SimulationEvent("hit", {})], sim_time=5.0)
        # 冷却结束
        assert ts.evaluate([SimulationEvent("hit", {})], sim_time=11.0)

    def test_state_check(self) -> None:
        ts = TriggerSystem()
        ts.add_rule(
            TriggerRule(
                trigger_id="time",
                condition=TriggerCondition(
                    type="state_check", check="sim_time_greater_than", threshold=100.0
                ),
                activate=TriggerAction(dialogue="d"),
            )
        )
        assert ts.evaluate([], sim_time=200.0)
        assert not ts.evaluate([], sim_time=50.0)


# ---------------------------------------------------------------------------
# commands.py
# ---------------------------------------------------------------------------


class TestCommandFactory:
    def test_from_action_valid(self) -> None:
        cmd = CommandFactory.from_action(
            {"type": "HOLD", "target_id": "company", "params": {"formation": "line"}}
        )
        assert cmd is not None
        assert cmd.type == CommandType.HOLD
        assert cmd.target_id == "company"
        assert cmd.params == {"formation": "line"}

    def test_from_action_unknown_type(self) -> None:
        assert CommandFactory.from_action({"type": "NOPE", "target_id": "x"}) is None

    def test_from_action_none(self) -> None:
        assert CommandFactory.from_action(None) is None
        assert CommandFactory.from_action({}) is None

    def test_from_actions_batch(self) -> None:
        cmds = CommandFactory.from_actions(
            [
                {"type": "STRIKE", "target_id": "t1"},
                {"type": "BAD", "target_id": "x"},
                {"type": "HOLD", "target_id": "t2"},
            ]
        )
        assert len(cmds) == 2
        assert cmds[0].type == CommandType.STRIKE


# ---------------------------------------------------------------------------
# recorder.py
# ---------------------------------------------------------------------------


class TestRecorder:
    def _snapshot(self, sim_time: float = 10.0, tick: int = 5) -> SimulationSnapshot:
        return SimulationSnapshot(
            sim_time=sim_time,
            tick_count=tick,
            pools={},
            logistics={},
            extra={},
        )

    def test_ring_buffer_capacity(self) -> None:
        rb = RingBuffer(capacity=3)
        for i in range(5):
            rb.append(CombatEventRecord(sim_time=float(i), event_type="fire"))
        assert len(rb) == 3
        assert rb.peek()[0].sim_time == 2.0  # 最旧的被丢弃

    def test_on_events_and_query(self) -> None:
        rec = CombatEventRecorder(
            db_path=":memory:",
            unit_registry={0: "M1A2 #0", 1: "T-90 #1"},
            side_map={0: "ally", 1: "enemy"},
        )
        rec.on_events(
            [
                SimulationEvent("fire", {"shooter": 0, "target": 1, "weapon": "m256", "ammo": "m829a3"}),
                SimulationEvent("hit", {"shooter": 0, "target": 1, "part": 1, "penetrated": True, "weapon": "m256", "ammo": "m829a3"}),
                SimulationEvent("destroyed", {"armor_idx": 1, "cause": "health"}),
            ],
            self._snapshot(),
        )
        rec.flush()

        events = rec.timeline()
        assert len(events) == 3
        assert events[0].source_name == "M1A2 #0"
        assert events[0].source_side == "ally"
        assert events[1].hit_part == "TURRET"
        assert events[1].penetration is True
        assert events[2].event_type == "destroyed"
        assert events[2].target_id == 1

    def test_briefing_generation(self) -> None:
        rec = CombatEventRecorder(
            db_path=":memory:",
            unit_registry={0: "M1A2", 1: "T-90"},
            side_map={0: "ally", 1: "enemy"},
            mission_id="test_01",
        )
        snap = self._snapshot(sim_time=45.0, tick=45)
        rec.on_events(
            [
                SimulationEvent("fire", {"shooter": 0, "target": 1, "weapon": "m256", "ammo": "m829a3"}),
                SimulationEvent("hit", {"shooter": 0, "target": 1, "part": 4, "penetrated": True}),
                SimulationEvent("destroyed", {"armor_idx": 1, "cause": "cookoff"}),
            ],
            snap,
        )
        rec.on_events(
            [SimulationEvent("fire", {"shooter": 1, "target": 0, "weapon": "2a46m", "ammo": "3bm42"})],
            snap,
        )

        briefing = rec.generate_briefing()
        assert briefing.mission_id == "test_01"
        assert briefing.enemies_destroyed == 1
        assert briefing.enemies_initial >= 1
        assert briefing.total_shots_fired == 2
        assert briefing.outcome == "victory"
        # 关键事件高亮（首次命中/首次击毁/殉爆）
        assert any("首次命中" in h for h in briefing.highlights)
        assert any("殉爆" in h for h in briefing.highlights)

        # 时间线
        assert len(briefing.timeline) >= 4

        # 单位战报
        by_idx = {r.armor_idx: r for r in briefing.unit_reports}
        assert by_idx[0].shots_fired == 1
        assert by_idx[0].kills == 1
        assert by_idx[1].destroyed is True

        # 文本导出
        text = briefing.to_text()
        assert "战斗简报" in text

    def test_friendly_fire_detection(self) -> None:
        gen = BriefingGenerator()
        rec = CombatEventRecorder(
            db_path=":memory:",
            unit_registry={0: "M1A2", 1: "M2"},
            side_map={0: "ally", 1: "ally"},
        )
        rec.on_events(
            [SimulationEvent("fire", {"shooter": 0, "target": 1})],
            self._snapshot(),
        )
        rec.flush()
        briefing = rec.generate_briefing()
        assert len(briefing.friendly_fire_incidents) == 1

    def test_observed_by_player_filter(self) -> None:
        gen = BriefingGenerator()
        rec = CombatEventRecorder(
            db_path=":memory:",
            unit_registry={0: "A", 1: "B"},
            side_map={0: "ally", 1: "enemy"},
        )
        rec.on_events(
            [SimulationEvent("destroyed", {"armor_idx": 1, "cause": "hit"})],
            self._snapshot(),
        )
        rec.flush()
        # 全部标记为已观测（默认 observed_by_player=1）
        briefing = rec.generate_briefing()
        assert briefing.enemies_destroyed == 1


# ---------------------------------------------------------------------------
# engine.py
# ---------------------------------------------------------------------------


class TestNarrativeEngine:
    def _make_engine(self, tmp_path: Path) -> tuple[NarrativeEngine, CommandQueue]:
        q = CommandQueue()
        eng = NarrativeEngine(command_queue=q)
        p = tmp_path / "p.yaml"
        p.write_text(PROLOGUE_YAML, encoding="utf-8")
        eng.dialogue_engine.load(p)
        return eng, q

    def test_tick_activates_trigger_dialogue(self, tmp_path: Path) -> None:
        eng, _ = self._make_engine(tmp_path)
        eng.trigger_system.add_rule(
            TriggerRule(
                trigger_id="on_hit",
                condition=TriggerCondition(type="event_match", event_kind="hit"),
                activate=TriggerAction(dialogue="prologue_001", priority=5),
            )
        )

        snap = SimulationSnapshot(sim_time=1.0, tick_count=1, pools={}, logistics={}, extra={})
        state = eng.tick(snap, [SimulationEvent("hit", {"target": 1})])

        assert state.active_dialogue is not None
        assert state.active_dialogue.speaker == "系统"
        assert eng.is_paused

    def test_tick_first_frame_opening(self, tmp_path: Path) -> None:
        """首帧触发开场对话（on_first_tick 触发器）。"""
        eng, _ = self._make_engine(tmp_path)
        eng.trigger_system.add_rule(
            TriggerRule(
                trigger_id="start",
                condition=TriggerCondition(type="on_first_tick"),
                activate=TriggerAction(dialogue="prologue_001", priority=10),
            )
        )
        snap = SimulationSnapshot(sim_time=0.0, tick_count=0, pools={}, logistics={}, extra={})
        state = eng.tick(snap, [])
        assert state.active_dialogue is not None

    def test_player_choice_pushes_command(self, tmp_path: Path) -> None:
        eng, q = self._make_engine(tmp_path)
        snap = SimulationSnapshot(sim_time=0.0, tick_count=0, pools={}, logistics={}, extra={})

        # 直接激活对话
        eng._activate_dialogue("prologue_001", pause_sim=True)
        # 前进到 choice
        eng.on_player_choice("prologue_001", 0)  # 错误：choice 需要先 advance

        # 正确流程：激活 → advance → choice → on_player_choice
        eng.reset()
        eng.dialogue_engine.load(tmp_path / "p.yaml")
        eng._activate_dialogue("prologue_001", pause_sim=True)
        # 手动推进：tick 里不自动推进，这里调用 on_player_choice(None) 不适用。
        # 改用 DialogueEngine 直接驱动：
        from Airia.narrative.dialogue import DialogueEngine as DE

        de = DE()
        de.load(tmp_path / "p.yaml")
        de.activate("prologue_001")
        de.advance(None)  # → choice
        # 模拟玩家点击「固守」
        actions: list[dict] = []
        st = de.advance(0, {"pending_actions": actions, "flags": set()})
        assert st.speaker == "副官"
        cmds = CommandFactory.from_actions(actions)
        for cmd in cmds:
            q.push(cmd)

        drained = q.drain()
        assert len(drained) == 1
        assert drained[0].type == CommandType.HOLD

    def test_player_choice_via_engine(self, tmp_path: Path) -> None:
        """端到端：engine 激活 → 推进 → 玩家选择 → 指令入队。"""
        eng, q = self._make_engine(tmp_path)

        eng._activate_dialogue("prologue_001", pause_sim=True)
        # 直接调用 DialogueEngine.advance 到 choice
        eng.dialogue_engine.advance(None, eng._eval_ctx())
        # 玩家选择「出击」（index 1）
        state = eng.on_player_choice("prologue_001", 1)

        drained = q.drain()
        assert len(drained) == 1
        assert drained[0].type == CommandType.REDEPLOY
        assert drained[0].params["waypoint"] == [1, 2]

        # 对话应已推进到 attack_node（未结束）
        assert state.active_dialogue is not None
        assert state.active_dialogue.text == "楔形推进！"

    def test_dialogue_finish_resumes_sim(self, tmp_path: Path) -> None:
        eng, _ = self._make_engine(tmp_path)
        eng._activate_dialogue("prologue_001", pause_sim=True)
        eng.dialogue_engine.advance(None, eng._eval_ctx())  # → choice
        eng.on_player_choice("prologue_001", 0)  # → hold_node
        state = eng.on_player_choice("prologue_001", 0)  # 再推进 → 结束
        # 注意：第二次调用时对话树已到 hold_node（text），advance 到结束
        assert eng._active_dialogue is None
        assert not eng.is_paused

    def test_reset(self, tmp_path: Path) -> None:
        eng, _ = self._make_engine(tmp_path)
        eng.set_flag("scout")
        eng._activate_dialogue("prologue_001", pause_sim=True)
        eng.reset()
        assert not eng.flags
        assert not eng.has_active_dialogue
        assert not eng.is_paused
