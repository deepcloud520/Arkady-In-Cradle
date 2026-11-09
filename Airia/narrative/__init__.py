"""Airia.narrative：独立叙事层（AGENTS.md Phase 4）。

消费 engine 的 simulation 状态（SimulationSnapshot + EventBus），
驱动剧情与对话；不嵌入无头引擎核心，不控制模拟流程
（通过 CommandQueue 间接交互）。

模块：
    dialogue.py   对话树加载/播放（DialogueEngine，YAML + Pydantic）
    triggers.py   条件触发器（TriggerSystem + ConditionEvaluator）
    commands.py   玩家选择 → Command（CommandFactory，纯函数）
    recorder.py   战斗事件记录 + 简报生成（CombatEventRecorder + BriefingGenerator）
    engine.py     中央调度器（NarrativeEngine）
"""

from Airia.narrative.commands import CommandFactory
from Airia.narrative.dialogue import (
    ChoiceOptionConfig,
    DialogueEngine,
    DialogueNodeConfig,
    DialogueState,
    DialogueTreeConfig,
)
from Airia.narrative.engine import NarrativeEngine, NarrativeState
from Airia.narrative.recorder import (
    BriefingGenerator,
    CombatBriefing,
    CombatEventRecorder,
    CombatEventRecord,
    RingBuffer,
    UnitStatsReport,
)
from Airia.narrative.triggers import (
    ConditionEvaluator,
    TriggerAction,
    TriggerCondition,
    TriggerRule,
    TriggerSystem,
)

__all__ = [
    "BriefingGenerator",
    "ChoiceOptionConfig",
    "CombatBriefing",
    "CombatEventRecorder",
    "CombatEventRecord",
    "CommandFactory",
    "ConditionEvaluator",
    "DialogueEngine",
    "DialogueNodeConfig",
    "DialogueState",
    "DialogueTreeConfig",
    "NarrativeEngine",
    "NarrativeState",
    "RingBuffer",
    "TriggerAction",
    "TriggerCondition",
    "TriggerRule",
    "TriggerSystem",
    "UnitStatsReport",
]
