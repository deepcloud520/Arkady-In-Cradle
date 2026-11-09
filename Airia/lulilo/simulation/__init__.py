"""engine.simulation 层：Tick 主循环 / CommandQueue / 状态存取 / 存档（无头模式）—— Phase 3。

这是引擎核心的入口：上层（render / narrative）通过它消费状态，但引擎不感知上层。

模块：
    commands.py     CommandQueue（玩家指令注入，线程安全；AGENTS.md 3.7）
    state.py        SimulationState / SimulationSnapshot（Simulation.get_state()）
    loop.py         SimulationLoop（Tick 主循环 + 事件驱动内循环 + 多分辨率 Tick）+ EventBus
    serialization.py 存档/读档（msgpack + numpy bytes，无损往返）
"""

from Airia.lulilo.simulation.commands import Command, CommandQueue, CommandType
from Airia.lulilo.simulation.loop import (
    TICK_AIR,
    TICK_BULLET,
    TICK_GROUND,
    EventBus,
    SimulationEvent,
    SimulationLoop,
)
from Airia.lulilo.simulation.serialization import (
    BattlefieldSave,
    deserialize_battlefield,
    load_from_file,
    save_to_file,
    serialize_battlefield,
)
from Airia.lulilo.simulation.state import SimulationSnapshot, SimulationState

__all__ = [
    "BattlefieldSave",
    "Command",
    "CommandQueue",
    "CommandType",
    "EventBus",
    "SimulationEvent",
    "SimulationLoop",
    "SimulationSnapshot",
    "SimulationState",
    "TICK_AIR",
    "TICK_BULLET",
    "TICK_GROUND",
    "deserialize_battlefield",
    "load_from_file",
    "save_to_file",
    "serialize_battlefield",
]
