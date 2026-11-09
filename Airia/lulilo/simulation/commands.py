"""玩家指令注入（AGENTS.md Phase 3：simulation/commands.py）。

对应 AGENTS.md 3.7「叙事层与模拟层的交互」：
    - 玩家在对话中选择 → 生成 Command 对象 → 推入 CommandQueue
    - 模拟的每个 Tick 检查队列，执行挂起的玩家命令
    - 叙事层永远不直接修改模拟状态，只通过 CommandQueue 下发意图

CommandQueue 设计：
    - 线程安全（Lock 保护，叙事层/UI 线程与模拟线程解耦）
    - drain() 一次性取走全部挂起命令（Tick 间隙执行，避免逐条加锁）
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class CommandType(IntEnum):
    """玩家可下发的指令类型（AGENTS.md 3.7：redeploy / strike / hold）。"""
    REDEPLOY = 0   # 重新部署 / 转移阵地
    STRIKE = 1     # 火力打击（指定目标）
    HOLD = 2       # 固守（停止机动）
    RESUPPLY = 3   # 申请补给
    DISENGAGE = 4  # 脱离接触


@dataclass
class Command:
    """一条玩家指令（叙事层 → 模拟层）。"""

    type: CommandType
    target_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Command({self.type.name}, target={self.target_id}, params={self.params})"


class CommandQueue:
    """线程安全的指令队列。

    用法（AGENTS.md 3.7）:
        叙事层:  queue.push(Command(CommandType.STRIKE, "t-90-03", {...}))
        模拟层:  for cmd in queue.drain(): simulation.execute(cmd)
    """

    def __init__(self) -> None:
        self._queue: list[Command] = []
        self._lock = threading.Lock()

    def push(self, cmd: Command) -> None:
        """入队一条指令（线程安全）。"""
        with self._lock:
            self._queue.append(cmd)

    def push_many(self, cmds: list[Command]) -> None:
        """批量入队（线程安全）。"""
        with self._lock:
            self._queue.extend(cmds)

    def drain(self) -> list[Command]:
        """取走并清空全部挂起指令（模拟 Tick 间隙调用）。"""
        with self._lock:
            cmds, self._queue = self._queue, []
        return cmds

    def peek(self) -> list[Command]:
        """查看当前挂起指令（不消费）。"""
        with self._lock:
            return list(self._queue)

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
