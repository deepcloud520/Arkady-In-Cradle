"""玩家选择 → Command 对象（AGENTS.md Phase 4：narrative/commands.py）。

对应 docs/phase4-narrative-design.md §3.4：
    - CommandFactory.from_action(action) → Command
    - CommandFactory.from_node(node, choice_index) → Command

纯函数，无状态。映射 AGENTS.md 3.7：
    action: { type: HOLD, target_id: "...", params: {...} }
    → Command(CommandType.HOLD, target_id, params)
"""

from __future__ import annotations

from typing import Any

from Airia.lulilo.core.logging import get_logger
from Airia.lulilo.simulation.commands import Command, CommandType

logger = get_logger(__name__)

# 字符串 → CommandType 枚举的映射（YAML action.type 是字符串）
_TYPE_MAP: dict[str, CommandType] = {
    "REDEPLOY": CommandType.REDEPLOY,
    "STRIKE": CommandType.STRIKE,
    "HOLD": CommandType.HOLD,
    "RESUPPLY": CommandType.RESUPPLY,
    "DISENGAGE": CommandType.DISENGAGE,
    # 兼容小写 / 别名
    "redeploy": CommandType.REDEPLOY,
    "strike": CommandType.STRIKE,
    "hold": CommandType.HOLD,
    "resupply": CommandType.RESUPPLY,
    "disengage": CommandType.DISENGAGE,
}


class CommandFactory:
    """把对话 action dict 转换成 Command 对象。

    纯函数，无状态（docs §3.4）。
    """

    @staticmethod
    def from_action(action: dict[str, Any] | None) -> Command | None:
        """从 action dict 生成一条 Command。

        Args:
            action: YAML 中的 action 定义。
                形如: {type: "HOLD", target_id: "7th_armor", params: {...}}

        Returns:
            Command 或 None（action 为空 / type 非法时）。
        """
        if not action:
            return None

        raw_type = action.get("type")
        if not isinstance(raw_type, str):
            logger.warning("action 缺少 type 字段: %r", action)
            return None

        cmd_type = _TYPE_MAP.get(raw_type)
        if cmd_type is None:
            logger.warning("未知指令类型: %r（action=%r）", raw_type, action)
            return None

        return Command(
            type=cmd_type,
            target_id=str(action.get("target_id", "")),
            params=dict(action.get("params") or {}),
        )

    @staticmethod
    def from_node(node: Any, choice_index: int) -> Command | None:
        """从对话节点（choice 的某个选项）生成一条 Command。

        Args:
            node: DialogueNodeConfig（choice 节点）。
            choice_index: 玩家选择的选项索引。

        Returns:
            Command 或 None（该选项无 action 或非法）。
        """
        options = getattr(node, "options", None)
        if not options or not (0 <= choice_index < len(options)):
            return None
        action = getattr(options[choice_index], "action", None)
        return CommandFactory.from_action(action)

    @staticmethod
    def from_actions(pending_actions: list[dict[str, Any]]) -> list[Command]:
        """批量转换（advance 收集的 pending_actions → Command 列表）。

        Returns:
            list[Command]（跳过非法项）。
        """
        cmds: list[Command] = []
        for action in pending_actions:
            cmd = CommandFactory.from_action(action)
            if cmd is not None:
                cmds.append(cmd)
        return cmds
