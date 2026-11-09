"""对话树加载 / 播放（AGENTS.md Phase 4：narrative/dialogue.py）。

对应 docs/phase4-narrative-design.md §3.2：
    - DialogueTreeConfig / DialogueNodeConfig / ChoiceOptionConfig（Pydantic，YAML 加载）
    - DialogueState（dataclass，运行时状态，渲染层消费）
    - DialogueEngine（对话树状态机：激活 / 推进 / 分支 / 条件选项）

设计要点（AGENTS.md 4.2）：
    - 配置类全部继承 pydantic.BaseModel（YAML → Pydantic 加载管线）
    - 运行时状态用轻量 dataclass（非配置，不落盘）
    - 条件选项 / 条件分支由外部注入 eval_ctx（ConditionEvaluator，见 triggers.py）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from Airia.lulilo.core.logging import get_logger

logger = get_logger(__name__)

# 节点类型（docs §2.2）
NodeType = Literal["text", "choice", "branch", "action"]


# ---------------------------------------------------------------------------
# Pydantic 配置模型（YAML → 引擎）
# ---------------------------------------------------------------------------


class ChoiceOptionConfig(BaseModel):
    """对话选项（docs §2.3）。"""

    text: str
    next: str
    action: dict[str, Any] | None = None      # 选择副作用 → Command（可选）
    condition: str | None = None              # 可见条件表达式（可选）


class DialogueNodeConfig(BaseModel):
    """对话树中的一个节点（docs §2.2）。"""

    id: str
    type: NodeType
    speaker: str = ""
    text: str = ""
    next: str | None = None                    # text/action 节点前进目标（None=结束）
    options: list[ChoiceOptionConfig] = Field(default_factory=list)
    action: dict[str, Any] | None = None       # action 节点要执行的指令
    set_flags: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)  # branch 节点条件列表
    next_true: str | None = None               # branch 节点：条件全真 → next_true
    next_false: str | None = None              # branch 节点：否则 → next_false


class DialogueTreeConfig(BaseModel):
    """一个完整的对话树（从 YAML 加载，docs §2.1）。"""

    dialogue_id: str
    title: str = ""
    nodes: list[DialogueNodeConfig]

    def node_map(self) -> dict[str, DialogueNodeConfig]:
        """节点列表 → dict（id → 节点），重复 id 报错。"""
        m: dict[str, DialogueNodeConfig] = {}
        for node in self.nodes:
            if node.id in m:
                raise ValueError(
                    f"对话树 {self.dialogue_id} 节点 id 重复: {node.id}"
                )
            m[node.id] = node
        return m


# ---------------------------------------------------------------------------
# 运行时状态（dataclass，渲染层消费）
# ---------------------------------------------------------------------------


@dataclass
class DialogueState:
    """当前对话的运行时状态（docs §3.2，渲染层消费）。"""

    dialogue_id: str
    speaker: str = ""
    text: str = ""
    options: list[str] = field(default_factory=list)  # 仅 choice 节点有（已按条件过滤）
    # 可见选项索引 → 原始选项索引（docs §2.3）。UI 只看到过滤后的列表，
    # 引擎必须靠本表映射回原始索引，否则过滤后选错分支
    # （docs/架构评审报告.md §十 建议 10 明确警告此坑）。
    option_indices: list[int] = field(default_factory=list)
    history: list[str] = field(default_factory=list)  # 对话历史（滚动显示）
    is_finished: bool = False                         # 对话是否结束


# ---------------------------------------------------------------------------
# 对话树状态机
# ---------------------------------------------------------------------------


class DialogueEngine:
    """对话树状态机。

    职责（docs §3.2）：
        - 加载 YAML 对话树 → DialogueTreeConfig
        - 激活一棵树：activate(dialogue_id) → DialogueState
        - 推进状态：advance(option_index, eval_ctx) → DialogueState
        - 条件求值：branch 节点 / 条件选项依赖外部注入的 eval_ctx

    eval_ctx 约定（供 ConditionEvaluator 求值）：
        - flags: set[str]            全局标志位
        - state: Any                 模拟快照（SimulationSnapshot）
        - event: dict | None         最近事件摘要
        - sim_time: float            模拟时间
    """

    def __init__(self) -> None:
        self._trees: dict[str, DialogueTreeConfig] = {}
        self._active: DialogueTreeConfig | None = None
        self._nodes: dict[str, DialogueNodeConfig] = {}
        self._cur: DialogueNodeConfig | None = None
        self._history: list[str] = []
        self._finished: bool = False

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def register(self, tree: DialogueTreeConfig) -> DialogueTreeConfig:
        """注册一棵已构造的对话树（不做文件 IO）。

        供两种场景使用：
            - `load()` 内部：YAML → Pydantic 后入库；
            - 测试/工具：直接由 dict 构造 `DialogueTreeConfig` 后注入，
              免去临时文件（沙箱/无文件系统环境可用）。

        Args:
            tree: 已校验的对话树配置。

        Returns:
            同一 tree（便于链式使用）。
        """
        self._trees[tree.dialogue_id] = tree
        return tree

    def load(self, path: Path | str) -> DialogueTreeConfig:
        """加载一棵对话树（YAML → Pydantic 校验）。

        Args:
            path: YAML 文件路径。

        Returns:
            DialogueTreeConfig（同时注册到引擎）。
        """
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        tree = DialogueTreeConfig.model_validate(raw)
        self.register(tree)
        logger.info("对话树已加载: %s (%d 节点)", tree.dialogue_id, len(tree.nodes))
        return tree

    def load_all(self, dir_path: Path | str) -> int:
        """加载目录下全部 *.yaml / *.yml 对话树。

        Returns:
            加载的对话树数量。
        """
        d = Path(dir_path)
        count = 0
        for p in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            self.load(p)
            count += 1
        return count

    def has(self, dialogue_id: str) -> bool:
        return dialogue_id in self._trees

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    def activate(
        self,
        dialogue_id: str,
        eval_ctx: dict[str, Any] | None = None,
    ) -> DialogueState:
        """激活一棵对话树，从入口节点开始播放。

        Args:
            dialogue_id: 对话树 id。
            eval_ctx: 条件求值上下文（入口节点若为 choice，其条件选项需要）。

        Returns:
            DialogueState（首个节点的展示状态）。
        """
        if dialogue_id not in self._trees:
            raise KeyError(f"对话树不存在: {dialogue_id}")
        self._active = self._trees[dialogue_id]
        self._nodes = self._active.node_map()
        self._history = []
        self._finished = False

        # 入口 = 第一个节点
        first = self._active.nodes[0]
        self._cur = first
        return self._enter(first, eval_ctx)

    def advance(
        self,
        option_index: int | None = None,
        eval_ctx: dict[str, Any] | None = None,
    ) -> DialogueState:
        """推进到下一个节点。

        Args:
            option_index: choice 节点中玩家选择的选项索引；None 表示自动前进
                （text/action/branch 节点）。
            eval_ctx: 条件求值上下文（branch / 条件选项需要）。

        Returns:
            新的 DialogueState（若对话结束，is_finished=True）。
        """
        if self._cur is None:
            return self._finish_state()

        ctx = eval_ctx or {}

        # 根据当前节点类型决定下一步目标节点 id
        nxt: str | None = None
        node = self._cur

        if node.type == "choice":
            if option_index is None:
                raise ValueError("choice 节点必须提供 option_index")
            # option_index 是「可见索引」：先按条件过滤，再映射回原始索引
            visible = self._visible_option_indices(node, ctx)
            if not (0 <= option_index < len(visible)):
                raise ValueError(
                    f"选项索引越界: {option_index} (0~{len(visible) - 1})"
                )
            opt = node.options[visible[option_index]]
            nxt = opt.next
            self._apply_action(opt.action, ctx)
        elif node.type == "branch":
            # 所有 conditions 全真 → next_true，否则 next_false
            ok = all(self._eval_cond(c, ctx) for c in node.conditions)
            nxt = node.next_true if ok else node.next_false
        elif node.type == "action":
            self._apply_action(node.action, ctx)
            nxt = node.next
        else:  # text
            nxt = node.next

        # 记录对话历史
        if node.speaker and node.text:
            self._history.append(f"[{node.speaker}] {node.text}")

        # 应用 set_flags（docs §2.1：节点被进入时设置标志位）
        # 1) 离开当前节点时，若其带 set_flags 也应用（兼容旧语义）
        if node.set_flags:
            flags = ctx.get("flags")
            if isinstance(flags, set):
                flags.update(node.set_flags)
        # 2) 进入目标节点时应用其 set_flags（选项 → 目标节点的标志）
        if nxt is not None and nxt in self._nodes:
            target_node = self._nodes[nxt]
            if target_node.set_flags:
                flags = ctx.get("flags")
                if isinstance(flags, set):
                    flags.update(target_node.set_flags)

        if nxt is None:
            # 对话结束
            self._finished = True
            self._cur = None
            return self._finish_state()

        if nxt not in self._nodes:
            raise ValueError(f"对话树 {self._active.dialogue_id} 引用了不存在的节点: {nxt}")
        self._cur = self._nodes[nxt]
        # 进入新节点时同样要带 eval_ctx：新节点若是 choice，其条件选项需过滤
        return self._enter(self._cur, ctx)

    def current_state(
        self,
        eval_ctx: dict[str, Any] | None = None,
    ) -> DialogueState | None:
        """当前展示状态（无活跃对话返回 None）。"""
        if self._cur is None or self._finished:
            return None
        return self._enter(self._cur, eval_ctx)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _enter(
        self,
        node: DialogueNodeConfig,
        eval_ctx: dict[str, Any] | None = None,
    ) -> DialogueState:
        """节点 → 展示状态（choice 节点只列出条件满足的选项）。"""
        opts: list[str] = []
        indices: list[int] = []
        if node.type == "choice":
            indices = self._visible_option_indices(node, eval_ctx or {})
            opts = [node.options[i].text for i in indices]
        return DialogueState(
            dialogue_id=self._active.dialogue_id if self._active else "",
            speaker=node.speaker,
            text=node.text,
            options=opts,
            option_indices=indices,
            history=list(self._history),
            is_finished=False,
        )

    @staticmethod
    def _visible_option_indices(
        node: DialogueNodeConfig,
        ctx: dict[str, Any],
    ) -> list[int]:
        """满足可见条件的选项**原始索引**（未写 condition → 全部可见）。"""
        from Airia.narrative.triggers import ConditionEvaluator

        out: list[int] = []
        for i, opt in enumerate(node.options):
            if opt.condition and not ConditionEvaluator.evaluate(opt.condition, ctx):
                continue
            out.append(i)
        return out

    def _finish_state(self) -> DialogueState:
        return DialogueState(
            dialogue_id=self._active.dialogue_id if self._active else "",
            history=list(self._history),
            is_finished=True,
        )

    def _apply_action(self, action: dict[str, Any] | None, ctx: dict[str, Any]) -> None:
        """执行节点动作。

        骨架期：仅记录到 eval_ctx['pending_actions']（CommandFactory 消费）。
        """
        if not action:
            return
        pending = ctx.setdefault("pending_actions", [])
        pending.append(action)

    def _eval_cond(self, expr: str, ctx: dict[str, Any]) -> bool:
        """求值一个条件表达式。

        NOTE: 骨架期用安全布尔求值（见 triggers.ConditionEvaluator）。
        这里延迟导入避免循环依赖。
        """
        from Airia.narrative.triggers import ConditionEvaluator

        return ConditionEvaluator.evaluate(expr, ctx)
