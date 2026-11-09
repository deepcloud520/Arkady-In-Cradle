"""条件触发器（AGENTS.md Phase 4：narrative/triggers.py）。

对应 docs/phase4-narrative-design.md §3.3 + §2.5/§2.6：
    - TriggerRule / TriggerCondition / TriggerAction（Pydantic，YAML 加载）
    - ConditionEvaluator（安全条件表达式求值，dialogue 也复用）
    - TriggerSystem（每帧 evaluate → 被触发的 TriggerAction 列表，按 priority 降序）

触发器双模式（docs §1.1）：
    - event_match：匹配实时事件（如「首次击毁」）
    - state_check：检查累积状态（如「血量 < 30%」）
    - on_first_tick / flag_set / composite：辅助模式

设计要点：
    - ConditionEvaluator 用 ast 解析白名单校验（拒绝任意名称访问），非裸 eval
    - TriggerSystem 无状态求值器；once/cooldown 状态由外部（NarrativeEngine）管理
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from Airia.lulilo.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 条件表达式求值器（安全子集，供 dialogue + triggers 共用）
# ---------------------------------------------------------------------------


class ConditionEvaluator:
    """安全的条件表达式求值器。

    支持子集（docs §2.4）：
        - 白名单名称：flags / state / event / sim_time / target_unit
        - 运算符：and or not == != < <= > >= in is + - * / % 括号 下标
        - 方法调用：仅 flags.contains(...) / flags.add(...) 类白名单调用

    实现：ast 解析 + 白名单校验（拒绝 Name 未注册 / 任意 Call / 属性访问），
    然后编译求值。非法表达式返回 False 并告警（不抛异常，避免打断 Tick）。
    """

    # 允许引用的顶层名称
    _ALLOWED_NAMES = {
        "flags", "state", "event", "sim_time", "target_unit",
        "True", "False", "None",
    }
    # 允许调用的方法（name.method → True）
    _ALLOWED_CALLS = {
        "flags.contains", "flags.add",
        "state.armor_count", "state.any_ally_health_below",
    }
    _BINOPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }
    _UNARY = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
    _CMP = {
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.Is: operator.is_, ast.IsNot: operator.is_not,
    }

    @classmethod
    def evaluate(cls, expr: str, ctx: dict[str, Any]) -> bool:
        """求值一个布尔表达式。

        Args:
            expr: 表达式字符串（空串 → True，视为无条件）。
            ctx: 求值上下文（flags / state / event / sim_time / target_unit）。

        Returns:
            bool；非法/异常表达式返回 False。
        """
        if not expr or not expr.strip():
            return True
        try:
            tree = ast.parse(expr, mode="eval")
            cls._validate(tree.body)
            # flags.contains(x) → x in flags（Python3 set 无 .contains 方法）
            transformed = _ContainsTransformer().visit(tree)
            ast.fix_missing_locations(transformed)
            code = compile(transformed, "<narrative-condition>", "eval")
            return bool(eval(code, {"__builtins__": {}}, ctx))
        except (SyntaxError, NameError, TypeError, AttributeError, ZeroDivisionError):
            logger.warning("条件表达式求值失败: %r", expr, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # AST 白名单校验
    # ------------------------------------------------------------------

    @classmethod
    def _validate(cls, node: ast.AST) -> None:
        """递归校验 AST 节点，非法节点抛 SyntaxError。"""
        if isinstance(node, ast.Expression):
            cls._validate(node.body)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                cls._validate(v)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in cls._UNARY:
                raise SyntaxError(f"非法一元运算符: {type(node.op).__name__}")
            cls._validate(node.operand)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in cls._BINOPS:
                raise SyntaxError(f"非法二元运算符: {type(node.op).__name__}")
            cls._validate(node.left)
            cls._validate(node.right)
        elif isinstance(node, ast.Compare):
            if any(type(op) not in cls._CMP for op in node.ops):
                raise SyntaxError("非法比较运算符")
            cls._validate(node.left)
            for c in node.comparators:
                cls._validate(c)
        elif isinstance(node, ast.Name):
            if node.id not in cls._ALLOWED_NAMES:
                raise SyntaxError(f"非法名称访问: {node.id}")
        elif isinstance(node, ast.Constant):
            return  # 字面量允许
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                cls._validate(elt)
        elif isinstance(node, ast.List):
            for elt in node.elts:
                cls._validate(elt)
        elif isinstance(node, ast.Subscript):
            cls._validate(node.value)
            cls._validate(node.slice)
        elif isinstance(node, ast.Slice):
            if node.lower:
                cls._validate(node.lower)
            if node.upper:
                cls._validate(node.upper)
        elif isinstance(node, ast.Attribute):
            # 仅允许白名单链（如 flags.contains / state.armor_count）
            chain = cls._chain_name(node)
            if chain not in cls._ALLOWED_CALLS and not cls._is_allowed_attr(chain):
                raise SyntaxError(f"非法属性访问: {chain}")
            cls._validate(node.value)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise SyntaxError("仅允许白名单方法调用")
            chain = cls._chain_name(node.func)
            if chain not in cls._ALLOWED_CALLS:
                raise SyntaxError(f"非法方法调用: {chain}")
            for a in node.args:
                cls._validate(a)
            for kw in node.keywords:
                cls._validate(kw.value)
        elif isinstance(node, ast.keyword):
            cls._validate(node.value)
        else:
            raise SyntaxError(f"不支持的表达式节点: {type(node).__name__}")

    @classmethod
    def _chain_name(cls, node: ast.AST) -> str:
        """提取属性链名（flags.contains → "flags.contains"）。"""
        if isinstance(node, ast.Attribute):
            return f"{cls._chain_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        return "?"

    @classmethod
    def _is_allowed_attr(cls, chain: str) -> bool:
        """属性链是否允许访问（如 state.sim_time / state.pools）。"""
        # 允许 state.* 的公开字段访问（快照 dataclass 属性）
        if chain.startswith("state."):
            return True
        # flags 本身是 set，contains/add 已在白名单；其他 flags.* 拒绝
        return False


class _ContainsTransformer(ast.NodeTransformer):
    """把 flags.contains(x) 调用转换为 x in flags（Python3 set 无 contains 方法）。"""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute):
            chain = self._chain_name(node.func)
            if chain == "flags.contains" and node.args:
                arg = node.args[0]
                in_op = ast.Compare(
                    left=arg,
                    ops=[ast.In()],
                    comparators=[ast.Name(id="flags")],
                )
                return in_op
        return self.generic_visit(node)

    @staticmethod
    def _chain_name(node: ast.AST) -> str:
        if isinstance(node, ast.Attribute):
            return f"{_ContainsTransformer._chain_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        return "?"


# ---------------------------------------------------------------------------
# Pydantic 配置模型（YAML → 引擎）
# ---------------------------------------------------------------------------


class TriggerCondition(BaseModel):
    """触发器条件（docs §2.6）。"""

    type: str = "event_match"            # event_match | state_check | on_first_tick | flag_set | composite
    event_kind: str = ""                 # event_match：事件类型
    check: str = ""                      # state_check：检查函数名
    threshold: Any = None                # state_check：阈值
    filter: dict[str, Any] = Field(default_factory=dict)  # event_match：过滤器
    count: int = 1                       # event_match：第 n 次触发
    flag_name: str = ""                  # flag_set：标志名
    operator: str = "and"                # composite：and / or
    conditions: list["TriggerCondition"] = Field(default_factory=list)


class TriggerAction(BaseModel):
    """触发结果（docs §2.5 activate）。"""

    dialogue: str = ""                   # 激活的对话树 id
    priority: int = 0                    # 数字越大越优先
    pause_sim: bool = True               # 弹出对话时暂停模拟
    once: bool = True                    # 触发后自动注销
    cooldown: float = 0.0                # 冷却秒数


class TriggerRule(BaseModel):
    """一条完整触发器配置。"""

    trigger_id: str
    description: str = ""
    condition: TriggerCondition
    activate: TriggerAction


# ---------------------------------------------------------------------------
# 触发器系统
# ---------------------------------------------------------------------------


class TriggerSystem:
    """触发器系统。

    每帧 evaluate(events, snapshot, flags, sim_time) → 返回被触发的
    TriggerAction 列表（按 priority 降序）。

    once / cooldown 状态管理：
        - once：触发后从 _fired_once 记录，不再触发
        - cooldown：记录最近触发时间，冷却期内不再触发
    """

    def __init__(self) -> None:
        self._rules: list[TriggerRule] = []
        self._fired_once: set[str] = set()
        self._last_fired: dict[str, float] = {}
        # 事件计数（event_match count 用）
        self._event_counts: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, path: Path | str) -> int:
        """加载一个触发器 YAML 文件。

        YAML 结构：
            triggers:
              - trigger_id: ...
                condition: {...}
                activate: {...}

        Returns:
            加载的触发器数量。
        """
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw or "triggers" not in raw:
            logger.warning("触发器文件缺少 triggers 键: %s", path)
            return 0
        count = 0
        for item in raw["triggers"]:
            rule = TriggerRule.model_validate(item)
            self._rules.append(rule)
            count += 1
        logger.info("触发器已加载: %s (%d 条)", path, count)
        return count

    def add_rule(self, rule: TriggerRule) -> None:
        """编程式注册一条触发器。"""
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # 求值
    # ------------------------------------------------------------------

    def evaluate(
        self,
        events: list[Any],
        snapshot: Any | None = None,
        flags: set[str] | None = None,
        sim_time: float = 0.0,
        target_unit: int | None = None,
        side_map: dict[int, str] | None = None,
    ) -> list[TriggerAction]:
        """评估所有触发器，返回被触发的 TriggerAction 列表（priority 降序）。

        Args:
            events: 本帧 EventBus 事件列表（SimulationEvent，含 kind/data）。
            snapshot: 模拟快照（SimulationSnapshot 或 None）。
            flags: 全局标志位（可被事件匹配副作用更新）。
            sim_time: 当前模拟时间。
            target_unit: 当前上下文目标单位索引（可选，供条件引用）。
            side_map: armor_idx → 阵营（"ally"/"enemy"/"neutral"）。事件 filter
                的 ``side`` 键用它推断事件涉及方的阵营（模拟层事件不含 side）。

        Returns:
            list[TriggerAction]（按 priority 降序；空列表 = 无触发）。
        """
        flags = flags if flags is not None else set()
        ctx = {
            "flags": flags,
            "state": snapshot,
            "event": None,
            "sim_time": sim_time,
            "target_unit": target_unit,
        }

        fired: list[TriggerAction] = []
        for rule in self._rules:
            if rule.trigger_id in self._fired_once:
                continue  # once 已触发过
            # 冷却检查
            last = self._last_fired.get(rule.trigger_id, -1e18)
            if sim_time - last < rule.activate.cooldown:
                continue

            if self._match(rule.condition, events, ctx, side_map):
                fired.append(rule.activate)
                self._last_fired[rule.trigger_id] = sim_time
                if rule.activate.once:
                    self._fired_once.add(rule.trigger_id)

        # priority 降序
        fired.sort(key=lambda a: a.priority, reverse=True)
        return fired

    def reset(self) -> None:
        """重置所有状态（新任务开始时调用）。"""
        self._fired_once.clear()
        self._last_fired.clear()
        self._event_counts.clear()

    # ------------------------------------------------------------------
    # 条件匹配
    # ------------------------------------------------------------------

    def _match(
        self,
        cond: TriggerCondition,
        events: list[Any],
        ctx: dict[str, Any],
        side_map: dict[int, str] | None = None,
    ) -> bool:
        if cond.type == "event_match":
            return self._match_event(cond, events, side_map)
        if cond.type == "state_check":
            return self._match_state(cond, ctx)
        if cond.type == "on_first_tick":
            # 任意事件到来即视为已运行（首帧事件由 NarrativeEngine 保证）
            return bool(events)
        if cond.type == "flag_set":
            flags = ctx.get("flags")
            return isinstance(flags, set) and cond.flag_name in flags
        if cond.type == "composite":
            results = [self._match(c, events, ctx, side_map) for c in cond.conditions]
            return all(results) if cond.operator == "and" else any(results)
        logger.warning("未知触发器条件类型: %s", cond.type)
        return False

    def _match_event(
        self,
        cond: TriggerCondition,
        events: list[Any],
        side_map: dict[int, str] | None = None,
    ) -> bool:
        """事件匹配：event_kind + filter 过滤 + count 计数。

        filter 特殊键：
            - ``side`` / ``source_side`` / ``target_side``：用 side_map 推断
              事件涉及方的阵营（模拟层事件不含 side 字段）。
        """
        matched = 0
        for ev in events:
            kind = getattr(ev, "kind", None)
            if kind != cond.event_kind:
                continue
            data = getattr(ev, "data", {}) or {}
            if not self._filter_matches(cond.filter, data, kind, side_map):
                continue
            key = (cond.event_kind, str(sorted(cond.filter.items())))
            self._event_counts[key] = self._event_counts.get(key, 0) + 1
            matched = self._event_counts[key]
        return matched >= cond.count

    def _filter_matches(
        self,
        f: dict[str, Any],
        data: dict[str, Any],
        kind: str,
        side_map: dict[int, str] | None,
    ) -> bool:
        """filter 匹配：普通键直接比对；side 键用 side_map 推断阵营。"""
        side_map = side_map or {}
        for k, v in f.items():
            if k == "side":
                if self._event_side(data, kind, side_map) != v:
                    return False
            elif k == "source_side":
                if self._event_source_side(data, kind, side_map) != v:
                    return False
            elif k == "target_side":
                if self._event_target_side(data, kind, side_map) != v:
                    return False
            else:
                if data.get(k) != v:
                    return False
        return True

    # -- 事件涉及方的 id / 阵营推断（模拟层事件字段不统一）--

    @staticmethod
    def _event_target_id(data: dict[str, Any], kind: str) -> int:
        """事件目标单位 id（destroyed 用 armor_idx，其余用 target）。"""
        if kind == "destroyed":
            return int(data.get("armor_idx", -1))
        return int(data.get("target", -1))

    @staticmethod
    def _event_source_id(data: dict[str, Any], kind: str) -> int:
        """事件发起单位 id（fire/hit/miss 用 shooter，detection 用 source）。"""
        return int(data.get("shooter", data.get("source", -1)))

    def _event_side(self, data: dict[str, Any], kind: str, side_map: dict[int, str]) -> str:
        """事件涉及方阵营：优先 target（被影响方），其次 source。"""
        tgt = self._event_target_id(data, kind)
        if tgt in side_map:
            return side_map[tgt]
        src = self._event_source_id(data, kind)
        return side_map.get(src, "")

    def _event_source_side(self, data: dict[str, Any], kind: str, side_map: dict[int, str]) -> str:
        return side_map.get(self._event_source_id(data, kind), "")

    def _event_target_side(self, data: dict[str, Any], kind: str, side_map: dict[int, str]) -> str:
        return side_map.get(self._event_target_id(data, kind), "")

    def _match_state(self, cond: TriggerCondition, ctx: dict[str, Any]) -> bool:
        """状态检查：check 函数名 + threshold。"""
        name = cond.check
        threshold = cond.threshold
        try:
            # 不依赖快照的检查
            if name == "sim_time_greater_than":
                return float(ctx.get("sim_time", 0.0)) > float(threshold)
            if name == "flag_set_check":
                flags = ctx.get("flags")
                return isinstance(flags, set) and str(threshold) in flags

            # 以下检查需要快照
            state = ctx.get("state")
            if state is None:
                return False
            if name == "any_ally_health_below":
                return self._any_ally_health_below(state, float(threshold))
            if name == "armor_count_less_than":
                return self._armor_count(state) < int(threshold)
            if name == "enemy_destroyed_count":
                return self._enemy_destroyed_count(ctx) >= int(threshold)
            logger.warning("未知状态检查: %s", name)
            return False
        except (TypeError, ValueError, AttributeError):
            logger.warning("状态检查失败: %s", name, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 状态检查实现（读快照）
    # ------------------------------------------------------------------

    @staticmethod
    def _any_ally_health_below(state: Any, threshold: float) -> bool:
        """快照中是否存在血量 < threshold 的活跃单位。"""
        pool = state.pools.get("armor") if hasattr(state, "pools") else None
        if pool is None:
            return False
        box = state.pools.get("box") if hasattr(state, "pools") else None
        active = pool.active
        # 整车血量 = HULL 部位血（docs/part-health-design.md §三；哨兵槽 -1 → 0 安全）
        health = box.hp[pool.hull_idx] if box is not None else np.zeros(pool.capacity, dtype=np.float32)
        return bool(active & (health < threshold * 100.0)).any()

    @staticmethod
    def _armor_count(state: Any, threshold: Any) -> int:
        pool = state.pools.get("armor") if hasattr(state, "pools") else None
        if pool is None:
            return 0
        return int(pool.active.sum())

    @staticmethod
    def _enemy_destroyed_count(ctx: dict[str, Any]) -> int:
        """骨架期简化：从 ctx['event'] 计数（正式实现由 recorder 统计）。"""
        event = ctx.get("event")
        if event and event.get("type") == "destroyed":
            return 1
        return 0
