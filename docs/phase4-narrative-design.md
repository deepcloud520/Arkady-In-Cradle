# Phase 4 叙事层管线设计

> 状态：**设计定稿，待实现**（2026-08-12）
> 对应 AGENTS.md Phase 4（叙事层集成）
> 注意：经研究，暂时不设计“跳过战斗”无头引擎自主模拟，跳过战斗后不显示

## 零、现有接口盘点（叙事层上下游）

### 上游：叙事层消费的模拟层接口

| 接口 | 位置 | 叙事层用法 |
|------|------|-----------|
| `SimulationState.snapshot()` | `simulation/state.py` | `Snapshot`（frozen dataclass），含 `sim_time` / `pools`(armor/box/projectile) / `logistics`(ammo/fuel) |
| `EventBus.drain()` | `simulation/loop.py` | 每帧 drain 事件列表：`fire` / `hit` / `penetration` / `destroyed` / `miss` |
| `SimulationLoop.sim_paused` | `simulation/loop.py` | 叙事层可设置暂停（如等待玩家选择） |

### 下游：叙事层产出的接口

| 接口 | 位置 | 叙事层用法 |
|------|------|-----------|
| `CommandQueue.push(cmd)` | `simulation/commands.py` | 玩家选择 → `Command(type, target_id, params)` → 入队 |
| `CommandType` 枚举 | `simulation/commands.py` | REDEPLOY / STRIKE / HOLD / RESUPPLY / DISENGAGE |

### 现有事件类型（EventBus emit 的 kind）

| 事件 | data 字段 | 语义 | 埋点位置 |
|------|----------|------|---------|
| `fire` | shooter, target, proj, weapon, ammo, range, sigma | 开火 | `engagement.py::fire()` |
| `hit` | proj, armor(目标), part(str), pen(bool), pen_mm, angle | 命中（含穿透结果）| `engagement.py::_resolve_hit()` |
| `destroyed` | armor_idx, cause | 单位摧毁 | `engagement.py::_resolve_hit()` |
| `miss` | proj, shooter | 脱靶/弹丸回收 | `engagement.py::_recycle_stale()` |
| `detection` | source, target, sensor, state_change | 探测状态跃升（Phase 4 新增埋点） | `detection/sensors.py`（待加） |
| `command` | type, target, params | 玩家/AI 指令执行 | `loop.py::_apply_command()`（Phase 3 已有骨架） |

### 叙事层不被允许做的事

- ❌ 直接修改 `SimulationState`（AGENTS.md 3.7）
- ❌ 直接调用模拟层的 `step()` / `tick()`
- ❌ 直接操作 SoA 数组（armor.pos 等）
- ✅ 只通过 `CommandQueue` 下发意图、通过 `Snapshot` + `EventBus` 消费状态


## 一、管线总览

```
┌──────────────────────────────────────────────────────────────┐
│                    UI 层（render / main_window）              │
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │  GLWidget（战场渲染）│    │  NarrativeWidget（对话面板） │  │
│  │  摄像机自由控制      │    │  QWidget overlay 半透明底栏  │  │
│  └─────────────────────┘    └──────────┬──────────────────┘  │
│                                        │                     │
│                         玩家点击选项 → NarrativeEngine       │
└────────────────────────────────────────┼─────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────┐
│                    叙 事 层（Airis/narrative/）               │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ DialogueEngine│◄──│TriggerSystem │◄──│ NarrativeEngine  │  │
│  │ (dialogue.py) │   │(triggers.py) │   │  (engine.py)     │  │
│  │              │   │              │   │  中央调度器       │  │
│  │ 对话树状态机  │   │ 条件求值器   │   │  - 每帧 tick()   │  │
│  │ YAML 加载    │   │ 事件监听器   │   │  - 触发器→对话树  │  │
│  └──────┬───────┘   └──────┬───────┘   └────┬─────┬──────┘  │
│         │                  │                 │     │         │
│  ┌──────▼──────────────────▼─────────────────▼──┐  │         │
│  │           CommandFactory (commands.py)        │  │         │
│  │    对话选项 action → Command → push()         │  │         │
│  └──────────────────────────────────────────────┘  │         │
│                                                    │         │
│  ┌─────────────────────────────────────────────────▼───────┐ │
│  │          CombatEventRecorder (recorder.py)              │ │
│  │  - 环形缓冲区（内存 1000 条）                            │ │
│  │  - SQLite 持久化（完整事件流）                           │ │
│  │  - 每帧 drain EventBus → 结构化写入                     │ │
│  │  - 生成 CombatBriefing（战果统计 + 时间线 + 单位战报）   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │NarrativeRenderer (renderer.py)(需为未来UI精细设计预留空间) ││
│  │  - 对话文本 + 选项按钮（QWidget overlay）                  ││
│  │  - 角色立绘/头像占位                                       ││
│  │  - 战斗简报面板（表格/时间线/统计图表占位）                 ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
              每帧 drain: EventBus + Snapshot
              玩家选择: CommandQueue.push()
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                  模 拟 层（lulilo/simulation/）               │
│                                                              │
│  EventBus  ← fire / hit / destroyed / miss / detection       │
│  SimulationState  → snapshot()（只读）                       │
│  CommandQueue  ← 玩家指令                                    │
│  SimulationLoop  sim_paused ← 等待玩家选择时暂停             │
└──────────────────────────────────────────────────────────────┘
```

> 注：`detection` 事件为 Phase 4 新增埋点（`detection/sensors.py`），
> 在探测状态发生跃升时 emit（首次发现某目标 / 丢失目标 / 传感器切换）。
> `command` 事件由 `SimulationLoop._apply_command()` 自动 emit（Phase 3 已有骨架）。

### 1.1 核心设计决策

| 决策 | 理由 |
|------|------|
| **叙事层每帧 tick，与模拟 Tick 同频** | 消费 EventBus 事件不丢帧；对话状态机轻量（<1ms） |
| **触发器用「事件匹配 + 状态条件」双模式** | 事件匹配响应实时事件（击中/摧毁）；状态条件响应累积状态（血量 < 30%） |
| **对话树用 YAML 定义，Pydantic 加载** | 对齐项目规范（武器/弹药也用 Pydantic）；YAML 嵌套结构适合对话分支 |
| **“战斗可跳过”= 模拟层 `run_headless(duration)`** | 叙事层设置跳过标志 → 模拟全速跑 → drain 最终 Snapshot → 叙事层生成简报 |
| **渲染层独立 QWidget overlay** | 对话面板不侵入 GLWidget，透明背景叠在战场上方，互不干扰 |


## 二、YAML 数据格式

### 2.1 对话树（dialogue tree）
以下文本仅供测试，与Arkady in Cradle剧情无关。
```yaml
# story/dialogues/prologue.yaml
dialogue_id: "prologue_001"
title: "序章：摇篮之中"
nodes:
  - id: start
    type: text
    speaker: "系统通报"
    text: |
      第7装甲旅已抵达集结点「摇篮」。
      前方侦察报告：敌装甲集群正在逼近，预计接触时间 180 秒。
    next: choice_deploy

  - id: choice_deploy
    type: choice
    speaker: "指挥官"
    text: "如何部署？"
    options:
      - text: "展开防御阵型，固守待援"
        next: deploy_defensive
        action:
          type: HOLD
          target_id: "7th_armor_company"
          params:
            formation: "defensive_line"
      - text: "主动出击，抢占北部高地"
        next: deploy_offensive
        action:
          type: REDEPLOY
          target_id: "7th_armor_company"
          params:
            waypoint: [120, 80]
            formation: "wedge"
      - text: "（等待更多情报...）"
        next: wait_intel
        condition: "flags.contains('has_scout_report')"

  - id: deploy_defensive
    type: text
    speaker: "副官 伊万诺娃"
    text: "明白。各单位注意，展开防御阵型！炮手自由开火。"
    next: null         # null = 结束对话，控制权交还模拟
    set_flags: ["defensive_posture"]

  - id: deploy_offensive
    type: text
    speaker: "副官 伊万诺娃"
    text: "Roger that! 楔形队形，向北推进！"
    next: null
    set_flags: ["offensive_posture"]

  - id: wait_intel
    type: text
    speaker: "指挥官"
    text: "再等等。让侦察无人机再飞一圈。"
    next: null
    set_flags: ["cautious_commander"]
```

### 2.2 节点类型

| type | 说明 | 必需字段 |
|------|------|---------|
| `text` | 纯文本叙述 | speaker, text, next |
| `choice` | 玩家选择分支 | speaker, text, options[] |
| `branch` | 条件自动分支（无 UI） | conditions[], next_true, next_false |
| `action` | 执行动作后自动前进 | action, next |

### 2.3 选项字段

| 字段 | 说明 |
|------|------|
| `text` | 显示文本 |
| `next` | 下一个节点 id |
| `action` | 可选：选择的副作用（Command 入队） |
| `condition` | 可选：该选项的可见条件（表达式字符串） |

### 2.4 条件表达式语法

```
"flags.contains('scout_complete')"
"state.armor_count('enemy') < 5"
"state.sim_time > 120"
"event.last('destroyed').armor_idx == target_unit"
"!flags.contains('alerted')"
```

支持 `&&`、`||`、`!`、`()` 组合。

### 2.5 触发器配置
以下trigger仅供测试和格式参考。
```yaml
# story/triggers/chapter1.yaml
triggers:
  - trigger_id: "first_blood"
    description: "首次击毁敌方单位"
    condition:
      type: event_match
      event_kind: "destroyed"
      filter:
        side: "enemy"          # 目标方：enemy / ally / any
      count: 1                 # 第 n 次触发（1=第一次）
    activate:
      dialogue: "first_blood_dialogue"
      priority: 5              # 数字越大越优先
      once: true               # 触发后自动注销
      pause_sim: true          # 弹出对话时暂停模拟

  - trigger_id: "half_health"
    description: "我方任意单位血量 < 30%"
    condition:
      type: state_check
      check: "any_ally_health_below"
      threshold: 0.3
    activate:
      dialogue: "commander_warning"
      priority: 3
      once: false
      cooldown: 30.0           # 冷却 30 秒，避免反复弹出

  - trigger_id: "mission_time_elapsed"
    description: "战斗超过 300 秒"
    condition:
      type: state_check
      check: "sim_time_greater_than"
      threshold: 300.0
    activate:
      dialogue: "time_warning"
      priority: 2
      once: true

  - trigger_id: "battle_start"
    description: "模拟首次 tick 触发（开场对话）"
    condition:
      type: on_first_tick
    activate:
      dialogue: "prologue_001"
      priority: 10
      once: true
      pause_sim: true
```

### 2.6 条件类型

| type | 说明 | 参数 |
|------|------|------|
| `event_match` | 匹配特定事件类型 | event_kind, filter, count |
| `state_check` | 检查模拟状态 | check(函数名), threshold |
| `on_first_tick` | 模拟首次 tick | 无 |
| `flag_set` | 标志位被设置 | flag_name |
| `composite` | 组合条件 | operator(and/or), conditions[] |


## 三、模块接口定义（Python 代码骨架）

### 3.1 NarrativeEngine（中央调度器）

```python
# narrative/engine.py

class NarrativeEngine:
    """叙事层中央调度器。

    每帧 tick(snapshot, events) → 评估触发器 → 激活对话树 → 管理状态。
    对话激活期间 sim_paused=True（模拟暂停等待玩家输入）。
    """

    def __init__(self, command_queue: CommandQueue) -> None:
        self.command_queue = command_queue
        self.dialogue_engine = DialogueEngine()
        self.trigger_system = TriggerSystem()
        self._flags: set[str] = set()
        self._active_dialogue: str | None = None
        self._skip_combat = False
        # 条件求值上下文
        self._eval_ctx: dict[str, Any] = {}

    def tick(self, snapshot: SimulationSnapshot, events: list[SimulationEvent]) -> NarrativeState:
        """每帧调用。

        Returns:
            NarrativeState：当前叙事状态（渲染层据此决定显示什么）。
        """
        ...

    def on_player_choice(self, dialogue_id: str, option_index: int) -> None:
        """玩家选择了某个选项。"""
        ...

    def request_skip_combat(self) -> None:
        """玩家请求跳过当前战斗。"""
        ...

    @property
    def is_paused(self) -> bool:
        """模拟是否应暂停（等待玩家选择）。"""
        ...

    @property
    def has_active_dialogue(self) -> bool:
        """是否有待玩家选择的对话。"""
        ...


@dataclass
class NarrativeState:
    """渲染层消费的叙事状态。"""
    active_dialogue: DialogueState | None   # 当前活跃对话（None = 无对话）
    pending_commands: list[Command]          # 待推入队列的指令
    flags: set[str]                          # 全局标志位
    skip_combat: bool                        # 是否请求跳过战斗
    recent_events: list[dict]                # 最近的事件摘要（供 HUD 滚动日志）
```

### 3.2 DialogueEngine（对话树状态机）

```python
# narrative/dialogue.py

@dataclass
class DialogueTree:
    """一个完整的对话树（从 YAML 加载）。"""
    dialogue_id: str
    title: str
    nodes: dict[str, DialogueNode]


@dataclass
class DialogueNode:
    """对话树中的一个节点。"""
    id: str
    type: Literal["text", "choice", "branch", "action"]
    speaker: str = ""
    text: str = ""
    next: str | None = None
    options: list[ChoiceOption] = field(default_factory=list)
    action: dict | None = None
    set_flags: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    next_true: str | None = None
    next_false: str | None = None


@dataclass
class ChoiceOption:
    text: str
    next: str
    action: dict | None = None
    condition: str | None = None


@dataclass
class DialogueState:
    """当前对话的运行时状态（渲染层消费）。"""
    dialogue_id: str
    speaker: str
    text: str
    options: list[str]        # 选项文本列表（仅 choice 节点有）
    history: list[str]        # 对话历史（滚动显示）
    is_finished: bool         # 对话是否结束


class DialogueEngine:
    """对话树状态机。

    职责：
        - 加载 YAML 对话树 → DialogueTree
        - 激活一棵树：active(node_id) → DialogueState
        - 推进状态：advance(option_index) → DialogueState
        - 条件求值：外部注入 eval_ctx
    """

    def load(self, path: Path) -> DialogueTree: ...
    def load_all(self, dir_path: Path) -> None: ...
    def activate(self, dialogue_id: str) -> DialogueState: ...
    def advance(self, option_index: int, eval_ctx: dict) -> DialogueState: ...
    def current_state(self) -> DialogueState | None: ...
```

### 3.3 TriggerSystem（条件触发器）

```python
# narrative/triggers.py

@dataclass
class TriggerRule:
    trigger_id: str
    condition: TriggerCondition
    activate: TriggerAction
    once: bool = True
    cooldown: float = 0.0


@dataclass
class TriggerCondition:
    type: Literal["event_match", "state_check", "on_first_tick", "flag_set", "composite"]
    event_kind: str = ""
    check: str = ""
    threshold: Any = None
    filter: dict = field(default_factory=dict)
    count: int = 1
    operator: str = "and"
    conditions: list["TriggerCondition"] = field(default_factory=list)


@dataclass
class TriggerAction:
    dialogue: str             # 激活的对话树 id
    priority: int = 0
    pause_sim: bool = True
    once: bool = True
    cooldown: float = 0.0


class TriggerSystem:
    """触发器系统。

    每帧 evaluate(events, snapshot, flags) → 返回被触发的 TriggerAction 列表，
    按 priority 降序排列。
    """

    def load(self, path: Path) -> None: ...
    def evaluate(
        self,
        events: list[SimulationEvent],
        snapshot: SimulationSnapshot,
        flags: set[str],
    ) -> list[TriggerAction]: ...
```

### 3.4 CommandFactory（玩家选择 → 指令）

```python
# narrative/commands.py

class CommandFactory:
    """对话选项 action → Command 对象。

    纯函数，无状态。映射 AGENTS.md 3.7：
        action: { type: HOLD, target_id: "...", params: {...} }
        → Command(CommandType.HOLD, target_id, params)
    """

    @staticmethod
    def from_action(action: dict) -> Command | None: ...

    @staticmethod
    def from_node(node: DialogueNode, choice_index: int) -> Command | None: ...
```

### 3.5 NarrativeRenderer（Qt 渲染）

```python
# narrative/renderer.py

class NarrativeWidget(QWidget):
    """对话面板 Overlay。

    定位：覆盖在 GLWidget 上方、屏幕底部 1/3。
    样式：半透明深色背景 + 白色文字 + 选项按钮。

    信号：
        option_selected(dialogue_id: str, option_index: int)
        skip_combat_requested()
    """

    option_selected = Signal(str, int)
    skip_combat_requested = Signal()

    def show_dialogue(self, state: DialogueState) -> None: ...
    def show_briefing(self, summary: CombatBriefing) -> None: ...
    def hide(self) -> None: ...


@dataclass
class CombatBriefing:
    """战斗简报（由 CombatEventRecorder 生成）。"""
    duration_s: float
    allies_remaining: int
    enemies_destroyed: int
    casualties: list[str]        # 阵亡单位名称
    key_events: list[str]        # 关键事件摘要
    outcome: str                 # "victory" / "defeat" / "stalemate"
```

### 3.6 CombatEventRecorder（战斗事件记录 + 简报生成）

```python
# narrative/recorder.py

# ---------------------------------------------------------------------------
# 事件数据模型（SQLite 行 → Python dataclass）
# ---------------------------------------------------------------------------

@dataclass
class CombatEventRecord:
    """一条战斗事件记录（与 SQLite combat_events 表一一对应）。"""
    id: int = -1                           # SQLite 自增主键（内存中为 -1）
    sim_time: float = 0.0
    tick_count: int = 0
    event_type: str = ""                   # fire / hit / destroyed / detection / command / miss
    source_id: int = -1                    # 发起者 armor_idx（-1=系统/玩家）
    source_name: str = ""                  # 如 "M1A2 #3"
    source_side: str = ""                  # ally / enemy / neutral
    target_id: int = -1
    target_name: str = ""
    target_side: str = ""
    weapon_id: str = ""
    ammo_id: str = ""
    hit_part: str = ""                     # HULL / TURRET / ENGINE / TRACK / AMMO_RACK
    penetration: bool = False
    pen_mm: float = 0.0
    angle_deg: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    extra_json: str = ""                   # 扩展字段 JSON
    observed_by_player: int = 0            # 0=未观测 / 1=已观测 / 2=推测（模糊信息）


class CombatEventRecorder:
    """战斗事件记录器：环形缓冲区 + SQLite 持久化 + 简报生成。

    设计原则（参考 CMO Message Log + DCS Debrief）：
        - 事件驱动：模拟层 EventBus 每帧 drain → 结构化写入
        - 数据与表现分离：SQLite 存原始事件，BriefingGenerator 做聚合分析
        - 战争迷雾：observed_by_player 标记观测状态，简报只展示已观测信息
        - Arcadia 认知层预留：简报可注入主观解读文本

    用法:
        recorder = CombatEventRecorder(db_path=":memory:")  # 或 "logs/mission_001.db"
        recorder.on_events(events)                           # 每帧 drain 后调用
        briefing = recorder.generate_briefing()              # 生成简报
        recorder.close()
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        ring_size: int = 1000,
        unit_registry: dict[int, str] | None = None,  # armor_idx → 单位名
        side_map: dict[int, str] | None = None,        # armor_idx → ally/enemy
    ) -> None: ...

    # -- 事件写入 --

    def on_events(self, events: list[SimulationEvent], snapshot: SimulationSnapshot) -> None:
        """每帧调用：drain EventBus 事件 → 结构化写入环形缓冲区 + SQLite。"""
        ...

    def flush(self) -> None:
        """强制将缓冲区刷入 SQLite（阶段切换/存档时调用）。"""
        ...

    # -- 查询 --

    def query(
        self,
        event_type: str | None = None,
        source_id: int | None = None,
        target_id: int | None = None,
        time_range: tuple[float, float] | None = None,
        observed_only: bool = True,              # 仅返回玩家已观测的事件
        limit: int = 500,
    ) -> list[CombatEventRecord]: ...

    def timeline(self, observed_only: bool = True) -> list[CombatEventRecord]:
        """按时间排序的完整事件时间线。"""
        ...

    # -- 简报生成 --

    def generate_briefing(self) -> CombatBriefing:
        """从事件流生成战斗简报。

        聚合维度：
            1. 战果统计：双方摧毁数 / 弹药消耗 / 存活单位
            2. 时间线：关键事件按时间排列
            3. 单位战报：每个单位的射击/命中/击杀/损失
            4. 关键事件高亮：首次命中、友军误击、弹药殉爆等
        """
        ...

    def unit_stats(self, armor_idx: int) -> UnitStatsReport:
        """查询单个单位的服役记录。"""
        ...

    # -- 生命周期 --

    def close(self) -> None: ...
    def reset(self) -> None:
        """清空缓冲区 + SQLite 表（新任务开始时）。"""
        ...


@dataclass
class UnitStatsReport:
    """单位服役记录。"""
    armor_idx: int
    unit_name: str
    side: str
    shots_fired: int = 0
    hits_landed: int = 0
    hits_received: int = 0
    kills: int = 0
    destroyed: bool = False
    destroyed_by: str = ""           # 击杀者名称
    ammo_consumed: float = 0.0
    survival_time: float = 0.0
    notable_events: list[str] = field(default_factory=list)  # 关键事件摘要


class BriefingGenerator:
    """简报生成器（纯函数，无状态），从事件列表生成 CombatBriefing。

    设计原则：
        - 多维度聚合：战果统计 + 时间线 + 单位战报 + 关键事件高亮
        - 战争迷雾：observed_only=True 时过滤未观测/推测事件
        - Arcadia 认知：可注入 subjective_notes（主观解读文本，预留）
    """

    def __init__(self, side_map: dict[int, str] | None = None) -> None: ...

    def generate(
        self,
        events: list[CombatEventRecord],
        unit_names: dict[int, str],
        duration_s: float,
        observed_only: bool = True,
        subjective_notes: dict[str, str] | None = None,  # Arcadia 主观解读
    ) -> CombatBriefing: ...

    def _count_kills(self, events: list[CombatEventRecord]) -> dict[str, int]: ...
    def _build_timeline(self, events: list[CombatEventRecord]) -> list[str]: ...
    def _detect_highlights(self, events: list[CombatEventRecord]) -> list[str]: ...
    def _per_unit_stats(
        self, events: list[CombatEventRecord], unit_names: dict[int, str]
    ) -> list[UnitStatsReport]: ...
```


## 四、战斗跳过机制

> ⚠️ **暂缓设计**（2026-08-12 决策）：跳过战斗后不显示，无头引擎自主模拟暂不设计。
> 当前阶段战斗必须完整运行；简报由 `CombatEventRecorder` 在战斗过程中实时生成，
> 任务结束时通过 `generate_briefing()` 产出完整战报。


## 五、数据流时序

### 5.1 正常 Tick（含事件记录）

```
[模拟层] step(dt)
    ├── 消费 CommandQueue
    ├── 推进物理状态
    ├── events = EventBus.drain()
    │
    ▼
[叙事层] engine.tick(snapshot, events)
    ├── recorder.on_events(events, snapshot)     ← 新增：结构化事件写入
    │   ├── SimulationEvent → CombatEventRecord（字段映射）
    │   ├── 查 snapshot 补全 source_name / side
    │   └── 环形缓冲区 + SQLite 写入
    │
    ├── trigger_system.evaluate(events, snapshot, flags)
    │   └── 匹配到 trigger "first_blood"
    ├── dialogue_engine.activate("first_blood_dialogue")
    ├── sim_paused = True
    ├── return NarrativeState(active_dialogue=..., is_paused=True)
    │
    ▼
[渲染层] renderer.show_dialogue(state.active_dialogue)
    ├── 显示文本 + 选项按钮
    │
    ▼
[玩家] 点击选项 1 → renderer.option_selected.emit("first_blood", 1)
    │
    ▼
[叙事层] engine.on_player_choice("first_blood", 1)
    ├── choice.action → CommandFactory → CommandQueue.push()
    ├── dialogue_engine.advance(1)
    ├── 如果 next == null → 对话结束，sim_paused = False
    ├── return NarrativeState(active_dialogue=None, pending_commands=[...])
    │
    ▼
[渲染层] renderer.hide()
    │
    ▼
[模拟层] sim_paused = False → 恢复正常 Tick
    → 下一个 step() 中消费 CommandQueue 指令
```

### 5.2 简报生成（任务结束时）

```
[任务结束] → NarrativeEngine.generate_briefing()
    │
    ├── recorder.generate_briefing()
    │   ├── flush() 缓冲区残余 → SQLite
    │   ├── BriefingGenerator.generate(events, ...)
    │   │   ├── 战果统计（双方摧毁/存活/弹药消耗）
    │   │   ├── 时间线（关键事件按 sim_time 排列）
    │   │   ├── 单位战报（每个单位的射击/命中/击杀/损失）
    │   │   └── 关键事件高亮（首次命中/友军误击/弹药殉爆）
    │   └── → CombatBriefing
    │
    ▼
[渲染层] renderer.show_briefing(briefing)
    ├── 显示战果统计表格
    ├── 显示时间线（滚动列表）
    ├── 显示单位战报
    └── [可选] 导出为 .json / .txt
```


## 六、实现阶段划分

### Phase 4a：骨架（数据模型 + 无 UI 的逻辑管线）

| 模块 | 输出 | 预估 |
|------|------|------|
| `narrative/dialogue.py` | DialogueTree/DialogueNode (Pydantic) + YAML 加载 + DialogueEngine 状态机 | 1d |
| `narrative/triggers.py` | TriggerRule/TriggerCondition/TriggerAction (Pydantic) + TriggerSystem.evaluate | 1d |
| `narrative/commands.py` | CommandFactory（纯函数） | 0.5d |
| `narrative/recorder.py` | CombatEventRecorder（环形缓冲区 + SQLite）+ BriefingGenerator（聚合分析） | 1.5d |
| `narrative/engine.py` | NarrativeEngine 中央调度器（集成 recorder） | 1d |
| 单元测试 | 对话树加载/推进/分支 + 触发器匹配/优先级/冷却 + 事件写入/查询/简报生成 | 1.5d |

### Phase 4b：渲染（Qt Widget）

| 模块 | 输出 | 预估 |
|------|------|------|
| `narrative/renderer.py` | NarrativeWidget（QWidget overlay，含简报面板） | 1d |
| `main_window.py` 集成 | 叠加 NarrativeWidget 到 GLWidget 上方 | 0.5d |

### Phase 4c：集成 + 示例数据 ✅（2026-08-12）

| 模块 | 输出 | 状态 |
|------|------|------|
| 示例对话树 | `story/dialogues/prologue.yaml`（序章）+ `first_blood.yaml`（首次交战，含 branch 演示） | ✅ 完成 |
| 示例触发器 | `story/triggers/chapter1.yaml`（开场 / 首次击毁 / 时间警告） | ✅ 完成 |
| 简报面板 UI | 表格/时间线/统计图表面板 | ⏸️ 暂缓（依赖 4b Qt 面板，与 4b 一起做） |
| 集成测试 | `tests/test_narrative_integration.py`（4 用例：完整 Tick 循环含对话 + 事件记录 + 简报生成 + 指令消费 + 跳过 + 重置） | ✅ 完成（288 passed） |

> 用户决策（2026-08-12）：`detection` 事件埋点**不加**；Phase 4b（NarrativeWidget Qt 面板）**暂缓**，简报面板 UI 一并推迟到 UI 阶段。


## 七、关键约束（对齐 AGENTS.md）

| 约束 | 说明 |
|------|------|
| **叙事不碰模拟** | `NarrativeEngine` 永远不持有 `SimulationState` 引用，只读 `Snapshot` + `EventBus` |
| **线程安全** | `CommandQueue` 已含 `threading.Lock`，叙事层 `push()` 天然安全 |
| **暂停语义** | `sim_paused=True` 期间模拟层只消费指令不推进物理（AGENTS.md 3.1） |
| **依赖方向** | `narrative` 只依赖 `lulilo/simulation`（commands/state/loop），不依赖 lulilo 内部子模块 |
| **YAML → Pydantic** | 所有配置类继承 `BaseModel`，对齐项目规范 |
| **编码** | 所有文件读写显式 `encoding='utf-8'` |
| **SQLite 不阻塞 Tick** | `CombatEventRecorder.on_events()` 只写环形缓冲区（O(1)）；`flush()` 在 Tick 间隙/阶段切换时批量写入，不进入热路径 |
| **战争迷雾** | 简报只展示 `observed_by_player >= 1` 的事件；推测事件（=2）以模糊形式呈现（如「未知目标」） |
| **不可变追加** | `combat_events` 表只 INSERT 不 UPDATE/DELETE（事件日志语义）；`mission_summary` / `unit_stats` 在任务结束时一次性聚合写入 |


## 八、文件规划

```
Airis/narrative/
    __init__.py           # 导出 NarrativeEngine / DialogueEngine / TriggerSystem / ...
    dialogue.py           # DialogueTree / DialogueNode / DialogueEngine
    triggers.py           # TriggerRule / TriggerSystem / ConditionEvaluator
    commands.py           # CommandFactory
    engine.py             # NarrativeEngine + NarrativeState
    recorder.py           # CombatEventRecorder + BriefingGenerator + UnitStatsReport
    renderer.py           # NarrativeWidget（Qt 渲染）

story/
    dialogues/            # 对话树 YAML
        prologue.yaml
        ...
    triggers/             # 触发器配置 YAML
        chapter1.yaml
        ...

tests/
    test_narrative.py     # 单元测试：对话/触发/指令/事件记录/简报生成
```


## 九、战斗事件记录系统（SQLite + 环形缓冲区 + 简报生成）

> 参考：CMO Message Log + DCS Debrief 系统。
> 核心思路：**事件驱动 + 数据与表现分离**。模拟层埋点触发事件 → 结构化存储 → 简报生成器多维度聚合。

### 9.1 事件类型定义

| 事件类型 | 来源（模拟层埋点） | 关键字段 |
|---------|-------------------|---------|
| `fire` | `FiringControl.fire()` → EventBus | source_id, target_id, weapon_id, ammo_id, pos |
| `hit` | `FiringControl._resolve_hit()` → EventBus | source_id, target_id, hit_part, pen_mm, angle_deg |
| `destroyed` | `FiringControl._resolve_hit()` → EventBus | target_id, source_id（击杀者）, cause |
| `miss` | `FiringControl._recycle_stale()` → EventBus | source_id, proj |
| `detection` | `detection/sensors.py`（新增埋点） | source_id（传感器方）, target_id（被探测方）, sensor_type |
| `command` | `SimulationLoop._apply_command()` → EventBus | target_id, command_type, params |

### 9.2 SQLite 数据库设计

```sql
-- 战斗事件主表（每条事件一行，不可变追加）
CREATE TABLE IF NOT EXISTS combat_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_time        REAL NOT NULL,              -- 模拟时间 (s)
    tick_count      INTEGER NOT NULL,           -- Tick 序号
    event_type      TEXT NOT NULL,              -- fire | hit | destroyed | detection | command | miss

    -- 涉事实体
    source_id       INTEGER DEFAULT -1,         -- 发起者 armor_idx（-1=系统/玩家指令）
    source_name     TEXT DEFAULT '',             -- 如 "M1A2 #3"
    source_side     TEXT DEFAULT '',             -- ally | enemy | neutral
    target_id       INTEGER DEFAULT -1,         -- 目标 armor_idx
    target_name     TEXT DEFAULT '',
    target_side     TEXT DEFAULT '',

    -- 武器/弹药
    weapon_id       TEXT DEFAULT '',             -- 武器注册表 id
    ammo_id         TEXT DEFAULT '',             -- 弹药注册表 id

    -- 命中详情（仅 hit 事件填充）
    hit_part        TEXT DEFAULT '',             -- HULL | TURRET | ENGINE | TRACK | AMMO_RACK
    penetration     INTEGER DEFAULT 0,           -- 0/1
    pen_mm          REAL DEFAULT 0.0,            -- 穿透深度 (mm)
    angle_deg       REAL DEFAULT 0.0,            -- 入射角

    -- 位置
    pos_x           REAL DEFAULT 0.0,
    pos_y           REAL DEFAULT 0.0,
    pos_z           REAL DEFAULT 0.0,

    -- 扩展 + 观测
    extra_json      TEXT DEFAULT '{}',           -- 扩展字段（JSON，存放武器名/弹药名等人类可读信息）
    observed_by_player INTEGER DEFAULT 1,        -- 0=未观测 / 1=已观测 / 2=推测（模糊）

    created_at      TEXT DEFAULT (datetime('now'))
);

-- 索引：按时间查询（时间线视图）
CREATE INDEX IF NOT EXISTS idx_events_time   ON combat_events(sim_time);
-- 索引：按类型过滤（战果统计）
CREATE INDEX IF NOT EXISTS idx_events_type   ON combat_events(event_type);
-- 索引：按单位查询（单位战报）
CREATE INDEX IF NOT EXISTS idx_events_source ON combat_events(source_id);
CREATE INDEX IF NOT EXISTS idx_events_target ON combat_events(target_id);

-- 任务摘要表（每次任务/战役一行）
CREATE TABLE IF NOT EXISTS mission_summary (
    id              INTEGER PRIMARY KEY,
    mission_id      TEXT NOT NULL,
    start_time      REAL,
    end_time        REAL,
    total_ticks     INTEGER,
    outcome         TEXT,                       -- victory | defeat | stalemate
    summary_json    TEXT,                       -- 完整简报 JSON（缓存，避免重复生成）
    created_at      TEXT DEFAULT (datetime('now'))
);

-- 单位统计表（每次任务每个单位一行，任务结束时聚合写入）
CREATE TABLE IF NOT EXISTS unit_stats (
    id              INTEGER PRIMARY KEY,
    mission_id      TEXT NOT NULL,
    armor_idx       INTEGER NOT NULL,
    unit_name       TEXT,
    side            TEXT,
    shots_fired     INTEGER DEFAULT 0,
    hits_landed     INTEGER DEFAULT 0,
    hits_received   INTEGER DEFAULT 0,
    kills           INTEGER DEFAULT 0,
    destroyed       INTEGER DEFAULT 0,
    destroyed_by    TEXT DEFAULT '',
    ammo_consumed   REAL DEFAULT 0.0,
    fuel_consumed   REAL DEFAULT 0.0,
    survival_time   REAL,
    UNIQUE(mission_id, armor_idx)
);
```

### 9.3 环形缓冲区设计

```python
class RingBuffer:
    """固定容量环形缓冲区（Numba 不参与，Python list 操作）。"""

    def __init__(self, capacity: int = 1000) -> None:
        self._buf: list[CombatEventRecord] = []
        self._capacity = capacity

    def append(self, record: CombatEventRecord) -> None:
        if len(self._buf) >= self._capacity:
            self._buf.pop(0)  # 丢弃最旧记录（或可配置为溢出到 SQLite）
        self._buf.append(record)

    def drain(self) -> list[CombatEventRecord]:
        """取走全部并清空（写入 SQLite 前 drain）。"""
        records, self._buf = self._buf, []
        return records

    def __len__(self) -> int: ...
    def __iter__(self): ...
```

### 9.4 数据流：EventBus → Recorder → SQLite

```
[模拟层] EventBus.emit("hit", {...})
    │
    ▼ (每帧 drain)
[NarrativeEngine.tick()]
    ├── events = EventBus.drain()
    ├── trigger_system.evaluate(events, ...)
    │
    ├── recorder.on_events(events, snapshot)
    │       │
    │       ├── 1. 将 SimulationEvent → CombatEventRecord（映射字段）
    │       ├── 2. 从 snapshot 查 entity name / side（armor_idx → 名称映射）
    │       ├── 3. 追加到环形缓冲区
    │       └── 4. 每 N 帧或缓冲区满 → flush → SQLite INSERT
    │
    └── return NarrativeState
```

### 9.5 简报生成流程

```
CombatEventRecorder.generate_briefing()
    │
    ├── 1. flush()：缓冲区残余 → SQLite
    ├── 2. SQLite SELECT 查询所有事件
    ├── 3. BriefingGenerator.generate(events, ...)
    │       │
    │       ├── _count_kills()      → 双方击毁数
    │       ├── _build_timeline()   → 时间线文本列表
    │       ├── _per_unit_stats()   → 每个单位的服役记录
    │       └── _detect_highlights()→ 关键事件高亮
    │
    └── 4. → CombatBriefing（dataclass）
```

### 9.6 简报输出格式（CombatBriefing 扩展）

```python
@dataclass
class CombatBriefing:
    """战斗简报（由 BriefingGenerator 生成，渲染层消费）。"""
    # 基础信息
    mission_id: str = ""
    duration_s: float = 0.0
    outcome: str = "stalemate"              # victory / defeat / stalemate

    # 战果统计
    allies_initial: int = 0
    allies_remaining: int = 0
    allies_lost: int = 0
    enemies_initial: int = 0
    enemies_destroyed: int = 0
    enemies_remaining: int = 0
    total_shots_fired: int = 0
    total_ammo_consumed: float = 0.0

    # 时间线（摘要文本列表）
    timeline: list[str] = field(default_factory=list)

    # 单位战报
    unit_reports: list[UnitStatsReport] = field(default_factory=list)

    # 关键事件高亮
    highlights: list[str] = field(default_factory=list)
    # 如: "T+23.5s  M1A2 #3 首次命中 T-90A #7（尾翼稳定脱壳穿甲弹，穿透炮塔正面）"
    # 如: "T+45.0s  T-90A #7 被 M1A2 #3 击毁（弹药殉爆）"

    # 友军误击警告
    friendly_fire_incidents: list[str] = field(default_factory=list)

    # Arcadia 认知层（可选主观解读）
    arcadia_notes: str = ""                 # 如: "根据碎片分析，这发炮弹的穿透深度远超标准..."

---

> **一句话总结：叙事层 = TriggerSystem 盯事件 → DialogueEngine 放对话 → CommandFactory 产指令 → CommandQueue 注入模拟。**
> **战斗事件记录 = EventBus → CombatEventRecorder（环形缓冲区 + SQLite） → BriefingGenerator（多维聚合） → CombatBriefing（战果统计 + 时间线 + 单位战报 + Arcadia 认知）。**
