# Phase 8 数据链体系作战设计 —— 认知孪生战争迷雾 + 三军 AI 升级

> 状态：**设计定稿（2026-08-14，规划）**
> ✅ **实施状态（2026-08-17）：步骤①~⑧⑩ 落地**（磨盘战术 ⑨ 按用户要求留待后续；
> ④ 渲染切换 = `WorldAdapter.from_snapshot_twin` 数据层 + 状态映射，视觉主题映射
> 留渲染层迭代）。全量回归通过（376+，含既有 6b/6c/7）。
> 定位：本文是 `数据链与战争迷雾设计.md`（2026-08-12，概念设计）的**实施级细化**——
> 把「物理世界 / 认知孪生」概念落到 SoA 接缝；同时是 `phase6-ai-design.md`（编队 AI）、
> `phase7-air-design.md`（空中部队）的**消费者**：数据链是 AI 的感知总线，战争迷雾是
> AI 与玩家的共同视界。
>
> 读前必读：`数据链与战争迷雾设计.md`（Twin 概念）、`phase6-ai-design.md`（接触表/指挥链）、
> `phase6-guidance-pipeline.md`（`guid_comm`/`guid_target_pos` 坐标制导）、
> `phase7-air-design.md`（空中编队 AI 框架）。本文复用其 SoA 约定与既有接缝，不重复论证。

---

## 〇、总纲：一切 AI 只认 Twin，数据链是 Twin 的唯一载体

### 0.1 一句话定义

> **Phase 8 = 封印上帝视角。** 从本阶段起，AI（编队/单位/玩家）接触到的所有敌情一律来自
> **认知孪生（Twin）**，孪生由「传感器观测 + 数据链共享 + 时间衰减 + 情报注入」驱动；
> 数据链好坏（通信簇大小、链路质量）直接决定孪生与物理世界的偏差，即战争迷雾的浓度。

### 0.2 核心原则（延续 6c 拍板）

1. **物理世界 Real 永远只属于引擎**：渲染、AI、叙事、玩家指令全部只读 Twin，绝不读真值
   （唯一例外：引擎内部的物理结算）。
2. **数据链去中心化**（6c 已拍板）：通信簇按链路质量动态成簇；中央簇（含玩家 HQ）与
   失联簇各自为战；**玩家指令只达中央簇且优先级最高**。
3. **AI 只决策不执行**（沿用）：支援请求、火力分配、制导中继全部走既有接口——
   `FiringControl.fire()`、`guid_target_pos` 写入、`EventBus` 事件、命令队列。
4. **热路径只做向量化**：孪生更新、链路图重建是 1s 级 Python 遍历（< 0.1ms 预算，
   概念文档已论证）；毁伤/碰撞/气动内核全部不动。

### 0.3 与既有接缝的对应关系（已逐一核实源码）

| 本阶段需求 | 现成接缝 | 来源 |
|---|---|---|
| AI 换用孪生感知 | `FormationAI(contact_provider=...)` 注入点（默认全知，可换源） | 6c 步骤② |
| 通信质量 | `ArmorPool.comm` / `FormationAIPool.comm`（0~1） | 6c 步骤① |
| 通信断裂自保 | `FormationAI.COMM_THRESHOLD`（<0.5 收不到指令→自主自保） | 6c 步骤② |
| 数据链制导 | `guid_target_pos` 坐标制导（**发射后改坐标 = 第三方中继制导，天然支持**） | 6b |
| 人在回路门控 | `guid_comm`（< comm_threshold 停导）+ `ManualOverride.HOLD/REIGNITE` | 6b |
| 延迟情报注入 | `EventBus.emit/drain`（调度队列本阶段在 TwinEngine 内实现） | Phase 3 |
| 渲染消费 | `WorldAdapter.from_snapshot`（改读 twin 数组，§四） | Phase 5a |
| 巡飞弹 | `ProjectilePool` 弹丸 + `LOITER=5`（兼侦察上报，§五.4） | 6c 定稿 |
| CAS 任务 | STRIKE 指令的 target_kind（执行者=Phase 7 空中单位） | 6c 拍板 |
| 网络图 | `networkx` 依赖已就位 | Phase 0 |

---

## 一、认知孪生：`TwinPool`（新建 SoA 池）

### 1.1 为什么新建池而不是给 ArmorPool 加 twin_* 字段

- **物理池保持纯净**：`ArmorPool`/`AircraftPool`（Phase 7）是 Numba 热路径，孪生更新是
  Python 1s 级遍历——写分离保证扩容/并行纪律不打架（AGENTS.md 4.3）；
- **泛化到所有单位类型**：装甲、飞机（Phase 7）、未来舰船共用一套孪生，不需要每种池加字段；
- **玩家侧也需要孪生**：友军状态同样有通信延迟（自己部队也可能「失联」），Twin 覆盖敌我双方。

### 1.2 SoA 布局（`entities/twin.py` 新建）

```python
MAX_TWIN = 4096   # 覆盖 2048 装甲 + 128 飞机 + 余量

class TwinStatus(IntEnum):
    UNKNOWN = 0    # 未观测/超时降级（概念文档：? 标记）
    ACTIVE = 1     # 观测中（置信度决定渲染精度）
    KIA = 2        # 确认击毁（残骸 + 红叉）
    MIA = 3        # 信号丢失（灰色轮廓闪烁）

class TwinPool:
    # 键：unit_kind（0=armor / 1=aircraft）+ unit_idx → twin_idx（1:1 绑定，spawn 同步建立）
    twin_idx    = np.zeros(MAX_TWIN, np.int32)       # 回指物理单位索引（fancy 键）
    unit_kind   = np.zeros(MAX_TWIN, np.uint8)
    active      = np.zeros(MAX_TWIN, np.bool_)
    # ---- 孪生状态（玩家/ AI 看到的「现实」）----
    pos         = np.zeros((MAX_TWIN, 3), np.float32)   # 估计位置（含误差）
    vel         = np.zeros((MAX_TWIN, 3), np.float32)   # 估计速度（航迹外推用）
    status      = np.zeros(MAX_TWIN, np.uint8)          # TwinStatus
    confidence  = np.zeros(MAX_TWIN, np.float32)        # 0~1 置信度
    last_update = np.zeros(MAX_TWIN, np.float32)        # 最近观测时间（tick 计数）
    pos_err     = np.zeros(MAX_TWIN, np.float32)        # 位置误差半径（m，随衰减增长）
    source      = np.zeros(MAX_TWIN, np.uint8)          # 最近更新源（0无/1本队观测/2数据链/3友军报告/4叙事）
    # ---- 外推 ----
    last_vel_known = np.zeros((MAX_TWIN, 3), np.float32)  # 最后一次确知速度（dead-reckon 用）
```

### 1.3 绑定生命周期

- `ArmorPool.spawn` / `AircraftPool.spawn` → 同步 `TwinPool.bind(kind, idx)`（激活孪生，
  初始 `UNKNOWN` + `confidence=0`，友军单位初始由己方 C2 直接注入 ACTIVE）；
- `destroy` → 孪生**不销毁**（残骸/失联状态保留），仅由毁伤结算提交 KIA 观测（§二.1）；
- 序列化：twin 段进 `SimulationSnapshot.pools`，存档往返无损。

---

## 二、孪生更新引擎（`simulation/twin_engine.py` 新建，Python 1s tick）

### 2.1 四条更新通道（对齐概念文档 §三）

```python
class TwinEngine:
    def __init__(self, twin: TwinPool, detection, datalink, events, rng): ...
    def tick(self, loop, dt):
        # ① 传感器注入：detection 层输出（友军 comm > 0.8 才算「可上报」的观测）
        #    写 pos/vel/status=ACTIVE/confidence=观测质量/last_update=now
        # ② 数据链航迹共享：DataLinkNetwork.broadcast_contacts()（§三.3）→ 同簇单位
        #    互相校正孪生（多源融合取最高置信度，误差取最小）
        # ③ 时间衰减：now - last_update > T_STALE(10s) → confidence *= 0.99^Δt
        #    < 0.3 → 降级 UNKNOWN；原 KIA 但置信衰减 → 回退（误报重估）
        # ④ 延迟报告注入：pending 队列 (delay_ticks, report) 逐 tick 倒计时 →
        #    到点写孪生（叙事/友军报告，confidence 由报告源定）
```

### 2.2 观测质量 → 置信度/误差映射

| 观测源 | confidence | pos_err | 备注 |
|---|---|---|---|
| 本编队直瞄/雷达（comm ≥ 0.8） | 0.9 | 传感器噪声（detection 输出已有滤波质量） | 最高精度 |
| 数据链中继航迹 | min(源置信度, 链路质量) | 源误差 + 传播误差 | 逐跳衰减 |
| 友军无线电报告 | 0.6 | 100m（目视级） | 延迟 5~30 tick |
| 叙事层注入 | 按剧情参数 | 按剧情参数 | §九 |

### 2.3 死亡的双通道报告（概念文档 §五 场景的落地）

物理击毁 ≠ 孪生 KIA：
- **直接观测击毁**（击毁者 comm ≥ 0.8）→ 立即 KIA + confidence 0.95；
- **未观测** → 孪生保持 ACTIVE → 通信中断 → MIA（信号丢失）→ 后续侦察/报告 → KIA。
- 概念文档「地雷命中 → 5 tick 通讯中断 → 30 tick 无人机确认」场景作为**验收用例**写死进测试（§十一）。

---

## 三、数据链网络（`simulation/datalink.py` 新建，networkx）

### 3.1 链路模型

图节点 = 活跃单位 + 编队指挥节点 + 玩家 HQ；连边条件（两点间链路质量）：

```
link_q(a, b) = comm_a × comm_b × LOS(地形遮挡) × 距离衰减 × (1 - ECM 压制)
```

- `comm`：SoA 已有（6c），单位级通信能力；
- `LOS`：复用 `detection/los.py` DDA（地形遮挡断链）；
- 距离衰减：超视距（BVR）节点只连卫星/中继/空中中继（数据链拓扑的自然优势：**飞机天然是中继节点**）；
- 建边阈值 `LINK_THRESHOLD=0.5`，连边权重 = link_q。

### 3.2 簇与可达性

- `networkx.connected_components` → 通信簇；含玩家 HQ 的簇 = **中央簇**（玩家指令可达）；
- `route_quality(a, b)`：两节点间最优路径的连乘质量（决定航迹中继精度与指令延迟）；
- **重建频率**：1s tick 全量重建（≤ 2200 节点规模 < 1ms，可接受）；爆炸/单位死亡不即时重建。

### 3.3 航迹广播（孪生的数据链通道）

```python
def broadcast_contacts(self, twin, formations):
    # 逐簇：收集簇内所有成员的「本队观测」孪生航迹
    # → 按目标 ID 合并（confidence 取 max，pos_err 取 min，位置加权融合）
    # → 写回簇内所有成员可读的共享接触视图（AI 的 contact_provider 消费此视图）
```

### 3.4 指令流

- **玩家指令**：只注入中央簇（6c 约定落地）——非中央簇成员执行 `Commander` 队列时直接跳过；
- **编队指令**：FormationAI 对 comm < `COMM_THRESHOLD` 的成员不写 order（6c 已实现，
  本阶段只需把 `comm` 换成「链路质量」语义，见 §五.1）；
- **支援请求/火力任务**（§五/§六）：走 `EventBus` + 簇内广播，跨簇不可达 → 自然丢失。

---

## 四、战争迷雾的渲染与 UI 消费

### 4.1 `WorldAdapter` 切换到 Twin（`render/adapter.py`）

`from_snapshot` 目前读 `armor.pos/yaw/active` 真值。Phase 8 后：

- 单位渲染数据源改为 `TwinPool`：位置用 `twin.pos`（**真值不进 adapter**）；
- `UnitRenderStatus`（已存在）映射孪生状态：

| TwinStatus | confidence | 渲染表现（主题商店兼容） |
|---|---|---|
| ACTIVE | > 0.8 | 完整模型 + 高亮光环 |
| ACTIVE | 0.5 ~ 0.8 | 半透明模型 + 抖动偏移（pos_err 驱动） |
| ACTIVE | < 0.5 | 模糊接触框 |
| UNKNOWN | — | 大「?」标记（最后已知位置） |
| KIA | > 0.9 | 静态残骸 + 红叉 |
| MIA | — | 灰色轮廓 + 闪烁 |

- 构成主义/日系主题各自实现映射（概念文档 §六：虚线框+标尺 / 半透明虚影+马赛克），
  数据层统一，仅视觉差异。

### 4.2 情报偏差警告

`twin.pos` 与 `real.pos` 偏差超阈值 → UI 提示「情报偏差警告」（概念文档 §四）。
实现：引擎侧 1s tick 算偏差（仅引擎有真值权限），作为事件 emit，UI 订阅。

### 4.3 玩家视角 = 中央簇视角

玩家 UI 只显示**中央簇可报告的孪生**：失联簇的单位自动变为 MIA/UNKNOWN——
「自己部队消失在屏幕上」的战争迷雾体验（概念文档核心玩法）。

---

## 五、装甲部队 AI 升级：呼叫支援 + 态势规划

### 5.1 态势感知：接触表换源

`FormationAI` 的 `contact_provider` 注入孪生接触视图（§三.3）：

- `Contact` 扩展 `confidence/pos_err/kind`（空中/地面，配合 Phase 7）；
- 目标位置 = `twin.pos`（带误差），血量 = `twin` 报告值（可能滞后）；
- 评分内核 `score_targets` 升级：价值 = f(类型, 血量, 位置) × **confidence 折扣**
  （看不清的目标不该被优先攻击——迷雾影响决策）；
- 编队 `comm` 语义升级：成员指令可达性 = `datalink.route_quality(formation, member)`
  （替代简单阈值比较，6c 逻辑结构不变）。

### 5.2 支援请求链（新流程）

```python
class SupportRequest:        # EventBus 载荷（不是新 SoA 池）
    kind: Literal["artillery", "loiter", "cas"]   # 炮兵 / 巡飞弹 / CAS
    target_twin: tuple[int, int]                  # (unit_kind, twin_idx) 孪生目标
    priority: int                                  # 紧急度
    requester: int                                 # 编队 id

# 编队 AI 新增意图分支：目标超射程 / 高价值 / 自身火力不足 →
#   发 SupportRequest（仅当本编队在通信簇内）→ EventBus.emit("support_request", ...)
```

- **接收方**：炮兵编队（§六）、空中编队（CAS 任务，Phase 7 STRIKE 语义）、巡飞弹管理
  （§五.4）；
- **裁决**：同一簇内请求按 priority 排序，火力富余才接单（拒绝 → 请求方降级自处理）；
- **闭环验收**：装甲呼叫 → 炮兵响应 → 落点毁伤 → 孪生 KIA 报告回流（§十一集成用例）。

### 5.3 态势规划：编队意图升级

在 6c 意图 FSM（MANEUVER/ADVANCE/DEFEND/RETREAT/HOLD）之上加**规划层**：

- 输入：孪生接触表 + 己方编队状态 + 支援可用性（簇内炮兵/空中编队清单）；
- 输出：意图切换 + 支援请求决策（`_update_intent` 的扩展分支，不新增层）；
- 目标：让「数据链良好」的部队打得出配合（炮兵支援 + 编队钳形），「数据链中断」的
  部队退回 6c 的自保行为——**迷雾浓度直接决定 AI 战力**，这是 Phase 8 的玩法核心。

### 5.4 巡飞弹侦察上报（案例）（6c 定稿的落地）

巡飞弹（`ProjectilePool` + LOITER 阶段，6c 预留）在 Phase 8 增加**侦察上报**：

- LOITER 期间把 `guid_target_pos` 附近敌方真值（**仅当巡飞弹 comm 在簇内**）上报孪生——实际为AI从电视制导图像筛选目标
  （source=1 本队观测）——它是「可消耗的无人机」；
- 被击落 → 上报中断 → 孪生衰减回 UNKNOWN。

---

## 六、炮兵 AI 升级：自主接收 + 反炮规避

### 6.1 自主接受友军请求

- 炮兵编队（role=1）的 `FormationAI` 订阅 `EventBus` 的 support_request 事件；
- 任务排序：priority × 自身弹药 × 射程覆盖 → 接受/拒绝（拒绝回执请求方）；
- 执行：`fc.fire(shooter, target_idx=-1, aim_at=twin.pos)`（6b 的 aim_at 先例，零扩展）——
  **注意：炮兵瞄准的是孪生坐标（带误差）**，迷雾直接影响落点散布——设计意图。

### 6.2 反炮规避（shoot & scoot）

- **反炮链**：射击瞬间炮口闪光 = 瞬时 RF/IR emitter（`EmitterPool` 已有瞬时源机制，
  6b 诱饵雏形）→ 敌方 RWR/声测检测 → 敌方炮兵反击落点 = 我炮位孪生估计；
- **规避**：射击 N 发（`SHOOT_AND_SCOOT_SALVO=6`）→ 沿预案路线转移阵地
  （`FormationAI` 的 MANEUVER 现成路径）→ 新阵地重新解算；
- **验收**：不转移的炮兵被反炮命中率显著高于转移的（统计测试）。

---

## 七、空军数据链：A 导 B 射 + 超视距 + 协同规避

### 7.1 A 导 B 射（中继制导，架构零改动）

6b 定稿已写明 `guid_target_pos` 是坐标制导。Phase 8 落地其数据链语义：

```
发射平台 A（后机）发射 AAM（guid_comm 门控开启）
    │ 导引头未锁定 / 目标在数据链航迹中
    ▼
前出侦察/预警机 B（数据链成员）持续广播目标孪生航迹
    ▼
DataLinkNetwork 把航迹写入 A 的导弹 guid_target_pos（每 1s tick 中继校正）
    ▼
末端导引头激活（ACTIVE_RADAR 锁定）→ 交接 → 数据链退出（导引头接管）
```

- 导弹侧零改动：`guid_target_pos` 本来就是外部可写字段；`guid_comm < comm_threshold`
  停导逻辑（6b）自动处理「链路丢失 → HOLD/MEMORY」；
- 新增：`combat/engagement.py` 一个中继更新接口 `relay_guidance(proj_idx, new_pos, link_q)`。

### 7.2 超视距空战（BVR，Phase 7 的基础上加链路）

Phase 7 的 BVR 是「本机雷达锁定 → 发射」。Phase 8 升级为：

- **发射源两种**：本机雷达航迹 / 数据链共享航迹（A 导 B 射的简化）——
  共享航迹发射时 confidence 折扣进入命中概率修正；
- **超视距控制**：导弹飞行中允许编队指挥节点改写 `guid_target_pos`（战术改目标/诱饵识别），
  复用 `ManualOverride` 语义（HOLD=链路保持、REIGNITE=中继重启）。

### 7.3 协同规避

Phase 7 的单位级 RWR 规避升级为**编队级协同**：

- 来袭 AAM 航迹（本机 RWR + 数据链共享）→ 编队 AI 评估威胁扇区 →
  **协同诱饵**：一架抛干扰弹吸引导引头、另一架机动脱离（牺牲优先级：弹药耗尽者先诱敌）；
- 实现为 `AirFormationAI` 的战术子状态机（§八），不新增检测模型。

---

## 八、编队战术库（磨盘战术等）

### 8.1 战术位（`AirFormationIntent` 的战术子状态机，Phase 7 §十已预留枚举位）

```python
class AirTactic(IntEnum):
    NONE = 0
    GRINDER = 1    # 磨盘：双/多机轮转交替攻击——前机攻击后高速脱离，
                   #       后机在掩护位接力攻击并掩护前机脱离，如磨盘旋转持续施压，
                   #       使敌无法锁定单一目标（战术细节待用户确认）
    DRAG = 2       # 拖带：诱敌机故意暴露吸引，友机侧翼伏击
    SCISSORS = 3   # 双机剪式：交替交叉机动消解追尾
    BOOM_ZOOM = 4  # 能量战：俯冲攻击→高速爬升脱离，不对等缠斗
```

- 战术 = `AirFormationAI` 的意图扩展：CAP/ENGAGE 状态下根据兵力比/高度差/链路质量
  选择战术模板（Python FSM，复用 6c 模式）；
- 战术执行 = 单位 AI 机动原语组合（Phase 7 §三.3 的占位/规避/俯冲原语）；
- **磨盘战术的触发条件**：编队兵力 ≥ 2 且数据链质量好（轮转需要航迹共享），
  链路差时退回各自为战——再证「数据链决定战术上限」。

### 8.2 与 Phase 7 的关系

Phase 7 交付「单机 OODA + 编队任务 FSM」，Phase 8 只加**战术子状态机**与
**链路质量输入**，不重写飞控/气动/交战链。

---

## 九、叙事层接口

- **叙事读孪生**：`narrative/triggers.py` 的触发器改读 TwinPool（「指挥官以为单位被毁，
  其实只是失联」成为剧情分支素材）；
- **叙事写孪生**：`TwinEngine.submit_report(kind, source, delay_ticks, confidence)`
  供剧情注入虚假/真实情报（概念文档 §三.3 的友军报告通道）；
- **玩家选择**：Command 对象经中央簇广播（§三.4），失联部队收不到命令 → 剧情张力。

---

## 十、实施管线（步骤拆解，后续写 `phase8-datalink-pipeline.md`）

| 步 | 内容 | 依赖 | 验收 |
|---|---|---|---|
| ① | `TwinPool` + 绑定生命周期 + 序列化 | — | ✅ `entities/twin.py`（`test_phase8_twin_pool.py` 11 用例） |
| ② | `TwinEngine` 四通道（观测/共享/衰减/延迟注入） | ① | ✅ `simulation/twin_engine.py`（`test_phase8_twin_engine.py` 10 用例，含 MIA→KIA 时序） |
| ③ | `DataLinkNetwork`（链路图/簇/路由质量/航迹广播） | ① | ✅ `simulation/datalink.py`（`test_phase8_datalink.py` 11 用例；scipy 稀疏连通 + networkx 惰性） |
| ④ | 渲染切换：`WorldAdapter` 读 Twin + 主题视觉映射 | ①② | ✅ 数据层 `from_snapshot_twin` + `map_twin_render_status`（`test_phase8_adapter.py` 8 用例）；主题视觉留渲染迭代 |
| ⑤ | 装甲 AI：contact_provider 换源 + 支援请求链 + 评分折扣 | ②③ | ✅ `TwinContactProvider` + `_maybe_request_support` + `score_targets` confidence 折扣（`test_phase8_armor_ai.py`） |
| ⑥ | 炮兵 AI：自主接单 + shoot & scoot | ③⑤ | ✅ `_handle_support_events` 订阅接单（反炮闭环 6c 已有；`test_phase8_armor_ai.py`） |
| ⑦ | 巡飞弹侦察上报 | ②③ | ✅ `LoiterRecon` + `ProjectileType.LOITER`（`test_phase8_loiter.py` 5 用例） |
| ⑧ | 空军：`relay_guidance` + BVR 数据链发射 + 协同规避 | ③ + Phase 7 | ✅ `relay_guidance`/`fire_air(confidence)`/`_tick_coordinated_evasion`（`test_phase8_air_datalink.py` 5 用例 + 集成用例 4） |
| ⑨ | 编队战术库（磨盘等） | Phase 7 ⑥ | ⏸ **用户明确留待后续**（`AirTactic` 枚举位已预留） |
| ⑩ | 集成测试 + 性能 + 文档更新 | 全部 | ✅ `test_phase8_integration.py`（4 集成用例）+ `benchmark_phase8.py`（现实预算达标）+ AGENTS/dev-progress |

---

## 十一、测试与性能目标

- **单元测试**：每步 `tests/test_phase8_*.py`，全量回归零衰减（当前 402）；
- **集成用例**（概念文档场景写死）：
  1. 地雷命中 → 通讯中断 MIA → 无人机确认 KIA 的完整时序；
  2. 装甲呼叫炮兵 → 孪生坐标误差落点 → 毁伤 → 报告回流；
  3. 数据链中断的编队退回 6c 自保行为，链好的编队打出支援配合；
  4. A 导 B 射命中超视距目标；链路丢失 → HOLD → 导引头末端接管；
- **性能目标**：

| 场景 | 目标 |
|---|---|
| 孪生更新（2048 装甲 + 128 飞机） | < 0.1ms / 1s tick |
| 链路图重建（≤ 2200 节点） | < 1ms / 1s tick |
| 全帧影响（相对 Phase 8 前） | 增加 < 2ms（纯 Python 决策层） |
| 压测 10min | 无内存增长、孪生无泄漏（destroy 解绑） |

---

## 十二、范围外（明确不做）

- 卫星/预警机专职平台实体（数据链中继由既有单位兼任）；
- 电子战专用机（ECM 是单位属性不是平台类型）；
- 玩家手动操作数据链界面（Phase 5b UI 面板另立）；
- 完整信息论/多假设跟踪（孪生是单假设 + 置信度，不做粒子滤波）。

---

> 一句话：**Phase 8 的敌人不是敌军，是「不知道」——数据链有多好，部队就有多聪明，
> 迷雾有多浓，指挥官就有多瞎。引擎只做一件事：让 Real 与 Twin 的差距成为所有决策的
> 真实成本。** 🌫️
