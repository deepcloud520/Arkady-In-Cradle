# 《Arkady in Cradle》story/ 与设计架构协调程度评审

> 评审日期：2026-09-11
> 评审范围：`story/`（33 文件）+ `Airia/narrative/` + `Airia/ui/` + `Airia/Arkady_In_Cradle/` + `docs/` + `AGENTS.md`
> 方法：主评审人逐点复核关键代码与 YAML 契约 + 2 个并行子代理通读 14 份设定稿/草稿/宣发稿
> 标注约定：
> - ✅ **已复核** = 主评审人亲自读代码/数据确认，附行号
> - 📋 **子代理抽取** = 来自设定稿通读，未由主评审人逐字复核
>
> **范围决策（2026-09-11，用户明确指示）**：**不新增海军/水面舰艇等新平台类子系统**，
> 相关设定条目一律记为「范围外」，**不作为待实现缺口**。本评审不提议为任何设定新增引擎子系统。

---

## 一、总体判断

**零件协调度很高，接缝协调度接近为零。**

`story/` 与架构在**概念层**高度同源——玩家身份、指挥粒度、暂停语义、战后简报、
"数据链不可信"这些细节两边都对得上；但在**生产路径**上，两者唯一的接触面是 3 个 YAML，
而这 3 个 YAML 在自己的注释里声明"与剧情无关、仅供测试"。

> 一句话：**代码消费的是占位符，canon 是代码消费不到的 markdown。**

协调度分档：

| 档 | 含义 | 条目数 |
|---|---|---|
| 🟢 **协调** | 设定承诺 ↔ 架构已实现且已接线 | 5 |
| 🟡 **有设计、未接线** | 架构有完整零件，生产路径关断 | 5 |
| 🔴 **引擎缺口** | 设定有承诺，架构无对应物 | 3（+3 范围外） |
| ⚪ **冲突/待裁决** | story 内部矛盾，或 story ↔ 架构直接背离 | 8 |

---

## 二、🟢 高度协调（这部分做得很好，不是客套）

| story 承诺 | 架构对应物 | 证据 |
|---|---|---|
| 玩家=指挥官 **Arcadia**、副官 **Kateryna** | 示例对话的 speaker 正是这两个名字 | ✅ `story/dialogues/prologue.yaml:22,46` —— 示例数据与设定**同源**，不是随手编的 |
| **"你下达指令，编队自主执行"** | `FormationAI` / `AirFormationAI` + `CommandQueue` + AGENTS.md 3.7 单向约束 | ✅ `Airia/lulilo/simulation/ai.py`、`simulation/commands.py` |
| **"战斗可以暂停"** | `NarrativeEngine._sync_pause` → `loop.sim_paused` + 叙事菜单/快捷键 | ✅ `Airia/narrative/engine.py:317-320` |
| 战后战报 / 翻阅情报 / 查看编队状态 | `CombatEventRecorder`（SQLite + 环形缓冲）+ 四 Tab 面板 | ✅ `Airia/narrative/recorder.py`、`Airia/ui/{archive,briefing,detail}_panel.py` |
| 对话/触发器**数据驱动**（YAML→Pydantic） | `dialogue.py` 3 个 BaseModel + `triggers.py` 4 个 BaseModel + 白名单 AST 求值 | ✅ schema 与 `story/dialogues/*.yaml` 完全吻合 |

**结论**：`story/` 与架构是同一个设计思路的两个投影，**不需要方向性调整**——缺的是收口。

---

## 三、🟡 有设计、代码在、但生产路径未接线

**这是协调程度最严重的失守区**：设定承诺的机制"存在但玩家永远看不到"。

| # | story 承诺 | 架构实况 |
|---|---|---|
| 1 | 剧情主线（`story/` 全部 canon md） | ✅ `Airia/ui/main_window.py:132` 函数体第一行 `return  # 暂不加载 story/ 对话树`，其后 8 行加载逻辑是**不可达死代码**（调用点 `:80` 存在但被早退拦住）。产品定位"剧情永远在场"，实际"剧情根本不在场" |
| 2 | 选项按条件显示 | ✅ `ChoiceOptionConfig.condition`（`dialogue.py:42`）**被完全忽略**：`_enter` 只取 `o.text`（`:273-274`），`advance` 不校验条件（`:212-221`）。`prologue.yaml:42` 的"等待更多侦察情报"条件选项**永远可见可点**。📋 `tests/test_narrative.py:195-203` 更以相反断言固化了该 bug（既有 `docs/架构评审报告.md` §3.1 已记录） |
| 3 | **"数据链共享情报，也可能在共享错误信息"** | ✅ Phase 8 零件**全部写完**（`TwinPool`/`TwinEngine`/`DataLinkNetwork`），但全库**只被测试引用**；`render/adapter.py from_snapshot_twin` 无生产调用者 |
| 4 | **"屏幕上的一切都不需要被完全信任"** | ✅ 同上：`FormationAI` 仍走全知接触；执行/瞄准仍读真值（`ai.py:974/1005/1135`） |
| 5 | 玩家选择**改变战局** | ✅ **断的**：`loop._apply_command`（`loop.py:156-169`）对非 STRIKE 指令**只 emit 一个 `command` 事件**；`Arkady_In_Cradle/world1.py:372-378` 的 handler 只处理 STRIKE（`ai.py:1347-1348` 非 STRIKE 直接 `return False`）。`HOLD`/`REDEPLOY` **零实际效果** = 玩家选择目前是纯日志 |

### 3.1 主评审人补充的三条（既有 `docs/架构评审报告.md` 未覆盖）

1. **story 数据与场景装配已脱节**
   `prologue.yaml:38` 的 `waypoint: [120, 80]` 与当前地图不匹配——`world1.py:114` 现为
   15km 扁平图（`amplitude=0`），战场在实际世界坐标 `(7500, 7500)` 附近、栅格 400×400。
   即便 `REDEPLOY` 实现，`[120, 80]` 也是一个任意点。

2. **`pause_sim` 三方不一致 = 那 2 个失败测试的真实根因**
   `story/triggers/chapter1.yaml:19,33,44` 三条触发器全设 `pause_sim: false`；
   而 `TriggerAction.pause_sim` 默认 `True`（`triggers.py:235`），
   `docs/phase4-narrative-design.md:228` 的示例也是 `true`。
   `tests/test_narrative_integration.py:83` 断言 `engine.is_paused` 为真 → 必然失败。
   `docs/dev-progress.md:107-108` 把它记为"环境/工作区既有状态"——
   ✅ 实为 **story 数据与引擎暂停契约不一致**，不是环境问题。

3. **~~`CommandFactory` 指令词汇表落后于引擎~~（2026-09-11 撤回：原判断有误）**
   ✅ 复核更正：`narrative/commands.py:22-34` **已完整映射全部 5 个指令**
   （REDEPLOY/STRIKE/HOLD/RESUPPLY/DISENGAGE + 小写别名），与
   `simulation/commands.py:23-27` 的枚举一一对应，**此处无缺口**。
   （原判断源于一次不完整的关键字检索。）真正的断点不在词汇表，而在**消费端**：
   模拟层只消费 STRIKE，见 §三.5。

---

## 四、🔴 引擎缺口与范围外条目

### 4.1 真正的引擎缺口（3 条，均在既有能力内可收口）

| # | story 承诺 | 缺口 | 建议 |
|---|---|---|---|
| 1 | **"超过 2000 个单位同屏推演"**（`story/.../关于AIC.md:21`） | 池上限（2048 装甲/16384 弹丸）存在，但 `world1.py:134-143` 实装 `capacity=96` 装甲 + 4 机；且实测 AI 788ms/tick（1s 预算的 79%） | 这是**性能与文案口径**问题，不是新子系统：要么压性能，要么改文案 |
| 2 | **"战斗本身可以跳过，但模拟结果会影响剧情走向"**（`:21`） | `loop.run_headless`（`loop.py:220`）存在，但 `docs/phase4-narrative-design.md:663-667` 明写"**暂缓设计**"；引擎只有 `skip_active_dialogue`（跳过**对话** ≠ 跳过**战斗**） | 复用既有 `run_headless`，属管线收口 |
| 3 | UI 污染/噪点乱码、朋友列表灰度化、双重文字、VR 会议室 0.5s 第二层画面、100+ 封邮件、调查系统 | 叙事层只有 `text/choice/branch/action` **四类节点**，无"叙事表现指令"概念 | 建议给叙事层加**表现层扩展点**，而非改引擎 |

### 4.2 范围外条目（**已决策：不新增子系统，不纳入计划**）

> 本节条目不构成待办，仅作为"设定与实现边界"的档案记录，防止后续反复提出。

| story 设定 | 范围外原因 |
|---|---|
| **水面舰艇 / CSG / 航母**（`关于AIC.md:7,21`；`ACT猫娘脚本.md:30,32` 伊丽莎白级、指挥 5 个 CSG 登陆） | **用户 2026-09-11 明确决定：不新增海军类子系统**。引擎池现有 `Armor/Box/Projectile/Emitter/Aircraft/AirFormation/Twin/FormationAI`，无 `ShipPool`；全库 `naval\|舰` 仅 `entities/twin.py:10` 一句"未来舰船"注释。**Act 49-50 的航母/海峡登陆场景在当前架构内不表达**，属设定的"展示性内容"，不是引擎职责 |
| **太空/轨道战、天基电磁炮、七代机亚轨道 80–120km** | 同上：不新增轨道/太空子系统。引擎飞机巡航 3000m、气动 3-DOF 点质量；世界为 15km 栅格 |
| **采购系统**（Provider/Solution/License/序列号生命周期/外交定价） | engine 零实现，但 **`第4版.md` 已自行丢弃该设定**，故只需阻止其回流 canon |
| **"声音是信息通道"**（`关于AIC.md:23-35`：近/中/远三层、编队共享感知、通讯中断白噪音） | 全库 `QSound\|audio\|sounddevice` **零命中**。**不新增音频子系统**——但这是商店页的**独立卖点章节**，属对外承诺漂移，建议改文案或降级为 HUD 文字通道（见 §七.2） |

---

## 五、⚪ 冲突 / 待裁决

### 5.1 story 内部（📋 子代理抽取，需一次 canon 裁决）

| 冲突项 | 两条（或多条）线 |
|---|---|
| 核爆地点 | 第一版"新英格兰" ↔ 设定集/整合版/第4版"伦敦" |
| Act 编号制 | 4 Act（任务 1-15/15-30/31-45/46-50）↔ 第4版"50 ACTs × 7 章节" |
| Act 48 地点 | `ACT猫娘脚本.md:20` "顿特莫克" ↔ 第一版/设定集"巴黎 Aracradle 总部" |
| 第二次乌克兰战争 | 2031–2042 / 2032–2040 / 2034–2046 **三说** |
| Cerber 改名 | 2041 装罐改名 ↔ 2056 崩溃后改名 Cerberus |
| Kateryna | "0.01% 误差" ↔ "她的家在那 1% 里" |
| Tsamica | 有读心能力（旧稿）↔ 渗透用机体（新稿，能力消失） |
| 组织名 | CC公司 / Contelligenc Group / Contelligenc System / CC系统 / CCG |
| Act 49 目标城市 | 设定集同节自相矛盾："沿海城市（朴茨茅斯/南安普顿），**非伦敦**" ↔ "进攻内陆伦敦" |
| 标语 | "to be **seen** and connected" ↔ 整合版 §5.1 "**saw**"（§6.4 自认语法错误） |
| 整合版 §1.4 | 表内"车臣战争(1994–2000)"排在"沙漠风暴(1991)"之前（时序错） |
| 设定集 | Kateryna"形象参考"段与 Rhein-chan 完全重复（复制粘贴） |

### 5.2 story ↔ 架构直接背离（需立刻处理）

| # | 冲突 | 说明 |
|---|---|---|
| 1 | **`story/` 承载了已废弃的引擎设计** | 第一版/设定集写"逻辑层：**八叉树**存储体素数据…Morton 序"，而 AGENTS.md 明载八叉树已于 **2026-08-10 弃用**，改列式 span 高度场。**设定文档在教后来者一个已被否决的架构** |
| 2 | **尺度** | 设定是**战区—洲际 + 轨道级**（Act 49 指挥 5 个 CSG；场景含巴黎/伦敦/新英格兰/南非/台海/北非/全轨道）；AGENTS.md Phase 9 目标 15km×15km~400km×400km。400km 可覆盖区域战役，**洲际与轨道跨域不在规划内**（轨道部分见 §4.2 范围外） |
| 3 | ✅ **引擎名已统一（2026-09-11）** | 原为 `ACT猫娘脚本.md:2`"Airia 指挥系统" ↔ 代码包名/预告"Airis" 混用。**已全库改名为 `Airia`**（包目录 `Airis/` → `Airia/`，770 处 / 158 文件），设定正文与代码名实一致。唯一遗漏面：仓库外 Copilot memory 目录（见 §八） |
| 4 | **示例 YAML 场景感与当前地图不符** | `prologue.yaml:16-17` "沿河谷逼近"（当前地图 `amplitude=0` **无河谷**）；`first_blood.yaml:16` "T-90 被击毁"（`world1.py` 现为 M1A2/T-14）；`tests/test_narrative_integration.py:48,56` 的 registry 也是 `"T-90 #1"` |
| 5 | **DLC 双线互斥且都未归档** | `AIC-DLC.md`（世界政府/激光帆/记忆裁剪）↔ `AIC-DLC-设定-未归档.md`《Out of History》（**Arcadia 没有被设计出来**、冯·诺依曼探测器、450 艘聚变舰队）。同一目录两条**互斥时间线**，需 canon 裁决 |

### 5.3 目录治理本身（"协调程度"的元问题）

- `story/` 混放 **7 种体裁**：正式设定 5 份、剧本 1（`ACT猫娘脚本.md`）、
  LLM 草稿 2（`Anny与mini.md` 残留"用户提供了新的设定细节"等生成过程文本）、
  未归档 DLC 2、宣发 2（`关于AIC.md`/`AIC超前公开预告.md`）、
  无关文件 1（`Untitled-1.tex` 只有 Maxwell 方程组）、Obsidian 配置、参考图。
- **5 个版本稿并存且无 `canon.md` 指路**（第一版/整合版/第三版/第4版/设定集）——
  取用时无法判断哪份权威；📋 例证：`ACT猫娘脚本.md` 与 `第4版.md` 的 Act 编号制就不一致。
- ✅ **目录名拼写已修（2026-09-11）**：`story/Refererence/` → `story/Reference/`
  （原拼写与项目根 `Reference/` 形成两个近似目录，易误引）
- ✅ **canon 脚手架已建（2026-09-11）**：`story/README.md`（体裁分层 + 版本清单 +
  代码接口契约 + 范围外条目 + 维护约定）；AGENTS.md §九 已加指向
- AGENTS.md 对 `story/` 的全部描述只有一句："Obsidian 世界观笔记（中文 markdown），非源码"
  ——**没有 canon 标记、没有体裁分层、没有"哪份驱动代码"的说明**。

### 5.4 两个必须澄清的"看起来像缺口"的点

- **"存档"**：📋 10 份设定稿中**完全没有存档/战役编辑器/关卡编辑器描述**；
  唯一提及的是 `第二版Q&A.md` 的**明确否认**："我完全没说过存档，那是你自己加的设定"。
  → 引擎的 msgpack/h5py 存档是**纯工程能力**，**不得反向写入剧情设定**。
- **视觉小说表现力**：`Airia/ui/narrative_overlay.py` 只有文本 + 选项按钮 + 历史
  （`:41-72,82-99`），无立绘/表情/打字机/语音；`assets/textures/` 为空目录，
  而 `story/Reference/*.webp` 已有 6 张角色参考图。
  → 若 canon 要走"视觉小说"路线（`AIC超前公开预告.md`："视觉小说 + CMO 类军棋推演"），
  这是**必须补的表现层**，但属 UI 层而非引擎子系统（见 §七 P2）。

---

## 六、三个最关键缺口（排序）

1. **canon(md) ↔ code(yaml) 之间没有桥。**
   唯一上线的 3 个 YAML 自称"与剧情无关、仅供测试"，
   而 canon 全是代码消费不到的 markdown，且入口被 `return` 关断。
2. **对外契约与实现能力漂移。** `关于AIC.md` 是商店页（对外承诺）：
   海军（已决策范围外）、声音通道（无子系统）、2000 单位（实测 788ms/tick）。
   **对外承诺漂移比内部文档不符更危险。**
3. **战争迷雾是 story 的灵魂（"信息永远不足"），零件齐全但未接线。**
   投入产出比最高的一块：代码都在，缺的只是装配。

---

## 七、建议（分级）

| 优先级 | 动作 |
|---|---|
| **P0** | ✅ **已完成（2026-09-11）** 建 `story/README.md`：目录体裁分层、版本清单、代码接口契约、范围外条目、维护约定；**版本权威标注为「待作者确认」**（工程侧建议 `第三版.md` + `第4版.md`，未擅自生效） |
| **P0** | ✅ **部分完成（2026-09-11）** 两处废弃引擎描述（八叉树/Morton）已加 **⚠️ 工程侧标注**（正文未改）；`Refererence` → `Reference` 已改名；AGENTS.md §九 已加指向。⏳ **仍待裁决**：DLC 双线 canon、§5.1 冲突清单、§4.2 范围外的设定正文标注 |
| **P1** | 删 `Airia/ui/main_window.py:132` 的 `return`；实现 `condition` 选项过滤（**注意保留原始索引映射**，否则过滤后选错分支，参见 `docs/架构评审报告.md:347`）；对齐 `chapter1.yaml` 的 `pause_sim` 与 `tests/test_narrative_integration.py:83` 的断言 |
| **P1** | 接线 Phase 8（`TwinEngine`/`DataLinkNetwork`）→ 兑现"信息不可信"；给 `HOLD`/`REDEPLOY` 接真实效果（`CommandFactory` 侧无需改动，见 §3.1.3 更正） |
| **P2** | 出一份**机制需求映射表**（见附录 A）：story 承诺 → 引擎模块 → ✅/🟡/🔴/范围外 状态；把 §4.2 的范围外条目一并登记，止损商店页漂移 |
| **P2** | 叙事表现层：给 `NarrativeOverlay` 加立绘/表情槽位（`assets/textures/` + `story/Reference/`），并按需扩展 YAML 表现节点类型（§4.1.3） |
| **P3** | 让 `docs/dev-progress.md` 的失败归因更精确：把 narrative 的 2 个失败从"环境问题"改为"story 数据 `pause_sim` 契约不一致" |

---

## 八、施工记录

### 2026-09-11（首轮施工：P1 代码/契约 + P3 归因）

**已完成（代码 + 数据 + 测试，均已验证）**

| 项 | 落点 | 验证 |
|---|---|---|
| 条件选项**引擎侧**过滤 + 可见索引→原始索引映射 | `Airia/narrative/dialogue.py`（`_visible_option_indices` / `DialogueState.option_indices` / `activate/advance/current_state` 透传 `eval_ctx`） | 新增回归 `test_visible_index_maps_to_original_option`（隐藏中间选项 → 可见索引 1 必须走原始索引 2） |
| 叙事层**接线** | `Airia/ui/main_window.py::_load_story` 删除 `return`；加载 3 棵树 + 3 条触发器；按场景注入 `has_scout_report` | `tests/test_narrative_integration.py` 5 用例全绿 |
| 暂停契约对齐（§3.1.2） | `story/triggers/chapter1.yaml`：battle_start / first_blood → `pause_sim: true`；时间警告 → `false` | 原 2 个失败用例转绿 |
| 时间警告对话引用修正 | 新增 `story/dialogues/time_warning.yaml`（原指向 `prologue_001`，会把序章重放一遍） | 触发器加载计数用例 |
| **玩家选择真正改变战局**（§三.5） | 新增 `Airia/lulilo/simulation/command_router.py`（`GroundCommandRouter`：HOLD→DEFEND 停机动仍可开火 / REDEPLOY→路标点+目标区 / RESUPPLY→弹药燃料结算 / DISENGAGE→背离敌重心）+ `world1.py` 装配（叙事别名 `7th_armor` → 本方坦克编队） | 新增 `tests/test_command_router.py` **12 用例全绿** |
| waypoint 坐标失配（§3.1.1） | `story/dialogues/prologue.yaml` `[120, 80]`（网格）→ 世界坐标 `[7350.0, 7200.0]` | 路由测试 + 集成测试 |
| 失败归因更正（P3） | `docs/dev-progress.md`："环境问题" → **"story 数据暂停契约不一致"** | —— |
| 测试可脱文件运行 | 新增公开 API `DialogueEngine.register()`（dict → Pydantic → 注册）；`tests/test_narrative.py` 5 个用例改内存构造 | `TestDialogueEngine` 类 5 passed（原全被沙箱 tmp_path 阻断） |

**验证口径**：`tests/test_narrative_integration.py` + `tests/test_command_router.py` = **17 passed**；
另有一份 22 项独立校验（条件过滤 / 索引映射 / 触发器契约 / CommandFactory 5 指令）全绿。

### 2026-09-11（P0 文档轮）

| 项 | 落点 |
|---|---|
| **canon 脚手架** | 新增 `story/README.md`：7 类体裁分层表、版本清单（12 项）、代码接口契约（6 条约定）、已废弃引擎描述索引、范围外条目、DLC 双线待裁决、维护约定。**版本权威标注「待作者确认」**——工程侧只给建议，不代作者决定 |
| **废弃引擎描述标注** | `story/arkady in cradle/第一版.md` §三、`story/arkady in cradle/《Arkady in Cradle》世界观设定集.md` §三：加 ⚠️ 工程侧标注（八叉树 / Morton 序 → 列式 span 高度场），**正文未改动** |
| **目录拼写** | `story/Refererence/` → `story/Reference/`（6 张角色参考图随目录迁移） |
| **备忘录指向** | `AGENTS.md` §九：`story/` 条目补 canon / 契约 / 评审文档指引；`assets/` 描述据实修正（shaders 已有 3 个主题，非空） |

**待办**

- **P0（仍需作者裁决）**：DLC 双线归属、§5.1 冲突清单（核爆地点 / Act 编号制 /
  乌克兰战争年代 / Cerber 改名 / Kateryna 误差 / 组织名）、§4.2 范围外条目在设定正文的
  显式标注、版本权威生效
- **P1**：Phase 8 接线（`TwinEngine`/`DataLinkNetwork` → 兑现"信息不可信"）
- **P2**：叙事表现层（立绘/表情/表现节点）+ 机制映射表落档
- **改名遗留（需你决定）**：仓库外 Copilot memory 目录残留 19 处 `Airis`
  （`dev-progress.md` 1 + `dev-progress-archive.md` 18）；沙箱禁止写仓库外路径，
  需 `danger-full-access` 授权或由你在非沙箱会话处理。

### 2026-09-11（引擎改名轮：`Airis` → `Airia`）

| 项 | 落点 / 结果 |
|---|---|
| **动机** | 设定正文用「Airia 指挥系统 / Airia 模块」，而代码包名是 `Airis` → 名实不符（§5.2.3） |
| **可行性前提（实测）** | 764→770 处全部为 `Airis`，**0 处** `airis`/`AIRIS`；正则 `Airis\w` **零命中**（无更长标识符）；全库无 `import_module`/`sys.modules["Airis…"]` 动态导入；`.vscode`/`.github`/`pyproject.toml` 零引用 |
| **工具** | 新增一次性脚本 `rename_airis_to_airia.py`（UTF-8 显式 + `newline=""` 保换行；dry-run / apply / verify 三模式；先备份再改） |
| **备份** | `_backup_airis_to_airia_20260911.zip`（878 文件 / 6.2 MB）——验证无误后可删 |
| **执行** | 替换 **158 文件 / 770 处** → 目录 `Airis/` → `Airia/`；清 `__pycache__` + Numba 缓存（`clear_numba_cache.py --all -y`） |
| **验证** | 残留 `Airis` **0 处**；**83 个模块全子包导入成功、0 失败**；全量回归 **715 passed / 0 failed / 11 errors**（基线 680 passed / 2 failed / 15 errors —— 失败清零，errors 全为沙箱 `tmp_path` 限制） |
| **耦合点（替换已覆盖）** | `main.py` import、`clear_numba_cache.py` 的 `default=["Airia"]`、`code_statistc.py` 统计根目录、各 `__init__.py` docstring、`AGENTS.md` §4.1/§五/§九 |
| **未留兼容 shim** | AGENTS.md 规则 5（`_` 前缀 + 兼容别名）针对**模块废弃**，不适用于包改名；留同名空壳包会重新引入混淆且 `Airis.lulilo.*` 子模块导入仍会断 |
- **环境限制（已部分缓解）**：pytest `tmp_path` 在本沙箱下 `PermissionError`
  （既有 15 errors，非本次引入）。本次新增公开 API `DialogueEngine.register()`
  （dict → Pydantic → 注册，免文件 IO），并把 5 个不依赖文件读写的用例改为内存构造
  → **42 passed / 10 errors / 0 failed**（变化：原 15 errors + **2 failed** → 0 failed）。
  剩余 10 errors 全部是**真正需要临时文件**的用例（YAML 加载语义本身），属沙箱限制

**未改动（避免越权改创作内容）**

- `story/**/*.md` 设定稿正文：八叉树等已废弃引擎描述仍在，**待 canon 定了再动**
  （其中"哪些版本权威 / DLC 走哪条线"属创作决策，不由工程侧单方面决定）

---

## 附录 A：机制需求映射表（P2 交付物草案）

| story 承诺（出处） | 引擎模块 | 状态 |
|---|---|---|
| 编队自主作战（`关于AIC.md:15`） | `FormationAI`/`AirFormationAI` | 🟢 已接线 |
| 战斗可以暂停（`:19`） | `sim_paused` + 叙事菜单 | 🟢 已接线 |
| 战后战报/情报翻阅（`:17`） | `CombatEventRecorder` + 3 面板 | 🟢 已接线 |
| 剧情主线在场（全部 canon md） | `ui/main_window.py` | 🟢 已接线（2026-09-11） |
| 条件选项可见性 | `dialogue.py:_visible_option_indices` | 🟢 已接线（引擎侧过滤 + 索引映射） |
| 数据链共享/错误信息（`:17`） | `TwinEngine`/`DataLinkNetwork` | 🟡 仅测试调用 |
| 信息不可信/战争迷雾（`:17`） | `from_snapshot_twin` | 🟡 仅测试调用 |
| 玩家选择改变战局（`prologue.yaml` actions） | `simulation/command_router.py` + `world1.py` | 🟢 已接线（HOLD/REDEPLOY/RESUPPLY/DISENGAGE） |
| 战斗可跳过（`:21`） | `run_headless` | 🔴 设计暂缓（`phase4 §四`） |
| 2000 单位同屏（`:21`） | 池上限 | 🔴 实装 96 + 性能未达标 |
| UI 污染/邮件/调查表现 | 叙事节点类型 | 🔴 无表现指令概念 |
| 水面舰艇/CSG/航母（`:7,21`） | — | ⚪ **范围外（已决策）** |
| 太空/轨道战（设定集 §3.4） | — | ⚪ **范围外（已决策）** |
| 声音是信息通道（`:23-35`） | — | ⚪ **不新增子系统；文案待收口** |
| 采购/解决方案式系统（设定集 §三） | — | ⚪ 设定已自行丢弃 |

## 附录 B：评审方法与可复现性

- 设定侧：2 个并行子代理分别通读
  （a）设定集 / 整合版 / 第一版 / 第二版Q&A；
  （b）第三版 / 第4版 / 两份 DLC / 两份宣发 / `Anny与mini.md` / `ACT猫娘脚本.md` / `Untitled-1.tex`，
  并按"世界基本盘 / 军事设定 / 玩家定位 / 叙事结构 / 机制描述 / 矛盾点"六段结构化抽取。
- 架构侧：主评审人亲自读取 `narrative/{dialogue,triggers,engine,commands}.py`、
  `simulation/{loop,commands,ai}.py`、`ui/{main_window,narrative_overlay}.py`、
  `Arkady_In_Cradle/world1.py`、`story/dialogues/*.yaml`、`story/triggers/chapter1.yaml`、
  `tests/test_narrative_integration.py`、`docs/phase4-narrative-design.md`、`docs/phase5-ui-pipeline-design.md`。
- 本评审**未运行测试**（只读环境）；§3.1 的 2 个失败用例由
  `docs/架构评审报告.md` 实测 + 本次静态证据链双向确认。
- 与 `docs/架构评审报告.md` 的关系：**复核并确认**其 §3.1、§3.4；
  **新增** §3.1(1)(2) 两条 + §3.1(3) 的自我更正（原判断有误，已撤回）、
  §4.2 范围外登记、§5.3 目录治理、§5.4 两点澄清。

---

**一句话结论**：`story/` 的世界观与 `Airia/` 的架构在**想法上是一家人**（连"数据链不可信"这种细节都两边都有），
但两者之间**缺一份 canon 契约和最后一段接线**——最贵的不是写代码，是决定"哪份设定说了算"。
