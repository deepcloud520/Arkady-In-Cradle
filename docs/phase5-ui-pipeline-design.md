# Phase 5 UI 管线设计

> 状态：**Phase 5a/5b/5c/5d/5e 全部实施**（5a-5d 2026-08-13；5e 收尾 2026-08-15）
> 对应 AGENTS.md Phase 5（UI 集成与主题商店）
> 前置：Phase 0-3（引擎）✅ / Phase 4（叙事层，已实现骨架）✅

---

## 实施进度（2026-08-13 / 2026-08-15）

### Phase 5a 已完成 ✅

| 项 | 状态 |
|----|------|
| `Airia/render/adapter.py` WorldAdapter | ✅ 透传模式 + `_build_damaged_parts`（5 部位损坏聚合） |
| `Airia/render/theme.py` ThemeManager | ✅ discover / load / switch 热切换 |
| 着色器外提 `assets/shaders/default/` + `constructivism/` | ✅（anime 待 5d） |
| `gl_widget.py`：主题接入 + WorldAdapter + 单位标记实例化 | ✅ |
| `main_window.py` 骨架：QSplitter + 主题菜单 + 主循环 | ✅ |
| 验证 | ✅ 288 passed；单位红绿标记渲染正确；主题来回切换稳定 |

### Phase 5b + 5c 已完成 ✅

| 项 | 状态 |
|----|------|
| `Airia/ui/` 四面板（NarrativeOverlay / DetailPanel / ArchivePanel / BriefingPanel） | ✅ QTabWidget 4 Tab（对话/档案/简报/详情）+ 信号槽接线 |
| `Airia/render/picking.py` Picker | ✅ 屏幕坐标 → O(N) 最近单位（project_world 向量化，与 glm 逐元素一致） |
| `Airia/render/hud/overlay.py` HUDOverlay | ✅ 屏幕空间圆点标记 + 选中高亮（NDC 实例化，关闭深度测试） |
| `Airia/render/hud/log_widget.py` LogWidget | ✅ Qt 透明滚动日志（左下角叠加，200 行上限） |
| GLWidget 拾取 | ✅ hover/单击/Esc 信号 `unit_hovered`/`unit_selected`；拖拽阈值区分 |
| MainWindow 装配 | ✅ recorder + AmmoManager/FuelManager + set_unit_context + QTabWidget + LogWidget |
| 验证 | ✅ 299 passed；smoke_gui 通过；HUD 像素统计 ally=1585/enemy=2596 |

### Phase 5d 已完成 ✅（docs/anime-theme-design.md）

| 项 | 状态 |
|----|------|
| `assets/shaders/anime/`（vert + frag + theme.toml） | ✅ cel 3 级色阶 + 法线描边（平方锐化）+ 粉彩 + 大气透视 |
| 主题热切换（default↔constructivism↔anime） | ✅ 5 轮稳定 |
| 验证 | ✅ 299 passed；smoke_theme_anime 描边开关对照 0.89%；lum_peaks=2 |

### Phase 5e 已完成 ✅（2026-08-15，Phase 5 全部收尾）

| 项 | 状态 |
|----|------|
| `Airia/render/hud/text.py` FontAtlas + TextRenderer | ✅ PIL 字体图集（ASCII + 中文部位/阵营名 + ⚠）+ 实例化字符 quad 绘制 |
| HUD 单位文字标签（单位名，选中/悬浮高亮色） | ✅ GLWidget._render_labels，读 WorldAdapter.unit_name |
| HUD 损坏 ⚠ 摘要文字 | ✅ `⚠ 引擎 履带`（橙色，读 damaged_parts） |
| `Airia/render/effects.py` EffectLayer（弹道线/命中标记缓冲） | ✅ 纯逻辑可无头单测：fire→弹道线，hit/ricochet/destroyed→命中标记，墙钟时间老化 + 上限防刷屏 |
| 弹道线（射手→目标，渐隐） | ✅ HUDOverlay.render_lines 线段实例化渲染 |
| 命中标记（扩张圆环 + 十字，渐隐） | ✅ HUDOverlay.render_ring（u_mode=1 软边圆环）+ render_lines |
| DetailPanel 级别 2 真实速度 | ✅ 渲染层位置差分 + EMA（`_estimate_speed`，不动引擎；adapter.armor_speed） |
| 对话历史跨 Tab / 结束后保留 | ✅ NarrativeOverlay._set_idle 不再清空历史（新对话激活时整体替换） |
| `Airia/ui/controls.py` ControlMap（键盘→动作） | ✅ 空格暂停 / 回车推进 / 1-9 选选项（→ CommandQueue）/ S 跳过 / Ctrl+1/2/3 主题 / F1 帮助 |
| MainWindow 手动暂停（与对话暂停独立） | ✅ `_manual_paused` + 事件过滤器（GLWidget.installEventFilter） |
| 验证 | ✅ 435 passed（新增 tests/test_phase5_effects.py 15 用例） |

### 待办（仅剩 5d 留档项）

- **5d 留档**：深度剪影描边（early-z 失效，需 G-Buffer 方案 B）；HUD 主题化（Phase 8 孪生置信度）
- **5e 备注**：HUD 弹道线目前用「射手→目标当前世界位置」直线近似（不跟踪弹丸飞行轨迹，避免渲染层读抛射物池）；命中标记位置 = 目标单位位置（非精确命中点）——足够演示级视觉效果

### 实施中踩坑（已修复，重要）

1. **moderngl 实例化必须加 `/i` 后缀**：实例 buffer format `"3f 1f 3f"` 缺 `/i` 时 divisor=0（per-vertex），实例数据被当顶点连续读取 → 只画前几个错乱三角形。修复 = `"3f 1f 3f /i"`（源码 `FormatIterator::info()` 与官方 `docs/topics/buffer_format.rst` 确认）
2. **主题切回报 `InvalidObject`**：`switch` 释放旧 program 后必须移出缓存（`self._themes.pop(old.id, None)`），否则 `ensure()` 复用已释放 program
3. **主题切换地形变黑（GL context 非 current）**：菜单槽触发时 context 不保证 current，编译 program / 重建 VAO / 写 uniform 静默失败。修复 = `MainWindow._switch_theme` 整个 switch 包进 `gl_widget.makeCurrent()/doneCurrent()`；`_on_theme_switched` 里 GL 操作再包一层（防御）
4. **glm 矩阵转 numpy 的 to_list 转置陷阱（2026-08-13，HUD 圆重合/飞出根因）**：`glm.mat4.to_list()` 返回 **4 个列向量**（非扁平 16 元素），`np.array(to_list()).reshape(4,4, order='F')` 是 **no-op**（已 (4,4)）→ 矩阵被转置 → clip_w 错乱 500 倍（224.6→115480）→ **所有单位 HUD 圆投影到同一屏幕点（友军/敌军完全重合）+ 特殊角度 NDC 爆炸圆点飞出**。修复 = 改用 `np.frombuffer(mvp.to_bytes(), ...)`（与 GPU uniform 完全一致的 column-major 扁平）。教训：**identity 矩阵转置不变 → 单测测不出，必须加非 identity 相机投影对比测试**（test_project_matches_glm）
5. **HUD 深度测试剔除（2026-08-13）**：HUD 圆点 NDC z=0（深度 ~0.5），GL_LESS 下近处地形（深度<0.5）把标记剔除 → 低角度观察圆点消失。修复 = `_render_hud` 期间 `ctx.disable(DEPTH_TEST)`（HUD 语义=永远最上层），finally 恢复
6. **文字图集必须回退（2026-08-15，5e）**：HUD 文字依赖 CJK 字体文件，找不到字体时 `FontAtlas.available=False`，`TextRenderer` 不创建、渲染层自动跳过文字（不崩溃）。测试用 `pytest.skip` 处理无字体环境
7. **特效位置渲染时刻解析（2026-08-15，5e）**：弹道线/命中标记**不缓存事件发生时的世界坐标**，而是渲染时用 `WorldAdapter.armor_pos` 解析（单位移动后特效跟随，无需事件携带位置）；缺点 = 非精确轨迹/命中点（演示级足够，见「待办」备注）
8. **快捷键修饰键判定（2026-08-15，5e）**：`ControlMap` 的 Ctrl 分支必须**先判 Ctrl+数字**（且排除 shift/alt/meta），否则 Ctrl 会落入「其它修饰键不拦截」分支；事件过滤器只装在 GLWidget 上，避免抢走按钮/输入框的默认按键行为

---

## 零、现有资产盘点

### 0.1 渲染层（`Airia/render/`）— 已有

| 模块 | 文件 | 职责 | Phase 5 改动 |
|------|------|------|-------------|
| 轨道相机 | `camera.py` | `OrbitCamera`：透视投影 + 球坐标旋转/缩放 | **不改** ✅ |
| GL 控件 | `gl_widget.py` | `QOpenGLWidget`：地形网格渲染 + 相机交互 | **重构**：着色器外提 → 主题系统；内联 VERTEX_SHADER/FRAGMENT_SHADER 移到外部文件 |
| HUD 占位 | `hud/__init__.py` | 空壳，注释标明「待实现」 | **实现**：单位状态标记、弹道线、日志滚动条 |

### 0.2 UI 层（`Airia/ui/`）— 已有

| 模块 | 文件 | 职责 | Phase 5 改动 |
|------|------|------|-------------|
| 主窗口 | `main_window.py` | `MainWindow(QMainWindow)`：仅含 GLWidget + 状态栏相机信息 | **重度重构**：增加面板布局、菜单栏、工具栏、信号槽 |

### 0.3 叙事层（`Airia/narrative/`）— 已有

| 模块 | 文件 | 职责 | Phase 5 改动 |
|------|------|------|-------------|
| 中央调度 | `engine.py` | `NarrativeEngine.tick()` → `NarrativeState` | **不改** ✅（但需要 UI 层消费其输出） |
| 对话引擎 | `dialogue.py` | `DialogueEngine` + `DialogueState` | **不改** ✅ |
| 触发器 | `triggers.py` | `TriggerSystem` + `ConditionEvaluator` | **不改** ✅ |
| 指令工厂 | `commands.py` | `CommandFactory`（纯函数） | **不改** ✅ |
| 事件记录 | `recorder.py` | `CombatEventRecorder` + 简报 | **不改** ✅ |

### 0.4 模拟层接口（叙事层/UI 层消费）— 已有

| 接口 | 文件 | 用途 |
|------|------|------|
| `SimulationSnapshot` | `simulation/state.py` | 只读快照（pools + logistics + sim_time） |
| `EventBus.drain()` | `simulation/loop.py` | 事件列表（fire/hit/destroyed/miss/detection/command） |
| `CommandQueue.push()` | `simulation/commands.py` | 玩家指令注入 |
| `SimulationLoop.sim_paused` | `simulation/loop.py` | 暂停/恢复模拟 |

### 0.5 数据资产 — 已有

| 路径 | 内容 | Phase 5 用途 |
|------|------|-------------|
| `story/dialogues/*.yaml` | 对话树（prologue.yaml, first_blood.yaml） | `NarrativeOverlay` 加载显示 |
| `story/triggers/*.yaml` | 触发器配置（chapter1.yaml） | 已由 NarrativeEngine 消费 |
| `assets/shaders/` | 空目录 | 着色器主题文件的存放位置 |
| `assets/textures/` | 空目录 | UI 纹理/图标（预留） |

### 0.6 战争迷雾设计（Roadmap 预留）— 参考

> 详见 `docs/数据链与战争迷雾设计.md`（Phase 8 预留，当前不实现）。

该设计定义了「物理世界（Real）→ 认知孪生（Twin）」的双重世界模型：
- **Real**：模拟引擎维护的绝对真相（玩家永远不可直接见）
- **Twin**：传感器融合 + 时间衰减 + 叙事注入驱动的玩家认知（渲染层唯一数据源）

**Phase 5 的架构要求**：虽然 Twin 世界尚未实现，但渲染管线必须通过 **WorldAdapter** 抽象层读取数据，今日它是 Real 数据的透传代理，未来切换为 Twin 数据源时 **GLWidget / HUD / DetailPanel 零改动**。

---

## 一、管线总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    main.py（入口）                                │
│  QApplication → MainWindow → 组装全部面板                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                MainWindow（QMainWindow）                          │
│                                                                  │
│  ┌─ MenuBar ───────────────────────────────────────────────────┐ │
│  │  主题菜单 → ThemeManager.switch("constructivism"|"anime"|…) │ │
│  │  视图菜单 → 面板显隐切换                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ CentralWidget（QSplitter 水平）────────────────────────────┐ │
│  │                                                              │ │
│  │  ┌──────────────────────┐  ┌──────────────────────────────┐ │ │
│  │  │                      │  │                              │ │ │
│  │  │   GLWidget           │  │   SidePanel (QTabWidget)     │ │ │
│  │  │   (QOpenGLWidget)    │  │                              │ │ │
│  │  │                      │  │  Tab1: 对话面板               │ │ │
│  │  │   3D 战场渲染         │  │  Tab2: 档案库（预留骨架）     │ │ │
│  │  │   + 相机控制          │  │  Tab3: 简报面板               │ │ │
│  │  │   + HUD 叠加层        │  │  Tab4: 单位详情（三级信息）   │ │ │
│  │  │   + 拾取交互(悬浮/点击)│  │                              │ │ │
│  │  │                      │  │                              │ │ │
│  │  └──────────────────────┘  └──────────────────────────────┘ │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ StatusBar ─────────────────────────────────────────────────┐ │
│  │  模拟时间 | FPS | 相机参数 | 当前主题                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 1.1 布局策略（核心决策）

| 决策 | 理由 |
|------|------|
| **水平 QSplitter**：GLWidget（左，拉伸因子 3）+ SidePanel（右，拉伸因子 1） | 战场为主视图，侧边栏可拖拽调整宽度，最小宽度保护（SidePanel ≥ 250px） |
| **SidePanel 用 QTabWidget** | 对话/档案/简报三个面板切换，节省空间，符合战术推演工具惯例 |
| **HUD 叠加在 GLWidget 上**（非独立 widget） | 单位状态标记/弹道线/命中标记需在世界坐标中定位，用 moderngl 绘制（与 3D 场景共享深度缓冲）；日志滚动条可用 QLabel overlay |
| **对话面板在 SidePanel 中**（非 GLWidget overlay） | 对话需要完整文本渲染和按钮交互，Qt widget 比 moderngl 文本渲染更成熟；且对话时玩家注意力在文本而非战场 |

---

## 二、数据流管线（关键架构）

### 2.0 核心约束：渲染层永不直读 Real 数据

> **法则**（来自 `docs/数据链与战争迷雾设计.md`）：
> 渲染管线只绘制 Twin 数据，永远不直接绘制 Real 数据。
> 玩家永远在「看地图上的标记」，而不是「看物理世界的真相」。

Phase 5 通过 **WorldAdapter** 抽象层落实这一法则：

```
┌─────────────────────────────────────────────────────────────┐
│                 WorldAdapter（渲染数据源）                    │
│                                                             │
│  Phase 5（现在）：透传 Real 数据（上帝视角，无战争迷雾）       │
│  Phase 8（未来）：切换为 Twin 数据 + 置信度 + 状态衰减        │
│                                                             │
│  接口（不变）：                                              │
│    adapter.armor_pos[i]       → 渲染用的位置（Real or Twin）  │
│    adapter.armor_health[i]    → 渲染用的血量                  │
│    adapter.armor_confidence[i]→ 置信度 0~1（Phase 5 恒为 1） │
│    adapter.armor_status[i]    → 渲染状态枚举                  │
│    adapter.damaged_parts(i)   → 该单位损坏部位名列表          │
│    adapter.unit_count()       → 活跃单位数                    │
│                                                             │
│  consumers（只依赖 WorldAdapter，不依赖具体数据源）：             │
│    GLWidget.paintGL()         ← adapter.get_render_data()    │
│    HUD overlay                ← adapter.get_hud_data()       │
│    DetailPanel                ← adapter.get_stats()          │
└─────────────────────────────────────────────────────────────┘
```

**Phase 5 实现**：`WorldAdapter` 包装 `SimulationSnapshot` 的 SoA 数组引用，透传所有字段，`confidence` 恒为 1.0，`status` 恒为 ACTIVE。对外接口与未来 Twin 版完全一致。

### 2.1 每帧数据流

```
                       每帧循环（16ms / 60fps）
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  [模拟层]  SimulationLoop.step(dt)                                │
│     │                                                            │
│     ├── 更新 Real 世界（位置/血量/状态…）                          │
│     ├── EventBus.emit(fire/hit/destroyed/…)                      │
│     │                                                            │
│     ▼                                                            │
│  [孪生层]  WorldAdapter.update(snapshot, events)   ← Phase 5: 透传 │
│     │                                         ← Phase 8: 传感器融合 │
│     │                                                            │
│     ▼                                                            │
│  [叙事层]  NarrativeEngine.tick(snapshot, events)                 │
│     │                                                            │
│     ├── Recorder.on_events()     ← 环形缓冲区写入                 │
│     ├── TriggerSystem.evaluate() ← 触发器匹配                     │
│     ├── DialogueEngine.advance() ← 对话状态机推进                 │
│     │                                                            │
│     ▼                                                            │
│  NarrativeState {                                                 │
│     active_dialogue: DialogueState | None   ← 对话面板消费        │
│     recent_events: list[dict]               ← HUD 滚动日志消费    │
│     flags: set[str]                         ← 条件判断            │
│     paused: bool                            ← 模拟暂停控制        │
│     pending_commands: list[Command]          ← 已自动入队          │
│  }                                                               │
│     │                                                            │
│     ▼                                                            │
│  [UI 层]  MainWindow._on_frame(narrative_state, adapter)         │
│     │                                                            │
│     ├── GLWidget.set_adapter(adapter)          ← 注入渲染数据源   │
│     ├── NarrativeOverlay.update(state.active_dialogue)           │
│     │     └── 显示对话文本 + 选项按钮                              │
│     ├── HUD.update(adapter, state.recent_events)                 │
│     │     └── 更新单位标签/状态标记/命中标记（读 adapter 而非 snapshot）│
│     ├── DetailPanel.update(adapter)                              │
│     │     └── 按当前 hover/selected 刷新三级信息（读 adapter）     │
│     └── StatusBar.update(snapshot.sim_time, fps)                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

                   用户交互（异步，信号驱动）
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  玩家点击对话选项                                                 │
│     │                                                            │
│     ▼                                                            │
│  NarrativeOverlay._on_option_clicked(index)                       │
│     │                                                            │
│     ▼                                                            │
│  NarrativeEngine.on_player_choice(dialogue_id, index)            │
│     │                                                            │
│     ├── DialogueEngine.advance()   ← 推进到下一节点               │
│     ├── CommandFactory.from_node() ← 选项 action → Command       │
│     │                                                            │
│     ▼                                                            │
│  CommandQueue.push(cmd)                                           │
│     │                                                            │
│     ▼                                                            │
│  [模拟层] 下一个 Tick 消费 Command                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

                   单位拾取交互（hover/click，信号驱动）
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  鼠标移动到单位标记上（无按键）                                    │
│     │                                                            │
│     ▼                                                            │
│  Picking.pick(screen_x, screen_y, adapter)                        │
│     │  （O(N) 屏幕距离最近，< 1ms）                                │
│     ▼                                                            │
│  GLWidget.unit_hovered(idx) 信号                                  │
│     │                                                            │
│     ▼                                                            │
│  DetailPanel.set_hover(idx)                                       │
│     └── 显示级别 1：单位名/阵营/状态/损坏部位摘要                   │
│                                                                  │
│  左键点击单位标记                                                 │
│     │                                                            │
│     ▼                                                            │
│  GLWidget.unit_selected(idx) 信号                                 │
│     │                                                            │
│     ▼                                                            │
│  DetailPanel.set_selection(idx)                                   │
│     ├── 显示级别 2：血量/部件逐项/后勤/最近事件                    │
│     └── HUD 高亮描边该单位                                         │
│                                                                  │
│  点击空白 / 按 Esc                                                │
│     │                                                            │
│     ▼                                                            │
│  unit_selected(-1) → DetailPanel 回退级别 0（全局概况）            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 WorldAdapter 接口定义 ✅（已实施）

> 实现注（已实施）：`_build_damaged_parts(box, armor_active, n_armor)` 额外传入 `armor_active`
> （仅存活单位聚合损坏部位，避免父单位已毁时误标）；`armor.side` 通过 `getattr`
> 读取（ArmorPool 暂无该字段，Phase 8 正式接入，演示可动态附加）。
>
> **Phase 5b/5c 扩展**：dataclass 追加 `ammo_reserve/ammo_capacity/fuel_reserve/fuel_capacity`
> （`from_snapshot` 读 `snapshot.logistics`，未装配为 None）+ 便捷查询
> `unit_name(i)` / `side_name(i)`（渲染层命名规则，不碰引擎）/ `speed_at(i)`（占位 0）/
> `ammo_at(i)` / `fuel_at(i)` / `get_stats()`（级别 0 聚合：双方兵力 / 平均弹药燃料 / 模拟时间）

```python
# Airia/render/adapter.py（Phase 5 新增）

from enum import IntEnum
from dataclasses import dataclass
import numpy as np

class UnitRenderStatus(IntEnum):
    """渲染用单位状态（Phase 5 全为 ACTIVE；Phase 8 扩展 UNKNOWN/KIA/MIA）。"""
    ACTIVE = 0
    UNKNOWN = 1   # Phase 8：变成问号
    KIA = 2       # Phase 8：确认击毁
    MIA = 3       # Phase 8：信号丢失

@dataclass
class WorldAdapter:
    """渲染数据源抽象层。

    Phase 5：透传 SimulationSnapshot 的 SoA 数组，所有 confidence=1.0。
    Phase 8：切换为 Twin 数组，含传感器融合 + 时间衰减。

    核心约定：GLWidget / HUD / DetailPanel 只通过本适配器读取数据，
    绝不直接访问 SimulationSnapshot.pools 或 SoA 数组。
    """

    # ── 视图引用（Phase 5：直指 Real 数组；Phase 8：指 Twin 数组）──
    armor_pos: np.ndarray          # (N, 3) float32
    armor_yaw: np.ndarray          # (N,) float32
    armor_health: np.ndarray       # (N,) float32
    armor_active: np.ndarray       # (N,) bool
    armor_confidence: np.ndarray   # (N,) float32  ← Phase 5 恒为 1.0
    armor_status: np.ndarray       # (N,) uint8    ← Phase 5 恒为 ACTIVE
    armor_side: np.ndarray         # (N,) uint8    ← 绝对阵营（Side：0=北约 1=华约 2=中立）
    armor_damaged_parts: np.ndarray  # (N, 5) bool  ← 每单位 5 个部位的损坏掩码（PartType 顺序）

    # ── 部件详情（命中时联动，HUD/详情面板读取）──
    box_parent: np.ndarray         # (M,) int32    ← 碰撞箱所属装甲
    box_part: np.ndarray           # (M,) int32    ← 部位类型（PartType 枚举）
    box_active: np.ndarray         # (M,) bool     ← 部件存活（False=已损坏）

    sim_time: float = 0.0
    tick_count: int = 0

    @classmethod
    def from_snapshot(cls, snapshot: SimulationSnapshot) -> "WorldAdapter":
        """从模拟快照构造（Phase 5 透传模式）。"""
        armor = snapshot.pools["armor"]
        box = snapshot.pools["box"]
        n = armor.max_capacity
        return cls(
            armor_pos=armor.pos,
            armor_yaw=armor.yaw,
            armor_health=armor.health,
            armor_active=armor.active,
            armor_confidence=np.ones(n, dtype=np.float32),  # 上帝视角
            armor_status=np.zeros(n, dtype=np.uint8),        # 全 ACTIVE
            armor_side=armor.side if hasattr(armor, "side") else np.zeros(n, dtype=np.uint8),
            armor_damaged_parts=_build_damaged_parts(box),
            box_parent=box.parent,
            box_part=box.part,
            box_active=box.active,
            sim_time=snapshot.sim_time,
            tick_count=snapshot.tick_count,
        )

    @property
    def active_mask(self) -> np.ndarray:
        """活跃单位掩码（Phase 5：等价于 armor_active）。"""
        return self.armor_active

    def damaged_parts(self, unit_idx: int) -> list[str]:
        """返回某单位已损坏的部位名列表（空列表 = 完好）。

        Phase 5 直接来自 BoxPool 部位毁伤（box.active=False）；
        Phase 8 叠加孪生置信度（低置信时不展示或标记「推测」）。
        """
        ...

    def get_render_data(self) -> dict:
        """返回 GLWidget paintGL 需要的渲染数据。
        Phase 8 扩展：根据 confidence/status 附加不同的渲染模式标记。
        """
        ...


def _build_damaged_parts(box: Any) -> np.ndarray:
    """聚合 BoxPool：每个装甲单位 → 5 部位损坏掩码 (N, 5) bool。

    实现：遍历 box.active 中 parent>=0 的部件，active=False 且 parent=i
    即标记 i 单位的对应 PartType 为损坏。Numba 或 numpy 向量化（每次
    from_snapshot 调用，2048 单位 × 4096 箱 < 1ms）。
    """
    ...
```

### 2.3 主循环集成（MainWindow 持有全部）✅ 已实施

`main.py` + `Airia/ui/main_window.py` 已按本数据流落地（含 recorder 装配、`set_unit_context`、
面板回调 `_recent_events_for` / `_timeline_for` / `_generate_briefing`）。下方伪代码为设计参照：

```python
# main.py（伪代码示意）
class MainWindow(QMainWindow):
    def __init__(self):
        ...
        # 模拟层
        self.simulation = SimulationLoop(state)
        self.simulation.attach("movement", move_hook)   # Phase 3 已实现
        self.simulation.attach("combat", combat_hook)   # Phase 3 已实现
        ...

        # 孪生适配器（Phase 5：透传模式；Phase 8：传感器融合模式）
        self.adapter: WorldAdapter | None = None

        # 叙事层（Phase 4 已实现）
        self.narrative = NarrativeEngine(
            command_queue=self.simulation.commands,
            recorder=CombatEventRecorder(...),
            loop=self.simulation,
        )

        # 帧定时器（替换 GLWidget 内部的 self.timer）
        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._on_frame)
        self._tick_timer.start(16)  # ~60fps

    def _on_frame(self):
        # 1. 模拟层推进（对话暂停时跳过）
        if not self._paused:
            self.simulation.step(1/60)

        # 2. 孪生世界更新（Phase 5：透传；Phase 8：传感器融合+衰减）
        events = self.simulation.events.drain()
        snapshot = self.simulation.state.snapshot()
        self.adapter = WorldAdapter.from_snapshot(snapshot)

        # 3. 叙事层消费
        narrative_state = self.narrative.tick(snapshot, events)
        self._paused = narrative_state.paused

        # 4. UI 层刷新（注入 adapter，永不注入原始 snapshot）
        self._update_ui(narrative_state, self.adapter)

        # 5. 渲染刷新
        self.gl_widget.update()
```

**关键约束**：模拟 `step()` 和叙事 `tick()` 在 UI 线程执行（Qt 主线程），因为：
- Phase 3 的 Numba 并行在 step 内部已天然多线程（`prange`），无需额外线程
- 叙事层计算量 < 1ms（docs 设计保证），不会阻塞帧率
- 避免跨线程信号槽的序列化开销

### 2.4 暂停机制

```
对话激活（需要玩家选择）
  → NarrativeState.paused = True
  → MainWindow._on_frame() 检测 paused → 跳过 simulation.step()
  → GLWidget 继续渲染（相机仍可旋转），但模拟冻结
  → 玩家选择 → NarrativeEngine.on_player_choice()
  → NarrativeState.paused = False
  → 下一帧恢复 simulation.step()
```

---

## 三、着色器主题系统

### 3.1 目录结构

```
assets/shaders/
├── default/                    # 默认（现实军事风格）← 当前 gl_widget.py 内写着色器移入
│   ├── terrain.vert
│   ├── terrain.frag
│   └── theme.toml              # 主题元数据
├── constructivism/             # 构成主义（苏式红黑几何）
│   ├── terrain.vert
│   ├── terrain.frag
│   └── theme.toml
├── anime/                      # 日系动漫（cel shading + 描边）
│   ├── terrain.vert
│   ├── terrain.frag
│   └── theme.toml
└── _template/                  # 模板（供社区/未来主题参考）
    ├── terrain.vert
    ├── terrain.frag
    └── theme.toml
```

### 3.2 theme.toml 元数据

```toml
# assets/shaders/constructivism/theme.toml
[theme]
id = "constructivism"
name = "构成主义"
author = "Arkady in Cradle"
version = "1.0"
description = "苏式构成主义风格：红黑配色、几何色块、硬边阴影"

[uniforms]
# 主题可声明自定义 uniform（在主题切换时注入）
light_dir = [0.45, -0.6, 0.65]
light_color = [1.0, 0.85, 0.75]
sky_color = [0.8, 0.15, 0.1]
ground_color = [0.12, 0.10, 0.08]
ambient_strength = 0.35
```

### 3.3 ThemeManager 设计 ✅（已实施）

```python
# Airia/render/theme.py（新增）

class ThemeManager:
    """着色器主题管理器：加载/切换/卸载。"""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._themes: dict[str, Theme] = {}     # id → Theme
        self._current: Theme | None = None
        self._on_switch: list[Callable] = []    # 切换回调（通知 GLWidget 重绑 VAO）

    def discover(self) -> list[str]:
        """扫描 assets/shaders/ 发现所有可用主题，返回 id 列表。"""

    def load(self, theme_id: str) -> Theme:
        """加载一个主题（编译着色器 + 读取元数据）。"""

    def switch(self, theme_id: str) -> None:
        """热切换到指定主题：编译新着色器 → 重建 program → 通知 GLWidget 重绑 VAO。"""

    @property
    def current(self) -> Theme | None:
        ...

    def on_switch(self, callback: Callable) -> None:
        """注册切换回调。"""

@dataclass
class Theme:
    id: str
    name: str
    program: moderngl.Program         # 已编译的着色器程序
    uniforms: dict[str, Any]          # theme.toml 中声明的默认 uniform 值
    metadata: dict[str, str]
```

### 3.4 热切换流程

```
用户菜单选择「构成主义」
  → ThemeManager.switch("constructivism")
  → 编译 constructivism/terrain.vert + terrain.frag
  → 新 program = ctx.program(...)
  → 卸载旧 program（program.release()）
  → 触发 on_switch 回调
  → GLWidget._on_theme_switched(new_program):
      → self.program = new_program
      → 重新绑定 VAO（vao.program = new_program）
      → 写入所有 uniform 默认值
      → self.update() 触发重绘
```

**性能目标**：主题切换 < 100ms（AGENTS.md Phase 5 验收标准）。编译两个短着色器 + 重绑 VAO 应在 ~50ms 内完成。

### 3.5 三种主题效果设计

| 主题 | 视觉特征 | 着色器关键技术 |
|------|---------|---------------|
| **default**（现实军事） | 当前的半球环境光 + 方向光，自然色 | 现有 shader 外提，不动逻辑 |
| **constructivism**（构成主义） | 红黑主导、色块硬边、几何化阴影 | `step()` 量化连续光照 → 2-3 级硬边；`diff > 0.5 ? 1.0 : 0.2` |
| **anime**（日系动漫） | Cel shading 2-3 级色阶 + Sobel 边缘描边 + 柔和粉彩色 | 量化光照 + 屏幕空间法线边缘检测；天空色偏粉紫，地面偏暖灰 |

> 注：anime 主题的描边需要在 fragment shader 中采样相邻像素的法线/深度，用 `gl_FragCoord` + `textureSize` 实现简单 Sobel。ModernGL 支持 `texture` 采样，当前 FBO 渲染到默认帧缓冲，可用 `gl_FragCoord` 取当前像素坐标。

### 3.6 主题 × 战争迷雾：置信度渲染（Phase 8 预留）

> 本节为 Phase 8 预留设计，Phase 5 不实现，但着色器 uniform 接口需预留 `u_confidence`。

每个主题对孪生置信度的视觉响应不同，但底层数据流完全一致（`WorldAdapter.armor_confidence[i]`）：

| 孪生状态 | default（军事） | constructivism（构成主义） | anime（日系动漫） |
|----------|----------------|--------------------------|-------------------|
| confidence > 0.8 | 完整模型 + 高亮光环 | 实心红色块 + 白色测量标尺 | 清晰描边 + 饱和色 |
| confidence 0.5~0.8 | 半透明模型 + 虚线框 | 虚线边框 + 标注「推定」 | 半透明 + 马赛克粒子 |
| confidence < 0.5 | 模糊抖动偏移 | 虚线框 + 大「?」标记 | 虚影轮廓 + 闪烁 |
| UNKNOWN | 灰色「?」在最后已知位置 | 红色「?」+ 坐标标注 | 半透明虚影 + 问号气泡 |
| KIA (confirmed) | 静态残骸 + 红叉 | 黑色残骸块 + 红叉 | 灰白残骸 + ✕ 标记 |

**着色器接口预留**：每个主题的 vertex shader 接受 `u_confidence` uniform（per-instance），fragment shader 根据置信度调整 alpha / 色相 / 描边强度。Phase 5 中该 uniform 恒为 1.0。

---

## 四、UI 模块详细设计

### 4.1 模块清单（5a/5b/5c 后全部就位 ✅）

| 新文件 | 职责 | 状态 |
|--------|------|------|
| `Airia/render/adapter.py` | WorldAdapter 渲染数据源抽象（Phase 5 透传 Real，Phase 8 切换 Twin；含部位损坏聚合 + 后勤视图 + 命名/聚合查询 + armor_speed 速度视图） | ✅ 5a+5b+5e |
| `Airia/render/picking.py` | 屏幕坐标 → 单位拾取（hover/click 判定，O(N) 屏幕距离最近） | ✅ 5c |
| `Airia/ui/main_window.py` | **重构**：完整的主窗口布局 + 快捷键事件过滤器 + 手动暂停 + 速度估计 + 特效喂入 | ✅ 5a+5b+5e |
| `Airia/ui/narrative_overlay.py` | 对话面板（SidePanel Tab1；历史跨 Tab/结束后保留） | ✅ 5b+5e |
| `Airia/ui/detail_panel.py` | 单位详情面板（SidePanel Tab4，**三级信息详细度**；级别 2 真实速度） | ✅ 5b+5e |
| `Airia/ui/archive_panel.py` | 档案库骨架（SidePanel Tab2） | ✅ 5b 骨架 |
| `Airia/ui/briefing_panel.py` | 战斗简报面板（SidePanel Tab3） | ✅ 5b 骨架 |
| `Airia/render/theme.py` | ThemeManager + Theme | ✅ 5a |
| `Airia/render/hud/overlay.py` | moderngl HUD 渲染（圆点标记 + 选中高亮 + 圆环模式 + 线段 + 文字） | ✅ 5c+5e |
| `Airia/render/hud/text.py` | **新增（5e）**：FontAtlas（PIL 字体图集）+ TextRenderer（实例化字符 quad） | ✅ 5e |
| `Airia/render/effects.py` | **新增（5e）**：EffectLayer 战场特效事件缓冲（弹道线/命中标记，纯逻辑可单测） | ✅ 5e |
| `Airia/render/hud/log_widget.py` | Qt 透明滚动日志（左下角叠加） | ✅ 5c |
| `Airia/ui/controls.py` | **新增（5e）**：ControlMap 键盘 → 动作映射（叙事/命令队列快捷键） | ✅ 5e |

### 4.2 NarrativeOverlay（对话面板）✅ 已实施

消费 `NarrativeState.active_dialogue`（DialogueState：speaker/text/options/history）；选项按钮 → `option_selected(dialogue_id, index)` 信号 → MainWindow → `NarrativeEngine.on_player_choice()`；无对话显示「待命中...」。

```
┌─────────────────────────────────┐
│  📻 前线通讯                    │  ← 标题栏（speaker 名）
│                                 │
│  "第7装甲连已抵达集结点「摇篮」。│  ← QLabel 对话文本（自动换行）
│   前方无人机侦察：敌装甲纵队... │
│                                 │
│  ┌───────────────────────────┐  │
│  │ 展开防御阵型，依托掩体固守 │  │  ← QPushButton 选项（choice 节点）
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ 主动出击，抢占河谷北侧高地 │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │（等待更多侦察情报...）     │  │
│  └───────────────────────────┘  │
│                                 │
│  ──── 历史 ──────────────────   │
│  [系统] 第7装甲连已抵达...      │  ← QScrollArea 对话历史
│  [Arcadia] 展开防御阵型。       │
│  [Kateryna] 明白，自由开火...   │
└─────────────────────────────────┘
```

**信号槽**：
- `option_selected(dialogue_id: str, index: int)` → MainWindow 槽：调用 `narrative.on_player_choice()`
- 无对话时隐藏选项区域，显示「待命中...」占位文本

### 4.3 HUD 叠加层（moderngl 绘制）✅ 已实施（5c 基础 + 5e 全量）

**已落地（5c）**：单位状态圆点（阵营色）+ 选中高亮环（HUDOverlay 屏幕空间 NDC 实例化绘制，HUD pass 期间关闭深度测试）+ LogWidget 滚动日志。
**已落地（5e）**：单位文字标签（FontAtlas 字体图集，选中亮黄/悬浮亮白/默认阵营色相）+ 损坏 ⚠ 摘要文字（橙色）+ 弹道线（fire 事件，射手→目标直线渐隐）+ 命中标记（hit/ricochet/destroyed 事件，扩张圆环 + 十字渐隐）。

在 GLWidget 的 `paintGL()` 中，地形渲染完成后额外绘制：

```
┌──────────────────────────────────────────┐
│                                    FPS:60 │  ← 右上角（正交投影 HUD pass）
│   🟢 T-90 #03                      3.2km │  ← 单位标签（世界坐标 → 屏幕投影）
│   ▲ 敌方 T-72B3 #12（低置信）             │  ← 状态标记（无血条）
│   ⚠ 引擎 履带                             │  ← 损坏部位标记（仅损坏时显示）
│                                          │
│        ╱ 弹道线（开火时短暂显示）          │
│       ╱                                  │
│   💥 HIT!                                │  ← 命中标记（世界坐标，渐隐）
│                                          │
│  ┌─ 日志 ─────────────────────────────┐  │
│  │ [12.3s] T-90 #03 击毁 BTR-82A #07 │  │  ← QLabel overlay（透明背景）
│  │ [11.8s] T-90 #02 命中 T-72B3 #12  │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

**HUD 渲染技术方案**：
- **世界空间元素**（单位状态标记/损坏标记/标签/命中标记）：在 3D 透视投影之后，用相同的 MVP 矩阵把世界坐标投影到屏幕坐标 → 正交投影绘制 2D 元素
- **屏幕空间元素**（FPS/日志）：独立正交投影 pass（`glm.ortho(0, w, h, 0)`）

**单位状态标记图例**（读 `WorldAdapter.armor_status` + `armor_confidence` + `armor_damaged_parts`，**不显示血量**）：

| 状态 | 表现（default） | 数据来源 |
|------|---------------|---------|
| ACTIVE（友军） | 绿色实心圆 + 单位名 | `status=ACTIVE`, `side=ally` |
| ACTIVE（敌军） | 红色实心圆 + 单位名 | `status=ACTIVE`, `side=enemy` |
| 有部件损坏 | 状态圆旁追加 `⚠ 引擎 履带` 等文字 | `armor_damaged_parts[i]` 中 True 的部位名 |
| UNKNOWN | 灰色「?」在最后已知位置 | `status=UNKNOWN` |
| KIA | 静态残骸标记 + 红叉 | `status=KIA`, `confidence>0.9` |
| MIA | 灰色轮廓 + 闪烁边框 | `status=MIA` |

> 设计说明：Phase 5 透传模式下全为 ACTIVE（confidence=1.0），但**部位损坏数据已真实可用**（BoxPool 部件毁伤，Phase 3 已实现）——HUD 的 `⚠` 标记在 Phase 5 就能生效。UNKNOWN/KIA/MIA 由 Phase 8 孪生世界驱动，视觉表现由主题决定（见 §3.6）。

### 4.4 单位详情面板（SidePanel Tab4，三级信息详细度）✅ 已实施

`DetailPanel`（QTextBrowser HTML 渲染）按 hover/selection 切换三级信息；数据源全部经 `WorldAdapter` + 外部注入的最近事件回调（消费 `CombatEventRecorder.query`）。

**核心交互**：右侧窗口的信息详细程度随鼠标交互状态递增——正常（无交互）→ 悬浮 → 点击。

```
┌─ Tab4: 单位详情 ──────────────────────────┐
│                                            │
│  ┌─ 级别 0：无交互（全局概况）─────────────┐ │
│  │  友军装甲：12  敌军装甲：8              │ │
│  │  友军伤亡：0   敌军伤亡：2              │ │
│  │  平均弹药存量：87%  平均燃料存量：94%           │ │
│  │  [模拟]时间：00:02:34                     │ │
│  └────────────────────────────────────────┘ │
│                                            │
│  ┌─ 级别 1：鼠标悬浮某单位 ────────────────┐ │
│  │  📌 T-90 #03（友军）                    │ │
│  │  状态：ACTIVE                           │ │
│  │  损坏：⚠ 引擎 履带                     │ │
│  │  位置：(124.3, 88.1)                    │ │
│  │  ────────────────                       │ │
│  │  （提示：点击查看完整信息）             │ │
│  └────────────────────────────────────────┘ │
│                                            │
│  ┌─ 级别 2：点击选中某单位 ────────────────┐ │
│  │  📌 T-90 #03（友军）[已选中]            │ │
│  │  状态：ACTIVE  │ 阵营：友军             │ │
│  │  血量：62/100  │ 置信度：100%           │ │
│  │  位置：(124.3, 88.1, 4.0) 航向：32°     │ │
│  │  速度：0 m/s                            │ │
│  │  ── 部件状态 ──                         │ │
│  │  ✓ 车体   ✓ 炮塔                       │ │
│  │  ✕ 引擎（失速）  ✕ 履带（无法机动）     │ │
│  │  ✓ 弹药架                               │ │
│  │  ── 后勤 ──                             │ │
│  │  弹药：38/40 发  燃料：82%              │ │
│  │  ── 最近事件 ──                         │ │
│  │  [12.3s] 命中 T-72B3 #12（未击穿）      │ │
│  │  [11.8s] 引擎受损（起火风险）           │ │
│  └────────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

#### 三级信息详细度定义

| 级别 | 触发 | 信息内容 | 数据来源 |
|------|------|---------|---------|
| **0（全局概况）** | 无交互 / 失去选择 | 双方兵力、伤亡、弹药/燃料总存、模拟时间 | `WorldAdapter` 聚合 + `snapshot.logistics` |
| **1（悬浮基础）** | 鼠标悬停在单位标签/标记上（不点击） | 单位名、阵营、整体状态、**损坏部位摘要**、位置 | `armor_status` + `damaged_parts()` 摘要 |
| **2（点击完整）** | 点击单位标签/标记（保持选中） | 级别1 + 血量、置信度、航向/速度、**部件逐项状态**、后勤弹药/燃料、最近事件 | `armor_health` + `box_part/box_active` 逐项 + `recorder` 最近事件 |

#### 拾取交互（GLWidget）

| 事件 | 行为 |
|------|------|
| **鼠标移动（无按键）** | 屏幕坐标 → 由摄像机位置发出射线 → 命中检测最近单位（屏幕空间距离最小）→ 触发 hover 信号 `unit_hovered(idx)` |
| **鼠标悬停单位** | 发射 `unit_hovered(int idx)` → 详情面板显示级别 1 |
| **鼠标悬停空白** | 发射 `unit_hovered(-1)` → 详情面板回退级别 0 |
| **左键点击单位** | 发射 `unit_selected(int idx)` → 详情面板显示级别 2，HUD 该单位高亮描边 |
| **左键点击空白 / Esc** | 发射 `unit_selected(-1)` → 回退级别 0 |

> **拾取实现**：单位是 2D 屏幕标记（不是 3D 模型），拾取用「标记屏幕位置距离最近」即可（O(N) 遍历 2048 单位，每帧 < 1ms）；无需 GPU 拾取缓冲。与相机拖拽（左键按住拖动）通过「按下时记录位置，移动超过阈值才判定为拖拽」区分，单击判定为选中。

**信号槽**：
- `unit_hovered(int) / unit_selected(int)` → MainWindow 槽 → `DetailPanel.set_hover(idx)` / `set_selection(idx)`
- `DetailPanel` 内部按状态聚合不同详细度的数据并渲染

---

## 五、文件改动清单

### 5.1 修改文件

| 文件 | 改动 |
|------|------|
| `main.py` | 集成 SimulationLoop + NarrativeEngine + ThemeManager；组装 MainWindow |
| `Airia/ui/main_window.py` | **重度重构**：QSplitter 布局 + MenuBar + 信号槽连接 |
| `Airia/render/gl_widget.py` | **外提着色器**：VERTEX_SHADER/FRAGMENT_SHADER 常量 → `assets/shaders/default/`；接收外部 program（ThemeManager 注入）；通过 WorldAdapter 读取渲染数据；增加 HUD pass |
| `Airia/render/__init__.py` | 导出 ThemeManager + WorldAdapter |

### 5.2 新增文件

| 文件 | 行数估算 |
|------|---------|
| `Airia/render/adapter.py` | ~150 行 |
| `Airia/render/picking.py` | ~120 行 |
| `Airia/ui/narrative_overlay.py` | ~200 行 |
| `Airia/ui/detail_panel.py` | ~200 行 |
| `Airia/ui/archive_panel.py` | ~80 行（骨架） |
| `Airia/ui/briefing_panel.py` | ~120 行（骨架） |
| `Airia/render/theme.py` | ~180 行 |
| `Airia/render/hud/overlay.py` | ~250 行 |
| `Airia/render/hud/log_widget.py` | ~100 行 |
| `assets/shaders/default/terrain.vert` | ~20 行 |
| `assets/shaders/default/terrain.frag` | ~35 行 |
| `assets/shaders/default/theme.toml` | ~10 行 |
| `assets/shaders/constructivism/terrain.vert` | ~20 行 |
| `assets/shaders/constructivism/terrain.frag` | ~40 行 |
| `assets/shaders/constructivism/theme.toml` | ~12 行 |
| `assets/shaders/anime/terrain.vert` | ~20 行 |
| `assets/shaders/anime/terrain.frag` | ~55 行 |
| `assets/shaders/anime/theme.toml` | ~12 行 |
| `docs/phase5-ui-pipeline-design.md` | 本文档 |

### 5.3 不移改的文件（Phase 5 不碰）

- 所有 `Airia/lulilo/` 子包（Phase 0-3，稳定）
- 所有 `Airia/narrative/` 模块（Phase 4，稳定）
- `Airia/render/camera.py`（稳定）
- `tests/` 目录（Phase 5 结束后单独补 GUI 测试）

---

## 六、实施顺序

### 6.1 Phase 5a：WorldAdapter + 着色器外提 + 主题系统（最小破坏性）✅ 已完成（2026-08-13）

1. **实现 `Airia/render/adapter.py`**：`WorldAdapter.from_snapshot()` 透传模式
2. 从 `gl_widget.py` 提取 VERTEX_SHADER / FRAGMENT_SHADER → `assets/shaders/default/`
3. 实现 `Airia/render/theme.py`（ThemeManager + Theme）
4. GLWidget 改为接收外部 program（构造注入），默认加载 default 主题
5. GLWidget 改为通过 `WorldAdapter` 读取渲染数据（而非直接读 terrain + snapshot）
6. 验证：`uv run python main.py` 渲染效果与改动前完全一致

### 6.2 Phase 5b：MainWindow 重构 + 面板布局 ✅ 已完成（2026-08-13）

1. ✅ 重构 `Airia/ui/main_window.py`：QSplitter + SidePanel(QTabWidget) + MenuBar
2. ✅ 实现 `NarrativeOverlay`（对话面板 Tab1）
3. ✅ 实现 `DetailPanel`（单位详情 Tab4，**三级信息**：全局概况→悬浮→点击）
4. ✅ 实现 `ArchivePanel`（档案库骨架 Tab2）
5. ✅ 实现 `BriefingPanel`（简报面板骨架 Tab3）
6. ✅ 连接信号槽：选项点击 → NarrativeEngine → CommandQueue（+ recorder/logistics 装配）

### 6.3 Phase 5c：HUD 叠加层 + 拾取交互 ✅ 已完成（2026-08-13）

1. ✅ 实现 `Airia/render/hud/overlay.py`（moderngl 圆点标记 + 选中高亮；损坏部位/文字/弹道线/命中标记待补）
2. ✅ 实现 `Airia/render/hud/log_widget.py`（Qt 透明日志）
3. ✅ 实现 `Airia/render/picking.py`（屏幕坐标 → 单位拾取：hover/click 判定）
4. ✅ GLWidget 鼠标事件接入拾取（移动→hover、单击→选中、Esc→取消，拖拽阈值 4px）
5. ✅ DetailPanel 接入 `unit_hovered` / `unit_selected` 信号，三级信息切换
6. ✅ GLWidget 增加 HUD pass（投影→屏幕空间圆点，关闭深度测试）

### 6.4 Phase 5d：主题实现 + 完善

1. 实现 constructivism 主题（着色器 + theme.toml）
2. 实现 anime 主题（cel shading + 描边着色器 + theme.toml）
3. 菜单栏主题切换 → ThemeManager.switch() → 热切换
4. GUI 冒烟测试：切换主题 + 对话流程 + HUD 显示

---

## 七、验收标准

| 标准 | 指标 | 状态 |
|------|------|------|
| 主题切换耗时 | < 100ms（无卡顿） | ✅ 5a 已达成 |
| UI 帧率（地形 + 对话面板） | ≥ 30fps | ✅ 5a/5b 冒烟通过 |
| 战斗帧率（2048 装甲 + 16384 弹丸 + UI） | ≥ 30fps | ⏳ Phase 6 压测 |
| 鼠标/键盘事件 | 全部正确响应（拖拽旋转 + 滚轮缩放 + 选项点击 + 拾取/Esc + 键盘快捷键） | ✅ 5c+5e |
| 对话流程 | YAML 加载 → 触发器激活 → 选项显示 → 选择 → Command 入队，端到端通 | ✅ 5b（smoke 开场对话暂停/恢复） |
| 暂停/恢复 | 对话激活时模拟暂停，选项选择后恢复，相机始终可控 | ✅ |
| WorldAdapter 抽象 | GLWidget / HUD / DetailPanel 全部通过 WorldAdapter 读取数据，不直接访问 SimulationSnapshot.pools | ✅ |
| 拾取交互 | 悬浮单位 → 级别 1；点击 → 级别 2；点击空白/Esc → 回退级别 0 | ✅ 5c |
| 损坏部位显示 | HUD 状态标记在部件损坏时显示 `⚠` 摘要；DetailPanel 级别 2 显示部件逐项 ✓/✕ | ✅ 5e（HUD）+ 5b（DetailPanel） |
| 键盘快捷键（5e） | 空格暂停 / 回车推进 / 1-9 选选项（→ CommandQueue）/ S 跳过 / Ctrl+1/2/3 主题 / F1 帮助 | ✅ 5e（tests/test_phase5_effects.py ControlMap） |
| HUD 战场特效（5e） | 弹道线（fire）+ 命中标记（hit/ricochet/destroyed）渐隐 | ✅ 5e（EffectLayer 单测） |
| 单位真实速度（5e） | DetailPanel 级别 2 速度 = 渲染层位置差分 EMA（不动引擎） | ✅ 5e |
| 对话历史跨 Tab 保留（5e） | 对话结束 / 切换 Tab 后历史保留，新对话激活时替换 | ✅ 5e |
| 向后兼容 | `uv run python -m pytest` 全部通过（不含新增 GUI 测试） | ✅ 435 passed |

---

## 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| 着色器外提后路径解析问题（`__file__` 在 `uv run` 下行为差异） | 使用 `importlib.resources` 或相对于 `pyproject.toml` 的路径解析 |
| 热切换着色器时 moderngl 资源泄漏 | ThemeManager 显式调用 `old_program.release()` + `old_program.glo` 检查 |
| NarrativeOverlay 对话历史滚动性能 | QScrollArea + 限制最大行数（200 行），超出丢弃旧行 |
| HUD 单位状态标记投影在相机极端角度下飞出屏幕 | NDC 裁剪：`if ndc.z > 1.0 or ndc.z < -1.0: skip`（project_world 已按 clip_w>0 + ndc_z∈[-1,1] 过滤） |
| SimulationLoop 在主线程 step() 阻塞 UI | 已验证：Numba `prange` 释放 GIL；step() 内无阻塞 I/O；单帧耗时 < 16ms（60fps 安全） |
| 实例化渲染只画前几个实例（✅ 已修复） | moderngl 实例 buffer format 必须加 `/i` 后缀（divisor=1）；缺省是 per-vertex |
| 主题切换地形变黑（✅ 已修复） | GL context 非 current：`makeCurrent()` 包裹 `ThemeManager.switch()` + 回调里的 GL 操作 |
| 主题切回 InvalidObject（✅ 已修复） | `switch` 释放旧 program 后移出缓存，切回时重新编译 |
| **glm to_list 转置（✅ 已修复 2026-08-13）** | `to_list()` 返回 4 列向量，reshape 是 no-op → 用 `to_bytes()` 转 numpy；必须配非 identity 投影对比测试 |
| **HUD 深度测试剔除（✅ 已修复 2026-08-13）** | HUD pass 期间 `ctx.disable(DEPTH_TEST)`，标记永远最上层 |
