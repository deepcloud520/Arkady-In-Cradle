# UI 改造实施记录：C4I 浮层（P0）

> 状态：**P0 已落地并通过冒烟验证**（2026-11-03）
> 上位设计：`docs/ui-cmo-redesign.md`（V2 视觉定稿，未接入正式代码之前）
> 关联：`docs/phase5-ui-pipeline-design.md`（Phase 5 基线）、`AGENTS.md` Phase 5
> 本文只记录**实施口径**与**已交付范围**，不重复设计论证。

---

## 一、定稿补充（相对 V2 设计的修订）

| # | 议题 | V2 设计 | 实施口径（2026-11-03） |
|---|---|---|---|
| 1 | 单位缩略图 | YAML → PIL 蓝图线框图 | **改用无版权装备图片**；原 UI-2a/2b（蓝图管线 + schema）整体作废 |
| 2 | 系统对话 / 教程 | 未定义 | **进 Message Log**（`kind=system`），不阻塞推演 |
| 3 | 实时通话 | 侧栏 Tab 对话面板 | **左侧弹出窗**：头像（无 → `Sound only`）+ STT 逐字显现 |
| 4 | 战斗简报 | 侧栏 Tab | **中央独立窗口**（非模态）；档案同款窗口 |
| 5 | 窗口形态 | 隐含独占全屏 | **可缩放窗口**（用户拖边缘 / 最大化铺满），非独占全屏 |
| 6 | 面板 alpha | 内容卡 58% | 深色主题 **58% → 72%**（亮地形上小字对比度 AA）；浅色保持 58% |
| 7 | 时间栏 | 白字 + 黑阴影 | **按配色切换**：浅色主题改深字 + 浅阴影（否则浅底白字不可读） |
| 8 | 底部常驻栏 | Message Log 常驻底部 96px | **不做常驻栏**；MESSAGE LOG 做成**可折叠面板**（默认折叠，标题栏显示未读数，`L` 展开），完整历史在其中滚动 |
| 9 | 绘制方式 | 允许 QPainter 自绘浮层 | **除战场（GL）外一律用控件**：面板 = `DockPanel`（QFrame 标题栏 + QToolButton）、时间栏 = QFrame + QLabel、卡片 = QFrame + QSS、头像/缩略图 = QLabel、日志 = `QListWidget`。**UI 层无 `paintEvent`** |
| 10 | 控件风格（CMO） | 半透明无边框浮层 | **明显控件感**：面板 = 渐变标题栏 + 立体边 + `−`/`x` 按钮 + 投影；按钮/列表/滚动条走 QSS 渐变面与凹陷槽（**不再全局 `background: transparent`**，那会把控件剥成平涂矩形）；同时设置 `QPalette` 让原生控件保持质感 |
| 11 | 可拖动 | 固定位置 | 面板标题栏**可拖动**（在容器内自由摆放，带边界钳制）；左侧通话窗标题栏同样可拖；玩家拖过的面板 resize 时**不再被强行归位** |
| 12 | 暂停权限 | 玩家/对话可随意暂停 | **暂停只能由玩家或教程触发**；真正战役（`real_campaign` flag）**禁用一切暂停**（含剧情暂停），上锁时立即解除既有暂停（`narrative/pause.py`） |
| 13 | 时间栏形态 | 分块卡片 + 大字时钟 | **CMO 原味**：一块半透明黑底 + 白字，单行 `Time / Local / Zulu / 23 hr 59 min to go / Camera Alt`，下面一行 `Selected`；不分块、不加边框渐变 |
| 14 | 面板关闭 | `x` 只 hide | 修「关不掉」：`DockPanel.user_closed` 标记，布局重排**不再自动 show**；只有 `L`/菜单显式调用才 `open()` |

---

## 二、响应式尺寸（`Airia/ui/metrics.py`）

所有尺寸的唯一来源；**控件内不得再写字面量像素**。

```
s = clamp(窗口高 / 900, 0.80, 2.00)     # 基准 1440×900（原型验证分辨率）
k = clamp(s × 用户偏好, 0.75, 2.40)      # 偏好 0.85 / 1.00 / 1.15 / 1.30
侧栏 = min(200k, 0.22 × 窗口宽)
```

宽度断点：

| 宽度 | mode | 侧栏 | 时间栏分块 |
|---|---|---|---|
| ≥1600 | `full` | 200k | 地点/日期/ZULU/剩余/高度 |
| 1200–1599 | `compact` | 200k | 地点/日期/ZULU |
| 900–1199 | `tight` | 0.22W | ZULU |
| <900 | `drawer` | 抽屉（默认隐藏，Tab 唤出） | 仅大字时钟 |

高度 < 700 时：时间栏 54→46、底栏 96→72（基线值，再乘 k）。

**DPI 纪律**：只按**逻辑像素**算 k；Qt6 的 High-DPI 缩放已由 `devicePixelRatio`
处理，**绝不再乘 DPR**。字体一律 `setPixelSize()`，不用 pointSize。

---

## 三、已交付文件

| 文件 | 作用 |
|---|---|
| `Airia/render/palette.py` | C4I light/dark 调色板 + 语义色 + 全局 QSS（**下沉到 render**，因 `hud/log_widget` 也要用，避免 render→ui 反向依赖） |
| `Airia/ui/palette.py` | 转发层，保持 `from Airia.ui import palette as pal` 写法 |
| `Airia/ui/metrics.py` | 响应式尺寸（纯函数，可单测） |
| `Airia/ui/dock.py` | 可停靠面板：标题栏（`-` 折叠 / `x` 关闭）+ 拖动 + 边界钳制 |
| `Airia/ui/cockpit.py` | 地图铺满 + 面板定位 + resize 节流（33ms）+ 断点；玩家拖过的面板不再归位 |
| `Airia/ui/timebar.py` | 时间栏（QFrame 卡片 + QLabel 分块 + Selected 三态；宽度贴合内容） |
| `Airia/ui/message_panel.py` | MESSAGE LOG 面板（默认折叠；折叠态显示 `· N` 未读数；`L` 展开） |
| `Airia/ui/comms_popup.py` | 左侧通话弹窗；`TypewriterModel`（字符/秒）+ `AvatarBox(QLabel)` + 标题栏可拖 / 可关闭 |
| `Airia/ui/unit_thumb.py` | 装备缩略图（QLabel）：`MANIFEST.toml` → 同名文件 → `NO IMAGE` 文本占位；DPR 感知 |
| `Airia/ui/windows.py` | `CenterWindow` 基类 + `BriefingWindow` / `ArchiveWindow`（+ `LogWindow` 备用） |
| `Airia/ui/_ticker.py` | **已废弃**（被 `message_panel` 取代，保留文件不再引用） |
| `Airia/ui/fonts.py` | 界面字体加载（雅黑/等宽回退链；offscreen 下避免豆腐块） |
| `Airia/render/hud/log_widget.py` | 重写：按 kind 着色 + `append_system` + MESSAGE LOG 标题（现由 `LogWindow` 承载） |
| `Airia/ui/main_window.py` | 接入 Cockpit；4 Tab 退役；菜单改为 视图/叙事/面板/帮助 |
| `Airia/render/gl_widget.py` | 相机 `far` 按地图对角线放大；HUD 图集/半径/字号按屏幕高度缩放 |
| `Airia/render/camera.py` | 新增 `far_for_extent()` |

### 暂停权限（`Airia/narrative/pause.py`）

```
玩家（空格 / 按钮） ─┐
教程步骤          ─┼─→ PauseController ─→ loop.sim_paused
剧情对话(pause_sim)┘        ↑
                        locked = real_campaign flag
```

- **允许来源**：玩家、教程、（非战役时的）剧情对话
- **战役锁**：`real_campaign` flag 置位后 `locked=True` → 所有来源被拒，既有暂停立即清空
- 引擎侧 `sim_paused` 是**唯一**暂停真值：UI 不再各自维护 `manual/narrative` 的与运算
- 菜单「叙事 → 战役模式（禁用暂停）」供演示/测试切换

### 相机远裁剪面修复

原先 `far=12000` 固定，15km 地图远侧边缘（距相机约 20000m）被裁掉。
现由 `far_for_extent(w, h) = max(12000, 3 × 对角线)`（15km → 约 63.6km）。

---

## 四、数据来源（仍守「UI 只读 WorldAdapter」）

- 详情 / 时间栏 / 单位名与编队：`WorldAdapter`（`unit_name` / `unit_formation_status` / `armor_*`）
- 通话文本与选项：`NarrativeState.active_dialogue`（`option_selected` 信号签名不变）
- 系统消息：`LogWidget.append_system`
- 日志事件：`simulation.events.drain()` → `{kind, data}`

---

## 五、运行与验收

```powershell
# 真实主窗口（含 GL + 模拟层）截图 → tests/output/ui_c4i_window.png
.venv\Scripts\python.exe tests\smoke_c4i_window.py

# 纯布局/配色冒烟（无 GL，快）→ tests/output/ui_c4i_shell_{dark,light}.png
.venv\Scripts\python.exe tests\smoke_c4i_shell.py

# 单元测试（尺寸/三态/着色/STT/缩略图/浮层几何/远裁剪面）
.venv\Scripts\python.exe -m pytest tests/test_ui_c4i_shell.py -q

# 正式运行
.venv\Scripts\python.exe main.py
```

> 本机 `uv run` 会因 uv 缓存目录被沙箱拒绝而失败，直接用 `.venv\Scripts\python.exe`。

### 界面元素一览（沉浸感口径）

| 元素 | 位置 | 显隐 / 交互 |
|---|---|---|
| 时间栏 | 左上（卡片宽度贴合文字） | 常驻 |
| UNIT STATUS 侧栏 | 右侧 | 常驻；标题栏可拖动 / 可折叠；`Tab` 隐藏；窄窗口→抽屉 |
| 实时通话 | 左侧，时间栏下方 | 仅通话时出现；标题栏可拖动，`x` 关闭本次通话 |
| MESSAGE LOG | 左下 | **面板**：默认折叠（标题栏显示未读数），点标题栏或 `L` 展开，`x` 关闭，可拖动 |
| 战斗档案 / 战斗简报 | `面板` 菜单 | 独立窗口（原生标题栏，可拖动） |
| 调试状态栏 | 窗口底部 | **默认隐藏**，`F3` 唤出 |

---

## 六、尚未交付（后续批次）

| 批次 | 内容 | 依赖 |
|---|---|---|
| P1 | ~~通话弹窗接线~~（已随 P0 完成） | — |
| P2 | ~~简报 / 档案窗口~~（已随 P0 完成） | — |
| P3 | 缩略图**真图**（`assets/units/thumbs/` + MANIFEST 授权登记） | **缺图片素材** |
| P4 | 折叠分组侧栏 + MSR 口径 + 悬浮只换头部 + 告警动态展开 + 多选 `SelectionStore` | — |
| P5 | 军标朝向楔（HUD `mode 2`）+ 像素尺寸 LOD + 密度保护 + 线型绑可信度 | — |
| P6 | 比例尺 + 光标坐标 + 射程圈（`circle_kind: weapon/sensor`） | 引擎侧射程视图 |
| P7 | 日志点击定位（`camera.focus_on`）+ 告警导航 | — |
| P8 | 框选 `pick_in_rect` + 多选聚合视图 | P4 |

### 已知缺口

1. `assets/units/thumbs/` **当前为空**：界面显示 `NO IMAGE` 占位。放图后按
   `MANIFEST.toml` 登记（`file` / `license` / `source` / `credit`）即生效。
2. 侧栏仍是旧 `DetailPanel`（HTML + 三级信息），**尚未**改为折叠分组 / MSR 口径。
3. 地图单位仍是圆点（未见朝向楔 / 军标框）。
4. `UI_SCALES` 滑块只在菜单里（视图 → UI 缩放），未做设置页。
5. 窗口几何不持久化（开发期有意如此）。
6. `tests/test_narrative.py::test_load_yaml` 在受限环境下因 pytest `tmp_path`
   不可写而 ERROR——环境问题，与本轮改动无关。
