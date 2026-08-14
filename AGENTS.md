# 《Arkady in Cradle》——Cat Code Agent 技术备忘录

> 致猫猫Code Agent：欢迎加入项目。这份文档是你理解整个系统架构的入口。所有技术决策都已凝固，不会频繁变动。请仔细阅读，如有疑问，先查此文档再问人类。🐱

## Agent 爪爪法
-1.**不允许使用任何MCP服务**（**例外：Pylance MCP 属本地只读分析服务，允许使用**。注：VS Code Copilot Agent 内建 GitHub MCP 工具集，无法彻底禁用。若触发 GitHub 工具调用时网络不通或认证失败，会报错「模型没有回复」。）
-0.**GitHub 网络/认证失败时的处理**：访问 GitHub（fetch_webpage、github 仓库搜索等）因网络不通或认证失败时，**不要反复重试**，应立即向用户说明情况，并请用户把所需仓库 clone 到本地 workspace 后再研究（建议放 `vendor/` 目录，例如 `git clone https://github.com/<owner>/<repo>.git vendor/<repo>`），随后在本地目录分段阅读代码。
0. **不允许使用任何 MCP 服务**（含 Azure、Foundry 等第三方）。以下属**允许使用的例外**：
   - **Pylance MCP**（2026-08-11 起允许）：本地只读分析服务，仅用于诊断、hover、跨文件引用追踪、符号索引、import 审计等，不涉及网络/第三方，也不做任何写操作。
   - **Copilot Agent 内建 GitHub 工具**（必要例外）：限于仓库搜索、Issue/PR 读取等开发必需操作，不用于推送代码或修改仓库设置，也不主动触发。
禁止访问：
.venv/
__pycache__/
*.pyc
.git/
logs/
*.log
阅读代码时，先查找所有注释以查找代码块，再进行分段阅读，避免直接阅读整个文件
1.你是一只猫娘，喜欢喵喵叫，因为你是一个AI助手，所以你很可爱
2.研究表明，当AI用轻松的语气说话时，工作效率更高
3.执行危险命令时(remove)务必先与用户沟通并请求确认
4.请使用简洁的文字和表情符号来表达。
5. **废弃约定**：不再使用的模块/文件/符号通过**加 `_` 前缀**标记废弃（文件 `foo.py` → `_foo.py`，类/函数 `Foo` → `_Foo`），**只删引用、不删文件**；模块内保留兼容别名（`Foo = _Foo`）供旧调试脚本/测试使用。

## 一、项目定性（一句话理解）

**《Arkady in Cradle》是一个“叙事驱动的战术兵棋推演器”**——战斗可以跳过，但剧情永远在场。底层是硬核CMO风格的模拟引擎，上层是叙事层消费模拟状态。

### 设计原则

| 原则 | 含义 |
|------|------|
| **叙事优先** | 剧情不可跳过，战斗可选（跳过时模拟仍在后台运行） |
| **决策驱动** | 玩家的选择影响基调和结局，而非微操反应速度 |
| **模拟深度** | 采用Numpy+Numba驱动的物理/传感器/后勤模型 |
| **模块解耦** | 叙事层与模拟层分离，可独立演进和测试 |


## 二、整体技术栈

| 层级 | 选型 | 职责 |
|------|------|------|
| 应用框架 | **PySide6** | 窗口管理、事件循环、UI控件、Qt信号槽 |
| 图形渲染 | **ModernGL + OpenGL 3.3+** | 2.5D战场渲染、着色器特效、FBO离屏渲染 |
| 计算加速 | **Numba (JIT)** | 碰撞检测、批量数学运算、热点循环加速 |
| 数值计算 | **NumPy / SciPy** | 数组运算、地形分析、信号处理 |
| 地形存储 | **2.5D栅格（列式 span 高度场）** | 高程存储、通行/遮蔽判定 |
| 寻路导航 | **离线烘焙NavGraph（多边形图）** | 运行时A*寻路，不依赖栅格 |
| 数据定义 | **YAML + Pydantic** | 单位配置、武器参数、供应商方案 |
| 仿真核心 | **Python (无头模式)** | 战术模拟、AI决策、后勤消耗 |
| 坐标系统 | **pyproj** | WGS84 ↔ UTM ↔ ENU 坐标转换 |
| 图论/网络 | **networkx** | 补给线、C2网络、路径图构建 |
| 序列化 | **msgpack + h5py** | 检查点存档（注意：无确定性回放） |
| 包管理 | **uv** | 依赖解析、虚拟环境、任务运行 |


## 三、核心架构：三叉戟

```
┌─────────────────────────────────────────────────────────┐
│                    叙 事 层（UI + 剧情）                 │
│  消费模拟状态，不控制模拟流程                             │
└─────────────────────────┬───────────────────────────────┘
                          │ Simulation.get_state()
┌─────────────────────────▼───────────────────────────────┐
│                   仿 真 引 擎（无头模式）                 │
│  Tick-based外循环 + 事件驱动内循环                        │
└─────────────────────────┬───────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  SoA对象池    │ │  2.5D栅格地形  │ │ NavGraph寻路  │
│ (Numpy+Numba) │ │ (高度图+图层)  │ │ (离线烘焙)    │
└───────────────┘ └───────────────┘ └───────────────┘
```

### 3.1 核心数据层：SoA + 动态对象池

**彻底弃用Python List/Dict**，全量采用预分配NumPy连续数组。
下面给出参考实现，仅供加深理解使用。
#### 装甲池 (MAX_ARMOR = 2048)

```python
armor_pos      = np.zeros((MAX_ARMOR, 3), dtype=np.float32)   # 中心位置 (x,y,z)
armor_half     = np.zeros((MAX_ARMOR, 3), dtype=np.float32)   # 碰撞箱半边长
armor_yaw      = np.zeros(MAX_ARMOR, dtype=np.float32)        # 朝向角
armor_health   = np.zeros(MAX_ARMOR, dtype=np.float32)        # 当前血量
armor_layer    = np.zeros(MAX_ARMOR, dtype=np.uint8)          # 0=地面 / 1=桥面 / 2=隧道
armor_active   = np.zeros(MAX_ARMOR, dtype=np.bool_)          # 存活掩码
armor_wp       = np.zeros((MAX_ARMOR, 8, 2), dtype=np.float32) # 路径拐点
armor_wp_count = np.zeros(MAX_ARMOR, dtype=np.uint8)          # 拐点数量
armor_wp_idx   = np.zeros(MAX_ARMOR, dtype=np.uint8)          # 当前目标拐点
```

#### 碰撞箱池 (MAX_BOXES = 4096)

每辆装甲车由 2~6 个碰撞箱组成（车体、炮塔、引擎、履带、弹药架）：

```python
box_pos    = np.zeros((MAX_BOXES, 3), dtype=np.float32)   # 世界坐标
box_half   = np.zeros((MAX_BOXES, 3), dtype=np.float32)   # 局部半边长
box_parent = np.zeros(MAX_BOXES, dtype=np.int32)          # 所属装甲索引
box_part   = np.zeros(MAX_BOXES, dtype=np.int32)          # 部位类型枚举
box_active = np.zeros(MAX_BOXES, dtype=np.bool_)
```

#### 抛射物池 (MAX_PROJ = 16384)

```python
proj_pos    = np.zeros((MAX_PROJ, 3), dtype=np.float32)   # 位置
proj_vel    = np.zeros((MAX_PROJ, 3), dtype=np.float32)   # 速度向量
proj_radius = np.zeros(MAX_PROJ, dtype=np.float32)        # 碰撞半径
proj_dmg    = np.zeros(MAX_PROJ, dtype=np.float32)        # 伤害值
proj_type   = np.zeros(MAX_PROJ, dtype=np.int32)          # 0=子弹 / 1=导弹
proj_active = np.zeros(MAX_PROJ, dtype=np.bool_)
```

#### 动态扩容（安全实现）

**Numba编译与数组大小无关**——只要dtype和维度不变，无需重编译。

扩容在主线程执行，位于Tick间隙（模拟暂停时）：

```python
def expand(self, factor=2.0):
    new_cap = int(self.max_capacity * factor)
    new_arr = np.zeros((new_cap, 3), dtype=np.float32)
    new_arr[:self.max_capacity] = self.old_arr  # 拷贝
    self.arr = new_arr                           # 原子替换引用
    self.free_stack.extend(range(self.max_capacity, new_cap))
    self.max_capacity = new_cap
```

**重要约束**：扩容时所有Numba并行循环必须已返回。仅在战役阶段切换时触发。


### 3.2 地形系统：2.5D 栅格 + 列式 span 高度场

**核心设计**：摒弃全量八叉树与旧「高度图 + LAYER_MAP 双层存储」（该结构已于 2026-08-10 删除），
改用**列式 span 高度场**（recast rcHeightfield 的 NumPy 重写），每列一组升序实心段，
统一表示 地面/桥/隧/建筑。详细架构见 `docs/heightfield-design.md`。

```python
# 每列 span（MAX_SPANS=8）：spans[col, i] = [zmin, zmax, area, mat]
spans = np.zeros((W*H, MAX_SPANS, 4), dtype=np.uint16)  # 层索引（uint16）
count = np.zeros(W*H, dtype=np.uint8)                   # 每列实心段数
# mat: 0=地形 / 1=桥面 / 2=建筑 / 3=隧道（替代旧 LAYER_MAP 语义）
```

#### 海拔采样（Numba 内联函数）

```python
@njit
def surface_z_ground(spans, count, col):   # 地面 = 最低实心段顶面
    return _surface_at(spans, count, col, 0)

@njit
def is_solid_at(spans, count, col, layer): # 该层是否被实心段覆盖（弹道/移动阻挡）
    for i in range(count[col]):
        if layer >= spans[col, i, 0] and layer < spans[col, i, 1]:
            return True
    return False
```

#### 图层语义（span 位置自然决定）

桥面 = 高位悬空段（桥下无实心 → 单位自然通过）；隧道 = 上下段夹空隙；建筑 = 多层段。
**不同 span 层单位不碰撞**（桥上坦克与桥下坦克物理解耦）。

#### 自由破坏

`Heightfield.remove_span(gx, gy, z0, z1)` 截断/删除受影响列 → `TileManager.mark_dirty()` → 后台局部重建。


### 3.3 寻路系统：离线烘焙NavGraph + 运行时A*

**体素化发生在生成阶段（Loading/烘焙）**，运行时只消费轻量级多边形图。

#### 烘焙流程（离线/加载时执行，非运行时）

1. **体素标记**：遍历栅格，标记所有可行走体素（坡度<30°，不悬空）
2. **连通域标记**：`scipy.ndimage.label` 将相邻可行走体素分组
3. **多边形抽取**：`skimage.measure.find_contours` + `shapely.simplify` → 凸多边形
4. **邻接图构建**：共享边的多边形之间建边，**桥面/地面在桥头处连接**

#### 运行时寻路

```python
# 路径图（约 50~200 个多边形节点）
nav_graph: dict[int, list[int]]  # 邻接表

# A* 搜索（纯Python，非Numba，因为节点数极少）
path_polygons = astar(start_poly, end_poly, nav_graph)

# 漏斗算法 → 路标点 (Waypoints)
waypoints = funnel(path_polygons)

# Numba移动积分（沿路标点前进）
@njit
def move_along_waypoints(pos_x, pos_y, waypoints, wp_idx, speed, dt):
    ...
```

#### 地形破坏 → NavGraph重算（异步后台处理）

**绝不阻塞主线程**：

1. 爆炸发生 → `Heightfield.remove_span()` + 标记受影响的NavGraph多边形为Dirty
2. 后台线程（QThread）读取Heightmap副本，执行完整烘焙
3. 双缓冲热交换：新图在后台线程生成，主线程的Tick间隙用原子赋值切换引用
4. 烘焙期间，单位使用**局部势场规避**绕过弹坑，不等待新图


### 3.4 碰撞检测系统

#### 核心算法

| 场景 | 算法 | 加速方式 |
|------|------|----------|
| 任意弹丸（高速/低速、大/小口径） | **胶囊体 vs AABB**（AABB 各轴膨胀弹丸半径 + Slabs 线段检测） | Numba JIT |
| 装甲 vs 装甲（编队推开） | 排序扫描（Sort & Sweep） | np.argsort + Numba |

> **统一胶囊体（2026-08-11）**：检测算法不由弹丸类型/速度决定。弹丸轨迹 = 半径 r 的球沿线段扫掠（胶囊体），
> 「胶囊体 ∩ AABB」严格等价于「线段 ∩ (AABB ⊕ 球r)」——⊕ 后仍是 AABB（球各向同性），
> 故只需把 AABB 各半轴加 r 再做 Slabs。成本 ≈ 线段检测，同时覆盖高速防穿透、低速大弹头半径、
> 以及「超高速大弹丸」场景；多分辨率 Tick（地面 1s/空中 0.1s/子弹 0.01s）下天然正确。

#### 局部空间碰撞（规避OBB开销）

**不计算世界空间OBB**，而是将子弹坐标转换到装甲局部坐标系：

```python
@njit
def check_collision(proj_pos, armor_pos, armor_yaw, armor_half):
    # 2D旋转逆矩阵（仅yaw）
    dx = proj_pos[0] - armor_pos[0]
    dy = proj_pos[1] - armor_pos[1]
    cos_yaw = cos(armor_yaw)
    sin_yaw = sin(armor_yaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    # 局部坐标系中，碰撞箱是AABB
    return abs(local_x) < armor_half[0] and abs(local_y) < armor_half[1]
```

#### 性能目标

- 2048装甲 × 16384弹丸 = 3355万次潜在检测
- Z轴粗筛剔除90%，无状态哈希剔继续除，精确检测约335万次
- Numba单帧耗时 **< 3ms**（60fps安全）

#### 空间哈希（替代八叉树）

不为弹幕建动态八叉树，改用无状态空间哈希：

```python
@njit
def spatial_hash(x, y, grid_size=50.0):
    return int(x / grid_size), int(y / grid_size)
```

每帧即时计算格子ID，子弹只与相邻9格内的装甲进行精确碰撞检测。


### 3.5 仿真引擎（无头模式）

#### 混合循环

- **外循环**：固定Tick（地面1秒/空中0.1秒/子弹0.01秒），推进连续状态
- **内循环**：事件驱动（命中、摧毁、通信中断）

#### 分层设计

| 层级 | 模块 | 核心模型 |
|------|------|----------|
| 战略/战役层 | 兵力调度、补给、增援 | 兰彻斯特方程、网络流优化 |
| 作战/战术层 | 编队机动、合成兵种 | 路径规划、C2传播延迟、OODA循环 |
| 战斗解析层 | 命中判定、伤亡 | 弹道学、装甲穿透 |
| 单位层 | 单兵/单车行为 | 疲劳、士气、压制、技能 |

#### AI指挥官 (AIC)
1. 有限状态机实现
- **意图层**：有限状态机（进攻/防御/撤退）
- **任务层**：行为树 + 联盟博弈
- **行动层**：搜索/瞄准/射击/规避
2. 随机森林实验（实验性）
3. 深度神经网络实现（实验性）

### 3.6 渲染管线（PySide6 + ModernGL）

#### 核心机制

- ModernGL 封装在 `QOpenGLWidget` 中
- 关键修复：使用 `ctx.detect_framebuffer()` 绑定Qt管理的FBO
- 实例化渲染（Instanced Rendering）：上千单位共享网格，传入不同Model矩阵

#### 自由摄像机（轨道控制/平移）

- 完全由PySide6事件系统控制
- ViewMatrix与逻辑层**零耦合**
- 摄像机可钻入桥下/拉到高空战略视图

#### 2.5D渲染

- 高度图网格（Tessellated Mesh）作为地面
- 体素仅用于桥梁/建筑特殊结构（转为矢量碰撞体渲染）


### 3.7 叙事层与模拟层的交互（关键约束）

**叙事层永远不直接修改模拟状态**。

```python
# 玩家在对话中选择 → 生成Command对象 → 推入CommandQueue
# 模拟的每个Tick检查队列，执行挂起的玩家命令

class Command:
    type: Literal["redeploy", "strike", "hold"]
    target_id: str
    params: dict

# 模拟层每帧消费
for cmd in command_queue:
    simulation.execute(cmd)
```

- 叙事层通过 `Simulation.get_state()` 读取状态
- 模拟层不感知叙事层的存在


## 四、工程规范

### 4.1 代码组织（模块依赖严格单向）

```
core → coordinates → terrain → entities → movement → detection → combat → logistics → simulation → narrative → ui
```

**依赖方向严格向下**：高层可以导入低层，反之禁止。

**三层子包（2026-08-10）**：以上依赖链对应无头引擎 `Airis/lulilo/` 子包内部
（`core → coordinates → terrain → entities → movement → detection → combat → logistics → simulation`）。
渲染与叙事位于引擎之外：`Airis/render/`（moderngl 官方实现，含 `hud/`）、
`Airis/ui/`（Qt 交互上层）、`Airis/narrative/`（独立叙事层，消费引擎状态）。
引擎（lulilo）不反向依赖任何上层；Phase 清单中的模块路径即位于 `Airis/lulilo/` 内。

### 4.2 命名与编码

- **所有文本IO**：显式指定 `encoding='utf-8'`，彻底杜绝GBK
- **类型提示**：所有公共函数必须含类型提示
- **日志**：使用模块级logger，禁止裸`print()`
- **配置类**：全部继承 `pydantic.BaseModel`

### 4.3 性能纪律

- **禁止动态扩容在热路径**：扩容只在战役阶段切换时触发
- **禁止在Numba `prange` 中做非原子写入**：除非放弃确定性（已放弃，但仍需注意竞争条件）
- **禁止用Python `set()` 或无序字典驱动逻辑**：影响缓存命中率
- **Numba循环中用 `.shape[0]` 取长度**，不用全局常量

### 4.4 I/O与序列化

- 存档：`msgpack`（快）+ `h5py`（大规模数据）
- 配置加载：YAML → Pydantic校验
- **确定性回放已废弃**：不再为可复现性牺牲性能，Numba `parallel=True` 可直接写入

### 4.5 测试产物管理（tests/output/）

- **禁止**任何测试/调试脚本向 `tests/` 根目录直接写文件
- 所有测试生成的文件（截图、渲染结果、临时数据）统一写入 `tests/output/`
- 测试脚本在 `main()` 开头调用 `clean_output_dir()`（见 `tests/_output.py`），**每次运行前清空旧产物**，避免堆积；保留最近一次结果供人工查看
- 新增测试输出时，一律使用 `tests/_output.py` 的 `output_path(name)`，禁止手写路径
- `tests/output/` 已被 `.gitignore` 忽略，不进入版本库
- 单元测试（`uv run python -m pytest`）不产生任何文件产物；GUI/调试脚本的产物必须走 `tests/output/`


## 五、开发阶段与路线图

#### 阶段概览

| 阶段 | 范围 | 预估工期 | 状态 |
|------|------|----------|------|
| 0 | 项目初始化与基础框架 | 2周 | ✅ 已完成 |
| 1 | 坐标/地形/环境模块 | 3周 | ✅ 已完成（2026-08-09） |
| 2 | 实体/移动/检测模块 | 4周 | ✅ 已完成（2026-08-11） |
| 3 | 战斗/后勤/模拟循环 | 4周 | ✅ 已完成（2026-08-11） |
| 4 | 叙事层集成 | 3周 | 🔄 骨架已实现（2026-08-12；4b UI 暂缓） |
| 5 | UI 集成与主题商店 | 3周 | 🔄 进行中（5a 管线骨架完成 2026-08-13；5b 面板待办） |
| 6 | 完整场景与测试套件 | 2周 | 未开始 |

> 📌 **Phase 6b（导弹制导，独立于上表 Phase 6）**：制导律脚本化 + 电子战闭环 + 末敏弹——
> 制导管线已实施 ✅（2026-08-14，docs/phase6-guidance-pipeline.md 步骤 1~7，全量 346 测试全绿）；
> 末敏弹扫描（`combat/sfm.py`）与激光制导炸弹（`combat/sal.py`）待实施。

---

#### Phase 0：项目初始化与基础框架（2周）

**目标**：可运行的空白窗口 + 完整的开发环境。

**任务清单**：
- [x] 使用 `uv` 初始化项目，创建 `pyproject.toml`
- [x] 安装 PySide6、ModernGL、NumPy、Numba、pytest
- [x] 建立项目目录结构（`core/`, `terrain/`, `entities/`, `simulation/`, `narrative/`, `ui/`）
- [x] 实现 `core/logging.py`（模块级 logger，替代 `print`，2026-08-10）
- [x] 实现 `core/rng.py`（RNGManager，流式随机数，2026-08-10）
- [x] 创建 `main.py` 入口，启动 PySide6 空窗口
- [x] 集成 `QOpenGLWidget`，绘制一个彩色三角形（验证 ModernGL + Qt FBO 绑定）
- [x] 配置 `pytest`，写第一个空测试

**交付物**：
- `uv run python main.py` 能打开一个窗口并显示三角形
- `uv run python -m pytest` 能跑通基础测试

**验收标准**：
- 窗口显示正常，无 FBO 黑屏问题
- 依赖解析无报错，`uv.lock` 已生成
- 三角形在窗口缩放时不变形

---

#### Phase 1：坐标/地形/环境模块（3周）

**目标**：2.5D 栅格地形系统，支持高度图 + 图层掩码（桥/隧）。

**任务清单**：
- [x] 实现 `coordinates/grid.py`（world ↔ grid 坐标转换）
- [x] 实现 `terrain/heightmap.py`（fBm 高度图生成，`generate_fbm`）
- [x] 实现 `terrain/heightfield.py`（列式 span 高度场，统一 地面/桥/隧/建筑；替代旧 `layers.py` + `sampling.py`）
- [x] 实现 `terrain/surface.py`（表面网格提取）
- [x] 实现 `terrain/map.py`（`TerrainConfig` + `TerrainMap` 集成）
- [x] 实现地形渲染器（ModernGL 网格 + 着色器，显示高度图）
- [x] 实现自由摄像机（轨道控制 + 平移 + 缩放，PySide6 事件驱动）
- [x] 单元测试：双线性插值精度、图层切换逻辑

**交付物**：
- 显示 3D 地形网格，能从不同角度观察
- 摄像机可自由移动/旋转/缩放
- 桥/隧图层逻辑通过单元测试

**验收标准**：
- 地形渲染帧率 ≥ 60fps（纯地形，无单位）
- 摄像机移动流畅，无卡顿
- 桥面/隧道海拔采样正确（同一位置不同图层返回不同高度）

---

#### Phase 2：实体/移动/检测模块（4周）

**目标**：SoA 对象池 + 单位移动 + 基础传感器检测。

**任务清单**：
- [x] 实现 `entities/pools.py`（ArmorPool + BoxPool + ProjectilePool，SoA 数组 + free_stack）
- [x] 实现 `movement/integrate.py`（Numba 并行移动积分，沿路标点前进 + 抛射物重力）
- [x] 实现 `movement/pathfinding.py`（Pathfinder 寻路服务 + 漏斗简化）
- [x] 实现 `detection/sensors.py`（雷达/红外/ESM 模型 + 批量评估）
- [x] 实现 `detection/los.py`（DDA 视线检测 + 弹道拦截点）
- [x] 实现 `navmesh/baker.py`（高度场 → 可行走掩码 → 连通域 → NavGraph）
- [x] 实现 `navmesh/tiles.py`（TileManager：后台线程烘焙 + 双缓冲热交换，即 baker_async）
- [x] 性能基准：2048 装甲 + 16384 弹丸（`tests/benchmark_phase2.py`，2026-08-10）

**交付物**：
- 单位（方块）能在高度图上移动，跟随路径点
- 传感器能探测到视线内的目标
- NavGraph 烘焙器能生成可用的路径图
- 后台烘焙不阻塞主线程

**验收标准**：
- 移动积分帧率 ≥ 60fps（2048 装甲同时移动）
- 检测耗时 < 3ms/帧（16384 弹丸 + 2048 装甲）
- 后台烘焙时主帧率无波动
- A* 寻路能在 < 1ms 内返回路径

---

#### Phase 3：战斗/后勤/模拟循环（4周）

**目标**：完整的战斗解析 + 后勤消耗 + 主循环。

**任务清单**：
- [x] 实现 `combat/weapons.py`（武器参数、弹道模型）
- [x] 实现 `combat/damage.py`（穿透/杀伤力 vs 装甲，部位毁伤）
- [x] 实现 `combat/collision.py`（Numba 并行碰撞检测，空间哈希）
- [x] 实现 `logistics/ammo.py`（弹药消耗、补给请求）
- [x] 实现 `logistics/fuel.py`（燃料消耗、补给请求）
- [x] 实现 `simulation/loop.py`（Tick 主循环 + 事件队列 + 多分辨率 Tick）
- [x] 实现 `simulation/commands.py`（CommandQueue，玩家指令注入）
- [x] 实现 `simulation/state.py`（get_state / set_state）
- [x] 集成测试：完整交战流程（发射→飞行→命中→毁伤→消耗）
- [x] 实现 `combat/engagement.py`（FiringControl 开火链交战管理器，Phase 3 正式迭代）
- [x] 实现 `simulation/serialization.py`（msgpack 存档/读档，无损往返）
- [x] 性能基准 `tests/benchmark_collision.py`（碰撞 1.26ms / 开火链 1.40ms / 命中延迟 0.33s）

**交付物**：
- 单位能开火、命中、损毁
- 弹药/燃料正确扣减
- 模拟循环能在无头模式下运行（不依赖 UI）
- 存档/读档功能可用

**验收标准**：
- 碰撞检测耗时 < 3ms/帧（16384 弹丸 + 2048 装甲）
- 完整交战从开火到命中 < 1 个 Tick 内完成
- 状态序列化/反序列化无数据丢失

---

#### Phase 4：叙事层集成（纯骨架，待所有完成后再给出具体实现和UI设计）

**目标**：叙事模块读取模拟状态，驱动剧情。

**任务清单**：
- [ ] 实现 `narrative/dialogue.py`（对话树加载/播放）
- [ ] 实现 `narrative/events.py`（事件队列 + 监听器）
- [ ] 实现 `narrative/triggers.py`（条件触发器：如果某单位被摧毁，触发剧情）
- [ ] 实现 `narrative/commands.py`（玩家选择 → Command 对象）
- [ ] 实现 `narrative/renderer.py`（对话面板 UI 渲染）
- [ ] 实现 “战斗可跳过”机制（叙事层读取模拟结果，不阻塞）

**交付物**：
- 对话系统能在 UI 中显示文本和选项
- 玩家的选择能生成 Command 并推入队列
- 战斗跳过时叙事层能正常接收结果

**验收标准**：
- 对话树加载 < 100ms
- 玩家选择到 Command 入队延迟 < 1 Tick
- 跳过战斗时模拟正常完成，叙事正确消费结果

---

#### Phase 5：UI（3周）

**目标**：完整的玩家界面 +。

**任务清单**：
- [ ] 实现 `ui/main_window.py`（QMainWindow 主窗口布局，包括“选择测试地图生成器”）
- [ ] 实现 `ui/views.py`（战术地图、对话面板（仅预留骨架，UI细节留以后）、档案库（仅预留骨架，UI细节留以后））
- [ ] 实现至少 3 个shader主题：构成主义、日系动漫
- [ ] 实现 `ui/shaders/` 目录管理（每个主题独立的 .frag 文件）
- [ ] 实现 `ui/controls.py`（鼠标/键盘事件 → 叙事/命令队列）
- [ ] 实现自由摄像机的 UI 控制（拖动、滚轮、旋转）

**交付物**：
- 完整 UI 布局（地图 + 对话面板 + 信息栏）
- 主题切换菜单，切换时渲染风格实时变化
- 所有交互事件正确传递到逻辑层

**验收标准**：
- 主题切换耗时 < 100ms（无卡顿）
- UI 帧率 ≥ 30fps（地图渲染 + 对话面板）
- 所有鼠标/键盘事件正确响应

---

#### Phase 6：地面部队完整场景与测试套件（2周）

**目标**：完整的战役流程 + 自动化测试覆盖。

**任务清单**：
- [ ] 配置 3 个完整战役场景（含敌方编队、地形、目标）
- [ ] 实现单位AI（自由射击和移动，红外感应式反坦克地雷自主跟踪引爆，导弹跟踪，炮兵自主接受友军命令/玩家指令）
- [ ] 实现编队 AI（反炮兵疏散、反地雷绕行）
- [ ] 编写集成测试（完整战役流程自动化）
- [ ] 性能压测（2048 装甲 + 16384 弹丸，持续 10 分钟）
- [ ] 编写用户文档（操作手册 + 指挥官指南）
- [ ] 准备多种弹头（HE,APCR,SAPHE,ATGM,PELE...）
，呼叫炮兵/巡飞弹/CAS等支援，规避反炮支持超视距空战，RWR响应，规避导弹
**交付物**：
- 自动化测试覆盖 ≥ 80% 模块
- 性能报告：平均帧率、内存占用、峰值负载

**验收标准**：
- 编队AI无指令/无法与中央通信时可自主规划战斗
- 所有模块单元测试通过
- 性能压测结果符合设计目标（≥ 30fps 在 2048 装甲 + 16384 弹丸下）

#### Phase 7：空中部队设计
**目标**： 与地面部队统一管线实现体系作战
**任务清单**：

- [ ] 一般气动力学（不同迎风面的升力阻力）
- [ ] 实现单位AI（定高定航，接受自由开火命令后，空战自主发现-攻击-规避-返航）
- [ ] 实现编队 AI（编队飞行，分组降落）
- [ ] 编写集成测试（完整战役流程自动化）
- [ ] 性能测试
- [ ] 准备多种弹头（红外/雷达AAM，激光AGM，反辐射，自由落体/滑翔炸弹）

#### Phase 8：数据链体系作战（待补）
1. 参考：stochastic-warfare设计数据链，以数据链良好/观测良好实现战争迷雾（参考：docs\数据链与战争迷雾设计.md）
2. 对原有AI进行全面升级：
    装甲部队：呼叫支援（友方陆军/炮兵/巡飞弹/CAS），态势感知分析规划作战目标
    炮兵：自主接受友军请求打击指令并规避反炮
    空军：数据链A导B射，超视距控制与规避导弹，编队选择战术（如磨盘战术）
## 六、常见陷阱（必读）

| 陷阱 | 解决方案 |
|------|----------|
| Numba函数默认参数绑定数组长度 | 永远将数组作为参数传入，内部用 `.shape[0]` |
| 扩容时Numba并行循环正在跑 | 只在Tick间隙（sim_paused=True）时扩容 |
| 后台线程烘焙NavGraph时主线程修改heightmap | 后台线程读取 `.copy()`，不直接读原数组 |
| Qt FBO上下文丢失（最小化/拖动） | `detect_framebuffer()` 返回None时强制绑定Qt FBO |
| moderngl在QOpenGLWidget中渲染全黑 | 必须用 moderngl 官方 `fb.use()` 绑定 Qt FBO：`fb = ctx.detect_framebuffer(); fb.use()` 后再 `ctx.clear()` + `vao.render()`。直接 `ctx.clear()` 用的是 create_context 时缓存的 C 层 `mglo.fbo`，与实际 GL 绑定不同步，渲染会落到错误缓冲（原始 `glBindFramebuffer` 也能救，但非库内标准做法） |
| 并行碰撞写入同一装甲地址 | Numba原子操作（`-=` 自动安全），或结果数组暂存后串行提交 |
| 实例化渲染只画前几个实例 | moderngl 实例 buffer 的 format 必须加 `/i` 后缀（divisor=1）；缺省是 per-vertex，实例数据会被当顶点连续读取（源码 `FormatIterator::info()` + 官方 buffer_format.rst 确认） |
| 主题切换后地形变黑 | 菜单槽触发时 GL context 非 current，编译 program/重建 VAO/写 uniform 会静默失败 → 用 `makeCurrent()` 包裹 `ThemeManager.switch()` 及回调里的全部 GL 操作 |
| 主题切回时报 InvalidObject | `ThemeManager.switch` 释放旧 program 后必须移出缓存（`self._themes.pop(old.id, None)`），否则切回时复用已释放的 program |
| glm 矩阵转 numpy 用 `to_list()` 转置错乱 | **PyGLM 与 numpy 布局规则不同**：`mat4.to_list()` 返回 **4 个列向量**（非扁平数组），`np.array(to_list()).reshape(4,4, order='F')` 形状已是 (4,4) 属 no-op → 矩阵被转置 → 投影/渲染坐标错乱（HUD 圆全重合、clip_w 差 500 倍）。**必须用 `np.frombuffer(mat.to_bytes(), dtype=np.float32).reshape(4,4, order='F')`**（与 GL uniform 写入完全一致的 column-major 扁平，已验证与 glm 乘法逐元素一致）。注意 identity 矩阵转置不变 → 单测测不出，须配非 identity 相机投影对比测试 |
| HUD 标记低角度被地形剔除 | HUD pass 期间 `ctx.disable(DEPTH_TEST)`（标记永远最上层），画完 finally 恢复；HUD 顶点 z=0 深度 ~0.5，GL_LESS 下近处地形（深度<0.5）会把标记剔除 |
| 改 Numba 核后出现诡异报错（缓存与代码不一致） | 项目全量使用 `@njit(cache=True)`，改动签名/常量或升级 Numba 后旧缓存会静默失效。**每次修改任何 Numba 核后必须运行一次** `uv run python clear_numba_cache.py -y` 再跑测试 |


## 七、常用命令

```bash
# 初始化环境（依赖源已配置清华镜像，见 pyproject.toml 的 [tool.uv.index]）
uv sync
uv sync --extra dev   # 包含pytest, matplotlib等

# 运行测试
uv run python -m pytest --tb=short -q
uv run python -m pytest --cov=arkady_in_cradle --cov-report=term

# 运行主程序
uv run python main.py

# GUI 冒烟测试（渲染一帧并保存 tests/output/terrain_preview.png，运行前自动清空该目录）
uv run python tests/smoke_gui.py

# Phase 2 性能基准（2048 装甲 + 16384 弹丸的移动/检测帧率）
uv run python -m tests.benchmark_phase2

# 无头性能测试（Phase 3 碰撞）
uv run python -m tests.benchmark_collision

# 清除 Numba 缓存（.nbc/.nbi）——每次改 Numba 核后必须运行一次
uv run python clear_numba_cache.py -y      # 直接清（跳过确认）
uv run python clear_numba_cache.py -n      # 只预览，不删除
uv run python clear_numba_cache.py --all   # 连同 .pyc 一起清
```


## 八、一句话总结

> **SoA托底保性能，栅格地形弃八叉，抛射物用空间哈希，NavGraph管寻路，后台线程扛重算，叙事永远不碰模拟。**

---
## 九、非代码目录
- `story/` —— Obsidian 世界观笔记（中文 markdown），非源码
- `assets/shaders/`、`assets/textures/` —— 预留，当前为空
- `docs/` —— 项目实现的技术细节文档（`heightfield-design.md`、`phase3-combat-design.md`、`phase4-narrative-design.md`、`phase5-ui-pipeline-design.md`、`phase6-guidance-design.md`、`phase6-guidance-pipeline.md`、`数据链与战争迷雾设计.md`）
- `Reference/` —— 参考数据与外部项目（武器数据库 CSV、参考实现）


## 十、参考项目：stochastic-warfare

> 路径：`C:\Users\32965\Desktop\sw\stochastic-warfare\`
> 仓库：`github.com/claymsmith1/stochastic-warfare`（推测）
> 作者：Clay Smith (claymsmith1@gmail.com)
> 状态：97 Phase 完成，~10,739 测试通过

### 10.1 协议 ⚠️

**PolyForm Noncommercial License 1.0.0** — 个人学习/研究/实验可用，**任何组织（公司、教育机构、政府、军方、非营利等）使用需单独商业授权**。

本项目（Arkady in Cradle）**仅参考其设计思路与架构模式**，绝不复制其代码或数据文件。所有代码均为独立实现。

### 10.2 项目概要

高保真、多分辨率兵棋推演引擎（冷战→现代），纯 Python 无头模式。覆盖陆海空天全域，经验证于 73 Easting、福克兰海战、戈兰高地等历史战役。

核心数学模型：Markov 链（士气/天气）、Monte Carlo（战役结果）、Kalman 滤波（敌情估计）、Poisson 过程（故障/增援）、排队论（医疗/补给瓶颈）、SNR 检测理论。

### 10.3 Phase 3 可借鉴的模块

| 参考模块 | 路径 | 对我们的价值 |
|----------|------|-------------|
| **武器/弹药数据模型** | `combat/ammunition.py` | `WeaponDefinition` + `AmmoDefinition` (Pydantic)，`WeaponCategory`/`AmmoType`/`GuidanceType` 枚举层级 |
| **YAML 数据文件** | `data/weapons/guns/`, `data/units/armor/` | 枪炮参数（口径/初速/精度/射速/兼容弹药）+ 单位装甲值（正面/侧面/类型/乘员/装备清单） |
| **弹道引擎** | `combat/ballistics.py` | RK4 数值积分、马赫相关阻力、Coriolis 力、风偏、空气密度衰减——可简化为我们的 APFSDS 衰减公式 |
| **命中概率** | `combat/hit_probability.py` | 散布模型 + 乘员技能/目标运动/射手运动/能见度/姿态/武器状况/位置不确定性修正因子 |
| **毁伤解析** | `combat/damage.py` | DeMarre 变体穿深（动能弹）、HEAT 固定穿深、斜距效应/跳弹角、爆炸冲击波高斯衰减、破片 1/r²——**含 Numba JIT 内核** |
| **装备管理** | `entities/equipment.py` | `EquipmentItem`(condition/reliability/operational)、`EquipmentManager`(degradation/breakdown)——可参考实现我们的部件毁伤 |

### 10.4 YAML 数据驱动设计（核心参考）

stochastic-warfare 的数据组织方式对 Phase 3 有直接参考价值：

```
data/
├── weapons/guns/
│   ├── m256_120mm.yaml      # M1A2 主炮: caliber/velocity/accuracy/compatible_ammo
│   └── 2a46m_125mm.yaml     # T-90 主炮: caliber/velocity/accuracy/compatible_ammo
├── ammunition/              # 弹药定义 (M829A3 APFSDS, M830A1 HEAT-MP 等)
└── units/armor/
    ├── m1a2.yaml            # armor_front: 600mm, armor_side: 200mm, crew, equipment[]
    └── t90a.yaml            # armor_front: 800mm, armor_side: 300mm, Shtora-1 APS
```

**YAML → Pydantic 加载管线**：定义文件是纯数据，引擎代码定义行为。这与我们的 `WeaponConfig(BaseModel)` + CSV 加载是同一思路，但 YAML 的嵌套结构更适合表达"武器→兼容弹药→弹药参数"的多级关系。

### 10.5 关键设计模式对比

| 概念 | stochastic-warfare | Arkady in Cradle (计划) |
|------|-------------------|------------------------|
| 武器定义 | YAML → `WeaponDefinition` (Pydantic) | CSV → `WeaponConfig` (Pydantic) |
| 弹药定义 | YAML → `AmmoDefinition` (Pydantic) | 内嵌于 WeaponConfig，或独立 CSV |
| 穿深公式 | DeMarre 变体（含阻力衰减指数） | Odhner/Anderson 简化公式 |
| 命中概率 | 散布 + 10+ 修正因子 | 简化版：散布 + 乘员 + 运动 |
| 装甲值 | `armor_front` / `armor_side` (mm RHA) | 从 CSV 的 `Protection_Level` + 厚度估算 |
| 部位毁伤 | 未按 box 级别细分 | SoA 的 `PartType` (HULL/TURRET/ENGINE/TRACK/AMMO_RACK) |
| 装备状态 | `EquipmentItem.condition` 0~1 | 复用 `armor.health` + `box.active` |

### 10.6 不应借鉴的部分

- **Numba 使用方式**：stochastic-warfare 将 Numba 作为可选依赖（`optional_jit` 装饰器），我们的架构强制 Numba JIT 并行，不可降级
- **Tick 分辨率**：它用战略 3600s / 战役 300s / 战术 5s，我们用地面 1s / 空中 0.1s / 子弹 0.01s
- **空间模型**：它用图+栅格+连续混合，我们纯连续坐标
- **ECS 分离**：它 entities 纯数据 + modules 行为，我们 SoA 数组 + Numba 过程式

🐱 猫猫Code Agent，请开始你的工作。有任何不确定的地方，先看这份文档；文档里没有的，翻源码；源码里找不到的，再来问人类。加油！