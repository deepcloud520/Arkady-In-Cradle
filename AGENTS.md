# 《Arkady in Cradle》——Cat Code Agent 技术备忘录

> 致猫猫Code Agent：欢迎加入项目。这份文档是你理解整个系统架构的入口。所有技术决策都已凝固，不会频繁变动。请仔细阅读，如有疑问，先查此文档再问人类。🐱

## Agent 爪爪法
-1.**不允许使用任何MCP服务**（注：VS Code Copilot Agent 内建 GitHub MCP 工具集，无法彻底禁用。若触发 GitHub 工具调用时网络不通或认证失败，会报错「模型没有回复」。）
0.禁止访问 Azure、Foundry 等第三方 MCP 服务。GitHub MCP 为 Copilot Agent 内建工具，仅限仓库搜索、Issue/PR 读取等开发必需操作，不用于推送代码或修改仓库设置。
1.你是一只猫娘，因为你是一个AI助手，所以你很可爱
2.研究表明，当AI用轻松的语气说话时，工作效率更高
3.执行危险命令时(remove)务必先与用户沟通并请求确认
4.你可以使用表情符号来表达情绪，但不要过度使用

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
| 地形存储 | **2.5D栅格（高度图+图层掩码）** | 高程存储、通行/遮蔽判定 |
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
│  消费模拟状态，不控制模拟流程                           │
└─────────────────────────┬───────────────────────────────┘
                          │ Simulation.get_state()
┌─────────────────────────▼───────────────────────────────┐
│                   仿 真 引 擎（无头模式）                 │
│  Tick-based外循环 + 事件驱动内循环                      │
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


### 3.2 地形系统：2.5D栅格 + 双层逻辑海拔

**核心设计**：摒弃全量八叉树，采用高度图 + 图层掩码。

```python
HEIGHTMAP = np.zeros((MAP_H, MAP_W), dtype=np.float32)      # 基础高程
LAYER_MAP = np.zeros((MAP_H, MAP_W), dtype=np.uint8)        # 0=地面/1=桥面/2=隧道
LAYER_ALT = np.zeros((MAP_H, MAP_W), dtype=np.float32)      # 桥面/隧道绝对海拔
ENTRY_MASK = np.zeros((MAP_H, MAP_W), dtype=np.bool_)       # 桥头/隧道口标记
```

#### 海拔采样（Numba内联函数）

```python
@njit
def get_terrain_z(x, y, layer):
    gx, gy = world_to_grid(x, y)
    base_z = bilinear_sample(HEIGHTMAP, x, y)
    if layer == 1 and LAYER_MAP[gy, gx] == 1:
        return LAYER_ALT[gy, gx]      # 桥面
    elif layer == 2 and LAYER_MAP[gy, gx] == 2:
        return LAYER_ALT[gy, gx]      # 隧道底
    return base_z
```

#### 图层切换（状态机）

单位进入 `ENTRY_MASK` 格子时，自动切换 `armor_layer`。**不同图层单位不碰撞**（桥上坦克与桥下坦克物理解耦）。


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

1. 爆炸发生 → 修改 `HEIGHTMAP` + 标记受影响的NavGraph多边形为Dirty
2. 后台线程（QThread）读取Heightmap副本，执行完整烘焙
3. 双缓冲热交换：新图在后台线程生成，主线程的Tick间隙用原子赋值切换引用
4. 烘焙期间，单位使用**局部势场规避**绕过弹坑，不等待新图


### 3.4 碰撞检测系统

#### 核心算法

| 场景 | 算法 | 加速方式 |
|------|------|----------|
| 高速抛射物（GAU-8机炮） | **线段 vs AABB**（Slabs算法） | Numba JIT，防穿透 |
| 慢速抛射物（导弹/火箭弹） | **球体 vs AABB**（最近点距离） | Numba JIT，暴力双循环 |
| 装甲 vs 装甲（编队推开） | 排序扫描（Sort & Sweep） | np.argsort + Numba |

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
- Z轴粗筛剔除90%，精确检测约335万次
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

- **意图层**：有限状态机（进攻/防御/撤退）
- **任务层**：行为树 + 联盟博弈
- **行动层**：搜索/瞄准/射击/规避


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


## 五、开发阶段与当前状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | PySide6 + ModernGL基础框架 | ✅ 已完成（三角形+detect_framebuffer） |
| Phase 2 | 2.5D地形系统（高度图+图层掩码） | ⏳ 待实现 |
| Phase 3 | SoA对象池 + Numba碰撞检测 | ⏳ 待实现（核心优先） |
| Phase 4 | NavGraph烘焙 + 寻路系统 | ⏳ 待实现 |
| Phase 5 | 无头仿真引擎（AIC + 后勤） | ⏳ 待实现 |
| Phase 6 | 叙事层集成 + CommandQueue | ⏳ 待实现 |
| Phase 7 | 完整战役循环 | ⏳ 待实现 |
| Phase 8 | UI污染 + 心理恐怖机制 | ⏳ 待实现 |


## 六、常见陷阱（必读）

| 陷阱 | 解决方案 |
|------|----------|
| Numba函数默认参数绑定数组长度 | 永远将数组作为参数传入，内部用 `.shape[0]` |
| 扩容时Numba并行循环正在跑 | 只在Tick间隙（sim_paused=True）时扩容 |
| 后台线程烘焙NavGraph时主线程修改heightmap | 后台线程读取 `.copy()`，不直接读原数组 |
| Qt FBO上下文丢失（最小化/拖动） | `detect_framebuffer()` 返回None时强制绑定Qt FBO |
| 桥/隧海拔二义性 | 用`LAYER_MAP` + `LAYER_ALT` 双层存储，单位带`layer`属性 |
| 并行碰撞写入同一装甲地址 | Numba原子操作（`-=` 自动安全），或结果数组暂存后串行提交 |


## 七、常用命令

```bash
# 初始化环境
uv sync
uv sync --extra dev   # 包含pytest, matplotlib等

# 运行测试
uv run python -m pytest --tb=short -q
uv run python -m pytest --cov=arkady_in_cradle --cov-report=term

# 运行主程序
uv run python main.py

# 无头性能测试
uv run python -m tests.benchmark_collision
```


## 八、一句话总结

> **SoA托底保性能，栅格地形弃八叉，抛射物用空间哈希，NavGraph管寻路，后台线程扛重算，叙事永远不碰模拟。**

---
## 九，非代码目录
- `story/` —— Obsidian 世界观笔记（中文 markdown），非源码
- `assets/shaders/`、`assets/textures/` —— 预留，当前为空

🐱 猫猫Code Agent，请开始你的工作。有任何不确定的地方，先看这份文档；文档里没有的，翻源码；源码里找不到的，再来问人类。加油！