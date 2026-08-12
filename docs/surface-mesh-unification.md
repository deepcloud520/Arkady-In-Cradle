# 表面网格统一架构（去 layer · GreedyMesh · 网格即导航）⚠️ 已归档

> 状态：**已归档**（2026-08-12），被 `docs/navmesh-region-growth-design.md` 替代
> 归档原因：GreedyMesh 按精确 z 分组导致斜坡碎面片，无法支撑 10000×10000 地图
> 对应 AGENTS.md 3.2 / 3.3 / 3.6 的**架构演进方向**：推翻 baker 多边形图管线，
> 以 GreedyMesh 表面网格统一 渲染 / 导航 / 碰撞 / 单位层。
> 本文档是多次设计讨论的收敛结论，供未来实施时遵循。

## 〇、一句话

> **所有 span 顶面 = GreedyMesh 表面网格；一个网格三用（渲染 + 导航 + 碰撞）；layer 概念删除，单位层由 z 坐标自决。**

---

## 一、背景与动机（为什么重构）

### 1.1 现状问题（2026-08-11 发现）

| # | 问题 | 根因 |
|---|------|------|
| 1 | **baker 只烘焙地面层**：`_build_walkable_mask` 硬编码 `spans[col, 0, 1]`（恒取第 0 段） | `layer=0` 隐含「地面」的语义耦合 |
| 2 | **真实烘焙几乎总是 1 个多边形**，portal_edges 恒空 | 连通域 = 多边形，且不做凸分解 |
| 3 | **桥面导航不可达**：桥 12m 高 vs 地面 2m，`max_climb=2` 判不连通 | 桥 = 垂直悬空面板，桥头无坡道 |
| 4 | **结构层渲染裸逐格**：每格 4 顶点独立四边形，不合并 | `surface.py::_extract_structures` 未做 GreedyMesh |
| 5 | **`add_span` 合并吞材质**：地面 `[0,12)`+桥面贴地 `[12,14)` 合并成 1 段 BRIDGE | 合并条件只看 z 相邻，不看 mat 相同 |
| 6 | **`armor.layer` 冗余**：单位硬编码「在地面/桥/隧道层」 | 单位 z + 碰撞箱尺寸本可自决 |

### 1.2 核心洞察（讨论收敛）

1. **「Layer」是两样东西**：
   - ① 存储下标 = 同列第几段实心（`spans[col, 0]`）—— **去不掉**，是数组索引
   - ② 语义标签 = 「layer 0=地面 / 1=桥」硬编码—— **应该去掉**
2. **破坏语义应由 span 的 `mat` 承担**（第 4 维），不该由 layer 序号承担
3. **可走性判定应遍历所有 span 顶面**（「本质都是 span 扫描出的表面」），不该只看 layer 0
4. **baker 多边形图依赖 layer 分层**：layer 一删，baker 管线（walkable→连通域→多边形→邻接）失去输入 → 死亡
5. **GreedyMesh 面片本身就是导航需要的凸多边形**：面片=四边形（凸）、共享边=portal、无需凸分解

---

## 二、目标架构：三位一体表面网格

```
Heightfield spans[col] = [[zmin,zmax,area,mat], ...]   ← 唯一数据源
        │
        ▼
GreedyMesh 提取
  ├── 遍历所有列的所有 span 顶面
  ├── 按 (world_z, mat) 分组
  ├── 每组的 (gx,gy) 网格做 2D greedy mesh 合并（同平面面片合并）
  ├── 生成四边形面片 SurfacePatch[poly, z, mat, normal]
  ├── 可走性筛选（面积 ≥ 阈值 ∧ 坡度 < max_slope ∧ 非悬空）
  └── 面片邻接图（共享边 + 相邻面片 z 落差 < max_climb）
        │
        ├──→ render_data（顶点+索引，同材质面片合并 → 少 drawcall）
        └──→ NavGraph 替代（patch_centroids + patch_adjacency）
              │
              └── A* 面片邻接图 → 面片序列 → SSFA（沿共享边收紧）→ 路标点
```

### 2.1 三个核心收益

| 用途 | 机制 | 价值 |
|------|------|------|
| **渲染** | 同 (z, mat) 面片合并 → 大四边形 | 减少 drawcall / 材质切换（结构层可减 ~9×） |
| **导航** | 面片 = 导航节点；共享边 = portal | **免费得到导航图**（A* 面片序列 + SSFA 沿 portal 收紧） |
| **碰撞/单位层** | 面片 z 高度 → 单位 z 自决层 | 删除 `armor.layer`，物理按 z 重叠判定 |

> **注：面片不一定是凸四边形**。GreedyMesh 对规则矩形桥合并成大矩形（凸），但建筑可能是
> L 形 / T 形 / 任意凹多边形。因此导航 = A*（面片序列）+ **SSFA funnel（沿共享边收紧）**，
> 而非「形心连线」——凹多边形形心可能落在多边形外，形心路径不可靠。funnel 是核心组件。

---

## 三、顶点数实测（2026-08-11）

### 3.1 当前基线

| 场景 | 地面层 | 结构层（桥） | 总顶点 | 提取耗时 |
|------|:---:|:---:|:---:|:---:|
| 200×200 + 40×6 桥 | 164,641 | 960 | 165,601 | 61.6ms |
| 300×300 + 80×8 桥 | 371,441 | 2,560 | 374,001 | 156.8ms |

> 地面层 = 平滑连续网格（SUBDIV 细分），占 99%+ 顶点——高度连续变化，**无法像平面一样合并**。

### 3.2 GreedyMesh 对结构层收益（理论）

| 桥尺寸 | 当前结构顶点 | 理想 GreedyMesh | 缩减 |
|:---:|:---:|:---:|:---:|
| 40×6 | 960 | ~372 | 2.6× |
| 80×8 | 2,560 | ~708 | 3.6× |
| 200×20 | 16,000 | ~1,764 | **9.1×** |

### 3.3 结论：收益不在顶点数，在架构

- 结构层只占总顶点 **<1%** → 追求顶点数不值得做
- **真正价值**：① 导航免费（面片=多边形+portal）② 渲染状态合并 ③ 语义统一
- **性能影响**：提取耗时 60~157ms（加载期可接受）；运行期导航从「baker 多边形图」变为「面片图」，A* 规模相当或更小

---

## 四、实施清单（激进方案）

### 4.1 删除

| 文件/模块 | 原因 |
|-----------|------|
| `navmesh/baker.py` | 多边形提取管线失去输入（依赖 layer） |
| `navmesh/tiles.py` | 表面局部重建替代 tile 脏标记 |
| `entities/pools.py::LayerType` / `armor.layer` | 单位 z 自决层 |
| `heightfield.py::surface_z(layer=N)` API | 改按 z 查 surface mesh 投影 |

> ⚠️ **funnel 不删**（见 §4.4）：面片可能是凹多边形（建筑 L 形等），形心路径不可靠；
> 且即便凸面片，形心连线也非最短。funnel 沿 portal 收紧是核心组件，现有实现直接复用。

### 4.2 新建/重写

| 文件 | 内容 |
|------|------|
| `terrain/surface_mesh.py`（新建） | GreedyMesh 核心：`SurfaceMesh`（面片 + 邻接 + portal） |
| `terrain/surface.py`（重写） | 统一对所有 span 顶面做 GreedyMesh，替代地面/结构分离 |
| `movement/nav.py`（新建） | A* 在面片邻接图上跑，返回**面片 ID 序列**（替代 `baker.astar`） |
| `movement/pathfinding.py`（适配） | `find_path` → A* 面片序列 → **复用现有 `funnel`（SSFA）** → 路标点 |

### 4.4 funnel 复用（核心，不删）

现有 `funnel(poly_ids, graph, start, end)` 已完整实现 SSFA 且消费 `portal_edges`：

- 把 `NavGraph` 替换为 `SurfaceMesh`（面片 + 共享边 portal）即可直接复用，接口兼容
- 管线：`A* 面片邻接图 → 面片序列 → funnel 沿共享边收紧 → 路标点`
- 面片可能是凹多边形（建筑 L/T 形）→ funnel 收紧保证路径贴着导航面走，不穿出

### 4.3 顺带修复

| 问题 | 修复 |
|------|------|
| `add_span` 合并吞材质 | 合并条件加 `and mat 相同`（`s[3] == merged[-1][3]`） |

---

## 五、实施路线图

```
Step 1: add_span 合并按 mat 区分（基础 bug 修复）
Step 2: surface_mesh.py —— GreedyMesh 面片提取（核心）
        ├── 按 (z, mat) 分组 → 2D greedy mesh 合并
        ├── 面片可走性筛选（面积/坡度/悬空）
        └── 面片邻接图构建（共享边 + z 落差）
Step 3: surface.py 统一 —— 地面/结构合并为单一 GreedyMesh 提取
Step 4: 导航适配 —— A* 走面片邻接图，Pathfinder 重写
Step 5: 去 layer —— 删 LayerType / armor.layer，碰撞改按 z 重叠
Step 6: 清理 —— 删 baker/tiles 死代码 + 测试重写（funnel 保留）
```

---

## 六、风险与决策记录

| 风险 | 应对 |
|------|------|
| GreedyMesh 合并算法复杂 | 标准 2D sweep（参考 Recast rcBuildContours），可先用朴素「逐行扩展矩形」 |
| 面片邻接节点数 | 合并后通常 <2000，A* 可承受 |
| 删除 baker 丢失「桥/隧分区域」 | GreedyMesh 按 z 落差自然隔离不同高度面片 |
| 现有测试大面积失效 | 逐一重写（integration_heightfield / test_pathfinding / test_terrain 等） |
| 地面层不合并 → 顶点不降 | 接受：地面是曲面，GreedyMesh 只优化平面结构层 |

### 决策记录

| 日期 | 决策 |
|------|------|
| 2026-08-11 | 弃 Sort&Sweep，编队推开用空间哈希（复用 build_grid_index） |
| 2026-08-11 | 确立「去 layer + GreedyMesh 三位一体」为最终架构方向 |
| 2026-08-11 | 确认 GreedyMesh 价值在导航/统一，不在顶点数（结构层仅占 <1%） |
