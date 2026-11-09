# 建筑实体池 + span 判稳 + 结构坍塌 · 设计待办

> **状态：未开工（待办）** · 2026-11-05 定稿（第三版：判稳回到 span 上算）
> 本文是设计结论 + 任务清单，**不是已实现功能**。目前代码里没有 `structure/` 包、
> 没有 `span_owner`、没有 `MAT_RUBBLE`、没有局部重建，请勿当作现状引用。
>
> 相关现状文档：`docs/heightfield-design.md`、`docs/worldmesh-ground-patch-policy.md`、
> `docs/架构评审报告.md` §3.3（自由破坏是半闭环）。

---

## 1. 问题

所有混凝土结构（桥、楼、隧道顶板）在现有模型里只有「实体」，没有「结构」：

- `heightfield` span 只有 `[zmin, zmax, area, mat]`，没有归属、没有构件、没有受力类型；
- 破坏只有 `remove_span()`（SAL 炸桥 → 半径 3 m 柱体）——结果是**挖洞**，不是塌；
- 结构面片被豁免悬空判定（`world_mesh.py:502-505`），柱被挖掉后楼板仍 `walkable`；
- 破坏后 `invalidate_geometry()` 只清缓存；移动 hook 持加载期旧 WorldMesh、
  `GraphPathfinder` 持加载期 NavGraph → 单位踩空气、AI 照原路走；
- **存档不含地形**（`serialize_battlefield()` 只打包各池 + 后勤 + 孪生），
  所以连现在的 SAL 弹坑都是**读档即消失**。

## 2. 职责划分：几何在 span，身份在 building

**不做"建筑状态 → span 投影"的缓存模型**（那会引入第二条同步路径）。分成三层：

| 数据 | 权威 | 谁写 | 说明 |
|---|---|---|---|
| 结构几何（柱/板/墙/桥面实体的分布） | **heightfield spans** | `BuildingPool.hit()` 经 `remove_span` / 坍塌提交 | 破坏 = 改 span，不重建、不投影 |
| 归属（哪段 span 属于哪栋） | `span_owner` | 生成期写入，`rebuild` 时维护 | 批处理、统计、守卫、坍塌范围 |
| 生命周期（完好/受损/坍塌中/废墟）+ 完整性 | **BuildingPool** | 判稳结束后更新 | AI / 叙事 / 事件 / 存档锚点 |
| 构件级几何（索、斜杆、拱肋） | **AssemblyModel** | 装配体加载器 | 只有英雄结构；span 放代理 |
| 破坏记录（持久化） | **terrain delta** | `hit()` 与坍塌提交同步写 | 见 §3.6 |

一句话：**span 是几何，也是判稳的输入；building 是身份与状态，不重复存几何。**

### 2.1 为什么判稳直接在 span 上跑

1. 破坏天然就是 span 变更（`remove_span`），无需"先改模型再投影"；
2. 支撑 BFS 的节点本来就是"列 × 段"，span 就是它的自然表示；
3. 不存在"影子与本体不一致"——**唯一的失败模式是 delta 漏记**（见 §3.6 与守卫）；
4. 生成器不需要另外产出柱网/层表（少一份要与几何对齐的数据）。

### 2.2 材质退化为"支撑规则提示"

`MAT_COLUMN / MAT_BEAM / MAT_SLAB / MAT_WALL / MAT_DECK` 仍要加，但**只用来给 BFS 提供
规则参数**（能不能横向传力、悬挑极限多少），不是用来"标出整个结构模型"。

若生成器把楼板写成 `MAT_BUILDING`，判稳就会按"墙"处理（不允许悬挑）→ 误判。
所以生成器写材质必须正确，这是新增的一条数据契约。

## 3. 数据模型

### 3.1 `span_owner`

```python
# heightfield 内新增，与 spans 同形状（不新增数组维度）
span_owner = np.full((n_cols, MAX_SPANS), -1, dtype=np.int16)  # -1 = 地形 / 废墟
```

- 生成期：建筑投影自己的 span 时写 `bid`；
- 运行时：`hit()` 删 span 时按 owner 校验；坍塌提交把失稳段整段清空并写 `owner=-1` 的废墟；
- 统计：一栋楼"还剩多少实体体积" = 该 owner 的 span 体积和 → 直接喂完整性。

### 3.2 BuildingPool（精简版，不再持有几何）

```python
MAX_BUILDINGS = 512

bld_kind      = zeros(MAX_BUILDINGS, uint8)   # FRAME / BRIDGE / ASSEMBLY / TUNNEL_LID
bld_state     = zeros(MAX_BUILDINGS, uint8)   # INTACT / DAMAGED / COLLAPSING / RUINED
bld_aabb      = zeros((MAX_BUILDINGS, 6), float32)
bld_integrity = zeros(MAX_BUILDINGS, float32) # 0~1（由剩余 span 体积算）
bld_active    = zeros(MAX_BUILDINGS, bool)

bld_model = [None] * MAX_BUILDINGS            # 仅 ASSEMBLY 用；FRAME 为 None
```

`FRAME` 建筑**没有 model 对象**：判稳只看 span + 材质规则。

### 3.3 支撑规则表（BFS 的边权与上限）

| 材质 | 竖向传力 | 横向传力 | 悬挑/跨距上限 | 备注 |
|---|---|---|---|---|
| `MAT_COLUMN` | ✅（0 代价） | ❌ | — | 必须堆到基础，不能悬挑 |
| `MAT_WALL` | ✅ | ⚠️ 有限（1 格） | 1 m | 承重墙，侧向可带一点 |
| `MAT_BEAM` | ✅ | ✅（1 代价/格） | 6~8 m | 梁：跨在柱之间 |
| `MAT_SLAB` | ✅ | ✅ | 3~4 m | 楼板：靠梁/柱 |
| `MAT_DECK` | ✅ | ✅ | 4 m | 桥面板（代理） |
| `MAT_BUILDING` | ✅ | ❌ | — | 非承重装饰：不传力，自身靠邻居 |
| `MAT_RUBBLE` | ✅ | ❌ | — | 废墟：只堆叠 |

BFS：节点 = `(col, span_idx)`；根 = `zmin ≤ 地面/基础顶面` 的段；
边 = 竖向接触（代价 0）+ 横向同 z 重叠的邻列段（代价 1，受上表上限）；
`dist > 上限` 的段 = 失稳。

### 3.4 AssemblyModel（仅英雄结构）

```yaml
id: bridge_alpha
kind: cable_stayed
nodes:  [{id: p0, x: 512, y: 300, z: 12}, {id: p1, x: 512, y: 300, z: 62}]
members:
  - {a: p0, b: p1, type: COLUMN, mat: steel, cap_c: 8.0e6}
  - {a: p1, b: d12, type: CABLE, mat: cable, cap_t: 2.0e6}
slabs:  [{poly: [...], z: 14, mat: DECK, thickness: 0.3}]
parts:  [{id: main_span, members: [...], collapse: DROP, debris: 24}]
```

- 机构判定：地基节点沿 active 构件 BFS（柱/撑双向、索只受拉、梁需两端支撑）；
- 载荷重分配：设计期预计算 `infl[m][k]`，运行时查表累加超 `cap` 再失效；
- 投影：柱/塔/桥面 → span（写 owner）；**索与斜杆不投影**，显式接受"弹道可穿索"。

### 3.5 废墟分布

```
V    = Σ 坍塌体积（m³，含松散系数 K：密实 1.4~1.6 / 含空腔 2.0~2.5）
R    = (3V / (π·tanθ))^(1/3)          θ = 30~40°
h(d) = (R - d)·tanθ,  d < R
```

三钳制：① `h ≤ H_RUINS_MAX`（楼 8 m / 桥 2 m）；② `h ≤ 上方残存段 zmin − base − 0.2`；
③ `h < 0.15` 不写。写回时把该建筑失稳段清空，再 `add_span(..., mat=MAT_RUBBLE, owner=-1)`。
**必须批次化**（一次坍塌所有构件一起算 V/重心），逐构件写会丢体积、堆出多峰。

### 3.6 存档：terrain delta（新增，必须做）

现状 `serialize_battlefield()` 不打包地形 → 任何 span 破坏都会在读档时消失。
span 成为几何权威后，必须持久化**列级差量**：

```python
# 只存被改过的列（塌一栋楼通常几百列，msgpack 几十 KB）
terrain_delta = [
    {"col": (gx, gy),
     "spans": [[z0, z1, area, mat, owner], ...]},   # 该列当前完整 span 快照
]
```

规则：`hit()` 与坍塌提交在改 span 的同时写 delta（同一函数内完成，不允许分叉）；
读档时先按 delta 重写这些列，再恢复 WorldMesh / NavGraph 缓存。
**守卫**：dev 模式下每 N tick 校验"当前 span 快照 vs 存档重建结果"的哈希，漏记即报错。

## 4. 运行时流程

```
combat 命中
  → BuildingPool.hit(x, y, z, pen, kind)
        ├ 命中某建筑 bbox → 换算 → 直接在 span 上打洞（remove_span，owner 校验）
        ├ 记录 terrain delta
        ├ bld_state = DAMAGED，bld_dirty[bid] = True
        └ 返回 HitResult（未命中建筑 → 调用方走地形弹坑）
  → BuildingPool.step(dt)                        # 1 s tick，只处理脏建筑
        ├ evaluate_support(hf, bid)              # §3.3 的 BFS，仅该建筑 bbox 内的 span
        ├ 失稳段 → commit_collapse()（删段 + 废墟 + delta）
        ├ 更新 bld_integrity / bld_state
        └ emit("building_damaged" / "building_collapsed", {...})
  → 下游（零改动，继续只读 heightfield）
       渲染 VAO 重建 / 局部 NavGraph 重烘焙 + set_graph / 单位 mesh_id 重绑 / 叙事
```

### 4.1 下游四条断线（必须一起补）

| 断线 | 现状 | 修法 |
|---|---|---|
| 渲染 | `gl_widget._build_meshes()` 只在初始化调用 | 监听 `building_*` 事件 → `makeCurrent()` 后重建 VAO |
| 物理 | `build_armor_movement_hook()` 加载期 `flatten_world_mesh()` 固化 | hook 内每帧从 `terrain.world_mesh` 取；重建后重新 attach |
| 寻路 | `GraphPathfinder` 持加载期快照，`set_graph()` 零调用 | 异步局部烘焙 → `set_graph()` 热交换 |
| 面片句柄 | `armor.mesh_id` 是面片下标 | 重建后按 `hint_z` 重绑；无面片且仍在原高度 → 进入 falling |

## 5. 解耦判定

| 耦合面 | 现状 | 契约 | 判定 |
|---|---|---|---|
| 伤害入口 | combat 直接 `remove_span` | **唯一 `BuildingPool.hit()`**（改 span + 写 delta + 标脏） | ❌ 必须补 |
| span 写入 | 谁都能改 | **owner 守卫**：非 hit 路径禁止删 `owner >= 0` 的段 | ❌ 必须补 |
| 碰撞/遮挡/可行走 | 读 hf.spans | 无额外契约（几何本来就在 span） | ✅ 天然满足 |
| 存档 | 不含地形 | **terrain delta** | ❌ 必须补 |
| 渲染 | 仅地形 + 单位 | 装配体需要 **render adapter**（不 import structure 常量） | ⚠️ 仅英雄 |
| 性能 | 破坏后全量重建（≈4 s） | **局部重建** | ❌ 硬门槛 |

结论：**这版比"缓存投影"解耦更彻底**——碰撞/遮挡/可行走三条天然满足，只剩 5 个明确契约，
且其中 3 个（入口/hit、守卫、delta）在同一个函数里一起完成。

## 6. 分期任务

### P0 · 建筑池 + span 判稳（预计 1.5 天）

- [ ] `heightfield`：新增 `span_owner` + `spans_of()/write_col()` 公开；新增 `MAT_RUBBLE`；
      `MAX_SPANS` 8 → 16；`_write_col` 截断改为「保上下两端」
- [ ] 新增 `Airia/lulilo/structure/`（`pool.py` / `support.py` / `spans.py`）
- [ ] `support.py::evaluate_support()`：§3.3 的 BFS（材质规则表可配置）
- [ ] `terrain/factory.py`：建筑生成器写正确材质 + 写 `span_owner`
- [ ] `BuildingPool.hit()` 唯一入口；`engagement._remove_terrain_crater()` 改为经此路由
- [ ] owner 守卫（`remove_span` 遇 `owner >= 0` 抛错）
- [ ] 四条断线接线
- [ ] 验收：炸柱脚 → 上层楼板立即不可站立、AI 绕行；单栋判稳 < 0.1 ms

### P1 · 坍塌提交 + 废墟 + 存档（预计 1 天）

- [ ] `terrain/debris.py::commit_collapse()`：批次化体积守恒 + 休止角 + 三钳制
- [ ] terrain delta 写入/读档（`serialization.py` 扩 `SAVE_VERSION`）
- [ ] 局部重建（脏块，接 `navmesh/_tiles.py` 的 `mark_dirty`）+ 渲染 VAO 刷新
- [ ] `MAT_RUBBLE` 渲染色 / 坡度代价 / 导航代价
- [ ] 单位掉落 / 被埋判定
- [ ] 验收：塌完留下可走废墟丘、桥下被堵；单次结算 < 0.3 ms；存读档后废墟与破坏一致

### P2 · 装配体英雄结构（预计 2~3 天）

- [ ] `AssemblyModel` + YAML 加载器（与 span 投影同源）
- [ ] 机构判定 + 影响矩阵查表 + 坍塌编排（`DROP/TOPPLE/SNAP/SETTLE`）
- [ ] 碎片池 + 落地折算废墟
- [ ] `render/adapter.py::render_batches()`（不 import structure 常量）
- [ ] 验收：斜拉桥断索 → 主跨下挠 → 整跨坠落 + 索塔倾倒；事件结算 < 1 ms

## 7. 开放问题

1. **材质契约**：生成器写错材质（楼板写成 `MAT_BUILDING`）会静默误判；要不要在加载期做
   「材质-几何一致性」校验（如板厚 ≤1 层必须是 SLAB）？
2. **多建筑共用列**：临街两栋贴在一起，一栋塌后邻居失去侧向支撑要不要建模？
3. **重建粒度**：WorldMesh/NavGraph 目前是全局重建（≈4 s），必须做脏块（`_tiles.py` 有骨架）。
4. **`MAX_SPANS` 8 → 16**：内存 +50%~100%（400×400 约 +1.2 MB），Numba 核需重跑基准。
5. **废墟再破坏/清理**：owner=-1 的废墟可 `remove_span`，清掉后地面高度如何回落？
6. **建筑生成粒度**：按"一栋一栋"生成，还是先出街区再切分建筑？

## 8. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| delta 漏记 | 破坏不持久（读档复活/消失） | hit 与 delta 写在同一函数；dev 哈希自检 |
| 材质误标 | 判稳按错规则 → 该塌不塌 / 乱塌 | 加载期一致性校验 + 生成器单元测试 |
| 性能特例化 | 全局重建导致塌一次卡 4 s | P1 必做局部重建，否则不进 P2 |
| 表达力天花板 | 索/斜杆不投影，弹道可穿 | 记录为已知近似，不做"打到索"判定 |
| 演示误导 | 2D 演示场景是"机构"（细柱/无斜撑） | 演示仅验证判定与废墟分布 |

## 9. 附录

- 2D 演示：`demo/collapse_2d.py`（出帧图）、`demo/collapse_2d.html`（可交互）
- 现状半闭环证据：`docs/架构评审报告.md` §3.3；`combat/engagement.py::_remove_terrain_crater()`；
  `terrain/map.py::invalidate_geometry()`
- `demo/` 不属于引擎依赖链，不参与 pytest
