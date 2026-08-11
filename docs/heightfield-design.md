# 高度场（Heightfield）架构设计 —— 调用结构契约

> 状态：**结构已定，内容待填**（2026-08-10）
> 对应 AGENTS.md 3.2 / 3.3 / 3.6 的重构方向。

## 一、背景与动机

以「列式 span 高度场」统一表示 地面/桥/隧道/建筑，替代 `heightmap + layer_map/layer_alt` 双层存储：

- 地面 = 最低实心段顶面；桥 = 高位悬空段（桥下无实心，可被炸断）；隧道 = 上下段夹空隙；建筑 = 多层段
- 无需 `LAYER_MAP` / `ENTRY_MASK` 特判，"图层"由 span 在列内位置自然决定
- **自由破坏** = `remove_span()` 截断/删除受影响列
- **NavMesh 局部更新** = tile 脏标记 + 后台局部重建（`DetourTileCache` 思路）

## 二、许可证合规

算法思路参考 [recastnavigation](https://github.com/recastnavigation/recastnavigation)
(c) 2009 Mikko Mononen，**zlib License**：允许商用/修改/再分发，但不得谎称原创、修改版需标记、版权声明不得移除。
**每个复刻文件头须保留上述 zlib 版权注释。**（本设计所述文件均已加注）

## 三、分层依赖（严格单向，AGENTS.md 4.1）

```
core → lulilo → terrain.heightfield
                          ├→ terrain.surface
                          └→ navmesh.baker → navmesh.tiles
上层（movement/detection/combat/simulation/render）消费，不反向依赖。
```

## 四、模块接口契约

### 4.1 `terrain/heightfield.py` —— `Heightfield`
| 方法 | 签名 | 时机 |
|------|------|------|
| 构造 | `Heightfield(width, height, cs, ch, z_min, z_max)` | 加载期 |
| 写·增 | `add_span(gx, gy, z0, z1, area=0, mat=0) -> bool` | 加载期 |
| 写·删 | `remove_span(gx, gy, z0, z1) -> bool` | 运行时破坏 |
| 读 | `ground_z(gx, gy) -> float` | 热路径 |
| 读 | `surface_z(gx, gy, layer=0) -> float` | 热路径 |
| 读 | `solid_at(gx, gy, z) -> bool` | 热路径 |
| 内 | `span_count(gx, gy) -> int` | 调试/测试 |

状态：**已实现**（`Airis/lulilo/terrain/heightfield.py`），含 Numba 热路径 + 单测 `tests/test_heightfield.py`。

### 4.2 `terrain/surface.py` —— `SurfaceExtractor`
| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `SurfaceExtractor(hf)` | 持有只读引用 |
| 全量 | `extract() -> RenderData` | **加载期统一**（竖棱柱/greedy） |
| 分块 | `extract_tile(gx0, gy0, gx1, gy1) -> RenderData` | 运行时局部重提取 |

`RenderData`：`vertices (N,6) float32 [x,y,z,r,g,b]` + `indices (M,) uint32`（与现有 `to_render_data` 同构）。

### 4.3 `navmesh/baker.py` —— `Baker` / `NavGraph`
| 方法 | 签名 | 说明 |
|------|------|------|
| 全量 | `bake() -> NavGraph` | 高度场→可行走面→region/contour/polygon |
| 分块 | `bake_tile(gx0, gy0, gx1, gy1) -> NavGraph` | 局部重建 |

`NavGraph`：多边形节点 + 邻接表（AGENTS.md 3.3），运行时 A*。

### 4.4 `navmesh/tiles.py` —— `TileManager`
| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `TileManager(hf, tile_size)` | 按格划分 tile |
| 标记 | `mark_dirty(gx0, gy0, gx1, gy1)` | 破坏后调用 |
| 重建 | `rebuild_dirty(async_=True)` | 后台线程局部重建 + 双缓冲热交换 |
| 查询 | `nav_graph()` | 当前生效 NavGraph（原子引用） |

## 五、数据流

```text
加载期：TerrainMap.generate() → 填 Heightfield(add_span)
        → surface.extract() 出网格 → render
        → baker.bake() 出 NavGraph → 运行时 A*

运行时：单位查询 ground_z / surface_z / solid_at（Numba 热路径）
       爆炸事件 → heightfield.remove_span()
               → tiles.mark_dirty()
               → tiles.rebuild_dirty(后台线程)
               → 双缓冲热交换（主线程 Tick 间隙，AGENTS.md 3.3）
```

## 六、决策记录

| 决策 | 内容 |
|------|------|
| 2026-08-10 | 表面提取**加载期统一做**，运行时**按 tile 脏处理** |
| 2026-08-10 | 自研高度场+span（NumPy/Numba），不绑 C++：Python 3.13 装不上 PyRecastDetour 的 pyd（最高 Py310），且绑库不支持局部更新 |
| 2026-08-10 | recast 官方 / recastlib / PyRecastDetour 仅作**算法与 API 蓝本**，不入运行时依赖 |

## 七、已知问题（待办）

### 7.1 隧道 span 无「上下段夹空隙」（2026-08-10 发现，**2026-08-11 已修复**）
> 状态：**已修复**（2026-08-11）。修复后隧道格 `span_count == 2`（隧道底 + 隧道顶），`ground_z == tunnel_alt`（-6.0），空隙可通行。下方保留根因与验证记录供追溯。

**现象（修复前）**：隧道格 `Heightfield.span_count == 1`（预期 ≥2：隧道底 + 隧道顶），`ground_z` 为 `tunnel_alt + 2`（-4.0）而非 `tunnel_alt`（-6.0）。

**根因（修复前）**：`TerrainMap.generate()` 先填隧道底段 `[-50, tunnel_alt)`（MAT_TERRAIN），再填隧道顶板段 `[tunnel_alt, tunnel_alt + 2)`（MAT_TUNNEL）。两段**相邻无空隙**，`Heightfield.add_span` 的合并规则（`s[0] <= merged[-1][1]`，半开区间相邻即合并）把它们并成单段 `[-50, tunnel_alt + 2)`，`mat` 取后加入者 MAT_TUNNEL。

**影响（修复前）**：
- `solid_at(gx, gy, z)` 在隧道空间（`z ∈ (tunnel_alt, tunnel_alt+2)`）返回 True → 单位无法穿过隧道（设计意图是「上下段夹空隙」通行）
- 隧道格 `ground_z` 与 `continuous_ground`（渲染，= `tunnel_alt`）不一致（差 2m）
- `surface.py` 地面材质取 `spans[col, 0, 3]` → 隧道格地面为 MAT_TUNNEL（暗岩）

**修复（2026-08-11）**：`map.py:generate()` 隧道顶板起点抬高到 `tunnel_alt + 2`，即顶板段 `[tunnel_alt + 2, tunnel_alt + 4)`，留出 2m 空隙。修复后结构：

- 底段 `[-50, tunnel_alt)`（MAT_TERRAIN）+ 空隙 `[tunnel_alt, tunnel_alt + 2)` + 顶段 `[tunnel_alt + 2, tunnel_alt + 4)`（MAT_TUNNEL）
- 隧道格 `span_count == 2`，`ground_z == tunnel_alt`（-6.0）
- `solid_at` 在空隙返回 False → 单位可穿过隧道 ✅
- 隧道顶（layer 1）作为结构层由 `surface.py` 渲染（暗岩顶面 + 悬空侧面）🆕

**验证（2026-08-11）**：
1. `surface.py` 渲染：地面层使用 `continuous_ground`（= `tunnel_alt`），修复前后一致；隧道顶结构层首次渲染 ✅
2. `navmesh/baker.py` 可行走掩码：隧道格 `ground_z` -4.0 → -6.0；walkable 判定不变；climb 约束（|Δz| > max_climb=2）隧道区与周围断开，修复前后一致 ✅
3. 全量 pytest：95 passed（含新增 `test_tunnel_gap_not_merged`）；GUI 冒烟 `GUI_SMOKE_OK` ✅

（同步记录于 `/memories/repo/dev-progress.md`）

### 7.2 SSFA portal edges 在真实烘焙下常为空（2026-08-11 记录）

> 状态：**设计约束已明确**，非 bug。SSFA 算法已实现并有走廊 NavGraph 专项测试（`tests/test_pathfinding.py`）。

**现象**：真实 Heightfield 烘焙（`Baker.bake()`）几乎总是产生 **1 个多边形**，`NavGraph.portal_edges` 为空 → `Pathfinder.find_path` 走退化分支（形心路径 + `_simplify_path` 简化），SSFA 的漏斗收紧实际不生效。

**根因**：
- `_build_walkable_mask()` 只烘焙 ground 层（桥/隧道不参与导航）
- 连通域标记 `_label_regions()` 把「一个连通地面」标为 1 个区域 → 1 个多边形
- `_extract_region_polygons()` 用 `unary_union` + `simplify`，**不做凸分解** → 凹多边形也不拆分成相邻凸多边形
- 相邻多边形共享边（portal）只存在于「凸分解后的相邻凸片」之间，当前管线不产出

**影响**：AGENTS.md 3.3 的「50~200 多边形 + portal 漏斗」设计在**当前 bake 管线**下无法充分发挥；寻路退化为形心路径 + 简化（功能可用，非最优）。

**未来方向**（不阻塞 Phase 3 收尾）：
- `_extract_region_polygons` 增加凸分解（如 shapely 的 triangulate / Hertel-Mehlhorn 近似），把大区域拆成相邻凸多边形 → portal_edges 自然产生 → SSFA 生效
- 桥/隧道加入导航层（walkable mask 按图层烘焙）
- 届时 SSFA 无需改动，只需喂入带 portal 的图

**SSFA 死循环教训**（2026-08-11，重要）：
- 「闭合后不前进重试」的 SSFA 变体在 `left==right==apex`（三态全等）时会死循环 + 无限 append（内存爆炸 ~2G）
- 修复 = 每条 portal 只消费一次（单调推进）+ `closed_left/right` 标记 + `max_iter` 保险丝

