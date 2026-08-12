# 导航网格重构：按可走性区域生长 + 三角剖分

> 状态：**已完成**（2026-08-12）
> 替代：`docs/surface-mesh-unification.md`（GreedyMesh 方案）
> 参考：Recast Navigation 管线（`Reference/navi.md`）

## 〇、一句话

> **遍历所有 span 顶面提取可行走格 → BFS 区域生长（逐对 z 比较 + mat 硬约束）→ shapely 轮廓 → Ear Clipping 三角剖分 → 三角形 = 导航节点，共享边 = portal。**

---

## 一、动机：为什么推翻旧方案

### 1.1 旧方案（GreedyMesh / `surface_mesh.py` 现状）

```
_scan_groups: 按 (layer_z, mat) 精确分组 → GreedyMesh 矩形合并
```

| 场景 | 表现 | 问题 |
|------|------|------|
| 纯斜坡 1000×1000 | 每格 z 不同 → 每一步长一个 group → 100 万个面片 | **OOM/寻路不可用** |
| z-bucket 2m + 斜坡 | ~8200 个面片 | 仍然太多，且 bucket 边界是人工切割 |
| 丘陵地形 | 大量碎面片 | 面片边界与地形语义无关 |

### 1.2 核心矛盾

> **分组依据是 "z 值是否相等"，而非 "能否从一个格走到另一个格"。**

### 1.3 新方案核心思想

> **分组依据改为 "相邻格是否可步行走到"（区域生长），面片边界 = 真正的导航边界（悬崖/陡坡/材质切换/水体）。**

---

## 二、管线总览

```
Heightfield spans (W×H 列 × ≤MAX_SPANS 段)
       │
       ▼  ① 可行走格点提取（遍历所有列所有 span 顶面）
CompactWalkableCells
  ┌─ 每格一个或多个 WalkableCell: (gx, gy, z_layer, z_world, mat)
  └─ 筛除：非顶面、坡度超标、面积过小、悬空
       │
       ▼  ② 区域生长（BFS，逐对 |Δz|<max_climb ∧ mat 相同 ∧ 邻域相邻）
Regions
  ┌─ list[set[cell_idx]]
  └─ 同材质、z 邻近、8 邻域连通 → 同一区域
       │
       ▼  ③ 轮廓提取（每区域格方块 → shapely unary_union → simplify）
Polygons
  ┌─ 简单多边形（可能带洞）
  └─ MultiPolygon → 拆为多个独立多边形
       │
       ▼  ④ 三角剖分（Ear Clipping + 桥接法处理洞）
Triangles
  ┌─ 每个三角形天然凸 → 两点直线在内部
  └─ 三角形 = 导航节点
       │
       ▼  ⑤ 邻接图 + portal 边（扫描线算法 O(n log n)）
NavGraph
  ┌─ 共享边长度 > cs*0.5 → portal
  └─ portal_edges: {tri_id: [(p1, p2), ...]}
       │
       ▼  ⑥ 跨层链接（桥头坡道 / 楼梯 / 隧道口）
  ┌─ 不同 mat、XY 重叠、|z|<max_climb → 垂直 portal（点 portal）
  └─ 寻路时：A* 跨面片 → funnel 沿 portal 收紧 → 路标点 → 高度场查 z
```

---

## 三、各步详细设计

### ③.① 可行走格点提取

#### 3.1.1 数据模型

```python
@dataclass
class WalkableCell:
    gx: int            # 格坐标 x
    gy: int            # 格坐标 y
    z_layer: int       # span 顶面层索引（uint16 编码）
    z_world: float     # 世界高度 (m)
    mat: int           # 材质 MAT_TERRAIN/BRIDGE/BUILDING/TUNNEL
    col: int           # 列索引 = gx + gy * width（冗余，加速）
```

**关键**：同一 (gx, gy) 列上可能存在多个 WalkableCell——例如地面 (z=2m, TERRAIN) 和上方桥面 (z=12m, BRIDGE)。各自独立参与区域生长。

#### 3.1.2 提取算法

```python
def extract_walkable_cells(hf: Heightfield, cfg: WalkableConfig) -> list[WalkableCell]:
    """遍历所有列的所有 span，提取可行走格点。"""
    cells = []
    for gy in range(hf.height):
        for gx in range(hf.width):
            col = hf._col(gx, gy)
            for si in range(int(hf.count[col])):
                # ① 必须是暴露表面（顶面）
                if not _is_top_surface(hf, col, si):
                    continue
                # ② 坡度可接受
                if _slope_at(hf, gx, gy, si) > cfg.max_slope_deg:
                    continue
                # ③ 面积够大（孤立单格不考虑）
                # ④ 非悬空（下方有支撑）
                if _is_hanging(hf, gx, gy, si):
                    continue

                zmax = int(hf.spans[col, si, _ZMAX])
                mat = int(hf.spans[col, si, _MAT])
                cells.append(WalkableCell(
                    gx=gx, gy=gy,
                    z_layer=zmax,
                    z_world=hf._to_z(zmax),
                    mat=mat, col=col,
                ))
    return cells
```

#### 3.1.3 顶面判定

一个 span 是「暴露表面」⇔ 上方无其他 span 紧贴覆盖：

```python
def _is_top_surface(hf, col, si):
    """si 段上方是否有紧贴的 span。"""
    n = hf.count[col]
    if si == n - 1:
        return True  # 最后一层 → 一定是顶面
    next_zmin = hf.spans[col, si + 1, _ZMIN]
    this_zmax = hf.spans[col, si, _ZMAX]
    return (next_zmin - this_zmax) >= GAP_THRESHOLD  # ≥ 1 层间隙
```

效果：
- 地面 si=0，上方 12m 处有桥 → 间隙大 → 地面和桥面都是顶面 ✅
- 地面 si=0，上方紧贴建筑底层 → 间隙小 → 地面被覆盖，不可走 ✅

#### 3.1.4 坡度判定

用四邻域高度差计算局部坡度，取最大值：

```python
def _slope_at(hf, gx, gy, si):
    z_center = hf._to_z(hf.spans[hf._col(gx, gy), si, _ZMAX])
    mat = hf.spans[hf._col(gx, gy), si, _MAT]

    max_grad = 0.0
    for nx, ny in [(gx-1,gy), (gx+1,gy), (gx,gy-1), (gx,gy+1)]:
        if not (0 <= nx < hf.width and 0 <= ny < hf.height):
            continue
        n_col = hf._col(nx, ny)
        # 找邻居同 mat 同层 span 的高度
        n_z = _find_neighbor_z(hf, n_col, mat, z_center)
        if n_z is None:
            continue
        grad = abs(z_center - n_z) / hf.cs
        max_grad = max(max_grad, grad)

    return math.degrees(math.atan(max_grad))
```

#### 3.1.5 悬空判定

```python
def _is_hanging(hf, gx, gy, si):
    """面片下方是否有支撑。TERRAIN 紧邻下方 span、人工结构自身可站立。"""
    mat = hf.spans[hf._col(gx, gy), si, _MAT]
    if mat != MAT_TERRAIN:
        return False  # BRIDGE/BUILDING/TUNNEL 本身可站立
    # si == 0（地面）→ 有支撑
    if si == 0:
        return False
    # si > 0：检查是否紧贴下方 span
    this_zmin = hf.spans[hf._col(gx, gy), si, _ZMIN]
    prev_zmax = hf.spans[hf._col(gx, gy), si - 1, _ZMAX]
    return (this_zmin - prev_zmax) > GAP_THRESHOLD
```

---

### ③.② 区域生长

#### 3.2.1 核心算法

```python
def grow_regions(cells: list[WalkableCell], max_climb: float) -> list[set[int]]:
    """BFS 区域生长，逐对比较 z + mat 硬约束。"""
    neighbors = _build_neighbor_map(cells)  # cell_idx → [neighbor_idx, ...]
    visited = set()
    regions = []

    for seed_idx in range(len(cells)):
        if seed_idx in visited:
            continue

        region = set()
        queue = deque([seed_idx])
        visited.add(seed_idx)

        while queue:
            cur_idx = queue.popleft()
            region.add(cur_idx)
            cur = cells[cur_idx]

            for nb_idx in neighbors[cur_idx]:
                if nb_idx in visited:
                    continue
                nb = cells[nb_idx]

                # ── 合并条件 ──
                if nb.mat != cur.mat:               # mat 不同 → 不合并
                    continue
                if abs(nb.z_world - cur.z_world) > max_climb:  # z 差过大
                    continue

                visited.add(nb_idx)
                queue.append(nb_idx)

        if len(region) >= MIN_REGION_CELLS:
            regions.append(region)

    return regions
```

#### 3.2.2 邻域图构建

```python
def _build_neighbor_map(cells):
    # 按 (gx, gy, mat) 索引：同一格可能有多个不同 mat 的 cell
    cell_map: dict[tuple[int,int,int], list[int]] = {}
    for idx, c in enumerate(cells):
        key = (c.gx, c.gy, c.mat)
        cell_map.setdefault(key, []).append(idx)

    neighbors = [[] for _ in cells]
    for idx, c in enumerate(cells):
        for nx, ny in [(c.gx-1,c.gy),(c.gx+1,c.gy),(c.gx,c.gy-1),(c.gx,c.gy+1)]:
            for nb_idx in cell_map.get((nx, ny, c.mat), []):
                neighbors[idx].append(nb_idx)
    return neighbors
```

#### 3.2.3 与旧 Bake 的关键区别

| | 旧 Baker `_label_regions` | 新 `grow_regions` |
|---|---|---|
| 数据源 | 仅 `spans[col, 0]`（地面层） | **所有 span 顶面** |
| z 比较 | 与种子点 `base_z` 比（绝对） | **逐对相邻格比**（相对） |
| mat 约束 | 无 | **硬约束**（不同 mat 绝不合并） |
| 纯斜坡 1000×1000 | BFS 约 400 格后断连 → ~15 个区域 | **1 个区域** |
| 桥面 z=12m | 不可达（只看地面层） | **独立区域** |
| 地面+桥上坡道 | 不连接（z 差 > max_climb） | 两区域 + 跨层垂直 portal |

---

### ③.③ 轮廓提取

```python
def region_to_polygon(cells_in_region, hf) -> Polygon | MultiPolygon | None:
    """区域 → shapely 多边形（可能带洞）。"""
    from shapely.ops import unary_union
    from shapely.geometry import box

    cs = hf.cs
    squares = [
        box(c.gx * cs, c.gy * cs, (c.gx + 1) * cs, (c.gy + 1) * cs)
        for c in cells_in_region
    ]
    merged = unary_union(squares)
    if merged.is_empty:
        return None

    return merged.simplify(cs * 0.5, preserve_topology=True)
```

**处理 MultiPolygon**：

```python
def polygon_to_list(poly):
    """统一接口：Polygon → [Polygon]；MultiPolygon → 拆为列表。"""
    if poly.geom_type == "MultiPolygon":
        return [g for g in poly.geoms if not g.is_empty]
    elif poly.geom_type == "Polygon":
        return [poly] if not poly.is_empty else []
    return []
```

---

### ③.④ 三角剖分

#### 3.4.1 为什么是三角剖分，不是凸分解

| 方案 | 节点数 | 面片内寻路 | 实现 |
|------|:---:|------|:---:|
| 凸分解 (Hertel-Mehlhorn) | 最多（每凸片一个节点） | 直线 | 高 |
| **三角剖分 (Ear Clipping)** | **中等** | **直线（三角形天然凸）** | **低** |
| 不分解 | 最少 | 凹处直线穿出，需额外处理 | 中 |

三角形天然凸 + 算法简单 + 无外部依赖 → 选择三角剖分。

#### 3.4.2 Ear Clipping 核心

```python
def triangulate_polygon(polygon: Polygon) -> list[tuple[int,int,int]]:
    """Ear Clipping 三角剖分，支持带洞多边形。

    Returns:
        三角形索引列表，每个元素为 (i, j, k) 指向顶点数组。
    """
    # 1. 提取外环 + 内环
    outer = list(polygon.exterior.coords)[:-1]  # 首尾去重
    holes = [list(h.coords)[:-1] for h in polygon.interiors]

    # 2. 桥接处理洞（缝合到外环）
    vertices = list(outer)
    for hole in holes:
        vertices = _bridge_hole(vertices, hole)

    # 3. 标准 ear clipping
    triangles = []
    remaining = list(range(len(vertices)))

    while len(remaining) > 3:
        for i in range(len(remaining)):
            prev, cur, next = _get_triple(remaining, i)
            if _is_ear(vertices, prev, cur, next, remaining):
                triangles.append((prev, cur, next))
                remaining.pop(i)
                break

    triangles.append(tuple(remaining[:3]))
    return triangles
```

#### 3.4.3 桥接法处理洞

```
  外环（逆时针）
      内部洞（顺时针）    ← 洞边界

桥接：
  ① 找洞的最右顶点 H
  ② 在外环上找可见顶点 V（HV 线段不穿越任何边界）
  ③ 插入桥边 H→V 和 V→H（正反各一次）
  ④ 洞顶点按桥边缝合到外环序列中
```

缝合后，整个带洞多边形变成单一简单多边形边界，ear clipping 直接适用。

```python
def _bridge_hole(outer, hole):
    """把洞缝合到外环上。"""
    # 1. 找洞的最右顶点
    h_idx = max(range(len(hole)), key=lambda i: hole[i][0])
    h_pt = hole[h_idx]

    # 2. 在外环上找可见顶点
    v_idx = _find_visible_vertex(outer, h_pt, hole)

    # 3. 缝合
    result = outer[:v_idx + 1]
    result.extend(hole[h_idx:] + hole[:h_idx + 1])  # 洞顺时针走
    result.extend(outer[v_idx:])
    return result
```

#### 3.4.4 Ear 判定

一个顶点是 ear ⇔ 其前驱-自身-后继构成的三角形满足：
1. 三角形内部没有其他顶点（点-三角形包含测试）
2. 三角形的"内角" < 180°（不是 reflex vertex）

---

### ③.⑤ 邻接图 + portal 边

#### 3.5.1 数据模型

```python
@dataclass
class NavTriangle:
    tri_id: int
    vertices: np.ndarray        # (3, 2) float32, 世界坐标
    centroid: tuple[float, float]
    z_world: float              # 代表高度（取三顶点均值或中位数）
    mat: int

@dataclass
class NavGraph:
    triangles: list[NavTriangle]
    adjacency: dict[int, list[int]]   # {tri_id: [neighbor_id]}
    portal_edges: dict[int, list[tuple[vec2, vec2]]]
```

#### 3.5.2 邻接判定

两个三角形邻接 ⇔ 共享一条边（两个顶点都相同，容差 `cs * 0.1`）。

```python
def build_adjacency(triangles):
    # 边 → 三角形映射
    edge_to_tri: dict[tuple[vec2, vec2], list[int]] = {}
    for tid, tri in enumerate(triangles):
        for e in edges_of(tri):
            key = _canonical_edge(e)
            edge_to_tri.setdefault(key, []).append(tid)

    # 共享边的三角形对
    adjacency = {}
    portals = {}
    for e, tids in edge_to_tri.items():
        if len(tids) == 2:
            a, b = tids
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
            portals.setdefault(a, []).append(e)
            portals.setdefault(b, []).append((e[1], e[0]))  # 反向

    return adjacency, portals
```

> 三角形数通常在 200-10000（视地图复杂度），O(n) 边映射足够。

#### 3.5.3 面片定位

起点/终点所在三角形：

```python
def locate_triangle(x, y, triangles):
    """点定位：哪个三角形包含 (x,y)。"""
    from shapely.geometry import Point, Polygon
    p = Point(x, y)
    for tri in triangles:
        if Polygon(tri.vertices).contains(p):
            return tri.tri_id
    # fallback：最近三角形
    return min(triangles, key=lambda t: 
        (t.centroid[0]-x)**2 + (t.centroid[1]-y)**2).tri_id
```

---

### ③.⑥ 跨层链接（垂直 portal）

#### 3.6.1 场景

```
桥面（z=12m, BRIDGE）            建筑屋顶（z=20m, BUILDING）
  ════════════════                  ┌──────┐
        ‖ 坡道 / 桥头                    │ 楼梯 │
  地面（z=2m, TERRAIN）            地面（z=2m）
```

#### 3.6.2 检测

```python
def link_cross_layer(triangles_by_mat, max_climb):
    """跨 mat 垂直 portal 检测。"""
    vertical_portals = []

    pairs = [
        (MAT_TERRAIN, MAT_BRIDGE),
        (MAT_TERRAIN, MAT_BUILDING),
        (MAT_BRIDGE, MAT_TUNNEL),
    ]
    for mat_a, mat_b in pairs:
        tris_a = triangles_by_mat.get(mat_a, [])
        tris_b = triangles_by_mat.get(mat_b, [])
        # 空间哈希加速
        spatial_a = SpatialHash(tris_a, cell_size=50.0)
        for tb in tris_b:
            for ta in spatial_a.query_near(tb):
                # XY 投影三角形相交？
                if not xy_overlap(ta, tb):
                    continue
                if abs(ta.z_world - tb.z_world) > max_climb:
                    continue
                # 记录为垂直 portal（点连接）
                vertical_portals.append((ta.tri_id, tb.tri_id))
    return vertical_portals
```

#### 3.6.3 在邻接图中的表示

垂直 portal 加入邻接表，`portal_edges` 中标记为特殊值（如 `((x,y), (x,y))`，即退化为点）。funnel 算法沿垂直 portal 收紧时，将 portal 视为单点处理。

---

## 四、寻路集成

### 4.1 已有基础设施（可复用）

| 模块 | 文件 | 状态 |
|------|------|:---:|
| A\* 面片搜索 | `movement/nav.py::patch_astar` | ✅ 已实现 |
| 形心路径 | `movement/nav.py::patch_centroid_path` | ✅ 已实现 |
| 漏斗算法 | 需实现 | ⚠️ 待写 |
| 路径服务 | `movement/pathfinding.py::Pathfinder` | ✅ 已实现 |
| 坐标转换 | `coordinates/grid.py` | ✅ 已实现 |

### 4.2 需新增：SSFA Funnel

```python
def funnel_path(
    triangulated_path: list[int],
    start: tuple[float, float],
    end: tuple[float, float],
    portals: dict[int, list[tuple[vec2, vec2]]],
) -> list[tuple[float, float]]:
    """SSFA (Simple Stupid Funnel Algorithm)。
    
    Args:
        triangulated_path: A* 返回的三角形 ID 序列。
        start, end: 起终点世界坐标。
        portals: portal_edges 字典。
    
    Returns:
        收紧后的路标点列表。
    """
    # 门户边序列
    edges = []
    for i in range(len(triangulated_path) - 1):
        a, b = triangulated_path[i], triangulated_path[i+1]
        for p1, p2 in portals.get(a, []):
            # 找与 b 共享的 portal 边
            if any((p2, p1) == q or (p1, p2) == q for q in portals.get(b, [])):
                edges.append((p1, p2))
                break

    # SSFA 主循环
    # 维护左右漏斗边，逐步收紧
    ...
```

### 4.3 高度查询

路标点均为 2D 坐标，在 `movement/integrate.py` 中通过高度场查实际 z：

```python
# 路标点 → 3D 位置
for wx, wy in waypoints:
    gx, gy = world_to_grid(wx, wy, cs)
    wz = surface_z_at(hf, gx, gy)  # 从高度场查
```

---

## 五、规模分析

### 5.1 节点数估算

| 地图 | 格数 | 区域数 | 三角形数（估） | 加载耗时（估） |
|------|:---:|:---:|:---:|:---:|
| 200×200 | 4万 | 10-50 | 50-300 | ~150ms |
| 1000×1000 | 100万 | 30-200 | 200-1500 | ~1s |
| 4000×4000 | 1600万 | 100-500 | 500-4000 | ~10s |
| 10000×10000 | 1亿 | 200-2000 | 1000-15000 | ~60s |

> 加载仅在启动/读档时执行一次。存档时序列化 NavGraph 到 h5/msgpack。运行时只读。

### 5.2 内存估算

- 10000×10000 地图 → ~15000 个三角形
- 每个三角形：3 个顶点 (3×2×4B=24B) + 形心 (8B) + z+mat+id (12B) ≈ 44B
- 邻接表：~30000 条边 × 8B ≈ 240KB
- **总计 < 2MB**

### 5.3 运行时寻路

- A\* 在 15000 节点图上 → < 1ms
- Funnel 沿 ~20 个 portal 收紧 → < 0.1ms
- 路标点 z 查询 → < 0.01ms

---

## 六、复杂拓扑处理

### 6.1 桥下地面上的单位

```
桥面 z=12m（BRIDGE 区域 → triangles_upper）
地面 z=2m  （TERRAIN 区域 → triangles_lower）
         ‖
    桥墩（实心 span，非顶面 → 不产生 WalkableCell）
```

- 桥下地面仍是可行走的（没有被覆盖）
- 桥上单位：z≈12m，定位到 triangles_upper 中的一个三角形
- 桥下单位：z≈2m，定位到 triangles_lower 中的一个三角形
- 两者不碰撞（不同 z、不同三角形集合）

### 6.2 隧道

```
隧道顶 z=8m  （TUNNEL span → 顶面可见 + 非悬空 → WalkableCell）
隧道底 z=0m  （TERRAIN span → 隧道顶下方 → 非顶面 → 不提取）
         ‖
    隧道口（z=0m→8m 过渡，地面区域与隧道顶区域在口部 XY 重叠且 z 差 < max_climb）
```

隧道口 = 跨层垂直 portal。

### 6.3 建筑群

```
建筑 A（BUILDING）：屋顶 z=15m，各层楼板
建筑 B（BUILDING）：屋顶 z=20m
地面（TERRAIN）：z=2m

每个建筑的每个暴露楼板 = 独立区域（如果被上层楼板覆盖则不提取）
建筑间如果相邻且 z 差 < max_climb → 水平 portal
```

---

## 七、与现有代码的关系

| 文件 | 操作 |
|------|------|
| `terrain/surface_mesh.py` | **重写** — 新 `NavMeshBuilder` 替代 `SurfaceMeshBuilder` |
| `terrain/surface_mesh.py::SurfaceMesh` | **保留结构，调整字段** — patches→triangles, z→per-triangle |
| `navmesh/baker.py` | **删除** — 已标记废弃，所有功能由新管线接管 |
| `movement/nav.py` | **微调** — 适配 NavTriangle 接口 |
| `movement/pathfinding.py` | **微调** — 适配新 NavGraph + 集成 funnel |
| `movement/integrate.py` | **不变** |
| `terrain/heightfield.py` | **微调** — 可能需要暴露辅助查询方法 |
| `terrain/surface.py` | **不变** — 渲染管线独立 |
| `tests/test_pathfinding.py` 等 | **更新** — 适配新接口 |
| `docs/surface-mesh-unification.md` | **归档** — 被本文档替代 |

---

## 八、实施顺序

| 步骤 | 内容 | 预估工作量 |
|:---:|------|:---:|
| 1 | 实现 `WalkableCell` 数据模型 + `extract_walkable_cells` + 顶面/坡度/悬空判定 | 中 |
| 2 | 实现 `grow_regions`（BFS 区域生长 + mat 硬约束 + 邻域图构建） | 小 |
| 3 | 实现 `region_to_polygon`（已有 shapely 管线可复用） | 小 |
| 4 | 实现 `triangulate_polygon`（Ear Clipping + 桥接法处理洞） | 中 |
| 5 | 实现 `build_navgraph`（三角形邻接图 + portal_edges） | 小 |
| 6 | 实现跨层链接（垂直 portal 检测） | 小 |
| 7 | 实现 SSFA Funnel 算法 | 中 |
| 8 | 集成到 `Pathfinder` / `nav.py` | 小 |
| 9 | 更新测试（斜坡/桥/建筑/洞） | 中 |
| 10 | 删除 `navmesh/baker.py` 废弃代码 | 小 |
| 11 | 性能测试（10000×10000 地图） | 中 |

---

## 九、与 Recast Navigation 的对应

| Recast 模块 | 我们的对应 | 说明 |
|---|---|---|
| `rcHeightfield` | `Heightfield` | 已有 |
| `rcCompactHeightfield` | `extract_walkable_cells` | 第①步 |
| `rcBuildRegions` | `grow_regions` | 第②步，我们用 BFS+mat 替代分水岭 |
| `rcBuildContours` | `region_to_polygon` | 第③步，用 shapely |
| `rcBuildPolyMesh` | 跳过 | 我们不走凸分解 |
| `rcBuildPolyMeshDetail` | `triangulate_polygon` | 第④步，Ear Clipping |
| 邻接图 | `build_navgraph` | 第⑤步，等效 Detour 的 tile 连接 |
| 跨层 | `link_cross_layer` | 第⑥步，Recast 不做（纯 3D 游戏无此需求） |

---

## 十、未解决的开放问题

1. **动态障碍物**（被摧毁的建筑/桥梁）→ 运行时移除三角形 + 局部重新三角剖分？还是预留动态 tile 机制？
2. **坡度判定的精度**：用 `surface.py` 平滑网格的双线性插值坡度替代离散格梯度？
3. **垂直 portal 的 funnel 行为**：点 portal 在 SSFA 中的精确处理方式？
4. **加载耗时优化**：10000×10000 地图 60s 是否可接受？是否需要分块并行提取？

---

## 附录 A：关键参数（推荐默认值）

| 参数 | 默认值 | 含义 |
|------|:---:|------|
| `max_slope_deg` | 30.0 | 最大可行走坡度（度） |
| `max_climb` | 2.0 | 区域生长最大 z 差（m） |
| `min_region_cells` | 8 | 最小区域格数 |
| `gap_threshold` | 1 | 顶面检测的间隙阈值（层数） |
| `cell_size` | 10.0 | 栅格尺寸（m） |
| `simplify_tolerance` | cs × 0.5 | 轮廓简化容差 |
| `portal_min_length` | cs × 0.5 | portal 边最小长度 |
