# WorldMesh 3D 世界模型设计

> 创建日期：2026-08-12
> 状态：**Phase A + B + D 已实施（2026-08-12）**；Phase C 缓办（NavGraph 依赖 spans，与 WorldMesh 结果类似）
>
> Phase A 落地（`terrain/world_mesh.py` + ArmorPool.mesh_id + `integrate_armor_movement_3d`）：
> - WorldPatch 3D 化 + STRtree 空间索引 + hint_z 点定位 + ray_plane_z
> - §5.5 面片穿越方案已更新为「局部性 + hint_z 直接定位」（弃邻接三阶段）
>
> Phase B 落地（`terrain/dual_contour.py`，DualContouring 简化版）：
> - 顶面/底面/侧面完整 3D 提取 + 逐格地面平滑 + 法线
> - 地面顶面逐格（GreedyMesh 合并丢内部顶点）；结构/底面/侧面合并
> - 地面不生成侧面（span 从世界底延伸 → 边界 52m 巨墙；地面由平滑网格+Heightfield 兜底）
>
> Phase D 落地（`terrain/surface.py::WorldMeshRenderer`）：
> - 渲染数据源统一为 WorldMesh（DualContour 完整 3D），不再从 Heightfield 独立提取
> - 地面平滑细分 + 结构（顶/底/侧）patch 三角化 → 桥侧墙/桥底进入渲染
>
> ⚠️ **执行条款（2026-11-03）**：本文件 §1「放弃 2.5D 高度图思维」的强制约束见
> **`docs/worldmesh-ground-patch-policy.md`**——地面必须保留显式 `WorldPatch`，
> 禁止「地面不入面片集合 / 直接查高度场」的退化方案（冲突时以该文件为准）。

## 一、动机

当前架构的 world model（世界模型）是 **Heightfield column-span + continuous_ground (H,W) 扁数组**。这一组合有以下根本局限：

| 局限 | 具体表现 |
|------|----------|
| **无拓扑** | `continuous_ground` 是 (H,W) 死数组，无法表达洞（弹坑）、环（废墟）、多层 |
| **无斜面** | 所有表面隐式 `normal = (0,0,1)`，无法表达斜桥、坡道、洞壁、天花板 |
| **多表面二义性** | 桥面和桥下地面 xy 重叠，数组只能存一个 z 值 |
| **渲染与逻辑割裂** | 渲染用 `SurfaceExtractor`（SUBDIV 细分），逻辑用 Heightfield 量化层 |
| **单位定位间接** | `(x,y)` → grid → Heightfield 采样 z → 再转换，不直接 |
| **无空中单位预留** | 当前所有单位假设在地面，无表面凭依/豁免机制 |

**结论：放弃 2.5D 高度图思维，全面转向 3D 面片集合作为世界模型。**

## 二、目标架构

```
                    ┌─────────────────────────────────┐
                    │       Heightfield column-span    │
                    │   (体素场：碰撞检测 · 视线判定)   │
                    │   保持 ch=1.0 量化，轴对齐柱     │
                    └──────────────┬──────────────────┘
                                   │ Dual Contouring 提取表面
                    ┌──────────────▼──────────────────┐
                    │         WorldMesh                │
                    │   (面片集合 = 世界模型)           │
                    │   · 物理 · 碰撞精算 · z 采样     │
                    │   · 渲染 · 视线                   │
                    │   面片: 3D 顶点 + 法线 + 材质     │
                    └──────────────┬──────────────────┘
                                   │ 烘焙
                    ┌──────────────▼──────────────────┐
                    │     NavGraph (三角形)             │
                    │   纯寻路加速 · 不存世界状态        │
                    └──────────────────────────────────┘
```

### 关键原则

1. **WorldMesh 是物理真实**：所有物理计算（碰撞精算、z 采样、弹道终点）走 WorldMesh
2. **Heightfield 是碰撞快查**：column-span 轴对齐柱用于粗筛（胶囊体 vs AABB 粗检）
3. **NavGraph 是寻路捷径**：三角形邻接图只用于 A* 搜索，不参与世界状态
4. **渲染直接从 WorldMesh 取**：面片顶点就是渲染顶点，无中间层

## 三、数据模型

### 3.1 WorldPatch（世界面片）

```python
@dataclass
class WorldPatch:
    """3D 世界面片 —— 世界模型的原子单元。

    Attributes:
        patch_id: 唯一 ID（在 WorldMesh 内索引）。
        vertices: (V, 3) float32  世界 3D 坐标 (x, y, z)，逆时针闭环。
        normal: (3,) float32  面片法线（单位向量，指向可站立表面外侧）。
        mat: 材质（MAT_TERRAIN / MAT_BRIDGE / MAT_BUILDING / MAT_TUNNEL）。
        walkable: 是否可行走（由坡度 + 悬空 + 材质判定）。
        area: 面积 (m²)。
        centroid: (3,) float32  形心世界坐标。
    """

    patch_id: int
    vertices: np.ndarray        # (V, 3) float32
    normal: np.ndarray          # (3,) float32
    mat: int
    walkable: bool = False
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
```

与旧 `SurfacePatch` 的核心差别：

| | SurfacePatch (旧) | WorldPatch (新) |
|---|---|---|
| 顶点维度 | `(V, 2)` + `z: float` | `(V, 3)` 真正 3D |
| 法线 | 隐式 `(0,0,1)` | 显式 `normal (3,)` |
| 面朝向 | 仅水平面 | **任意朝向**（斜面/墙壁/天花板） |
| z 采样 | `patch.z`（直接取） | `ray_plane_z(patch, x, y)` 竖直射线与面片平面求交 |

### 3.2 WorldMesh（面片集合 + 空间索引）

面片定位使用 **STRtree（R-tree）**：O(log N) 查询，内存与面片数成比例（与格数无关），
城市战 80% 覆盖率或 10000×20000 大地图均适用。

```python
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree

@dataclass
class WorldMesh:
    """3D 世界面片集合 + 空间索引 + 邻接图。

    Attributes:
        patches: 所有面片（按 patch_id 索引）。
        adjacency: {patch_id: [neighbor_id]}  共享边邻接。
        portal_edges: {patch_id: [(p1_3d, p2_3d)]}  门户边（3D 世界坐标）。
        walkable_patches: {patch_id}  可行走面片缓存。
        _polygons: [shapely.Polygon, ...]  面片 2D 轮廓（与 patches 同序）。
        _tree: STRtree 空间索引（懒构建）。
    """

    patches: list[WorldPatch]
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    portal_edges: dict[int, list[tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    walkable_patches: set[int] = field(default_factory=set)

    # 空间索引（内部）
    _polygons: Any | None = field(default=None, repr=False)
    _tree: Any | None = field(default=None, repr=False)

    def _build_index(self) -> None:
        """构建 STRtree 空间索引（首次查询时懒加载）。"""
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree

        self._polygons = [
            Polygon(p.vertices[:, :2]) for p in self.patches
        ]
        self._tree = STRtree(self._polygons)

    def locate_patch(
        self, x: float, y: float,
        mat: int | None = None,
        hint_z: float | None = None,
    ) -> int:
        """点定位：返回覆盖 (x,y) 的面片 ID。O(log N)。

        Args:
            x, y: 世界坐标。
            mat: 材质过滤（可选）。
            hint_z: 期望高度（可选）。当同 (x,y) 有多个面片时，
                比较各面片在 (x,y) 处的 ray_plane_z 值与 hint_z 的
                接近程度，返回最接近的面片（而非表面最高的）。
                用于桥下单位定位等场景。

        Returns:
            面片 ID，未找到返回 -1。
        """
        if self._tree is None:
            self._build_index()

        from shapely.geometry import Point
        pt = Point(x, y)
        candidates: list[int] = list(self._tree.query(pt))

        best_pid = -1
        best_dist = float("inf")
        for pid in candidates:
            p = self.patches[pid]
            if mat is not None and p.mat != mat:
                continue
            if not self._polygons[pid].contains(pt):
                continue
            z = self._ray_plane_z_impl(p, x, y)
            if hint_z is not None:
                d = abs(z - hint_z)
                if d < best_dist:
                    best_dist = d
                    best_pid = pid
            elif z > best_dist:  # 无 hint → 取表面最高的面片
                best_dist = z
                best_pid = pid
        return best_pid

    def ray_plane_z(self, patch_id: int, x: float, y: float) -> float:
        """竖直射线 (x,y,±∞) 与面片的交点 z。

        ray(t) = (x, y, t), plane: n·(p - v0) = 0
        → t = (n·v0 - n.x·x - n.y·y) / n.z
        """
        return self._ray_plane_z_impl(self.patches[patch_id], x, y)

    @staticmethod
    def _ray_plane_z_impl(patch: WorldPatch, x: float, y: float) -> float:
        n = patch.normal
        v0 = patch.vertices[0]
        if abs(n[2]) < 1e-9:
            return -1e9  # 竖直面，不可站立
        return (float(np.dot(n, v0)) - n[0] * x - n[1] * y) / n[2]
```

### 3.3 单位定位模型

**所有单位统一用 `(mesh_id, x, y, z)` 表示世界位置**，不区分地面/空中：

```python
# ArmorPool 扩展
armor_mesh_id = np.zeros(MAX_UNITS, dtype=np.int32)  # 所在表面面片 ID
                                                       # -1 = 空中（无凭依）
armor_pos_x   = np.zeros(MAX_UNITS, dtype=np.float32) # 世界 x
armor_pos_y   = np.zeros(MAX_UNITS, dtype=np.float32) # 世界 y
armor_pos_z   = np.zeros(MAX_UNITS, dtype=np.float32) # 世界 z（地面置为 NaN 触发自动查表）
```

| 单位类型 | `mesh_id` | z 来源 | 渲染行为 |
|----------|-----------|--------|----------|
| 地面装甲 | `patch_id` | `ray_plane_z(patch_id, x, y)` | 贴表面法线防止穿模 |
| 空中单位 | `-1` | 自由（飞行高度） | 无表面约束，主动控制朝向 |

**z 同步逻辑**：

```python
def sync_unit_z(armor_mesh_id, armor_pos):
    """每 tick：地面单位 z 从表面重算，空中单位保留自由 z。"""
    for i in range(n):
        mid = armor_mesh_id[i]
        if mid >= 0:
            armor_pos[i, 2] = world_mesh.ray_plane_z(mid, armor_pos[i, 0], armor_pos[i, 1])
        # mid < 0 → 空中单位，z 不处理
```

## 四、表面提取：Dual Contouring

### 4.1 选型理由

| 算法 | 适用性 | 问题 |
|------|--------|------|
| Marching Cubes | 3D 密度场 | 模糊棱角，不能保留 Heightfield 的轴对齐特征 |
| **Dual Contouring** | ✅ 从体素场提取表面 | 保留锐利特征（桥边、建筑棱角），自然输出面片+法线 |
| GreedyMesh (现有) | 2D 合并 | 只能处理同 z 层，无法处理斜面 |

### 4.2 简化 Dual Contouring（适配 column-span）

完整 Dual Contouring 需要解 QEF（Quadratic Error Function）定位每体素内的顶点。对于 column-span 的轴对齐柱体素，可简化为：

1. **体素格点**：`(gx, gy, z_layer)` 的 3D 格，每个格点存「是否实心」标记
2. **边交叉点**：格边的 `实心↔空心` 过渡处放置表面顶点
3. **面片生成**：每格最多 1~3 个面片（顶面、侧面、底面）
4. **顶点平滑**：边交叉点的 z 用 `continuous_ground` 双线性插值替代量化层 z

对于斜面/非轴对齐表面，则需要完整 QEF 或 Hermite 数据（法线估计）来定位顶点。

### 4.3 与 Heightfield 的关系

- Heightfield 保持 ch=1.0 量化，用于碰撞粗筛（胶囊体 vs AABB 的 broad phase）
- WorldMesh 提供碰撞精算（弹丸命中点、精确 z 坐标）
- 地形破坏：先更新 Heightfield span → 标记受影响区域 → 局部重建 WorldMesh 面片

## 五、单位定位

### 5.1 统一位置模型

```
所有单位：mesh_id (int32) + (x, y, z) (float32)
  mesh_id = -1  → 空中单位，z 自由，渲染自由朝向
  mesh_id >= 0  → 地面单位，z 从面片求交，渲染贴表面法线
```

### 5.2 z 同步（每 tick）

```python
def sync_ground_z(mesh_id, pos, world_mesh):
    """地面单位：z = ray_plane_z(mesh_id, x, y)；空中单位跳过。"""
    for i in range(len(mesh_id)):
        mid = mesh_id[i]
        if mid >= 0:
            pos[i, 2] = world_mesh.ray_plane_z(mid, pos[i, 0], pos[i, 1])
```

### 5.3 移动积分（地面单位）

```
每个 ground tick:
  1. 查 NavGraph A* → 三角形序列 → funnel → 路标点 (x,y)
  2. 在面片内按速度推进 (x, y)，portal 穿越时更新 mesh_id
  3. sync_ground_z()
```

### 5.4 移动积分（空中单位）

```
每个 air tick:
  1. 按航向/速度自由更新 (x, y, z)
  2. mesh_id = -1，无表面约束
```

### 5.5 面片穿越（mesh_id 更新）

> **2026-08-12 设计更新（与用户讨论后拍板）**：弃「当前优先 → 邻接穿越 → 全局 fallback」三阶段，
> 改为 **hint_z 直接定位 + 局部性检查**。理由：
> 1. **歧义已由 hint_z 解决**——`locate_patch(x, y, hint_z=当前z)` 选 ray_plane_z 与当前 z 最接近的
>    面片，桥下单位自然选地面面片，无需邻接穿越。
> 2. **局部性消解 99% 查询**——单位移动连续，每 tick 先查「仍在当前面片？」（Numba point_in_poly），
>    命中则保持 mesh_id 并同步 z；只有越出（~1%）才回退 STRtree。
> 3. **生产规模内存更优**——不建全量 Grid Hash（4000×4000 的 cell_start 即 ~13MB），
>    STRtree 内存 O(面片数)，与格数无关。
> 4. **邻接表职责收窄**——仅服务 A*/SSFA 寻路（纯 Python），不进移动热路径。

#### 问题：同 (x,y) 多面片时表面高度排序不可靠

```
单位在桥面下方、地面上方：
  桥面 patch_B: ray_plane_z(B, x, y) = 14.0, mat=BRIDGE
  地面 patch_A: ray_plane_z(A, x, y) = 5.3,  mat=TERRAIN
  → 按表面最高返回桥面 ❌ (单位实际在地面)
```

**正解：`locate_patch` 带 `hint_z`（单位当前 z），选 ray_plane_z 与 hint_z 最接近的面片。**

#### 方案：局部性检查 + hint_z 直接定位

```
每个 ground tick（Numba 热路径）：
  1. 推进 (x, y) 沿路标点
  2. 若 mesh_id[i] >= 0 且 (x,y) 仍在当前面片内（point_in_poly）
     → mesh_id 不变，z = ray_plane_z(mesh_id, x, y)    [99%+ 命中]
  3. 越出 → mesh_id[i] = -1
每个 ground tick（Python 侧，仅对 -1 的单位）：
  4. mesh_id[i] = locate_patch(x, y, hint_z=pos.z)     [STRtree O(log N)]
```

```python
# Numba 内核（integrate_armor_movement_3d，展平数组）
@njit(parallel=True)
def integrate_armor_movement_3d(pos, yaw, active, wp, wp_count, wp_idx, speed, dt,
                                mesh_id, patch_v, patch_vstart, patch_vcnt, patch_norm):
    for i in prange(n):
        # ... 推进 (x,y) ...
        mid = mesh_id[i]
        if mid >= 0:
            if _point_in_poly_flat(pos[i,0], pos[i,1], patch_v, patch_vstart, patch_vcnt, mid):
                pos[i,2] = _ray_plane_z_flat(..., mid, pos[i,0], pos[i,1])  # 同步 z
            else:
                mesh_id[i] = -1  # 标记待 Python fallback

# Python fallback（非热路径）
def resolve_mesh_ids(pos, mesh_id, world_mesh):
    for i in np.flatnonzero(mesh_id == -1):
        pid = world_mesh.locate_patch(pos[i,0], pos[i,1], hint_z=pos[i,2])
        mesh_id[i] = pid
        if pid >= 0:
            pos[i,2] = world_mesh.ray_plane_z(pid, pos[i,0], pos[i,1])
```

#### 悬挂触点跨面片

四触点独立使用上述逻辑，各带自己的 `mesh_id_contact[j]`：

```python
# 每个触点独立追踪面片（局部性 + hint_z 同 §5.5）
for j in range(4):
    cid = update_mesh_id(contact_mesh_id[i,j], cx, cy, world_mesh)
    contact_mesh_id[i,j] = cid
    contact_z[i,j] = world_mesh.ray_plane_z(cid, cx, cy)
```

这样车头已上桥（c2/c3 → BRIDGE）、车尾还在岸（c0/c1 → TERRAIN），各触点自动追踪各自的面片。

#### 新单位初始化

```python
# spawn 时用全局 locate_patch（唯一需要表面高度排序的场景）
mid = world_mesh.locate_patch(x, y)
# locate_patch 内部：STRtree 候选 → 无 hint 时取表面最高的面片 → 返回
# 对于 spawn 在桥面上的单位：表面最高 = 桥面 ✅
# 对于 spawn 在桥下的单位：调用方传 hint_z=地面z，选 ray_plane_z 接近的面片
```

### 5.6 底盘悬挂模拟

装甲单位通过**多触点投影 + 弹簧阻尼**模拟颠簸路面/翻越障碍时的真实姿态。

#### 5.6.1 触点模型

```
        前
        ↑
   ┌─────────┐
   │ c2    c3│  ← 四角触点（车身坐标系）
   │         │     c_i = (x, y) + R(yaw) · (±half_len, ±half_wid)
   │ c0    c1│
   └─────────┘
        ↓
        后
```

```python
# ArmorPool 新增字段
armor_contact_z  = np.zeros((MAX_ARMOR, 4), dtype=np.float32)  # 四触点 z
armor_pitch      = np.zeros(MAX_ARMOR, dtype=np.float32)       # 俯仰角 (rad)
armor_roll       = np.zeros(MAX_ARMOR, dtype=np.float32)       # 侧倾角 (rad)
armor_suspension = np.zeros(MAX_ARMOR, dtype=np.float32)       # 悬挂压缩量 (m)
```

#### 5.6.2 每 tick 解算

```python
@njit(parallel=True)
def solve_suspension(
    pos, yaw, half_len, half_wid, mesh_id, world_mesh,
    contact_z, pitch, roll, suspension, dt,
):
    """四触点投影 → 最小二乘平面拟合 → 弹簧阻尼平滑。"""
    for i in prange(n):
        if mesh_id[i] < 0:
            continue

        # 1. 计算四触点世界坐标
        cos_y = cos(yaw[i]); sin_y = sin(yaw[i])
        offsets = [
            (-half_len[i], -half_wid[i]),  # c0: 左后
            (-half_len[i],  half_wid[i]),  # c1: 右后
            ( half_len[i], -half_wid[i]),  # c2: 左前
            ( half_len[i],  half_wid[i]),  # c3: 右前
        ]
        for j in range(4):
            cx = pos[i,0] + cos_y*offsets[j][0] - sin_y*offsets[j][1]
            cy = pos[i,1] + sin_y*offsets[j][0] + cos_y*offsets[j][1]
            contact_z[i,j] = world_mesh.ray_plane_z(mesh_id[i], cx, cy)

        # 2. 从四触点最小二乘拟合平面 → 俯仰/侧倾
        #    平面方程: z = a*x + b*y + c
        #    用四触点做线性回归
        zs = contact_z[i]
        a, b, c = _fit_plane_4points(offsets, zs)
        target_pitch = atan(a)   # ∂z/∂x → 俯仰
        target_roll  = atan(b)   # ∂z/∂y → 侧倾
        target_z     = c         # 底盘中心高度

        # 3. 弹簧阻尼平滑
        #    pitch += (target_pitch - pitch) * stiffness * dt
        #           - pitch_vel * damping * dt
        stiffness = 8.0    # 弹簧刚度 (1/s²)
        damping = 5.0      # 阻尼系数 (1/s)
        pitch[i] += (target_pitch - pitch[i]) * stiffness * dt * dt
        roll[i]  += (target_roll  - roll[i])  * stiffness * dt * dt

        # 4. 悬挂压缩量（底盘 z - 平均触点 z）
        suspension[i] = target_z - zs.mean()
        pos[i, 2] = target_z  # 底盘中心 z 覆盖 sync_ground_z
```

#### 5.6.3 简化版（Phase A 先落地）

完整四触点需要 `_fit_plane_4points` 最小二乘，Phase A 可简化为**三点确定平面**（取 c0/c1/c3，或 c0/c2/c3），公式简洁：

```python
# 三点法线: n = (c1-c0) × (c3-c0)
# target_pitch = atan2(-n.x, n.z)
# target_roll  = atan2( n.y, n.z)
```

#### 5.6.4 碰撞箱跟随

`_sync_boxes_kernel` 在做 `box_pos = armor_pos + R(yaw)*local_offset` 时，
需叠加俯仰/侧倾旋转：

```python
R_total = Rz(yaw) · Rx(roll) · Ry(pitch)  # Z → X → Y 旋转连乘
box_world = armor_pos + R_total @ box_local
```

#### 5.6.5 渲染消费

渲染层从 `armor_pitch/armor_roll` 构造 Model 矩阵，替代当前的纯 yaw 旋转。

#### 5.6.6 性能

- 2048 装甲 × 4 触点 = 8192 次 `ray_plane_z`/tick
- `ray_plane_z` = 1 次数组查表 + dot 运算，< 1μs
- 总耗时 < 0.01ms，完全不影响帧率

当前 `NavMeshBuilder` 从 `Heightfield` + `WalkableCell` 烘焙。WorldMesh 落地后：

```
旧：Heightfield → WalkableCell → 区域生长 → 轮廓 → triangulate → NavGraph
新：WorldMesh.patches → 筛选 walkable → 面片→三角形 → 邻接 → portal → NavGraph
```

核心简化：WorldMesh 每个面片已经是提取好的表面，不需要再重复做表面提取 + 轮廓三角剖分。只需把面片三角化（若面片非三角形）+ 构建邻接。

## 七、渲染适配

当前 `SurfaceExtractor` 从 Heightfield 独立提取渲染网格。WorldMesh 落地后：

```
surface.py 不再从 Heightfield 提取
↓
直接从 WorldMesh.patches 生成渲染网格
  - 地面面片：取 continuous_ground 连续 z
  - 结构面片：z 为量化值（结构本身是平的）
  - 顶点法线 = patch.normal
```

`continuous_ground` 作为渲染输入保留（工厂生成 fBm 连续场），但不再用于单位 z 采样（单位走 WorldMesh）。

## 八、实施路线图

### Phase A：WorldPatch 3D 化（最小改动，可并行）—— ✅ 已实施（2026-08-12）

| 步骤 | 内容 | 影响面 | 状态 |
|------|------|--------|------|
| A1 | `SurfacePatch` 废弃 → 新建 `WorldPatch` (3D vertices + normal) | **新建 world_mesh.py** | ✅ |
| A2 | `SurfaceMesh` → `WorldMesh`，`locate_patch(x,y)` 点定位 + STRtree + hint_z | **新建 world_mesh.py** | ✅ |
| A3 | TerrainMap 持有 `world_mesh`（懒加载，从 Heightfield 第一次构建） | map.py | ✅ |
| A4 | ArmorPool 加 `mesh_id`，`pos.z` 从 `ray_plane_z` 派生 | pools.py | ✅ |
| A5 | `integrate_armor_movement_3d`（Numba 局部性 + z 同步）+ `resolve_mesh_ids`（STRtree fallback） | integrate.py | ✅ |
| A6 | 测试 `tests/test_world_mesh.py`（15 用例） | tests/ | ✅ |

**Phase A 已完成**：装甲 z 跟随面片表面（地面 z = 量化值，结构面 z = span.zmax）。功能跑通。
§5.5 面片穿越已按用户决策更新为「局部性 + hint_z 直接定位」（弃邻接三阶段，见 §5.5）。
pitch/roll/suspension 数组已预留（恒 0），完整悬挂（§5.6）留 Phase B 后启用。

### Phase B：Dual Contouring 提取（3D 表面重建）—— ✅ 已实施（2026-08-12，简化版）

| 步骤 | 内容 | 状态 |
|------|------|------|
| B1 | 实现 Dual Contouring（从 column-span 提取 3D 面片）—— 简化版：轴对齐盒暴露面判定 + 顶/底/侧提取 | ✅ |
| B2 | 地面顶点 z 用 `continuous_ground` 替代量化层 z —— 地面顶面**逐格**四边形，顶点 z 双线性插值 | ✅ |
| B3 | 侧面/底面提取 —— 结构/底面 GreedyMesh 合并 + 侧面沿墙一维合并 | ✅ |
| B4 | WorldPatch.normal 正确计算 —— FACE_TOP/BOTTOM(±z) + 侧面(±x/±y) | ✅ |

**设计决策（2026-08-12）**：
- column-span 是轴对齐盒 → 暴露面恒为轴对齐 → **不需解 QEF**（顶点定位在盒边，保留锐利特征）
- **地面顶面逐格**（不是 GreedyMesh 合并）：合并会丢内部顶点 → 连续场内部起伏 z 丢失
  （Phase B 开发中实测失败暴露）。结构/底面是平面，合并不丢信息。
- 完整 QEF/Hermite（非轴对齐斜面：弹坑/坡道）留后续迭代

### Phase C：NavGraph 从 WorldMesh 重烘焙

| 步骤 | 内容 |
|------|------|
| C1 | `NavMeshBuilder` 改为消费 `WorldMesh.patches` |
| C2 | 面片三角化（非三角形面片 → 约束 Delaunay） |
| C3 | 垂直 portal（跨层连接） |

### Phase D：渲染直接从 WorldMesh —— ✅ 已实施（2026-08-12）

| 步骤 | 内容 | 状态 |
|------|------|------|
| D1 | `surface.py` 改为从 `WorldMesh.patches` 构建渲染网格 | ✅ |
| D2 | 移除 `SurfaceExtractor` 的 Heightfield 依赖 | ✅（to_render_data 已切 WorldMeshRenderer；SurfaceExtractor 保留供旧脚本） |

**实现**（`surface.py::WorldMeshRenderer`）：
- 地面：continuous_ground（若提供）或 WorldMesh 地面 patch 导出格中心 z_map →
  SUBDIV 细分平滑网格（保留连续法线 + 高度色，无斜纹回归）
- 结构（顶/底/侧）：WorldMesh patch 直接三角化（法线 = patch.normal，颜色 = mat 色）
  —— Phase B 完整 3D（桥侧墙/桥底）首次进入渲染
- `TerrainMap.render_world_mesh`（DualContour 完整 3D）与逻辑 `world_mesh`（Phase A 顶面）解耦

## 九、风险与缓解

| 风险 | 缓解 |
|------|------|
| Dual Contouring 复杂度高 | 先 Phase A 用简化版（轴对齐面片）让功能跑通，再迭代 |
| 性能：面片定位 2048 次/tick | 面片数 50~200，暴力遍历 2048×200=0.4M 次，Python 可承受；hot path 用 Numba |
| 面片局部坐标 → 世界坐标不一致 | 三角形重心坐标唯一且 Numba 友好；一般多边形用 MVC |

---

> **一句话**：Heightfield 管碰撞，WorldMesh 管世界，NavGraph 管寻路。三者职责清晰，各有边界。
