"""Numba 移动积分：装甲沿路标点前进（AGENTS.md Phase 2）。

对应 AGENTS.md 3.3「Numba 移动积分（沿路标点前进）」：
- 每个 Tick 推进单位位置，沿当前路标点方向移动
- 到达路标点后切换到下一个
- 速度由外部传入（不同单位/地形/编队有不同的速度）

WorldMesh 集成（2026-08-12，docs/worldmesh-3d-surface-design.md §5）：
- 地面单位统一用 (mesh_id, x, y, z) 表示世界位置
- integrate_armor_movement_3d：推进 (x,y) + 面片局部性检查 + z 同步
- resolve_mesh_ids：对 miss 的单位回退 STRtree 定位（Python 侧）

性能：
    2048 装甲全量移动积分 < 0.5ms（Numba JIT 并行）。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numba import njit, prange

from Airis.lulilo.entities.pools import MAX_ARMOR
from Airis.lulilo.terrain.world_mesh import flatten_world_mesh


@njit(cache=True, parallel=True)
def integrate_armor_movement(
    pos: np.ndarray,           # (N, 3) float32
    yaw: np.ndarray,           # (N,) float32
    active: np.ndarray,        # (N,) bool_
    wp: np.ndarray,            # (N, 8, 2) float32
    wp_count: np.ndarray,      # (N,) uint8
    wp_idx: np.ndarray,        # (N,) uint8
    speed: np.ndarray,         # (N,) float32  每单位当前速度 (m/s)
    dt: float,                 # Tick 时长 (s)
    terrain_z_func: callable | None = None,  # 预留：地形采样（Phase 3 接入）
) -> None:
    """将所有活跃装甲沿路标点推进 dt 秒。

    对每个活跃单位：
    1. 计算到当前目标路标点的方向向量
    2. 按 speed * dt 前进
    3. 若到达路标点（距离 < 阈值），切换到下一个
    4. 更新朝向角 yaw 指向运动方向

    Args:
        pos: 装甲位置数组 (N, 3)，原地修改。
        yaw: 朝向角数组，原地修改。
        active: 存活掩码。
        wp: 路标点数组 (N, 8, 2)。
        wp_count: 路标点数量。
        wp_idx: 当前目标路标点索引，原地修改。
        speed: 当前速度 (m/s)。
        dt: Tick 时长。
        terrain_z_func: 预留，地形高度查询函数 f(x, y) -> z。

    Note:
        terrain_z_func 无法在 nopython 模式下传入 Python callable。
        留作接口占位，Phase 3 用 Numba 函数指针或内联解决。
    """
    n = pos.shape[0]
    for i in prange(n):
        if not active[i]:
            continue

        c = int(wp_idx[i])
        total = int(wp_count[i])
        if total == 0:
            continue  # 无路径，静止

        # 当前目标路标点
        tx = wp[i, c, 0]
        ty = wp[i, c, 1]

        # 方向向量
        dx = tx - pos[i, 0]
        dy = ty - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)

        step = speed[i] * dt

        if dist <= step or dist < 0.01:
            # 到达当前路标点
            pos[i, 0] = tx
            pos[i, 1] = ty
            # 切换到下一个
            if c + 1 < total:
                wp_idx[i] = np.uint8(c + 1)
            else:
                # 已到达最后一个路标点，清除路径
                wp_count[i] = 0
                wp_idx[i] = 0
        else:
            # 沿方向前进
            nx = dx / dist
            ny = dy / dist
            pos[i, 0] += nx * step
            pos[i, 1] += ny * step
            # 更新朝向
            yaw[i] = np.arctan2(ny, nx)


@njit(cache=True)
def integrate_single_armor(
    px: float,
    py: float,
    pz: float,
    tx: float,
    ty: float,
    speed: float,
    dt: float,
) -> tuple[float, float, float, bool]:
    """单个单位的移动积分（供非并行场景使用）。

    Returns:
        (new_px, new_py, new_yaw, arrived) —— 新位置、新朝向、是否到达。
    """
    dx = tx - px
    dy = ty - py
    dist = np.sqrt(dx * dx + dy * dy)
    step = speed * dt

    if dist <= step or dist < 0.01:
        new_yaw = np.arctan2(dy, dx)
        return tx, ty, new_yaw, True

    nx = dx / dist
    ny = dy / dist
    new_yaw = np.arctan2(ny, nx)
    return px + nx * step, py + ny * step, new_yaw, False


@njit(cache=True, parallel=True)
def integrate_projectile_movement(
    pos: np.ndarray,       # (M, 3) float32
    vel: np.ndarray,       # (M, 3) float32
    active: np.ndarray,    # (M,) bool_
    dt: float,
    gravity: float = 9.81,
) -> None:
    """将所有活跃抛射物推进 dt 秒（含重力）。

    Args:
        pos: 抛射物位置，原地修改。
        vel: 速度向量，原地修改（导弹可能会改方向，预留）。
        active: 存活掩码。
        dt: Tick 时长（抛射物 Tick=0.01s）。
        gravity: 重力加速度 (m/s²)，沿 -Z 方向。
    """
    m = pos.shape[0]
    for i in prange(m):
        if not active[i]:
            continue
        # 位置积分
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt
        # 重力影响 Z 速度
        vel[i, 2] -= gravity * dt


@njit(cache=True)
def compute_speed(
    base_speed: float,
    slope_deg: float,
    max_slope_deg: float = 30.0,
    min_speed_factor: float = 0.2,
) -> float:
    """根据坡度计算实际速度。

    Args:
        base_speed: 平地速度 (m/s)。
        slope_deg: 当前坡度 (°)。
        max_slope_deg: 最大可通行坡度。
        min_speed_factor: 最陡坡时的速度因子。

    Returns:
        实际速度 (m/s)。坡度超过 max_slope_deg 时返回 0（不可通行）。
    """
    if slope_deg >= max_slope_deg:
        return 0.0
    # 线性插值：平地 100%，最陡坡 min_speed_factor
    factor = 1.0 - (1.0 - min_speed_factor) * (slope_deg / max_slope_deg)
    return base_speed * max(factor, 0.0)


# ---------------------------------------------------------------------------
# WorldMesh 集成：mesh_id 追踪 + z 同步
# ---------------------------------------------------------------------------
# 设计决策（2026-08-12，与用户讨论）：
# - 移动热路径**不需要邻接表**——利用「单位移动连续」的物理事实：
#   先查当前面片（point_in_poly），99%+ 的 tick 直接命中；
#   miss（mesh_id 置 -1）才回退 Python 侧 STRtree（resolve_mesh_ids）。
# - 邻接表（WorldMesh.adjacency）仅供 A*/SSFA 寻路使用，不进热路径。
# ---------------------------------------------------------------------------


@njit(cache=True)
def _point_in_poly_flat(
    px: float,
    py: float,
    patch_v: np.ndarray,      # (total_v, 3) float32
    vstart: np.ndarray,       # (M+1,) int32
    vcnt: np.ndarray,         # (M,) int32
    pid: int,
) -> bool:
    """射线法：点在面片内（展平数组版）。"""
    n = int(vcnt[pid])
    s = int(vstart[pid])
    inside = False
    j = n - 1
    for i in range(n):
        xi = patch_v[s + i, 0]
        yi = patch_v[s + i, 1]
        xj = patch_v[s + j, 0]
        yj = patch_v[s + j, 1]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


@njit(cache=True)
def _ray_plane_z_flat(
    patch_v: np.ndarray,      # (total_v, 3) float32
    vstart: np.ndarray,       # (M+1,) int32
    vcnt: np.ndarray,         # (M,) int32
    norm: np.ndarray,         # (M, 3) float32
    pid: int,
    x: float,
    y: float,
) -> float:
    """竖直射线与面片求交 z（展平数组版）。

    plane: n·(p - v0) = 0 → t = (n·v0 - n.x·x - n.y·y) / n.z
    """
    nz = norm[pid, 2]
    if abs(nz) < 1e-9:
        return -1e9  # 竖直面，不可站立
    s = int(vstart[pid])
    nx = norm[pid, 0]
    ny = norm[pid, 1]
    v0 = patch_v[s]
    nv0 = nx * v0[0] + ny * v0[1] + nz * v0[2]
    return (nv0 - nx * x - ny * y) / nz


@njit(cache=True)
def _z_bilinear_flat(
    patch_v: np.ndarray,      # (total_v, 3) float32
    vstart: np.ndarray,       # (M+1,) int32
    vcnt: np.ndarray,         # (M,) int32
    norm: np.ndarray,         # (M, 3) float32
    pid: int,
    x: float,
    y: float,
) -> float:
    """轴对齐矩形面片双线性 z 采样（展平数组版，docs/suspension-slope-design.md §4.2 E1）。

    顶点顺序约定 [v0,v1,v2,v3] = [x0y0, x1y0, x1y1, x0y1]（dual_contour /
    WorldMeshBuilder 输出一致）。结构面片 4 顶点 z 相同 → 退化为常量。
    非 4 顶点 / 非矩形（如侧面）→ 回退 ray_plane_z（保守）。
    """
    c = int(vcnt[pid])
    s = int(vstart[pid])
    if c != 4:
        return _ray_plane_z_flat(patch_v, vstart, vcnt, norm, pid, x, y)
    v0 = patch_v[s]
    v1 = patch_v[s + 1]
    v2 = patch_v[s + 2]
    v3 = patch_v[s + 3]
    dx = v1[0] - v0[0]
    dy = v3[1] - v0[1]
    if dx < 1e-9 or dy < 1e-9:
        return _ray_plane_z_flat(patch_v, vstart, vcnt, norm, pid, x, y)
    u = (x - v0[0]) / dx
    v = (y - v0[1]) / dy
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return (
        (1.0 - u) * (1.0 - v) * v0[2]
        + u * (1.0 - v) * v1[2]
        + u * v * v2[2]
        + (1.0 - u) * v * v3[2]
    )


@njit(cache=True)
def integrate_armor_movement_3d(
    pos: np.ndarray,           # (N, 3) float32
    yaw: np.ndarray,           # (N,) float32
    active: np.ndarray,        # (N,) bool_
    wp: np.ndarray,            # (N, 8, 2) float32
    wp_count: np.ndarray,      # (N,) uint8
    wp_idx: np.ndarray,        # (N,) uint8
    speed: np.ndarray,         # (N,) float32
    dt: float,                 # Tick 时长 (s)
    mesh_id: np.ndarray,       # (N,) int32  —— 原地更新
    patch_v: np.ndarray,       # (total_v, 3) float32
    patch_vstart: np.ndarray,  # (M+1,) int32
    patch_vcnt: np.ndarray,    # (M,) int32
    patch_norm: np.ndarray,    # (M, 3) float32
) -> None:
    """将所有活跃装甲推进 dt 秒，并同步 mesh_id 与 z（WorldMesh 版）。

    每个活跃单位：
    1. 沿路标点推进 (x, y)（同 integrate_armor_movement）
    2. 面片局部性检查：仍在当前面片 → 保持 mesh_id，用 ray_plane_z 同步 z
    3. 已越出当前面片 → mesh_id 置 -1（由 Python 侧 resolve_mesh_ids 回退
       STRtree 定位——Numba 无法调用 shapely）
    4. 空中/未绑定（mesh_id < 0）→ z 不处理

    🔥 顺序化（2026-08-13）：短内核并行是负优化——prange 每帧唤醒
    16 线程（omp）忙等同步，CPU 占用 15.7x（60fps 烧 ~22% CPU）而
    顺序版 2048 单位仅 0.137ms/帧、无线程忙等。移动积分非碰撞热点，
    顺序已远超 60fps 预算。

    Args:
        pos/yaw/active/wp/wp_count/wp_idx/speed/dt: 同 integrate_armor_movement。
        mesh_id: 所在面片 ID，原地修改。
        patch_v/patch_vstart/patch_vcnt/patch_norm:
            WorldMesh 展平数组（terrain.world_mesh.flatten_world_mesh）。
    """
    n = pos.shape[0]
    for i in range(n):
        if not active[i]:
            continue

        c = int(wp_idx[i])
        total = int(wp_count[i])
        if total > 0:
            tx = wp[i, c, 0]
            ty = wp[i, c, 1]
            dx = tx - pos[i, 0]
            dy = ty - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            step = speed[i] * dt

            if dist <= step or dist < 0.01:
                pos[i, 0] = tx
                pos[i, 1] = ty
                if c + 1 < total:
                    wp_idx[i] = np.uint8(c + 1)
                else:
                    wp_count[i] = 0
                    wp_idx[i] = 0
            else:
                nx = dx / dist
                ny = dy / dist
                pos[i, 0] += nx * step
                pos[i, 1] += ny * step
                yaw[i] = np.arctan2(ny, nx)

        # ---- mesh_id 局部性检查 + z 同步（双线性 bbox 判定，docs E1）----
        mid = mesh_id[i]
        if mid >= 0:
            inside, z = _bilinear_rect_sample_flat(
                patch_v, patch_vstart, patch_vcnt, patch_norm,
                mid, pos[i, 0], pos[i, 1],
            )
            if inside:
                # 仍在当前面片 → 同步 z（地面面片双线性插值，含边界命中）
                pos[i, 2] = z
            else:
                # 已越出 → 标记待 Python fallback
                mesh_id[i] = -1
        # mid < 0：空中/未绑定，z 不处理


def resolve_mesh_ids(
    pos: np.ndarray,
    mesh_id: np.ndarray,
    world_mesh: Any,
) -> None:
    """对 mesh_id == -1 的单位做 STRtree 定位（Python 侧，非热路径）。

    仅在 Numba 局部性检查 miss 后调用（每 tick 通常只有少量单位越出
    面片）。用 hint_z=当前 z 定位——桥下单位选 ray_plane_z 最接近的面片。

    Args:
        pos: (N, 3) float32，原地修改（命中后同步 z）。
        mesh_id: (N,) int32，原地修改。
        world_mesh: WorldMesh（提供 locate_patch / ray_plane_z）。
    """
    idxs = np.flatnonzero(mesh_id == -1)
    for i in idxs:
        pid = world_mesh.locate_patch(
            float(pos[i, 0]), float(pos[i, 1]), hint_z=float(pos[i, 2])
        )
        mesh_id[i] = pid
        if pid >= 0:
            # 双线性采样（docs E1）：地面面片顶点 z 各异 → 插值而非平面近似
            pos[i, 2] = world_mesh.patches[pid].z_at(
                float(pos[i, 0]), float(pos[i, 1])
            )


# ---------------------------------------------------------------------------
# 悬挂模拟（docs/suspension-slope-design.md §3，2026-08-13）
# ---------------------------------------------------------------------------
# 四触点投影 + 三点平面拟合 + 弹簧阻尼：
#   - 触点局部偏移（车身坐标系）：c0=(-L,-W) 左后, c1=(-L,+W) 右后,
#     c2=(+L,-W) 左前, c3=(+L,+W) 右前（L=half[1] 车长半, W=half[0] 车宽半）
#   - 触点 z 采样：contact_mesh_id 局部性检查 → _z_bilinear_flat（E1 双线性）；
#     miss 置 -1，由 Python resolve_contact_mesh_ids 回退 STRtree
#   - 三点平面（c0/c1/c3，车身局部坐标 z = a·lx + b·ly + c）：
#       b = (z1-z0)/(2W) → roll；a = (z3-z1)/(2L) → pitch；c = 底盘中心 z
#   - 弹簧阻尼（方案 B 临界阻尼）：k=stiffness, c_damp=2√k，半隐式欧拉


@njit(cache=True)
def _bilinear_rect_sample_flat(
    patch_v: np.ndarray,      # (total_v, 3) float32
    vstart: np.ndarray,       # (M+1,) int32
    vcnt: np.ndarray,         # (M,) int32
    norm: np.ndarray,         # (M, 3) float32
    pid: int,
    x: float,
    y: float,
) -> tuple[bool, float]:
    """轴对齐矩形面片采样，返回 (在面片内?, 双线性 z)。

    🔥 边界点修正（2026-08-13）：射线法 `_point_in_poly_flat` 对边界点
    （u=0/v=0 等，如车长 6m=3×cs 对称放置时触点全落格线上）误判 miss
    → 触点一直钳制近似 → 姿态偏小。矩形 bbox 判定（u,v∈[0,1] 含边界）
    与 shapely `covers` 一致。非 4 顶点 / 非矩形 → (False, ray_plane_z)。
    """
    c = int(vcnt[pid])
    s = int(vstart[pid])
    if c != 4:
        return False, _ray_plane_z_flat(patch_v, vstart, vcnt, norm, pid, x, y)
    v0 = patch_v[s]
    v1 = patch_v[s + 1]
    v2 = patch_v[s + 2]
    v3 = patch_v[s + 3]
    dx = v1[0] - v0[0]
    dy = v3[1] - v0[1]
    if dx < 1e-9 or dy < 1e-9:
        return False, _ray_plane_z_flat(patch_v, vstart, vcnt, norm, pid, x, y)
    u = (x - v0[0]) / dx
    v = (y - v0[1]) / dy
    inside = (u >= 0.0) and (u <= 1.0) and (v >= 0.0) and (v <= 1.0)
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    z = (
        (1.0 - u) * (1.0 - v) * v0[2]
        + u * (1.0 - v) * v1[2]
        + u * v * v2[2]
        + (1.0 - u) * v * v3[2]
    )
    return inside, z


@njit(cache=True)
def _sample_contact_z(
    lx: float,
    ly: float,
    i: int,
    j: int,
    pos: np.ndarray,          # (N,3) float32
    yaw: np.ndarray,          # (N,) float32
    mesh_id: np.ndarray,      # (N,) int32
    contact_mesh_id: np.ndarray,  # (N,4) int32
    contact_z: np.ndarray,    # (N,4) float32 旧值（越出时保留）
    patch_v: np.ndarray,      # (total_v,3) float32
    patch_vstart: np.ndarray,  # (M+1,) int32
    patch_vcnt: np.ndarray,    # (M,) int32
    patch_norm: np.ndarray,    # (M,3) float32
) -> tuple[bool, float, int]:
    """第 j 触点世界 xy + z 采样，返回 (有效?, z, 面片 id)。

    🔥 跳动修复（2026-08-13）：触点跨面片时若用钳制近似，本帧错、
    resolve 修复后下帧对 → 帧间 z 跳变（实测 p99=0.12m）。改为：
    越出触点/单位面片时返回 (False, 旧 contact_z, -1) —— 触点 z 延续
    旧值平滑过渡（一帧移动仅 0.08m，误差 = 0.08×坡度，可忽略），
    Python resolve 定位新面片后后续帧精确。
    """
    cos_y = np.cos(yaw[i])
    sin_y = np.sin(yaw[i])
    cx = pos[i, 0] + cos_y * lx - sin_y * ly
    cy = pos[i, 1] + sin_y * lx + cos_y * ly
    mid = mesh_id[i]
    cid = contact_mesh_id[i, j]
    if cid >= 0:
        inside, z = _bilinear_rect_sample_flat(
            patch_v, patch_vstart, patch_vcnt, patch_norm, cid, cx, cy
        )
        if inside:
            return True, z, cid
    if mid >= 0:
        inside, z = _bilinear_rect_sample_flat(
            patch_v, patch_vstart, patch_vcnt, patch_norm, mid, cx, cy
        )
        if inside:
            return True, z, mid
    # 越出：保留旧 contact_z（平滑延续），标记 -1 由 Python resolve 修正
    return False, contact_z[i, j], -1


@njit(cache=True)
def solve_suspension(
    pos: np.ndarray,           # (N,3) float32  —— pos.z 原地覆盖为底盘 z
    yaw: np.ndarray,           # (N,) float32
    half: np.ndarray,          # (N,3) float32  half[1]=L 车长半, half[0]=W 车宽半
    active: np.ndarray,        # (N,) bool_
    mesh_id: np.ndarray,       # (N,) int32
    contact_z: np.ndarray,     # (N,4) float32  —— out 四触点表面 z
    contact_mesh_id: np.ndarray,  # (N,4) int32  —— in/out（miss 置 -1）
    pitch: np.ndarray,         # (N,) float32  —— in/out
    roll: np.ndarray,          # (N,) float32  —— in/out
    suspension: np.ndarray,    # (N,) float32  —— out 悬挂压缩量
    pitch_vel: np.ndarray,     # (N,) float32  —— in/out 弹簧速度
    roll_vel: np.ndarray,      # (N,) float32  —— in/out
    dt: float,
    stiffness: float,
    patch_v: np.ndarray,       # (total_v,3) float32
    patch_vstart: np.ndarray,  # (M+1,) int32
    patch_vcnt: np.ndarray,    # (M,) int32
    patch_norm: np.ndarray,    # (M,3) float32
) -> None:
    """四触点投影 → 三点平面拟合 → 弹簧阻尼平滑（docs §3.6/§3.7）。

    每个活跃地面单位：
    1. 四触点世界 xy = pos + R(yaw)·(±L, ±W)，z 用触点面片双线性采样
    2. 三点平面（c0/c1/c3）→ target_pitch/roll/底盘 z
    3. 临界阻尼弹簧平滑 pitch/roll（带速度项）
    4. suspension = 底盘 z − 平均触点 z；pos.z = 底盘 z（覆盖 E1 单点同步）

    🔥 顺序化（2026-08-13）：同 integrate_armor_movement_3d——prange
    每帧 16 线程忙等烧 CPU（60fps ~22%），顺序版 2048 单位 0.137ms。
    """
    n = pos.shape[0]
    for i in range(n):
        if not active[i]:
            continue
        mid = mesh_id[i]
        if mid < 0:
            continue  # 空中/未绑定：姿态不处理
        L = half[i, 1]
        W = half[i, 0]
        if L < 1e-6 or W < 1e-6:
            continue

        # 1) 四触点世界 xy + z 采样
        z0 = z1 = z2 = z3 = 0.0
        for j in range(4):
            lx = -L if j < 2 else L
            ly = -W if (j % 2) == 0 else W
            valid, z, cid = _sample_contact_z(
                lx, ly, i, j, pos, yaw, mesh_id, contact_mesh_id, contact_z,
                patch_v, patch_vstart, patch_vcnt, patch_norm,
            )
            if valid:
                contact_z[i, j] = z
                contact_mesh_id[i, j] = cid
            else:
                # 越出（跨面片过渡帧）：旧 z 参与拟合（平滑延续，无跳变），
                # contact_mesh_id 置 -1 → 下帧 resolve_contact_mesh_ids 定位新面片
                z = contact_z[i, j]
                contact_mesh_id[i, j] = -1
            if j == 0:
                z0 = z
            elif j == 1:
                z1 = z
            elif j == 2:
                z2 = z
            else:
                z3 = z

        # 2) 三点平面（c0/j0, c1/j1, c3/j3，车身局部坐标）
        a = (z3 - z1) / (2.0 * L)      # ∂z/∂lx → pitch
        b = (z1 - z0) / (2.0 * W)      # ∂z/∂ly → roll
        target_pitch = np.arctan(a)
        target_roll = np.arctan(b)
        target_z = z0 + a * L + b * W  # 底盘中心目标 z

        # 3) 弹簧阻尼（临界阻尼 k, c=2√k，半隐式欧拉 + 子步）
        #    稳定域：k·dt_sub² < 1。地面 Tick=1s 时 dt 过大（k=8 → K=8 发散），
        #    内部拆子步 dt_sub ≤ 0.1s（K=0.08）保证稳定（docs §3.6 方案 B）。
        k = stiffness
        c_damp = 2.0 * np.sqrt(k)
        n_sub = max(1, int(np.ceil(dt / 0.1)))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            pitch_vel[i] += (
                (target_pitch - pitch[i]) * k - pitch_vel[i] * c_damp
            ) * dt_sub
            pitch[i] += pitch_vel[i] * dt_sub
            roll_vel[i] += (
                (target_roll - roll[i]) * k - roll_vel[i] * c_damp
            ) * dt_sub
            roll[i] += roll_vel[i] * dt_sub

        # 4) 悬挂压缩量 + 底盘 z
        suspension[i] = target_z - (z0 + z1 + z2 + z3) * 0.25
        pos[i, 2] = target_z


def resolve_contact_mesh_ids(
    pos: np.ndarray,
    yaw: np.ndarray,
    half: np.ndarray,
    contact_z: np.ndarray,
    contact_mesh_id: np.ndarray,
    world_mesh: Any,
) -> None:
    """对 contact_mesh_id == -1 的触点做 STRtree 定位（Python 侧，非热路径）。

    触点世界坐标 = pos + R(yaw)·(±L, ±W)，用 hint_z=触点当前 z 定位
    （桥下触点选 ray_plane_z 接近的面片，避免被桥面歧义）。

    Args:
        pos/yaw/half: ArmorPool 主数组（(N,3)/(N,)/(N,3)）。
        contact_z: (N,4) float32，原地更新（命中后精确 z）。
        contact_mesh_id: (N,4) int32，原地更新。
        world_mesh: WorldMesh（locate_patch / patches[].z_at）。
    """
    n = pos.shape[0]
    for i in range(n):
        cos_y = float(np.cos(yaw[i]))
        sin_y = float(np.sin(yaw[i]))
        L = float(half[i, 1])
        W = float(half[i, 0])
        hint = float(pos[i, 2])  # 单位当前 z（触点在单位周围同层表面，最可靠）
        for j in range(4):
            if contact_mesh_id[i, j] != -1:
                continue
            lx = -L if j < 2 else L
            ly = -W if (j % 2) == 0 else W
            cx = float(pos[i, 0]) + cos_y * lx - sin_y * ly
            cy = float(pos[i, 1]) + sin_y * lx + cos_y * ly
            # 🔥 hint 用单位 z：contact_z 初始 0 在「桥面/地面共存」时会把
            # 触点错误定位到地面（z=0），单位 z 平滑且与触点同层
            pid = world_mesh.locate_patch(cx, cy, hint_z=hint)
            contact_mesh_id[i, j] = pid
            if pid >= 0:
                contact_z[i, j] = world_mesh.patches[pid].z_at(cx, cy)


def build_armor_movement_hook(
    armor: Any,
    world_mesh: Any,
    speed: np.ndarray,
    stiffness: float = 8.0,
) -> Callable:
    """构建 movement 子系统 hook：移动 → mesh resolve → 悬挂 → 触点 resolve。

    供 ``SimulationLoop.attach("movement", hook)`` 使用（docs/suspension-slope-design.md
    第二节管线总览）。WorldMesh 展平数组在 hook 创建时缓存一次（WorldMesh
    不变时复用；地形破坏重建后应重新构建 hook）。

    Args:
        armor: ArmorPool（含 wp/mesh_id/pitch/roll/contact_* 等悬挂字段）。
        world_mesh: WorldMesh（z 采样 + STRtree 定位）。
        speed: (N,) float32 单位速度（沿路标点）。
        stiffness: 悬挂弹簧刚度 (rad/s²)，默认 8.0（临界阻尼）。

    Returns:
        hook(loop, dt) -> None，可直接 attach 到 SimulationLoop。
    """
    flat = flatten_world_mesh(world_mesh)

    def hook(_loop: Any, dt: float) -> None:
        pv, vstart, vcnt, norm = flat
        integrate_armor_movement_3d(
            armor.pos, armor.yaw, armor.active,
            armor.wp, armor.wp_count, armor.wp_idx, speed, float(dt),
            armor.mesh_id, pv, vstart, vcnt, norm,
        )
        resolve_mesh_ids(armor.pos, armor.mesh_id, world_mesh)
        # 🔥 跳动修复：先解析触点面片，再 solve_suspension —— 触点在跨面片
        # 过渡帧也能用最新面片采样，消除「本帧近似、下帧精确」的 z 跳变
        resolve_contact_mesh_ids(
            armor.pos, armor.yaw, armor.half,
            armor.contact_z, armor.contact_mesh_id, world_mesh,
        )
        solve_suspension(
            armor.pos, armor.yaw, armor.half, armor.active, armor.mesh_id,
            armor.contact_z, armor.contact_mesh_id,
            armor.pitch, armor.roll, armor.suspension,
            armor.pitch_vel, armor.roll_vel,
            float(dt), stiffness, pv, vstart, vcnt, norm,
        )

    return hook


# ---------------------------------------------------------------------------
# Phase 6b 制导积分：导弹控制回调（docs/phase6-guidance-pipeline.md §四）
# ---------------------------------------------------------------------------
# 架构（2026-08-14 用户澄清）：
#   - guid_law = 指向 MissileControlRegistry 的函数索引；末敏弹也有索引
#     （sfm_ballistic），但 BALLISTIC 阶段该函数不制导（输出 0 → 直线+重力）。
#   - guid_stage / guid_override 是传给控制回调的状态机输入。
#   - 内核按 guid_stage 分派物理：
#       BALLISTIC → 直线 + 重力（函数不制导，与 guid_law=-1 直线积分等价）
#       SCAN      → 悬停（末敏弹开伞扫描，位置不动；扫描/引爆由 Python 层驱动）
#       TRACK/MEMORY/TERMINAL → 调 control_fn（回调读 stage/override 决策）
#   - 执行器饱和（|a| ≤ max_accel）/ FOV 门限 / 重力由本内核统一施加，
#     回调只算总加速度（陷阱 §八.1）。
#   - 阶段常量与 combat/guidance.GuidanceStage 对齐（movement 层不能反向
#     import combat，故用 int 字面量，勿改值）。
#   - Numba 多版本缓存：同一内核传不同 control_fn → 各自编译缓存
#     （陷阱 §八.2：重命名回调后清 __pycache__）。

_GUIDANCE_STAGE_TRACK = 0
_GUIDANCE_STAGE_MEMORY = 1
_GUIDANCE_STAGE_TERMINAL = 2
_GUIDANCE_STAGE_BALLISTIC = 3
_GUIDANCE_STAGE_SCAN = 4

# 模块级零速度缓存（陷阱 §八.6：地面目标速度恒 0，别在内核里每弹丸 np.zeros）
_ZERO_VEL = np.zeros(3, dtype=np.float32)


@njit(cache=True, parallel=True)
def integrate_guided_projectile_movement(
    pos: np.ndarray,                       # (M, 3) float32
    vel: np.ndarray,                       # (M, 3) float32
    active: np.ndarray,                    # (M,) bool_
    age: np.ndarray,                       # (M,) float32 飞行时间（推进剖面时序）
    guid_target_pos: np.ndarray,           # (M, 3) float32 检测到的目标坐标
    guid_memory_pos: np.ndarray,           # (M, 3) float32 最后已知坐标（MEMORY）
    guid_stage: np.ndarray,                # (M,) int32 阶段状态机（给回调的输入）
    guid_nav_gain: np.ndarray,             # (M,) float32 制导增益（per-ammo，SoA）
    guid_override: np.ndarray,             # (M,) int32 人在回路推进指令（已门控）
    guid_search_fov_cos: np.ndarray,       # (M,) float32 搜索视场余弦（per-ammo）
    guid_track_fov_cos: np.ndarray,        # (M,) float32 跟踪视场余弦（per-ammo）
    max_accel: float,                      # 总加速度饱和（按 law 分组，含推力+转向）
    dt: float,
    gravity: float,
    control_fn: Any,                       # @njit 控制回调（推进+制导合一）
) -> None:
    """把活跃的制导弹推进 dt 秒（按 guid_stage 分派物理 + 调控制回调）。

    - BALLISTIC：直线 + 重力（**该阶段控制回调不制导**，与 guid_law=-1 等价）。
    - SCAN：悬停（位置/速度不变）。
    - TRACK / MEMORY / TERMINAL：目标坐标（MEMORY 用记忆点，其余用检测坐标）→
      FOV 门限 → control_fn 算总加速度 → 执行器饱和 → 半隐式积分 + 重力。
    - 目标速度恒 0（模块级 _ZERO_VEL；Phase 7 空战接入真实 target_vel）。

    Note:
        control_fn 签名（docs §3.5，含 stage 状态机输入）：
            (pos, vel, tpos, tvel, age, stage, override, nav_gain, dt, out) -> None
    """
    m = pos.shape[0]
    for i in prange(m):
        if not active[i]:
            continue
        stage = int(guid_stage[i])

        # BALLISTIC：弹道阶段——函数不制导，直线 + 重力
        if stage == _GUIDANCE_STAGE_BALLISTIC:
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            pos[i, 2] += vel[i, 2] * dt
            vel[i, 2] -= gravity * dt
            continue

        # SCAN：开伞扫描——悬停（位置/速度不变），扫描探测由 Python 层驱动
        if stage == _GUIDANCE_STAGE_SCAN:
            continue

        # TRACK / MEMORY / TERMINAL：目标坐标（stage 决定来源）
        if stage == _GUIDANCE_STAGE_MEMORY:
            tpos = guid_memory_pos[i]
        else:
            tpos = guid_target_pos[i]

        # FOV 门限：TRACK/TERMINAL 用窄跟踪视场（抗跳变），MEMORY 用搜索视场
        if stage == _GUIDANCE_STAGE_TRACK or stage == _GUIDANCE_STAGE_TERMINAL:
            fov_cos = guid_track_fov_cos[i]
        else:
            fov_cos = guid_search_fov_cos[i]
        if fov_cos < 1.0:
            dx = tpos[0] - pos[i, 0]
            dy = tpos[1] - pos[i, 1]
            dz = tpos[2] - pos[i, 2]
            dr = np.sqrt(dx * dx + dy * dy + dz * dz)
            spd = np.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
            aligned = (
                (dx * vel[i, 0] + dy * vel[i, 1] + dz * vel[i, 2]) / (dr * spd)
                if dr > 1e-6 and spd > 1e-6
                else 1.0
            )
            if aligned < fov_cos:
                # 目标超出接收视场 → 停导直射（直线 + 重力，碰撞自然判脱靶）
                pos[i, 0] += vel[i, 0] * dt
                pos[i, 1] += vel[i, 1] * dt
                pos[i, 2] += vel[i, 2] * dt
                vel[i, 2] -= gravity * dt
                continue

        # 控制回调：总加速度 = 转向 + 轴向推力（回调内部按 stage/override 分派）
        accel = np.empty(3, dtype=np.float32)
        control_fn(
            pos[i], vel[i], tpos, _ZERO_VEL,
            age[i], guid_stage[i], guid_override[i], guid_nav_gain[i], dt, accel,
        )

        # 执行器饱和：总加速度上限（转向 + 轴向推力，40g 默认防截断推力）
        a2 = accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]
        if a2 > max_accel * max_accel:
            s = max_accel / np.sqrt(a2)
            accel[0] *= s
            accel[1] *= s
            accel[2] *= s

        # 半隐式积分：a → v → x + 重力（与直线积分同阶）
        vel[i, 0] += accel[0] * dt
        vel[i, 1] += accel[1] * dt
        vel[i, 2] += accel[2] * dt
        vel[i, 2] -= gravity * dt
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt
