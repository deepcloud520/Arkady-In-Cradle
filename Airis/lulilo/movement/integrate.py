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

from typing import Any

import numpy as np
from numba import njit, prange

from Airis.lulilo.entities.pools import MAX_ARMOR


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


@njit(cache=True, parallel=True)
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

    Args:
        pos/yaw/active/wp/wp_count/wp_idx/speed/dt: 同 integrate_armor_movement。
        mesh_id: 所在面片 ID，原地修改。
        patch_v/patch_vstart/patch_vcnt/patch_norm:
            WorldMesh 展平数组（terrain.world_mesh.flatten_world_mesh）。
    """
    n = pos.shape[0]
    for i in prange(n):
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

        # ---- mesh_id 局部性检查 + z 同步 ----
        mid = mesh_id[i]
        if mid >= 0:
            if _point_in_poly_flat(
                pos[i, 0], pos[i, 1], patch_v, patch_vstart, patch_vcnt, mid
            ):
                # 仍在当前面片 → 同步 z
                pos[i, 2] = _ray_plane_z_flat(
                    patch_v, patch_vstart, patch_vcnt, patch_norm,
                    mid, pos[i, 0], pos[i, 1],
                )
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
            pos[i, 2] = world_mesh.ray_plane_z(pid, float(pos[i, 0]), float(pos[i, 1]))
