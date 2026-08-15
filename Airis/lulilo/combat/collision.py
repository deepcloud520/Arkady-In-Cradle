"""Numba 并行碰撞检测 + 空间哈希（AGENTS.md Phase 3：combat/collision.py）。

对应 AGENTS.md 3.4「碰撞检测系统」—— 统一为「胶囊体 vs AABB」：
    | 任意弹丸（高速/低速、大/小口径） | 胶囊体 vs AABB（AABB 各轴膨胀弹丸半径 r 后做 Slabs 线段检测） | Numba JIT |
    | 装甲 vs 装甲（编队推开）        | 排序扫描（Sort & Sweep）                                     | np.argsort+Numba |

设计决策（2026-08-11）：检测算法**不再由弹丸类型/速度决定**，统一用
「胶囊体」模型——弹丸轨迹 = 半径 r 的球体沿线段扫掠。胶囊体 ∩ AABB
严格等价于「线段 ∩ (AABB ⊕ 球r)」，而 ⊕ 后仍是 AABB（球各向同性），
因此只需把 AABB 各半轴加 r 再做 Slabs 线段检测，成本 ≈ 线段检测。

这一设计同时覆盖：
    - 高速细弹丸（防穿透：线段本质）
    - 低速大弹丸（半径覆盖：AABB 膨胀）
    - 超高速大弹丸（两者兼得，不依赖弹种/速度判据）
    - 多分辨率 Tick（地面 1s / 空中 0.1s / 子弹 0.01s）下天然正确

空间哈希（替代八叉树）：无状态，每帧即时计算格子 ID，
子弹只与相邻 9 格内的装甲做精确碰撞检测（AGENTS.md 3.4 空间哈希小节）。

局部空间碰撞（规避 OBB 开销）：把弹丸坐标旋转到装甲局部坐标系（仅 yaw），
局部坐标系中碰撞箱是 AABB（AGENTS.md 3.4 check_collision 代码）。

性能目标（AGENTS.md 3.4）：精确检测约 335 万次，Numba 单帧 < 3ms。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from Airis.lulilo.core.logging import get_logger

logger = get_logger(__name__)

# 空间哈希格子尺寸（AGENTS.md 3.4：grid_size=50.0）
DEFAULT_GRID_SIZE = 50.0


# ---------------------------------------------------------------------------
# 基础数学内核（Numba）
# ---------------------------------------------------------------------------


@njit(cache=True)
def spatial_hash(x: float, y: float, grid_size: float = DEFAULT_GRID_SIZE) -> tuple[int, int]:
    """无状态空间哈希：世界坐标 → 格子 ID（AGENTS.md 3.4）。"""
    return int(x / grid_size), int(y / grid_size)


@njit(cache=True)
def segment_aabb_single(
    p0x: float, p0y: float, p0z: float,
    p1x: float, p1y: float, p1z: float,
    box_cx: float, box_cy: float, box_cz: float,
    box_hx: float, box_hy: float, box_hz: float,
    radius: float = 0.0,
) -> tuple[bool, float]:
    """胶囊体 vs AABB 碰撞（Slabs 线段检测 + AABB 半径膨胀，统一算法）。

    数学依据：弹丸轨迹 = 半径 r 的球沿线段扫掠（胶囊体）。胶囊体 ∩ AABB
    严格等价于「线段 ∩ (AABB ⊕ 球r)」；⊕ 后仍是 AABB（球各向同性），
    故只需把半轴各加 r 再做 Slabs——成本与纯线段检测相同，同时保留
    弹丸半径（大弹头）与防穿透（高速）两大特性，不依赖弹种/速度判据。

    Args:
        p0/p1: 线段两端点（世界坐标，弹丸上一帧/当前帧位置）。
        box_c: AABB 中心（世界坐标）。
        box_h: AABB 半边长。
        radius: 弹丸碰撞半径 r（≥0）。内部做 box_h + r 膨胀。

    Returns:
        (是否命中, 命中参数 t ∈ [0,1]，未命中返回 t=1e9)。
        Note: t 是「膨胀 AABB」的进入点，比真实表面接触早 ~r，属安全侧。
    """
    # AABB 各轴膨胀弹丸半径（胶囊体 ⊕ AABB 等价变换）
    ex = box_hx + radius
    ey = box_hy + radius
    ez = box_hz + radius

    d0 = p0x - box_cx
    d1 = p0y - box_cy
    d2 = p0z - box_cz
    e0 = p1x - p0x
    e1 = p1y - p0y
    e2 = p1z - p0z

    tmin = 0.0
    tmax = 1.0

    # 逐轴 Slabs 裁剪（膨胀后半轴）
    if abs(e0) < 1e-12:
        if d0 < -ex or d0 > ex:
            return False, 1e9
    else:
        inv = 1.0 / e0
        t1 = (-ex - d0) * inv
        t2 = (ex - d0) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax or tmax < 0.0:
            return False, 1e9

    if abs(e1) < 1e-12:
        if d1 < -ey or d1 > ey:
            return False, 1e9
    else:
        inv = 1.0 / e1
        t1 = (-ey - d1) * inv
        t2 = (ey - d1) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax or tmax < 0.0:
            return False, 1e9

    if abs(e2) < 1e-12:
        if d2 < -ez or d2 > ez:
            return False, 1e9
    else:
        inv = 1.0 / e2
        t1 = (-ez - d2) * inv
        t2 = (ez - d2) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax or tmax < 0.0:
            return False, 1e9

    if tmin > 1.0:
        return False, 1e9
    return True, tmin


# ---------------------------------------------------------------------------
# 局部空间碰撞（AGENTS.md 3.4：规避 OBB 开销）
# ---------------------------------------------------------------------------


@njit(cache=True)
def world_to_local(
    x: float, y: float,
    box_cx: float, box_cy: float,
    yaw: float,
) -> tuple[float, float]:
    """把世界坐标转换到装甲局部坐标系（2D 逆旋转，仅 yaw，z 不变）。

    局部坐标系中碰撞箱是 AABB（AGENTS.md 3.4 check_collision）。
    """
    dx = x - box_cx
    dy = y - box_cy
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return local_x, local_y


@njit(cache=True)
def check_collision(
    proj_x: float, proj_y: float,
    armor_x: float, armor_y: float,
    armor_yaw: float,
    armor_half_x: float, armor_half_y: float,
) -> bool:
    """单发弹丸点 vs 装甲 AABB（局部坐标判定，AGENTS.md 3.4 参考实现）。"""
    lx, ly = world_to_local(proj_x, proj_y, armor_x, armor_y, armor_yaw)
    return abs(lx) < armor_half_x and abs(ly) < armor_half_y


# ---------------------------------------------------------------------------
# 批量候选检测（单弹丸 vs 候选装甲数组，供空间哈希 narrow phase 调用）
# ---------------------------------------------------------------------------


@njit(cache=True)
def segment_hit_candidates(
    p0: np.ndarray,            # (3,) 上一帧位置
    p1: np.ndarray,            # (3,) 当前帧位置
    radius: float,             # 弹丸碰撞半径（胶囊体）
    armor_pos: np.ndarray,     # (M, 3) 候选装甲中心
    armor_half: np.ndarray,    # (M, 3) 候选装甲半边长
    armor_active: np.ndarray,  # (M,)
    out_hit: np.ndarray,       # (M,) bool 输出
    out_t: np.ndarray,         # (M,) float 输出（命中参数 t）
) -> None:
    """胶囊体 vs 一批装甲 AABB，一次内核处理单弹丸全部候选。

    Args:
        p0/p1: 线段端点（世界坐标，上一帧/当前帧位置）。
        radius: 弹丸半径，传入 segment_aabb_single 做 AABB 膨胀。
        各 armor_* 数组为候选集（来自空间哈希相邻 9 格）。
        out_hit/out_t: 输出数组，长度须与候选集一致，原地写入。

    Note:
        骨架先用世界坐标 AABB（z 轴不变）做近似；局部 yaw 精确版
        在 Phase 3 正式迭代中接入 world_to_local 后替换。
    """
    m = armor_pos.shape[0]
    for i in range(m):
        if not armor_active[i]:
            continue
        hit, t = segment_aabb_single(
            p0[0], p0[1], p0[2],
            p1[0], p1[1], p1[2],
            armor_pos[i, 0], armor_pos[i, 1], armor_pos[i, 2],
            armor_half[i, 0], armor_half[i, 1], armor_half[i, 2],
            radius,
        )
        out_hit[i] = hit
        out_t[i] = t


# ---------------------------------------------------------------------------
# 全 Numba 空间索引 + 并行检测内核（Phase 3 正式迭代）
#
# AGENTS.md 3.4：无状态空间哈希 + Numba 并行，替代 Python dict broad phase。
# 用密集网格索引（cell_count / cell_start / cell_armor）替代逐弹丸 Python
# 查表——16384 弹丸 × 2048 装甲的单帧检测下沉为一次并行内核调用，
# 性能目标 < 3ms/帧（原 Python dict 版实测 ~186ms，远不达标）。
# ---------------------------------------------------------------------------


@njit(cache=True)
def build_grid_index(
    armor_pos: np.ndarray,       # (N, 3) float32
    armor_half: np.ndarray,      # (N, 3) float32
    armor_active: np.ndarray,    # (N,) bool
    grid_size: float,
    grid_min: float,
    gx_count: int,
    gy_count: int,
    cell_count: np.ndarray,      # (G,) int32 每格候选数（out，先清零再填）
    cell_start: np.ndarray,      # (G+1,) int32 前缀和（out）
    cell_armor: np.ndarray,      # (MAX_CAND,) int32 排序后的装甲索引（out）
) -> None:
    """构建密集网格空间索引（两遍：计数 → 前缀和 → 填充）。

    装甲按中心所在格登记，并按半边长扩散到相邻格（clamp dx/dy ≤ 2，
    避免跨大装甲漏检；cell_armor 容量须 ≥ Σ 扩散格数）。
    """
    n = armor_pos.shape[0]
    total = cell_start.shape[0] - 1
    for c in range(total):
        cell_count[c] = 0

    # 第一遍：每格候选数
    for i in range(n):
        if not armor_active[i]:
            continue
        gx = int((armor_pos[i, 0] - grid_min) / grid_size)
        gy = int((armor_pos[i, 1] - grid_min) / grid_size)
        dx = min(int(np.ceil(armor_half[i, 0] / grid_size)), 2)
        dy = min(int(np.ceil(armor_half[i, 1] / grid_size)), 2)
        for ox in range(-dx, dx + 1):
            for oy in range(-dy, dy + 1):
                cx = gx + ox
                cy = gy + oy
                if 0 <= cx < gx_count and 0 <= cy < gy_count:
                    cell_count[cy * gx_count + cx] += 1

    # 前缀和（格少，串行即可）
    cell_start[0] = 0
    for c in range(1, total + 1):
        cell_start[c] = cell_start[c - 1] + cell_count[c - 1]

    # 第二遍：填充（cell_count 清零复用为写入游标）
    # ⚠️ 修复（2026-08-11）：此前直接复用第一遍统计值当游标 → pos 从
    # start+N 错位写入（越界到下一格），且 cell_count 被污染翻倍 →
    # 同一装甲重复登记（弹丸碰撞下重复检查无害，编队推开下重复推开累积）。
    for c in range(total):
        cell_count[c] = 0
    for i in range(n):
        if not armor_active[i]:
            continue
        gx = int((armor_pos[i, 0] - grid_min) / grid_size)
        gy = int((armor_pos[i, 1] - grid_min) / grid_size)
        dx = min(int(np.ceil(armor_half[i, 0] / grid_size)), 2)
        dy = min(int(np.ceil(armor_half[i, 1] / grid_size)), 2)
        for ox in range(-dx, dx + 1):
            for oy in range(-dy, dy + 1):
                cx = gx + ox
                cy = gy + oy
                if 0 <= cx < gx_count and 0 <= cy < gy_count:
                    cell = cy * gx_count + cx
                    pos = cell_start[cell] + cell_count[cell]
                    cell_armor[pos] = i
                    cell_count[cell] += 1


@njit(cache=True, parallel=True)
def detect_projectiles_batch(
    proj_pos: np.ndarray,        # (P, 3) float32 当前帧
    proj_vel: np.ndarray,        # (P, 3) float32
    proj_radius: np.ndarray,     # (P,) float32
    proj_active: np.ndarray,     # (P,) bool
    armor_pos: np.ndarray,       # (N, 3) float32
    armor_half: np.ndarray,      # (N, 3) float32
    armor_active: np.ndarray,    # (N,) bool
    dt: float,
    grid_size: float,
    grid_min: float,
    gx_count: int,
    gy_count: int,
    cell_start: np.ndarray,      # (G+1,) int32
    cell_count: np.ndarray,      # (G,) int32
    cell_armor: np.ndarray,      # (MAX_CAND,) int32
    out_hit_armor: np.ndarray,   # (P,) int32 命中装甲（-1 未命中）
    out_t: np.ndarray,           # (P,) float32 命中参数
) -> None:
    """并行胶囊体 vs AABB 检测：每弹丸查相邻 9 格候选，取 t 最小者。

    弹丸在网格外时 clamp 到边界格（近似，避免越界）。
    各弹丸写入独立输出槽位 → 无竞争。
    """
    p = proj_pos.shape[0]
    for i in prange(p):
        out_hit_armor[i] = -1
        out_t[i] = 1e9
        if not proj_active[i]:
            continue

        px0 = proj_pos[i, 0] - proj_vel[i, 0] * dt
        py0 = proj_pos[i, 1] - proj_vel[i, 1] * dt
        pz0 = proj_pos[i, 2] - proj_vel[i, 2] * dt

        gx = int((proj_pos[i, 0] - grid_min) / grid_size)
        gy = int((proj_pos[i, 1] - grid_min) / grid_size)
        if gx < 0:
            gx = 0
        elif gx >= gx_count:
            gx = gx_count - 1
        if gy < 0:
            gy = 0
        elif gy >= gy_count:
            gy = gy_count - 1

        best_t = 1e9
        best_a = -1
        radius = proj_radius[i]
        for ox in (-1, 0, 1):
            cx = gx + ox
            if cx < 0 or cx >= gx_count:
                continue
            for oy in (-1, 0, 1):
                cy = gy + oy
                if cy < 0 or cy >= gy_count:
                    continue
                cell = cy * gx_count + cx
                base = cell_start[cell]
                cnt = cell_count[cell]
                for k in range(cnt):
                    a = cell_armor[base + k]
                    if not armor_active[a]:
                        continue
                    hit, t = segment_aabb_single(
                        px0, py0, pz0,
                        proj_pos[i, 0], proj_pos[i, 1], proj_pos[i, 2],
                        armor_pos[a, 0], armor_pos[a, 1], armor_pos[a, 2],
                        armor_half[a, 0], armor_half[a, 1], armor_half[a, 2],
                        radius,
                    )
                    if hit and t < best_t:
                        best_t = t
                        best_a = a
        if best_a >= 0:
            out_hit_armor[i] = best_a
            out_t[i] = best_t


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def detect_proj_aircraft_batch(
    proj_pos: np.ndarray,        # (P, 3) float32 弹丸位置（当前帧）
    proj_vel: np.ndarray,        # (P, 3) float32 弹丸速度（反推上一帧）
    proj_radius: np.ndarray,     # (P,) float32 弹丸碰撞半径
    proj_active: np.ndarray,     # (P,) bool_
    air_pos: np.ndarray,         # (N, 3) float32 飞机机身箱位置（世界坐标，BoxPool sync 后）
    air_half: np.ndarray,        # (N, 3) float32 机身箱半边长
    air_active: np.ndarray,      # (N,) bool_
    dt: float,
    out_hit: np.ndarray,         # (P,) int32 out：命中飞机索引（-1 未命中）
    out_t: np.ndarray,           # (P,) float32 out：命中参数 t
) -> None:
    """弹丸 vs 飞机机身箱批量碰撞（Phase 7，docs §7.4 ①）。

    复用统一「胶囊体 vs AABB」算法（`segment_aabb_single`）：飞机挂的机身箱
    是 AABB（BoxPool parent_kind=1，sync 后世界坐标），弹丸轨迹是胶囊。
    飞机数 ≤128 → 全对全即可（不建空间索引；对比装甲 2048 需哈希）。
    """
    p = proj_pos.shape[0]
    n = air_pos.shape[0]
    for i in prange(p):  # noqa: F821
        if not proj_active[i]:
            continue
        p0x = proj_pos[i, 0] - proj_vel[i, 0] * dt
        p0y = proj_pos[i, 1] - proj_vel[i, 1] * dt
        p0z = proj_pos[i, 2] - proj_vel[i, 2] * dt
        p1x = proj_pos[i, 0]
        p1y = proj_pos[i, 1]
        p1z = proj_pos[i, 2]
        rad = proj_radius[i]
        best_t = 1e9
        best = -1
        for j in range(n):
            if not air_active[j]:
                continue
            hit, t = segment_aabb_single(
                p0x, p0y, p0z, p1x, p1y, p1z,
                air_pos[j, 0], air_pos[j, 1], air_pos[j, 2],
                air_half[j, 0], air_half[j, 1], air_half[j, 2],
                rad,
            )
            if hit and t < best_t:
                best_t = t
                best = j
        out_hit[i] = best
        out_t[i] = best_t


@dataclass(frozen=True)
class CollisionHit:
    """一次命中记录（供后续毁伤解析消费）。"""

    proj_idx: int
    armor_idx: int
    t: float                              # 命中参数 [0,1]（胶囊体 vs AABB）
    hit_x: float
    hit_y: float
    hit_z: float


# ---------------------------------------------------------------------------
# 碰撞检测器（broad phase 空间哈希 + narrow phase Numba 内核）
# ---------------------------------------------------------------------------


class CollisionDetector:
    """碰撞检测器（全 Numba：密集网格空间索引 + 并行胶囊体检测）。

    流程（AGENTS.md 3.4）：
        1. build_index():  Numba 内核构建密集网格索引（cell_count/start/armor）
        2. detect():       Numba 并行内核逐弹丸查相邻 9 格 → 胶囊体 vs AABB

    Note:
        - 已从骨架期「Python dict broad phase」升级为全 Numba 数组索引，
          16384 弹丸 × 2048 装甲单帧 < 3ms（原 dict 版 ~186ms 不达标）
        - 装甲 vs 装甲（Sort & Sweep 编队推开）仍留后续迭代
    """

    # 每装甲最大登记格数（dx/dy clamp ≤ 2 → 最多 5×5=25 格）
    _MAX_SPREAD = 25

    def __init__(
        self,
        grid_size: float = DEFAULT_GRID_SIZE,
        world_half: float = 2500.0,
    ) -> None:
        """初始化。

        Args:
            grid_size: 空间哈希格子尺寸 (m)，AGENTS.md 3.4 = 50.0。
            world_half: 网格覆盖的半边长 (m)，战场 [-world_half, world_half]²。
        """
        self.grid_size = float(grid_size)
        self.world_half = float(world_half)
        span = 2.0 * self.world_half
        self.gx_count = max(1, int(np.ceil(span / self.grid_size)))
        self.gy_count = max(1, int(np.ceil(span / self.grid_size)))
        total = self.gx_count * self.gy_count
        self._cell_count = np.zeros(total, dtype=np.int32)
        self._cell_start = np.zeros(total + 1, dtype=np.int32)
        self._cell_armor = np.zeros(total + self._MAX_SPREAD + 8, dtype=np.int32)
        self._built = False

    def build_index(
        self,
        armor_pos: np.ndarray,
        armor_half: np.ndarray,
        armor_active: np.ndarray,
    ) -> None:
        """重建密集网格空间索引（Numba 内核）。

        Args:
            armor_pos: (N, 3) 装甲位置。
            armor_half: (N, 3) 装甲半边长（跨格扩散，clamp dx/dy ≤ 2）。
            armor_active: (N,) 活跃掩码。
        """
        n = armor_pos.shape[0]
        gs = self.grid_size

        # 按扩散需求分配 cell_armor 容量（Σ 每装甲登记格数）
        active = armor_active.astype(np.bool_)
        if np.any(active):
            dx = np.minimum(
                np.ceil(armor_half[active, 0] / gs).astype(np.int32), 2
            )
            dy = np.minimum(
                np.ceil(armor_half[active, 1] / gs).astype(np.int32), 2
            )
            total_slots = int(np.sum((2 * dx + 1) * (2 * dy + 1)))
        else:
            total_slots = 0

        if self._cell_armor.shape[0] < total_slots:
            self._cell_armor = np.zeros(total_slots + 8, dtype=np.int32)

        build_grid_index(
            armor_pos, armor_half, armor_active,
            gs, -self.world_half,
            self.gx_count, self.gy_count,
            self._cell_count, self._cell_start, self._cell_armor,
        )
        self._built = True

    def detect(
        self,
        proj_pos: np.ndarray,     # (P, 3) 当前帧弹丸位置
        proj_vel: np.ndarray,     # (P, 3) 弹丸速度（用于反推上一帧）
        proj_radius: np.ndarray,  # (P,)
        proj_active: np.ndarray,  # (P,) bool
        armor_pos: np.ndarray,
        armor_half: np.ndarray,
        armor_active: np.ndarray,
        dt: float,
    ) -> list[CollisionHit]:
        """执行整帧碰撞检测，返回全部命中（统一胶囊体算法，Numba 并行）。

        Args:
            proj_*: 抛射物 SoA 数组。
            armor_*: 装甲 SoA 数组。
            dt: 上一帧时长（秒），用于从 vel 反推上一帧位置 p0 = pos - vel*dt。

        Returns:
            命中列表（每颗弹丸最多一次命中，取 t 最小者）。
        """
        if not self._built:
            self.build_index(armor_pos, armor_half, armor_active)

        p = proj_pos.shape[0]
        out_hit_armor = np.full(p, -1, dtype=np.int32)
        out_t = np.full(p, 1e9, dtype=np.float32)

        detect_projectiles_batch(
            proj_pos, proj_vel, proj_radius, proj_active,
            armor_pos, armor_half, armor_active,
            dt,
            self.grid_size, -self.world_half,
            self.gx_count, self.gy_count,
            self._cell_start, self._cell_count, self._cell_armor,
            out_hit_armor, out_t,
        )

        hits: list[CollisionHit] = []
        valid = np.flatnonzero(out_hit_armor >= 0)
        for i in valid:
            t = float(out_t[i])
            p0 = proj_pos[i] - proj_vel[i] * dt
            p1 = proj_pos[i]
            hit_pos = p0 + (p1 - p0) * t
            hits.append(CollisionHit(
                proj_idx=int(i),
                armor_idx=int(out_hit_armor[i]),
                t=t,
                hit_x=float(hit_pos[0]),
                hit_y=float(hit_pos[1]),
                hit_z=float(hit_pos[2]),
            ))
        return hits

    def clear(self) -> None:
        """清空索引（装甲大规模变动时调用）。"""
        self._built = False
        self._cell_count[:] = 0
        self._cell_start[:] = 0


# ---------------------------------------------------------------------------
# 装甲 vs 装甲：空间哈希编队推开（AGENTS.md 3.4，Phase 3 正式迭代）
#
# 需求：防编队单位重叠（队列/纵队行进时车距过近）。每地面 Tick 调用一次。
# 算法：复用 build_grid_index 建密集网格索引 → 对每个活跃装甲查相邻 3×3 格
#       内的候选装甲 → AABB 重叠判定 → 沿重叠最小的轴各推开一半。
#       （与弹丸碰撞检测共享同一空间哈希基础设施，网格 50m。）
# 去重：每对只由「索引较小者」处理（j > i），避免 i↔j 重复。
# 性能：2048 装甲 O(N) 查相邻格，Numba 串行内核 < 1ms（无需 prange，
#       重叠写入需串行避免竞争）。
# ---------------------------------------------------------------------------


@njit(cache=True)
def separate_armors_hash(
    pos: np.ndarray,          # (N, 3) float32 世界坐标（原地修改）
    half: np.ndarray,         # (N, 3) float32 半边长
    active: np.ndarray,       # (N,) bool_ 存活掩码
    grid_size: float,
    grid_min: float,
    gx_count: int,
    gy_count: int,
    cell_start: np.ndarray,   # (G+1,) int32 前缀和
    cell_count: np.ndarray,   # (G,) int32 每格候选数
    cell_armor: np.ndarray,   # (MAX_CAND,) int32 候选装甲索引
    sep_scale: float,         # 推开距离 = 重叠量 × sep_scale（0.5 = 各推一半）
    push_cap: float,          # 单帧最大推开距离（防弹射）
) -> int:
    """空间哈希编队推开：原地修正 pos，返回被推开的装甲对数。

    Args:
        pos: 装甲位置数组，原地修改（推开后写入）。
        half: 装甲半边长。
        active: 存活掩码。
        grid_*: 空间哈希网格参数（与 build_grid_index 一致）。
        cell_*: build_grid_index 输出的密集网格索引。
        sep_scale: 分离系数，0.5 = 重叠双方各退一半。
        push_cap: 单帧单轴最大推开距离（m），防数值弹射。

    Returns:
        发生重叠并推开的装甲对数量（0 = 无重叠，可用于早退）。

    Note:
        - 每对只由索引较小者处理（j > i），避免重复推开。
        - z 轴不判重叠（不同层单位不互相推，桥/隧物理解耦，AGENTS.md 3.2）。
        - 推开量 ≤ 重叠量×sep_scale 且 ≤ push_cap，原地累积。
    """
    n = pos.shape[0]
    # 临时拷贝：推开过程逐对累积，读取原始位置避免级联偏差
    orig_x = np.empty(n, dtype=np.float32)
    orig_y = np.empty(n, dtype=np.float32)
    for i in range(n):
        orig_x[i] = pos[i, 0]
        orig_y[i] = pos[i, 1]

    pairs = 0
    for i in range(n):
        if not active[i]:
            continue
        hxi = half[i, 0]
        hyi = half[i, 1]
        xi = orig_x[i]
        yi = orig_y[i]

        # 所在格（clamp 到边界，与 detect_projectiles_batch 一致）
        gx = int((xi - grid_min) / grid_size)
        gy = int((yi - grid_min) / grid_size)
        if gx < 0:
            gx = 0
        elif gx >= gx_count:
            gx = gx_count - 1
        if gy < 0:
            gy = 0
        elif gy >= gy_count:
            gy = gy_count - 1

        # 查相邻 3×3 格内的候选装甲
        for ox in (-1, 0, 1):
            cx = gx + ox
            if cx < 0 or cx >= gx_count:
                continue
            for oy in (-1, 0, 1):
                cy = gy + oy
                if cy < 0 or cy >= gy_count:
                    continue
                cell = cy * gx_count + cx
                base = cell_start[cell]
                cnt = cell_count[cell]
                for k in range(cnt):
                    j = cell_armor[base + k]
                    if j <= i:            # 每对只由索引较小者处理
                        continue
                    if not active[j]:
                        continue
                    # 去重关键：装甲因扩散登记到多格，这里只在候选 j 的
                    # 「中心格」== 当前格时才处理该对，确保 (i,j) 只推开一次
                    #（否则同一对会被 i 查多个相邻格重复推开，push_cap 失效）
                    if int((orig_x[j] - grid_min) / grid_size) != cx:
                        continue
                    if int((orig_y[j] - grid_min) / grid_size) != cy:
                        continue
                    xj = orig_x[j]
                    yj = orig_y[j]
                    # AABB 重叠判定（水平面）
                    oxx = (hxi + half[j, 0]) - abs(xj - xi)
                    oyy = (hyi + half[j, 1]) - abs(yj - yi)
                    if oxx <= 0.0 or oyy <= 0.0:
                        continue
                    # 沿重叠最小的轴分离（若相等用 x）
                    if oxx <= oyy:
                        push = min(oxx * sep_scale, push_cap)
                        sgn = 1.0 if xj >= xi else -1.0
                        pos[i, 0] -= sgn * push
                        pos[j, 0] += sgn * push
                    else:
                        push = min(oyy * sep_scale, push_cap)
                        sgn = 1.0 if yj >= yi else -1.0
                        pos[i, 1] -= sgn * push
                        pos[j, 1] += sgn * push
                    pairs += 1
    return pairs


class FormationResolver:
    """编队推开服务：空间哈希 + 装甲对重叠推开（AGENTS.md 3.4）。

    与弹丸碰撞检测共享 build_grid_index 空间哈希基础设施，
    每地面 Tick 调用一次：:

        res = FormationResolver()
        pairs = res.resolve(armor.pos, armor.half, armor.active)
        # pairs > 0 表示有编队重叠被推开

    Note:
        - 复用 build_grid_index 建索引（cell_armor 容量按扩散需求分配）。
        - z 轴不判重叠（不同层单位不互相推，桥/隧物理解耦，AGENTS.md 3.2）。
        - 编队推开应在模拟 Tick 间隙调用（与移动积分串行），避免与
          Numba 并行移动循环竞争写入。
    """

    # 每装甲最大登记格数（dx/dy clamp ≤ 2 → 最多 5×5=25 格）
    _MAX_SPREAD = 25

    def __init__(
        self,
        grid_size: float = DEFAULT_GRID_SIZE,
        world_half: float = 2500.0,
        sep_scale: float = 0.5,
        push_cap: float = 2.0,
    ) -> None:
        """初始化。

        Args:
            grid_size: 空间哈希格子尺寸 (m)，AGENTS.md 3.4 = 50.0。
            world_half: 网格覆盖的半边长 (m)，战场 [-world_half, world_half]²。
            sep_scale: 分离系数，0.5 = 重叠双方各退一半。
            push_cap: 单帧单轴最大推开距离 (m)。
        """
        self.grid_size = float(grid_size)
        self.world_half = float(world_half)
        self.sep_scale = float(sep_scale)
        self.push_cap = float(push_cap)
        span = 2.0 * self.world_half
        self.gx_count = max(1, int(np.ceil(span / self.grid_size)))
        self.gy_count = max(1, int(np.ceil(span / self.grid_size)))
        total = self.gx_count * self.gy_count
        self._cell_count = np.zeros(total, dtype=np.int32)
        self._cell_start = np.zeros(total + 1, dtype=np.int32)
        self._cell_armor = np.zeros(total + self._MAX_SPREAD + 8, dtype=np.int32)
        self._last_pairs = 0

    @property
    def last_pairs(self) -> int:
        """最近一次调用推开的装甲对数（0 = 无重叠）。"""
        return self._last_pairs

    def resolve(
        self,
        pos: np.ndarray,
        half: np.ndarray,
        active: np.ndarray,
    ) -> int:
        """执行编队推开，原地修改 pos，返回推开的装甲对数。"""
        n = pos.shape[0]
        if n == 0:
            self._last_pairs = 0
            return 0

        # 按扩散需求分配 cell_armor 容量（与 CollisionDetector.build_index 一致）
        a = active.astype(np.bool_)
        if np.any(a):
            dx = np.minimum(np.ceil(half[a, 0] / self.grid_size).astype(np.int32), 2)
            dy = np.minimum(np.ceil(half[a, 1] / self.grid_size).astype(np.int32), 2)
            total_slots = int(np.sum((2 * dx + 1) * (2 * dy + 1)))
        else:
            total_slots = 0
        if self._cell_armor.shape[0] < total_slots:
            self._cell_armor = np.zeros(total_slots + 8, dtype=np.int32)

        # 构建空间索引（复用弹丸碰撞的 build_grid_index 内核）
        build_grid_index(
            pos, half, active,
            self.grid_size, -self.world_half,
            self.gx_count, self.gy_count,
            self._cell_count, self._cell_start, self._cell_armor,
        )

        self._last_pairs = int(separate_armors_hash(
            pos, half, active,
            self.grid_size, -self.world_half,
            self.gx_count, self.gy_count,
            self._cell_start, self._cell_count, self._cell_armor,
            self.sep_scale, self.push_cap,
        ))
        return self._last_pairs
