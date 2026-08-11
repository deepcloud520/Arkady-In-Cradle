"""DDA 视线检测：射线 vs 高度场（AGENTS.md Phase 2）。

对应 AGENTS.md 3.2「视线检测，DDA 射线 + 高度图」。
采用标准 3D DDA (Amanatides & Woo) 算法遍历栅格，对每格查询 Heightfield 的
实心段是否阻挡射线。

性能：单次 DDA 遍历 < 1μs（纯 Numba），批量 2048 对检测 < 2ms。
"""

from __future__ import annotations

import numpy as np
from numba import njit

from Airis.lulilo.terrain.heightfield import NO_SURFACE, _ZMAX, is_solid_at as _is_solid_at


@njit(cache=True)
def line_of_sight(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    spans: np.ndarray,      # (N_COLS, MAX_SPANS, 4) uint16
    count: np.ndarray,      # (N_COLS,) uint8
    hf_width: int,
    cs: float,
    ch: float,
    z_min: float,
    max_range: float = 5000.0,
) -> bool:
    """DDA 视线检测：从 (x0,y0,z0) 到 (x1,y1,z1) 是否无阻挡。

    沿射线遍历栅格列，对每列查询实心段是否在射线高度处阻挡。

    Args:
        x0, y0, z0: 观察者位置（世界坐标）。
        x1, y1, z1: 目标位置（世界坐标）。
        spans: Heightfield.spans 数组。
        count: Heightfield.count 数组。
        hf_width: Heightfield.width。
        cs: 栅格单元尺寸 (m/格)。
        ch: 高度量化步长 (m/层)。
        z_min: 高度场最低海拔。
        max_range: 最大检测距离（超过此距离直接返回 False）。

    Returns:
        True 如果视线清晰（无阻挡），False 如果有阻挡或超距。
    """
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    ray_len = np.sqrt(dx * dx + dy * dy + dz * dz)

    if ray_len < 1e-6:
        return True  # 同一点
    if ray_len > max_range:
        return False

    # 单位方向向量
    rdx = dx / ray_len
    rdy = dy / ray_len
    rdz = dz / ray_len

    # 起始栅格
    gx = int(np.floor(x0 / cs))
    gy = int(np.floor(y0 / cs))

    # 步进方向
    step_x = 1 if rdx >= 0 else -1
    step_y = 1 if rdy >= 0 else -1

    # 到下一个格边界的距离（参数 t 单位）
    if abs(rdx) < 1e-10:
        t_delta_x = 1e30
        t_max_x = 1e30
    else:
        t_delta_x = abs(cs / rdx)
        if rdx > 0:
            t_max_x = ((gx + 1) * cs - x0) / rdx
        else:
            t_max_x = (gx * cs - x0) / rdx

    if abs(rdy) < 1e-10:
        t_delta_y = 1e30
        t_max_y = 1e30
    else:
        t_delta_y = abs(cs / rdy)
        if rdy > 0:
            t_max_y = ((gy + 1) * cs - y0) / rdy
        else:
            t_max_y = (gy * cs - y0) / rdy

    # 遍历栅格
    t = 0.0
    prev_gx = gx
    prev_gy = gy

    # 退化射线（纯垂直/近乎垂直）：按高度步进，不走 DDA 格边界
    degenerate = (t_delta_x >= 1e29 and t_delta_y >= 1e29)

    while t < ray_len:
        # 在当前参数 t 处的高度
        z = z0 + rdz * t

        # 检查当前格（边界内）
        if 0 <= gx < hf_width and 0 <= gy < (count.shape[0] // hf_width):
            col = gx + gy * hf_width
            # is_solid_at 期望层索引，需将世界 z 转为层
            layer = int((z - z_min) / ch)
            if _is_solid_at(spans, count, col, float(layer)):
                return False  # 射线被实心段阻挡

        # DDA 步进
        if degenerate:
            t += ch / max(abs(rdz), 1e-6)
        elif t_max_x < t_max_y:
            gx += step_x
            t = t_max_x
            t_max_x += t_delta_x
        else:
            gy += step_y
            t = t_max_y
            t_max_y += t_delta_y

        # 防止死循环（相邻格反复横跳）
        if gx == prev_gx and gy == prev_gy and not degenerate:
            t += cs / max(abs(rdx), abs(rdy), 1e-6)

        prev_gx = gx
        prev_gy = gy

    return True  # 到达终点，无阻挡


@njit(cache=True)
def line_of_sight_batch(
    ox: np.ndarray,         # (N,) float32  观察者 x
    oy: np.ndarray,         # (N,) float32  观察者 y
    oz: np.ndarray,         # (N,) float32  观察者 z
    tx: np.ndarray,         # (N,) float32  目标 x
    ty: np.ndarray,         # (N,) float32  目标 y
    tz: np.ndarray,         # (N,) float32  目标 z
    active: np.ndarray,     # (N,) bool_    哪些对有效
    spans: np.ndarray,
    count: np.ndarray,
    hf_width: int,
    cs: float,
    ch: float,
    z_min: float,
    max_range: float = 5000.0,
) -> np.ndarray:
    """批量 DDA 视线检测。

    Returns:
        (N,) bool_ 数组，True 表示视线清晰。
    """
    n = ox.shape[0]
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if not active[i]:
            continue
        result[i] = line_of_sight(
            ox[i], oy[i], oz[i],
            tx[i], ty[i], tz[i],
            spans, count, hf_width, cs, ch, z_min, max_range,
        )
    return result


@njit(cache=True)
def terrain_intercept(
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    max_t: float,
    spans: np.ndarray,
    count: np.ndarray,
    hf_width: int,
    cs: float,
    ch: float,
    z_min: float,
) -> float:
    """沿射线找到第一个与高度场实心段相交的参数 t。

    用于弹道终点计算（子弹/导弹命中地面或建筑）。

    Args:
        x0, y0, z0: 射线起点。
        dx, dy, dz: 方向向量（不需要归一化）。
        max_t: 最大搜索距离（沿方向向量的参数）。
        spans, count, hf_width, cs, ch, z_min: 高度场参数。

    Returns:
        命中时的参数 t（在 [0, max_t] 之间），或 max_t + 1 表示未命中。
    """
    ray_len = np.sqrt(dx * dx + dy * dy + dz * dz)
    if ray_len < 1e-10:
        return max_t + 1.0

    rdx = dx / ray_len
    rdy = dy / ray_len
    rdz = dz / ray_len

    gx = int(np.floor(x0 / cs))
    gy = int(np.floor(y0 / cs))

    step_x = 1 if rdx >= 0 else -1
    step_y = 1 if rdy >= 0 else -1

    if abs(rdx) < 1e-10:
        t_delta_x = 1e30
        t_max_x = 1e30
    else:
        t_delta_x = abs(cs / rdx)
        if rdx > 0:
            t_max_x = ((gx + 1) * cs - x0) / rdx
        else:
            t_max_x = (gx * cs - x0) / rdx

    if abs(rdy) < 1e-10:
        t_delta_y = 1e30
        t_max_y = 1e30
    else:
        t_delta_y = abs(cs / rdy)
        if rdy > 0:
            t_max_y = ((gy + 1) * cs - y0) / rdy
        else:
            t_max_y = (gy * cs - y0) / rdy

    t = 0.0
    prev_gx = gx
    prev_gy = gy

    # 退化射线（纯垂直/近乎垂直）：按高度步进，不走 DDA 格边界
    degenerate = (t_delta_x >= 1e29 and t_delta_y >= 1e29)

    while t < max_t:
        z = z0 + rdz * t

        if 0 <= gx < hf_width and 0 <= gy < (count.shape[0] // hf_width):
            col = gx + gy * hf_width
            layer = int((z - z_min) / ch)
            if _is_solid_at(spans, count, col, float(layer)):
                return t

        if degenerate:
            # 纯垂直射线：按 grid 高度步进
            t += ch / max(abs(rdz), 1e-6)
        elif t_max_x < t_max_y:
            gx += step_x
            t = t_max_x
            t_max_x += t_delta_x
        else:
            gy += step_y
            t = t_max_y
            t_max_y += t_delta_y

        if gx == prev_gx and gy == prev_gy and not degenerate:
            t += cs / max(abs(rdx), abs(rdy), 1e-6)
        prev_gx = gx
        prev_gy = gy

    return max_t + 1.0


# ---------------------------------------------------------------------------
# Python 封装（持有 Heightfield 引用，提供友好 API）
# ---------------------------------------------------------------------------


class LosChecker:
    """视线检测器：持有 Heightfield 引用，提供类型安全的 Python API。

    用法::

        hf = Heightfield(...)
        los = LosChecker(hf)
        can_see = los.check(ox, oy, oz, tx, ty, tz)
    """

    def __init__(self, hf, max_range: float = 5000.0) -> None:
        """Args:
            hf: Heightfield 实例。
            max_range: 最大检测距离。
        """
        self._hf = hf
        self._max_range = float(max_range)

    def check(
        self,
        ox: float, oy: float, oz: float,
        tx: float, ty: float, tz: float,
    ) -> bool:
        """单次视线检测。"""
        return line_of_sight(
            ox, oy, oz, tx, ty, tz,
            self._hf.spans, self._hf.count,
            self._hf.width, self._hf.cs, self._hf.ch, self._hf.z_min,
            self._max_range,
        )

    def check_batch(
        self,
        ox: np.ndarray, oy: np.ndarray, oz: np.ndarray,
        tx: np.ndarray, ty: np.ndarray, tz: np.ndarray,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        """批量视线检测。"""
        if active is None:
            active = np.ones(ox.shape[0], dtype=np.bool_)
        return line_of_sight_batch(
            ox, oy, oz, tx, ty, tz, active,
            self._hf.spans, self._hf.count,
            self._hf.width, self._hf.cs, self._hf.ch, self._hf.z_min,
            self._max_range,
        )

    def find_intercept(
        self,
        x0: float, y0: float, z0: float,
        dx: float, dy: float, dz: float,
        max_t: float,
    ) -> float:
        """弹道拦截点。"""
        return terrain_intercept(
            x0, y0, z0, dx, dy, dz, max_t,
            self._hf.spans, self._hf.count,
            self._hf.width, self._hf.cs, self._hf.ch, self._hf.z_min,
        )
