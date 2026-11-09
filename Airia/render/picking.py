"""屏幕坐标 → 单位拾取（Phase 5c 新增）。

对应 docs/phase5-ui-pipeline-design.md §4.4「拾取交互」：
    - 单位是 2D 屏幕标记（非 3D 模型），拾取用「标记屏幕距离最近」判定
    - O(N) 遍历 2048 单位，每帧 < 1ms，无需 GPU 拾取缓冲
    - 与相机拖拽通过「按下位置 + 位移阈值」区分（GLWidget 负责）

坐标系约定：
    - 屏幕坐标：左上角 (0,0)，x 向右、y 向下（Qt / 鼠标事件一致）
    - NDC：OpenGL 裁剪空间 [-1,1]^3（z 为深度）
"""
from __future__ import annotations

import logging

import glm
import numpy as np

from Airia.render.adapter import AIR_HANDLE_OFFSET, WorldAdapter

logger = logging.getLogger(__name__)


def project_world(
    world: np.ndarray,
    mvp: glm.mat4,
    viewport_w: int,
    viewport_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把 (N,3) 世界坐标投影到屏幕像素坐标。

    Args:
        world: (N,3) float32 世界坐标。
        mvp: 相机 MVP 矩阵（column-major，与写入 uniform 的矩阵一致）。
        viewport_w / viewport_h: 视口像素尺寸。

    Returns:
        (screen, ndc_z, visible)：
            screen  (N,2) float32 屏幕像素 (x, y)，y 向下
            ndc_z   (N,)  float32 裁剪空间深度
            clip_w  (N,)  float32 裁剪 w 分量（<0 = 在相机后方）
            visible (N,)  bool    是否在裁剪体内（z ∈ [-1,1] 且 w > 0）
    """
    n = world.shape[0]
    if n == 0:
        empty2 = np.zeros((0, 2), dtype=np.float32)
        empty1 = np.zeros(0, dtype=np.float32)
        emptyb = np.zeros(0, dtype=np.bool_)
        return empty2, empty1, empty1, emptyb

    # glm 矩阵 → numpy 数学矩阵 M（M[row, col]）。
    # 注意：不能用 mvp.to_list() —— PyGLM 的 to_list() 返回 4 个列向量，
    # np.array 后已是 (4,4)，再 reshape(4,4, order='F') 是 no-op 返回原数组
    # （行 = 数学列），导致矩阵转置错误（2026-08-13 排查：clip_w 差 500 倍，
    # 全部单位投影挤到同一点）。改用 to_bytes()（与 GPU uniform 完全一致的
    # column-major 扁平），已验证与 glm 直接乘法逐元素一致。
    m_col = np.frombuffer(mvp.to_bytes(), dtype=np.float32).reshape(4, 4, order="F")

    # 齐次坐标 (N,4)
    h = np.empty((n, 4), dtype=np.float32)
    h[:, 0] = world[:, 0]
    h[:, 1] = world[:, 1]
    h[:, 2] = world[:, 2]
    h[:, 3] = 1.0

    # clip = M @ v  →  每行 = (M @ v_i)^T = v_i^T @ M^T
    clip = h @ m_col.T  # (N,4)
    clip_w = clip[:, 3]
    w_safe = np.where(np.abs(clip_w) < 1e-9, np.copysign(1e-9, clip_w), clip_w)
    ndc = clip[:, :3] / w_safe[:, None]  # (N,3)

    sx = (ndc[:, 0] * 0.5 + 0.5) * float(viewport_w)
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(viewport_h)
    screen = np.column_stack([sx, sy]).astype(np.float32)

    visible = (clip_w > 0) & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    return screen, ndc[:, 2].astype(np.float32), clip_w, visible


def pick_nearest(
    screen_x: float,
    screen_y: float,
    adapter: WorldAdapter,
    mvp: glm.mat4,
    viewport_w: int,
    viewport_h: int,
    max_px: float = 30.0,
) -> int:
    """拾取屏幕坐标附近最近的活跃单位（地面装甲 + 空中单位双池）。

    Args:
        screen_x / screen_y: 鼠标屏幕坐标（像素）。
        adapter: WorldAdapter（只读消费，不直读 Real SoA）。
        mvp: 相机 MVP 矩阵。
        viewport_w / viewport_h: 视口像素尺寸。
        max_px: 命中判定半径（像素），超出则视为空白。

    Returns:
        选择句柄 handle：
            - 地面装甲 = 装甲索引（与旧行为一致，向后兼容）；
            - 空中单位 = AIR_HANDLE_OFFSET + aircraft_idx（decode_handle 解码）；
            - 无命中返回 -1。
    """
    if adapter is None:
        return -1
    best_handle = -1
    best_dist = float(max_px)  # 初始 = 命中半径；只有更近的单位才覆盖

    # ---- 地面装甲 ----
    if adapter.armor_active is not None and int(np.count_nonzero(adapter.armor_active)) > 0:
        idx = np.flatnonzero(adapter.armor_active)
        pos = adapter.armor_pos[idx]
        screen, _ndc_z, _clip_w, visible = project_world(
            pos, mvp, viewport_w, viewport_h
        )
        dist = np.hypot(screen[:, 0] - screen_x, screen[:, 1] - screen_y)
        dist = np.where(visible, dist, np.inf)
        j = int(np.argmin(dist))
        if np.isfinite(dist[j]) and dist[j] < best_dist:
            best_dist = dist[j]
            best_handle = int(idx[j])

    # ---- 空中单位（2026-08-18 修复：此前拾取只查 armor，飞机无法选中）----
    ac_active = getattr(adapter, "aircraft_active", None)
    if ac_active is not None and int(np.count_nonzero(ac_active)) > 0:
        aidx = np.flatnonzero(ac_active)
        apos = adapter.aircraft_pos[aidx]
        screen, _ndc_z, _clip_w, visible = project_world(
            apos, mvp, viewport_w, viewport_h
        )
        dist = np.hypot(screen[:, 0] - screen_x, screen[:, 1] - screen_y)
        dist = np.where(visible, dist, np.inf)
        j = int(np.argmin(dist))
        if np.isfinite(dist[j]) and dist[j] < best_dist:
            best_dist = dist[j]
            best_handle = AIR_HANDLE_OFFSET + int(aidx[j])

    return best_handle


class Picker:
    """拾取服务（Phase 5c）：屏幕坐标 → 单位索引。

    无状态；GLWidget 在鼠标移动（hover）与点击（select）时调用。
    """

    def __init__(self, max_px: float = 30.0) -> None:
        self.max_px = float(max_px)

    def pick(
        self,
        screen_x: float,
        screen_y: float,
        adapter: WorldAdapter | None,
        mvp: glm.mat4,
        viewport_w: int,
        viewport_h: int,
    ) -> int:
        """见 ``pick_nearest``。adapter 为 None 时返回 -1。"""
        if adapter is None:
            return -1
        return pick_nearest(
            screen_x, screen_y, adapter, mvp, viewport_w, viewport_h, self.max_px
        )
