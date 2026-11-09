"""轨道相机：透视投影 + 自由旋转/缩放（render 层，官方实现）。

基于 pyglm（GLM 移植），矩阵为 column-major 约定，与 OpenGL/moderngl
的 uniform 布局天然一致（无需转置）。视图/投影矩阵与逻辑层零耦合
（AGENTS.md 3.6 节），完全由 UI 事件驱动。
坐标系约定：Z 轴向上（ENU），地图位于 XY 平面。
"""
from __future__ import annotations

import math

import glm


def far_for_extent(width_m: float, height_m: float, min_far: float = 12000.0) -> float:
    """按地图尺寸推算相机远裁剪面（2026-11-03 修复）。

    背景：15km×15km 演示地图的默认相机距离为 18000m，而 ``far`` 固定 12000m，
    导致地图中心与大部分地形落在远裁剪面之外（``ndc_z`` 被钳到 1.0、
    ``visible=False``），开局画面近乎空白（实测中心点 w=18000 > far）。
    Web/UI 改为「地图铺满」后此问题会被放大，故按对角线放大远裁剪面。

    Args:
        width_m / height_m: 地图世界尺寸（米）。
        min_far: 下限（保持旧行为，小地图不缩水）。

    Returns:
        远裁剪面距离（米），≥ ``3 × 对角线`` 且 ≥ ``min_far``。
    """
    diag = math.hypot(max(float(width_m), 1.0), max(float(height_m), 1.0))
    return max(float(min_far), diag * 3.0)


class OrbitCamera:
    """围绕目标点旋转的轨道相机（支持鼠标拖拽与滚轮缩放）。"""

    def __init__(
        self,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 300.0,
        yaw: float = 45.0,
        pitch: float = 35.0,
        fov: float = 50.0,
        # 深度精度优化（2026-08-11）：
        # - near 0.5→3.0：24 位定点深度在 near=0.5/far=20000 下 1500m 处分辨率仅 27cm →
        #   远处地形 Z-fighting；near=3 后分辨率提升约 30 倍（且 near<相机最近距离 5m 不裁剪）
        # - far 必须 > zoom 上限 10000（否则相机拉远时整场被 far 裁剪 → 地图消失）
        #   20000→12000 已足够；32 位深度缓冲下 12000m 处精度仍达 mm 级
        # 保持标准透视投影 → 景深（DoF）还原线性深度仅需一行公式，最友好
        near: float = 3.0,
        far: float = 12000.0,
    ) -> None:
        self.target = glm.vec3(*target)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)

    def eye(self) -> glm.vec3:
        """相机位置（球坐标 → 笛卡尔）。"""
        yr = glm.radians(self.yaw)
        pr = glm.radians(self.pitch)
        cp = glm.cos(pr)
        return self.target + glm.vec3(
            self.distance * glm.cos(yr) * cp,
            self.distance * glm.sin(yr) * cp,
            self.distance * glm.sin(pr),
        )

    def orbit(self, dx: float, dy: float) -> None:
        """鼠标拖拽：dx 水平旋转 yaw，dy 改变俯仰 pitch。"""
        self.yaw += dx * 0.3
        self.pitch = min(89.0, max(-89.0, self.pitch + dy * 0.3))

    def zoom(self, factor: float) -> None:
        """滚轮缩放：factor > 1 拉近，< 1 拉远。"""
        self.distance = min(10000.0, max(5.0, self.distance * factor))

    def view_matrix(self) -> glm.mat4:
        """GLM lookAt 视图矩阵（column-major）。"""
        up = glm.vec3(0.0, 0.0, 1.0)
        return glm.lookAt(self.eye(), self.target, up)

    def projection_matrix(self, aspect: float) -> glm.mat4:
        """GLM 透视投影矩阵（column-major）。"""
        return glm.perspective(
            glm.radians(self.fov), max(aspect, 1e-6), self.near, self.far
        )

    def mvp_matrix(self, aspect: float) -> glm.mat4:
        """组合后的 MVP 矩阵（column-major，可直接 .tobytes() 写 uniform）。"""
        return self.projection_matrix(aspect) * self.view_matrix()

