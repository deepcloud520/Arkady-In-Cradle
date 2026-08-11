"""轨道相机：透视投影 + 自由旋转/缩放（render 层，官方实现）。

基于 pyglm（GLM 移植），矩阵为 column-major 约定，与 OpenGL/moderngl
的 uniform 布局天然一致（无需转置）。视图/投影矩阵与逻辑层零耦合
（AGENTS.md 3.6 节），完全由 UI 事件驱动。
坐标系约定：Z 轴向上（ENU），地图位于 XY 平面。
"""
from __future__ import annotations

import glm


class OrbitCamera:
    """围绕目标点旋转的轨道相机（支持鼠标拖拽与滚轮缩放）。"""

    def __init__(
        self,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 300.0,
        yaw: float = 45.0,
        pitch: float = 35.0,
        fov: float = 50.0,
        near: float = 0.5,
        far: float = 20000.0,
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

