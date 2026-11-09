"""局部切平面坐标系（LocalFrame）与地球曲率修正。

对应 docs/curvature-3d-worldmesh-design.md §2 / §3：

- **LocalFrame 是唯一存储坐标**：锚点经纬高 + 锚点在地图内的位置 `(x0, y0)`；
  单位坐标 `(frame_id, x, y, z)` 全部是本地米（float32）。
- 局部 `z` 是“相对锚点切平面”的高度；海平面在远处按 `-d²/(2R)` 下坠。
- 曲率按 **A 方案** 只作为 WorldMesh 顶点变换，`spans` 保持平面
  （`z_span` 只给地形破坏；`z_wm` 给检测/移动/渲染/火控）。

命名纪律（设计文档 §2.3）：

    z_span  —— 平面高度（Heightfield 层高）
    z_wm    —— 弯曲后的世界高度（WorldMesh 顶点）
    msl     —— 正高（仅 UI/简报显示）

本模块只做几何换算，不读写任何池/地形对象（`bend_patches` 仅按鸭子类型
修改 patch 顶点，供 terrain 层在构建期调用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# 地球半径 / 自转角速度（设计文档 §3 / §8）
R_EARTH: float = 6_371_000.0
OMEGA_EARTH: float = 7.2921159e-5

# 单锚点适用范围上限（d²/2R ≤ 1km 且 atan(d/R) ≤ 1° → ≈110km，取整 100km）
MAX_SINGLE_FRAME_KM: float = 100.0


@dataclass(frozen=True)
class LocalFrame:
    """局部切平面坐标系（锚点 + 地图内位置）。

    Attributes:
        frame_id: 帧编号（单帧地图恒 0；800×100 分块时每块一个）。
        lat0 / lon0: 锚点纬经度（°，全球身份）。
        alt0: 锚点正高（m，仅 ECEF 换算用）。
        x0 / y0: 锚点在地图坐标中的位置（m）；锚点为原点时恒 0。
        heading: 局部 +X 相对正东的旋转（°，可选）。
        radius: 地球半径（m，测试可注入小半径验证公式）。
    """

    frame_id: int = 0
    lat0: float = 0.0
    lon0: float = 0.0
    alt0: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    heading: float = 0.0
    radius: float = R_EARTH

    # ------------------------------------------------------------------
    # 曲率
    # ------------------------------------------------------------------

    def drop(self, x: float, y: float) -> float:
        """相对锚点的曲率下坠（m）：`d²/(2R)`（抛物近似）。"""
        dx = float(x) - self.x0
        dy = float(y) - self.y0
        return (dx * dx + dy * dy) / (2.0 * self.radius)

    def wm_z(self, z_span: float, x: float, y: float) -> float:
        """平面高度 → 弯曲世界高度（WM 顶点用）。"""
        return float(z_span) - self.drop(x, y)

    def span_z(self, z_wm: float, x: float, y: float) -> float:
        """弯曲世界高度 → 平面高度（破坏 API 用）。"""
        return float(z_wm) + self.drop(x, y)

    def msl(self, z_wm: float, x: float, y: float) -> float:
        """弯曲世界高度 → 正高（仅 UI/简报显示用）。"""
        return float(z_wm) + self.drop(x, y)

    def earth_center(self) -> tuple[float, float, float]:
        """地心在地图坐标中的位置：锚点正下方 R。

        局部 z 以“锚点切平面处的海平面”为 0，因此地心 z = -R（与 `alt0` 无关；
        `alt0` 只参与 ECEF 换算）。
        """
        return (self.x0, self.y0, -self.radius)

    def omega(self) -> tuple[float, float, float]:
        """自转角速度在局部 ENU 下的分量（东, 北, 天）。"""
        phi = np.radians(self.lat0)
        return (0.0, OMEGA_EARTH * float(np.cos(phi)), OMEGA_EARTH * float(np.sin(phi)))

    # ------------------------------------------------------------------
    # ECEF（只做换算，不落库）
    # ------------------------------------------------------------------

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """局部 ENU 三轴在 ECEF 下的方向（含 heading 旋转）。"""
        phi = float(np.radians(self.lat0))
        lam = float(np.radians(self.lon0))
        east = np.array([-np.sin(lam), np.cos(lam), 0.0], dtype=np.float64)
        north = np.array(
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            dtype=np.float64,
        )
        up = np.array(
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
            dtype=np.float64,
        )
        if self.heading:
            h = float(np.radians(self.heading))
            c, s = float(np.cos(h)), float(np.sin(h))
            east, north = c * east - s * north, s * east + c * north
        return east, north, up

    def _anchor_ecef(self) -> np.ndarray:
        _, _, up = self._basis()
        return (self.radius + self.alt0) * up

    def to_ecef(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """局部坐标 → ECEF（球面近似；跨帧/跨图查询用）。"""
        east, north, up = self._basis()
        p = self._anchor_ecef() + float(x) * east + float(y) * north + float(z) * up
        return (float(p[0]), float(p[1]), float(p[2]))

    def from_ecef(self, ex: float, ey: float, ez: float) -> tuple[float, float, float]:
        """ECEF → 局部坐标（`to_ecef` 的逆）。"""
        east, north, up = self._basis()
        d = np.array([float(ex), float(ey), float(ez)], dtype=np.float64) - self._anchor_ecef()
        return (float(d @ east), float(d @ north), float(d @ up))

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def validate(self, width_m: float, height_m: float) -> None:
        """单帧适用范围校验：对角线超过 `MAX_SINGLE_FRAME_KM` 直接拒绝。

        设计文档 §0：单锚点适用 ≤100 km；更大必须分块。此处在加载期快速失败，
        避免“跑到一半才发现落点偏了几百米”。
        """
        diag_km = float(np.hypot(width_m, height_m)) / 1000.0
        if diag_km > MAX_SINGLE_FRAME_KM:
            raise ValueError(
                f"LocalFrame 单帧对角线 {diag_km:.1f} km > {MAX_SINGLE_FRAME_KM:.0f} km；"
                "请改用分块（每块 ≤100 km），见 docs/curvature-3d-worldmesh-design.md §1"
            )


# ---------------------------------------------------------------------------
# 构建期：把曲率施加到 WorldPatch 顶点
# ---------------------------------------------------------------------------


def _recompute_patch_geometry(patch: Any) -> None:
    """重算 patch 的法线 / 面积 / 形心（Newell's method）。

    `WorldPatch.__post_init__` 只在构造时算一次；压弯后顶点 z 变了，
    法线/面积/形心必须同步，否则 `ray_plane_z` 与渲染法线会偏。
    """
    v = patch.vertices
    n = len(v)
    if n < 3:
        return
    nx = ny = nz = 0.0
    for i in range(n):
        x1, y1, z1 = (float(v[i, 0]), float(v[i, 1]), float(v[i, 2]))
        x2, y2, z2 = (float(v[(i + 1) % n, 0]), float(v[(i + 1) % n, 1]), float(v[(i + 1) % n, 2]))
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    norm = float(np.sqrt(nx * nx + ny * ny + nz * nz))
    if norm > 1e-12:
        patch.normal = np.array([nx / norm, ny / norm, nz / norm], dtype=np.float32)
        patch.area = 0.5 * norm
    patch.centroid = v.mean(axis=0).astype(np.float32)


def bend_patches(patches: Iterable[Any], frame: LocalFrame | None) -> None:
    """把曲率下坠施加到一批 WorldPatch 顶点（原地修改）。

    `frame is None` 时不变（保持旧行为 / 测试基线）。

    Args:
        patches: WorldPatch 序列（只要求有 `.vertices` / `.normal` /
            `.area` / `.centroid`，避免 terrain ↔ coordinates 循环导入）。
        frame: 局部帧；None 表示不做曲率。
    """
    if frame is None:
        return
    r2 = 2.0 * frame.radius
    x0, y0 = frame.x0, frame.y0
    for patch in patches:
        v = patch.vertices
        if v is None or len(v) < 3:
            continue
        dx = v[:, 0].astype(np.float64) - x0
        dy = v[:, 1].astype(np.float64) - y0
        v[:, 2] = (v[:, 2].astype(np.float64) - (dx * dx + dy * dy) / r2).astype(np.float32)
        _recompute_patch_geometry(patch)
