"""高度图：fBm 生成器（AGENTS.md 3.2，Heightfield 的连续地表输入）。

依赖方向：coordinates → terrain（本模块仅依赖 numpy）。
- 2026-08-10 清理：移除 bilinear_sample（旧 sampling.py 专用，随旧查询接口废弃）。
- 2026-08-11 修复「地形尖锐」：白噪声 lattice 改用 smoothstep 插值上采样，
  消除 lattice 格点处双线性折痕（旧 scipy.ndimage.zoom order=1 导数不连续）。
"""
from __future__ import annotations

import numpy as np


def generate_fbm(
    width: int,
    height: int,
    amplitude: float,
    noise_scale: float,
    octaves: int,
    seed: int,
) -> np.ndarray:
    """分形布朗运动（fBm）value noise 高度图（smoothstep 插值，无折痕）。

    相比旧版（scipy.ndimage.zoom order=1 双线性放大白噪声），改用
    smoothstep（3t²-2t³）双线性插值：噪声在 lattice 格点处 C1 连续，
    消除尖锐折痕/尖峰（2026-08-11 修复「地形尖锐」）。高频八度 lattice
    下限 min_lattice=4，避免格点过稀疏产生全图锯齿。

    Args:
        width: 栅格宽（格）。
        height: 栅格高（格）。
        amplitude: 首八度振幅（m）。
        noise_scale: 噪声频率尺度（格）。
        octaves: 八度数。
        seed: 随机种子。

    Returns:
        (height, width) float32 高度图数组。
    """
    rng = np.random.default_rng(seed)
    heightmap = np.zeros((height, width), dtype=np.float32)
    amp = amplitude
    freq = 1.0
    min_lattice = 4  # 每八度最小 lattice 格数（防稀疏格点锯齿）
    for _ in range(octaves):
        gw = max(min_lattice, int(width / (noise_scale * freq)))
        gh = max(min_lattice, int(height / (noise_scale * freq)))
        lattice = (rng.random((gh, gw)) * 2.0 - 1.0).astype(np.float32)
        heightmap += _upsample_smoothstep(lattice, height, width) * amp
        amp *= 0.5
        freq *= 2.0
    return heightmap


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """smoothstep（3t²-2t³）：lattice 格点处导数连续 → 消除折痕/尖峰。"""
    return t * t * (3.0 - 2.0 * t)


def _upsample_smoothstep(
    lattice: np.ndarray, out_h: int, out_w: int
) -> np.ndarray:
    """lattice 噪声 → (out_h, out_w) 平滑插值放大（smoothstep 双线性）。

    全 numpy 向量化；边缘按最外层格钳制（与旧 zoom 行为一致）。
    """
    lh, lw = lattice.shape
    ys = np.linspace(0.0, float(lh - 1), out_h)
    xs = np.linspace(0.0, float(lw - 1), out_w)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, lh - 2)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, lw - 2)
    ty = _smoothstep(ys - y0)
    tx = _smoothstep(xs - x0)
    z00 = lattice[y0[:, None], x0[None, :]]
    z10 = lattice[y0[:, None], x0[None, :] + 1]
    z01 = lattice[y0[:, None] + 1, x0[None, :]]
    z11 = lattice[y0[:, None] + 1, x0[None, :] + 1]
    return (
        (z00 * (1.0 - tx)[None, :] + z10 * tx[None, :]) * (1.0 - ty)[:, None]
        + (z01 * (1.0 - tx)[None, :] + z11 * tx[None, :]) * ty[:, None]
    ).astype(np.float32)
