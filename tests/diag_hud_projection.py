"""诊断：单位标记投影位置（Phase 5c HUD 重合/位移问题）。

用法：uv run python tests/diag_hud_projection.py
在 offscreen 平台创建 MainWindow，用 main.py 相机参数打印全部
活跃单位的投影屏幕坐标 / clip_w / 可见性，定位重合与飞出原因。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

from Airia.render.adapter import WorldAdapter
from Airia.render.picking import project_world


def main() -> None:
    app = QApplication(sys.argv)
    from Airia.ui.main_window import MainWindow

    win = MainWindow()
    cam = win.gl_widget.camera
    # main.py 相机参数
    cam.target = (256.0, 256.0, 0.0)
    cam.distance = 200.0
    cam.yaw = 45.0
    cam.pitch = 40.0

    snapshot = win.simulation.state.snapshot()
    adapter = WorldAdapter.from_snapshot(snapshot)
    active = np.flatnonzero(adapter.armor_active)
    pos = adapter.armor_pos[active].copy()
    mvp = cam.mvp_matrix(1.6)  # 1280/800
    screen, ndc_z, clip_w, visible = project_world(pos, mvp, 1280, 800)

    print(f"相机 eye={tuple(round(v,1) for v in cam.eye())} d={cam.distance:.0f} yaw={cam.yaw} pitch={cam.pitch}")
    for k, idx in enumerate(active):
        p = adapter.armor_pos[idx]
        print(
            f"idx={idx:2d} side={int(adapter.armor_side[idx])} "
            f"world=({p[0]:6.1f},{p[1]:6.1f},{p[2]:5.1f}) "
            f"screen=({screen[k,0]:7.1f},{screen[k,1]:7.1f}) "
            f"clip_w={clip_w[k]:7.3f} ndc_z={ndc_z[k]:+.3f} visible={visible[k]}"
        )

    # 屏幕内点位分析
    inside = visible & (screen[:, 0] >= 0) & (screen[:, 0] <= 1280) & (screen[:, 1] >= 0) & (screen[:, 1] <= 800)
    print(f"\n屏幕内圆点: {int(np.count_nonzero(inside))}/{len(active)}")
    if np.any(inside):
        s = screen[inside]
        print(f"屏幕内范围 x=[{s[:,0].min():.0f},{s[:,0].max():.0f}] y=[{s[:,1].min():.0f},{s[:,1].max():.0f}]")


if __name__ == "__main__":
    main()
