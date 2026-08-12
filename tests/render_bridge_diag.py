"""临时诊断：多视角渲染默认桥，定位抖动/黑条纹/缺失面。运行后删除。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from Airis.render.gl_widget import GLWidget
from _output import output_path
from tests.terrain_factory import TerrainFactory

# (名称, 相机target(x,y,z), distance, yaw, pitch)
VIEWS = [
    ("far_side", (260.0, 204.0, 13.0), 420.0, 0.0, 8.0),   # 远距离正对侧面
    ("far_angle", (260.0, 204.0, 13.0), 400.0, 45.0, 18.0),  # 远距离斜视
    ("mid_side", (260.0, 204.0, 13.0), 180.0, 0.0, 5.0),   # 中等距离正对侧面
    ("mid_angle", (260.0, 204.0, 13.0), 200.0, 45.0, 15.0),  # 中等距离斜视
]


def main() -> int:
    app = QApplication(sys.argv)
    terrain = TerrainFactory.default()
    w = GLWidget(terrain)
    w.resize(960, 640)
    w.show()
    state = {"i": 0, "frame": 0}

    def grab() -> None:
        name, tgt, dist, yaw, pitch = VIEWS[state["i"]]
        w.camera.target = (tgt[0], tgt[1], tgt[2])
        w.camera.distance = dist
        w.camera.yaw = yaw
        w.camera.pitch = pitch
        state["frame"] += 1
        if state["frame"] >= 6:
            out = output_path(f"bridge_diag_{name}.png")
            w.grabFramebuffer().save(str(out))
            print(f"SAVED {out}")
            state["frame"] = 0
            state["i"] += 1
            if state["i"] >= len(VIEWS):
                app.quit()

    w.frameSwapped.connect(grab)
    QTimer.singleShot(15000, app.quit)
    app.exec()
    print("DIAG_EXIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
