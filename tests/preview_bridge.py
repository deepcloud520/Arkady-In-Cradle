"""临时：渲染默认地形中第一座桥的近景截图（验证默认桥可见）。运行后删除。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from Airis.render.gl_widget import GLWidget
from _output import output_path
from tests.terrain_factory import TerrainFactory


def main() -> int:
    app = QApplication(sys.argv)
    terrain = TerrainFactory.default()  # 默认配置自带 3 座桥
    w = GLWidget(terrain)
    # 第一座桥（栅格 x100~160, y100~104 → 世界中心 260,204），拉近低角度看桥面抬升
    w.camera.target = (260.0, 204.0, 0.0)
    w.camera.distance = 130.0
    w.camera.yaw = 45.0
    w.camera.pitch = 28.0
    w.resize(960, 640)
    w.show()

    frames = {"n": 0}

    def grab() -> None:
        frames["n"] += 1
        if frames["n"] >= 8:
            out = output_path("bridge_preview.png")
            w.grabFramebuffer().save(str(out))
            print(f"BRIDGE_PREVIEW_OK -> {out} (frames={frames['n']})")
            app.quit()

    w.frameSwapped.connect(grab)
    QTimer.singleShot(5000, app.quit)
    app.exec()
    print("BRIDGE_PREVIEW_EXIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
