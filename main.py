import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from Airis.ui.main_window import MainWindow


def main():
    # 请求 OpenGL 4.6 Core Profile（必须在 QApplication 创建之前设置）
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    # 2026-08-11：请求 32 位深度缓冲，配合 camera.py 的 near=3/far=5000 消除远处 Z-fighting
    fmt.setDepthBufferSize(32)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    # 调试/演示：相机对准地图中心（默认地形为平坦演示地形，见 gl_widget._default_terrain）
    hf = window.gl_widget.terrain.heightfield
    cam = window.gl_widget.camera
    cam.target = (hf.width * hf.cs * 0.5, hf.height * hf.cs * 0.5, 0.0)
    cam.distance = hf.width * hf.cs * 1.2
    cam.yaw = 45.0
    cam.pitch = 30.0
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()