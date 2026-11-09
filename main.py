import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from Airia.ui.fonts import load_ui_fonts
from Airia.ui.main_window import MainWindow


def main():
    # 请求 OpenGL 4.6 Core Profile（必须在 QApplication 创建之前设置）
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    # 2026-08-11：请求 32 位深度缓冲，配合 camera.py 的 near=3/far=5000 消除远处 Z-fighting
    fmt.setDepthBufferSize(32)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    # 界面字体（中文正文 + 等宽时间码；offscreen 下也能拿到，不会变豆腐块）
    load_ui_fonts(app)
    window = MainWindow()

    # 演示：三军交战（MainWindow._create_simulation 装配的完整调用链）——
    #   15km×15km 扁平地图，战场在地图中央 (7500,7500)：坦克编队相向接敌、
    #   双方炮兵曲射对轰 + 反炮闭环、双方空军迎头空战（AAM/坠毁链）。
    # 相机拉高俯瞰整个 15km 战场（远裁剪面由 GLWidget 按地图对角线自动放大，
    # 2026-11-03 起不再固定 12000，见 render/camera.py: far_for_extent）
    hf = window.gl_widget.terrain.heightfield
    cam = window.gl_widget.camera
    cam.target = (hf.width * hf.cs * 0.5, hf.height * hf.cs * 0.5, 0.0)
    cam.distance = 9500.0   # 俯瞰全图（zoom 上限 10000 内）
    cam.yaw = 45.0
    cam.pitch = 45.0
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
