from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QTimer
from Airis.render.gl_widget import GLWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arkady in Cradle")
        self.setGeometry(100, 100, 640, 480)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.gl_widget = GLWidget()
        layout.addWidget(self.gl_widget)
        
        self.setStyleSheet("background-color: #0a0a0f;")

        # 调试：状态栏实时显示相机参数（定位条纹角度用）
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._update_cam_info)
        self._cam_timer.start(100)

    def _update_cam_info(self) -> None:
        """状态栏显示当前相机参数（拖动/滚轮时实时更新）。"""
        cam = self.gl_widget.camera
        t = cam.target
        e = cam.eye()
        self.statusBar().showMessage(
            f"distance={cam.distance:.1f}  yaw={cam.yaw:.1f}  pitch={cam.pitch:.1f}  "
            f"target=({t[0]:.1f},{t[1]:.1f},{t[2]:.1f})  "
            f"eye=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})"
        )