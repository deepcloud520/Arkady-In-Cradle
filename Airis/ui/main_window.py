from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from render.gl_widget import GLWidget


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