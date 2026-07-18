import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Airis'))

from Airis.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()