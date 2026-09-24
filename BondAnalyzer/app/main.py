import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from app.gui.main_window import MainWindow
if __name__=='__main__':
 app=QApplication(sys.argv); window=MainWindow(Path(__file__).resolve().parents[1]); window.show(); sys.exit(app.exec())
