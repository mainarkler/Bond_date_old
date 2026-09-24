from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFrame,QVBoxLayout,QLabel,QPushButton
class DropZone(QFrame):
 file_dropped=Signal(str)
 def __init__(self):
  super().__init__(); self.setAcceptDrops(True); self.setObjectName('dropZone')
  layout=QVBoxLayout(self); label=QLabel('Перетащите PDF или DOCX сюда'); label.setStyleSheet('font-size: 18px; font-weight: 600;'); label.setAlignment(Qt.AlignCenter); layout.addWidget(label)
  self.choose=QPushButton('Выбрать файл'); layout.addWidget(self.choose)
 def dragEnterEvent(self,event):
  urls=event.mimeData().urls()
  if urls and urls[0].toLocalFile().lower().endswith(('.pdf','.docx')):event.acceptProposedAction()
 def dropEvent(self,event):self.file_dropped.emit(event.mimeData().urls()[0].toLocalFile())
