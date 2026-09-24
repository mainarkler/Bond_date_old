from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QTableWidget,QTableWidgetItem,QTextEdit,QMessageBox
from .widgets import DropZone
from ..core.document_loader import load_document
from ..core.rule_extractor import extract_rules
from ..core.llm_extractor import extract_with_llm
from ..core.merger import merge_results
from ..core.models import FIELDS
from ..export.json_export import export_json
from ..export.xlsx_export import export_xlsx
class MainWindow(QMainWindow):
 def __init__(self,root):
  super().__init__(); self.root=Path(root); self.document=None; self.result=None; self.setWindowTitle('BondAnalyzer Portable'); self.resize(1100,720)
  central=QWidget(); layout=QVBoxLayout(central); self.setCentralWidget(central)
  top=QHBoxLayout(); title=QLabel('BondAnalyzer'); title.setStyleSheet('font-size:24px;font-weight:bold'); top.addWidget(title); top.addStretch(); local=QLabel('● LOCAL MODE'); local.setStyleSheet('color:#18794e;font-weight:bold'); top.addWidget(local); layout.addLayout(top)
  self.drop=DropZone(); self.drop.setMinimumHeight(130); self.drop.setStyleSheet('#dropZone{border:2px dashed #718096;border-radius:10px;background:#f7fafc;}'); self.drop.file_dropped.connect(self.analyze); self.drop.choose.clicked.connect(self.pick); layout.addWidget(self.drop)
  self.status=QLabel('Выберите или перетащите документ. Анализ выполняется только локально.'); layout.addWidget(self.status)
  self.table=QTableWidget(0,6); self.table.setHorizontalHeaderLabels(['Параметр','Значение','Confidence','Страница','Раздел','Статус']); self.table.itemSelectionChanged.connect(self.show_source); self.table.setSelectionBehavior(QTableWidget.SelectRows); layout.addWidget(self.table)
  self.snippet=QTextEdit(); self.snippet.setReadOnly(True); self.snippet.setPlaceholderText('Выберите строку, чтобы увидеть подтверждающий фрагмент.'); self.snippet.setMaximumHeight(100); layout.addWidget(self.snippet)
  buttons=QHBoxLayout(); self.json=QPushButton('JSON'); self.xlsx=QPushButton('Excel'); self.open_source=QPushButton('Открыть источник'); self.json.clicked.connect(lambda:self.save('json')); self.xlsx.clicked.connect(lambda:self.save('xlsx')); self.open_source.clicked.connect(self.show_source); [buttons.addWidget(x) for x in (self.json,self.xlsx,self.open_source)]; buttons.addStretch(); layout.addLayout(buttons)
 def pick(self):
  path,_=QFileDialog.getOpenFileName(self,'Выберите документ','','Документы (*.pdf *.docx)');
  if path:self.analyze(path)
 def analyze(self,path):
  try:
   self.status.setText('Анализ документа...'); self.document=load_document(path); rules=extract_rules(self.document); llm=extract_with_llm(self.document,self.root); self.result=merge_results(rules,llm); self.populate(); self.status.setText(f'Готово: {self.document.filename}. LLM: {"использована" if self.result.metadata.get("llm_used") else "не установлена или не подтвердила данные"}.')
  except Exception as exc: QMessageBox.warning(self,'Не удалось проанализировать',str(exc)); self.status.setText('Ошибка анализа.')
 def populate(self):
  self.table.setRowCount(0)
  for field in FIELDS:
   value=self.result.parameters.get(field); source=self.result.sources.get(field); row=self.table.rowCount(); self.table.insertRow(row)
   values=[field, str(value) if value is not None else '—', f'{self.result.confidence.get(field,0):.2f}' if value is not None else '—',str(source.page) if source else '—',source.section or '—' if source else '—','Подтверждено' if source else 'Не найдено']
   for col,value in enumerate(values):self.table.setItem(row,col,QTableWidgetItem(value))
  self.table.resizeColumnsToContents()
 def show_source(self):
  if not self.result or not self.table.selectedItems():return
  field=self.table.item(self.table.currentRow(),0).text(); source=self.result.sources.get(field); self.snippet.setText(source.snippet if source else 'Для этого параметра подтверждающий источник не найден.')
 def save(self,kind):
  if not self.result:return
  suffix='json' if kind=='json' else 'xlsx'; path,_=QFileDialog.getSaveFileName(self,'Сохранить результат',str(self.root/'output'/f'{self.document.filename}.{suffix}'),f'{suffix.upper()} (*.{suffix})')
  if path: (export_json(path,self.document.filename,self.result) if kind=='json' else export_xlsx(path,self.result)); self.status.setText(f'Сохранено: {path}')
