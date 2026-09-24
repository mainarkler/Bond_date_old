@echo off
cd /d "%~dp0.."
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean --onedir --windowed --name BondAnalyzer app\main.py
