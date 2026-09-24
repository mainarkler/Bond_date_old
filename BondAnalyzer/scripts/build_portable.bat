@echo off
setlocal
cd /d "%~dp0.."
call scripts\build_windows.bat
if errorlevel 1 exit /b %errorlevel%
set TARGET=dist\BondAnalyzerPortable
if exist "%TARGET%" rmdir /s /q "%TARGET%"
mkdir "%TARGET%"
xcopy /e /i /y "dist\BondAnalyzer" "%TARGET%" >nul
move "%TARGET%\BondAnalyzer.exe" "%TARGET%\BondAnalyzer.exe" >nul 2>nul
xcopy /e /i /y runtime "%TARGET%\runtime" >nul
xcopy /e /i /y models "%TARGET%\models" >nul
xcopy /e /i /y output "%TARGET%\output" >nul
echo Portable build: %TARGET%\BondAnalyzer.exe
