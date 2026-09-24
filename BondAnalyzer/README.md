# BondAnalyzer Portable
A fully local PySide6 desktop MVP for extracting traceable bond placement terms from text-layer PDF and DOCX documents. Drag a document into the application, review values with page/section/snippet evidence, and export JSON or XLSX.

## Run in development
```bat
cd BondAnalyzer
py -m pip install -r requirements.txt
py -m app.main
```

## Build portable Windows folder
```bat
cd BondAnalyzer
scripts\build_portable.bat
```
The output is `dist\BondAnalyzerPortable\BondAnalyzer.exe`. Copy the **entire** `BondAnalyzerPortable` folder to a Windows PC; Python, Node, Docker and Ollama are not needed there.

## Optional local LLM
Copy `llama-server.exe` from a Windows llama.cpp build to `runtime\llama\`, and copy a GGUF model to `models\`. Do not commit either file. If either is absent the deterministic extractor still works. The only HTTP traffic is to `http://127.0.0.1:8765`; documents are never uploaded, and there is no telemetry or cloud API.

## Pipeline
PDF text/DOCX text and tables are loaded with page mapping; deterministic rules and section detection produce evidence. When present, llama.cpp receives bounded chunks (25,000 characters) and its JSON is accepted only when its snippet exactly occurs on the declared page. Rules take priority during merge.

## MVP limits
Scanned PDFs require OCR and show an explanatory message. DOCX is represented as logical page 1. Heading detection and Russian rule patterns are best-effort; review every extracted value and source before relying on it.
