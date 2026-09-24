# Bond_date_old

## News + Fundamental Engine

### New modules

```text
services/
  query_expander.py
news/
  fetcher.py
services/
  company_news_analysis.py
  signal_service.py
  fundamental_engine.py
api/
  company_news_api.py
```

### News status field

All analysis/signal/fundamental pipelines now propagate:

```json
{ "news_status": "ok | empty | error" }
```

### Example

```json
{
  "query": "AAPL",
  "expanded_queries": ["AAPL", "Apple", "Apple Inc"],
  "news_status": "ok",
  "news_count": 42
}
```

## Sell Stress Share batch web-report

The Share batch mode in `🧩 Sell_stres` can export an interactive web report (`.html`) with filtering/grouping.

### What is included

- UI/business/data separation:
  - UI integration in existing app: `app.py` (`🧩 Sell_stres` / `Share`)
  - Filters/data loading from MOEX index analytics API: `sell_stress_ui/data.py`
  - HTML report export builder: `sell_stress_ui/reporting.py`

### Run

```bash
streamlit run app.py
```

### Test/check

```bash
python -m compileall app.py sell_stress_ui
```

### Notes

- Asset list can be filtered by index (`IMOEX`, `RTS`) and stock text filter.
- Repeated calculations are cached in `sell_stress_ui/service.py` with `lru_cache`.
- Results can be exported as CSV from the UI.
- In Share batch mode, an additional interactive HTML report export is available (2 sheets/tabs).
- Sheet 1 contains the chart with vertical axis `DeltaP` and horizontal axis `Q`.
- HTML report includes filters by index, ticker, and ISIN.
- Batch report ranks ISINs by index inclusion across the full MOEX stock-index catalog (main/sector/thematic), using MOEX index analytics endpoint and ticker -> ISIN resolution via `.../markets/shares/securities/{ticker}`.

## Анализ эмиссионных документов

Панель **«Анализ эмиссионных документов»** принимает `PDF` и `DOCX`. PDF обрабатывается постранично: для страниц без текстового слоя используется OCR; затем текст разбивается на токенизированные блоки. Каждый блок локальная открытая модель преобразует в JSON фактов, а итоговая локальная модель формирует summary из этих JSON.

Внешние API, `OPENAI_API_KEY` и сетевые запросы для анализа **не используются**. Базовый структурированный анализ работает без модели. Локальная GGUF-модель (опционально) улучшает summary и требует вручную подготовленного окружения с `llama-cpp-python`; рекомендуемый вариант для русского языка — `Qwen2.5-3B-Instruct-Q4_K_M.gguf`. Скачайте/разместите файл модели заранее в окружении и задайте путь:

```bash
export LOCAL_LLM_MODEL_PATH=/opt/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

По умолчанию приложение ищет модель в `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf`. Если файла либо `llama-cpp-python` нет, приложение остаётся работоспособным и формирует структурированную локальную сводку по правилам.
