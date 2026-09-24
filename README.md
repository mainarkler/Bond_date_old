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
