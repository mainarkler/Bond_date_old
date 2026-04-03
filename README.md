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

## Sell Stress XML UI

A new XML-driven UI layer is embedded directly in the main Streamlit app inside `🧩 Sell_stres` → `Share`.

### What is included

- XML schema for the form: `sell_stress_ui/schemas/sell_stress_form.xml`
- Example index-membership dataset: `sell_stress_ui/data/index_membership.csv`
- UI/business/data separation:
  - UI integration in existing app: `app.py` (`🧩 Sell_stres` / `Share`)
  - XML parsing: `sell_stress_ui/form_config.py`
  - Filters/data loading: `sell_stress_ui/data.py`
  - Integration with `sell_stress`: `sell_stress_ui/service.py`

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
