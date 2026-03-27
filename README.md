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
