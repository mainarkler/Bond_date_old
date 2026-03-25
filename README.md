# Bond_date_old

## News + Investment Analysis + Factor Signal Extension

### File structure

```text
news/
  __init__.py
  fetcher.py
  parser.py
  deduplicator.py
  scorer.py
  models.py
agent/
  __init__.py
  agent.py
  prompts.py
  analyzer.py
  postprocessor.py
services/
  cache_backend.py
  company_news_analysis.py
  factor_engine.py
  signal_service.py
storage/
  __init__.py
  signals_store.py
api/
  company_news_api.py
news_agent_config.py
company_news_cli.py
```

### Environment variables

```bash
export NEWSAPI_KEY="..."
export GNEWS_KEY="..."
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export REDIS_URL="redis://localhost:6379/0"   # optional
export CACHE_TTL_SECONDS="300"
export SIGNAL_CACHE_TTL_SECONDS="600"
export SIGNAL_STORE_PATH="storage/signals.db"
```

### Unified analysis function

```python
from services.company_news_analysis import get_company_news_analysis

result = await get_company_news_analysis("AAPL")
print(result)
```

### Investment signal function

```python
from services.signal_service import get_investment_signal

signal = await get_investment_signal("AAPL")
print(signal)
```

### API usage

```bash
uvicorn api.company_news_api:app --reload
```

Analyze endpoint:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query":"AAPL"}'
```

Signal endpoint:

```bash
curl -X POST http://127.0.0.1:8000/signal \
  -H "Content-Type: application/json" \
  -d '{"query":"AAPL"}'
```

### Example signal output

```json
{
  "signal": "BUY",
  "score": 0.2541,
  "factors": {
    "earnings": 0.1601,
    "m&a": 0.0412,
    "regulation": -0.0204,
    "macro": 0.0189,
    "product": 0.0462,
    "litigation": 0.0081
  },
  "top_events": [
    {
      "event_type": "earnings",
      "sentiment": 0.5,
      "confidence": 0.82,
      "timestamp": "2026-03-25T09:15:00+00:00",
      "title": "Apple beats quarterly earnings estimates"
    }
  ]
}
```
