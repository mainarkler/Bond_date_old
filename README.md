# Bond_date_old

## Alpha Signal Engine

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
  market_context.py
  signal_refiner.py
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

### API usage

```bash
uvicorn api.company_news_api:app --reload
```

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
  "confidence": 0.71,
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
      "magnitude": 0.9,
      "surprise": 0.8,
      "timestamp": "2026-03-25T09:15:00+00:00",
      "title": "Apple better than expected quarterly earnings"
    }
  ],
  "market_context": {
    "price_change_1d": 0.008,
    "price_change_3d": 0.015,
    "volatility": 2.18,
    "volume_spike": 0.12
  },
  "explanation": "Signal=BUY; normalized_score=0.2541; strongest_factor=earnings (0.1601)."
}
```
