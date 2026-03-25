# Bond_date_old

## News + Investment Analysis Extension

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
  company_news_analysis.py
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
```

### Unified integration function

```python
from services.company_news_analysis import get_company_news_analysis

result = await get_company_news_analysis("AAPL")
print(result)
```

### CLI usage

```bash
python company_news_cli.py "AAPL"
```

### FastAPI usage

```bash
uvicorn api.company_news_api:app --reload
```

POST request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query":"AAPL"}'
```

### Sample output

```json
{
  "query": "AAPL",
  "news_count": 12,
  "news": [
    {
      "title": "Apple reports quarterly earnings beat",
      "source": "Reuters",
      "published_at": "2026-03-25T10:15:00+00:00",
      "url": "https://example.com/apple-earnings",
      "summary": "Apple beat consensus EPS and revenue guidance...",
      "relevance_score": 0.92
    }
  ],
  "analysis": {
    "sentiment_score": 0.44,
    "key_events": [
      "Quarterly earnings beat",
      "New buyback authorization"
    ],
    "risks": [
      "Regulatory pressure in the EU"
    ],
    "opportunities": [
      "Services margin expansion"
    ],
    "final_assessment": "Constructive medium-term outlook with policy risk overhang.",
    "confidence": 0.78
  }
}
```
