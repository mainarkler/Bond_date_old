# Bond_date_old

## Fundamental Analysis Engine (IFRS + News + AI)

### File structure

```text
ifrs/
  __init__.py
  loader.py
  parser.py
  extractor.py
  normalizer.py
news/
agent/
services/
  fundamental_metrics.py
  fundamental_engine.py
  signal_service.py
  company_news_analysis.py
api/
  company_news_api.py
```

### API usage

```bash
uvicorn api.company_news_api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/fundamental \
  -H "Content-Type: application/json" \
  -d '{"query":"SBER"}'
```

### Example fundamental response

```json
{
  "financials": {
    "revenue": 1250000000.0,
    "ebitda": 320000000.0,
    "net_income": 180000000.0,
    "assets": 5400000000.0,
    "liabilities": 2900000000.0,
    "equity": 2500000000.0,
    "cash_flow": 240000000.0
  },
  "ratios": {
    "revenue_growth": 0.136364,
    "ebitda_margin": 0.256,
    "net_margin": 0.144,
    "debt_to_equity": 1.16,
    "roe": 0.072,
    "free_cash_flow": 180000000.0
  },
  "news_summary": {
    "sentiment_score": 0.24,
    "trend_analysis": "Positive trend with improving earnings quality.",
    "valuation_view": "fair"
  },
  "strengths": ["Margin resilience", "Cash generation"],
  "risks": ["Regulatory uncertainty"],
  "valuation_view": "fair",
  "final_assessment": "Fundamentals are stable with balanced upside and risk.",
  "confidence": 0.71
}
```
