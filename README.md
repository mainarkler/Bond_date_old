# Bond_date_old

## Fundamental Analysis Engine (Financial-first)

### File structure

```text
ifrs/
  loader.py
  parser.py
  extractor.py
  normalizer.py
services/
  fundamental_metrics.py
  fundamental_engine.py
api/
  company_news_api.py
```

### Fundamental endpoint

```bash
uvicorn api.company_news_api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/fundamental \
  -H "Content-Type: application/json" \
  -d '{"query":"SBER"}'
```

### Example response

```json
{
  "mode": "financial_only",
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
  "strengths": [
    "Revenue growth is positive.",
    "EBITDA margin indicates strong operating profitability."
  ],
  "risks": [
    "ROE is below target range."
  ],
  "valuation_view": "fair",
  "trend_analysis": "Fundamental trend is improving.",
  "confidence": 0.65
}
```
