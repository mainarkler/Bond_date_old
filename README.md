# Bond_date_old

## API

### Variation margin (VM)

The Streamlit application currently exposes the PDF report through a query-string action:

```text
GET https://bonddate.streamlit.app/?action=vm_pdf&trade_name=gold-12.26&quantity=3000
```

- `trade_name` — futures contract name or ticker, for example `gold-12.26`.
- `quantity` — number of contracts in the position.
- Response — PDF report (not JSON).

#### Planned JSON endpoints

The following URLs describe the intended JSON API contract. **They are not available on `bonddate.streamlit.app` yet**; the corresponding request handlers must be implemented and deployed before clients can use them.

| Method | Full URL example | Intended response |
|---|---|---|
| Full quote and VM data | `GET /api/v1/vm?trade_name=gold-12.26&quantity=3000` | Contract ID, quote time, BID, previous price, price change, VM per contract, position VM |
| VM per contract | `GET /api/v1/vm/variation-margin?trade_name=gold-12.26` | VM per contract in RUB |
| Quote timestamp | `GET /api/v1/vm/quote-time?trade_name=gold-12.26` | Quote time and trade date |
| Contract price change | `GET /api/v1/vm/price-change?trade_name=gold-12.26` | Current BID, previous price, price change percentage |
| Position VM | `GET /api/v1/vm/position-margin?trade_name=gold-12.26&quantity=3000` | VM per contract, quantity, total position VM |

When deployed as a separate API service, the full URLs will have the form:

```text
https://<api-host>/api/v1/vm?trade_name=gold-12.26&quantity=3000
https://<api-host>/api/v1/vm/variation-margin?trade_name=gold-12.26
https://<api-host>/api/v1/vm/quote-time?trade_name=gold-12.26
https://<api-host>/api/v1/vm/price-change?trade_name=gold-12.26
https://<api-host>/api/v1/vm/position-margin?trade_name=gold-12.26&quantity=3000
```

The JSON API should return `Content-Type: application/json`. If token authentication is enabled, send the token in the `X-API-Token` request header.

#### Data definitions

- `quote_time` — update time reported by MOEX ISS.
- `variation_margin_per_contract_rub` — calculated using the current `BID`, previous `PREVPRICE`, and contract tick-price multiplier.
- `price_change_pct` — percentage change field reported by MOEX ISS (`LASTTOPREVPRICE`).
- `position_variation_margin_rub` — VM per contract multiplied by `quantity`. This is the calculated variation margin for the position, not exchange initial margin / collateral.

Example JSON response shape (values are illustrative, not live quotes):

```json
{
  "trade_name": "gold-12.26",
  "secid": "GOLD-12.26",
  "quantity": 3000,
  "quote_time": "2026-10-22 15:45:00",
  "bid": 0,
  "previous_price": 0,
  "price_change_pct": 0,
  "variation_margin_per_contract_rub": 0,
  "position_variation_margin_rub": 0,
  "source": "MOEX ISS"
}
```

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
