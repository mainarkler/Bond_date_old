from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


def build_share_batch_html_report(
    combined_delta_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> bytes:
    """Build interactive HTML report with index filter/grouping for batch share output."""
    ranking = ranking_df.copy() if not ranking_df.empty else pd.DataFrame(columns=["ISIN", "Indices", "RankScore"])
    meta = meta_df.copy() if not meta_df.empty else pd.DataFrame(columns=["ISIN", "T", "Sigma", "MDTV"])
    curves = combined_delta_df.copy()

    isins_payload = meta.merge(ranking, on="ISIN", how="outer")
    isins_payload["Indices"] = isins_payload.get("Indices", "").fillna("")
    isins_payload["RankScore"] = pd.to_numeric(isins_payload.get("RankScore", 0), errors="coerce").fillna(0).astype(int)
    isins_payload = isins_payload.sort_values(["RankScore", "ISIN"], ascending=[False, True]).fillna("")

    index_set = set()
    for indices in isins_payload["Indices"].astype(str):
        index_set.update(i.strip() for i in indices.split(";") if i.strip())
    index_options = ["ALL"] + sorted(index_set)

    group_rows = []
    for index_name in sorted(index_set):
        mask = isins_payload["Indices"].str.contains(index_name, na=False)
        group_rows.append(
            {
                "Index": index_name,
                "IsinCount": int(mask.sum()),
                "AvgRankScore": float(isins_payload.loc[mask, "RankScore"].mean()) if mask.any() else 0,
            }
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "index_options": index_options,
        "isins": isins_payload.to_dict(orient="records"),
        "curves": curves.to_dict(orient="records"),
        "groups": group_rows,
    }

    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Sell_stres Share Batch Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 13px; }}
    th {{ background: #f4f6f8; text-align: left; }}
    .row {{ display: flex; gap: 24px; align-items: center; margin: 8px 0; }}
    .muted {{ color: #666; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <h2>Sell_stres Share Batch Report</h2>
  <p class="muted">Generated at: <span id="generatedAt"></span></p>

  <div class="row">
    <label for="indexFilter"><strong>Фильтр по индексу:</strong></label>
    <select id="indexFilter"></select>

    <label for="isinFilter"><strong>ISIN:</strong></label>
    <input id="isinFilter" placeholder="RU..." />
  </div>

  <h3>Группировка по индексам</h3>
  <table id="groupTable">
    <thead><tr><th>Index</th><th>ISIN count</th><th>Avg rank score</th></tr></thead>
    <tbody></tbody>
  </table>

  <h3>Сводка ISIN</h3>
  <table id="isinTable">
    <thead><tr><th>ISIN</th><th>Indices</th><th>RankScore</th><th>T</th><th>Sigma</th><th>MDTV</th></tr></thead>
    <tbody></tbody>
  </table>

  <h3>Точки кривой (Q / ΔP)</h3>
  <table id="curveTable">
    <thead><tr><th>ISIN</th><th>Q</th><th>ΔP</th></tr></thead>
    <tbody></tbody>
  </table>

<script>
const DATA = {payload_json};

document.getElementById('generatedAt').textContent = DATA.generated_at;
const indexFilterEl = document.getElementById('indexFilter');
const isinFilterEl = document.getElementById('isinFilter');

DATA.index_options.forEach(opt => {{
  const option = document.createElement('option');
  option.value = opt;
  option.textContent = opt;
  indexFilterEl.appendChild(option);
}});

function render() {{
  const selectedIndex = indexFilterEl.value || 'ALL';
  const isinText = (isinFilterEl.value || '').trim().toUpperCase();

  const filteredIsins = DATA.isins.filter(row => {{
    const byIndex = selectedIndex === 'ALL' || (row.Indices || '').includes(selectedIndex);
    const byIsin = !isinText || (row.ISIN || '').toUpperCase().includes(isinText);
    return byIndex && byIsin;
  }});
  const allowedIsins = new Set(filteredIsins.map(r => r.ISIN));

  const filteredCurves = DATA.curves.filter(row => allowedIsins.has(row.ISIN));

  const groupBody = document.querySelector('#groupTable tbody');
  groupBody.innerHTML = '';
  DATA.groups
    .filter(g => selectedIndex === 'ALL' || g.Index === selectedIndex)
    .forEach(g => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{g.Index}}</td><td>${{g.IsinCount}}</td><td>${{Number(g.AvgRankScore).toFixed(2)}}</td>`;
      groupBody.appendChild(tr);
    }});

  const isinBody = document.querySelector('#isinTable tbody');
  isinBody.innerHTML = '';
  filteredIsins.forEach(row => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{row.ISIN || ''}}</td><td>${{row.Indices || ''}}</td><td>${{row.RankScore || 0}}</td><td>${{row.T || ''}}</td><td>${{row.Sigma || ''}}</td><td>${{row.MDTV || ''}}</td>`;
    isinBody.appendChild(tr);
  }});

  const curveBody = document.querySelector('#curveTable tbody');
  curveBody.innerHTML = '';
  filteredCurves.forEach(row => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{row.ISIN}}</td><td>${{row.Q}}</td><td>${{row.DeltaP}}</td>`;
    curveBody.appendChild(tr);
  }});
}}

indexFilterEl.addEventListener('change', render);
isinFilterEl.addEventListener('input', render);
render();
</script>
</body>
</html>"""

    return html.encode("utf-8")
