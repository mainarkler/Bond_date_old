from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


def build_share_batch_html_report(
    combined_delta_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> bytes:
    """Build interactive HTML report with two tabs/pages and chart on first page."""
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

    html_template = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Sell_stres Share Batch Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border: 1px solid #ddd; padding: 6px; font-size: 13px; }
    th { background: #f4f6f8; text-align: left; }
    .row { display: flex; gap: 24px; align-items: center; margin: 8px 0; flex-wrap: wrap; }
    .muted { color: #666; }
    .tabs { display: flex; gap: 8px; margin: 12px 0; }
    .tab-btn { border: 1px solid #ccc; background: #f8f8f8; padding: 8px 12px; cursor: pointer; border-radius: 6px; }
    .tab-btn.active { background: #dfefff; border-color: #8db3ff; }
    .tab { display: none; }
    .tab.active { display: block; }
    #chartSvg { width: 100%; height: 520px; border: 1px solid #ddd; background: white; }
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

  <div class="tabs">
    <button id="tabChartBtn" class="tab-btn active" type="button">Лист 1: График</button>
    <button id="tabDataBtn" class="tab-btn" type="button">Лист 2: Данные</button>
  </div>

  <section id="tabChart" class="tab active">
    <h3>ΔP(Q): вертикаль = DeltaP, горизонталь = Q</h3>
    <svg id="chartSvg" viewBox="0 0 1100 520" preserveAspectRatio="none"></svg>
  </section>

  <section id="tabData" class="tab">
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
  </section>

<script>
const DATA = __PAYLOAD__;
document.getElementById('generatedAt').textContent = DATA.generated_at;
const indexFilterEl = document.getElementById('indexFilter');
const isinFilterEl = document.getElementById('isinFilter');

DATA.index_options.forEach(opt => {
  const option = document.createElement('option');
  option.value = opt;
  option.textContent = opt;
  indexFilterEl.appendChild(option);
});

const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#17becf','#8c564b','#e377c2','#7f7f7f','#bcbd22'];

function groupByIsin(rows) {
  const map = new Map();
  rows.forEach(r => {
    if (!map.has(r.ISIN)) map.set(r.ISIN, []);
    map.get(r.ISIN).push(r);
  });
  for (const [k,v] of map.entries()) {
    v.sort((a,b) => Number(a.Q)-Number(b.Q));
  }
  return map;
}

function drawChart(rows) {
  const svg = document.getElementById('chartSvg');
  svg.innerHTML = '';
  if (!rows.length) {
    svg.innerHTML = '<text x="40" y="40" fill="#999">Нет данных для графика</text>';
    return;
  }

  const w = 1100, h = 520;
  const ml = 80, mr = 20, mt = 25, mb = 55;
  const pw = w - ml - mr, ph = h - mt - mb;

  const xs = rows.map(r => Number(r.Q));
  const ys = rows.map(r => Number(r.DeltaP));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xPad = (xMax - xMin) * 0.03 || 1;
  const yPad = (yMax - yMin) * 0.08 || 1e-6;

  const x0 = xMin - xPad, x1 = xMax + xPad;
  const y0 = yMin - yPad, y1 = yMax + yPad;

  const sx = x => ml + ((x - x0) / (x1 - x0)) * pw;
  const sy = y => mt + (1 - (y - y0) / (y1 - y0)) * ph;

  const axis = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  axis.setAttribute('stroke', '#444');
  axis.innerHTML = `
    <line x1="${ml}" y1="${mt + ph}" x2="${ml + pw}" y2="${mt + ph}" />
    <line x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + ph}" />
  `;
  svg.appendChild(axis);

  const labelX = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  labelX.setAttribute('x', String(ml + pw/2));
  labelX.setAttribute('y', String(h - 14));
  labelX.setAttribute('text-anchor', 'middle');
  labelX.textContent = 'Q';
  svg.appendChild(labelX);

  const labelY = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  labelY.setAttribute('x', '20');
  labelY.setAttribute('y', String(mt + ph/2));
  labelY.setAttribute('transform', `rotate(-90 20 ${mt + ph/2})`);
  labelY.setAttribute('text-anchor', 'middle');
  labelY.textContent = 'DeltaP';
  svg.appendChild(labelY);

  const byIsin = groupByIsin(rows);
  let i = 0;
  for (const [isin, points] of byIsin.entries()) {
    const color = colors[i % colors.length];
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', color);
    poly.setAttribute('stroke-width', '2');
    poly.setAttribute('points', points.map(p => `${sx(Number(p.Q))},${sy(Number(p.DeltaP))}`).join(' '));
    svg.appendChild(poly);

    const last = points[points.length-1];
    const tag = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    tag.setAttribute('x', String(sx(Number(last.Q)) + 6));
    tag.setAttribute('y', String(sy(Number(last.DeltaP))));
    tag.setAttribute('fill', color);
    tag.setAttribute('font-size', '11');
    tag.textContent = isin;
    svg.appendChild(tag);
    i += 1;
  }
}

function activateTab(name) {
  document.getElementById('tabChart').classList.toggle('active', name === 'chart');
  document.getElementById('tabData').classList.toggle('active', name === 'data');
  document.getElementById('tabChartBtn').classList.toggle('active', name === 'chart');
  document.getElementById('tabDataBtn').classList.toggle('active', name === 'data');
}

document.getElementById('tabChartBtn').addEventListener('click', () => activateTab('chart'));
document.getElementById('tabDataBtn').addEventListener('click', () => activateTab('data'));

function render() {
  const selectedIndex = indexFilterEl.value || 'ALL';
  const isinText = (isinFilterEl.value || '').trim().toUpperCase();

  const filteredIsins = DATA.isins.filter(row => {
    const byIndex = selectedIndex === 'ALL' || (row.Indices || '').includes(selectedIndex);
    const byIsin = !isinText || (row.ISIN || '').toUpperCase().includes(isinText);
    return byIndex && byIsin;
  });
  const allowedIsins = new Set(filteredIsins.map(r => r.ISIN));
  const filteredCurves = DATA.curves.filter(row => allowedIsins.has(row.ISIN));

  drawChart(filteredCurves);

  const groupBody = document.querySelector('#groupTable tbody');
  groupBody.innerHTML = '';
  DATA.groups
    .filter(g => selectedIndex === 'ALL' || g.Index === selectedIndex)
    .forEach(g => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${g.Index}</td><td>${g.IsinCount}</td><td>${Number(g.AvgRankScore).toFixed(2)}</td>`;
      groupBody.appendChild(tr);
    });

  const isinBody = document.querySelector('#isinTable tbody');
  isinBody.innerHTML = '';
  filteredIsins.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.ISIN || ''}</td><td>${row.Indices || ''}</td><td>${row.RankScore || 0}</td><td>${row.T || ''}</td><td>${row.Sigma || ''}</td><td>${row.MDTV || ''}</td>`;
    isinBody.appendChild(tr);
  });

  const curveBody = document.querySelector('#curveTable tbody');
  curveBody.innerHTML = '';
  filteredCurves.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.ISIN}</td><td>${row.Q}</td><td>${row.DeltaP}</td>`;
    curveBody.appendChild(tr);
  });
}

indexFilterEl.addEventListener('change', render);
isinFilterEl.addEventListener('input', render);
render();
</script>
</body>
</html>"""

    html = html_template.replace("__PAYLOAD__", payload_json)
    return html.encode("utf-8")
