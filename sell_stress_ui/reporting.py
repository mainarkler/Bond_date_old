from __future__ import annotations

from datetime import datetime
import json

import pandas as pd

INDEX_CATALOG = [
    {"name": "Индекс МосБиржи", "code": "IMOEX"},
    {"name": "IMOEX2 – значения индекса МосБиржи за весь торговый день, включая дополнительные сессии", "code": "IMOEX2"},
    {"name": "Индекс РТС", "code": "RTSI"},
    {"name": "Индекс МосБиржи в юанях", "code": "IMOEXCNY"},
    {"name": "Индекс МосБиржи – активное управление", "code": "IMOEXW"},
    {"name": "Индекс МосБиржи голубых фишек", "code": "MOEXBC"},
    {"name": "Индекс МосБиржи голубых фишек (долларовый)", "code": "MRBC"},
    {"name": "Индекс МосБиржи широкого рынка", "code": "MOEXBMI"},
    {"name": "Индекс МосБиржи широкого рынка (долларовый)", "code": "RUBMI"},
    {"name": "Индекс МосБиржи средней и малой капитализации", "code": "MCXSM"},
    {"name": "Индекс РТС средней и малой капитализации", "code": "RTSSM"},
    {"name": "Нефти и газа", "code": "MOEXOG"},
    {"name": "Нефти и газа (РТС)", "code": "RTSOG"},
    {"name": "Электроэнергетики", "code": "MOEXEU"},
    {"name": "Электроэнергетики (РТС)", "code": "RTSEU"},
    {"name": "Телекоммуникаций", "code": "MOEXTL"},
    {"name": "Телекоммуникаций (РТС)", "code": "RTSTL"},
    {"name": "Металлов и добычи", "code": "MOEXMM"},
    {"name": "Металлов и добычи (РТС)", "code": "RTSMM"},
    {"name": "Финансов", "code": "MOEXFN"},
    {"name": "Финансов (РТС)", "code": "RTSFN"},
    {"name": "Потребительского сектора", "code": "MOEXCN"},
    {"name": "Потребительского сектора (РТС)", "code": "RTSCR"},
    {"name": "Химии и нефтехимии", "code": "MOEXCH"},
    {"name": "Химии и нефтехимии (РТС)", "code": "RTSCH"},
    {"name": "Индекс МосБиржи информационных технологий", "code": "MOEXIT"},
    {"name": "Индекс РТС информационных технологий", "code": "RTSIT"},
    {"name": "Индекс МосБиржи недвижимости", "code": "MOEXRE"},
    {"name": "Индекс РТС недвижимости", "code": "RTSRE"},
    {"name": "Транспорта", "code": "MOEXTN"},
    {"name": "Транспорта (РТС)", "code": "RTSTN"},
    {"name": "Индекс МосБиржи 10", "code": "MOEX10"},
    {"name": "Индекс МосБиржи инноваций", "code": "MOEXINN"},
    {"name": "Индекс МосБиржи IPO", "code": "MIPO"},
]


def build_share_batch_html_report(
    combined_delta_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> bytes:
    ranking = ranking_df.copy() if not ranking_df.empty else pd.DataFrame(columns=["ISIN", "Ticker", "Indices", "RankScore"])
    meta = meta_df.copy() if not meta_df.empty else pd.DataFrame(columns=["ISIN", "Ticker", "T", "Sigma", "MDTV", "Indices", "RankScore"])
    curves = combined_delta_df.copy()

    if "Ticker" not in curves.columns:
        curves = curves.merge(ranking[["ISIN", "Ticker"]], on="ISIN", how="left")
    curves["Ticker"] = curves.get("Ticker", "").fillna("")

    isins_payload = meta.merge(ranking, on=["ISIN", "Ticker"], how="outer") if "Ticker" in meta.columns else meta.merge(ranking, on="ISIN", how="outer")
    isins_payload["Ticker"] = isins_payload.get("Ticker", "").fillna("")
    isins_payload["Indices"] = isins_payload.get("Indices", "").fillna("")
    isins_payload["RankScore"] = pd.to_numeric(isins_payload.get("RankScore", 0), errors="coerce").fillna(0).astype(int)
    isins_payload = isins_payload.sort_values(["RankScore", "Ticker", "ISIN"], ascending=[False, True, True]).fillna("")

    unique_indices_values = sorted(
        {
            str(indices).strip()
            for indices in isins_payload["Indices"].astype(str).tolist()
            if str(indices).strip()
        }
    )
    index_options = ["ALL"] + unique_indices_values

    curve_tickers = sorted([t for t in curves.get("Ticker", pd.Series(dtype=str)).astype(str).unique() if t])
    ticker_options = ["ALL"] + curve_tickers

    group_rows = (
        isins_payload.groupby("Indices", dropna=True)
        .size()
        .reset_index(name="IsinCount")
        .rename(columns={"Indices": "Index"})
    )
    group_rows["Index"] = group_rows["Index"].astype(str)
    group_rows = group_rows[group_rows["Index"].str.len() > 0].sort_values("Index")

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "index_options": index_options,
        "ticker_options": ticker_options,
        "isins": isins_payload.to_dict(orient="records"),
        "curves": curves.to_dict(orient="records"),
        "groups": group_rows.to_dict(orient="records"),
        "index_catalog": INDEX_CATALOG,
    }

    payload_json = json.dumps(payload, ensure_ascii=False)

    html_template = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8" />
<title>Sell_stres Share Batch Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th, td { border: 1px solid #ddd; padding: 6px; font-size: 13px; }
th { background: #f4f6f8; text-align: left; }
.row { display: flex; gap: 20px; align-items: center; margin: 8px 0; flex-wrap: wrap; }
.tabs { display: flex; gap: 8px; margin: 12px 0; }
.tab-btn { border: 1px solid #ccc; background: #f8f8f8; padding: 8px 12px; cursor: pointer; border-radius: 6px; }
.tab-btn.active { background: #dfefff; border-color: #8db3ff; }
.tab { display: none; } .tab.active { display: block; }
.meta-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:8px; }
.meta-card { border:1px solid #ddd; padding:8px; border-radius:6px; }
</style></head><body>
<h2>Sell_stres Share Batch Report</h2><p>Generated at: <span id="generatedAt"></span></p>
<div class="row">
<label><b>Индексы (множественный выбор):</b></label><div id="indexChecklist"></div>
<label>Тикер:</label><select id="tickerFilter"></select>
<label>ISIN:</label><input id="isinFilter" placeholder="RU..." />
</div>
<div class="tabs"><button id="tabChartBtn" class="tab-btn active" type="button">График</button><button id="tabDataBtn" class="tab-btn" type="button">Данные</button></div>
<section id="tabChart" class="tab active"><div id="plot" style="height:560px;"></div></section>
<section id="tabData" class="tab">
<h3>Метаданные (T / Sigma / MDTV)</h3><div id="metaBlock" class="meta-grid"></div>
<h3>Группировка по индексам</h3><table id="groupTable"><thead><tr><th>Index</th><th>ISIN count</th></tr></thead><tbody></tbody></table>
<h3>Сводка (Ticker/ISIN/Indices/RankScore)</h3><table id="isinTable"><thead><tr><th>Ticker</th><th>ISIN</th><th>Indices</th><th>RankScore</th></tr></thead><tbody></tbody></table>
<h3>Каталог индексов акций</h3><table id="catalogTable"><thead><tr><th>Наименование индекса</th><th>Код индекса</th></tr></thead><tbody></tbody></table>
</section>
<script>
const DATA = __PAYLOAD__;
document.getElementById('generatedAt').textContent = DATA.generated_at;
const indexChecklistEl = document.getElementById('indexChecklist');
const tickerFilterEl = document.getElementById('tickerFilter');
const isinFilterEl = document.getElementById('isinFilter');
DATA.index_options
  .filter(o => o !== 'ALL')
  .forEach(o=>{
    const label=document.createElement('label');
    label.style.marginRight='10px';
    const cb=document.createElement('input');
    cb.type='checkbox';
    cb.value=o;
    cb.className='index-cb';
    cb.checked=true;
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + o));
    indexChecklistEl.appendChild(label);
  });
DATA.ticker_options.forEach(o=>{const op=document.createElement('option');op.value=o;op.textContent=o;tickerFilterEl.appendChild(op);});

function activateTab(name){
 document.getElementById('tabChart').classList.toggle('active', name==='chart');
 document.getElementById('tabData').classList.toggle('active', name==='data');
 document.getElementById('tabChartBtn').classList.toggle('active', name==='chart');
 document.getElementById('tabDataBtn').classList.toggle('active', name==='data');
}
document.getElementById('tabChartBtn').addEventListener('click', ()=>activateTab('chart'));
document.getElementById('tabDataBtn').addEventListener('click', ()=>activateTab('data'));

function render(){
 const selectedIndices = Array.from(document.querySelectorAll('.index-cb:checked')).map(el => el.value);
 const ticker = tickerFilterEl.value || 'ALL';
 const isinText = (isinFilterEl.value||'').trim().toUpperCase();
 const useAllIndices = selectedIndices.length === 0 || selectedIndices.length === document.querySelectorAll('.index-cb').length;

 const filteredIsins = DATA.isins.filter(r => {
   const byIndex = useAllIndices || selectedIndices.includes(r.Indices||'');
   const byTicker = ticker==='ALL' || (r.Ticker||'')===ticker;
   const byIsin = !isinText || (r.ISIN||'').toUpperCase().includes(isinText);
   return byIndex && byTicker && byIsin;
 });
 const allowed = new Set(filteredIsins.map(r=>r.ISIN));
 const curves = DATA.curves.filter(r=>allowed.has(r.ISIN));
 const allowedAfterCurve = new Set(curves.map(r=>r.ISIN));
 const filteredIsinsFinal = filteredIsins.filter(r => allowedAfterCurve.has(r.ISIN));

 const tracesMap = new Map();
 curves.forEach(r=>{
   const key = r.Ticker || r.ISIN;
   if(!tracesMap.has(key)) tracesMap.set(key, {x:[], y:[], name:key, mode:'lines+markers', type:'scatter'});
   tracesMap.get(key).x.push(Number(r.Q) / 1000000.0);
   tracesMap.get(key).y.push(Number(r.DeltaP) * 100.0);
 });
 Plotly.newPlot('plot', Array.from(tracesMap.values()), {xaxis:{title:'реализацию позиции в рынок (Млн. руб)'}, yaxis:{title:'изменение цены в %'}, margin:{t:20}}, {responsive:true});

 const metaBlock = document.getElementById('metaBlock'); metaBlock.innerHTML='';
 filteredIsinsFinal.forEach(r=>{
   const card=document.createElement('div'); card.className='meta-card';
   card.innerHTML = `<b>${r.Ticker || '-'} / ${r.ISIN || '-'}</b><br>T: ${r.T || '-'}<br>Sigma: ${r.Sigma || '-'}<br>MDTV: ${r.MDTV || '-'}`;
   metaBlock.appendChild(card);
 });

 const groupBody=document.querySelector('#groupTable tbody'); groupBody.innerHTML='';
 DATA.groups.filter(g=>useAllIndices||selectedIndices.includes(g.Index)).forEach(g=>{
   const tr=document.createElement('tr'); tr.innerHTML=`<td>${g.Index}</td><td>${g.IsinCount}</td>`; groupBody.appendChild(tr);
 });

 const isinBody=document.querySelector('#isinTable tbody'); isinBody.innerHTML='';
 filteredIsinsFinal.forEach(r=>{
   const tr=document.createElement('tr'); tr.innerHTML=`<td>${r.Ticker||''}</td><td>${r.ISIN||''}</td><td>${r.Indices||''}</td><td>${r.RankScore||0}</td>`; isinBody.appendChild(tr);
 });

 const catBody=document.querySelector('#catalogTable tbody'); catBody.innerHTML='';
 DATA.index_catalog.forEach(c=>{ const tr=document.createElement('tr'); tr.innerHTML=`<td>${c.name}</td><td>${c.code}</td>`; catBody.appendChild(tr); });
}

document.querySelectorAll('.index-cb').forEach(el=>el.addEventListener('change', render));
tickerFilterEl.addEventListener('change', render);
isinFilterEl.addEventListener('input', render);
render();
</script></body></html>"""

    return html_template.replace("__PAYLOAD__", payload_json).encode("utf-8")
