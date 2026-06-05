import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv"
)

OUT_PATH = Path("error_dashboard.html")

N_MC = 20
MASK_FRAC = 0.20
RANDOM_SEED = 42


def parse_contract(col):
    """
    Extract strike and option type from names like:
    NIFTY27JAN2625200CE
    NIFTY27JAN2623800PE
    """
    m = re.search(r"(\d+)(CE|PE)$", col)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def rmse(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x * x)))


def make_bin(x):
    if pd.isna(x):
        return "missing"
    if x < 0.10:
        return "<0.10"
    if x < 0.15:
        return "0.10-0.15"
    if x < 0.20:
        return "0.15-0.20"
    if x < 0.30:
        return "0.20-0.30"
    return ">0.30"


def impute_1d(strikes, values):
    """
    Same-row cross-sectional imputation over strikes.
    Interior gaps: linear interpolation.
    Edge gaps: nearest observed value extrapolation.
    """
    strikes = np.asarray(strikes, dtype=float)
    values = np.asarray(values, dtype=float)

    out = values.copy()
    ok = ~np.isnan(values)

    if ok.sum() == 0:
        return out

    if ok.sum() == 1:
        out[~ok] = values[ok][0]
        return out

    out[~ok] = np.interp(strikes[~ok], strikes[ok], values[ok])
    return out


def build_error_data(df):
    rng = np.random.default_rng(RANDOM_SEED)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
    df["date"] = df["datetime"].dt.strftime("%d-%m-%Y")
    df["hour"] = df["datetime"].dt.hour
    df["time_frac"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60

    contract_cols = []
    meta = {}

    for col in df.columns:
        parsed = parse_contract(col)
        if parsed:
            strike, opt_type = parsed
            contract_cols.append(col)
            meta[col] = {"strike": strike, "opt_type": opt_type}

    if not contract_cols:
        raise ValueError("No option columns found. Expected columns ending in CE or PE, like NIFTY27JAN2625200CE.")

    for col in contract_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []

    for mc in range(N_MC):
        for idx, row in df.iterrows():
            for opt_type in ["CE", "PE"]:
                cols = [c for c in contract_cols if meta[c]["opt_type"] == opt_type]
                cols = sorted(cols, key=lambda c: meta[c]["strike"])

                strikes = np.array([meta[c]["strike"] for c in cols], dtype=float)
                vals = row[cols].to_numpy(dtype=float)

                observed_idx = np.where(~np.isnan(vals))[0]

                if len(observed_idx) < 4:
                    continue

                n_mask = max(1, int(round(len(observed_idx) * MASK_FRAC)))
                mask_idx = rng.choice(observed_idx, size=n_mask, replace=False)

                masked_vals = vals.copy()
                masked_vals[mask_idx] = np.nan

                pred = impute_1d(strikes, masked_vals)

                observed_after_mask = np.where(~np.isnan(masked_vals))[0]

                for j in mask_idx:
                    true_iv = vals[j]
                    pred_iv = pred[j]

                    if np.isnan(true_iv) or np.isnan(pred_iv):
                        continue

                    left_obs = observed_after_mask[observed_after_mask < j]
                    right_obs = observed_after_mask[observed_after_mask > j]

                    is_edge = len(left_obs) == 0 or len(right_obs) == 0
                    n_obs = len(observed_after_mask)

                    rows.append(
                        {
                            "mc": mc,
                            "row_idx": idx,
                            "datetime": row["datetime"].strftime("%d-%m-%Y %H:%M"),
                            "date": row["date"],
                            "hour": int(row["hour"]),
                            "time_frac": float(row["time_frac"]),
                            "contract": cols[j],
                            "strike": int(strikes[j]),
                            "k_rank": int(j),
                            "opt_type": opt_type,
                            "true_iv": float(true_iv),
                            "pred_iv": float(pred_iv),
                            "abs_err": float(abs(pred_iv - true_iv)),
                            "err": float(pred_iv - true_iv),
                            "sq_err": float((pred_iv - true_iv) ** 2),
                            "cell_type": "edge" if is_edge else "interior",
                            "is_j27": row["date"] == "27-01-2026",
                            "n_obs": int(n_obs),
                            "iv_bin": make_bin(true_iv),
                        }
                    )

    err = pd.DataFrame(rows)

    if err.empty:
        raise ValueError("Monte Carlo validation produced no error rows. Check that dataset.csv has enough non-missing IV values.")

    def agg(group_cols):
        return (
            err.groupby(group_cols, dropna=False)
            .agg(
                rmse=("sq_err", lambda x: float(np.sqrt(np.mean(x)))),
                mae=("abs_err", "mean"),
                mean_iv=("true_iv", "mean"),
                n=("abs_err", "size"),
            )
            .reset_index()
        )

    summary = {
        "overall_rmse": rmse(err["err"]),
        "interior_rmse": rmse(err.loc[err["cell_type"] == "interior", "err"]),
        "edge_rmse": rmse(err.loc[err["cell_type"] == "edge", "err"]),
        "j27_rmse": rmse(err.loc[err["is_j27"], "err"]),
        "other_rmse": rmse(err.loc[~err["is_j27"], "err"]),
        "ce_rmse": rmse(err.loc[err["opt_type"] == "CE", "err"]),
        "pe_rmse": rmse(err.loc[err["opt_type"] == "PE", "err"]),
        "n_total": int(len(err)),
    }

    pct = (
        err.groupby("k_rank")["abs_err"]
        .quantile([0.25, 0.50, 0.75, 0.95])
        .unstack()
        .reset_index()
    )
    pct.columns = ["k_rank", "p25", "p50", "p75", "p95"]

    scatter = err.sample(min(1500, len(err)), random_state=RANDOM_SEED)[
        ["true_iv", "abs_err", "cell_type", "is_j27"]
    ].to_dict("records")

    D = {
        "summary": summary,
        "by_krank": agg(["k_rank", "opt_type"]).to_dict("records"),
        "error_pctiles_by_krank": pct.to_dict("records"),
        "by_iv": agg(["iv_bin", "cell_type"]).to_dict("records"),
        "by_hour_other": agg(["hour"]).query("hour == hour").to_dict("records"),
        "j27_time_agg": agg(["time_frac", "is_j27"]).query("is_j27 == True").drop(columns=["is_j27"]).to_dict("records"),
        "by_date": agg(["date"]).to_dict("records"),
        "by_nobs": agg(["n_obs"]).to_dict("records"),
        "heatmap_other": agg(["k_rank", "hour", "is_j27"]).query("is_j27 == False").drop(columns=["is_j27"]).to_dict("records"),
        "heatmap_j27": agg(["k_rank", "hour", "is_j27"]).query("is_j27 == True").drop(columns=["is_j27"]).to_dict("records"),
        "by_date_krank": agg(["date", "k_rank"]).to_dict("records"),
        "j27_by_time_krank": agg(["time_frac", "k_rank", "opt_type", "is_j27"]).query("is_j27 == True").drop(columns=["is_j27"]).to_dict("records"),
        "scatter": scatter,
        "by_type": agg(["cell_type", "is_j27"]).to_dict("records"),
    }

    return D


df = pd.read_csv(DATA_PATH)
D = build_error_data(df)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IV Imputer Error Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}}
.topbar{{background:linear-gradient(135deg,#1a1f2e,#252d3d);padding:18px 28px;border-bottom:1px solid #2d3748;display:flex;align-items:center;gap:16px}}
.topbar h1{{font-size:1.25rem;font-weight:700;color:#63b3ed}}
.badge{{background:#2a4365;color:#90cdf4;padding:3px 10px;border-radius:999px;font-size:.78rem}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}}
.stat{{background:#1a2540;border:1px solid #2d4a7a;border-radius:8px;padding:10px 16px;text-align:center}}
.stat .v{{font-size:1.35rem;font-weight:700;color:#63b3ed}}
.stat .l{{font-size:.72rem;color:#718096;margin-top:2px}}
.stats-row{{display:flex;gap:12px;padding:16px 24px;flex-wrap:wrap;background:#0f1117;border-bottom:1px solid #1e2a3a}}
.main{{padding:20px 24px;display:grid;gap:20px}}
.section{{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:20px}}
.section-title{{font-size:.95rem;font-weight:600;color:#90cdf4;margin-bottom:14px;display:flex;align-items:center;gap:8px}}
.section-title .tag{{background:#1a3a5c;color:#63b3ed;font-size:.7rem;padding:2px 8px;border-radius:4px}}
.charts-2col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.chart-wrap{{position:relative;height:280px}}
.chart-wrap-tall{{position:relative;height:360px}}
.slider-row{{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}}
.slider-row label{{font-size:.8rem;color:#a0aec0;white-space:nowrap}}
.slider-row select,.slider-row input[type=range]{{background:#252d3d;border:1px solid #3a4a5c;color:#e2e8f0;padding:4px 8px;border-radius:6px;font-size:.82rem}}
input[type=range]{{width:180px;accent-color:#63b3ed}}
.val-display{{font-size:.82rem;color:#63b3ed;font-weight:600;min-width:60px}}
.heatmap-wrap{{overflow-x:auto}}
table.heatmap{{border-collapse:collapse;font-size:.72rem;width:100%}}
table.heatmap th,table.heatmap td{{padding:6px 8px;text-align:center;border:1px solid #2d3748}}
table.heatmap th{{background:#252d3d;color:#90cdf4;font-weight:600}}
.insight-box{{background:#1a2e1a;border:1px solid #2d5a2d;border-radius:8px;padding:12px 16px;font-size:.8rem;color:#9ae6b4;margin-bottom:12px;line-height:1.6}}
.insight-box strong{{color:#68d391}}
.tabs{{display:flex;gap:4px;margin-bottom:14px;flex-wrap:wrap}}
.tab{{padding:5px 14px;border-radius:6px;font-size:.8rem;cursor:pointer;border:1px solid #2d3748;color:#a0aec0;background:transparent;transition:all .2s}}
.tab.active{{background:#2a4365;color:#90cdf4;border-color:#2a4365}}
.panel{{display:none}}
.panel.active{{display:block}}
</style>
</head>

<body>
<div class="topbar">
  <h1>📊 IV Imputer — Monte Carlo Error Analysis</h1>
  <div class="badges">
    <span class="badge">{len(df):,} timestamps</span>
    <span class="badge">{N_MC} MC iterations</span>
    <span class="badge">{MASK_FRAC:.0%} validation mask</span>
  </div>
</div>

<div class="stats-row" id="stats-row"></div>

<div class="main">

<div class="insight-box">
  <strong>Method:</strong>
  This dashboard was generated directly from <strong>dataset.csv</strong>.
  Known IV values are temporarily hidden, imputed using same-row strike interpolation, and compared against the true hidden IVs.
  So the errors shown here are Monte Carlo validation errors, not live missing-cell errors.
</div>

<div class="section">
  <div class="section-title">📈 Cross-Section Error Profile <span class="tag">SMILE POSITION</span></div>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('cs',0,this)">RMSE by Strike Rank</button>
    <button class="tab" onclick="switchTab('cs',1,this)">Error Percentiles</button>
    <button class="tab" onclick="switchTab('cs',2,this)">CE vs PE</button>
    <button class="tab" onclick="switchTab('cs',3,this)">By IV Level</button>
  </div>
  <div id="cs-panel-0" class="panel active"><div class="chart-wrap"><canvas id="cs0"></canvas></div></div>
  <div id="cs-panel-1" class="panel"><div class="chart-wrap"><canvas id="cs1"></canvas></div></div>
  <div id="cs-panel-2" class="panel"><div class="charts-2col"><div class="chart-wrap"><canvas id="cs2a"></canvas></div><div class="chart-wrap"><canvas id="cs2b"></canvas></div></div></div>
  <div id="cs-panel-3" class="panel"><div class="chart-wrap"><canvas id="cs3"></canvas></div></div>
</div>

<div class="section">
  <div class="section-title">⏱ Temporal Error Profile <span class="tag">TIME / DATE</span></div>
  <div class="tabs">
    <button class="tab active" onclick="switchTab('tm',0,this)">By Hour</button>
    <button class="tab" onclick="switchTab('tm',1,this)">Jan 27 Intraday</button>
    <button class="tab" onclick="switchTab('tm',2,this)">By Date</button>
    <button class="tab" onclick="switchTab('tm',3,this)">Observed Neighbors</button>
  </div>
  <div id="tm-panel-0" class="panel active"><div class="chart-wrap"><canvas id="tm0"></canvas></div></div>
  <div id="tm-panel-1" class="panel"><div class="chart-wrap"><canvas id="tm1"></canvas></div></div>
  <div id="tm-panel-2" class="panel"><div class="chart-wrap-tall"><canvas id="tm2"></canvas></div></div>
  <div id="tm-panel-3" class="panel"><div class="chart-wrap"><canvas id="tm3"></canvas></div></div>
</div>

<div class="section">
  <div class="section-title">🔥 2D Error Heatmap: Strike × Hour</div>
  <div class="slider-row">
    <label>Day type:</label>
    <select id="hm-day" onchange="renderHeatmap()">
      <option value="other">Non-Jan27 trading days</option>
      <option value="j27">Jan 27</option>
    </select>
  </div>
  <div class="heatmap-wrap" id="heatmap-container"></div>
</div>

<div class="section">
  <div class="section-title">📅 Strike Profile by Date</div>
  <div class="slider-row">
    <label>Date:</label>
    <input type="range" id="date-slider" min="0" max="1" value="0" step="1" oninput="updateDateChart(this.value)">
    <span class="val-display" id="date-label">--</span>
  </div>
  <div class="chart-wrap"><canvas id="date-chart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">🕐 Jan 27 Strike Profile by Time</div>
  <div class="slider-row">
    <label>Time:</label>
    <input type="range" id="j27-slider" min="0" max="1" value="0" step="1" oninput="updateJ27Chart(this.value)">
    <span class="val-display" id="j27-label">--</span>
  </div>
  <div class="charts-2col">
    <div class="chart-wrap"><canvas id="j27-ce-chart"></canvas></div>
    <div class="chart-wrap"><canvas id="j27-pe-chart"></canvas></div>
  </div>
</div>

<div class="section">
  <div class="section-title">🔵 Error vs True IV</div>
  <div class="slider-row">
    <label>Filter:</label>
    <select id="scatter-filter" onchange="renderScatter()">
      <option value="all">All cells</option>
      <option value="interior">Interior only</option>
      <option value="edge">Edge only</option>
      <option value="j27">Jan 27 only</option>
      <option value="other">Non-Jan27 only</option>
    </select>
  </div>
  <div class="chart-wrap-tall"><canvas id="scatter-chart"></canvas></div>
</div>

<div class="section">
  <div class="section-title">⚡ Edge vs Interior Error Decomposition</div>
  <div class="charts-2col">
    <div class="chart-wrap"><canvas id="ei1"></canvas></div>
    <div class="chart-wrap"><canvas id="ei2"></canvas></div>
  </div>
</div>

</div>

<script>
const D = {json.dumps(D)};

const BLUE='rgba(99,179,237,.85)', RED='rgba(252,129,129,.85)',
GREEN='rgba(104,211,145,.85)', ORANGE='rgba(246,173,85,.85)',
TEAL='rgba(129,230,217,.85)';

function rmseColor(v, lo=0, hi=0.03){{
  const t=Math.min(1,Math.max(0,(v-lo)/(hi-lo)));
  const r=Math.round(50+t*200), g=Math.round(200-t*150), b=Math.round(100-t*50);
  return `rgba(${{r}},${{g}},${{b}},0.85)`;
}}

function baseOpts(title){{
  return {{
    responsive:true,
    maintainAspectRatio:false,
    plugins:{{
      legend:{{labels:{{color:'#a0aec0',font:{{size:11}}}}}},
      title:{{display:!!title,text:title,color:'#90cdf4',font:{{size:12,weight:'600'}}}}
    }},
    scales:{{
      x:{{ticks:{{color:'#718096',font:{{size:10}}}},grid:{{color:'rgba(255,255,255,.05)'}}}},
      y:{{ticks:{{color:'#718096',font:{{size:10}}}},grid:{{color:'rgba(255,255,255,.07)'}}}}
    }}
  }};
}}

function newChart(id,type,data,opts){{
  const ctx=document.getElementById(id);
  if(!ctx) return null;
  if(ctx._chart) ctx._chart.destroy();
  const c=new Chart(ctx,{{type,data,options:opts}});
  ctx._chart=c;
  return c;
}}

function switchTab(prefix,idx,btn){{
  document.querySelectorAll(`[id^="${{prefix}}-panel-"]`).forEach(p=>p.classList.remove('active'));
  btn.closest('.section').querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(`${{prefix}}-panel-${{idx}}`).classList.add('active');
  btn.classList.add('active');
}}

const s=D.summary;
document.getElementById('stats-row').innerHTML=[
  ['Overall RMSE',(s.overall_rmse*1000).toFixed(2)+'‰'],
  ['Interior RMSE',(s.interior_rmse*1000).toFixed(2)+'‰'],
  ['Edge RMSE',(s.edge_rmse*1000).toFixed(2)+'‰'],
  ['Jan27 RMSE',(s.j27_rmse*1000).toFixed(2)+'‰'],
  ['Other RMSE',(s.other_rmse*1000).toFixed(2)+'‰'],
  ['CE RMSE',(s.ce_rmse*1000).toFixed(2)+'‰'],
  ['PE RMSE',(s.pe_rmse*1000).toFixed(2)+'‰'],
  ['Validation Points',s.n_total.toLocaleString()],
].map(([l,v])=>`<div class="stat"><div class="v">${{v}}</div><div class="l">${{l}}</div></div>`).join('');

(function(){{
  const raw=D.by_krank;
  const ranks=[...new Set(raw.map(r=>r.k_rank))].sort((a,b)=>a-b);
  const ce=ranks.map(k=>{{const r=raw.find(x=>x.k_rank===k&&x.opt_type==='CE'); return r?r.rmse*1000:null;}});
  const pe=ranks.map(k=>{{const r=raw.find(x=>x.k_rank===k&&x.opt_type==='PE'); return r?r.rmse*1000:null;}});
  const meanIV=ranks.map(k=>{{
    const rc=raw.filter(x=>x.k_rank===k);
    return rc.length?rc.reduce((a,b)=>a+b.mean_iv,0)/rc.length*100:null;
  }});
  newChart('cs0','bar',{{labels:ranks.map(k=>`K${{k}}`),datasets:[
    {{label:'CE RMSE (‰)',data:ce,backgroundColor:BLUE,yAxisID:'y'}},
    {{label:'PE RMSE (‰)',data:pe,backgroundColor:GREEN,yAxisID:'y'}},
    {{label:'Mean IV (%)',data:meanIV,type:'line',borderColor:ORANGE,backgroundColor:'transparent',pointRadius:3,yAxisID:'y2',borderWidth:2}}
  ]}},{{...baseOpts('RMSE by Strike Rank'),scales:{{...baseOpts().scales,
    y:{{...baseOpts().scales.y,position:'left'}},
    y2:{{position:'right',ticks:{{color:'#718096',font:{{size:10}}}},grid:{{drawOnChartArea:false}}}}
  }}}});
}})();

(function(){{
  const p=D.error_pctiles_by_krank;
  newChart('cs1','bar',{{labels:p.map(r=>`K${{r.k_rank}}`),datasets:[
    {{label:'p25',data:p.map(r=>r.p25*1000),backgroundColor:'rgba(104,211,145,.4)',stack:'s'}},
    {{label:'p50',data:p.map(r=>(r.p50-r.p25)*1000),backgroundColor:'rgba(104,211,145,.7)',stack:'s'}},
    {{label:'p75',data:p.map(r=>(r.p75-r.p50)*1000),backgroundColor:'rgba(246,173,85,.7)',stack:'s'}},
    {{label:'p95',data:p.map(r=>(r.p95-r.p75)*1000),backgroundColor:'rgba(252,129,129,.8)',stack:'s'}},
  ]}},{{...baseOpts('Absolute Error Percentiles'),scales:{{...baseOpts().scales,y:{{...baseOpts().scales.y,stacked:true}}}}}});
}})();

(function(){{
  const raw=D.by_krank;
  const ce=raw.filter(r=>r.opt_type==='CE').sort((a,b)=>a.k_rank-b.k_rank);
  const pe=raw.filter(r=>r.opt_type==='PE').sort((a,b)=>a.k_rank-b.k_rank);
  newChart('cs2a','bar',{{labels:ce.map(r=>`K${{r.k_rank}}`),datasets:[{{data:ce.map(r=>r.rmse*1000),backgroundColor:ce.map(r=>rmseColor(r.rmse))}}]}},{{...baseOpts('CE Error by Rank'),plugins:{{...baseOpts().plugins,legend:{{display:false}}}}}});
  newChart('cs2b','bar',{{labels:pe.map(r=>`K${{r.k_rank}}`),datasets:[{{data:pe.map(r=>r.rmse*1000),backgroundColor:pe.map(r=>rmseColor(r.rmse))}}]}},{{...baseOpts('PE Error by Rank'),plugins:{{...baseOpts().plugins,legend:{{display:false}}}}}});
}})();

(function(){{
  const raw=D.by_iv;
  const bins=[...new Set(raw.map(r=>r.iv_bin))];
  const interior=bins.map(b=>{{const r=raw.find(x=>x.iv_bin===b&&x.cell_type==='interior'); return r?r.rmse*1000:0;}});
  const edge=bins.map(b=>{{const r=raw.find(x=>x.iv_bin===b&&x.cell_type==='edge'); return r?r.rmse*1000:0;}});
  newChart('cs3','bar',{{labels:bins,datasets:[
    {{label:'Interior',data:interior,backgroundColor:BLUE}},
    {{label:'Edge',data:edge,backgroundColor:RED}}
  ]}},baseOpts('RMSE by True IV Level'));
}})();

(function(){{
  const h=D.by_hour_other;
  newChart('tm0','line',{{labels:h.map(r=>`${{r.hour}}:00`),datasets:[{{label:'RMSE (‰)',data:h.map(r=>r.rmse*1000),borderColor:BLUE,backgroundColor:'rgba(99,179,237,.1)',fill:true,tension:.4}}]}},baseOpts('RMSE by Hour'));
}})();

const toHHMM=t=>{{const h=Math.floor(t),m=Math.round((t-h)*60);return`${{h}}:${{String(m).padStart(2,'0')}}`;}};

(function(){{
  const h=D.j27_time_agg.sort((a,b)=>a.time_frac-b.time_frac);
  newChart('tm1','line',{{labels:h.map(r=>toHHMM(r.time_frac)),datasets:[
    {{label:'RMSE (‰)',data:h.map(r=>r.rmse*1000),borderColor:RED,backgroundColor:'rgba(252,129,129,.1)',fill:true,tension:.4}},
    {{label:'Mean IV',data:h.map(r=>r.mean_iv),borderColor:ORANGE,backgroundColor:'transparent',tension:.4,yAxisID:'y2'}}
  ]}},{{...baseOpts('Jan 27 Intraday Error'),scales:{{...baseOpts().scales,y2:{{position:'right',ticks:{{color:'#718096'}},grid:{{drawOnChartArea:false}}}}}}}});
}})();

(function(){{
  const d=D.by_date;
  newChart('tm2','bar',{{labels:d.map(r=>String(r.date).slice(0,5)),datasets:[{{data:d.map(r=>r.rmse*1000),backgroundColor:d.map(r=>r.date==='27-01-2026'?RED:BLUE)}}]}},{{...baseOpts('RMSE by Date'),plugins:{{...baseOpts().plugins,legend:{{display:false}}}}}});
}})();

(function(){{
  const n=D.by_nobs;
  newChart('tm3','line',{{labels:n.map(r=>`${{r.n_obs}} obs`),datasets:[{{label:'RMSE (‰)',data:n.map(r=>r.rmse*1000),borderColor:TEAL,backgroundColor:'rgba(129,230,217,.1)',fill:true,tension:.3}}]}},baseOpts('RMSE vs Observed Points'));
}})();

function renderHeatmap(){{
  const day=document.getElementById('hm-day').value;
  const raw=day==='j27'?D.heatmap_j27:D.heatmap_other;
  if(!raw.length){{document.getElementById('heatmap-container').innerHTML='<p>No heatmap data.</p>';return;}}
  const ranks=[...new Set(raw.map(r=>r.k_rank))].sort((a,b)=>a-b);
  const hours=[...new Set(raw.map(r=>r.hour))].sort((a,b)=>a-b);
  const lookup={{}};
  raw.forEach(r=>lookup[`${{r.k_rank}}_${{r.hour}}`]=r.rmse);
  const maxV=Math.max(...raw.map(r=>r.rmse));
  let html='<table class="heatmap"><tr><th>Strike↓ Hour→</th>';
  hours.forEach(h=>html+=`<th>${{h}}:00</th>`);
  html+='</tr>';
  ranks.forEach(k=>{{
    html+=`<tr><th>K${{k}}</th>`;
    hours.forEach(h=>{{
      const v=lookup[`${{k}}_${{h}}`];
      if(v==null){{html+='<td>-</td>';return;}}
      const t=v/maxV;
      const r=Math.round(40+t*215),g=Math.round(210-t*160),b=Math.round(100-t*60);
      html+=`<td style="background:rgba(${{r}},${{g}},${{b}},.75);color:#fff;font-weight:600">${{(v*1000).toFixed(1)}}</td>`;
    }});
    html+='</tr>';
  }});
  html+='</table>';
  document.getElementById('heatmap-container').innerHTML=html;
}}
renderHeatmap();

const dates=[...new Set(D.by_date_krank.map(r=>r.date))].sort();
document.getElementById('date-slider').max=Math.max(0,dates.length-1);
let dateChart=null;
function updateDateChart(idx){{
  const d=dates[Number(idx)];
  if(!d)return;
  document.getElementById('date-label').textContent=d;
  const rows=D.by_date_krank.filter(r=>r.date===d).sort((a,b)=>a.k_rank-b.k_rank);
  if(dateChart)dateChart.destroy();
  dateChart=new Chart(document.getElementById('date-chart'),{{type:'bar',data:{{labels:rows.map(r=>`K${{r.k_rank}}`),datasets:[{{data:rows.map(r=>r.rmse*1000),backgroundColor:rows.map(r=>rmseColor(r.rmse))}}]}},options:{{...baseOpts(`Strike Error — ${{d}}`),plugins:{{...baseOpts().plugins,legend:{{display:false}}}}}}}});
}}
updateDateChart(0);

const j27times=[...new Set(D.j27_by_time_krank.map(r=>r.time_frac))].sort((a,b)=>a-b);
document.getElementById('j27-slider').max=Math.max(0,j27times.length-1);
let j27ceChart=null,j27peChart=null;
function updateJ27Chart(idx){{
  const tf=j27times[Number(idx)];
  if(tf==null)return;
  document.getElementById('j27-label').textContent=toHHMM(tf);
  ['CE','PE'].forEach(ot=>{{
    const rows=D.j27_by_time_krank.filter(r=>r.time_frac===tf&&r.opt_type===ot).sort((a,b)=>a.k_rank-b.k_rank);
    const id=ot==='CE'?'j27-ce-chart':'j27-pe-chart';
    if(ot==='CE'&&j27ceChart)j27ceChart.destroy();
    if(ot==='PE'&&j27peChart)j27peChart.destroy();
    const chart=new Chart(document.getElementById(id),{{type:'bar',data:{{labels:rows.map(r=>`K${{r.k_rank}}`),datasets:[{{data:rows.map(r=>r.rmse*1000),backgroundColor:rows.map(r=>rmseColor(r.rmse,0,0.05))}}]}},options:{{...baseOpts(`Jan27 ${{ot}} — ${{toHHMM(tf)}}`),plugins:{{...baseOpts().plugins,legend:{{display:false}}}}}}}});
    if(ot==='CE')j27ceChart=chart;else j27peChart=chart;
  }});
}}
updateJ27Chart(0);

let scatterChart=null;
function renderScatter(){{
  const f=document.getElementById('scatter-filter').value;
  let data=D.scatter;
  if(f==='interior')data=data.filter(r=>r.cell_type==='interior');
  else if(f==='edge')data=data.filter(r=>r.cell_type==='edge');
  else if(f==='j27')data=data.filter(r=>r.is_j27);
  else if(f==='other')data=data.filter(r=>!r.is_j27);
  const pts=data.map(r=>({{x:r.true_iv,y:r.abs_err*1000}}));
  if(scatterChart)scatterChart.destroy();
  scatterChart=new Chart(document.getElementById('scatter-chart'),{{type:'scatter',data:{{datasets:[{{label:'Cells',data:pts,backgroundColor:'rgba(99,179,237,.35)',pointRadius:3}}]}},options:{{...baseOpts('Absolute Error vs True IV'),scales:{{x:{{...baseOpts().scales.x,title:{{display:true,text:'True IV',color:'#718096'}}}},y:{{...baseOpts().scales.y,title:{{display:true,text:'|Error| (‰)',color:'#718096'}}}}}}}}}});
}}
renderScatter();

(function(){{
  const t=D.by_type;
  const cats=['interior/other','edge/other','interior/j27','edge/j27'];
  const vals=cats.map(c=>{{
    const [ct,dg]=c.split('/');
    const isj=dg==='j27';
    const r=t.find(x=>x.cell_type===ct&&Boolean(x.is_j27)===isj);
    return r?r.rmse*1000:0;
  }});
  newChart('ei1','bar',{{labels:['Interior Other','Edge Other','Interior Jan27','Edge Jan27'],datasets:[{{data:vals,backgroundColor:[BLUE,ORANGE,RED,'rgba(252,80,80,.9)']}}]}},{{...baseOpts('RMSE by Cell Type × Day'),plugins:{{...baseOpts().plugins,legend:{{display:false}}}}}});

  const mse=r=>r?r.rmse**2*r.n:0;
  const items=[
    t.find(x=>x.cell_type==='interior'&&!x.is_j27),
    t.find(x=>x.cell_type==='edge'&&!x.is_j27),
    t.find(x=>x.cell_type==='interior'&&x.is_j27),
    t.find(x=>x.cell_type==='edge'&&x.is_j27)
  ];
  const totals=items.map(mse);
  const tot=totals.reduce((a,b)=>a+b,0)||1;
  newChart('ei2','doughnut',{{labels:['Interior Other','Edge Other','Interior Jan27','Edge Jan27'],datasets:[{{data:totals.map(v=>(v/tot*100).toFixed(1)),backgroundColor:[BLUE,ORANGE,RED,'rgba(252,80,80,.9)'],borderColor:'#1a1f2e'}}]}},baseOpts('Share of Total Squared Error (%)'));
}})();
</script>
</body>
</html>
"""

OUT_PATH.write_text(html, encoding="utf-8")
print(f"Written {OUT_PATH.resolve()} ({len(html)//1024} KB)")