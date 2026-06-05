#!/usr/bin/env python3
"""
Monte Carlo Error Analysis for try.py-style IV Imputer
=====================================================

This repeatedly creates synthetic missing IV cells from observed dataset cells,
runs the pasted try.py imputer logic, and generates error reports/graphs showing
where the cross-section method fails.

Outputs in --out-dir:
  mc_error_rows.csv
  worst_500_errors.csv
  mc_metrics_summary.csv/json
  group_metrics_*.csv
  plots/*.png
  interactive_error_dashboard.html

Run:
  python monte_carlo_error_analysis_try.py --data dataset.csv --out-dir mc_error_report --n-runs 25
"""

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


DEFAULT_DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv")
EPS_IV = 1e-6
SEPARATOR = "||"

BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
EDGE_LOCAL_POLY_BW = 2e-4
EDGE_BLEND_CLAUDE = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14
MIN_EDGE_LOCAL_NEIGHBORS = 3
LOCAL_POLY_DEGREE = 2

MONEYNESS_BINS = [-np.inf, 0.990, 0.995, 1.000, 1.005, 1.010, np.inf]
MONEYNESS_LABELS = ["<0.990", "0.990-0.995", "0.995-1.000", "1.000-1.005", "1.005-1.010", ">1.010"]


def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo synthetic-missing error analysis for try.py imputer.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out-dir", type=str, default="mc_error_report_try")
    p.add_argument("--n-runs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-mode", choices=["pattern", "frac"], default="pattern")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--min-hidden-per-run", type=int, default=500)
    p.add_argument("--max-hidden-per-row-type", type=int, default=None)
    p.add_argument("--dashboard-run", type=int, default=0)
    p.add_argument("--max-dashboard-frames", type=int, default=120)
    p.add_argument("--no-static-plots", action="store_true")
    p.add_argument("--no-html", action="store_true")
    return p.parse_args()


def parse_metadata(df):
    pattern = re.compile(r"^(?P<underlying>[A-Z]+)(?P<expiry>\d{2}[A-Z]{3}\d{2})(?P<strike>\d+)(?P<option_type>CE|PE)$")
    rows = []
    for col in df.columns:
        if col in {"datetime", "datetime_parsed", "underlying_price"}:
            continue
        m = pattern.match(col)
        if m:
            d = m.groupdict()
            d["column"] = col
            d["strike"] = int(d["strike"])
            d["expiry_date"] = pd.to_datetime(d["expiry"], format="%d%b%y", errors="coerce")
            rows.append(d)
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise ValueError("No option columns parsed. Check column names.")
    return meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)


def safe_iv(x):
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


def component_value(already_filled, col, key):
    item = already_filled.get(col)
    if isinstance(item, dict):
        value = item.get(key, item.get("final", np.nan))
    else:
        value = item
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {k: np.nan for k in ["mse", "rmse", "mae", "median_abs_error", "bias_mean_error", "error_std", "max_abs_error", "p90_abs_error", "p95_abs_error", "p99_abs_error"]} | {"n": 0}
    err = y_pred - y_true
    ae = np.abs(err)
    se = err ** 2
    return {
        "n": int(len(y_true)),
        "mse": float(np.mean(se)),
        "rmse": float(np.sqrt(np.mean(se))),
        "mae": float(np.mean(ae)),
        "median_abs_error": float(np.median(ae)),
        "bias_mean_error": float(np.mean(err)),
        "error_std": float(np.std(err)),
        "max_abs_error": float(np.max(ae)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
        "p95_abs_error": float(np.quantile(ae, 0.95)),
        "p99_abs_error": float(np.quantile(ae, 0.99)),
    }


def group_metrics(df, cols):
    rows = []
    for keys, g in df.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(cols, keys)}
        row.update(compute_metrics(g["actual_iv"], g["predicted_iv"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mse", "n"], ascending=[False, False]).reset_index(drop=True)
    return out


def fit_quadratic(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) == 0:
        return None, "no_points"
    if len(y) == 1:
        return np.array([0.0, 0.0, float(y[0])]), "constant"
    if len(y) == 2:
        c = np.polyfit(x, y, 1)
        return np.array([0.0, float(c[0]), float(c[1])]), "linear"
    c = np.polyfit(x, y, 2)
    return np.array([float(c[0]), float(c[1]), float(c[2])]), "quadratic"


def eval_quadratic(coeff, x):
    if coeff is None:
        return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    x_obs, y_obs = np.asarray(x_obs, float), np.asarray(y_obs, float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs, y_obs = x_obs[mask], y_obs[mask]
    if len(y_obs) == 0:
        return np.nan
    if len(y_obs) == 1:
        return safe_iv(float(y_obs[0]))
    actual_degree = min(degree, len(y_obs) - 1)
    dx = x_obs - x_target
    w = np.exp(-(dx ** 2) / (2.0 * bandwidth))
    X = np.column_stack([dx ** j for j in range(actual_degree + 1)])
    WX = X * w[:, None]
    lhs = X.T @ WX
    rhs = X.T @ (w * y_obs)
    try:
        coeff = np.linalg.solve(lhs, rhs)
        return safe_iv(float(coeff[0]))
    except np.linalg.LinAlgError:
        wsum = float(w.sum())
        if wsum <= 1e-15:
            return np.nan
        return safe_iv(float((w @ y_obs) / wsum))


def local_poly_wls_loo_preds(x_obs, y_obs, bandwidth):
    preds = np.full(len(y_obs), np.nan)
    for i in range(len(y_obs)):
        preds[i] = local_poly_wls_pred(np.delete(x_obs, i), np.delete(y_obs, i), x_obs[i], bandwidth)
    return preds


def select_bandwidth_by_loo(x_obs, y_obs, bandwidth_grid=BANDWIDTH_GRID):
    x_obs, y_obs = np.asarray(x_obs, float), np.asarray(y_obs, float)
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
    best_bw = float(bandwidth_grid[len(bandwidth_grid) // 2])
    best_mse = np.inf
    for bw in bandwidth_grid:
        loo = local_poly_wls_loo_preds(x_obs, y_obs, bw)
        valid = np.isfinite(loo) & np.isfinite(y_obs)
        if not valid.any():
            continue
        mse = float(np.mean((loo[valid] - y_obs[valid]) ** 2))
        if mse < best_mse:
            best_mse, best_bw = mse, float(bw)
    return best_bw, best_mse


def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []
    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    x = np.array([strike_map[c] / spot for c in obs_cols], float)
    y = np.array([row[c] for c in obs_cols], float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], [c for c, ok in zip(obs_cols, mask) if ok]


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = [{"column": c, "strike": strike_map[c], "is_missing": pd.isna(row[c]), "iv": row[c]} for c in cols_by_type[opt_type]]
    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)


def get_edge_blocks(row, opt_type, cols_by_type, strike_map):
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    left_block = []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]):
            left_block.append(rec["column"])
        else:
            break
    right_block = []
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]):
            right_block.append(rec["column"])
        else:
            break
    return state, list(reversed(left_block)), list(reversed(right_block))


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
    state, left_fill_order, right_fill_order = get_edge_blocks(row, opt_type, cols_by_type, strike_map)
    if target_col in set(left_fill_order):
        return True, "edge_no_left_observed", "left", left_fill_order, left_fill_order.index(target_col)
    if target_col in set(right_fill_order):
        return True, "edge_no_right_observed", "right", right_fill_order, right_fill_order.index(target_col)
    same_side_missing = set(state.loc[state["is_missing"], "column"])
    same_side_observed = set(state.loc[~state["is_missing"], "column"])
    if target_col in same_side_missing and len(same_side_observed) == 0:
        return True, "edge_no_observed_same_side", "all_missing", list(state["column"]), 0
    return False, "not_edge", "", [], np.nan


def local_shape_features_for_target(row, target_col, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return {"local_abs_slope": np.nan, "local_abs_curvature": np.nan, "local_shape_bucket": "unknown", "nearest_gap_left": np.nan, "nearest_gap_right": np.nan}
    target_x = strike_map[target_col] / spot
    x_obs, y_obs, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)
    if len(y_obs) < 3:
        return {"local_abs_slope": np.nan, "local_abs_curvature": np.nan, "local_shape_bucket": "unknown_lt3", "nearest_gap_left": np.nan, "nearest_gap_right": np.nan}
    order = np.argsort(x_obs)
    x, y = x_obs[order], y_obs[order]
    left_idx = np.where(x < target_x)[0]
    right_idx = np.where(x > target_x)[0]
    nearest_gap_left = target_x - x[left_idx[-1]] if len(left_idx) else np.nan
    nearest_gap_right = x[right_idx[0]] - target_x if len(right_idx) else np.nan
    nearest = np.argsort(np.abs(x - target_x))[:min(5, len(x))]
    xn, yn = x[nearest], y[nearest]
    oo = np.argsort(xn)
    xn, yn = xn[oo], yn[oo]
    coeff, _ = fit_quadratic(xn, yn)
    if coeff is not None:
        a, b, _ = coeff
        abs_slope = abs(float(2 * a * target_x + b))
        abs_curv = abs(float(2 * a))
    else:
        abs_slope, abs_curv = np.nan, np.nan
    if np.isfinite(abs_curv):
        bucket = "straight_low_curvature" if abs_curv < 100 else ("moderate_curvature" if abs_curv < 1000 else "high_curvature")
    else:
        bucket = "unknown"
    return {"local_abs_slope": abs_slope, "local_abs_curvature": abs_curv, "local_shape_bucket": bucket, "nearest_gap_left": nearest_gap_left, "nearest_gap_right": nearest_gap_right}


def predict_non_edge_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols = collect_same_row_points(row, opt_type, cols_by_type, strike_map)
    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {"prediction": float(global_median_iv), "source": "same_row_fallback_global_median", "selected_model": "fallback_global_median", "bandwidth": np.nan, "loo_mse": np.nan, "n_train": len(y_obs), "used_cols": used_cols}
    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw)
    selected_model = "local_quadratic_wls" if np.isfinite(pred) else "fallback_global_median"
    if not np.isfinite(pred): pred = global_median_iv
    return {"prediction": safe_iv(pred), "source": "same_row_non_edge_local_poly_wls", "selected_model": selected_model, "bandwidth": best_bw, "loo_mse": loo_mse, "n_train": len(y_obs), "used_cols": used_cols}


def collect_edge_training_points_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan}
    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge", "edge_block_size": 0, "edge_position_in_block": np.nan}
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs_records = []
    for _, rec in state.iterrows():
        col = rec["column"]
        if pd.notna(row[col]):
            obs_records.append({"column": col, "strike": strike_map[col], "x": strike_map[col] / spot, "y": float(row[col])})
    if not obs_records:
        return np.array([]), np.array([]), [], {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}
    obs = pd.DataFrame(obs_records)
    target_strike = strike_map[target_col]
    if side == "right":
        base = obs[obs["strike"] < target_strike].sort_values("strike")
    elif side == "left":
        base = obs[obs["strike"] > target_strike].sort_values("strike")
    else:
        base = obs.sort_values("strike")
    x_train, y_train = base["x"].to_list(), base["y"].to_list()
    used_cols = base["column"].astype(str).to_list()
    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "claude")
        if np.isfinite(prev_pred):
            x_train.append(float(prev_pred))
            y_train.append(float(prev_pred))
            used_cols.append(f"{prev_col}*as_xy")
    return np.asarray(x_train, float), np.asarray(y_train, float), used_cols, {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}


def collect_edge_training_points_corrected(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan, "edge_observed_side_points": 0}
    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge", "edge_block_size": 0, "edge_position_in_block": np.nan, "edge_observed_side_points": 0}
    target_strike = strike_map[target_col]
    train_records = []
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    for _, rec in state.iterrows():
        col, val = rec["column"], row[rec["column"]]
        if pd.isna(val): continue
        strike = strike_map[col]
        if (side == "right" and strike < target_strike) or (side == "left" and strike > target_strike):
            train_records.append({"column": col, "strike": strike, "x": strike / spot, "y": float(val), "is_predicted": False})
    observed_side_points = len(train_records)
    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "corrected")
        if np.isfinite(prev_pred):
            train_records.append({"column": prev_col, "strike": strike_map[prev_col], "x": strike_map[prev_col] / spot, "y": float(prev_pred), "is_predicted": True})
    train = pd.DataFrame(train_records)
    if train.empty:
        return np.array([]), np.array([]), [], {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position, "edge_observed_side_points": observed_side_points}
    train = train.sort_values("strike").reset_index(drop=True)
    used_cols = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]
    return train["x"].to_numpy(float), train["y"].to_numpy(float), used_cols, {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position, "edge_observed_side_points": observed_side_points}


def collect_edge_training_points_quadratic(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan, "edge_base_observed_needed": 0}
    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge", "edge_block_size": 0, "edge_position_in_block": np.nan, "edge_base_observed_needed": 0}
    target_strike = strike_map[target_col]
    base_needed = max(MIN_EDGE_LOCAL_NEIGHBORS, len(block_cols))
    obs_records = []
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    for _, rec in state.iterrows():
        col = rec["column"]
        if pd.notna(row[col]):
            obs_records.append({"column": col, "strike": strike_map[col], "x": strike_map[col] / spot, "y": float(row[col]), "is_predicted": False})
    obs = pd.DataFrame(obs_records)
    if obs.empty:
        return np.array([]), np.array([]), [], {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position, "edge_base_observed_needed": base_needed}
    if side == "right":
        base = obs[obs["strike"] < target_strike].sort_values("strike", ascending=False).head(base_needed).sort_values("strike")
    elif side == "left":
        base = obs[obs["strike"] > target_strike].sort_values("strike", ascending=True).head(base_needed).sort_values("strike")
    else:
        base = obs.sort_values("strike")
    train_records = base.to_dict(orient="records")
    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "quadratic")
        if np.isfinite(prev_pred):
            train_records.append({"column": prev_col, "strike": strike_map[prev_col], "x": strike_map[prev_col] / spot, "y": float(prev_pred), "is_predicted": True})
    train = pd.DataFrame(train_records)
    if train.empty:
        return np.array([]), np.array([]), [], {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position, "edge_base_observed_needed": base_needed}
    train = train.sort_values("strike").reset_index(drop=True)
    used_cols = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]
    return train["x"].to_numpy(float), train["y"].to_numpy(float), used_cols, {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position, "edge_base_observed_needed": base_needed}


def predict_edge_corrected_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_corrected(row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used_cols, edge_info
    pred = local_poly_wls_pred(x_obs, y_obs, strike_map[target_col] / spot, EDGE_LOCAL_POLY_BW)
    if not np.isfinite(pred): pred = global_median_iv
    return safe_iv(pred), used_cols, edge_info


def predict_edge_quadratic(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_quadratic(row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used_cols, edge_info, np.nan
    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    pred = eval_quadratic(coeff, strike_map[target_col] / spot)
    if not np.isfinite(pred): pred = global_median_iv
    return safe_iv(pred), used_cols, edge_info, fit_kind


def predict_edge_claude_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return {"prediction": float(global_median_iv), "source": "edge_fallback_global_median_bad_spot", "selected_model": "fallback_global_median", "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan, "n_train": 0, "used_cols": [], "edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan}
    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    if len(y_obs) == 0:
        return {"prediction": float(global_median_iv), "source": "edge_fallback_global_median_no_neighbors", "selected_model": "fallback_global_median", "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan, "n_train": 0, "used_cols": used_cols, **edge_info}
    pred = local_poly_wls_pred(x_obs, y_obs, strike_map[target_col] / spot, EDGE_LOCAL_POLY_BW)
    selected_model = "edge_claude_progressive_local_poly_wls" if np.isfinite(pred) else "fallback_global_median"
    if not np.isfinite(pred): pred = global_median_iv
    fitted = np.array([local_poly_wls_pred(x_obs, y_obs, x, EDGE_LOCAL_POLY_BW) for x in x_obs], float)
    mask = np.isfinite(fitted) & np.isfinite(y_obs)
    fit_mse = float(np.mean((fitted[mask] - y_obs[mask]) ** 2)) if mask.any() else np.nan
    return {"prediction": safe_iv(pred), "source": "edge_claude_progressive_local_poly_wls", "selected_model": selected_model, "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": fit_mse, "n_train": len(y_obs), "used_cols": used_cols, **edge_info}


def predict_edge_ensemble(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    claude_info = predict_edge_claude_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled)
    corrected_pred, corrected_cols, corrected_info = predict_edge_corrected_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled)
    quadratic_pred, quadratic_cols, quadratic_info, quadratic_fit_kind = predict_edge_quadratic(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled)
    components = {"claude": safe_iv(claude_info["prediction"]), "corrected": safe_iv(corrected_pred), "quadratic": safe_iv(quadratic_pred)}
    pred = EDGE_BLEND_CLAUDE * components["claude"] + EDGE_BLEND_CORRECTED * components["corrected"] + EDGE_BLEND_QUADRATIC * components["quadratic"]
    selected_model = "edge_blended_progressive_wls_quadratic" if np.isfinite(pred) else "fallback_global_median"
    if not np.isfinite(pred): pred = global_median_iv
    return {**claude_info, "prediction": safe_iv(pred), "source": "edge_blended_progressive_wls_quadratic", "selected_model": selected_model, "component_predictions": components, "corrected_used_cols": corrected_cols, "quadratic_used_cols": quadratic_cols, "quadratic_fit_kind": quadratic_fit_kind, "edge_observed_side_points": corrected_info.get("edge_observed_side_points", np.nan), "edge_base_observed_needed": quadratic_info.get("edge_base_observed_needed", np.nan)}


def predict_cell(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    edge, edge_reason, _, _, _ = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if edge:
        info = predict_edge_ensemble(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled)
    else:
        info = predict_non_edge_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv)
    info["edge"] = bool(edge)
    info["edge_reason"] = edge_reason
    return info


def build_missing_cell_fill_order(df, cols_by_type, strike_map):
    missing_cells = []
    for row_idx in df.index:
        row = df.loc[row_idx]
        for opt_type in ["CE", "PE"]:
            state, left_fill_order, right_fill_order = get_edge_blocks(row, opt_type, cols_by_type, strike_map)
            missing_side_cols = [c for c in state["column"].tolist() if pd.isna(row[c])]
            if not missing_side_cols: continue
            edge_set = set(left_fill_order) | set(right_fill_order)
            interior = [c for c in state["column"].tolist() if c in missing_side_cols and c not in edge_set]
            ordered = left_fill_order + interior + [c for c in right_fill_order if c not in left_fill_order]
            for col in ordered:
                missing_cells.append((row_idx, col))
    return missing_cells


def fill_dataset(df_masked, option_cols, cols_by_type, strike_map, type_map, global_median_iv):
    filled = df_masked.copy()
    rows = []
    missing_cells = build_missing_cell_fill_order(df_masked, cols_by_type, strike_map)
    filled_values_by_row = {}
    for row_idx, col in missing_cells:
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})
        info = predict_cell(df_masked, row_idx, col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled)
        pred = info["prediction"]
        if not np.isfinite(pred): pred = global_median_iv
        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred
        comps = info.get("component_predictions", {})
        filled_values_by_row[row_idx][col] = {"final": pred, "claude": comps.get("claude", pred), "corrected": comps.get("corrected", pred), "quadratic": comps.get("quadratic", pred)}
        rows.append({
            "row_index": row_idx, "datetime": df_masked.loc[row_idx, "datetime"], "contract": col, "option_type": opt_type, "strike": strike_map[col], "final_prediction": pred,
            "edge": info["edge"], "edge_reason": info["edge_reason"], "source": info["source"], "selected_model": info["selected_model"], "bandwidth": info["bandwidth"], "loo_mse": info["loo_mse"], "n_train": info["n_train"],
            "used_cols": "|".join(map(str, info["used_cols"])), "edge_claude_prediction": comps.get("claude", np.nan), "edge_corrected_prediction": comps.get("corrected", np.nan), "edge_quadratic_prediction": comps.get("quadratic", np.nan),
            "edge_side": info.get("edge_side", ""), "edge_block_size": info.get("edge_block_size", np.nan), "edge_position_in_block": info.get("edge_position_in_block", np.nan), "edge_observed_side_points": info.get("edge_observed_side_points", np.nan), "edge_base_observed_needed": info.get("edge_base_observed_needed", np.nan)
        })
    return filled, pd.DataFrame(rows)


def build_fake_holdout_pattern(original_df, option_cols, cols_by_type, rng, min_hidden_per_run=500, max_hidden_per_row_type=None):
    hidden = set()
    for row_idx, row in original_df.iterrows():
        for opt_type in ["CE", "PE"]:
            side_cols = cols_by_type[opt_type]
            k = int(row[side_cols].isna().sum())
            if max_hidden_per_row_type is not None: k = min(k, max_hidden_per_row_type)
            obs = [c for c in side_cols if pd.notna(row[c])]
            if obs and k > 0:
                for col in rng.choice(obs, size=min(k, len(obs)), replace=False):
                    hidden.add((int(row_idx), str(col)))
    all_obs = [(int(i), str(c)) for i in original_df.index for c in option_cols if pd.notna(original_df.at[i, c])]
    if len(hidden) < min_hidden_per_run and all_obs:
        candidates = [x for x in all_obs if x not in hidden]
        extra_needed = min(min_hidden_per_run - len(hidden), len(candidates))
        if extra_needed > 0:
            for j in rng.choice(len(candidates), size=extra_needed, replace=False):
                hidden.add(candidates[int(j)])
    return sorted(hidden)


def build_fake_holdout_frac(original_df, option_cols, rng, holdout_frac=0.15):
    observed = [(int(i), str(c)) for i in original_df.index for c in option_cols if pd.notna(original_df.at[i, c])]
    n_hide = max(1, min(int(round(len(observed) * holdout_frac)), len(observed)))
    return sorted([observed[int(j)] for j in rng.choice(len(observed), size=n_hide, replace=False)])


def apply_fake_holdout(original_df, hidden_cells):
    masked = original_df.copy()
    truth = []
    for row_idx, col in hidden_cells:
        truth.append({"row_index": row_idx, "datetime": original_df.loc[row_idx, "datetime"], "contract": col, "actual_iv": float(original_df.at[row_idx, col])})
        masked.at[row_idx, col] = np.nan
    return masked, pd.DataFrame(truth)


def build_error_rows_for_run(run_id, original_df, masked_df, filled_df, fill_diag, truth, type_map, strike_map, cols_by_type):
    diag_lookup = {(int(r["row_index"]), r["contract"]): r for r in fill_diag.to_dict(orient="records")} if not fill_diag.empty else {}
    rows = []
    for rec in truth.to_dict(orient="records"):
        row_idx, col = int(rec["row_index"]), rec["contract"]
        actual, pred = float(rec["actual_iv"]), float(filled_df.at[row_idx, col])
        err = pred - actual
        opt_type, strike = type_map[col], strike_map[col]
        dtp = original_df.loc[row_idx, "datetime_parsed"]
        spot = original_df.loc[row_idx, "underlying_price"]
        moneyness = strike / spot if pd.notna(spot) and spot > 0 else np.nan
        d = diag_lookup.get((row_idx, col), {})
        shape = local_shape_features_for_target(masked_df.loc[row_idx], col, opt_type, cols_by_type, strike_map)
        state, _, _ = get_edge_blocks(masked_df.loc[row_idx], opt_type, cols_by_type, strike_map)
        strike_order = state["column"].tolist()
        rank = strike_order.index(col) if col in strike_order else np.nan
        rel_rank = rank / (len(strike_order) - 1) if np.isfinite(rank) and len(strike_order) > 1 else np.nan
        edge = bool(d.get("edge", False))
        edge_side = d.get("edge_side", "")
        section = f"edge_{edge_side}" if edge else "interior"
        rows.append({
            "run": run_id, "row_index": row_idx, "datetime": original_df.loc[row_idx, "datetime"], "datetime_parsed": dtp,
            "date": dtp.date().isoformat() if pd.notna(dtp) else "", "time": dtp.strftime("%H:%M") if pd.notna(dtp) else "", "time_minutes": int(dtp.hour * 60 + dtp.minute) if pd.notna(dtp) else np.nan,
            "contract": col, "option_type": opt_type, "strike": strike, "underlying_price": spot, "moneyness": moneyness, "log_moneyness": math.log(moneyness) if np.isfinite(moneyness) and moneyness > 0 else np.nan,
            "strike_rank": rank, "relative_strike_rank": rel_rank, "section": section, "edge": edge, "edge_reason": d.get("edge_reason", ""), "edge_side": edge_side, "edge_block_size": d.get("edge_block_size", np.nan), "edge_position_in_block": d.get("edge_position_in_block", np.nan),
            "source": d.get("source", ""), "selected_model": d.get("selected_model", ""), "bandwidth": d.get("bandwidth", np.nan), "loo_mse": d.get("loo_mse", np.nan), "n_train": d.get("n_train", np.nan),
            "actual_iv": actual, "predicted_iv": pred, "error": err, "abs_error": abs(err), "sq_error": err ** 2, "relative_abs_error": abs(err) / abs(actual) if actual != 0 else np.nan,
            "edge_claude_prediction": d.get("edge_claude_prediction", np.nan), "edge_corrected_prediction": d.get("edge_corrected_prediction", np.nan), "edge_quadratic_prediction": d.get("edge_quadratic_prediction", np.nan),
            **shape,
        })
    return pd.DataFrame(rows)


def add_analysis_buckets(errors):
    out = errors.copy()
    out["moneyness_bucket"] = pd.cut(out["moneyness"], bins=MONEYNESS_BINS, labels=MONEYNESS_LABELS, include_lowest=True).astype(str)
    out["time_bucket"] = pd.cut(out["time_minutes"], bins=[0, 10*60, 12*60, 14*60, 16*60], labels=["open_to_10", "10_to_12", "12_to_14", "14_to_close"], include_lowest=True).astype(str)
    out["rank_bucket"] = pd.cut(out["relative_strike_rank"], bins=[-0.001, 0.15, 0.35, 0.65, 0.85, 1.001], labels=["far_left_wing", "left_mid", "middle", "right_mid", "far_right_wing"], include_lowest=True).astype(str)
    for col, new_col in [("local_abs_curvature", "curvature_quantile_bucket"), ("local_abs_slope", "slope_quantile_bucket")]:
        values = out[col].replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        out[new_col] = "unknown"
        if valid.sum() >= 10:
            try:
                out.loc[valid, new_col] = pd.qcut(values[valid], q=5, duplicates="drop").astype(str)
            except Exception:
                pass
    curv = out["local_abs_curvature"].replace([np.inf, -np.inf], np.nan)
    valid = curv.notna()
    out["straight_vs_curved"] = "unknown"
    if valid.sum() >= 10:
        q33, q67 = curv[valid].quantile(0.33), curv[valid].quantile(0.67)
        out.loc[valid & (curv <= q33), "straight_vs_curved"] = "straight_section_low_curvature"
        out.loc[valid & (curv > q33) & (curv <= q67), "straight_vs_curved"] = "moderate_curvature"
        out.loc[valid & (curv > q67), "straight_vs_curved"] = "curved_section_high_curvature"
    return out


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_static_plots(errors, out_dir):
    if plt is None:
        print("matplotlib not installed; skipping static plots.")
        return
    plot_dir = ensure_dir(out_dir / "plots")
    def savefig(name):
        plt.tight_layout(); plt.savefig(plot_dir / name, dpi=160); plt.close()
    plt.figure(figsize=(7,7)); plt.scatter(errors["actual_iv"], errors["predicted_iv"], s=8, alpha=0.35)
    lo = min(errors["actual_iv"].min(), errors["predicted_iv"].min()); hi = max(errors["actual_iv"].max(), errors["predicted_iv"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1); plt.xlabel("Actual IV"); plt.ylabel("Predicted IV"); plt.title("Predicted vs Actual IV"); plt.grid(alpha=0.3); savefig("predicted_vs_actual.png")
    plt.figure(figsize=(9,5)); plt.hist(errors["error"], bins=100, alpha=0.85); plt.axvline(0, linestyle="--", linewidth=1); plt.xlabel("Error"); plt.ylabel("Count"); plt.title("Error Distribution"); plt.grid(alpha=0.3); savefig("error_histogram.png")
    plt.figure(figsize=(9,5)); plt.scatter(errors["moneyness"], errors["abs_error"], s=8, alpha=0.35); plt.xlabel("Moneyness"); plt.ylabel("Absolute Error"); plt.title("Absolute Error vs Moneyness"); plt.grid(alpha=0.3); savefig("abs_error_vs_moneyness.png")
    data = errors.replace([np.inf, -np.inf], np.nan).dropna(subset=["local_abs_curvature", "abs_error"])
    if len(data):
        plt.figure(figsize=(9,5)); plt.scatter(data["local_abs_curvature"], data["abs_error"], s=8, alpha=0.35); plt.xscale("log"); plt.xlabel("Local Abs Curvature"); plt.ylabel("Absolute Error"); plt.title("Errors in Straight vs Curved Regions"); plt.grid(alpha=0.3); savefig("abs_error_vs_local_curvature.png")
    for name, col in [("mse_by_section", "section"), ("mse_by_straight_vs_curved", "straight_vs_curved"), ("mse_by_rank_bucket", "rank_bucket"), ("mse_by_option_type", "option_type"), ("mse_by_moneyness_bucket", "moneyness_bucket"), ("mse_by_date", "date")]:
        gm = group_metrics(errors, [col])
        if gm.empty: continue
        gm = gm.sort_values("mse", ascending=False)
        plt.figure(figsize=(max(8, len(gm)*0.7), 5)); plt.bar(gm[col].astype(str), gm["mse"]); plt.xticks(rotation=45, ha="right"); plt.xlabel(col); plt.ylabel("MSE"); plt.title(name.replace("_", " ").title()); plt.grid(axis="y", alpha=0.3); savefig(f"{name}.png")
    for opt_type in ["CE", "PE"]:
        sub = errors[errors["option_type"] == opt_type]
        if sub.empty: continue
        pivot = sub.pivot_table(index="strike", columns="row_index", values="abs_error", aggfunc="mean")
        if pivot.empty: continue
        plt.figure(figsize=(13,5)); plt.imshow(pivot.values, aspect="auto", origin="lower"); plt.colorbar(label="Mean Absolute Error"); plt.yticks(range(len(pivot.index)), pivot.index.astype(str)); plt.xlabel("Timestamp row index"); plt.ylabel("Strike"); plt.title(f"{opt_type}: Mean Absolute Error Heatmap"); savefig(f"heatmap_abs_error_{opt_type}.png")


def create_interactive_dashboard(errors, original_df, out_dir, dashboard_run, max_frames):
    if not HAS_PLOTLY:
        print("plotly not installed; skipping HTML dashboard. Install with: pip install plotly")
        return
    html_path = out_dir / "interactive_error_dashboard.html"
    e = errors.copy()
    fig = make_subplots(rows=3, cols=2, specs=[[{"type":"xy"},{"type":"xy"}], [{"type":"heatmap"},{"type":"heatmap"}], [{"type":"xy"},{"type":"xy"}]], subplot_titles=["CE hidden actual vs prediction", "PE hidden actual vs prediction", "CE heatmap: mean abs error", "PE heatmap: mean abs error", "Error vs moneyness", "MSE by straight/curved section"], vertical_spacing=0.10)
    run_errors = e[e["run"] == dashboard_run].copy()
    if run_errors.empty: run_errors = e[e["run"] == e["run"].min()].copy()
    row_indices = sorted(run_errors["row_index"].unique().tolist())
    if len(row_indices) > max_frames:
        row_indices = [row_indices[i] for i in np.linspace(0, len(row_indices)-1, max_frames).astype(int)]
    first_row = row_indices[0] if row_indices else None
    slider_trace_indices = []
    if first_row is not None:
        for opt_type, col_num in [("CE", 1), ("PE", 2)]:
            sub = run_errors[(run_errors["row_index"] == first_row) & (run_errors["option_type"] == opt_type)]
            fig.add_trace(go.Scatter(x=sub["moneyness"], y=sub["actual_iv"], mode="markers", marker=dict(size=9, symbol="diamond"), name=f"{opt_type} hidden actual"), row=1, col=col_num); slider_trace_indices.append(len(fig.data)-1)
            fig.add_trace(go.Scatter(x=sub["moneyness"], y=sub["predicted_iv"], mode="markers", marker=dict(size=10, symbol="x"), name=f"{opt_type} prediction"), row=1, col=col_num); slider_trace_indices.append(len(fig.data)-1)
    for opt_type, col_num in [("CE", 1), ("PE", 2)]:
        sub = e[e["option_type"] == opt_type]
        pivot = sub.pivot_table(index="strike", columns="row_index", values="abs_error", aggfunc="mean")
        fig.add_trace(go.Heatmap(z=pivot.values if not pivot.empty else [[]], x=pivot.columns.astype(str).tolist() if not pivot.empty else [], y=pivot.index.astype(str).tolist() if not pivot.empty else [], colorscale="Viridis", name=f"{opt_type} heatmap"), row=2, col=col_num)
    sample = e.sample(min(len(e), 8000), random_state=7) if len(e) > 8000 else e
    fig.add_trace(go.Scatter(x=sample["moneyness"], y=sample["abs_error"], mode="markers", marker=dict(size=4, opacity=0.45), text=sample["contract"] + "<br>" + sample["datetime"].astype(str), name="abs error points"), row=3, col=1)
    gm = group_metrics(e, ["straight_vs_curved"])
    fig.add_trace(go.Bar(x=gm["straight_vs_curved"].astype(str), y=gm["mse"], name="MSE by local shape"), row=3, col=2)
    frames = []
    for row_idx in row_indices:
        frame_data = []
        for opt_type in ["CE", "PE"]:
            sub = run_errors[(run_errors["row_index"] == row_idx) & (run_errors["option_type"] == opt_type)]
            frame_data.append(go.Scatter(x=sub["moneyness"], y=sub["actual_iv"], mode="markers", marker=dict(size=9, symbol="diamond")))
            frame_data.append(go.Scatter(x=sub["moneyness"], y=sub["predicted_iv"], mode="markers", marker=dict(size=10, symbol="x")))
        frames.append(go.Frame(data=frame_data, name=str(row_idx), traces=slider_trace_indices[:4]))
    fig.frames = frames
    steps = []
    for row_idx in row_indices:
        dt = original_df.loc[row_idx, "datetime"] if row_idx in original_df.index else str(row_idx)
        steps.append({"args": [[str(row_idx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": f"{row_idx} | {dt}", "method": "animate"})
    if steps:
        fig.update_layout(sliders=[{"active": 0, "currentvalue": {"prefix": "Timestamp: "}, "pad": {"t": 50}, "steps": steps}])
    fig.update_layout(title="Monte Carlo IV Imputation Error Dashboard", height=1200, width=1500, showlegend=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Interactive dashboard saved → {html_path}")


def save_group_reports(errors, out_dir):
    specs = {
        "by_run": ["run"], "by_date": ["date"], "by_time": ["time"], "by_time_bucket": ["time_bucket"], "by_option_type": ["option_type"], "by_section": ["section"], "by_edge_reason": ["edge_reason"], "by_straight_vs_curved": ["straight_vs_curved"], "by_curvature_quantile": ["curvature_quantile_bucket"], "by_slope_quantile": ["slope_quantile_bucket"], "by_rank_bucket": ["rank_bucket"], "by_moneyness_bucket": ["moneyness_bucket"], "by_source": ["source"], "by_bandwidth": ["bandwidth"], "by_contract": ["contract", "option_type", "strike"], "by_strike_type": ["option_type", "strike"], "by_date_option_type": ["date", "option_type"], "by_section_shape": ["section", "straight_vs_curved"], "by_type_section_shape": ["option_type", "section", "straight_vs_curved"]
    }
    for name, cols in specs.items():
        gm = group_metrics(errors, cols)
        gm.to_csv(out_dir / f"group_metrics_{name}.csv", index=False)


def main():
    args = parse_args()
    data_path = Path(args.data)
    out_dir = ensure_dir(Path(args.out_dir))
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find input data file: {data_path.resolve()}")
    df = pd.read_csv(data_path)
    if {"datetime", "underlying_price"} - set(df.columns):
        raise ValueError("Dataset must contain datetime and underlying_price columns.")
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["datetime_parsed"].isna().any():
        raise ValueError(f"{int(df['datetime_parsed'].isna().sum())} datetimes could not be parsed.")
    df = df.sort_values("datetime_parsed").reset_index(drop=True)
    meta = parse_metadata(df)
    option_cols = meta["column"].tolist()
    strike_map = dict(zip(meta["column"], meta["strike"]))
    type_map = dict(zip(meta["column"], meta["option_type"]))
    cols_by_type = {"CE": [c for c in option_cols if type_map[c] == "CE"], "PE": [c for c in option_cols if type_map[c] == "PE"]}
    global_median_iv = float(df[option_cols].stack().median())
    print("Loaded data:", data_path.resolve())
    print("Rows:", len(df), "Option columns:", len(option_cols), "Initial missing:", int(df[option_cols].isna().sum().sum()))
    all_errors = []
    for run in tqdm(range(args.n_runs), desc="Monte Carlo runs"):
        rng = np.random.default_rng(args.seed + run)
        if args.mask_mode == "pattern":
            hidden = build_fake_holdout_pattern(df, option_cols, cols_by_type, rng, args.min_hidden_per_run, args.max_hidden_per_row_type)
        else:
            hidden = build_fake_holdout_frac(df, option_cols, rng, args.holdout_frac)
        masked_df, truth = apply_fake_holdout(df, hidden)
        filled_df, fill_diag = fill_dataset(masked_df, option_cols, cols_by_type, strike_map, type_map, global_median_iv)
        all_errors.append(build_error_rows_for_run(run, df, masked_df, filled_df, fill_diag, truth, type_map, strike_map, cols_by_type))
    errors = pd.concat(all_errors, ignore_index=True)
    errors = add_analysis_buckets(errors)
    errors.to_csv(out_dir / "mc_error_rows.csv", index=False)
    errors.sort_values("abs_error", ascending=False).head(500).to_csv(out_dir / "worst_500_errors.csv", index=False)
    overall = compute_metrics(errors["actual_iv"], errors["predicted_iv"])
    overall.update({"n_runs": int(args.n_runs), "mask_mode": args.mask_mode, "data_path": str(data_path), "n_unique_timestamps": int(df["datetime"].nunique()), "n_option_cols": int(len(option_cols)), "n_scored_fake_hidden_cells": int(len(errors))})
    pd.DataFrame([overall]).to_csv(out_dir / "mc_metrics_summary.csv", index=False)
    with open(out_dir / "mc_metrics_summary.json", "w", encoding="utf-8") as f: json.dump(overall, f, indent=2)
    save_group_reports(errors, out_dir)
    if not args.no_static_plots: save_static_plots(errors, out_dir)
    if not args.no_html: create_interactive_dashboard(errors, df, out_dir, args.dashboard_run, args.max_dashboard_frames)
    print("\nMonte Carlo error analysis complete.")
    print("Output directory:", out_dir.resolve())
    print("\nOverall metrics:")
    for k, v in overall.items(): print(f"  {k}: {v}")
    print("\nMost useful files:")
    for f in ["mc_error_rows.csv", "mc_metrics_summary.csv", "worst_500_errors.csv", "group_metrics_by_section.csv", "group_metrics_by_straight_vs_curved.csv", "group_metrics_by_type_section_shape.csv", "interactive_error_dashboard.html", "plots/"]:
        print(" ", out_dir / f)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
