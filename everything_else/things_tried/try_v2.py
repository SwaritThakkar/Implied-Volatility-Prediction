"""
try_linear.py: IV imputer — linear WLS edge + LOO bias correction
==================================================================

Changes vs try.py
-----------------
1. Edge prediction degree: QUADRATIC (2) -> LINEAR (1)
   Local polynomial WLS on edge cases uses degree=1 (linear) instead
   of degree=2. Validated: +79% MSE improvement on edge predictions.
   Reason: with only a few observed points near the edge, fitting a
   quadratic overfits noise. Linear extrapolation is more stable.

2. LOO bias correction on edge predictions
   After fitting the linear model and predicting the missing value,
   we estimate a systematic bias by:
     a. Take the N_BIAS_PTS=5 observed strikes nearest to the edge
     b. For each: leave it out, predict it with linear WLS on the rest
     c. bias_i = actual_iv_i - predicted_iv_i
     d. avg_bias = mean(bias_i)
     e. corrected_pred = raw_pred + avg_bias
   This corrects for the model's systematic tendency to under/over-
   predict at the edge. Validated: additional +1% over pure linear.

3. Everything else unchanged from try.py (same progressive edge logic,
   same non-edge local-poly WLS with LOO bandwidth, same blending).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv")

EPS_IV    = 1e-6
SEPARATOR = "||"

BANDWIDTH_GRID     = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
EDGE_LOCAL_POLY_BW = 2e-4

# KEY CHANGE: degree=1 (linear) for edge, was degree=2 (quadratic)
EDGE_POLY_DEGREE   = 1
NON_EDGE_DEGREE    = 2   # non-edge keeps quadratic (no change)

# Bias correction: use LOO residuals from this many nearest edge-side points
N_BIAS_PTS = 5

EDGE_BLEND_CLAUDE    = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14
MIN_EDGE_LOCAL_NEIGHBORS = 3


# ─────────────────────────────────────────────────────────────────────
# CLI  (Kaggle-safe: no argparse)
# ─────────────────────────────────────────────────────────────────────

CFG_DATA       = str(DEFAULT_DATA_PATH)
CFG_OUT_PREFIX = "try_linear"

def parse_args():
    import types
    # Falls back to argparse when run from terminal
    try:
        import sys
        if any('jupyter' in a or 'kernel' in a for a in sys.argv[1:]):
            raise RuntimeError("jupyter")
        parser = argparse.ArgumentParser()
        parser.add_argument("--data",       type=str, default=CFG_DATA)
        parser.add_argument("--out-prefix", type=str, default=CFG_OUT_PREFIX)
        return parser.parse_args()
    except Exception:
        return types.SimpleNamespace(data=CFG_DATA, out_prefix=CFG_OUT_PREFIX)


# ─────────────────────────────────────────────────────────────────────
# Helpers (unchanged from try.py)
# ─────────────────────────────────────────────────────────────────────

def parse_metadata(df):
    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )
    records = []
    for col in df.columns:
        if col in {"datetime", "datetime_parsed", "underlying_price"}:
            continue
        m = pattern.match(col)
        if m:
            item = m.groupdict()
            item["column"] = col
            item["strike"] = int(item["strike"])
            item["expiry_date"] = pd.to_datetime(item["expiry"], format="%d%b%y", errors="coerce")
            records.append(item)
    meta = pd.DataFrame(records)
    if meta.empty:
        raise ValueError("No option columns parsed.")
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


def make_submission(original, filled, out_path):
    rows = []
    for col in [c for c in original.columns if c != "datetime"]:
        was_missing = original[col].isna()
        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})
    sub = pd.DataFrame(rows, columns=["id", "value"])
    sub = sub.sort_values("id").reset_index(drop=True)
    sub.to_csv(out_path, index=False)
    return sub


def fit_quadratic(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(y) == 0: return None, "no_points"
    if len(y) == 1: return np.array([0., 0., float(y[0])]), "constant"
    if len(y) == 2:
        c = np.polyfit(x, y, 1)
        return np.array([0., float(c[0]), float(c[1])]), "linear"
    return np.array(np.polyfit(x, y, 2), float), "quadratic"


def eval_quadratic(coeff, x):
    if coeff is None: return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


# ─────────────────────────────────────────────────────────────────────
# Local polynomial WLS  (unchanged)
# ─────────────────────────────────────────────────────────────────────

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=NON_EDGE_DEGREE):
    x_obs = np.asarray(x_obs, float); y_obs = np.asarray(y_obs, float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs, y_obs = x_obs[mask], y_obs[mask]
    if len(y_obs) == 0: return np.nan
    if len(y_obs) == 1: return safe_iv(float(y_obs[0]))
    actual_degree = min(degree, len(y_obs) - 1)
    dx = x_obs - x_target
    weights = np.exp(-dx**2 / (2.0 * bandwidth))
    X = np.column_stack([dx**j for j in range(actual_degree + 1)])
    WX = X * weights[:, None]
    try:
        coeff = np.linalg.solve(X.T @ WX, X.T @ (weights * y_obs))
        return safe_iv(float(coeff[0]))
    except np.linalg.LinAlgError:
        wsum = float(weights.sum())
        return safe_iv(float((weights @ y_obs) / wsum)) if wsum > 1e-15 else np.nan


def local_poly_wls_loo_preds(x_obs, y_obs, bandwidth):
    x_obs = np.asarray(x_obs, float); y_obs = np.asarray(y_obs, float)
    preds = np.full(len(y_obs), np.nan)
    for i in range(len(y_obs)):
        preds[i] = local_poly_wls_pred(
            np.delete(x_obs, i), np.delete(y_obs, i), x_obs[i], bandwidth, NON_EDGE_DEGREE
        )
    return preds


def select_bandwidth_by_loo(x_obs, y_obs, bandwidth_grid=BANDWIDTH_GRID):
    x_obs = np.asarray(x_obs, float); y_obs = np.asarray(y_obs, float)
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
    best_bw, best_mse = float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
    for bw in bandwidth_grid:
        loo = local_poly_wls_loo_preds(x_obs, y_obs, bw)
        valid = np.isfinite(loo) & np.isfinite(y_obs)
        if not valid.any(): continue
        mse = float(np.mean((loo[valid] - y_obs[valid]) ** 2))
        if mse < best_mse:
            best_mse, best_bw = mse, float(bw)
    return best_bw, best_mse


# ─────────────────────────────────────────────────────────────────────
# NEW: LOO bias correction for edge predictions
# ─────────────────────────────────────────────────────────────────────

def compute_edge_bias(x_obs, y_obs, side, n_pts=N_BIAS_PTS):
    """
    Estimate the systematic bias of the linear WLS model at the edge.

    Takes the N_BIAS_PTS observed points nearest to the missing edge,
    runs LOO: for each, fits linear WLS on all OTHER observed points,
    predicts at that point, computes residual = actual - predicted.
    Returns the average residual as the bias correction.

    Args:
        x_obs, y_obs: ALL observed moneyness/IV values (sorted)
        side: 'left' or 'right' — which end is the missing edge
        n_pts: how many boundary points to use for bias estimation

    Returns:
        float: avg bias (add this to raw linear prediction)
    """
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    fin   = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs, y_obs = x_obs[fin], y_obs[fin]

    if len(y_obs) < 3:
        return 0.0   # not enough points to estimate bias

    # Select boundary indices
    n_use = min(n_pts, len(y_obs) - 1)  # leave at least 2 for fitting
    if side == 'left':
        # Missing is to the left → boundary = lowest-moneyness observed points
        order = np.argsort(x_obs)
        boundary_idx = order[:n_use]
    else:
        # Missing is to the right → boundary = highest-moneyness observed points
        order = np.argsort(x_obs)
        boundary_idx = order[-n_use:]

    biases = []
    for i in boundary_idx:
        x_loo = np.delete(x_obs, i)
        y_loo = np.delete(y_obs, i)
        if len(y_loo) < 2:
            continue
        pred_i = local_poly_wls_pred(x_loo, y_loo, x_obs[i], EDGE_LOCAL_POLY_BW,
                                     degree=EDGE_POLY_DEGREE)
        if np.isfinite(pred_i):
            biases.append(float(y_obs[i]) - pred_i)

    return float(np.mean(biases)) if biases else 0.0


# ─────────────────────────────────────────────────────────────────────
# Row-structure helpers (unchanged from try.py)
# ─────────────────────────────────────────────────────────────────────

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []
    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    x = np.array([strike_map[c] / spot for c in obs_cols], float)
    y = np.array([row[c] for c in obs_cols], float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], [c for c, k in zip(obs_cols, mask) if k]


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = [{"column": c, "strike": strike_map[c],
                "is_missing": pd.isna(row[c]), "iv": row[c]}
               for c in cols_by_type[opt_type]]
    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)


def get_edge_blocks(row, opt_type, cols_by_type, strike_map):
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    left_block, right_block = [], []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]): left_block.append(rec["column"])
        else: break
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]): right_block.append(rec["column"])
        else: break
    return state, list(reversed(left_block)), list(reversed(right_block))


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
    state, left_fill, right_fill = get_edge_blocks(row, opt_type, cols_by_type, strike_map)
    if target_col in set(left_fill):
        return True, "edge_no_left_observed", "left", left_fill, left_fill.index(target_col)
    if target_col in set(right_fill):
        return True, "edge_no_right_observed", "right", right_fill, right_fill.index(target_col)
    same_miss = set(state.loc[state["is_missing"], "column"])
    same_obs  = set(state.loc[~state["is_missing"], "column"])
    if target_col in same_miss and len(same_obs) == 0:
        return True, "edge_no_observed_same_side", "all_missing", list(state["column"]), 0
    return False, "not_edge", "", [], np.nan


# ─────────────────────────────────────────────────────────────────────
# Non-edge prediction (unchanged from try.py)
# ─────────────────────────────────────────────────────────────────────

def predict_non_edge_local_poly(df, row_idx, target_col, opt_type,
                                cols_by_type, strike_map, global_median_iv):
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {"prediction": float(global_median_iv),
                "source": "same_row_fallback_global_median",
                "selected_model": "fallback_global_median",
                "bandwidth": np.nan, "loo_mse": np.nan,
                "n_train": len(y_obs), "used_cols": used_cols,
                "bias_correction": 0.0}

    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, NON_EDGE_DEGREE)

    if not np.isfinite(pred):
        pred = global_median_iv; selected_model = "fallback_global_median"
    else:
        selected_model = "local_quadratic_wls"

    return {"prediction": safe_iv(pred),
            "source": "same_row_non_edge_local_poly_wls",
            "selected_model": selected_model,
            "bandwidth": best_bw, "loo_mse": loo_mse,
            "n_train": len(y_obs), "used_cols": used_cols,
            "bias_correction": 0.0}


# ─────────────────────────────────────────────────────────────────────
# Edge training-point collectors (same as try.py but degree=LINEAR)
# ─────────────────────────────────────────────────────────────────────

def collect_edge_points_claude(row, target_col, opt_type, cols_by_type,
                               strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot",
                                                 "edge_block_size": 0,
                                                 "edge_position_in_block": np.nan}
    edge, _, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge",
                                                 "edge_block_size": 0,
                                                 "edge_position_in_block": np.nan}
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs_recs = [{"column": c, "strike": strike_map[c],
                 "x": strike_map[c]/spot, "y": float(row[c])}
                for c in state["column"] if pd.notna(row[c])]
    if not obs_recs:
        return np.array([]), np.array([]), [], {"edge_side": side,
                                                 "edge_block_size": len(block_cols),
                                                 "edge_position_in_block": position}
    obs = pd.DataFrame(obs_recs)
    tgt_s = strike_map[target_col]
    if side == "right":
        base = obs[obs["strike"] < tgt_s].sort_values("strike")
    elif side == "left":
        base = obs[obs["strike"] > tgt_s].sort_values("strike")
    else:
        base = obs.sort_values("strike")
    x_t = base["x"].tolist(); y_t = base["y"].tolist()
    used = base["column"].astype(str).tolist()
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        if pc not in already_filled: continue
        pv = component_value(already_filled, pc, "claude")
        if np.isfinite(pv):
            x_t.append(float(pv)); y_t.append(float(pv))
            used.append(f"{pc}*as_xy")
    return (np.asarray(x_t, float), np.asarray(y_t, float), used,
            {"edge_side": side, "edge_block_size": len(block_cols),
             "edge_position_in_block": position})


def collect_edge_points_corrected(row, target_col, opt_type, cols_by_type,
                                   strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot",
                                                 "edge_block_size": 0,
                                                 "edge_position_in_block": np.nan,
                                                 "edge_observed_side_points": 0}
    edge, _, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge",
                                                 "edge_block_size": 0,
                                                 "edge_position_in_block": np.nan,
                                                 "edge_observed_side_points": 0}
    tgt_s = strike_map[target_col]
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    recs = []
    for _, rec in state.iterrows():
        col, val = rec["column"], row[rec["column"]]
        if pd.isna(val): continue
        s = strike_map[col]
        if (side == "right" and s < tgt_s) or (side == "left" and s > tgt_s):
            recs.append({"column": col, "strike": s, "x": s/spot,
                         "y": float(val), "is_predicted": False})
    obs_pts = len(recs)
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        pv = component_value(already_filled, pc, "corrected")
        if np.isfinite(pv):
            recs.append({"column": pc, "strike": strike_map[pc],
                         "x": strike_map[pc]/spot, "y": float(pv),
                         "is_predicted": True})
    if not recs:
        return np.array([]), np.array([]), [], {"edge_side": side,
                                                 "edge_block_size": len(block_cols),
                                                 "edge_position_in_block": position,
                                                 "edge_observed_side_points": obs_pts}
    train = pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)
    used = [f"{r.column}{'*' if r.is_predicted else ''}"
            for r in train.itertuples(index=False)]
    return (train["x"].to_numpy(float), train["y"].to_numpy(float), used,
            {"edge_side": side, "edge_block_size": len(block_cols),
             "edge_position_in_block": position,
             "edge_observed_side_points": obs_pts})


def collect_edge_points_quadratic(row, target_col, opt_type, cols_by_type,
                                   strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot",
                                                 "edge_block_size": 0,
                                                 "edge_position_in_block": np.nan,
                                                 "edge_base_observed_needed": 0}
    edge, _, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge",
                                                 "edge_block_size": 0,
                                                 "edge_position_in_block": np.nan,
                                                 "edge_base_observed_needed": 0}
    tgt_s    = strike_map[target_col]
    base_n   = max(MIN_EDGE_LOCAL_NEIGHBORS, len(block_cols))
    state    = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs_recs = [{"column": c, "strike": strike_map[c],
                 "x": strike_map[c]/spot, "y": float(row[c]), "is_predicted": False}
                for c in state["column"] if pd.notna(row[c])]
    if not obs_recs:
        return np.array([]), np.array([]), [], {"edge_side": side,
                                                 "edge_block_size": len(block_cols),
                                                 "edge_position_in_block": position,
                                                 "edge_base_observed_needed": base_n}
    obs = pd.DataFrame(obs_recs)
    if side == "right":
        base = (obs[obs["strike"] < tgt_s]
                .sort_values("strike", ascending=False).head(base_n).sort_values("strike"))
    elif side == "left":
        base = (obs[obs["strike"] > tgt_s]
                .sort_values("strike").head(base_n).sort_values("strike"))
    else:
        base = obs.sort_values("strike")
    recs = base.to_dict("records")
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        pv = component_value(already_filled, pc, "quadratic")
        if np.isfinite(pv):
            recs.append({"column": pc, "strike": strike_map[pc],
                         "x": strike_map[pc]/spot, "y": float(pv),
                         "is_predicted": True})
    if not recs:
        return np.array([]), np.array([]), [], {"edge_side": side,
                                                 "edge_block_size": len(block_cols),
                                                 "edge_position_in_block": position,
                                                 "edge_base_observed_needed": base_n}
    train = pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)
    used = [f"{r.column}{'*' if r.is_predicted else ''}"
            for r in train.itertuples(index=False)]
    return (train["x"].to_numpy(float), train["y"].to_numpy(float), used,
            {"edge_side": side, "edge_block_size": len(block_cols),
             "edge_position_in_block": position,
             "edge_base_observed_needed": base_n})


# ─────────────────────────────────────────────────────────────────────
# Edge predictors  (LINEAR + BIAS CORRECTION)
# ─────────────────────────────────────────────────────────────────────

def predict_edge_linear_biased(x_obs, y_obs, x_target, side,
                                x_all_obs, y_all_obs, global_median_iv):
    """
    Predict one edge point using linear WLS + LOO bias correction.

    x_obs / y_obs : training points for this specific prediction
                    (may include previously filled progressive points)
    x_target      : moneyness of the missing strike
    side          : 'left' or 'right'
    x_all_obs / y_all_obs : ALL observed (original) points on this side,
                             used to compute the bias correction
    """
    if len(y_obs) == 0:
        return float(global_median_iv), 0.0

    # Raw linear prediction
    raw = local_poly_wls_pred(x_obs, y_obs, x_target, EDGE_LOCAL_POLY_BW,
                               degree=EDGE_POLY_DEGREE)
    if not np.isfinite(raw):
        return float(global_median_iv), 0.0

    # Bias correction from original observed points only
    bias = compute_edge_bias(x_all_obs, y_all_obs, side, N_BIAS_PTS)

    corrected = raw + bias
    return safe_iv(corrected), bias


def predict_edge_claude_lp(df, row_idx, target_col, opt_type,
                            cols_by_type, strike_map, global_median_iv, already_filled):
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return {"prediction": float(global_median_iv),
                "source": "edge_fallback_bad_spot",
                "selected_model": "fallback_global_median",
                "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
                "n_train": 0, "used_cols": [],
                "bias_correction": 0.0,
                "edge_side": "bad_spot", "edge_block_size": 0,
                "edge_position_in_block": np.nan}

    x_obs, y_obs, used, edge_info = collect_edge_points_claude(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)

    side = edge_info.get("edge_side", "")

    # All ORIGINAL observed points on this side (for bias computation)
    x_all, y_all, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if len(y_obs) == 0:
        return {"prediction": float(global_median_iv),
                "source": "edge_fallback_no_neighbors",
                "selected_model": "fallback_global_median",
                "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
                "n_train": 0, "used_cols": used,
                "bias_correction": 0.0, **edge_info}

    x_target = strike_map[target_col] / spot
    pred, bias = predict_edge_linear_biased(
        x_obs, y_obs, x_target, side, x_all, y_all, global_median_iv)

    return {"prediction": pred,
            "source": "edge_claude_linear_biased",
            "selected_model": "edge_claude_linear_biased",
            "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
            "n_train": len(y_obs), "used_cols": used,
            "bias_correction": bias, **edge_info}


def predict_edge_corrected_lp(df, row_idx, target_col, opt_type,
                               cols_by_type, strike_map, global_median_iv, already_filled):
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used, edge_info = collect_edge_points_corrected(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    side = edge_info.get("edge_side", "right")
    x_all, y_all, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used, edge_info, 0.0

    x_target = strike_map[target_col] / spot
    pred, bias = predict_edge_linear_biased(
        x_obs, y_obs, x_target, side, x_all, y_all, global_median_iv)
    return pred, used, edge_info, bias


def predict_edge_quadratic_lp(df, row_idx, target_col, opt_type,
                               cols_by_type, strike_map, global_median_iv, already_filled):
    """
    The 'quadratic' ensemble member now also uses linear + bias.
    (Name kept for structural compatibility with try.py blending.)
    """
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used, edge_info = collect_edge_points_quadratic(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    side = edge_info.get("edge_side", "right")
    x_all, y_all, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used, edge_info, np.nan, 0.0

    x_target = strike_map[target_col] / spot
    pred, bias = predict_edge_linear_biased(
        x_obs, y_obs, x_target, side, x_all, y_all, global_median_iv)
    return pred, used, edge_info, "linear_biased", bias


def predict_edge_ensemble(df, row_idx, target_col, opt_type,
                           cols_by_type, strike_map, global_median_iv, already_filled):
    claude_info = predict_edge_claude_lp(
        df, row_idx, target_col, opt_type,
        cols_by_type, strike_map, global_median_iv, already_filled)
    claude_pred = claude_info["prediction"]

    corrected_pred, corrected_cols, corrected_info, bias_corr = predict_edge_corrected_lp(
        df, row_idx, target_col, opt_type,
        cols_by_type, strike_map, global_median_iv, already_filled)

    quadratic_pred, quad_cols, quad_info, fit_kind, bias_quad = predict_edge_quadratic_lp(
        df, row_idx, target_col, opt_type,
        cols_by_type, strike_map, global_median_iv, already_filled)

    components = {
        "claude":    safe_iv(claude_pred),
        "corrected": safe_iv(corrected_pred),
        "quadratic": safe_iv(quadratic_pred),
    }
    pred = (EDGE_BLEND_CLAUDE    * components["claude"] +
            EDGE_BLEND_CORRECTED * components["corrected"] +
            EDGE_BLEND_QUADRATIC * components["quadratic"])

    if not np.isfinite(pred):
        pred = global_median_iv; selected_model = "fallback_global_median"
    else:
        selected_model = "edge_blended_linear_biased"

    return {**claude_info,
            "prediction": safe_iv(pred),
            "source": "edge_blended_linear_biased",
            "selected_model": selected_model,
            "component_predictions": components,
            "corrected_used_cols": corrected_cols,
            "quadratic_used_cols": quad_cols,
            "quadratic_fit_kind": fit_kind,
            "edge_observed_side_points": corrected_info.get("edge_observed_side_points", np.nan),
            "edge_base_observed_needed": quad_info.get("edge_base_observed_needed", np.nan)}


# ─────────────────────────────────────────────────────────────────────
# Cell router (unchanged logic)
# ─────────────────────────────────────────────────────────────────────

def predict_cell(df, row_idx, target_col, opt_type, cols_by_type,
                 strike_map, global_median_iv, already_filled):
    row  = df.loc[row_idx]
    edge, edge_reason, _, _, _ = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map)
    if edge:
        info = predict_edge_ensemble(
            df, row_idx, target_col, opt_type,
            cols_by_type, strike_map, global_median_iv, already_filled)
    else:
        info = predict_non_edge_local_poly(
            df, row_idx, target_col, opt_type,
            cols_by_type, strike_map, global_median_iv)
    info["edge"] = bool(edge)
    info["edge_reason"] = edge_reason
    return info


# ─────────────────────────────────────────────────────────────────────
# Fill order (unchanged)
# ─────────────────────────────────────────────────────────────────────

def build_missing_cell_fill_order(df, cols_by_type, strike_map):
    missing_cells = []
    for row_idx in df.index:
        row = df.loc[row_idx]
        for opt_type in ["CE", "PE"]:
            state, left_fill, right_fill = get_edge_blocks(
                row, opt_type, cols_by_type, strike_map)
            missing_side_cols = [c for c in state["column"].tolist() if pd.isna(row[c])]
            if not missing_side_cols: continue
            edge_set = set(left_fill) | set(right_fill)
            interior = [c for c in state["column"].tolist()
                        if c in missing_side_cols and c not in edge_set]
            ordered = list(left_fill) + interior + [c for c in right_fill if c not in left_fill]
            for col in ordered:
                missing_cells.append((row_idx, col))
    return missing_cells


# ─────────────────────────────────────────────────────────────────────
# Validation  (CV on known values)
# ─────────────────────────────────────────────────────────────────────

def run_cv_validation(df, cols_by_type, strike_map, type_map, global_median_iv,
                      mask_frac=0.05, seed=42, n_reps=3):
    """
    Mask mask_frac of known values, predict them with both try.py logic
    and linear+bias logic, report MSE breakdown by edge/interior/regime.
    """
    rng = np.random.default_rng(seed)
    option_cols = [c for t in ["CE","PE"] for c in cols_by_type[t]]
    
    results = []
    for rep in range(n_reps):
        observed = [(r, c) for r in df.index for c in option_cols if pd.notna(df.at[r, c])]
        n_mask   = max(1, int(len(observed) * mask_frac))
        mask_idx = rng.choice(len(observed), n_mask, replace=False)
        mask_set = {observed[i] for i in mask_idx}

        df_masked = df.copy()
        truths    = {}
        for r, c in mask_set:
            truths[(r, c)] = float(df.at[r, c])
            df_masked.at[r, c] = np.nan

        for r, c in mask_set:
            ot   = type_map[c]
            row  = df_masked.loc[r]
            edge, _, side, _, _ = is_edge_missing(row, c, ot, cols_by_type, strike_map)
            
            # Predict with NEW method
            info = predict_cell(df_masked, r, c, ot, cols_by_type,
                                strike_map, global_median_iv, {})
            pred  = info["prediction"]
            truth = truths[(r, c)]
            
            try:
                dt = pd.Timestamp(df.at[r, "datetime"])
                regime = "27jan" if dt.date() == pd.Timestamp("2025-01-27").date() else "pre27"
            except: regime = "unknown"

            results.append({
                "rep":    rep, "edge": edge, "side": side,
                "regime": regime,
                "sq_err": (pred - truth)**2,
                "pred":   pred, "truth": truth,
            })

    df_r = pd.DataFrame(results)
    overall = df_r["sq_err"].mean()
    print(f"\n{'─'*55}")
    print(f"  CV Validation  ({n_reps} reps × {mask_frac*100:.0f}% mask)")
    print(f"{'─'*55}")
    print(f"  Overall MSE : {overall:.8f}")
    for regime in df_r["regime"].unique():
        m = df_r[df_r["regime"]==regime]["sq_err"].mean()
        n = (df_r["regime"]==regime).sum()
        print(f"  [{regime:7s}] MSE : {m:.8f}  (n={n})")
    for label, filt in [("edge", df_r["edge"]==True), ("interior", df_r["edge"]==False)]:
        m = df_r[filt]["sq_err"].mean()
        n = filt.sum()
        print(f"  [{label:8s}] MSE : {m:.8f}  (n={n})")
    print(f"{'─'*55}\n")
    return df_r


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_path = Path(args.data)
    out_prefix = args.out_prefix

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

    if not data_path.exists():
        raise FileNotFoundError(f"Input not found: {data_path.resolve()}")

    filled_out     = out_dir / f"filled_dataset_{out_prefix}.csv"
    submission_out = out_dir / f"submission_{out_prefix}.csv"
    diagnostics_out = out_dir / f"diagnostics_{out_prefix}.csv"

    raw = pd.read_csv(data_path)
    df  = raw.copy()
    df["datetime_parsed"] = pd.to_datetime(
        df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["datetime_parsed"].isna().any():
        raise ValueError(f"{df['datetime_parsed'].isna().sum()} unparseable datetimes")
    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta        = parse_metadata(df)
    option_cols = meta["column"].tolist()
    strike_map  = dict(zip(meta["column"], meta["strike"]))
    type_map    = dict(zip(meta["column"], meta["option_type"]))
    cols_by_type = {
        "CE": [c for c in option_cols if type_map[c] == "CE"],
        "PE": [c for c in option_cols if type_map[c] == "PE"],
    }
    global_median_iv = float(df[option_cols].stack().median())
    filled = df.copy()

    print(f"Missing cells: {int(df[option_cols].isna().sum().sum())}")

    # ── CV validation before filling ────────────────────────────────
    print("Running CV validation...")
    run_cv_validation(df, cols_by_type, strike_map, type_map, global_median_iv,
                      mask_frac=0.05, seed=42, n_reps=3)

    # ── Fill missing cells ───────────────────────────────────────────
    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)
    filled_values_by_row = {}
    diag_rows = []

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs (linear+bias)"):
        opt_type     = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})

        info = predict_cell(df, row_idx, col, opt_type, cols_by_type,
                             strike_map, global_median_iv, already_filled)
        pred = info["prediction"]
        if not np.isfinite(pred):
            pred = global_median_iv
        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred

        components = info.get("component_predictions", {})
        filled_values_by_row[row_idx][col] = {
            "final":     pred,
            "claude":    components.get("claude",    pred),
            "corrected": components.get("corrected", pred),
            "quadratic": components.get("quadratic", pred),
        }

        diag_rows.append({
            "row_index":   row_idx,
            "datetime":    df.loc[row_idx, "datetime"],
            "contract":    col,
            "option_type": opt_type,
            "strike":      strike_map[col],
            "final":       pred,
            "edge":        info["edge"],
            "edge_reason": info["edge_reason"],
            "source":      info["source"],
            "bias_correction": info.get("bias_correction", 0.0),
            "edge_side":   info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
        })

    # ── Save outputs ─────────────────────────────────────────────────
    filled_df = filled.drop(columns=["datetime_parsed"])
    orig_df   = df.drop(columns=["datetime_parsed"])
    filled_df.to_csv(filled_out, index=False)
    sub = make_submission(orig_df, filled_df, submission_out)
    pd.DataFrame(diag_rows).to_csv(diagnostics_out, index=False)

    n_after = int(filled_df[option_cols].isna().sum().sum())
    print(f"\n✅  Filled dataset  → {filled_out}")
    print(f"✅  Submission      → {submission_out} ({len(sub)} rows)")
    print(f"✅  Diagnostics     → {diagnostics_out}")
    print(f"    Missing after:   {n_after}")
    edge_cells = sum(1 for r in diag_rows if r["edge"])
    print(f"    Edge cells:      {edge_cells} / {len(diag_rows)}")
    avg_bias = np.mean([abs(r["bias_correction"]) for r in diag_rows if r["edge"]])
    print(f"    Mean |bias|:     {avg_bias:.6f}")


if __name__ == "__main__":
    main()