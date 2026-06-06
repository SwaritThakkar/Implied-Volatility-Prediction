"""
try_final_validated_tuned.py
============================

Base = user's saved best `try_final.py`:
  - Interior / non-edge: same-row same-option-type local quadratic WLS,
    bandwidth selected by LOO over [5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4].
  - Edge: same try_final edge ensemble, but each edge component selects
    BOTH degree {1,2} and bandwidth by LOO.

Small validated additions in this file:
  1) Interior CE-only PCHIP blend search:
       CE interior prediction = (1-w_ce) * WLS + w_ce * PCHIP
       PE interior remains WLS unless CV says otherwise.
     The script grid-searches w_ce on artificial interior masks and only adopts
     it if it improves the CV MSE.

  2) Edge ensemble-weight search:
       edge prediction = wc*claude + wcor*corrected + wq*quadratic
     The script grid-searches weights on artificial edge masks and only adopts
     them if they improve over the try_final weights (0.72, 0.14, 0.14).

This keeps the original try_final logic as the fallback. If validation finds no
improvement, it uses the exact original defaults.

Run:
    python try_final_validated_tuned.py --data dataset.csv

Outputs:
    filled_dataset_try_final_validated_tuned.csv
    submission_try_final_validated_tuned.csv
    diagnostics_try_final_validated_tuned.csv
    validation_tuning_try_final_validated_tuned.csv
"""

import argparse
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except Exception:
    PchipInterpolator = None
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_DATA_PATH = Path(
    "dataset.csv"
)

EPS_IV = 1e-6
SEPARATOR = "||"

BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
EDGE_LOCAL_POLY_BW = 2e-4
LOCAL_POLY_DEGREE = 2

# Original try_final weights.
DEFAULT_EDGE_WEIGHTS = (0.72, 0.14, 0.14)  # claude, corrected, quadratic
MIN_EDGE_LOCAL_NEIGHBORS = 3

# Tuning defaults. Use 5 reps to match your current validation style.
CV_MASK_FRAC = 0.08
CV_REPS = 5
CV_SEED = 42
PCHIP_WEIGHT_GRID = np.round(np.arange(0.0, 0.501, 0.05), 2)  # deliberately conservative
EDGE_WEIGHT_STEP = 0.05
MIN_REL_IMPROVEMENT_TO_ADOPT = 0.0005  # 0.05%; prevents adopting noise

CFG_DATA = str(DEFAULT_DATA_PATH)
CFG_OUT_PREFIX = "try_final_validated_tuned"


def parse_args():
    if any("jupyter" in a or "kernel" in a or ".json" in a for a in sys.argv[1:]):
        return types.SimpleNamespace(
            data=CFG_DATA,
            out_prefix=CFG_OUT_PREFIX,
            skip_tuning=False,
            ce_pchip_weight=None,
            edge_weights=None,
        )
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=CFG_DATA)
    p.add_argument("--out-prefix", type=str, default=CFG_OUT_PREFIX)
    p.add_argument("--skip-tuning", action="store_true", help="Use manual/default params and skip CV tuning.")
    p.add_argument("--ce-pchip-weight", type=float, default=None, help="Manual CE interior PCHIP weight.")
    p.add_argument(
        "--edge-weights",
        type=str,
        default=None,
        help="Manual edge weights as comma string, e.g. '0.72,0.14,0.14'.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

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
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def make_submission(original, filled, out_path):
    rows = []
    for col in [c for c in original.columns if c != "datetime"]:
        for idx in original.index[original[col].isna()]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})
    sub = pd.DataFrame(rows, columns=["id", "value"]).sort_values("id").reset_index(drop=True)
    sub.to_csv(out_path, index=False)
    return sub


# ---------------------------------------------------------------------
# Local polynomial WLS
# ---------------------------------------------------------------------

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs = x_obs[mask]
    y_obs = y_obs[mask]

    if len(y_obs) == 0:
        return np.nan
    if len(y_obs) == 1:
        return safe_iv(y_obs[0])

    actual_degree = min(int(degree), len(y_obs) - 1)
    dx = x_obs - x_target
    weights = np.exp(-(dx ** 2) / (2.0 * bandwidth))
    X = np.column_stack([dx ** j for j in range(actual_degree + 1)])
    WX = X * weights[:, None]

    try:
        coeff = np.linalg.solve(X.T @ WX, X.T @ (weights * y_obs))
        return safe_iv(coeff[0])
    except np.linalg.LinAlgError:
        wsum = float(weights.sum())
        if wsum <= 1e-15:
            return np.nan
        return safe_iv(float((weights @ y_obs) / wsum))


def local_poly_wls_loo_preds(x_obs, y_obs, bandwidth, degree=LOCAL_POLY_DEGREE):
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    preds = np.full(len(y_obs), np.nan)
    for i in range(len(y_obs)):
        preds[i] = local_poly_wls_pred(
            np.delete(x_obs, i), np.delete(y_obs, i), x_obs[i], bandwidth, degree
        )
    return preds


def select_bandwidth_by_loo(x_obs, y_obs, bandwidth_grid=BANDWIDTH_GRID):
    """Original try.py interior rule: fixed degree=2, LOO over bandwidth only."""
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf

    best_bw = float(bandwidth_grid[len(bandwidth_grid) // 2])
    best_mse = np.inf
    for bw in bandwidth_grid:
        loo = local_poly_wls_loo_preds(x_obs, y_obs, bw, degree=LOCAL_POLY_DEGREE)
        valid = np.isfinite(loo) & np.isfinite(y_obs)
        if not valid.any():
            continue
        mse = float(np.mean((loo[valid] - y_obs[valid]) ** 2))
        if mse < best_mse:
            best_mse, best_bw = mse, float(bw)
    return best_bw, best_mse


def select_bandwidth_and_degree_by_loo(x_obs, y_obs, bandwidth_grid=BANDWIDTH_GRID):
    """try_final edge rule: jointly choose degree in {1,2} and bandwidth by LOO."""
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), 1, np.inf

    best_bw = float(bandwidth_grid[len(bandwidth_grid) // 2])
    best_deg = 1
    best_mse = np.inf
    for degree in [1, 2]:
        for bw in bandwidth_grid:
            loo = local_poly_wls_loo_preds(x_obs, y_obs, bw, degree=degree)
            valid = np.isfinite(loo) & np.isfinite(y_obs)
            if not valid.any():
                continue
            mse = float(np.mean((loo[valid] - y_obs[valid]) ** 2))
            if mse < best_mse:
                best_mse, best_bw, best_deg = mse, float(bw), degree
    return best_bw, best_deg, best_mse


# ---------------------------------------------------------------------
# PCHIP helper for interior only
# ---------------------------------------------------------------------

def pchip_pred(x_obs, y_obs, x_target):
    if not SCIPY_AVAILABLE:
        return np.nan
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs = x_obs[mask]
    y_obs = y_obs[mask]
    if len(y_obs) < 3:
        return np.nan

    order = np.argsort(x_obs)
    x_obs, y_obs = x_obs[order], y_obs[order]

    ux, inv = np.unique(x_obs, return_inverse=True)
    if len(ux) < 3:
        return np.nan
    if len(ux) != len(x_obs):
        yy = np.zeros(len(ux), dtype=float)
        cnt = np.zeros(len(ux), dtype=float)
        for i, j in enumerate(inv):
            yy[j] += y_obs[i]
            cnt[j] += 1
        x_obs, y_obs = ux, yy / cnt

    if x_target < x_obs[0] or x_target > x_obs[-1]:
        return np.nan

    try:
        val = float(PchipInterpolator(x_obs, y_obs, extrapolate=False)(x_target))
        return safe_iv(val) if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Row structure helpers
# ---------------------------------------------------------------------

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []
    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    x = np.array([strike_map[c] / spot for c in obs_cols], dtype=float)
    y = np.array([row[c] for c in obs_cols], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], [c for c, keep in zip(obs_cols, mask) if keep]


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = [
        {"column": c, "strike": strike_map[c], "is_missing": pd.isna(row[c]), "iv": row[c]}
        for c in cols_by_type[opt_type]
    ]
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
    state, left_fill, right_fill = get_edge_blocks(row, opt_type, cols_by_type, strike_map)

    if target_col in set(left_fill):
        return True, "edge_no_left_observed", "left", left_fill, left_fill.index(target_col)
    if target_col in set(right_fill):
        return True, "edge_no_right_observed", "right", right_fill, right_fill.index(target_col)

    same_miss = set(state.loc[state["is_missing"], "column"])
    same_obs = set(state.loc[~state["is_missing"], "column"])
    if target_col in same_miss and len(same_obs) == 0:
        return True, "edge_no_observed_same_side", "all_missing", list(state["column"]), 0

    return False, "not_edge", "", [], np.nan


# ---------------------------------------------------------------------
# Non-edge prediction = original WLS + optional CE-only PCHIP blend
# ---------------------------------------------------------------------

def predict_non_edge_local_poly(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    ce_pchip_weight=0.0,
):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "same_row_fallback_global_median",
            "selected_model": "fallback_global_median",
            "bandwidth": np.nan,
            "loo_mse": np.nan,
            "n_train": len(y_obs),
            "used_cols": used_cols,
            "edge_degree": np.nan,
            "wls_prediction": np.nan,
            "pchip_prediction": np.nan,
            "ce_pchip_weight": ce_pchip_weight,
        }

    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    wls = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(wls):
        wls = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "local_quadratic_wls"

    pc = pchip_pred(x_obs, y_obs, x_target) if opt_type == "CE" and ce_pchip_weight > 0 else np.nan

    if opt_type == "CE" and ce_pchip_weight > 0 and np.isfinite(pc):
        pred = (1.0 - ce_pchip_weight) * wls + ce_pchip_weight * pc
        source = "same_row_non_edge_ce_wls_pchip_blend"
        selected_model = f"ce_wls_{1.0 - ce_pchip_weight:.2f}_pchip_{ce_pchip_weight:.2f}"
    else:
        pred = wls
        source = "same_row_non_edge_local_poly_wls"

    return {
        "prediction": safe_iv(pred),
        "source": source,
        "selected_model": selected_model,
        "bandwidth": best_bw,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        "edge_degree": np.nan,
        "wls_prediction": wls,
        "pchip_prediction": pc,
        "ce_pchip_weight": ce_pchip_weight,
    }


# ---------------------------------------------------------------------
# Edge collectors: try_final logic
# ---------------------------------------------------------------------

def collect_edge_training_points_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {"edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan}

    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {"edge_side": "not_edge", "edge_block_size": 0, "edge_position_in_block": np.nan}

    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs_recs = [
        {"column": c, "strike": strike_map[c], "x": strike_map[c] / spot, "y": float(row[c])}
        for c in state["column"]
        if pd.notna(row[c])
    ]
    if not obs_recs:
        return np.array([]), np.array([]), [], {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}

    obs = pd.DataFrame(obs_recs)
    target_strike = strike_map[target_col]
    if side == "right":
        base = obs[obs["strike"] < target_strike].sort_values("strike")
    elif side == "left":
        base = obs[obs["strike"] > target_strike].sort_values("strike")
    else:
        base = obs.sort_values("strike")

    x_train = base["x"].tolist()
    y_train = base["y"].tolist()
    used = base["column"].astype(str).tolist()

    previous_cols = block_cols[: int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        if prev_col not in already_filled:
            continue
        prev_pred = component_value(already_filled, prev_col, "claude")
        if np.isfinite(prev_pred):
            x_train.append(float(prev_pred))
            y_train.append(float(prev_pred))
            used.append(f"{prev_col}*as_xy")

    return (
        np.asarray(x_train, dtype=float),
        np.asarray(y_train, dtype=float),
        used,
        {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position},
    )


def collect_edge_training_points_corrected(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {
            "edge_side": "bad_spot",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "edge_observed_side_points": 0,
        }

    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {
            "edge_side": "not_edge",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "edge_observed_side_points": 0,
        }

    target_strike = strike_map[target_col]
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    recs = []
    for _, rec in state.iterrows():
        col = rec["column"]
        val = row[col]
        if pd.isna(val):
            continue
        strike = strike_map[col]
        if (side == "right" and strike < target_strike) or (side == "left" and strike > target_strike):
            recs.append({"column": col, "strike": strike, "x": strike / spot, "y": float(val), "is_predicted": False})

    observed_side_points = len(recs)
    previous_cols = block_cols[: int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "corrected")
        if np.isfinite(prev_pred):
            recs.append({
                "column": prev_col,
                "strike": strike_map[prev_col],
                "x": strike_map[prev_col] / spot,
                "y": float(prev_pred),
                "is_predicted": True,
            })

    if not recs:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_observed_side_points": observed_side_points,
        }

    train = pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)
    used = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]
    return (
        train["x"].to_numpy(dtype=float),
        train["y"].to_numpy(dtype=float),
        used,
        {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_observed_side_points": observed_side_points,
        },
    )


def collect_edge_training_points_quadratic(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {
            "edge_side": "bad_spot",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "edge_base_observed_needed": 0,
        }

    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge:
        return np.array([]), np.array([]), [], {
            "edge_side": "not_edge",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "edge_base_observed_needed": 0,
        }

    target_strike = strike_map[target_col]
    base_needed = max(MIN_EDGE_LOCAL_NEIGHBORS, len(block_cols))
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs_recs = [
        {"column": c, "strike": strike_map[c], "x": strike_map[c] / spot, "y": float(row[c]), "is_predicted": False}
        for c in state["column"]
        if pd.notna(row[c])
    ]
    if not obs_recs:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_base_observed_needed": base_needed,
        }

    obs = pd.DataFrame(obs_recs)
    if side == "right":
        base = obs[obs["strike"] < target_strike].sort_values("strike", ascending=False).head(base_needed).sort_values("strike")
    elif side == "left":
        base = obs[obs["strike"] > target_strike].sort_values("strike").head(base_needed).sort_values("strike")
    else:
        base = obs.sort_values("strike")

    recs = base.to_dict("records")
    previous_cols = block_cols[: int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "quadratic")
        if np.isfinite(prev_pred):
            recs.append({
                "column": prev_col,
                "strike": strike_map[prev_col],
                "x": strike_map[prev_col] / spot,
                "y": float(prev_pred),
                "is_predicted": True,
            })

    if not recs:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_base_observed_needed": base_needed,
        }

    train = pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)
    used = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]
    return (
        train["x"].to_numpy(dtype=float),
        train["y"].to_numpy(dtype=float),
        used,
        {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position, "edge_base_observed_needed": base_needed},
    )


# ---------------------------------------------------------------------
# Edge prediction = try_final degree selection + tunable ensemble weights
# ---------------------------------------------------------------------

def _edge_predict_with_deg_select(x_obs, y_obs, x_target, global_median_iv):
    if len(y_obs) == 0:
        return float(global_median_iv), EDGE_LOCAL_POLY_BW, 1, np.inf

    best_bw, best_deg, loo_mse = select_bandwidth_and_degree_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=best_deg)

    if not np.isfinite(pred):
        pred = local_poly_wls_pred(x_obs, y_obs, x_target, EDGE_LOCAL_POLY_BW, degree=1)
    if not np.isfinite(pred):
        pred = global_median_iv

    return safe_iv(pred), best_bw, best_deg, loo_mse


def predict_edge_components(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return {
            "components": {"claude": float(global_median_iv), "corrected": float(global_median_iv), "quadratic": float(global_median_iv)},
            "bandwidth": EDGE_LOCAL_POLY_BW,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": [],
            "edge_degree": 1,
            "edge_info": {"edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan},
            "corrected_used_cols": [],
            "quadratic_used_cols": [],
            "quadratic_fit_kind": "bad_spot",
            "corrected_info": {},
            "quadratic_info": {},
        }

    x_target = strike_map[target_col] / spot

    x_c, y_c, used_c, edge_info = collect_edge_training_points_claude(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled
    )
    pred_c, bw_c, deg_c, mse_c = _edge_predict_with_deg_select(x_c, y_c, x_target, global_median_iv)

    x_cor, y_cor, used_cor, cor_info = collect_edge_training_points_corrected(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled
    )
    pred_cor, _, _, _ = _edge_predict_with_deg_select(x_cor, y_cor, x_target, global_median_iv)

    x_q, y_q, used_q, q_info = collect_edge_training_points_quadratic(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled
    )
    pred_q, _, deg_q, _ = _edge_predict_with_deg_select(x_q, y_q, x_target, global_median_iv)

    return {
        "components": {"claude": safe_iv(pred_c), "corrected": safe_iv(pred_cor), "quadratic": safe_iv(pred_q)},
        "bandwidth": bw_c,
        "loo_mse": mse_c,
        "n_train": len(y_c),
        "used_cols": used_c,
        "edge_degree": deg_c,
        "edge_info": edge_info,
        "corrected_used_cols": used_cor,
        "quadratic_used_cols": used_q,
        "quadratic_fit_kind": f"deg{deg_q}_by_loo",
        "corrected_info": cor_info,
        "quadratic_info": q_info,
    }


def predict_edge_ensemble(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    already_filled,
    edge_weights=DEFAULT_EDGE_WEIGHTS,
):
    comp_info = predict_edge_components(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )
    comps = comp_info["components"]
    wc, wcor, wq = edge_weights
    pred = wc * comps["claude"] + wcor * comps["corrected"] + wq * comps["quadratic"]

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = f"edge_blended_deg_selected_w_{wc:.2f}_{wcor:.2f}_{wq:.2f}"

    return {
        "prediction": safe_iv(pred),
        "source": "edge_blended_deg_selected",
        "selected_model": selected_model,
        "component_predictions": comps,
        "bandwidth": comp_info["bandwidth"],
        "loo_mse": comp_info["loo_mse"],
        "n_train": comp_info["n_train"],
        "used_cols": comp_info["used_cols"],
        "edge_degree": comp_info["edge_degree"],
        "corrected_used_cols": comp_info["corrected_used_cols"],
        "quadratic_used_cols": comp_info["quadratic_used_cols"],
        "quadratic_fit_kind": comp_info["quadratic_fit_kind"],
        "edge_observed_side_points": comp_info["corrected_info"].get("edge_observed_side_points", np.nan),
        "edge_base_observed_needed": comp_info["quadratic_info"].get("edge_base_observed_needed", np.nan),
        **comp_info["edge_info"],
    }


# ---------------------------------------------------------------------
# Router / fill order
# ---------------------------------------------------------------------

def predict_cell(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    already_filled,
    ce_pchip_weight=0.0,
    edge_weights=DEFAULT_EDGE_WEIGHTS,
):
    row = df.loc[row_idx]
    edge, edge_reason, _, _, _ = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)

    if edge:
        info = predict_edge_ensemble(
            df,
            row_idx,
            target_col,
            opt_type,
            cols_by_type,
            strike_map,
            global_median_iv,
            already_filled,
            edge_weights=edge_weights,
        )
    else:
        info = predict_non_edge_local_poly(
            df,
            row_idx,
            target_col,
            opt_type,
            cols_by_type,
            strike_map,
            global_median_iv,
            ce_pchip_weight=ce_pchip_weight,
        )

    info["edge"] = bool(edge)
    info["edge_reason"] = edge_reason
    return info


def build_missing_cell_fill_order(df, cols_by_type, strike_map):
    missing_cells = []
    for row_idx in df.index:
        row = df.loc[row_idx]
        for opt_type in ["CE", "PE"]:
            state, left_fill, right_fill = get_edge_blocks(row, opt_type, cols_by_type, strike_map)
            missing_side_cols = [c for c in state["column"].tolist() if pd.isna(row[c])]
            if not missing_side_cols:
                continue
            edge_set = set(left_fill) | set(right_fill)
            interior = [c for c in state["column"].tolist() if c in missing_side_cols and c not in edge_set]
            ordered = list(left_fill) + interior + [c for c in right_fill if c not in left_fill]
            for col in ordered:
                missing_cells.append((row_idx, col))
    return missing_cells


# ---------------------------------------------------------------------
# Validation / tuning
# ---------------------------------------------------------------------

def artificial_masks(df, option_cols, mask_frac=CV_MASK_FRAC, n_reps=CV_REPS, seed=CV_SEED):
    rng = np.random.default_rng(seed)
    observed = [(r, c) for r in df.index for c in option_cols if pd.notna(df.at[r, c])]
    for rep in range(n_reps):
        idx = rng.choice(len(observed), int(len(observed) * mask_frac), replace=False)
        mask_set = [observed[i] for i in idx]
        df_m = df.copy()
        truths = {}
        for r, c in mask_set:
            truths[(r, c)] = float(df.at[r, c])
            df_m.at[r, c] = np.nan
        yield rep, df_m, truths, mask_set


def tune_interior_ce_pchip(df, option_cols, cols_by_type, strike_map, type_map, global_median_iv):
    if not SCIPY_AVAILABLE:
        return 0.0, pd.DataFrame()

    records = []
    for rep, df_m, truths, mask_set in artificial_masks(df, option_cols):
        for r, c in mask_set:
            opt_type = type_map[c]
            if opt_type != "CE":
                continue
            row = df_m.loc[r]
            edge, _, _, _, _ = is_edge_missing(row, c, opt_type, cols_by_type, strike_map)
            if edge:
                continue

            info = predict_non_edge_local_poly(
                df_m, r, c, opt_type, cols_by_type, strike_map, global_median_iv, ce_pchip_weight=0.0
            )
            wls = info.get("wls_prediction", info["prediction"])
            if not np.isfinite(wls):
                wls = global_median_iv

            spot = row["underlying_price"]
            x_obs, y_obs, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)
            pc = pchip_pred(x_obs, y_obs, strike_map[c] / spot) if pd.notna(spot) and spot > 0 else np.nan

            true = truths[(r, c)]
            for w in PCHIP_WEIGHT_GRID:
                pred = (1.0 - float(w)) * wls + float(w) * pc if np.isfinite(pc) else wls
                records.append({"family": "interior_ce_pchip", "rep": rep, "weight": float(w), "sq_err": (safe_iv(pred) - true) ** 2})

    val = pd.DataFrame(records)
    if val.empty:
        return 0.0, val

    summary = val.groupby("weight", as_index=False)["sq_err"].mean().sort_values("sq_err")
    base_mse = float(summary.loc[summary["weight"] == 0.0, "sq_err"].iloc[0])
    best_w = float(summary.iloc[0]["weight"])
    best_mse = float(summary.iloc[0]["sq_err"])

    if base_mse > 0 and (base_mse - best_mse) / base_mse >= MIN_REL_IMPROVEMENT_TO_ADOPT:
        chosen = best_w
    else:
        chosen = 0.0

    summary["chosen"] = summary["weight"] == chosen
    summary["baseline_mse"] = base_mse
    summary["relative_improvement_vs_w0"] = (base_mse - summary["sq_err"]) / base_mse if base_mse > 0 else np.nan
    return chosen, summary


def tune_edge_weights(df, option_cols, cols_by_type, strike_map, type_map, global_median_iv):
    records = []
    for rep, df_m, truths, mask_set in artificial_masks(df, option_cols):
        for r, c in mask_set:
            opt_type = type_map[c]
            row = df_m.loc[r]
            edge, _, _, _, _ = is_edge_missing(row, c, opt_type, cols_by_type, strike_map)
            if not edge:
                continue

            comp_info = predict_edge_components(
                df_m, r, c, opt_type, cols_by_type, strike_map, global_median_iv, already_filled={}
            )
            comps = comp_info["components"]
            true = truths[(r, c)]
            records.append({
                "rep": rep,
                "true": true,
                "claude": comps["claude"],
                "corrected": comps["corrected"],
                "quadratic": comps["quadratic"],
            })

    comp_df = pd.DataFrame(records)
    if comp_df.empty:
        return DEFAULT_EDGE_WEIGHTS, pd.DataFrame()

    rows = []
    step = EDGE_WEIGHT_STEP
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    for wc in grid:
        for wcor in grid:
            if wc + wcor > 1.0 + 1e-12:
                continue
            wq = round(1.0 - float(wc) - float(wcor), 2)
            pred = wc * comp_df["claude"] + wcor * comp_df["corrected"] + wq * comp_df["quadratic"]
            mse = float(np.mean((pred - comp_df["true"]) ** 2))
            rows.append({"family": "edge_weights", "w_claude": float(wc), "w_corrected": float(wcor), "w_quadratic": float(wq), "sq_err": mse})

    summary = pd.DataFrame(rows).sort_values("sq_err").reset_index(drop=True)
    wc0, wcor0, wq0 = DEFAULT_EDGE_WEIGHTS
    base_pred = wc0 * comp_df["claude"] + wcor0 * comp_df["corrected"] + wq0 * comp_df["quadratic"]
    base_mse = float(np.mean((base_pred - comp_df["true"]) ** 2))
    best = summary.iloc[0]
    best_weights = (float(best["w_claude"]), float(best["w_corrected"]), float(best["w_quadratic"]))
    best_mse = float(best["sq_err"])

    if base_mse > 0 and (base_mse - best_mse) / base_mse >= MIN_REL_IMPROVEMENT_TO_ADOPT:
        chosen = best_weights
    else:
        chosen = DEFAULT_EDGE_WEIGHTS

    summary["baseline_mse"] = base_mse
    summary["relative_improvement_vs_default"] = (base_mse - summary["sq_err"]) / base_mse if base_mse > 0 else np.nan
    summary["chosen"] = (
        np.isclose(summary["w_claude"], chosen[0])
        & np.isclose(summary["w_corrected"], chosen[1])
        & np.isclose(summary["w_quadratic"], chosen[2])
    )
    return chosen, summary


def parse_edge_weights_arg(s):
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError("--edge-weights must have exactly 3 comma-separated values.")
    total = sum(vals)
    if total <= 0:
        raise ValueError("--edge-weights must sum to a positive value.")
    return tuple(v / total for v in vals)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    data_path = Path(args.data)
    out_prefix = args.out_prefix
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

    if not data_path.exists():
        raise FileNotFoundError(f"Input not found: {data_path.resolve()}")

    filled_out = out_dir / f"filled_dataset_{out_prefix}.csv"
    submission_out = out_dir / f"submission_{out_prefix}.csv"
    diagnostics_out = out_dir / f"diagnostics_{out_prefix}.csv"
    tuning_out = out_dir / f"validation_tuning_{out_prefix}.csv"

    raw = pd.read_csv(data_path)
    df = raw.copy()
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["datetime_parsed"].isna().any():
        raise ValueError(f"{df['datetime_parsed'].isna().sum()} unparseable datetimes")
    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta = parse_metadata(df)
    option_cols = meta["column"].tolist()
    strike_map = dict(zip(meta["column"], meta["strike"]))
    type_map = dict(zip(meta["column"], meta["option_type"]))
    cols_by_type = {
        "CE": [c for c in option_cols if type_map[c] == "CE"],
        "PE": [c for c in option_cols if type_map[c] == "PE"],
    }
    global_median_iv = float(df[option_cols].stack().median())

    print(f"Timestamps     : {len(df)}")
    print(f"Missing cells  : {int(df[option_cols].isna().sum().sum())}")
    print(f"Scipy/PCHIP    : {'available' if SCIPY_AVAILABLE else 'NOT available'}")

    tuning_frames = []

    if args.ce_pchip_weight is not None:
        ce_pchip_weight = float(args.ce_pchip_weight)
        print(f"Using manual CE PCHIP weight: {ce_pchip_weight:.2f}")
    elif args.skip_tuning:
        ce_pchip_weight = 0.0
    else:
        print("Tuning CE-only interior PCHIP weight...")
        ce_pchip_weight, pchip_summary = tune_interior_ce_pchip(
            df, option_cols, cols_by_type, strike_map, type_map, global_median_iv
        )
        if not pchip_summary.empty:
            tuning_frames.append(pchip_summary)
        print(f"Chosen CE PCHIP weight: {ce_pchip_weight:.2f}")

    if args.edge_weights is not None:
        edge_weights = parse_edge_weights_arg(args.edge_weights)
        print(f"Using manual edge weights: {edge_weights}")
    elif args.skip_tuning:
        edge_weights = DEFAULT_EDGE_WEIGHTS
    else:
        print("Tuning edge ensemble weights...")
        edge_weights, edge_summary = tune_edge_weights(
            df, option_cols, cols_by_type, strike_map, type_map, global_median_iv
        )
        if not edge_summary.empty:
            tuning_frames.append(edge_summary)
        print(f"Chosen edge weights: claude={edge_weights[0]:.2f}, corrected={edge_weights[1]:.2f}, quadratic={edge_weights[2]:.2f}")

    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True, sort=False).to_csv(tuning_out, index=False)

    filled = df.copy()
    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)
    filled_values_by_row = {}
    rows = []
    diag = {
        "missing_initial": len(missing_cells),
        "filled": 0,
        "fallback_global_median": 0,
        "missing_after": None,
        "ce_pchip_weight": ce_pchip_weight,
        "edge_weight_claude": edge_weights[0],
        "edge_weight_corrected": edge_weights[1],
        "edge_weight_quadratic": edge_weights[2],
    }

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs"):
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})
        info = predict_cell(
            df,
            row_idx,
            col,
            opt_type,
            cols_by_type,
            strike_map,
            global_median_iv,
            already_filled,
            ce_pchip_weight=ce_pchip_weight,
            edge_weights=edge_weights,
        )
        pred = info["prediction"]
        if not np.isfinite(pred):
            pred = global_median_iv
            diag["fallback_global_median"] += 1
        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred

        components = info.get("component_predictions", {})
        filled_values_by_row[row_idx][col] = {
            "final": pred,
            "claude": components.get("claude", pred),
            "corrected": components.get("corrected", pred),
            "quadratic": components.get("quadratic", pred),
        }

        diag["filled"] += 1
        if info.get("selected_model") == "fallback_global_median":
            diag["fallback_global_median"] += 1

        rows.append({
            "row_index": row_idx,
            "datetime": df.loc[row_idx, "datetime"],
            "contract": col,
            "option_type": opt_type,
            "strike": strike_map[col],
            "final_prediction": pred,
            "edge": info["edge"],
            "edge_reason": info["edge_reason"],
            "source": info["source"],
            "selected_model": info["selected_model"],
            "bandwidth": info["bandwidth"],
            "loo_mse": info["loo_mse"],
            "edge_degree": info.get("edge_degree", np.nan),
            "n_train": info["n_train"],
            "used_cols": "|".join(map(str, info["used_cols"])),
            "wls_prediction": info.get("wls_prediction", np.nan),
            "pchip_prediction": info.get("pchip_prediction", np.nan),
            "ce_pchip_weight": info.get("ce_pchip_weight", 0.0),
            "edge_claude_prediction": info.get("component_predictions", {}).get("claude", np.nan),
            "edge_corrected_prediction": info.get("component_predictions", {}).get("corrected", np.nan),
            "edge_quadratic_prediction": info.get("component_predictions", {}).get("quadratic", np.nan),
            "edge_side": info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
            "edge_observed_side_points": info.get("edge_observed_side_points", np.nan),
            "edge_base_observed_needed": info.get("edge_base_observed_needed", np.nan),
        })

    diag["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled_df = filled.drop(columns=["datetime_parsed"])
    orig_df = df.drop(columns=["datetime_parsed"])
    filled_df.to_csv(filled_out, index=False)
    sub = make_submission(orig_df, filled_df, submission_out)
    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(diagnostics_out, index=False)

    print(f"✅ Filled dataset → {filled_out}")
    print(f"✅ Submission     → {submission_out} ({len(sub)} rows)")
    print(f"✅ Diagnostics    → {diagnostics_out}")
    if tuning_frames:
        print(f"✅ Tuning summary → {tuning_out}")
    print(f"Missing after     : {diag['missing_after']}")
    print(f"CE PCHIP weight   : {ce_pchip_weight:.2f}")
    print(f"Edge weights      : {edge_weights}")
    if diag_df[diag_df.edge].shape[0] > 0:
        print(f"Edge degree chosen: {diag_df[diag_df.edge]['edge_degree'].value_counts().to_dict()}")
    print("\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
