"""
try_regime_edge_submit.py
=========================

Submission generator matching Claude's validated improvement:

A) Interior/non-edge:
   same-row, same-option-type local quadratic WLS, but bandwidth grid is regime-specific:
       Jan 27  -> [7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4]
       Others  -> [7e-5, 1e-4, 1.5e-4, 2e-4]

B) Edge:
   same progressive edge ensemble as try.py, but for Jan 27 wings:
       CE right edge -> edge local-poly bandwidth = 5e-4
       PE left edge  -> edge local-poly bandwidth = 5e-4
   otherwise edge bandwidth = 2e-4.

Run:
    python try_regime_edge_submit.py --data /path/to/dataset.csv

Outputs:
    filled_dataset_regime_edge.csv
    submission_regime_edge.csv
    diagnostics_regime_edge.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv")

EPS_IV = 1e-6
SEPARATOR = "||"

# Claude validation grids
BW_CURR = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
BW_J27  = np.array([7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4], dtype=float)
BW_OTH  = np.array([7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)

EDGE_LOCAL_POLY_BW_DEFAULT = 2e-4
EDGE_LOCAL_POLY_BW_J27_WING = 5e-4

EDGE_BLEND_CLAUDE = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14

MIN_EDGE_LOCAL_NEIGHBORS = 3
LOCAL_POLY_DEGREE = 2
JAN27_DATE = pd.Timestamp("2026-01-27").date()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create submission using validated regime bandwidth + Jan27 wing-edge bandwidth improvement."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input dataset CSV.")
    parser.add_argument("--out-prefix", type=str, default="regime_edge", help="Output filename prefix.")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def safe_iv(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


def parse_metadata(df: pd.DataFrame) -> pd.DataFrame:
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


def make_submission(original: pd.DataFrame, filled: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for col in [c for c in original.columns if c != "datetime"]:
        was_missing = original[col].isna()
        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})

    sub = pd.DataFrame(rows, columns=["id", "value"]).sort_values("id").reset_index(drop=True)
    sub.to_csv(out_path, index=False)
    return sub


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


def fit_quadratic(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return None, "no_points"
    if len(y) == 1:
        return np.array([0.0, 0.0, float(y[0])]), "constant"
    if len(y) == 2:
        coeff = np.polyfit(x, y, 1)
        return np.array([0.0, float(coeff[0]), float(coeff[1])]), "linear"

    coeff = np.polyfit(x, y, 2)
    return np.array([float(coeff[0]), float(coeff[1]), float(coeff[2])]), "quadratic"


def eval_quadratic(coeff, x):
    if coeff is None:
        return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


def is_jan27(row) -> bool:
    return pd.notna(row["datetime_parsed"]) and row["datetime_parsed"].date() == JAN27_DATE


# ---------------------------------------------------------------------
# Local polynomial WLS
# ---------------------------------------------------------------------

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs = x_obs[mask]
    y_obs = y_obs[mask]

    if len(y_obs) == 0:
        return np.nan
    if len(y_obs) == 1:
        return safe_iv(float(y_obs[0]))

    actual_degree = min(degree, len(y_obs) - 1)
    dx = x_obs - x_target
    weights = np.exp(-(dx ** 2) / (2.0 * bandwidth))
    X = np.column_stack([dx ** j for j in range(actual_degree + 1)])

    WX = X * weights[:, None]
    lhs = X.T @ WX
    rhs = X.T @ (weights * y_obs)

    try:
        coeff = np.linalg.solve(lhs, rhs)
        return safe_iv(float(coeff[0]))
    except np.linalg.LinAlgError:
        wsum = float(weights.sum())
        if wsum <= 1e-15:
            return np.nan
        return safe_iv(float((weights @ y_obs) / wsum))


def local_poly_wls_loo_preds(x_obs, y_obs, bandwidth):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    preds = np.full(len(y_obs), np.nan)

    for i in range(len(y_obs)):
        preds[i] = local_poly_wls_pred(
            np.delete(x_obs, i),
            np.delete(y_obs, i),
            x_obs[i],
            bandwidth,
            degree=LOCAL_POLY_DEGREE,
        )

    return preds


def select_bandwidth_by_loo(x_obs, y_obs, bandwidth_grid):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

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
            best_mse = mse
            best_bw = float(bw)

    return best_bw, best_mse


# ---------------------------------------------------------------------
# Same-row helpers
# ---------------------------------------------------------------------

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []

    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    x = np.array([strike_map[c] / spot for c in obs_cols], dtype=float)
    y = np.array([row[c] for c in obs_cols], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    kept_cols = [c for c, keep in zip(obs_cols, mask) if keep]
    return x[mask], y[mask], kept_cols


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = []
    for col in cols_by_type[opt_type]:
        records.append({
            "column": col,
            "strike": strike_map[col],
            "is_missing": pd.isna(row[col]),
            "iv": row[col],
        })
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

    left_fill_order = list(reversed(left_block))
    right_fill_order = list(reversed(right_block))
    return state, left_fill_order, right_fill_order


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


def edge_bandwidth_for(row, opt_type, edge_side):
    """
    Claude's validated B:
        Jan27 CE right edge -> 5e-4
        Jan27 PE left edge  -> 5e-4
        all other edge cases -> 2e-4
    """
    if is_jan27(row) and ((opt_type == "CE" and edge_side == "right") or (opt_type == "PE" and edge_side == "left")):
        return EDGE_LOCAL_POLY_BW_J27_WING
    return EDGE_LOCAL_POLY_BW_DEFAULT


# ---------------------------------------------------------------------
# Prediction components
# ---------------------------------------------------------------------

def predict_non_edge_regime_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv):
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
        }

    grid = BW_J27 if is_jan27(row) else BW_OTH
    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, grid)
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "regime_local_quadratic_wls"

    return {
        "prediction": safe_iv(pred),
        "source": "same_row_non_edge_regime_local_poly_wls",
        "selected_model": selected_model,
        "bandwidth": best_bw,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
    }


def collect_edge_training_points_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {
            "edge_side": "bad_spot",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
        }

    edge, edge_reason, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map
    )

    if not edge:
        return np.array([]), np.array([]), [], {
            "edge_side": "not_edge",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
        }

    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    observed_records = []

    for _, rec in state.iterrows():
        col = rec["column"]
        if pd.notna(row[col]):
            observed_records.append({
                "column": col,
                "strike": strike_map[col],
                "x": strike_map[col] / spot,
                "y": float(row[col]),
                "is_predicted": False,
            })

    if not observed_records:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
        }

    obs = pd.DataFrame(observed_records)
    target_strike = strike_map[target_col]

    if side == "right":
        base = obs[obs["strike"] < target_strike].sort_values("strike", ascending=True)
    elif side == "left":
        base = obs[obs["strike"] > target_strike].sort_values("strike", ascending=True)
    else:
        base = obs.sort_values("strike", ascending=True)

    x_train = base["x"].to_list()
    y_train = base["y"].to_list()
    used_cols = base["column"].astype(str).to_list()

    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        if prev_col not in already_filled:
            continue

        prev_pred = component_value(already_filled, prev_col, "claude")
        if not np.isfinite(prev_pred):
            continue

        # Preserved Claude-style progressive update:
        # prediction is appended as BOTH x and y.
        x_train.append(float(prev_pred))
        y_train.append(float(prev_pred))
        used_cols.append(f"{prev_col}*as_xy")

    return (
        np.asarray(x_train, dtype=float),
        np.asarray(y_train, dtype=float),
        used_cols,
        {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
        },
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

    edge, edge_reason, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map
    )

    if not edge:
        return np.array([]), np.array([]), [], {
            "edge_side": "not_edge",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "edge_observed_side_points": 0,
        }

    target_strike = strike_map[target_col]
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    train_records = []
    for _, rec in state.iterrows():
        col = rec["column"]
        val = row[col]
        if pd.isna(val):
            continue

        strike = strike_map[col]
        if (side == "right" and strike < target_strike) or (side == "left" and strike > target_strike):
            train_records.append({
                "column": col,
                "strike": strike,
                "x": strike / spot,
                "y": float(val),
                "is_predicted": False,
            })

    observed_side_points = len(train_records)
    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "corrected")
        if not np.isfinite(prev_pred):
            continue

        train_records.append({
            "column": prev_col,
            "strike": strike_map[prev_col],
            "x": strike_map[prev_col] / spot,
            "y": float(prev_pred),
            "is_predicted": True,
        })

    train = pd.DataFrame(train_records)
    if train.empty:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_observed_side_points": observed_side_points,
        }

    train = train.sort_values("strike").reset_index(drop=True)
    used_cols = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]

    return (
        train["x"].to_numpy(dtype=float),
        train["y"].to_numpy(dtype=float),
        used_cols,
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

    edge, edge_reason, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map
    )

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

    observed_records = []
    for _, rec in state.iterrows():
        col = rec["column"]
        if pd.notna(row[col]):
            observed_records.append({
                "column": col,
                "strike": strike_map[col],
                "x": strike_map[col] / spot,
                "y": float(row[col]),
                "is_predicted": False,
            })

    obs = pd.DataFrame(observed_records)
    if obs.empty:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_base_observed_needed": base_needed,
        }

    if side == "right":
        base = (
            obs[obs["strike"] < target_strike]
            .sort_values("strike", ascending=False)
            .head(base_needed)
            .sort_values("strike")
        )
    elif side == "left":
        base = (
            obs[obs["strike"] > target_strike]
            .sort_values("strike", ascending=True)
            .head(base_needed)
            .sort_values("strike")
        )
    else:
        base = obs.sort_values("strike")

    train_records = base.to_dict(orient="records")
    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []
    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "quadratic")
        if not np.isfinite(prev_pred):
            continue

        train_records.append({
            "column": prev_col,
            "strike": strike_map[prev_col],
            "x": strike_map[prev_col] / spot,
            "y": float(prev_pred),
            "is_predicted": True,
        })

    train = pd.DataFrame(train_records)
    if train.empty:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_base_observed_needed": base_needed,
        }

    train = train.sort_values("strike").reset_index(drop=True)
    used_cols = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]

    return (
        train["x"].to_numpy(dtype=float),
        train["y"].to_numpy(dtype=float),
        used_cols,
        {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "edge_base_observed_needed": base_needed,
        },
    )


def predict_edge_claude_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_bad_spot",
            "selected_model": "fallback_global_median",
            "bandwidth": EDGE_LOCAL_POLY_BW_DEFAULT,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": [],
            "edge_side": "bad_spot",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
        }

    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_claude(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
        already_filled=already_filled,
    )

    bw = edge_bandwidth_for(row, opt_type, edge_info.get("edge_side", ""))

    if len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_no_neighbors",
            "selected_model": "fallback_global_median",
            "bandwidth": bw,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": used_cols,
            **edge_info,
        }

    x_target = strike_map[target_col] / spot
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_claude_progressive_local_poly_wls_adaptive_bw"

    fitted = np.array([
        local_poly_wls_pred(x_obs, y_obs, x, bw, degree=LOCAL_POLY_DEGREE)
        for x in x_obs
    ], dtype=float)
    mask = np.isfinite(fitted) & np.isfinite(y_obs)
    fit_mse = float(np.mean((fitted[mask] - y_obs[mask]) ** 2)) if mask.any() else np.nan

    return {
        "prediction": safe_iv(pred),
        "source": "edge_claude_progressive_local_poly_wls_adaptive_bw",
        "selected_model": selected_model,
        "bandwidth": bw,
        "loo_mse": fit_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        **edge_info,
    }


def predict_edge_corrected_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_corrected(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled
    )

    bw = edge_bandwidth_for(row, opt_type, edge_info.get("edge_side", ""))

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used_cols, edge_info, bw

    x_target = strike_map[target_col] / spot
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, bw, degree=LOCAL_POLY_DEGREE)
    if not np.isfinite(pred):
        pred = global_median_iv

    return safe_iv(pred), used_cols, edge_info, bw


def predict_edge_quadratic(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_quadratic(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled
    )

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used_cols, edge_info, np.nan

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    pred = eval_quadratic(coeff, strike_map[target_col] / spot)
    if not np.isfinite(pred):
        pred = global_median_iv

    return safe_iv(pred), used_cols, edge_info, fit_kind


def predict_edge_ensemble(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    claude_info = predict_edge_claude_local_poly(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )

    corrected_pred, corrected_cols, corrected_info, corrected_bw = predict_edge_corrected_local_poly(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )

    quadratic_pred, quadratic_cols, quadratic_info, quadratic_fit_kind = predict_edge_quadratic(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )

    components = {
        "claude": safe_iv(claude_info["prediction"]),
        "corrected": safe_iv(corrected_pred),
        "quadratic": safe_iv(quadratic_pred),
    }

    pred = (
        EDGE_BLEND_CLAUDE * components["claude"]
        + EDGE_BLEND_CORRECTED * components["corrected"]
        + EDGE_BLEND_QUADRATIC * components["quadratic"]
    )

    selected_model = "edge_blended_progressive_wls_quadratic_adaptive_bw"
    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"

    return {
        **claude_info,
        "prediction": safe_iv(pred),
        "source": "edge_blended_progressive_wls_quadratic_adaptive_bw",
        "selected_model": selected_model,
        "component_predictions": components,
        "corrected_used_cols": corrected_cols,
        "quadratic_used_cols": quadratic_cols,
        "quadratic_fit_kind": quadratic_fit_kind,
        "corrected_bandwidth": corrected_bw,
        "edge_observed_side_points": corrected_info.get("edge_observed_side_points", np.nan),
        "edge_base_observed_needed": quadratic_info.get("edge_base_observed_needed", np.nan),
    }


def predict_cell(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    edge, edge_reason, _, _, _ = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)

    if edge:
        info = predict_edge_ensemble(
            df=df,
            row_idx=row_idx,
            target_col=target_col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
            already_filled=already_filled,
        )
    else:
        info = predict_non_edge_regime_local_poly(
            df=df,
            row_idx=row_idx,
            target_col=target_col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
        )

    info["edge"] = bool(edge)
    info["edge_reason"] = edge_reason
    info["is_jan27"] = is_jan27(row)
    return info


# ---------------------------------------------------------------------
# Fill order
# ---------------------------------------------------------------------

def build_missing_cell_fill_order(df, cols_by_type, strike_map):
    missing_cells = []

    for row_idx in df.index:
        row = df.loc[row_idx]

        for opt_type in ["CE", "PE"]:
            state, left_fill_order, right_fill_order = get_edge_blocks(
                row, opt_type, cols_by_type, strike_map
            )

            missing_side_cols = [c for c in state["column"].tolist() if pd.isna(row[c])]

            if not missing_side_cols:
                continue

            edge_set = set(left_fill_order) | set(right_fill_order)

            interior = [
                c for c in state["column"].tolist()
                if c in missing_side_cols and c not in edge_set
            ]

            ordered = []
            ordered.extend(left_fill_order)
            ordered.extend(interior)
            ordered.extend([c for c in right_fill_order if c not in ordered])

            for col in ordered:
                missing_cells.append((row_idx, col))

    return missing_cells


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    data_path = Path(args.data)
    out_prefix = args.out_prefix

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find input file: {data_path.resolve()}")

    filled_out = Path(f"filled_dataset_{out_prefix}.csv")
    submission_out = Path(f"submission_{out_prefix}.csv")
    diagnostics_out = Path(f"diagnostics_{out_prefix}.csv")

    raw = pd.read_csv(data_path)
    df = raw.copy()

    df["datetime_parsed"] = pd.to_datetime(
        df["datetime"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if df["datetime_parsed"].isna().any():
        bad = int(df["datetime_parsed"].isna().sum())
        raise ValueError(f"{bad} datetime values could not be parsed.")

    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta = parse_metadata(df)
    option_cols = meta["column"].tolist()

    strike_map = dict(zip(meta["column"], meta["strike"]))
    type_map = dict(zip(meta["column"], meta["option_type"]))

    cols_by_type = {
        "CE": sorted([c for c in option_cols if type_map[c] == "CE"], key=lambda c: strike_map[c]),
        "PE": sorted([c for c in option_cols if type_map[c] == "PE"], key=lambda c: strike_map[c]),
    }

    global_median_iv = float(df[option_cols].stack().median())
    filled = df.copy()

    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)
    filled_values_by_row = {}

    diagnostics = {
        "missing_initial": int(df[option_cols].isna().sum().sum()),
        "filled": 0,
        "missing_after": None,
        "non_edge_regime_local_poly": 0,
        "edge_adaptive_bw_ensemble": 0,
        "fallback_global_median": 0,
        "jan27_missing_cells": 0,
        "jan27_wing_edge_bw_5e_4": 0,
        "edge_no_left_observed": 0,
        "edge_no_right_observed": 0,
        "edge_no_observed_same_side": 0,
        "not_edge": 0,
    }

    rows = []

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs with regime+edge improvement"):
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})

        info = predict_cell(
            df=filled,
            row_idx=row_idx,
            target_col=col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
            already_filled=already_filled,
        )

        pred = info["prediction"]

        if not np.isfinite(pred):
            pred = global_median_iv
            diagnostics["fallback_global_median"] += 1

        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred

        components = info.get("component_predictions", {})
        filled_values_by_row[row_idx][col] = {
            "final": pred,
            "claude": components.get("claude", pred),
            "corrected": components.get("corrected", pred),
            "quadratic": components.get("quadratic", pred),
        }

        diagnostics["filled"] += 1
        if info["edge"]:
            diagnostics["edge_adaptive_bw_ensemble"] += 1
        else:
            diagnostics["non_edge_regime_local_poly"] += 1

        if info.get("is_jan27", False):
            diagnostics["jan27_missing_cells"] += 1

        if info.get("bandwidth") == EDGE_LOCAL_POLY_BW_J27_WING:
            diagnostics["jan27_wing_edge_bw_5e_4"] += 1

        if info["edge_reason"] in diagnostics:
            diagnostics[info["edge_reason"]] += 1

        if info["selected_model"] == "fallback_global_median":
            diagnostics["fallback_global_median"] += 1

        rows.append({
            "row_index": row_idx,
            "datetime": filled.loc[row_idx, "datetime"],
            "contract": col,
            "option_type": opt_type,
            "strike": strike_map[col],
            "final_prediction": pred,
            "edge": info["edge"],
            "edge_reason": info["edge_reason"],
            "is_jan27": info.get("is_jan27", False),
            "source": info["source"],
            "selected_model": info["selected_model"],
            "bandwidth": info.get("bandwidth", np.nan),
            "corrected_bandwidth": info.get("corrected_bandwidth", np.nan),
            "loo_mse": info.get("loo_mse", np.nan),
            "n_train": info.get("n_train", np.nan),
            "edge_side": info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
            "edge_claude_prediction": info.get("component_predictions", {}).get("claude", np.nan),
            "edge_corrected_prediction": info.get("component_predictions", {}).get("corrected", np.nan),
            "edge_quadratic_prediction": info.get("component_predictions", {}).get("quadratic", np.nan),
            "edge_quadratic_fit_kind": info.get("quadratic_fit_kind", np.nan),
        })

    diagnostics["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled_out_df = filled.drop(columns=["datetime_parsed"])
    original_out_df = df.drop(columns=["datetime_parsed"])

    filled_out_df.to_csv(filled_out, index=False)
    submission = make_submission(original_out_df, filled_out_df, submission_out)
    pd.DataFrame(rows).to_csv(diagnostics_out, index=False)

    print(f"✅ Filled dataset saved → {filled_out}")
    print(f"✅ Submission saved → {submission_out} ({len(submission)} rows)")
    print(f"✅ Diagnostics saved → {diagnostics_out}")
    print()
    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
