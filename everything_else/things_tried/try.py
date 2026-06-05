"""
try.py: stronger IV imputer using local-polynomial WLS + progressive edge LP(2e-4)
=================================================================================

This is built from the strongest validated variant currently in the project:
strategies_and_results/m4_2nd_try_chatgpt/final_claude_progressive_edge_lp2e4_submission.py

Core rule
---------
Non-edge missing values:
    Use the current best method:
        same-row, same-option-type local quadratic WLS
        bandwidth selected by row-wise leave-one-out
        BANDWIDTH_GRID = [5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4]

Edge missing values:
    Use a small ensemble of three progressive edge models:
        72% Claude-style local polynomial edge
        14% corrected local polynomial edge
        14% M3-style progressive quadratic edge

    If there is a right-edge missing block:
        observed observed observed missing_1 missing_2 missing_3 ...

    Fill from the observed boundary outward:
        missing_1: fit local quadratic WLS on observed points, predict missing_1
        missing_2: fit local quadratic WLS on observed points + missing_1 prediction
        missing_3: fit local quadratic WLS on observed points + missing_1 + missing_2 predictions

    Same symmetric logic for left-edge blocks.

Important
---------
The progressive edge update intentionally follows the locally best-scoring
benchmark variant:
    train_x.append(prediction)
    train_y.append(prediction)

This is unusual, but it is preserved because that exact edge rule gave the best
stored CV score in this repo:
    MSE ~= 0.00015778
versus the active M3 local-poly adaptive family, which used a weaker edge rule.

Run
---
    python try.py --data cv_validation_system/dataset.csv

Outputs
-------
    filled_dataset_try.csv
    submission_try.csv
    diagnostics_try.csv
    cross_section_diagnostics_try.csv
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

# Best-till-now non-edge method.
BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)

# Edge ensemble. These weights were chosen on the local synthetic CV split.
EDGE_LOCAL_POLY_BW = 2e-4
EDGE_BLEND_CLAUDE = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14
MIN_EDGE_LOCAL_NEIGHBORS = 3
LOCAL_POLY_DEGREE = 2


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stronger IV imputer using local-poly WLS and progressive edge LP(2e-4)."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input dataset CSV.")
    parser.add_argument("--out-prefix", type=str, default="try", help="Output filename prefix.")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def parse_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Parse option columns like NIFTY27JAN2625200CE."""
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

        match = pattern.match(col)
        if match:
            item = match.groupdict()
            item["column"] = col
            item["strike"] = int(item["strike"])
            item["expiry_date"] = pd.to_datetime(item["expiry"], format="%d%b%y", errors="coerce")
            records.append(item)

    meta = pd.DataFrame(records)

    if meta.empty:
        raise ValueError("No option columns parsed. Check column names.")

    return meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)


def safe_iv(x: float) -> float:
    """Keep IV finite and positive."""
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


def component_value(already_filled, col, key):
    """Return a previous edge component prediction, supporting old scalar entries."""
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


def make_submission(original: pd.DataFrame, filled: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Create submission from cells missing in the input file."""
    rows = []

    for col in [c for c in original.columns if c != "datetime"]:
        was_missing = original[col].isna()

        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})

    submission = pd.DataFrame(rows, columns=["id", "value"])
    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv(out_path, index=False)
    return submission


def fit_quadratic(x, y):
    """Global quadratic fit with linear/constant fallbacks."""
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


# ---------------------------------------------------------------------
# Local polynomial WLS
# ---------------------------------------------------------------------

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    """
    Local polynomial regression at one target point.

    Fits:
        min_beta sum_i w_i * (y_i - beta_0 - beta_1 dx_i - beta_2 dx_i^2)^2

    where:
        dx_i = x_i - x_target
        w_i  = exp(-(x_i - x_target)^2 / (2 * bandwidth))

    Prediction is beta_0.
    """
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
    dist2 = dx ** 2
    weights = np.exp(-dist2 / (2.0 * bandwidth))

    X = np.column_stack([dx ** j for j in range(actual_degree + 1)])

    # Avoid constructing a full W matrix.
    WX = X * weights[:, None]
    lhs = X.T @ WX
    rhs = X.T @ (weights * y_obs)

    try:
        coeff = np.linalg.solve(lhs, rhs)
        pred = float(coeff[0])
        return safe_iv(pred)
    except np.linalg.LinAlgError:
        wsum = float(weights.sum())
        if wsum <= 1e-15:
            return np.nan
        return safe_iv(float((weights @ y_obs) / wsum))


def local_poly_wls_loo_preds(x_obs, y_obs, bandwidth):
    """Leave-one-out predictions for local polynomial WLS."""
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


def select_bandwidth_by_loo(x_obs, y_obs, bandwidth_grid=BANDWIDTH_GRID):
    """Select local-poly bandwidth by row-wise LOO MSE."""
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
# Same-row structure helpers
# ---------------------------------------------------------------------

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    """
    Collect observed points for one timestamp and one option type.

    x = strike / underlying_price
    y = IV
    """
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
    """Return CE or PE columns at this timestamp, ordered by strike."""
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
    """
    Identify left-edge and right-edge missing blocks for this row/option type.

    Left-edge block:
        missing missing observed observed ...

    Fill order:
        nearest observed boundary first, so reverse(left_block)

    Right-edge block:
        ... observed observed missing missing

    Fill order:
        nearest observed boundary first, so natural left-to-right order.
    """
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
    """Check whether target_col is edge missing within CE or PE only."""
    state, left_fill_order, right_fill_order = get_edge_blocks(row, opt_type, cols_by_type, strike_map)

    if target_col in set(left_fill_order):
        return True, "edge_no_left_observed", "left", left_fill_order, left_fill_order.index(target_col)

    if target_col in set(right_fill_order):
        return True, "edge_no_right_observed", "right", right_fill_order, right_fill_order.index(target_col)

    # If all same-side points are missing, both blocks may overlap.
    same_side_missing = set(state.loc[state["is_missing"], "column"])
    same_side_observed = set(state.loc[~state["is_missing"], "column"])
    if target_col in same_side_missing and len(same_side_observed) == 0:
        return True, "edge_no_observed_same_side", "all_missing", list(state["column"]), 0

    return False, "not_edge", "", [], np.nan


# ---------------------------------------------------------------------
# Prediction components
# ---------------------------------------------------------------------

def predict_non_edge_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv):
    """
    Non-edge prediction:
        same-row, same-option-type local quadratic WLS
        with bandwidth selected by LOO.
    """
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

    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)

    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "local_quadratic_wls"

    return {
        "prediction": safe_iv(pred),
        "source": "same_row_non_edge_local_poly_wls",
        "selected_model": selected_model,
        "bandwidth": best_bw,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
    }


def collect_edge_training_points_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    """
    Build Claude-style edge training points for the target.

    This follows the reference benchmark's spirit:

        x_train starts as observed-side moneyness values.
        y_train starts as observed-side IV values.

        For previously filled edge points:
            append prediction to x_train
            append prediction to y_train

    Yes, this means predicted IV is appended as x as well. This is intentionally
    kept because the user asked for Claude's exact logic.
    """
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
        # Right edge: use all actually observed points to the left.
        base = obs[obs["strike"] < target_strike].sort_values("strike", ascending=True)
    elif side == "left":
        # Left edge: use all actually observed points to the right.
        base = obs[obs["strike"] > target_strike].sort_values("strike", ascending=True)
    else:
        # Degenerate all-missing case.
        base = obs.sort_values("strike", ascending=True)

    x_train = base["x"].to_list()
    y_train = base["y"].to_list()
    used_cols = base["column"].astype(str).to_list()

    # Add previously predicted values inside this edge block.
    previous_cols = block_cols[:int(position)] if np.isfinite(position) else []

    for prev_col in previous_cols:
        if prev_col not in already_filled:
            continue

        prev_pred = component_value(already_filled, prev_col, "claude")

        if not np.isfinite(prev_pred):
            continue

        # This is the exact Claude-style progressive update.
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
    """
    Corrected progressive edge training set.

    It uses the real moneyness coordinate for prior edge predictions:
        x = strike / spot
        y = predicted IV
    """
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
    used_cols = [
        f"{r.column}{'*' if r.is_predicted else ''}"
        for r in train.itertuples(index=False)
    ]

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
    """
    M3-style progressive edge set: nearest observed same-side neighbours plus
    previous predictions in this edge block.
    """
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
    used_cols = [
        f"{r.column}{'*' if r.is_predicted else ''}"
        for r in train.itertuples(index=False)
    ]

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


def predict_edge_corrected_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_corrected(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled
    )

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used_cols, edge_info

    x_target = strike_map[target_col] / spot
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, EDGE_LOCAL_POLY_BW, degree=LOCAL_POLY_DEGREE)
    if not np.isfinite(pred):
        pred = global_median_iv
    return safe_iv(pred), used_cols, edge_info


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


def predict_edge_claude_local_poly(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    """
    Edge prediction using Claude progressive local-poly LP(2e-4).

    Prediction:
        local_poly_wls_pred(x_train, y_train, x_target, bandwidth=2e-4)
    """
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_bad_spot",
            "selected_model": "fallback_global_median",
            "bandwidth": EDGE_LOCAL_POLY_BW,
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

    if len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_no_neighbors",
            "selected_model": "fallback_global_median",
            "bandwidth": EDGE_LOCAL_POLY_BW,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": used_cols,
            **edge_info,
        }

    x_target = strike_map[target_col] / spot

    pred = local_poly_wls_pred(
        x_obs=x_obs,
        y_obs=y_obs,
        x_target=x_target,
        bandwidth=EDGE_LOCAL_POLY_BW,
        degree=LOCAL_POLY_DEGREE,
    )

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_claude_progressive_local_poly_wls"

    # Diagnostic in-sample fit error only.
    fitted = np.array([
        local_poly_wls_pred(x_obs, y_obs, x, EDGE_LOCAL_POLY_BW, degree=LOCAL_POLY_DEGREE)
        for x in x_obs
    ], dtype=float)
    mask = np.isfinite(fitted) & np.isfinite(y_obs)
    fit_mse = float(np.mean((fitted[mask] - y_obs[mask]) ** 2)) if mask.any() else np.nan

    return {
        "prediction": safe_iv(pred),
        "source": "edge_claude_progressive_local_poly_wls",
        "selected_model": selected_model,
        "bandwidth": EDGE_LOCAL_POLY_BW,
        "loo_mse": fit_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        **edge_info,
    }


def predict_edge_ensemble(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    """Blend the three strongest edge variants; leave non-edge logic unchanged."""
    claude_info = predict_edge_claude_local_poly(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )
    claude_pred = claude_info["prediction"]
    corrected_pred, corrected_cols, corrected_info = predict_edge_corrected_local_poly(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )
    quadratic_pred, quadratic_cols, quadratic_info, quadratic_fit_kind = predict_edge_quadratic(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
    )

    components = {
        "claude": safe_iv(claude_pred),
        "corrected": safe_iv(corrected_pred),
        "quadratic": safe_iv(quadratic_pred),
    }
    pred = (
        EDGE_BLEND_CLAUDE * components["claude"]
        + EDGE_BLEND_CORRECTED * components["corrected"]
        + EDGE_BLEND_QUADRATIC * components["quadratic"]
    )

    selected_model = "edge_blended_progressive_wls_quadratic"
    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"

    return {
        **claude_info,
        "prediction": safe_iv(pred),
        "source": "edge_blended_progressive_wls_quadratic",
        "selected_model": selected_model,
        "component_predictions": components,
        "corrected_used_cols": corrected_cols,
        "quadratic_used_cols": quadratic_cols,
        "quadratic_fit_kind": quadratic_fit_kind,
        "edge_observed_side_points": corrected_info.get("edge_observed_side_points", np.nan),
        "edge_base_observed_needed": quadratic_info.get("edge_base_observed_needed", np.nan),
    }


def predict_cell(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    """Route one missing cell to edge or non-edge model."""
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
        info = predict_non_edge_local_poly(
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

    return info


# ---------------------------------------------------------------------
# Fill-order construction
# ---------------------------------------------------------------------

def build_missing_cell_fill_order(df, cols_by_type, strike_map):
    """
    Build deterministic fill order:
        for each row and option type:
            left edge block, nearest boundary first
            interior missing cells by strike order
            right edge block, nearest boundary first
    """
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
    cross_diag_out = Path(f"cross_section_diagnostics_{out_prefix}.csv")

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
        "CE": [c for c in option_cols if type_map[c] == "CE"],
        "PE": [c for c in option_cols if type_map[c] == "PE"],
    }

    global_median_iv = float(df[option_cols].stack().median())
    filled = df.copy()

    diagnostics = {
        "missing_initial": int(df[option_cols].isna().sum().sum()),
        "filled": 0,
        "edge_blended_progressive_wls_quadratic": 0,
        "edge_claude_progressive_local_poly_wls": 0,
        "same_row_non_edge_local_poly_wls": 0,
        "fallback_global_median": 0,
        "edge_no_left_observed": 0,
        "edge_no_right_observed": 0,
        "edge_no_observed_same_side": 0,
        "not_edge": 0,
        "missing_after": None,
    }

    rows = []
    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)

    filled_values_by_row = {}

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs with edge ensemble"):
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})

        info = predict_cell(
            df=df,
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

        # Make component predictions available to the next missing value in the same edge block.
        components = info.get("component_predictions", {})
        filled_values_by_row[row_idx][col] = {
            "final": pred,
            "claude": components.get("claude", pred),
            "corrected": components.get("corrected", pred),
            "quadratic": components.get("quadratic", pred),
        }

        diagnostics["filled"] += 1

        if info["source"] in diagnostics:
            diagnostics[info["source"]] += 1

        if info["edge_reason"] in diagnostics:
            diagnostics[info["edge_reason"]] += 1

        if info["selected_model"] == "fallback_global_median":
            diagnostics["fallback_global_median"] += 1

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
            "n_train": info["n_train"],
            "used_cols": "|".join(map(str, info["used_cols"])),
            "edge_claude_prediction": info.get("component_predictions", {}).get("claude", np.nan),
            "edge_corrected_prediction": info.get("component_predictions", {}).get("corrected", np.nan),
            "edge_quadratic_prediction": info.get("component_predictions", {}).get("quadratic", np.nan),
            "edge_corrected_used_cols": "|".join(map(str, info.get("corrected_used_cols", []))),
            "edge_quadratic_used_cols": "|".join(map(str, info.get("quadratic_used_cols", []))),
            "edge_quadratic_fit_kind": info.get("quadratic_fit_kind", np.nan),
            "edge_side": info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
            "edge_observed_side_points": info.get("edge_observed_side_points", np.nan),
            "edge_base_observed_needed": info.get("edge_base_observed_needed", np.nan),
        })

    diagnostics["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled_out_df = filled.drop(columns=["datetime_parsed"])
    original_out_df = df.drop(columns=["datetime_parsed"])

    filled_out_df.to_csv(filled_out, index=False)
    submission = make_submission(original_out_df, filled_out_df, submission_out)

    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df.to_csv(diagnostics_out, index=False)
    diagnostics_df.to_csv(cross_diag_out, index=False)

    print(f"✅ Filled dataset saved → {filled_out}")
    print(f"✅ Submission saved → {submission_out} ({len(submission)} rows)")
    print(f"✅ Diagnostics saved → {diagnostics_out}")
    print(f"✅ Cross-section diagnostics saved → {cross_diag_out}")
    print()
    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
