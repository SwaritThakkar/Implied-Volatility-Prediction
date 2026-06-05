"""
Pure Cross-Section IV Imputer — Local Polynomial WLS + Smoothed Progressive Edge Edition
=======================================================================================

This is the continuation of the previous best script:

    Pure Cross-Section IV Imputer — Local Polynomial WLS Edition

What stays unchanged
--------------------
1. Pure cross-section only.
2. Same row only.
3. Same option type only: CE and PE are fitted separately.
4. Non-edge missing cells still use local quadratic WLS with per-row LOO
   bandwidth selection.
5. No time model, no regime split, no future-row information.

What changes
------------
Only the edge-block model changes.

Previous edge method:
    progressive global quadratic through a small nearest-neighbour set.

New edge method:
    progressive local quadratic WLS with a wider fixed edge bandwidth.

For an edge missing block, the fill order is still boundary-outward:

    right edge example:
        observed observed observed missing_1 missing_2 missing_3

        missing_1 is filled first, then missing_2, then missing_3.

    left edge example:
        missing_3 missing_2 missing_1 observed observed observed

        missing_1 is filled first, then missing_2, then missing_3.

For each edge prediction, the training set uses:
    - all originally observed same-row, same-option-type points on the valid side
      of the edge block, and
    - previously predicted values from the same edge block.

The edge prediction itself is local quadratic WLS:

    min_beta sum_i w_i * (y_i - beta_0 - beta_1 dx_i - beta_2 dx_i^2)^2

where:
    dx_i = x_i - x_target
    x = strike / underlying_price
    w_i = exp(-(x_i - x_target)^2 / (2 * EDGE_BANDWIDTH))

The final prediction is beta_0.

Why the edge change helps
-------------------------
The benchmark that motivated this file showed that progressive local-polynomial
edge filling with EDGE_BANDWIDTH = 2e-4 had much lower edge-block error than the
old progressive quadratic edge fill when edge blocks were simulated from observed
smiles. The important detail is that the corrected benchmark must append the
missing point's x-coordinate after prediction, not the predicted IV as an
x-coordinate.

Validated corrected benchmark on dataset.csv, masking left/right edge blocks of
size 1, 2, and 3:

    all-observed-side progressive global quadratic:
        RMSE ≈ 0.024887, MSE ≈ 0.00061939

    all-observed-side progressive local poly WLS, bw = 2e-4:
        RMSE ≈ 0.020582, MSE ≈ 0.00042361

    improvement in edge-block benchmark:
        about 31.6% MSE reduction

Note:
    The 89.32% number from the first benchmark was directionally useful, but that
    particular quick script had a bug in the progressive update: it appended the
    predicted IV to the x-list instead of appending the target x-coordinate.
    This file uses the corrected progressive update.

Run
---
    python final_locpoly_wls_edge_lp2e4.py --data dataset.csv

For CV:
    python final_locpoly_wls_edge_lp2e4.py --data cv_split/not_dataset.csv --out-prefix cv_final_locpoly_edge_lp2e4

Outputs
-------
    filled_dataset_<out-prefix>.csv
    submission_<out-prefix>.csv
    diagnostics_<out-prefix>.csv
    cross_section_diagnostics_<out-prefix>.csv
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

# Non-edge local polynomial bandwidths from the previous best script.
BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)

# Edge-specific bandwidth.
# This is intentionally wider than the best non-edge bandwidth because edge
# prediction is one-sided extrapolation. Wider smoothing stabilizes the boundary.
EDGE_BANDWIDTH = 2e-4

LOCAL_POLY_DEGREE = 2


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure cross-section IV imputer: local-poly WLS with smoothed progressive edge handling."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input CSV file.")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="final_locpoly_edge_lp2e4",
        help="Prefix for output files.",
    )
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
            item["expiry_date"] = pd.to_datetime(
                item["expiry"],
                format="%d%b%y",
                errors="coerce",
            )
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


def make_submission(original: pd.DataFrame, filled: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Create Kaggle-style submission from cells missing in the input file."""
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


# ---------------------------------------------------------------------
# Polynomial and local polynomial WLS
# ---------------------------------------------------------------------

def fit_quadratic(x, y):
    """
    Global quadratic fit with fallbacks.

    Used only as a diagnostic fallback and not as the new edge estimator.
    """
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


def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    """
    Local polynomial WLS prediction at one target point.

    Fits around x_target:
        y_i ≈ beta_0 + beta_1 (x_i - x_target) + beta_2 (x_i - x_target)^2

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

    # If there are too few points for degree 2, lower the degree gracefully.
    degree = min(int(degree), len(y_obs) - 1)

    dx = x_obs - x_target
    dist2 = dx ** 2
    weights = np.exp(-dist2 / (2.0 * bandwidth))

    X = np.column_stack([dx ** j for j in range(degree + 1)])

    # Avoid building a dense diagonal W matrix; this is X.T @ W @ X.
    xtwx = X.T @ (weights[:, None] * X)
    xtwy = X.T @ (weights * y_obs)

    try:
        beta = np.linalg.solve(xtwx, xtwy)
        pred = float(beta[0])
    except np.linalg.LinAlgError:
        weight_sum = weights.sum()
        if weight_sum <= 1e-15:
            return np.nan
        pred = float((weights @ y_obs) / weight_sum)

    return safe_iv(pred)


def local_poly_wls_loo(x_obs, y_obs, bandwidth, degree=LOCAL_POLY_DEGREE):
    """LOO predictions for local polynomial WLS on all observed points."""
    n = len(y_obs)
    preds = np.full(n, np.nan)

    for i in range(n):
        xi = np.delete(x_obs, i)
        yi = np.delete(y_obs, i)
        preds[i] = local_poly_wls_pred(xi, yi, x_obs[i], bandwidth, degree)

    return preds


def select_bandwidth(x_obs, y_obs, bw_grid=BANDWIDTH_GRID):
    """Select non-edge bandwidth by same-row LOO MSE."""
    if len(y_obs) <= 2:
        return float(bw_grid[len(bw_grid) // 2]), np.inf

    best_bw = float(bw_grid[len(bw_grid) // 2])
    best_mse = np.inf

    for bw in bw_grid:
        loo = local_poly_wls_loo(x_obs, y_obs, bw)
        valid = np.isfinite(loo)

        if not valid.any():
            continue

        mse = float(np.mean((loo[valid] - y_obs[valid]) ** 2))

        if mse < best_mse:
            best_mse = mse
            best_bw = float(bw)

    return best_bw, best_mse


# ---------------------------------------------------------------------
# Same-row point collection and edge-block detection
# ---------------------------------------------------------------------

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    """
    Collect same-row, same-option-type observed points.

    x = strike / underlying_price
    y = observed IV
    """
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []

    obs_cols = [col for col in cols_by_type[opt_type] if pd.notna(row[col])]

    x = np.array([strike_map[col] / spot for col in obs_cols], dtype=float)
    y = np.array([row[col] for col in obs_cols], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    kept_cols = [col for col, keep in zip(obs_cols, mask) if keep]

    return x[mask], y[mask], kept_cols


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    """Return same-row CE or PE state, ordered by strike."""
    records = []

    for col in cols_by_type[opt_type]:
        records.append({
            "column": col,
            "strike": strike_map[col],
            "is_missing": pd.isna(row[col]),
            "iv": row[col],
        })

    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
    """
    Edge means target has no observed same-type strike on one side.
    """
    target_strike = strike_map[target_col]

    observed_strikes = [
        strike_map[col]
        for col in cols_by_type[opt_type]
        if pd.notna(row[col])
    ]

    if not observed_strikes:
        return True, "edge_no_observed_same_side"

    has_left = any(k < target_strike for k in observed_strikes)
    has_right = any(k > target_strike for k in observed_strikes)

    if not has_left:
        return True, "edge_no_left_observed"
    if not has_right:
        return True, "edge_no_right_observed"

    return False, "not_edge"


def get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map):
    """
    Identify whether target_col belongs to a contiguous left/right edge block.

    Fill order is boundary-outward:
        left edge: highest-strike missing first, then move left.
        right edge: lowest-strike missing first, then move right.
    """
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    if target_col not in set(state["column"]):
        return {
            "side": "not_edge_block",
            "block_cols": [],
            "block_size": 0,
            "position_in_block": np.nan,
        }

    left_block = []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]):
            left_block.append(rec["column"])
        else:
            break

    if target_col in left_block:
        fill_order = list(reversed(left_block))
        return {
            "side": "left",
            "block_cols": fill_order,
            "block_size": len(left_block),
            "position_in_block": fill_order.index(target_col),
        }

    right_block = []
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]):
            right_block.append(rec["column"])
        else:
            break

    if target_col in right_block:
        fill_order = list(reversed(right_block))
        return {
            "side": "right",
            "block_cols": fill_order,
            "block_size": len(right_block),
            "position_in_block": fill_order.index(target_col),
        }

    return {
        "side": "not_edge_block",
        "block_cols": [],
        "block_size": 0,
        "position_in_block": np.nan,
    }


def collect_edge_training_points_all_observed_side(
    row,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    already_filled,
):
    """
    Collect training points for the new edge estimator.

    For a right-edge block, all originally observed same-side points are to the
    left of the block, so use all observed strikes below the target, plus earlier
    predictions in the same block.

    For a left-edge block, use all observed strikes above the target, plus earlier
    predictions in the same block.

    This differs from the old nearest-neighbour edge rule. The wide local-poly
    kernel handles distance weighting; keeping all valid-side points stabilizes
    the one-sided fit.
    """
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {
            "side": "bad_spot",
            "block_size": 0,
            "position_in_block": np.nan,
            "observed_side_points": 0,
        }

    block_info = get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map)
    side = block_info["side"]

    if side not in {"left", "right"}:
        return np.array([]), np.array([]), [], {
            **block_info,
            "observed_side_points": 0,
        }

    target_strike = strike_map[target_col]
    block_cols = block_info["block_cols"]
    pos = int(block_info["position_in_block"])

    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    train_records = []

    for _, rec in state.iterrows():
        col = rec["column"]
        val = row[col]

        if pd.isna(val):
            continue

        strike = strike_map[col]

        if side == "right" and strike < target_strike:
            train_records.append({
                "column": col,
                "strike": strike,
                "moneyness": strike / spot,
                "iv": float(val),
                "is_predicted": False,
            })
        elif side == "left" and strike > target_strike:
            train_records.append({
                "column": col,
                "strike": strike,
                "moneyness": strike / spot,
                "iv": float(val),
                "is_predicted": False,
            })

    observed_side_points = len(train_records)

    # Add previously predicted edge-block values.
    # This is the corrected progressive update:
    #   x coordinate = strike(prev_col) / spot
    #   y value      = predicted IV(prev_col)
    previous_missing_cols = block_cols[:pos]

    for prev_col in previous_missing_cols:
        if prev_col not in already_filled:
            continue

        prev_iv = already_filled[prev_col]

        if not np.isfinite(prev_iv):
            continue

        train_records.append({
            "column": prev_col,
            "strike": strike_map[prev_col],
            "moneyness": strike_map[prev_col] / spot,
            "iv": float(prev_iv),
            "is_predicted": True,
        })

    train = pd.DataFrame(train_records)

    if train.empty:
        return np.array([]), np.array([]), [], {
            **block_info,
            "observed_side_points": observed_side_points,
        }

    train = train.sort_values("strike").reset_index(drop=True)

    x = train["moneyness"].to_numpy(dtype=float)
    y = train["iv"].to_numpy(dtype=float)
    used_cols = [
        f"{r.column}{'*' if r.is_predicted else ''}"
        for r in train.itertuples(index=False)
    ]

    return x, y, used_cols, {
        **block_info,
        "observed_side_points": observed_side_points,
    }


# ---------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------

def predict_edge_local_poly_wls(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    already_filled,
):
    """Edge prediction using progressive local quadratic WLS with bw=2e-4."""
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    fallback = {
        "prediction": float(global_median_iv),
        "source": "edge_fallback_global_median",
        "selected_model": "fallback_global_median",
        "fit_kind": np.nan,
        "bandwidth": np.nan,
        "loo_mse": np.nan,
        "n_train": 0,
        "used_cols": [],
        "edge_side": "bad_spot",
        "edge_block_size": 0,
        "edge_position_in_block": np.nan,
        "edge_observed_side_points": 0,
    }

    if pd.isna(spot) or spot <= 0:
        return fallback

    x_obs, y_obs, used_cols, block_info = collect_edge_training_points_all_observed_side(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
        already_filled=already_filled,
    )

    if len(y_obs) == 0:
        return {
            **fallback,
            "source": "edge_fallback_global_median_no_neighbors",
            "edge_side": block_info.get("side", ""),
            "edge_block_size": block_info.get("block_size", 0),
            "edge_position_in_block": block_info.get("position_in_block", np.nan),
            "edge_observed_side_points": block_info.get("observed_side_points", 0),
        }

    x_target = strike_map[target_col] / spot
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, EDGE_BANDWIDTH, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_progressive_local_poly_wls"

    # Diagnostic in-sample WLS fit error. This is not a validation metric.
    fitted = np.array([
        local_poly_wls_pred(x_obs, y_obs, x, EDGE_BANDWIDTH, degree=LOCAL_POLY_DEGREE)
        for x in x_obs
    ], dtype=float)
    mask = np.isfinite(fitted) & np.isfinite(y_obs)
    fit_mse = float(np.mean((fitted[mask] - y_obs[mask]) ** 2)) if mask.any() else np.nan

    fit_kind = "quadratic" if len(y_obs) >= 3 else ("linear" if len(y_obs) == 2 else "constant")

    return {
        "prediction": safe_iv(pred),
        "source": "edge_progressive_local_poly_wls",
        "selected_model": selected_model,
        "fit_kind": fit_kind,
        "bandwidth": EDGE_BANDWIDTH,
        "loo_mse": fit_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        "edge_side": block_info.get("side", ""),
        "edge_block_size": block_info.get("block_size", np.nan),
        "edge_position_in_block": block_info.get("position_in_block", np.nan),
        "edge_observed_side_points": block_info.get("observed_side_points", np.nan),
    }


def predict_non_edge_local_poly(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    row_type_cache=None,
):
    """Non-edge prediction: previous best local quadratic WLS with LOO bandwidth."""
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    cache_key = (int(row_idx), opt_type)

    if row_type_cache is not None and cache_key in row_type_cache:
        cached = row_type_cache[cache_key]
        x_obs = cached["x_obs"]
        y_obs = cached["y_obs"]
        used_cols = cached["used_cols"]
        best_bw = cached["best_bw"]
        loo_mse = cached["loo_mse"]
    else:
        x_obs, y_obs, used_cols = collect_same_row_points(
            row=row,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
        )

        if pd.notna(spot) and spot > 0 and len(y_obs) > 0:
            best_bw, loo_mse = select_bandwidth(x_obs, y_obs, BANDWIDTH_GRID)
        else:
            best_bw, loo_mse = np.nan, np.nan

        if row_type_cache is not None:
            row_type_cache[cache_key] = {
                "x_obs": x_obs,
                "y_obs": y_obs,
                "used_cols": used_cols,
                "best_bw": best_bw,
                "loo_mse": loo_mse,
            }

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "same_row_fallback_global_median",
            "selected_model": "fallback_global_median",
            "fit_kind": np.nan,
            "bandwidth": np.nan,
            "loo_mse": np.nan,
            "n_train": len(y_obs),
            "used_cols": used_cols,
        }

    x_target = strike_map[target_col] / spot

    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "same_row_non_edge_local_poly_wls"

    fit_kind = "quadratic" if len(y_obs) >= 3 else ("linear" if len(y_obs) == 2 else "constant")

    return {
        "prediction": safe_iv(pred),
        "source": "same_row_non_edge_local_poly_wls",
        "selected_model": selected_model,
        "fit_kind": fit_kind,
        "bandwidth": best_bw,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
    }


def predict_cell(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    already_filled,
    row_type_cache=None,
):
    """Predict one missing IV cell."""
    row = df.loc[row_idx]

    edge, edge_reason = is_edge_missing(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
    )

    if edge:
        info = predict_edge_local_poly_wls(
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
            row_type_cache=row_type_cache,
        )

    info["edge"] = edge
    info["edge_reason"] = edge_reason
    return info


# ---------------------------------------------------------------------
# Fill-order construction
# ---------------------------------------------------------------------

def build_missing_fill_order(df, cols_by_type, strike_map):
    """
    Build row-wise fill order.

    Edge blocks must be filled boundary-outward so that later edge values can use
    earlier edge predictions from the same block.
    """
    missing_cells = []

    for row_idx in df.index:
        row = df.loc[row_idx]

        for opt_type in ["CE", "PE"]:
            side_cols = cols_by_type[opt_type]
            missing_side_cols = [col for col in side_cols if pd.isna(row[col])]

            if not missing_side_cols:
                continue

            state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

            left_block = []
            for _, rec in state.iterrows():
                if bool(rec["is_missing"]):
                    left_block.append(rec["column"])
                else:
                    break
            left_fill_order = list(reversed(left_block))

            right_block = []
            for _, rec in state.iloc[::-1].iterrows():
                if bool(rec["is_missing"]):
                    right_block.append(rec["column"])
                else:
                    break
            right_fill_order = list(reversed(right_block))

            edge_set = set(left_fill_order) | set(right_fill_order)
            interior = [
                col for col in state["column"].tolist()
                if col in missing_side_cols and col not in edge_set
            ]

            ordered = left_fill_order + interior + [
                col for col in right_fill_order if col not in left_fill_order
            ]

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
        "CE": [col for col in option_cols if type_map[col] == "CE"],
        "PE": [col for col in option_cols if type_map[col] == "PE"],
    }

    global_median_iv = float(df[option_cols].stack().median())
    filled = df.copy()

    diagnostics = {
        "missing_initial": int(df[option_cols].isna().sum().sum()),
        "filled": 0,
        "edge_progressive_local_poly_wls": 0,
        "same_row_non_edge_local_poly_wls": 0,
        "fallback_global_median": 0,
        "edge_no_left_observed": 0,
        "edge_no_right_observed": 0,
        "edge_no_observed_same_side": 0,
        "not_edge": 0,
    }

    rows = []
    missing_cells = build_missing_fill_order(df, cols_by_type, strike_map)

    # Already-filled predictions are stored by row so edge blocks can be filled
    # progressively inside the same timestamp.
    filled_values_by_row: dict[int, dict] = {}
    row_type_cache: dict[tuple[int, str], dict] = {}

    for row_idx, col in tqdm(missing_cells, desc="Final local-poly cross-section filling"):
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
            row_type_cache=row_type_cache,
        )

        pred = info["prediction"]

        if not np.isfinite(pred):
            pred = global_median_iv
            diagnostics["fallback_global_median"] += 1

        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred

        # Correct progressive update:
        # Store only the predicted IV by column. The x-coordinate will be rebuilt
        # later as strike_map[col] / spot.
        filled_values_by_row[row_idx][col] = pred

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
            "fit_kind": info["fit_kind"],
            "bandwidth": info["bandwidth"],
            "loo_mse": info["loo_mse"],
            "n_train": info["n_train"],
            "used_cols": "|".join(map(str, info["used_cols"])),
            "edge_side": info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
            "edge_observed_side_points": info.get("edge_observed_side_points", np.nan),
        })

    diagnostics["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled_out_df = filled.drop(columns=["datetime_parsed"])
    original_out_df = df.drop(columns=["datetime_parsed"])

    filled_out_df.to_csv(filled_out, index=False)
    submission = make_submission(original_out_df, filled_out_df, submission_out)

    pd.DataFrame(rows).to_csv(diagnostics_out, index=False)
    pd.DataFrame(rows).to_csv(cross_diag_out, index=False)

    print(f"✅ Filled dataset saved → {filled_out}")
    print(f"✅ Submission saved → {submission_out} ({len(submission)} rows)")
    print(f"✅ Diagnostics saved → {diagnostics_out}")
    print(f"✅ Cross-section diagnostics saved → {cross_diag_out}")

    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
