"""
Pure Cross-Section IV Imputer:
Quadratic + Kernel for normal points, PROGRESSIVE edge-block quadratic for edge points

This fixes the edge-case handling.

The important change:
    If a missing value is on the left/right edge of the same timestamp smile,
    we DO NOT pool nearest timestamps anymore.

    Instead, for that missing value:
        1. Stay at the SAME timestamp.
        2. Stay within the SAME option type: CE or PE.
        3. Find the 3 nearest non-missing observed strikes/moneyness values.
        4. Fit a quadratic through those 3 local observed points.
        5. Evaluate that quadratic at the missing value's moneyness.

This is exactly for cases like:
    leftmost PE missing:
        use the 3 nearest observed PE values to the right
        fit local quadratic
        extrapolate/interpolate to the missing left edge

For non-edge missing values:
    keep the existing method:
        same-row quadratic + kernel smoothing,
        bandwidth/blend selected by leave-one-out.

No regime split.
No time model.
Same method applies to every date, including 27 Jan.

Run:
    python pure_cross_section_quad_kernel_edge_fixed.py --data dataset.csv

For CV:
    python pure_cross_section_quad_kernel_edge_fixed.py --data cv_split/not_dataset.csv

Outputs:
    filled_dataset_pure_cross_section_quad_kernel_edge_fixed.csv
    submission_pure_cross_section_quad_kernel_edge_fixed.csv
    diagnostics_pure_cross_section_quad_kernel_edge_fixed.csv
    cross_section_diagnostics_pure_cross_section_quad_kernel_edge_fixed.csv
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

DEFAULT_DATA_PATH = Path("dataset.csv")
EPS_IV = 1e-6
SEPARATOR = "||"

# Edge fix:
# use the nearest 3 non-missing same-row, same-option-type points.
EDGE_LOCAL_NEIGHBORS = 3
# If there are B consecutive missing values on an edge, the first missing value
# uses B nearest observed points, the second uses B+1 points, etc.
# Keep at least 3 because a quadratic needs 3 points.
MIN_EDGE_LOCAL_NEIGHBORS = 3

BANDWIDTH_GRID_BY_TYPE = {
    "CE": np.array([
        1e-7, 2e-7, 5e-7,
        1e-6, 2e-6, 5e-6,
        7.5e-6, 1e-5, 1.25e-5, 1.5e-5,
        2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5,
        7.5e-5, 1e-4, 2e-4, 5e-4,
    ], dtype=float),
    "PE": np.array([
        5e-7, 1e-6, 2e-6, 5e-6,
        7.5e-6, 1e-5, 1.25e-5, 1.5e-5,
        2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5,
        7.5e-5, 1e-4, 2e-4, 5e-4,
    ], dtype=float),
}

BLEND_GRID_BY_TYPE = {
    "CE": np.array([
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00,
    ], dtype=float),
    "PE": np.array([
        0.40, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.725, 0.75, 0.775,
        0.80, 0.825, 0.85, 0.875,
        0.90, 0.925, 0.95, 0.975, 1.00,
    ], dtype=float),
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure cross-section IV imputer with progressive edge-block quadratic handling."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input CSV.")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="pure_cross_section_quad_kernel_edge_progressive",
        help="Output filename prefix.",
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
# Quadratic fit
# ---------------------------------------------------------------------

def fit_quadratic(x, y):
    """
    Fit IV = a*m^2 + b*m + c.

    Fallbacks:
        0 points -> no fit
        1 point  -> constant
        2 points -> linear
        3+       -> quadratic least-squares
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
    """Evaluate fitted quadratic/linear/constant curve safely."""
    if coeff is None:
        return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


def loo_quadratic_preds(x, y):
    """Leave-one-out predictions for the quadratic component."""
    preds = np.full(len(y), np.nan)

    for i in range(len(y)):
        coeff, _ = fit_quadratic(np.delete(x, i), np.delete(y, i))
        preds[i] = eval_quadratic(coeff, x[i])

    return preds


# ---------------------------------------------------------------------
# Kernel smoother
# ---------------------------------------------------------------------

def kernel_predict_many(x_obs, y_obs, x_targets, bandwidth):
    """
    Gaussian kernel smoother in ratio-moneyness space.

    weights:
        w_j = exp(-(m_j - m_target)^2 / (2 * bandwidth))
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_targets = np.asarray(x_targets, dtype=float)

    if len(y_obs) == 0:
        return np.full(len(x_targets), np.nan)

    if len(y_obs) == 1:
        return np.full(len(x_targets), safe_iv(y_obs[0]))

    dist2 = (x_targets[:, None] - x_obs[None, :]) ** 2
    weights = np.exp(-dist2 / (2.0 * bandwidth))
    weight_sums = weights.sum(axis=1)

    preds = np.empty(len(x_targets), dtype=float)
    good = weight_sums > 1e-15

    preds[good] = (weights[good] @ y_obs) / weight_sums[good]

    if (~good).any():
        nearest = np.argmin(dist2[~good], axis=1)
        preds[~good] = y_obs[nearest]

    return np.maximum(preds, EPS_IV)


def loo_kernel_preds(x, y, bandwidth):
    """Leave-one-out predictions for the kernel smoother."""
    n = len(y)

    if n <= 1:
        return np.full(n, np.nan)

    dist2 = (x[:, None] - x[None, :]) ** 2
    weights = np.exp(-dist2 / (2.0 * bandwidth))
    np.fill_diagonal(weights, 0.0)

    weight_sums = weights.sum(axis=1)
    preds = np.empty(n, dtype=float)

    good = weight_sums > 1e-15
    preds[good] = (weights[good] @ y) / weight_sums[good]

    if (~good).any():
        dist2_no_self = dist2.copy()
        np.fill_diagonal(dist2_no_self, np.inf)
        nearest = np.argmin(dist2_no_self[~good], axis=1)
        preds[~good] = y[nearest]

    return np.maximum(preds, EPS_IV)


def tune_cross_section_blend(x, y, option_type):
    """
    Tune:
        pred = blend * quadratic_pred + (1 - blend) * kernel_pred

    by leave-one-out MSE on the training set.
    """
    if len(y) <= 1:
        return BANDWIDTH_GRID_BY_TYPE[option_type][0], 1.0, "constant", np.inf

    q_loo = loo_quadratic_preds(x, y)
    valid_q = np.isfinite(q_loo)

    best_mse = float(np.mean((q_loo[valid_q] - y[valid_q]) ** 2)) if valid_q.any() else np.inf
    best_bandwidth = float(BANDWIDTH_GRID_BY_TYPE[option_type][len(BANDWIDTH_GRID_BY_TYPE[option_type]) // 2])
    best_blend = 1.0
    best_model = "pure_quadratic"

    for bandwidth in BANDWIDTH_GRID_BY_TYPE[option_type]:
        k_loo = loo_kernel_preds(x, y, bandwidth)

        for blend in BLEND_GRID_BY_TYPE[option_type]:
            pred = blend * q_loo + (1.0 - blend) * k_loo
            mask = np.isfinite(pred)

            if not mask.any():
                continue

            mse = float(np.mean((pred[mask] - y[mask]) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_bandwidth = float(bandwidth)
                best_blend = float(blend)

                if blend == 0.0:
                    best_model = "pure_kernel"
                elif blend == 1.0:
                    best_model = "pure_quadratic"
                else:
                    best_model = "quadratic_kernel_blend"

    return best_bandwidth, best_blend, best_model, best_mse


# ---------------------------------------------------------------------
# Training point collection
# ---------------------------------------------------------------------

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    """
    Collect same-option-type observed points from one timestamp.

    x = strike / underlying_price
    y = observed IV
    """
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []

    obs_cols = [
        col for col in cols_by_type[opt_type]
        if pd.notna(row[col])
    ]

    x = np.array([strike_map[col] / spot for col in obs_cols], dtype=float)
    y = np.array([row[col] for col in obs_cols], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    kept_cols = [col for col, keep in zip(obs_cols, mask) if keep]

    return x[mask], y[mask], kept_cols


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
    """
    Check whether the missing value is an edge within CE or PE only.

    Edge means:
        - no observed same-type strike to the left, or
        - no observed same-type strike to the right.
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



def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    """
    Return same-row, same-option-type columns split into observed/missing,
    ordered by strike.

    This is the core object used to detect edge missing blocks.
    """
    records = []

    for col in cols_by_type[opt_type]:
        records.append({
            "column": col,
            "strike": strike_map[col],
            "is_missing": pd.isna(row[col]),
            "iv": row[col],
        })

    state = pd.DataFrame(records).sort_values("strike").reset_index(drop=True)
    return state


def get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map):
    """
    Identify whether target_col belongs to a contiguous left-edge or right-edge
    missing block.

    Example right edge:
        observed observed observed missing missing missing missing
                                ^ first missing closest to observed

    For a block size B=4:
        first missing uses B observed points to the left
        second missing uses B+1 points: B observed + first predicted
        third missing uses B+2 points: B observed + first two predicted
        ...

    Returns:
        {
            side: "left" / "right" / "not_edge_block",
            block_cols: list of missing columns in fill order,
            block_size: B,
            position_in_block: 0-based position in fill order,
        }
    """
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    if target_col not in set(state["column"]):
        return {
            "side": "not_edge_block",
            "block_cols": [],
            "block_size": 0,
            "position_in_block": np.nan,
        }

    # Left edge missing block: consecutive missing values from lowest strike upward.
    left_block = []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]):
            left_block.append(rec["column"])
        else:
            break

    if target_col in left_block:
        # Fill from nearest observed boundary outward.
        # For left edge, the nearest observed boundary is on the right,
        # so fill highest-strike missing first, then move left.
        fill_order = list(reversed(left_block))
        return {
            "side": "left",
            "block_cols": fill_order,
            "block_size": len(left_block),
            "position_in_block": fill_order.index(target_col),
        }

    # Right edge missing block: consecutive missing values from highest strike downward.
    right_block = []
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]):
            right_block.append(rec["column"])
        else:
            break

    if target_col in right_block:
        # For right edge, fill lowest-strike missing first, then move right.
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


def collect_progressive_edge_training_points(
    row,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    already_filled_row_values,
):
    """
    PROGRESSIVE EDGE HANDLING.

    Suppose the rightmost 4 values are missing.

    Fill order:
        missing_1, missing_2, missing_3, missing_4

    Training set:
        missing_1 uses 4 nearest observed points to the left
        missing_2 uses 5 points: those 4 observed + missing_1 prediction
        missing_3 uses 6 points: those 4 observed + missing_1 + missing_2
        missing_4 uses 7 points: those 4 observed + missing_1 + missing_2 + missing_3

    Symmetric logic for left edge:
        fill from the observed boundary outward.
    """
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {
            "side": "bad_spot",
            "block_size": 0,
            "position_in_block": np.nan,
            "base_observed_needed": 0,
        }

    block_info = get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map)
    side = block_info["side"]

    if side not in {"left", "right"}:
        return np.array([]), np.array([]), [], block_info

    block_cols = block_info["block_cols"]
    block_size = int(block_info["block_size"])
    pos = int(block_info["position_in_block"])

    # Number of actual observed points required for the first missing point.
    # If block size is 4, use 4 observed. But always at least 3 for quadratic.
    base_needed = max(MIN_EDGE_LOCAL_NEIGHBORS, block_size)

    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    observed_records = []
    for _, rec in state.iterrows():
        col = rec["column"]
        original_val = row[col]

        if pd.notna(original_val):
            observed_records.append({
                "column": col,
                "strike": strike_map[col],
                "moneyness": strike_map[col] / spot,
                "iv": float(original_val),
                "is_predicted": False,
            })

    obs = pd.DataFrame(observed_records)

    if obs.empty:
        return np.array([]), np.array([]), [], {
            **block_info,
            "base_observed_needed": base_needed,
        }

    target_strike = strike_map[target_col]

    if side == "right":
        # Need observed values to the left of the first/right-edge block.
        base_obs = obs[obs["strike"] < target_strike].sort_values("strike", ascending=False).head(base_needed)
        base_obs = base_obs.sort_values("strike")
    else:
        # Need observed values to the right of the left-edge block.
        base_obs = obs[obs["strike"] > target_strike].sort_values("strike", ascending=True).head(base_needed)
        base_obs = base_obs.sort_values("strike")

    train_records = base_obs.to_dict(orient="records")

    # Add previously predicted missing values in this same edge block.
    # These are the "then look at 5, then 6..." points.
    previous_missing_cols = block_cols[:pos]

    for prev_col in previous_missing_cols:
        if prev_col not in already_filled_row_values:
            # This should not happen if we fill in the correct order,
            # but keep it safe.
            continue

        prev_iv = already_filled_row_values[prev_col]

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
            "base_observed_needed": base_needed,
        }

    train = train.sort_values("strike").reset_index(drop=True)

    x = train["moneyness"].to_numpy(dtype=float)
    y = train["iv"].to_numpy(dtype=float)
    used_cols = [
        f"{row.column}{'*' if row.is_predicted else ''}"
        for row in train.itertuples(index=False)
    ]

    return x, y, used_cols, {
        **block_info,
        "base_observed_needed": base_needed,
    }


def predict_edge_progressive_quadratic(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    already_filled_row_values,
):
    """
    Predict an edge missing value using progressive same-row local quadratic.

    The training list grows as edge missing values are filled sequentially.
    Predicted values already filled in the current edge block are marked with *
    in diagnostics used_cols.
    """
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_bad_spot",
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan,
            "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": [],
            "edge_side": "bad_spot",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "edge_base_observed_needed": 0,
        }

    x_obs, y_obs, used_cols, block_info = collect_progressive_edge_training_points(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
        already_filled_row_values=already_filled_row_values,
    )

    if len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_no_neighbors",
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan,
            "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": [],
            "edge_side": block_info.get("side", ""),
            "edge_block_size": block_info.get("block_size", 0),
            "edge_position_in_block": block_info.get("position_in_block", np.nan),
            "edge_base_observed_needed": block_info.get("base_observed_needed", 0),
        }

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    x_target = strike_map[target_col] / spot
    pred = eval_quadratic(coeff, x_target)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_progressive_local_quadratic"

    # Diagnostic in-sample fit error.
    if coeff is not None and len(y_obs) > 0:
        fitted = np.array([eval_quadratic(coeff, x) for x in x_obs], dtype=float)
        mask = np.isfinite(fitted) & np.isfinite(y_obs)
        fit_mse = float(np.mean((fitted[mask] - y_obs[mask]) ** 2)) if mask.any() else np.nan
    else:
        fit_mse = np.nan

    return {
        "prediction": safe_iv(pred),
        "source": "edge_progressive_same_row_quadratic",
        "selected_model": selected_model,
        "quadratic_fit_kind": fit_kind,
        "bandwidth": np.nan,
        "blend_quadratic_weight": 1.0,
        "loo_mse": fit_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        "edge_side": block_info.get("side", ""),
        "edge_block_size": block_info.get("block_size", 0),
        "edge_position_in_block": block_info.get("position_in_block", np.nan),
        "edge_base_observed_needed": block_info.get("base_observed_needed", 0),
    }


# ---------------------------------------------------------------------
# Single-cell prediction
# ---------------------------------------------------------------------

def predict_edge_local_quadratic(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
):
    """
    Predict an edge missing value using progressive same-row local quadratic.

    This function deliberately does NOT use kernel smoothing and does NOT pool
    timestamps. It follows the exact edge rule:
        nearest 3 observed points -> quadratic -> missing value.
    """
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_bad_spot",
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan,
            "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": [],
        }

    x_obs, y_obs, used_cols = collect_progressive_edge_training_points(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
    )

    if len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "edge_fallback_global_median_no_neighbors",
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan,
            "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan,
            "loo_mse": np.nan,
            "n_train": 0,
            "used_cols": [],
        }

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    x_target = strike_map[target_col] / spot
    pred = eval_quadratic(coeff, x_target)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_local_quadratic"

    # For diagnostics only: in-sample fit error on the 3 local points.
    if coeff is not None and len(y_obs) > 0:
        fitted = np.array([eval_quadratic(coeff, x) for x in x_obs], dtype=float)
        mask = np.isfinite(fitted) & np.isfinite(y_obs)
        loo_mse = float(np.mean((fitted[mask] - y_obs[mask]) ** 2)) if mask.any() else np.nan
    else:
        loo_mse = np.nan

    return {
        "prediction": safe_iv(pred),
        "source": "edge_progressive_same_row_quadratic",
        "selected_model": selected_model,
        "quadratic_fit_kind": fit_kind,
        "bandwidth": np.nan,
        "blend_quadratic_weight": 1.0,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
    }


def predict_non_edge_cross_section(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
):
    """
    Predict a non-edge missing IV using the original quadratic/kernel method.
    """
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    x_obs, y_obs, used_cols = collect_same_row_points(
        row=row,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
    )

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "same_row_fallback_global_median",
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan,
            "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan,
            "loo_mse": np.nan,
            "n_train": len(y_obs),
            "used_cols": used_cols,
        }

    x_target = np.array([strike_map[target_col] / spot], dtype=float)

    bandwidth, blend, selected_model, loo_mse = tune_cross_section_blend(
        x=x_obs,
        y=y_obs,
        option_type=opt_type,
    )

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    pred_quad = eval_quadratic(coeff, x_target[0])
    pred_kernel = kernel_predict_many(x_obs, y_obs, x_target, bandwidth)[0]

    pred = blend * pred_quad + (1.0 - blend) * pred_kernel

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"

    return {
        "prediction": safe_iv(pred),
        "source": "same_row_non_edge_quad_kernel",
        "selected_model": selected_model,
        "quadratic_fit_kind": fit_kind,
        "bandwidth": bandwidth,
        "blend_quadratic_weight": blend,
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
    already_filled_row_values,
):
    """
    Predict one missing IV.

    Edge:
        progressive same-row local quadratic

    Non-edge:
        same-row quadratic/kernel smoothing
    """
    row = df.loc[row_idx]

    edge, edge_reason = is_edge_missing(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
    )

    if edge:
        info = predict_edge_progressive_quadratic(
            df=df,
            row_idx=row_idx,
            target_col=target_col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
            already_filled_row_values=already_filled_row_values,
        )
    else:
        info = predict_non_edge_cross_section(
            df=df,
            row_idx=row_idx,
            target_col=target_col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
        )

    info["edge"] = edge
    info["edge_reason"] = edge_reason
    return info


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
        "edge_progressive_same_row_quadratic": 0,
        "same_row_non_edge_quad_kernel": 0,
        "edge_progressive_local_quadratic_selected": 0,
        "fallback_global_median": 0,
        "edge_no_left_observed": 0,
        "edge_no_right_observed": 0,
        "edge_no_observed_same_side": 0,
        "not_edge": 0,
    }

    rows = []

    # Important:
    # We fill row-by-row and option-type-by-option-type so progressive edge
    # predictions can be reused inside the same missing edge block.
    missing_cells = []
    for row_idx in df.index:
        row = df.loc[row_idx]

        for opt_type in ["CE", "PE"]:
            side_cols = cols_by_type[opt_type]
            missing_side_cols = [c for c in side_cols if pd.isna(row[c])]

            if not missing_side_cols:
                continue

            # Left edge block fill order: closest to observed boundary first.
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

            ordered = []
            ordered.extend(left_fill_order)

            # Interior missing/non-edge values next in strike order.
            edge_set = set(left_fill_order) | set(right_fill_order)
            interior = [
                c for c in state["column"].tolist()
                if c in missing_side_cols and c not in edge_set
            ]
            ordered.extend(interior)

            ordered.extend([c for c in right_fill_order if c not in ordered])

            for col in ordered:
                missing_cells.append((row_idx, col))

    # Stores predictions already made for the current row so that the 2nd/3rd/4th
    # missing edge value can use the 1st/2nd/3rd predicted edge value.
    filled_values_by_row = {}

    for row_idx, col in tqdm(missing_cells, desc="Progressive-edge pure cross-section filling"):
        opt_type = type_map[col]
        already_filled_row_values = filled_values_by_row.setdefault(row_idx, {})

        info = predict_cell(
            df=df,
            row_idx=row_idx,
            target_col=col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
            already_filled_row_values=already_filled_row_values,
        )

        pred = info["prediction"]

        if not np.isfinite(pred):
            pred = global_median_iv
            diagnostics["fallback_global_median"] += 1

        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred

        # Make this prediction available to the next missing value in the same
        # edge block, e.g. 4 observed -> predict first, then 5 points -> second.
        filled_values_by_row.setdefault(row_idx, {})[col] = pred

        diagnostics["filled"] += 1

        if info["source"] in diagnostics:
            diagnostics[info["source"]] += 1

        if info["edge_reason"] in diagnostics:
            diagnostics[info["edge_reason"]] += 1

        if info["selected_model"] == "fallback_global_median":
            diagnostics["fallback_global_median"] += 1
        elif info["selected_model"] == "edge_progressive_local_quadratic":
            diagnostics["edge_progressive_local_quadratic_selected"] += 1

        row_record = {
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
            "quadratic_fit_kind": info["quadratic_fit_kind"],
            "bandwidth": info["bandwidth"],
            "blend_quadratic_weight": info["blend_quadratic_weight"],
            "loo_mse": info["loo_mse"],
            "n_train": info["n_train"],
            "used_cols": "|".join(map(str, info["used_cols"])),
            "edge_side": info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
            "edge_base_observed_needed": info.get("edge_base_observed_needed", np.nan),
        }

        rows.append(row_record)

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
