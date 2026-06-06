"""
try_progressive_linear_edge_v3_bias_eda.py
================================

Based on try.py, but with a deliberately simpler edge rule.

Core rule
---------
Non-edge missing values:
    Same as try.py:
        same-row, same-option-type local quadratic WLS
        bandwidth selected by row-wise leave-one-out
        BANDWIDTH_GRID = [5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4]

Edge missing values:
    Progressive linear regression only.

    If the edge block is:
        observed observed observed missing_1 missing_2 missing_3 ...

    Fill from the observed boundary outward:
        missing_1: use the 2 nearest non-missing values and fit a line
        missing_2: use previous filled value + 3 nearest known-side values and fit a line
        missing_3: use previous filled values + 4 nearest known-side values and fit a line
        ...

    Same symmetric logic for left-edge blocks.

Important
---------
No global bias correction.
Adds a local timestamp-specific edge wrongness correction only for CE-right and PE-left.
No quadratic edge model.
No edge ensemble.
No cross-option-type transfer.

Validated edge-only upgrade over the previous simple version:
    - CE right edge and PE left edge use one extra known-side point.
    - All edge linear extrapolations use a small side-specific slope shrink.

The validation was done by hiding observed edge values and comparing only the
hidden cells. Non-edge logic remains exactly the same as try.py.

Run
---
    python try_progressive_linear_edge_v3_bias_eda.py --data dataset.csv

Outputs
-------
    filled_dataset_try_progressive_linear_edge.csv
    submission_try_progressive_linear_edge.csv
    diagnostics_try_progressive_linear_edge.csv
    cross_section_diagnostics_try_progressive_linear_edge.csv
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

DEFAULT_DATA_PATH = Path(
    "dataset.csv"
)

EPS_IV = 1e-6
SEPARATOR = "||"

# Same non-edge method as try.py.
BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
LOCAL_POLY_DEGREE = 2

# Edge-only validated tweaks.
# CE right edge and PE left edge behaved better with 3 nearest known-side
# points for the first missing value, then progressively more points.
EDGE_EXTRA_POINT_SIDES = {("CE", "right"), ("PE", "left")}

# Slope shrink: pred = anchor_y + lambda * (linear_pred - anchor_y).
# Tuned by masking observed edge values. No additive bias term.
EDGE_SLOPE_SHRINK = {
    ("CE", "left"): 0.83,
    ("CE", "right"): 0.92,
    ("PE", "left"): 0.88,
    ("PE", "right"): 0.87,
}

# EDA finding from edge holdout validation:
# The largest systematic misses are the OTM wings: CE right edge and PE left edge.
# A simple line fitted from the nearest known-side points tends to extrapolate in
# the wrong direction when the local edge curvature changes. We estimate that
# line's timestamp-specific wrongness from the immediately adjacent observed
# boundary point, without looking at the missing target. Validation on observed
# edge holdouts preferred a negative correction around -0.70 for only these two
# wing sides. Other sides got worse, so they are deliberately left unchanged.
EDGE_BIAS_SIDES = {("CE", "right"), ("PE", "left")}
EDGE_BIAS_GAMMA = -0.70
EDGE_BIAS_DECAY_PER_STEP = 0.25


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IV imputer: try.py non-edge logic + simple progressive linear edge logic."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input dataset CSV.")
    parser.add_argument("--out-prefix", type=str, default="try_progressive_linear_edge", help="Output filename prefix.")
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
    """Return a previous progressive prediction, supporting old scalar entries."""
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


def fit_linear(x, y):
    """Linear fit with constant fallback."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return None, "no_points"

    if len(y) == 1:
        return np.array([0.0, float(y[0])]), "constant"

    coeff = np.polyfit(x, y, 1)
    return np.array([float(coeff[0]), float(coeff[1])]), "linear"


def eval_linear(coeff, x):
    if coeff is None:
        return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


def estimate_edge_line_wrongness(row, opt_type, side, cols_by_type, strike_map):
    """
    Estimate timestamp-specific edge line wrongness from known values only.

    For a right-edge extrapolation, pretend the last observed same-side point is
    unknown and predict it using the two observed points immediately behind it.
    The residual

        true_boundary_iv - predicted_boundary_iv

    tells us whether the local line is systematically under/over-shooting at
    this timestamp and side. For a left edge, use the symmetric first-observed
    point and the next two observed points.

    This never uses the actual missing target value.
    """
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return 0.0, "no_bias_bad_spot"

    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    if len(obs_cols) < 4:
        return 0.0, "no_bias_too_few_obs"

    obs_cols = sorted(obs_cols, key=lambda c: strike_map[c])

    if side == "right":
        boundary_col = obs_cols[-1]
        train_cols = obs_cols[-3:-1]
    elif side == "left":
        boundary_col = obs_cols[0]
        train_cols = obs_cols[1:3]
    else:
        return 0.0, "no_bias_unknown_side"

    if len(train_cols) < 2:
        return 0.0, "no_bias_too_few_train"

    x_train = np.array([strike_map[c] / spot for c in train_cols], dtype=float)
    y_train = np.array([row[c] for c in train_cols], dtype=float)
    x_boundary = strike_map[boundary_col] / spot
    y_boundary = float(row[boundary_col])

    coeff, _ = fit_linear(x_train, y_train)
    boundary_pred = eval_linear(coeff, x_boundary)

    if not np.isfinite(boundary_pred):
        return 0.0, "no_bias_bad_boundary_pred"

    bias = float(y_boundary - boundary_pred)
    if not np.isfinite(bias):
        return 0.0, "no_bias_nonfinite"

    return bias, "inner_boundary_residual_bias"


# ---------------------------------------------------------------------
# Local polynomial WLS for non-edge cells only
# ---------------------------------------------------------------------

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    """
    Local polynomial regression at one target point.

    Same as try.py for non-edge cells.
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

    This is intentionally unchanged from try.py.
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


def collect_edge_training_points_linear(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    """
    Simple progressive linear edge model.

    Rule:
        First edge missing value:
            use the 2 nearest non-missing values from the observed side.

        Later edge missing values:
            include previous edge predictions and go farther back into known points.

    Always fit a linear line. No bias correction.
    """
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), [], {
            "edge_side": "bad_spot",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "linear_points_used": 0,
        }

    edge, edge_reason, side, block_cols, position = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map
    )

    if not edge:
        return np.array([]), np.array([]), [], {
            "edge_side": "not_edge",
            "edge_block_size": 0,
            "edge_position_in_block": np.nan,
            "linear_points_used": 0,
        }

    target_strike = strike_map[target_col]
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    observed_records = []

    for _, rec in state.iterrows():
        col = rec["column"]
        val = row[col]

        if pd.isna(val):
            continue

        strike = strike_map[col]

        if side == "left" and strike > target_strike:
            observed_records.append({
                "column": col,
                "strike": strike,
                "x": strike / spot,
                "y": float(val),
                "is_predicted": False,
            })

        elif side == "right" and strike < target_strike:
            observed_records.append({
                "column": col,
                "strike": strike,
                "x": strike / spot,
                "y": float(val),
                "is_predicted": False,
            })

    if not observed_records:
        return np.array([]), np.array([]), [], {
            "edge_side": side,
            "edge_block_size": len(block_cols),
            "edge_position_in_block": position,
            "linear_points_used": 0,
        }

    obs = pd.DataFrame(observed_records)
    pos_int = int(position) if np.isfinite(position) else 0

    # Default: first missing value uses 2 nearest known-side points.
    # Validated tweak: CE right and PE left are noisier wing/center-transition
    # edges, and using 3 nearest known-side points reduced edge holdout MSE.
    base_start = 3 if (opt_type, side) in EDGE_EXTRA_POINT_SIDES else 2
    base_needed = base_start + pos_int

    if side == "left":
        # Observed points are to the right. Nearest ones have smallest strike above target.
        base = (
            obs.sort_values("strike", ascending=True)
            .head(base_needed)
            .sort_values("strike")
        )

    elif side == "right":
        # Observed points are to the left. Nearest ones have largest strike below target.
        base = (
            obs.sort_values("strike", ascending=False)
            .head(base_needed)
            .sort_values("strike")
        )

    else:
        base = obs.sort_values("strike").head(base_needed)

    train_records = base.to_dict(orient="records")

    # Include previous predictions in the same edge block.
    previous_cols = block_cols[:pos_int]

    for prev_col in previous_cols:
        prev_pred = component_value(already_filled, prev_col, "linear")

        if not np.isfinite(prev_pred):
            prev_pred = component_value(already_filled, prev_col, "final")

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
            "linear_points_used": 0,
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
            "linear_points_used": len(train),
        },
    )


def predict_edge_linear(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    """
    Edge prediction:
        progressive linear regression only.
    """
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    x_obs, y_obs, used_cols, edge_info = collect_edge_training_points_linear(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
        already_filled=already_filled,
    )

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        pred = global_median_iv
        fit_kind = "fallback_global_median"
    else:
        x_target = strike_map[target_col] / spot
        coeff, fit_kind = fit_linear(x_obs, y_obs)
        raw_pred = eval_linear(coeff, x_target)

        if np.isfinite(raw_pred) and len(y_obs) > 0:
            # Slope shrink toward the nearest training point.
            # This keeps the simple linear edge idea, but avoids over-extrapolating
            # when the edge smile is locally curved.
            nearest = int(np.argmin(np.abs(np.asarray(x_obs, dtype=float) - x_target)))
            anchor_y = float(np.asarray(y_obs, dtype=float)[nearest])
            lam = EDGE_SLOPE_SHRINK.get((opt_type, edge_info.get("edge_side", "")), 1.0)
            pred = anchor_y + lam * (raw_pred - anchor_y)

            # Timestamp-specific systematic error correction.
            # Apply only where EDA found consistent gains: CE right and PE left.
            edge_side = edge_info.get("edge_side", "")
            if (opt_type, edge_side) in EDGE_BIAS_SIDES:
                wrongness, bias_source = estimate_edge_line_wrongness(
                    row=row,
                    opt_type=opt_type,
                    side=edge_side,
                    cols_by_type=cols_by_type,
                    strike_map=strike_map,
                )
                pos = edge_info.get("edge_position_in_block", 0)
                pos = int(pos) if np.isfinite(pos) else 0
                decay = 1.0 / (1.0 + EDGE_BIAS_DECAY_PER_STEP * pos)
                pred = pred + EDGE_BIAS_GAMMA * decay * wrongness
                edge_info["edge_bias_wrongness"] = wrongness
                edge_info["edge_bias_source"] = bias_source
                edge_info["edge_bias_gamma_used"] = EDGE_BIAS_GAMMA * decay
            else:
                edge_info["edge_bias_wrongness"] = 0.0
                edge_info["edge_bias_source"] = "not_bias_side"
                edge_info["edge_bias_gamma_used"] = 0.0
        else:
            pred = raw_pred

        if not np.isfinite(pred):
            pred = global_median_iv
            fit_kind = "fallback_global_median"

    pred = safe_iv(pred)

    return {
        "prediction": pred,
        "source": "edge_progressive_linear",
        "selected_model": fit_kind,
        "bandwidth": np.nan,
        "loo_mse": np.nan,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        "component_predictions": {
            "final": pred,
            "linear": pred,
            "claude": pred,
            "corrected": pred,
            "quadratic": pred,
        },
        "corrected_used_cols": [],
        "quadratic_used_cols": [],
        "quadratic_fit_kind": fit_kind,
        "edge_observed_side_points": len(y_obs),
        "edge_base_observed_needed": np.nan,
        **edge_info,
    }


def predict_cell(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    """Route one missing cell to edge or non-edge model."""
    row = df.loc[row_idx]

    edge, edge_reason, _, _, _ = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)

    if edge:
        info = predict_edge_linear(
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
        "edge_progressive_linear": 0,
        "same_row_non_edge_local_poly_wls": 0,
        "same_row_fallback_global_median": 0,
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

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs: try.py non-edge + linear edge"):
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
            "linear": components.get("linear", pred),
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
            "edge_linear_prediction": info.get("component_predictions", {}).get("linear", np.nan),
            "edge_side": info.get("edge_side", ""),
            "edge_block_size": info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
            "linear_points_used": info.get("linear_points_used", np.nan),
            "edge_slope_shrink": EDGE_SLOPE_SHRINK.get((opt_type, info.get("edge_side", "")), np.nan) if info["edge"] else np.nan,
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
