"""
Pure Cross-Section IV Imputer:
Quadratic + Kernel for normal points, LOCAL 3-point quadratic for edge points

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
        description="Pure cross-section IV imputer with fixed local 3-point edge handling."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input CSV.")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="pure_cross_section_quad_kernel_edge_fixed",
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


def collect_local_edge_neighbors(row, target_col, opt_type, cols_by_type, strike_map):
    """
    FIXED EDGE HANDLING.

    For an edge missing value:
        - stay on the same timestamp
        - stay in same option type
        - choose the 3 nearest non-missing observed points by moneyness distance
        - fit a local quadratic on those 3 points

    This is what prevents the bad behavior seen in the plots, where a left-edge
    missing value was being pulled by unrelated pooled/timestamp data.
    """
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []

    target_m = strike_map[target_col] / spot

    records = []
    for col in cols_by_type[opt_type]:
        val = row[col]
        if pd.isna(val):
            continue

        m = strike_map[col] / spot
        if not np.isfinite(m) or not np.isfinite(val):
            continue

        records.append({
            "column": col,
            "moneyness": float(m),
            "iv": float(val),
            "distance": abs(float(m) - float(target_m)),
            "strike": strike_map[col],
        })

    if not records:
        return np.array([]), np.array([]), []

    neighbors = (
        pd.DataFrame(records)
        .sort_values(["distance", "strike"])
        .head(EDGE_LOCAL_NEIGHBORS)
    )

    x = neighbors["moneyness"].to_numpy(dtype=float)
    y = neighbors["iv"].to_numpy(dtype=float)
    used_cols = neighbors["column"].tolist()

    return x, y, used_cols


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
    Predict an edge missing value using same-row nearest-3 local quadratic.

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

    x_obs, y_obs, used_cols = collect_local_edge_neighbors(
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
        "source": "edge_same_row_nearest3_local_quadratic",
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
):
    """
    Predict one missing IV.

    Edge:
        same-row nearest-3 local quadratic

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
        info = predict_edge_local_quadratic(
            df=df,
            row_idx=row_idx,
            target_col=target_col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
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
        "edge_same_row_nearest3_local_quadratic": 0,
        "same_row_non_edge_quad_kernel": 0,
        "fallback_global_median": 0,
        "edge_no_left_observed": 0,
        "edge_no_right_observed": 0,
        "edge_no_observed_same_side": 0,
        "not_edge": 0,
    }

    rows = []

    missing_cells = [
        (idx, col)
        for idx in df.index
        for col in option_cols
        if pd.isna(df.at[idx, col])
    ]

    for row_idx, col in tqdm(missing_cells, desc="Fixed-edge pure cross-section filling"):
        opt_type = type_map[col]

        info = predict_cell(
            df=df,
            row_idx=row_idx,
            target_col=col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
        )

        pred = info["prediction"]

        if not np.isfinite(pred):
            pred = global_median_iv
            diagnostics["fallback_global_median"] += 1

        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred
        diagnostics["filled"] += 1

        if info["source"] in diagnostics:
            diagnostics[info["source"]] += 1

        if info["edge_reason"] in diagnostics:
            diagnostics[info["edge_reason"]] += 1

        if info["selected_model"] == "fallback_global_median":
            diagnostics["fallback_global_median"] += 1

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
