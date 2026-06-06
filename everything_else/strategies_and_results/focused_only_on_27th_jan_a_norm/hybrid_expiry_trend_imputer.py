"""
Hybrid Expiry-Day IV Imputer

This script implements the model exactly as discussed.

Prediction formula:
    pred = a * pred_t + (1 - a) * pred_cross_section

Where:
    - Before 27 Jan:
          a = 0
          pred = pred_cross_section only

    - On 27 Jan:
          a is fixed per option contract for the whole day
          pred_t is a same-contract time model using only previous 27 Jan points
          pred_cross_section is the cross-sectional model at that timestamp

Inputs:
    dataset.csv
    or cv_split/not_dataset.csv if you are using the synthetic CV system

Outputs:
    filled_dataset_hybrid_expiry_trend.csv
    submission_hybrid_expiry_trend.csv
    diagnostics_hybrid_expiry_trend.csv
    a_weights_27jan.csv
    cross_section_diagnostics_hybrid_expiry_trend.csv
    time_model_diagnostics_hybrid_expiry_trend.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_PATH = Path("dataset.csv")

EXPIRY_DAY = pd.Timestamp("2026-01-27").date()

EPS_IV = 1e-6
SEPARATOR = "||"

EDGE_NEAREST_TIMESTAMPS = 3
MIN_TIME_POINTS_FOR_QUADRATIC = 3

BANDWIDTH_GRID_BY_TYPE = {
    "CE": np.array([
        1e-7, 2e-7, 5e-7,
        1e-6, 2e-6, 5e-6,
        7.5e-6, 1e-5, 1.25e-5, 1.5e-5,
        2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5,
        7.5e-5, 1e-4, 2e-4, 5e-4
    ], dtype=float),
    "PE": np.array([
        5e-7, 1e-6, 2e-6, 5e-6,
        7.5e-6, 1e-5, 1.25e-5, 1.5e-5,
        2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5,
        7.5e-5, 1e-4, 2e-4, 5e-4
    ], dtype=float),
}

BLEND_GRID_BY_TYPE = {
    "CE": np.array([
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00
    ], dtype=float),
    "PE": np.array([
        0.40, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.725, 0.75, 0.775,
        0.80, 0.825, 0.85, 0.875,
        0.90, 0.925, 0.95, 0.975, 1.00
    ], dtype=float),
}

FILLED_OUT = Path("everything_else/strategies_and_results/focused_only_on_27th_jan_a_norm/filled_dataset_hybrid_expiry_trend.csv")
SUBMISSION_OUT = Path("everything_else/strategies_and_results/focused_only_on_27th_jan_a_norm/submission_hybrid_expiry_trend.csv")
DIAGNOSTICS_OUT = Path("everything_else/strategies_and_results/focused_only_on_27th_jan_a_norm/diagnostics_hybrid_expiry_trend.csv")
A_WEIGHTS_OUT = Path("everything_else/strategies_and_results/focused_only_on_27th_jan_a_norm/a_weights_27jan.csv")
CROSS_DIAGNOSTICS_OUT = Path("everything_else/strategies_and_results/focused_only_on_27th_jan_a_norm/cross_section_diagnostics_hybrid_expiry_trend.csv")
TIME_DIAGNOSTICS_OUT = Path("everything_else/strategies_and_results/focused_only_on_27th_jan_a_norm/time_model_diagnostics_hybrid_expiry_trend.csv")


def parse_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Parse option columns into column, strike, option_type, and expiry metadata."""
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


def safe_positive_iv(x: float) -> float:
    """Return a positive finite IV, or NaN if x is unusable."""
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


def make_submission(original: pd.DataFrame, filled: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Create Kaggle submission from cells missing in the original input file."""
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
    """
    Fit y = a*x^2 + b*x + c.

    Fallbacks:
        0 points -> None
        1 point  -> constant
        2 points -> linear
        3+       -> quadratic
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

    degree = min(2, len(y) - 1)

    if degree == 1:
        coeff = np.polyfit(x, y, 1)
        return np.array([0.0, float(coeff[0]), float(coeff[1])]), "linear"

    coeff = np.polyfit(x, y, 2)
    return np.array([float(coeff[0]), float(coeff[1]), float(coeff[2])]), "quadratic"


def eval_quadratic(coeff, x):
    """Evaluate a fitted quadratic/linear/constant curve safely."""
    if coeff is None:
        return np.nan

    pred = float(np.polyval(coeff, x))
    return safe_positive_iv(pred)


def loo_quadratic_preds(x, y):
    """Leave-one-out predictions for the quadratic model."""
    n = len(y)
    preds = np.full(n, np.nan)

    for i in range(n):
        coeff, _ = fit_quadratic(np.delete(x, i), np.delete(y, i))
        preds[i] = eval_quadratic(coeff, x[i])

    return preds


def kernel_predict_many(x_obs, y_obs, x_targets, bandwidth):
    """
    Local kernel smoother:
        w_j = exp(-(x_j - x_target)^2 / (2b))
        pred = sum(w_j y_j) / sum(w_j)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_targets = np.asarray(x_targets, dtype=float)

    if len(y_obs) == 0:
        return np.full(len(x_targets), np.nan)

    if len(y_obs) == 1:
        return np.full(len(x_targets), safe_positive_iv(y_obs[0]))

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


def choose_cross_section_blend_by_loo(x, y, option_type):
    """
    Tune:
        pred_cross = blend * quadratic_pred + (1 - blend) * kernel_pred

    using leave-one-out MSE on the current prediction training list.
    """
    n = len(y)

    if n <= 1:
        return BANDWIDTH_GRID_BY_TYPE[option_type][0], 1.0, "constant", np.inf

    bandwidth_grid = BANDWIDTH_GRID_BY_TYPE[option_type]
    blend_grid = BLEND_GRID_BY_TYPE[option_type]

    q_loo = loo_quadratic_preds(x, y)
    valid_q = np.isfinite(q_loo)

    if valid_q.any():
        best_mse = float(np.mean((q_loo[valid_q] - y[valid_q]) ** 2))
    else:
        best_mse = np.inf

    best_bandwidth = float(bandwidth_grid[len(bandwidth_grid) // 2])
    best_blend = 1.0
    best_model = "pure_quadratic"

    for bandwidth in bandwidth_grid:
        k_loo = loo_kernel_preds(x, y, bandwidth)

        for blend in blend_grid:
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


def is_edge_missing(row, missing_col, side_cols, strike_map):
    """
    Determine edge status within CE or PE only.

    A missing value is an edge if, among the observed values of the same option
    type at that timestamp, there is no observed strike to its left or no
    observed strike to its right.
    """
    k_missing = strike_map[missing_col]

    observed_strikes = [
        strike_map[col]
        for col in side_cols
        if pd.notna(row[col])
    ]

    if not observed_strikes:
        return True, "edge_no_observed_same_side"

    has_left = any(k < k_missing for k in observed_strikes)
    has_right = any(k > k_missing for k in observed_strikes)

    if not has_left:
        return True, "edge_no_left_observed"
    if not has_right:
        return True, "edge_no_right_observed"

    return False, "not_edge"


def collect_same_row_training_points(row, opt_type, cols_by_type, strike_map):
    """Collect x = strike/spot and y = IV from one timestamp and one option type."""
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []

    cols = cols_by_type[opt_type]
    obs_cols = [c for c in cols if pd.notna(row[c])]

    x = np.array([strike_map[c] / spot for c in obs_cols], dtype=float)
    y = np.array([row[c] for c in obs_cols], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    kept_cols = [c for c, keep in zip(obs_cols, mask) if keep]

    return x[mask], y[mask], kept_cols


def collect_edge_training_points(df, row_idx, opt_type, cols_by_type, strike_map):
    """
    Edge prediction training set:
        use up to 3 nearest timestamps and pool their same-side observed points.
    """
    candidate_rows = [(abs(int(j) - int(row_idx)), int(j)) for j in df.index]
    candidate_rows.sort(key=lambda z: (z[0], z[1]))

    used_rows = []
    x_all = []
    y_all = []

    for _, j in candidate_rows:
        if len(used_rows) >= EDGE_NEAREST_TIMESTAMPS:
            break

        row = df.loc[j]
        x, y, _ = collect_same_row_training_points(row, opt_type, cols_by_type, strike_map)

        if len(y) == 0:
            continue

        used_rows.append(j)
        x_all.extend(x.tolist())
        y_all.extend(y.tolist())

    return np.asarray(x_all, dtype=float), np.asarray(y_all, dtype=float), used_rows


def predict_cross_section_for_cell(
    df,
    row_idx,
    missing_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
):
    """
    Produce pred_cross_section for a single missing cell.

    Rule:
        - If edge: use nearest 3 timestamps to fit the quadratic/kernel.
        - Otherwise: use same-row quadratic/kernel.
        - In both cases: tune the final quadratic/kernel blend by LOO.
    """
    row = df.loc[row_idx]
    side_cols = cols_by_type[opt_type]

    edge, edge_reason = is_edge_missing(row, missing_col, side_cols, strike_map)

    if edge:
        x_obs, y_obs, used_rows = collect_edge_training_points(
            df=df,
            row_idx=row_idx,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
        )
        source = "edge_nearest_3_timestamps"
    else:
        x_obs, y_obs, _ = collect_same_row_training_points(
            row=row,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
        )
        used_rows = [row_idx]
        source = "same_row_non_edge"

    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": source,
            "edge_reason": edge_reason,
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan,
            "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan,
            "loo_mse": np.nan,
            "n_train": len(y_obs),
            "used_rows": used_rows,
        }

    x_target = np.array([strike_map[missing_col] / spot], dtype=float)

    bandwidth, blend, selected_model, loo_mse = choose_cross_section_blend_by_loo(
        x=x_obs,
        y=y_obs,
        option_type=opt_type,
    )

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    q_pred = np.array([eval_quadratic(coeff, x_target[0])], dtype=float)
    k_pred = kernel_predict_many(x_obs, y_obs, x_target, bandwidth)

    pred = blend * q_pred[0] + (1.0 - blend) * k_pred[0]

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"

    return {
        "prediction": safe_positive_iv(pred),
        "source": source,
        "edge_reason": edge_reason,
        "selected_model": selected_model,
        "quadratic_fit_kind": fit_kind,
        "bandwidth": bandwidth,
        "blend_quadratic_weight": blend,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_rows": used_rows,
    }


def fit_time_quadratic(t_values, iv_values):
    """
    Fit 27 Jan time model:
        IV = a*time^2 + b*time + c

    using previous observed 27 Jan points only.
    """
    t_values = np.asarray(t_values, dtype=float)
    iv_values = np.asarray(iv_values, dtype=float)

    mask = np.isfinite(t_values) & np.isfinite(iv_values)
    t_values = t_values[mask]
    iv_values = iv_values[mask]

    if len(iv_values) < MIN_TIME_POINTS_FOR_QUADRATIC:
        return None, "not_enough_previous_points"

    coeff = np.polyfit(t_values, iv_values, 2)
    return np.array([float(coeff[0]), float(coeff[1]), float(coeff[2])]), "quadratic_time"


def predict_time_quadratic(coeff, t_value):
    """Evaluate the previous-only time quadratic."""
    if coeff is None:
        return np.nan

    pred = float(np.polyval(coeff, t_value))
    return safe_positive_iv(pred)


def compute_27jan_a_weights(df, meta):
    """
    Compute one fixed a weight per option for 27 Jan.

    For each option:
        derivative = (current_IV - last_known_IV) / time_between_points
        signed_derivative_sum += derivative

    Then:
        a_i = sum_i / sum(sum_i)

    Since signed sums can produce negative or >1 weights, the final blend weight
    is clipped to [0, 1]. The raw value is also saved as a_raw.
    """
    dates = df["datetime_parsed"].dt.date
    jan27_indices = np.where(dates == EXPIRY_DAY)[0]
    time_index = np.arange(len(df), dtype=float)

    rows = []

    for _, rec in meta.iterrows():
        col = rec["column"]
        values = df[col].to_numpy(dtype=float)

        observed_27 = [
            idx for idx in jan27_indices
            if np.isfinite(values[idx])
        ]

        derivative_sum = 0.0
        derivative_values = []

        for prev_idx, cur_idx in zip(observed_27[:-1], observed_27[1:]):
            dt = time_index[cur_idx] - time_index[prev_idx]
            if dt <= 0:
                continue

            derivative = (values[cur_idx] - values[prev_idx]) / dt

            if np.isfinite(derivative):
                derivative_sum += float(derivative)
                derivative_values.append(float(derivative))

        rows.append({
            "contract": col,
            "option_type": rec["option_type"],
            "strike": rec["strike"],
            "signed_derivative_sum": derivative_sum,
            "n_observed_27jan": len(observed_27),
            "n_derivative_terms": len(derivative_values),
            "mean_signed_derivative": float(np.mean(derivative_values)) if derivative_values else np.nan,
            "min_signed_derivative": float(np.min(derivative_values)) if derivative_values else np.nan,
            "max_signed_derivative": float(np.max(derivative_values)) if derivative_values else np.nan,
        })

    weights = pd.DataFrame(rows)
    denom = float(weights["signed_derivative_sum"].sum())

    if np.isfinite(denom) and abs(denom) > 1e-15:
        weights["a_raw"] = weights["signed_derivative_sum"] / denom
    else:
        weights["a_raw"] = 0.0

    weights["a"] = weights["a_raw"].clip(0.0, 1.0)

    return weights


def predict_time_for_27jan_cell(df, row_idx, col, jan27_indices, time_index):
    """
    Compute pred_t for one 27 Jan missing cell.

    It uses previous observed 27 Jan points only:
        s < t
    """
    values = df[col].to_numpy(dtype=float)

    previous_observed = [
        idx for idx in jan27_indices
        if idx < row_idx and np.isfinite(values[idx])
    ]

    if len(previous_observed) < MIN_TIME_POINTS_FOR_QUADRATIC:
        return {
            "prediction": np.nan,
            "fit_kind": "not_enough_previous_points",
            "n_train": len(previous_observed),
            "train_indices": previous_observed,
        }

    t_train = time_index[previous_observed]
    y_train = values[previous_observed]

    coeff, fit_kind = fit_time_quadratic(t_train, y_train)
    pred = predict_time_quadratic(coeff, time_index[row_idx])

    return {
        "prediction": pred,
        "fit_kind": fit_kind,
        "n_train": len(previous_observed),
        "train_indices": previous_observed,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.resolve()}.\n"
            "Put dataset.csv in the same folder as this script, or edit DATA_PATH."
        )

    raw = pd.read_csv(DATA_PATH)
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

    dates = df["datetime_parsed"].dt.date
    jan27_indices = np.where(dates == EXPIRY_DAY)[0]
    time_index = np.arange(len(df), dtype=float)

    a_weights = compute_27jan_a_weights(df, meta)
    a_map = dict(zip(a_weights["contract"], a_weights["a"]))
    a_raw_map = dict(zip(a_weights["contract"], a_weights["a_raw"]))
    a_weights.to_csv(A_WEIGHTS_OUT, index=False)

    main_diag_rows = []
    cross_diag_rows = []
    time_diag_rows = []

    diagnostics = {
        "missing_initial": int(df[option_cols].isna().sum().sum()),
        "filled": 0,
        "pre27_cross_only": 0,
        "jan27_blend_time_cross": 0,
        "jan27_cross_only_time_unavailable": 0,
        "fallback_global_median": 0,
    }

    missing_cells = [
        (idx, col)
        for idx in df.index
        for col in option_cols
        if pd.isna(df.at[idx, col])
    ]

    for row_idx, col in tqdm(missing_cells, desc="Hybrid expiry-trend filling"):
        opt_type = type_map[col]
        current_date = df.loc[row_idx, "datetime_parsed"].date()

        cross_info = predict_cross_section_for_cell(
            df=df,
            row_idx=row_idx,
            missing_col=col,
            opt_type=opt_type,
            cols_by_type=cols_by_type,
            strike_map=strike_map,
            global_median_iv=global_median_iv,
        )
        pred_cross = cross_info["prediction"]

        cross_diag_rows.append({
            "row_index": row_idx,
            "datetime": df.loc[row_idx, "datetime"],
            "contract": col,
            "option_type": opt_type,
            "strike": strike_map[col],
            "pred_cross_section": pred_cross,
            "cross_source": cross_info["source"],
            "edge_reason": cross_info["edge_reason"],
            "selected_model": cross_info["selected_model"],
            "quadratic_fit_kind": cross_info.get("quadratic_fit_kind", np.nan),
            "bandwidth": cross_info["bandwidth"],
            "blend_quadratic_weight": cross_info["blend_quadratic_weight"],
            "loo_mse": cross_info["loo_mse"],
            "n_train": cross_info["n_train"],
            "used_rows": "|".join(map(str, cross_info["used_rows"])),
        })

        if current_date < EXPIRY_DAY:
            a = 0.0
            pred_t = np.nan
            final_pred = pred_cross
            method = "pre27_cross_only"
            diagnostics["pre27_cross_only"] += 1

        elif current_date == EXPIRY_DAY:
            a = float(a_map.get(col, 0.0))

            time_info = predict_time_for_27jan_cell(
                df=df,
                row_idx=row_idx,
                col=col,
                jan27_indices=jan27_indices,
                time_index=time_index,
            )

            pred_t = time_info["prediction"]

            time_diag_rows.append({
                "row_index": row_idx,
                "datetime": df.loc[row_idx, "datetime"],
                "contract": col,
                "option_type": opt_type,
                "strike": strike_map[col],
                "pred_t": pred_t,
                "time_fit_kind": time_info["fit_kind"],
                "n_train_previous_27jan": time_info["n_train"],
                "train_indices": "|".join(map(str, time_info["train_indices"])),
                "a": a,
                "a_raw": float(a_raw_map.get(col, np.nan)),
            })

            if np.isfinite(pred_t):
                final_pred = a * pred_t + (1.0 - a) * pred_cross
                method = "jan27_blend_time_cross"
                diagnostics["jan27_blend_time_cross"] += 1
            else:
                final_pred = pred_cross
                method = "jan27_cross_only_time_unavailable"
                diagnostics["jan27_cross_only_time_unavailable"] += 1

        else:
            a = 0.0
            pred_t = np.nan
            final_pred = pred_cross
            method = "post_expiry_cross_only"

        if not np.isfinite(final_pred):
            final_pred = global_median_iv
            diagnostics["fallback_global_median"] += 1

        final_pred = safe_positive_iv(final_pred)
        filled.at[row_idx, col] = final_pred
        diagnostics["filled"] += 1

        main_diag_rows.append({
            "row_index": row_idx,
            "datetime": df.loc[row_idx, "datetime"],
            "contract": col,
            "option_type": opt_type,
            "strike": strike_map[col],
            "date": str(current_date),
            "method": method,
            "a": a,
            "pred_t": pred_t,
            "pred_cross_section": pred_cross,
            "final_prediction": final_pred,
        })

    diagnostics["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled_out = filled.drop(columns=["datetime_parsed"])
    original_out = df.drop(columns=["datetime_parsed"])

    filled_out.to_csv(FILLED_OUT, index=False)
    submission = make_submission(original_out, filled_out, SUBMISSION_OUT)

    pd.DataFrame(main_diag_rows).to_csv(DIAGNOSTICS_OUT, index=False)
    pd.DataFrame(cross_diag_rows).to_csv(CROSS_DIAGNOSTICS_OUT, index=False)
    pd.DataFrame(time_diag_rows).to_csv(TIME_DIAGNOSTICS_OUT, index=False)

    print(f"✅ Filled dataset saved → {FILLED_OUT}")
    print(f"✅ Submission saved → {SUBMISSION_OUT} ({len(submission)} rows)")
    print(f"✅ Main diagnostics saved → {DIAGNOSTICS_OUT}")
    print(f"✅ Cross-section diagnostics saved → {CROSS_DIAGNOSTICS_OUT}")
    print(f"✅ Time-model diagnostics saved → {TIME_DIAGNOSTICS_OUT}")
    print(f"✅ 27 Jan a-weights saved → {A_WEIGHTS_OUT}")

    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
