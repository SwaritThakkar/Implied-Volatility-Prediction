"""
Fast Ratio-Moneyness Quadratic Spline + Kernel Smoother IV Imputer

Requested change:
    moneyness = strike / underlying_price
not:
    log(strike / underlying_price)

For each timestamp and option type CE/PE separately:

1. Fit quadratic spline:
       IV = spline(m)
   where:
       m = strike / underlying_price
       spline degree k = 2

2. Fit local kernel smoother:
       w_j = exp(-(m_j - m_i)^2 / (2b))
       IV_hat_i = sum_j w_j IV_j / sum_j w_j

3. Tune bandwidth b and blend weight by row-wise leave-one-out:
       final = blend * quadratic_spline + (1 - blend) * kernel

No future rows are used.
This is purely same-row cross-sectional fitting.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
except Exception as exc:
    raise ImportError("This script requires scipy. Install with: pip install scipy") from exc


DATA_PATH = Path("dataset.csv")

EPS_IV = 1e-6
SEPARATOR = "||"

# Ratio moneyness K/S is very close to log-moneyness near ATM, but not identical.
# Keep the useful tuned region around 1e-5 to 5e-5, with some tails.
BANDWIDTH_GRID_BY_TYPE = {
    "CE": np.array([
        1e-7, 2e-7, 5e-7,
        1e-6, 2e-6, 5e-6,
        7.5e-6, 1e-5, 1.25e-5, 1.5e-5,
        2e-5, 2.5e-5, 3e-5, 5e-5,
        7.5e-5, 1e-4, 2e-4, 5e-4
    ], dtype=float),
    "PE": np.array([
        5e-7, 1e-6, 2e-6, 5e-6,
        7.5e-6, 1e-5, 1.25e-5, 1.5e-5,
        2e-5, 2.5e-5, 3e-5, 5e-5,
        7.5e-5, 1e-4, 2e-4, 5e-4
    ], dtype=float),
}

BLEND_GRID_BY_TYPE = {
    "CE": np.array([
        0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
    ], dtype=float),
    "PE": np.array([
        0.50, 0.60, 0.70, 0.75, 0.80,
        0.825, 0.85, 0.875, 0.90, 0.925,
        0.95, 0.975, 1.00
    ], dtype=float),
}

FILLED_OUT = Path("filled_dataset_ratio_quad_spline_kernel_fast.csv")
SUBMISSION_OUT = Path("submission_ratio_quad_spline_kernel_fast.csv")
SELECTION_OUT = Path("ratio_quad_spline_kernel_fast_diagnostics.csv")


def parse_metadata(df):
    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )

    candidate_cols = [c for c in df.columns if c not in ["datetime", "underlying_price"]]
    records = []

    for col in candidate_cols:
        match = pattern.match(col)
        if match:
            item = match.groupdict()
            item["column"] = col
            item["strike"] = int(item["strike"])
            records.append(item)

    meta = pd.DataFrame(records)

    if meta.empty:
        raise ValueError("No option columns parsed.")

    return meta


def collapse_duplicate_x(x, y):
    tmp = pd.DataFrame({"x": x, "y": y})
    tmp = tmp.groupby("x", as_index=False)["y"].mean().sort_values("x")
    return tmp["x"].to_numpy(dtype=float), tmp["y"].to_numpy(dtype=float)


def fit_quad_spline(x, y):
    """
    Quadratic spline with k=2.

    Fallbacks:
        0 points -> None
        1 point  -> constant
        2 points -> linear
        >=3      -> interpolated quadratic spline
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return None

    if len(y) == 1:
        return ("constant", float(y[0]))

    x, y = collapse_duplicate_x(x, y)

    if len(y) == 1:
        return ("constant", float(y[0]))

    if len(y) == 2:
        coeff = np.polyfit(x, y, 1)
        return ("poly", np.array([0.0, coeff[0], coeff[1]]))

    try:
        spline = InterpolatedUnivariateSpline(x, y, k=2)
        return ("spline", spline)
    except Exception:
        coeff = np.polyfit(x, y, min(2, len(y)-1))
        if len(coeff) == 2:
            coeff = np.array([0.0, coeff[0], coeff[1]])
        return ("poly", coeff)


def predict_spline(fit_obj, x):
    if fit_obj is None:
        return np.nan

    kind, obj = fit_obj

    if kind == "constant":
        pred = float(obj)
    elif kind == "poly":
        pred = float(np.polyval(obj, x))
    elif kind == "spline":
        pred = float(obj(x))
    else:
        pred = np.nan

    if not np.isfinite(pred):
        return np.nan

    return max(pred, EPS_IV)


def loo_spline_preds(x, y):
    n = len(y)
    preds = np.full(n, np.nan)

    for i in range(n):
        fit_obj = fit_quad_spline(np.delete(x, i), np.delete(y, i))
        preds[i] = predict_spline(fit_obj, x[i])

    return preds


def kernel_predict_many(x_obs, y_obs, x_targets, bandwidth):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_targets = np.asarray(x_targets, dtype=float)

    if len(y_obs) == 0:
        return np.full(len(x_targets), np.nan)

    if len(y_obs) == 1:
        return np.full(len(x_targets), float(y_obs[0]))

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


def choose_model_by_loo(x, y, option_type):
    n = len(y)

    if n <= 1:
        return BANDWIDTH_GRID_BY_TYPE[option_type][0], 1.0, "constant", np.inf

    bandwidth_grid = BANDWIDTH_GRID_BY_TYPE[option_type]
    blend_grid = BLEND_GRID_BY_TYPE[option_type]

    spline_loo = loo_spline_preds(x, y)
    valid_sp = np.isfinite(spline_loo)

    if valid_sp.any():
        best_mse = float(np.mean((spline_loo[valid_sp] - y[valid_sp]) ** 2))
    else:
        best_mse = np.inf

    best_bandwidth = bandwidth_grid[len(bandwidth_grid) // 2]
    best_blend = 1.0
    best_model = "pure_spline"

    for bandwidth in bandwidth_grid:
        k_loo = loo_kernel_preds(x, y, bandwidth)

        for blend in blend_grid:
            pred = blend * spline_loo + (1.0 - blend) * k_loo
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
                    best_model = "pure_spline"
                else:
                    best_model = "spline_kernel_blend"

    return best_bandwidth, best_blend, best_model, best_mse


def make_submission(original, filled, out_path):
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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Put dataset.csv in the same folder as this script.")

    df = pd.read_csv(DATA_PATH)
    meta = parse_metadata(df)

    option_cols = meta["column"].tolist()
    strike_map = dict(zip(meta["column"], meta["strike"]))
    type_map = dict(zip(meta["column"], meta["option_type"]))

    cols_by_type = {
        "CE": [c for c in option_cols if type_map[c] == "CE"],
        "PE": [c for c in option_cols if type_map[c] == "PE"],
    }

    global_median_iv = df[option_cols].stack().median()
    filled = df.copy()

    diagnostics = {
        "missing_initial": int(df[option_cols].isna().sum().sum()),
        "filled": 0,
        "pure_spline_selected": 0,
        "pure_kernel_selected": 0,
        "spline_kernel_blend_selected": 0,
        "fallback_global_median": 0,
    }

    selection_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ratio spline + kernel filling"):
        spot = row["underlying_price"]

        for opt_type, cols in cols_by_type.items():
            vals = row[cols]
            obs_cols = vals.index[vals.notna()].tolist()
            miss_cols = vals.index[vals.isna()].tolist()

            if not miss_cols:
                continue

            if pd.isna(spot) or spot <= 0 or len(obs_cols) == 0:
                for col in miss_cols:
                    filled.at[idx, col] = global_median_iv
                    diagnostics["filled"] += 1
                    diagnostics["fallback_global_median"] += 1
                continue

            # Ratio moneyness, not log moneyness.
            x_obs = np.array([strike_map[c] / spot for c in obs_cols], dtype=float)
            y_obs = np.array([row[c] for c in obs_cols], dtype=float)

            mask = np.isfinite(x_obs) & np.isfinite(y_obs)
            x_obs = x_obs[mask]
            y_obs = y_obs[mask]

            if len(y_obs) == 0:
                for col in miss_cols:
                    filled.at[idx, col] = global_median_iv
                    diagnostics["filled"] += 1
                    diagnostics["fallback_global_median"] += 1
                continue

            bandwidth, blend, selected_model, loo_mse = choose_model_by_loo(
                x_obs,
                y_obs,
                opt_type,
            )

            fit_obj = fit_quad_spline(x_obs, y_obs)

            if selected_model == "pure_spline":
                diagnostics["pure_spline_selected"] += 1
            elif selected_model == "pure_kernel":
                diagnostics["pure_kernel_selected"] += 1
            elif selected_model == "spline_kernel_blend":
                diagnostics["spline_kernel_blend_selected"] += 1

            selection_rows.append(
                {
                    "row_index": idx,
                    "datetime": row["datetime"],
                    "option_type": opt_type,
                    "n_observed": len(y_obs),
                    "n_missing": len(miss_cols),
                    "selected_model": selected_model,
                    "bandwidth": bandwidth,
                    "blend_spline_weight": blend,
                    "loo_mse": loo_mse,
                }
            )

            x_miss = np.array([strike_map[c] / spot for c in miss_cols], dtype=float)
            spline_preds = np.array([predict_spline(fit_obj, x) for x in x_miss], dtype=float)
            kernel_preds = kernel_predict_many(x_obs, y_obs, x_miss, bandwidth)

            final_preds = blend * spline_preds + (1.0 - blend) * kernel_preds

            for col, pred in zip(miss_cols, final_preds):
                if not np.isfinite(pred):
                    pred = global_median_iv
                    diagnostics["fallback_global_median"] += 1

                filled.at[idx, col] = max(float(pred), EPS_IV)
                diagnostics["filled"] += 1

    diagnostics["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled.to_csv(FILLED_OUT, index=False)
    submission = make_submission(df, filled, SUBMISSION_OUT)
    pd.DataFrame(selection_rows).to_csv(SELECTION_OUT, index=False)

    print(f"✅ Filled dataset saved → {FILLED_OUT}")
    print(f"✅ Submission saved → {SUBMISSION_OUT} ({len(submission)} rows)")
    print(f"✅ Selection diagnostics saved → {SELECTION_OUT}")

    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
