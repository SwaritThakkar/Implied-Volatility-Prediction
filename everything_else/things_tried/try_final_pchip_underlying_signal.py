"""
try_final_pchip_underlying_signal.py
====================================

Best baseline retained:
    - Edge logic: try_final.py degree+bandwidth LOO edge ensemble.
    - Interior cross-section logic: 0.75 * local quadratic WLS + 0.25 * PCHIP.

New validated-style interior correction:
    - Only for interior / non-edge cells.
    - Uses underlying-price movement through the signal found in EDA:

          signal = local_smile_slope * dmoneyness
          dmoneyness = K / S_t - K / S_{t-1}

    - Learns alpha by CV per bucket:

          bucket = (option_type, k_rank)
          final = cross_pred + alpha[bucket] * signal

    - A CV gate is used: if the correction does not improve validation MSE,
      all alphas are set to 0 and the script falls back to the baseline.
    - Edges are not touched by this correction.

Run:
    python try_final_pchip_underlying_signal.py --data dataset.csv

Outputs:
    filled_dataset_try_final_pchip_underlying_signal.csv
    submission_try_final_pchip_underlying_signal.csv
    diagnostics_try_final_pchip_underlying_signal.csv
    cross_section_diagnostics_try_final_pchip_underlying_signal.csv
    underlying_signal_alpha_validation_try_final_pchip_underlying_signal.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = Path(
    "dataset.csv"
)

EPS_IV = 1e-6
SEPARATOR = "||"

BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
EDGE_LOCAL_POLY_BW = 2e-4
LOCAL_POLY_DEGREE = 2

EDGE_BLEND_CLAUDE = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14
MIN_EDGE_LOCAL_NEIGHBORS = 3

PCHIP_INTERIOR_WEIGHT = 0.25
MIN_PCHIP_POINTS = 4

# New underlying-price correction config.
UNDERLYING_SIGNAL_CV_MASK_FRAC = 0.08
UNDERLYING_SIGNAL_CV_REPS = 5
UNDERLYING_SIGNAL_CV_SEED = 42
ALPHA_CLIP_ABS = 8.0               # alpha itself is clipped for stability
CORRECTION_CLIP_GRID = [0.0025, 0.005, 0.0075, 0.01]
DEFAULT_CORRECTION_CLIP = 0.005
MIN_BUCKET_RECORDS = 10            # below this, use global/broader fallback alpha

CFG_DATA = str(DEFAULT_DATA_PATH)
CFG_OUT_PREFIX = "try_final_pchip_underlying_signal"


def parse_args():
    import types, sys
    if any("jupyter" in a or "kernel" in a or ".json" in a for a in sys.argv[1:]):
        return types.SimpleNamespace(
            data=CFG_DATA,
            out_prefix=CFG_OUT_PREFIX,
            skip_cv=False,
            correction_clip=None,
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=CFG_DATA)
    parser.add_argument("--out-prefix", type=str, default=CFG_OUT_PREFIX)
    parser.add_argument("--skip-cv", action="store_true", help="Skip signal CV and use zero correction.")
    parser.add_argument("--correction-clip", type=float, default=None, help="Manual correction clip; CV still learns alphas unless --skip-cv.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Basic helpers
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


def is_jan27_ts(x):
    try:
        return pd.Timestamp(x).date() == pd.Timestamp("2026-01-27").date()
    except Exception:
        return False


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


def fit_quadratic(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) == 0:
        return None, "no_points"
    if len(y) == 1:
        return np.array([0., 0., float(y[0])]), "constant"
    if len(y) == 2:
        c = np.polyfit(x, y, 1)
        return np.array([0., float(c[0]), float(c[1])]), "linear"
    return np.array(np.polyfit(x, y, 2), float), "quadratic"


def eval_quadratic(coeff, x):
    if coeff is None:
        return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


# ─────────────────────────────────────────────────────────────────────
# Local polynomial WLS
# ─────────────────────────────────────────────────────────────────────

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs, y_obs = x_obs[mask], y_obs[mask]

    if len(y_obs) == 0:
        return np.nan
    if len(y_obs) == 1:
        return safe_iv(float(y_obs[0]))

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
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
    best_bw, best_mse = float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
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
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), 1, np.inf
    best_bw = float(bandwidth_grid[len(bandwidth_grid) // 2])
    best_deg, best_mse = 1, np.inf
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


def pchip_same_row_pred(x_obs, y_obs, x_target):
    if PchipInterpolator is None:
        return np.nan

    x = np.asarray(x_obs, float)
    y = np.asarray(y_obs, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(y) < MIN_PCHIP_POINTS:
        return np.nan

    order = np.argsort(x)
    x, y = x[order], y[order]

    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) < MIN_PCHIP_POINTS:
        return np.nan
    if len(ux) != len(x):
        y2 = np.zeros(len(ux), dtype=float)
        cnt = np.zeros(len(ux), dtype=float)
        for i, g in enumerate(inv):
            y2[g] += y[i]
            cnt[g] += 1
        x, y = ux, y2 / np.maximum(cnt, 1)
    else:
        x = ux

    if not (x[0] <= x_target <= x[-1]):
        return np.nan

    try:
        p = float(PchipInterpolator(x, y, extrapolate=False)(x_target))
    except Exception:
        return np.nan

    return safe_iv(p) if np.isfinite(p) else np.nan


# ─────────────────────────────────────────────────────────────────────
# Row-structure helpers
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
    records = [
        {"column": c, "strike": strike_map[c], "is_missing": pd.isna(row[c]), "iv": row[c]}
        for c in cols_by_type[opt_type]
    ]
    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)


def get_edge_blocks(row, opt_type, cols_by_type, strike_map):
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    left_block, right_block = [], []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]):
            left_block.append(rec["column"])
        else:
            break
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


def get_k_rank(target_col, cols_by_type, strike_map, opt_type):
    cols_sorted = sorted(cols_by_type[opt_type], key=lambda c: strike_map[c])
    return int(cols_sorted.index(target_col))


# ─────────────────────────────────────────────────────────────────────
# Interior cross-section + underlying signal correction
# ─────────────────────────────────────────────────────────────────────

def local_smile_slope_at_target(x_obs, y_obs, x_target):
    """
    Estimate dIV/dmoneyness near x_target using the nearest observed points.
    Prefer a bracketing secant. Fall back to nearest two observed points.
    """
    x = np.asarray(x_obs, float)
    y = np.asarray(y_obs, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < 2:
        return np.nan

    order = np.argsort(x)
    x, y = x[order], y[order]

    # Collapse duplicates.
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) < 2:
        return np.nan
    if len(ux) != len(x):
        yy = np.zeros(len(ux), dtype=float)
        cnt = np.zeros(len(ux), dtype=float)
        for i, g in enumerate(inv):
            yy[g] += y[i]
            cnt[g] += 1
        x, y = ux, yy / np.maximum(cnt, 1)
    else:
        x = ux

    right = int(np.searchsorted(x, x_target, side="right"))
    left = right - 1

    if 0 <= left < len(x) and 0 <= right < len(x) and x[right] != x[left]:
        return float((y[right] - y[left]) / (x[right] - x[left]))

    nearest = np.argsort(np.abs(x - x_target))[:2]
    if len(nearest) < 2:
        return np.nan
    i, j = sorted(nearest)
    if x[j] == x[i]:
        return np.nan
    return float((y[j] - y[i]) / (x[j] - x[i]))


def dmoneyness_from_underlying(df_like, row_idx, target_col, strike_map):
    """K/S_t - K/S_{t-1}; returns nan if no valid previous timestamp."""
    if row_idx <= 0:
        return np.nan
    s_t = df_like.at[row_idx, "underlying_price"]
    s_prev = df_like.at[row_idx - 1, "underlying_price"]
    if pd.isna(s_t) or pd.isna(s_prev) or s_t <= 0 or s_prev <= 0:
        return np.nan
    k = strike_map[target_col]
    return float(k / s_t - k / s_prev)


def underlying_signal_for_cell(df_like, row_idx, target_col, opt_type, cols_by_type, strike_map):
    row = df_like.loc[row_idx]
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.nan, np.nan, np.nan
    x_obs, y_obs, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)
    x_target = strike_map[target_col] / spot
    slope = local_smile_slope_at_target(x_obs, y_obs, x_target)
    dm = dmoneyness_from_underlying(df_like, row_idx, target_col, strike_map)
    if not (np.isfinite(slope) and np.isfinite(dm)):
        return np.nan, slope, dm
    return float(slope * dm), slope, dm


def predict_non_edge_cross_section(df, row_idx, target_col, opt_type,
                                   cols_by_type, strike_map, global_median_iv):
    """Current best interior baseline: 0.75 * local quadratic WLS + 0.25 * PCHIP."""
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
            "pchip_used": False,
            "base_prediction": np.nan,
            "pchip_prediction": np.nan,
        }

    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    base_pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(base_pred):
        base_pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "local_quadratic_wls"

    pchip_pred = pchip_same_row_pred(x_obs, y_obs, x_target)
    if np.isfinite(pchip_pred):
        pred = (1.0 - PCHIP_INTERIOR_WEIGHT) * base_pred + PCHIP_INTERIOR_WEIGHT * pchip_pred
        source = "same_row_non_edge_wls_pchip_blend"
        selected_model = f"local_quad_wls_plus_pchip_w{PCHIP_INTERIOR_WEIGHT:.2f}"
        pchip_used = True
    else:
        pred = base_pred
        source = "same_row_non_edge_local_poly_wls"
        pchip_used = False

    return {
        "prediction": safe_iv(pred),
        "source": source,
        "selected_model": selected_model,
        "bandwidth": best_bw,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        "edge_degree": np.nan,
        "pchip_used": pchip_used,
        "base_prediction": safe_iv(base_pred),
        "pchip_prediction": pchip_pred if np.isfinite(pchip_pred) else np.nan,
    }


def correction_from_alpha(signal, alpha, correction_clip):
    if not np.isfinite(signal) or not np.isfinite(alpha):
        return 0.0
    raw = float(alpha) * float(signal)
    return float(np.clip(raw, -correction_clip, correction_clip))


def predict_non_edge_with_underlying_signal(df, row_idx, target_col, opt_type,
                                            cols_by_type, strike_map, global_median_iv,
                                            alpha_table=None, global_alpha=0.0,
                                            correction_clip=DEFAULT_CORRECTION_CLIP):
    info = predict_non_edge_cross_section(
        df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv
    )
    cross_pred = info["prediction"]

    k_rank = get_k_rank(target_col, cols_by_type, strike_map, opt_type)
    alpha = global_alpha
    if alpha_table is not None:
        alpha = alpha_table.get((opt_type, k_rank), alpha_table.get((opt_type, "ALL"), global_alpha))

    signal, slope, dm = underlying_signal_for_cell(df, row_idx, target_col, opt_type, cols_by_type, strike_map)
    corr = correction_from_alpha(signal, alpha, correction_clip)

    if np.isfinite(signal) and abs(corr) > 0:
        pred = safe_iv(cross_pred + corr)
        info["prediction"] = pred
        info["source"] = info["source"] + "_underlying_signal_corrected"
        info["selected_model"] = info["selected_model"] + f"_uSignal_alpha{alpha:.4g}"
        info["underlying_signal_used"] = True
    else:
        info["underlying_signal_used"] = False

    info["underlying_signal"] = signal
    info["underlying_slope"] = slope
    info["dmoneyness"] = dm
    info["underlying_alpha"] = alpha
    info["underlying_correction"] = corr
    info["k_rank"] = k_rank
    return info


# ─────────────────────────────────────────────────────────────────────
# Edge collectors and predictors: try_final.py logic
# ─────────────────────────────────────────────────────────────────────

def collect_edge_training_points_claude(row, target_col, opt_type, cols_by_type,
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
    obs_recs = [{"column": c, "strike": strike_map[c], "x": strike_map[c]/spot, "y": float(row[c])}
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
    x_t = base["x"].tolist()
    y_t = base["y"].tolist()
    used = base["column"].astype(str).tolist()
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        if pc not in already_filled:
            continue
        pv = component_value(already_filled, pc, "claude")
        if np.isfinite(pv):
            x_t.append(float(pv))
            y_t.append(float(pv))
            used.append(f"{pc}*as_xy")
    return (np.asarray(x_t, float), np.asarray(y_t, float), used,
            {"edge_side": side, "edge_block_size": len(block_cols),
             "edge_position_in_block": position})


def collect_edge_training_points_corrected(row, target_col, opt_type, cols_by_type,
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
        if pd.isna(val):
            continue
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
                         "x": strike_map[pc]/spot, "y": float(pv), "is_predicted": True})
    if not recs:
        return np.array([]), np.array([]), [], {"edge_side": side,
                                                 "edge_block_size": len(block_cols),
                                                 "edge_position_in_block": position,
                                                 "edge_observed_side_points": obs_pts}
    train = pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)
    used = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]
    return (train["x"].to_numpy(float), train["y"].to_numpy(float), used,
            {"edge_side": side, "edge_block_size": len(block_cols),
             "edge_position_in_block": position, "edge_observed_side_points": obs_pts})


def collect_edge_training_points_quadratic(row, target_col, opt_type, cols_by_type,
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
    tgt_s = strike_map[target_col]
    base_n = max(MIN_EDGE_LOCAL_NEIGHBORS, len(block_cols))
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs_recs = [{"column": c, "strike": strike_map[c], "x": strike_map[c]/spot,
                 "y": float(row[c]), "is_predicted": False}
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
        base = obs[obs["strike"] > tgt_s].sort_values("strike").head(base_n).sort_values("strike")
    else:
        base = obs.sort_values("strike")
    recs = base.to_dict("records")
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        pv = component_value(already_filled, pc, "quadratic")
        if np.isfinite(pv):
            recs.append({"column": pc, "strike": strike_map[pc],
                         "x": strike_map[pc]/spot, "y": float(pv), "is_predicted": True})
    if not recs:
        return np.array([]), np.array([]), [], {"edge_side": side,
                                                 "edge_block_size": len(block_cols),
                                                 "edge_position_in_block": position,
                                                 "edge_base_observed_needed": base_n}
    train = pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)
    used = [f"{r.column}{'*' if r.is_predicted else ''}" for r in train.itertuples(index=False)]
    return (train["x"].to_numpy(float), train["y"].to_numpy(float), used,
            {"edge_side": side, "edge_block_size": len(block_cols),
             "edge_position_in_block": position, "edge_base_observed_needed": base_n})


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


def predict_edge_claude_local_poly(df, row_idx, target_col, opt_type,
                                    cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return {"prediction": float(global_median_iv),
                "source": "edge_fallback_global_median_bad_spot",
                "selected_model": "fallback_global_median",
                "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
                "n_train": 0, "used_cols": [], "edge_degree": 1,
                "edge_side": "bad_spot", "edge_block_size": 0,
                "edge_position_in_block": np.nan,
                "pchip_used": False,
                "underlying_signal_used": False}

    x_obs, y_obs, used, edge_info = collect_edge_training_points_claude(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)

    if len(y_obs) == 0:
        return {"prediction": float(global_median_iv),
                "source": "edge_fallback_global_median_no_neighbors",
                "selected_model": "fallback_global_median",
                "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
                "n_train": 0, "used_cols": used, "edge_degree": 1,
                "pchip_used": False,
                "underlying_signal_used": False,
                **edge_info}

    x_target = strike_map[target_col] / spot
    pred, bw, deg, loo_mse = _edge_predict_with_deg_select(x_obs, y_obs, x_target, global_median_iv)

    return {"prediction": pred,
            "source": "edge_claude_linear_or_quad_by_loo",
            "selected_model": f"edge_claude_deg{deg}_bw{bw:.0e}",
            "bandwidth": bw, "loo_mse": loo_mse,
            "n_train": len(y_obs), "used_cols": used,
            "edge_degree": deg,
            "pchip_used": False,
            "underlying_signal_used": False,
            **edge_info}


def predict_edge_corrected_local_poly(df, row_idx, target_col, opt_type,
                                       cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used, edge_info = collect_edge_training_points_corrected(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used, edge_info

    x_target = strike_map[target_col] / spot
    pred, _, _, _ = _edge_predict_with_deg_select(x_obs, y_obs, x_target, global_median_iv)
    return pred, used, edge_info


def predict_edge_quadratic(df, row_idx, target_col, opt_type,
                            cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used, edge_info = collect_edge_training_points_quadratic(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), used, edge_info, "deg_select"

    x_target = strike_map[target_col] / spot
    pred, _, deg, _ = _edge_predict_with_deg_select(x_obs, y_obs, x_target, global_median_iv)
    return pred, used, edge_info, f"deg{deg}_by_loo"


def predict_edge_ensemble(df, row_idx, target_col, opt_type,
                           cols_by_type, strike_map, global_median_iv, already_filled):
    claude_info = predict_edge_claude_local_poly(
        df, row_idx, target_col, opt_type,
        cols_by_type, strike_map, global_median_iv, already_filled)
    claude_pred = claude_info["prediction"]

    corrected_pred, corrected_cols, corrected_info = predict_edge_corrected_local_poly(
        df, row_idx, target_col, opt_type,
        cols_by_type, strike_map, global_median_iv, already_filled)

    quadratic_pred, quad_cols, quad_info, fit_kind = predict_edge_quadratic(
        df, row_idx, target_col, opt_type,
        cols_by_type, strike_map, global_median_iv, already_filled)

    components = {
        "claude": safe_iv(claude_pred),
        "corrected": safe_iv(corrected_pred),
        "quadratic": safe_iv(quadratic_pred),
    }
    pred = (EDGE_BLEND_CLAUDE * components["claude"] +
            EDGE_BLEND_CORRECTED * components["corrected"] +
            EDGE_BLEND_QUADRATIC * components["quadratic"])

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_blended_deg_selected"

    return {**claude_info,
            "prediction": safe_iv(pred),
            "source": "edge_blended_deg_selected",
            "selected_model": selected_model,
            "component_predictions": components,
            "corrected_used_cols": corrected_cols,
            "quadratic_used_cols": quad_cols,
            "quadratic_fit_kind": fit_kind,
            "edge_observed_side_points": corrected_info.get("edge_observed_side_points", np.nan),
            "edge_base_observed_needed": quad_info.get("edge_base_observed_needed", np.nan)}


# ─────────────────────────────────────────────────────────────────────
# Router and fill order
# ─────────────────────────────────────────────────────────────────────

def predict_cell(df, row_idx, target_col, opt_type, cols_by_type,
                 strike_map, global_median_iv, already_filled,
                 alpha_table=None, global_alpha=0.0,
                 correction_clip=DEFAULT_CORRECTION_CLIP):
    row = df.loc[row_idx]
    edge, edge_reason, _, _, _ = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map)
    if edge:
        info = predict_edge_ensemble(
            df, row_idx, target_col, opt_type,
            cols_by_type, strike_map, global_median_iv, already_filled)
    else:
        info = predict_non_edge_with_underlying_signal(
            df, row_idx, target_col, opt_type,
            cols_by_type, strike_map, global_median_iv,
            alpha_table=alpha_table,
            global_alpha=global_alpha,
            correction_clip=correction_clip,
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
            interior = [c for c in state["column"].tolist()
                        if c in missing_side_cols and c not in edge_set]
            ordered = list(left_fill) + interior + [c for c in right_fill if c not in left_fill]
            for col in ordered:
                missing_cells.append((row_idx, col))
    return missing_cells


# ─────────────────────────────────────────────────────────────────────
# CV for underlying signal alpha
# ─────────────────────────────────────────────────────────────────────

def _fit_alpha(signals, residuals):
    s = np.asarray(signals, float)
    r = np.asarray(residuals, float)
    m = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > 1e-12)
    s, r = s[m], r[m]
    if len(s) < 2:
        return 0.0
    denom = float(np.dot(s, s))
    if denom <= 1e-18:
        return 0.0
    alpha = float(np.dot(s, r) / denom)
    return float(np.clip(alpha, -ALPHA_CLIP_ABS, ALPHA_CLIP_ABS))


def _apply_alpha_records(records_df, alpha_table, global_alpha, correction_clip):
    preds = []
    for r in records_df.itertuples(index=False):
        alpha = alpha_table.get((r.option_type, int(r.k_rank)), alpha_table.get((r.option_type, "ALL"), global_alpha))
        corr = correction_from_alpha(r.signal, alpha, correction_clip)
        preds.append(safe_iv(r.cross_pred + corr))
    return np.asarray(preds, float)


def grid_search_underlying_signal_alpha(df, option_cols, cols_by_type, strike_map, type_map,
                                        global_median_iv, out_path=None,
                                        mask_frac=UNDERLYING_SIGNAL_CV_MASK_FRAC,
                                        n_reps=UNDERLYING_SIGNAL_CV_REPS,
                                        seed=UNDERLYING_SIGNAL_CV_SEED,
                                        manual_clip=None):
    """
    Mask observed cells, keep only cells that are interior under the mask,
    compute cross-section baseline and signal, fit alpha per (option_type,k_rank),
    and CV-gate the correction.
    """
    rng = np.random.default_rng(seed)
    observed = [(r, c) for r in df.index for c in option_cols if pd.notna(df.at[r, c])]
    records = []

    for rep in range(n_reps):
        chosen_idx = rng.choice(len(observed), int(len(observed) * mask_frac), replace=False)
        mask_set = [observed[i] for i in chosen_idx]
        df_m = df.copy()
        truths = {}
        for r, c in mask_set:
            truths[(r, c)] = float(df.at[r, c])
            df_m.at[r, c] = np.nan

        for r, c in mask_set:
            ot = type_map[c]
            row = df_m.loc[r]
            edge, _, _, _, _ = is_edge_missing(row, c, ot, cols_by_type, strike_map)
            if edge:
                continue

            cross_info = predict_non_edge_cross_section(
                df_m, r, c, ot, cols_by_type, strike_map, global_median_iv
            )
            cross_pred = cross_info["prediction"]
            true = truths[(r, c)]
            signal, slope, dm = underlying_signal_for_cell(df_m, r, c, ot, cols_by_type, strike_map)
            if not (np.isfinite(cross_pred) and np.isfinite(true) and np.isfinite(signal)):
                continue

            k_rank = get_k_rank(c, cols_by_type, strike_map, ot)
            records.append({
                "rep": rep,
                "row_idx": r,
                "datetime": df.loc[r, "datetime"],
                "contract": c,
                "option_type": ot,
                "strike": strike_map[c],
                "k_rank": k_rank,
                "true": true,
                "cross_pred": cross_pred,
                "base_sq_err": (cross_pred - true) ** 2,
                "residual": true - cross_pred,
                "signal": signal,
                "slope": slope,
                "dmoneyness": dm,
                "pchip_used": bool(cross_info.get("pchip_used", False)),
            })

    val = pd.DataFrame(records)
    if val.empty:
        print("WARNING: Underlying-signal CV produced no records. Using zero correction.")
        return {}, 0.0, 0.0, val

    # Fit alpha using all CV records, bucketed by option_type+k_rank.
    global_alpha = _fit_alpha(val["signal"].values, val["residual"].values)
    alpha_table = {}

    for ot, sub in val.groupby("option_type"):
        alpha_table[(ot, "ALL")] = _fit_alpha(sub["signal"].values, sub["residual"].values)

    for (ot, kr), sub in val.groupby(["option_type", "k_rank"]):
        if len(sub) >= MIN_BUCKET_RECORDS:
            alpha_table[(ot, int(kr))] = _fit_alpha(sub["signal"].values, sub["residual"].values)

    # Choose correction clip by validation. Since alpha is already fit from this CV,
    # this is a useful but optimistic gate; if it fails, use zero correction.
    clips = [manual_clip] if manual_clip is not None else CORRECTION_CLIP_GRID
    summary_rows = []
    base_mse = float(val["base_sq_err"].mean())
    for clip in clips:
        preds = _apply_alpha_records(val, alpha_table, global_alpha, float(clip))
        new_mse = float(np.mean((preds - val["true"].values) ** 2))
        summary_rows.append({
            "correction_clip": float(clip),
            "base_mse": base_mse,
            "new_mse": new_mse,
            "improvement_pct": (base_mse - new_mse) / base_mse * 100 if base_mse > 0 else 0.0,
            "n_records": len(val),
            "n_alpha_buckets": len([k for k in alpha_table if k[1] != "ALL"]),
            "global_alpha": global_alpha,
        })

    summary = pd.DataFrame(summary_rows).sort_values("new_mse")
    best_clip = float(summary.iloc[0]["correction_clip"])
    best_mse = float(summary.iloc[0]["new_mse"])

    if out_path is not None:
        alpha_rows = []
        for key, alpha in alpha_table.items():
            alpha_rows.append({"option_type": key[0], "k_rank": key[1], "alpha": alpha})
        alpha_df = pd.DataFrame(alpha_rows)
        out_payload = val.copy()
        out_payload.to_csv(out_path, index=False)
        alpha_df.to_csv(str(out_path).replace(".csv", "_alphas.csv"), index=False)
        summary.to_csv(str(out_path).replace(".csv", "_summary.csv"), index=False)

    print("\n" + "─" * 76)
    print(f"Underlying signal CV ({n_reps} reps × {mask_frac*100:.0f}% mask, interior only)")
    print("─" * 76)
    print(summary.to_string(index=False))
    print("─" * 76)
    print(f"Best correction clip : {best_clip:.4f}")
    print(f"Baseline MSE         : {base_mse:.9f}")
    print(f"Best corrected MSE   : {best_mse:.9f}")
    if base_mse > 0:
        print(f"Improvement          : {(base_mse - best_mse) / base_mse * 100:.2f}%")
    print(f"Alpha buckets        : {len([k for k in alpha_table if k[1] != 'ALL'])}")
    print("─" * 76 + "\n")

    # CV gate: only use correction if it improves.
    if best_mse < base_mse:
        return alpha_table, global_alpha, best_clip, val

    print("CV gate rejected underlying signal correction. Using alpha=0 fallback.")
    return {}, 0.0, best_clip, val


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

    filled_out = out_dir / f"filled_dataset_{out_prefix}.csv"
    submission_out = out_dir / f"submission_{out_prefix}.csv"
    diagnostics_out = out_dir / f"diagnostics_{out_prefix}.csv"
    cross_diag_out = out_dir / f"cross_section_diagnostics_{out_prefix}.csv"
    signal_val_out = out_dir / f"underlying_signal_alpha_validation_{out_prefix}.csv"

    raw = pd.read_csv(data_path)
    df = raw.copy()
    df["datetime_parsed"] = pd.to_datetime(
        df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
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
    filled = df.copy()

    print(f"Timestamps        : {len(df)}")
    print(f"Missing cells     : {int(df[option_cols].isna().sum().sum())}")
    print(f"PCHIP available   : {PchipInterpolator is not None}")
    print(f"Interior PCHIP w  : {PCHIP_INTERIOR_WEIGHT}")
    print(f"Jan27 timestamps  : {int(df['datetime_parsed'].map(is_jan27_ts).sum())}")

    if args.skip_cv:
        alpha_table, global_alpha = {}, 0.0
        correction_clip = args.correction_clip if args.correction_clip is not None else DEFAULT_CORRECTION_CLIP
        print("Skipping underlying-signal CV; using zero correction.")
    else:
        print("Running underlying-price signal CV...")
        alpha_table, global_alpha, correction_clip, _ = grid_search_underlying_signal_alpha(
            df, option_cols, cols_by_type, strike_map, type_map,
            global_median_iv, out_path=signal_val_out,
            manual_clip=args.correction_clip,
        )

    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)
    filled_values_by_row = {}
    diag = {
        "missing_initial": len(missing_cells),
        "filled": 0,
        "fallback_global_median": 0,
        "pchip_used": 0,
        "underlying_signal_used": 0,
        "underlying_alpha_buckets": len([k for k in alpha_table if k[1] != "ALL"]),
        "global_alpha": global_alpha,
        "correction_clip": correction_clip,
        "missing_after": None,
    }
    rows = []

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs"):
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})
        info = predict_cell(
            df, row_idx, col, opt_type, cols_by_type,
            strike_map, global_median_iv, already_filled,
            alpha_table=alpha_table,
            global_alpha=global_alpha,
            correction_clip=correction_clip,
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
        if info.get("pchip_used", False):
            diag["pchip_used"] += 1
        if info.get("underlying_signal_used", False):
            diag["underlying_signal_used"] += 1
        if info["selected_model"] == "fallback_global_median":
            diag["fallback_global_median"] += 1

        rows.append({
            "row_index": row_idx,
            "datetime": df.loc[row_idx, "datetime"],
            "contract": col,
            "option_type": opt_type,
            "strike": strike_map[col],
            "k_rank": info.get("k_rank", get_k_rank(col, cols_by_type, strike_map, opt_type)),
            "final_prediction": pred,
            "edge": info["edge"],
            "edge_reason": info["edge_reason"],
            "source": info["source"],
            "selected_model": info["selected_model"],
            "bandwidth": info["bandwidth"],
            "loo_mse": info["loo_mse"],
            "edge_degree": info.get("edge_degree", np.nan),
            "pchip_used": info.get("pchip_used", False),
            "base_prediction": info.get("base_prediction", np.nan),
            "pchip_prediction": info.get("pchip_prediction", np.nan),
            "underlying_signal_used": info.get("underlying_signal_used", False),
            "underlying_signal": info.get("underlying_signal", np.nan),
            "underlying_slope": info.get("underlying_slope", np.nan),
            "dmoneyness": info.get("dmoneyness", np.nan),
            "underlying_alpha": info.get("underlying_alpha", np.nan),
            "underlying_correction": info.get("underlying_correction", np.nan),
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

    diag["missing_after"] = int(filled[option_cols].isna().sum().sum())

    filled_df = filled.drop(columns=["datetime_parsed"])
    orig_df = df.drop(columns=["datetime_parsed"])
    filled_df.to_csv(filled_out, index=False)
    sub = make_submission(orig_df, filled_df, submission_out)
    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(diagnostics_out, index=False)
    diag_df.to_csv(cross_diag_out, index=False)

    print(f"✅  Filled dataset → {filled_out}")
    print(f"✅  Submission     → {submission_out} ({len(sub)} rows)")
    print(f"✅  Diagnostics    → {diagnostics_out}")
    print(f"✅  Signal CV      → {signal_val_out}")
    print(f"    Missing after  : {diag['missing_after']}")
    if diag_df[diag_df.edge].shape[0] > 0:
        deg_counts = diag_df[diag_df.edge]["edge_degree"].value_counts().to_dict()
        print(f"    Edge degree chosen: {deg_counts}")
    print(f"    Interior PCHIP used       : {diag['pchip_used']}")
    print(f"    Underlying signal used    : {diag['underlying_signal_used']}")
    print(f"    Underlying alpha buckets  : {diag['underlying_alpha_buckets']}")
    print(f"    Global alpha              : {diag['global_alpha']:.6g}")
    print(f"    Correction clip           : {diag['correction_clip']}")
    print("\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
