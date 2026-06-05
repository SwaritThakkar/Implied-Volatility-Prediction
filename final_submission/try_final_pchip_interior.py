"""
try_final_pchip_interior.py
===========================

Same edge logic as try_final.py:
    edge predictions use the validated degree+bandwidth LOO selector.

Only interior / non-edge logic is changed:
    base_try_final = same-row local quadratic WLS from try.py
    pchip_pred     = same-row PCHIP interpolation, only when target is inside
                     the observed strike/moneyness range
    final interior = (1 - PCHIP_INTERIOR_WEIGHT) * base_try_final
                     + PCHIP_INTERIOR_WEIGHT * pchip_pred

Edges are not touched.
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
    "/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv"
)

EPS_IV    = 1e-6
SEPARATOR = "||"

BANDWIDTH_GRID     = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
EDGE_LOCAL_POLY_BW = 2e-4
LOCAL_POLY_DEGREE  = 2

EDGE_BLEND_CLAUDE    = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14
MIN_EDGE_LOCAL_NEIGHBORS = 3

# The only new interior change.
# Conservative because previous validation showed PCHIP helps only as a small blend.
PCHIP_INTERIOR_WEIGHT = 0.25
MIN_PCHIP_POINTS = 4


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

CFG_DATA       = str(DEFAULT_DATA_PATH)
CFG_OUT_PREFIX = "try_final_pchip_interior"

def parse_args():
    import types, sys
    if any('jupyter' in a or 'kernel' in a or '.json' in a for a in sys.argv[1:]):
        return types.SimpleNamespace(data=CFG_DATA, out_prefix=CFG_OUT_PREFIX)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, default=CFG_DATA)
    parser.add_argument("--out-prefix", type=str, default=CFG_OUT_PREFIX)
    parser.add_argument("--skip-cv", action="store_true", help="Skip internal CV validation.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Helpers
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
            uid = f"{original.loc[idx,'datetime']}{SEPARATOR}{col}"
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
    """Original try.py non-edge method: fixed degree=2, LOO over bandwidth only."""
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
    """Edge-only selector from try_final.py: choose degree in {1,2} and bandwidth by LOO."""
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
    """
    Interior-only PCHIP interpolation.
    Strictly no extrapolation: returns nan outside observed x range.
    """
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

    # Collapse duplicate x values, if any, by averaging y.
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
    same_obs  = set(state.loc[~state["is_missing"], "column"])
    if target_col in same_miss and len(same_obs) == 0:
        return True, "edge_no_observed_same_side", "all_missing", list(state["column"]), 0
    return False, "not_edge", "", [], np.nan


# ─────────────────────────────────────────────────────────────────────
# Non-edge prediction: try.py base + PCHIP blend
# ─────────────────────────────────────────────────────────────────────

def predict_non_edge_local_poly(df, row_idx, target_col, opt_type,
                                cols_by_type, strike_map, global_median_iv):
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {"prediction": float(global_median_iv),
                "source": "same_row_fallback_global_median",
                "selected_model": "fallback_global_median",
                "bandwidth": np.nan, "loo_mse": np.nan,
                "n_train": len(y_obs), "used_cols": used_cols,
                "edge_degree": np.nan,
                "pchip_used": False,
                "base_prediction": np.nan,
                "pchip_prediction": np.nan}

    x_target = strike_map[target_col] / spot

    # Original try.py interior prediction.
    best_bw, loo_mse = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    base_pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=LOCAL_POLY_DEGREE)

    if not np.isfinite(base_pred):
        base_pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "local_quadratic_wls"

    # New: PCHIP only for true interior interpolation.
    pchip_pred = pchip_same_row_pred(x_obs, y_obs, x_target)
    if np.isfinite(pchip_pred):
        pred = (1.0 - PCHIP_INTERIOR_WEIGHT) * base_pred + PCHIP_INTERIOR_WEIGHT * pchip_pred
        source = "same_row_non_edge_local_poly_wls_pchip_blend"
        selected_model = f"local_quad_wls_plus_pchip_w{PCHIP_INTERIOR_WEIGHT:.2f}"
        pchip_used = True
    else:
        pred = base_pred
        source = "same_row_non_edge_local_poly_wls"
        pchip_used = False

    return {"prediction": safe_iv(pred),
            "source": source,
            "selected_model": selected_model,
            "bandwidth": best_bw, "loo_mse": loo_mse,
            "n_train": len(y_obs), "used_cols": used_cols,
            "edge_degree": np.nan,
            "pchip_used": pchip_used,
            "base_prediction": safe_iv(base_pred),
            "pchip_prediction": pchip_pred if np.isfinite(pchip_pred) else np.nan}


# ─────────────────────────────────────────────────────────────────────
# Edge training-point collectors copied from try_final.py
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
    tgt_s  = strike_map[target_col]
    base_n = max(MIN_EDGE_LOCAL_NEIGHBORS, len(block_cols))
    state  = get_same_side_state(row, opt_type, cols_by_type, strike_map)
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


# ─────────────────────────────────────────────────────────────────────
# Edge predictors copied from try_final.py
# ─────────────────────────────────────────────────────────────────────

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
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return {"prediction": float(global_median_iv),
                "source": "edge_fallback_global_median_bad_spot",
                "selected_model": "fallback_global_median",
                "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
                "n_train": 0, "used_cols": [], "edge_degree": 1,
                "edge_side": "bad_spot", "edge_block_size": 0,
                "edge_position_in_block": np.nan,
                "pchip_used": False}

    x_obs, y_obs, used, edge_info = collect_edge_training_points_claude(
        row, target_col, opt_type, cols_by_type, strike_map, already_filled)

    if len(y_obs) == 0:
        return {"prediction": float(global_median_iv),
                "source": "edge_fallback_global_median_no_neighbors",
                "selected_model": "fallback_global_median",
                "bandwidth": EDGE_LOCAL_POLY_BW, "loo_mse": np.nan,
                "n_train": 0, "used_cols": used, "edge_degree": 1,
                "pchip_used": False,
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
            **edge_info}


def predict_edge_corrected_local_poly(df, row_idx, target_col, opt_type,
                                       cols_by_type, strike_map, global_median_iv, already_filled):
    row  = df.loc[row_idx]
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
    row  = df.loc[row_idx]
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
        "claude":    safe_iv(claude_pred),
        "corrected": safe_iv(corrected_pred),
        "quadratic": safe_iv(quadratic_pred),
    }
    pred = (EDGE_BLEND_CLAUDE    * components["claude"] +
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
                 strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    edge, edge_reason, _, _, _ = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map)
    if edge:
        info = predict_edge_ensemble(
            df, row_idx, target_col, opt_type,
            cols_by_type, strike_map, global_median_iv, already_filled)
    else:
        info = predict_non_edge_local_poly(
            df, row_idx, target_col, opt_type,
            cols_by_type, strike_map, global_median_iv)
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
# CV Validation
# ─────────────────────────────────────────────────────────────────────

def run_cv_validation(df, option_cols, cols_by_type, strike_map, type_map,
                      global_median_iv, mask_frac=0.08, seed=42, n_reps=5):
    rng = np.random.default_rng(seed)
    results = []
    observed = [(r, c) for r in df.index for c in option_cols if pd.notna(df.at[r, c])]

    for rep in range(n_reps):
        idx = rng.choice(len(observed), int(len(observed) * mask_frac), replace=False)
        mask_set = {observed[i] for i in idx}
        df_m = df.copy()
        truths = {}
        for r, c in mask_set:
            truths[(r, c)] = float(df.at[r, c])
            df_m.at[r, c] = np.nan
        for r, c in mask_set:
            ot = type_map[c]
            info = predict_cell(df_m, r, c, ot, cols_by_type, strike_map, global_median_iv, {})
            pred = info["prediction"]
            t = truths[(r, c)]
            row = df_m.loc[r]
            cols_s = sorted(cols_by_type[ot], key=lambda x: strike_map[x])
            obs_f = [pd.notna(row[col]) for col in cols_s]
            cidx = cols_s.index(c)
            is_edge = not any(obs_f[:cidx]) or not any(obs_f[cidx+1:])
            results.append({
                "sq_err": (pred - t) ** 2,
                "is_edge": is_edge,
                "pchip_used": bool(info.get("pchip_used", False)),
                "rep": rep,
            })

    df_r = pd.DataFrame(results)
    print(f"\n{'─'*55}")
    print(f"  CV Validation ({n_reps} reps × {mask_frac*100:.0f}% mask, {len(df_r)} predictions)")
    print(f"{'─'*55}")
    print(f"  Overall MSE  : {df_r.sq_err.mean():.8f}")
    for label, filt in [("edge", df_r.is_edge), ("interior", ~df_r.is_edge), ("pchip_used", df_r.pchip_used)]:
        sub = df_r[filt]
        if len(sub):
            print(f"  [{label:10s}] MSE : {sub.sq_err.mean():.8f}  (n={len(sub)})")
    print(f"{'─'*55}\n")
    return df_r


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_path  = Path(args.data)
    out_prefix = args.out_prefix
    out_dir    = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

    if not data_path.exists():
        raise FileNotFoundError(f"Input not found: {data_path.resolve()}")

    filled_out      = out_dir / f"filled_dataset_{out_prefix}.csv"
    submission_out  = out_dir / f"submission_{out_prefix}.csv"
    diagnostics_out = out_dir / f"diagnostics_{out_prefix}.csv"
    cross_diag_out  = out_dir / f"cross_section_diagnostics_{out_prefix}.csv"

    raw = pd.read_csv(data_path)
    df  = raw.copy()
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

    print(f"Timestamps     : {len(df)}")
    print(f"Missing cells  : {int(df[option_cols].isna().sum().sum())}")
    print(f"PCHIP available: {PchipInterpolator is not None}")
    print(f"Interior PCHIP weight: {PCHIP_INTERIOR_WEIGHT}")

    if not getattr(args, "skip_cv", False):
        print("Running CV validation...")
        run_cv_validation(df, option_cols, cols_by_type, strike_map, type_map,
                          global_median_iv, mask_frac=0.08, seed=42, n_reps=5)

    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)
    filled_values_by_row = {}
    diag = {"missing_initial": len(missing_cells), "filled": 0,
            "fallback_global_median": 0, "pchip_used": 0, "missing_after": None}
    rows = []

    for row_idx, col in tqdm(missing_cells, desc="Filling IVs"):
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})
        info = predict_cell(df, row_idx, col, opt_type, cols_by_type,
                            strike_map, global_median_iv, already_filled)
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
        if info["selected_model"] == "fallback_global_median":
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
            "pchip_used": info.get("pchip_used", False),
            "base_prediction": info.get("base_prediction", np.nan),
            "pchip_prediction": info.get("pchip_prediction", np.nan),
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
    print(f"    Missing after  : {diag['missing_after']}")
    if diag_df[diag_df.edge].shape[0] > 0:
        deg_counts = diag_df[diag_df.edge]["edge_degree"].value_counts().to_dict()
        print(f"    Edge degree chosen: {deg_counts}")
    print(f"    Interior PCHIP used: {diag['pchip_used']}")
    print("\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
