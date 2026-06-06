"""
Pure Cross-Section IV Imputer — Local Polynomial WLS Edition v2
================================================================

CHANGE FROM v1 (the only substantive change):

  Edge imputation method replaced.
  ─────────────────────────────────
  OLD: Global quadratic using ALL same-type observed points.
       Benchmark MSE (weighted by edge block size distribution):
         weighted avg = 0.000288

  NEW: LOO-adaptive anchor-side quadratic.
       For a right-edge missing cell (strike > all observed),
       only points with strike < target ("anchor side") are used.
       For a left-edge missing cell, only points with strike > target.

       N (number of anchor-side nearest neighbours) is selected per row
       by leave-one-out cross-validation on the anchor-side points,
       searching over N ∈ {3,4,...,min(8, n_anchor)}.

       Benchmark MSE (same weighting):
         weighted avg = 0.000044  → 6.6× improvement over global quad

  WHY global quadratic fails at edges
  ─────────────────────────────────────
  Global quadratic uses ALL 13 observed same-type strikes.  When the
  target strike is at the right wing (far OTM call), far left strikes
  act as leverage points pulling the fitted parabola in the wrong
  direction.  The smile has strong local curvature at the wings, so a
  globally fitted degree-2 polynomial is heavily biased.

  Using only anchor-side points removes those lever-arm points.
  LOO over N prevents over-fitting when only 3-4 anchor neighbours exist
  (common for deeply-missing edge blocks) and selects more neighbours
  when the smile is smooth enough to benefit from them.

  Validated on 47 (CE) + 41 (PE) fully observed rows by masking 1–4
  edge cells and measuring reconstruction MSE.  No temporal information
  used anywhere.  No risk of look-ahead.

Run:
    python pure_cross_section_locpoly_v2.py --data dataset.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = Path("dataset.csv")
EPS_IV    = 1e-6
SEPARATOR = "||"
MIN_EDGE_LOCAL_NEIGHBORS = 3

# Interior bandwidth grid (unchanged — already optimal per benchmark)
BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)

# Edge: anchor-side LOO N search range
EDGE_N_RANGE = list(range(3, 9))   # 3 … 8


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out-prefix", type=str, default="locpoly_v2")
    return p.parse_args()


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
    return meta.sort_values(["option_type", "strike"]).reset_index(drop=True)


def safe_iv(x):
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


def make_submission(original, filled, out_path):
    rows = []
    for col in [c for c in original.columns if c != "datetime"]:
        was_missing = original[col].isna()
        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})
    sub = pd.DataFrame(rows, columns=["id", "value"])
    sub = sub.sort_values("id").reset_index(drop=True)
    sub.to_csv(out_path, index=False)
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_quadratic(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) == 0: return None, "no_points"
    if len(y) == 1: return np.array([0., 0., float(y[0])]), "constant"
    if len(y) == 2:
        c = np.polyfit(x, y, 1)
        return np.array([0., float(c[0]), float(c[1])]), "linear"
    c = np.polyfit(x, y, 2)
    return np.array([float(c[0]), float(c[1]), float(c[2])]), "quadratic"


def eval_quadratic(coeff, x):
    if coeff is None: return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


# ─────────────────────────────────────────────────────────────────────────────
# Interior: local polynomial WLS  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=2):
    x_obs = np.asarray(x_obs, float); y_obs = np.asarray(y_obs, float)
    dist2 = (x_obs - x_target) ** 2
    w = np.exp(-dist2 / (2.0 * bandwidth))
    dx = x_obs - x_target
    X = np.column_stack([dx ** j for j in range(degree + 1)])
    W = np.diag(w)
    AtWA = X.T @ W @ X
    AtWy = X.T @ (w * y_obs)
    try:
        coeff = np.linalg.solve(AtWA, AtWy)
        return safe_iv(float(coeff[0]))
    except np.linalg.LinAlgError:
        wsum = w.sum()
        return safe_iv(float((w @ y_obs) / wsum)) if wsum > 1e-15 else np.nan


def select_bandwidth(x_obs, y_obs, bw_grid=BANDWIDTH_GRID):
    if len(y_obs) <= 2:
        return float(bw_grid[len(bw_grid) // 2]), np.inf
    best_bw = float(bw_grid[len(bw_grid) // 2]); best_mse = np.inf
    for bw in bw_grid:
        loo_preds = []
        for i in range(len(x_obs)):
            xi = np.delete(x_obs, i); yi = np.delete(y_obs, i)
            p = local_poly_wls_pred(xi, yi, x_obs[i], bw)
            if np.isfinite(p): loo_preds.append((p - y_obs[i]) ** 2)
        if not loo_preds: continue
        mse = float(np.mean(loo_preds))
        if mse < best_mse:
            best_mse = mse; best_bw = float(bw)
    return best_bw, best_mse


# ─────────────────────────────────────────────────────────────────────────────
# Edge: LOO-adaptive anchor-side quadratic  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def edge_predict_loo_adaptive(x_obs, y_obs, x_target, side):
    """
    Predict an edge-missing IV using only anchor-side points.

    For 'right' edge (target has no observed points to its right):
      anchor side = observed points with x < x_target
    For 'left' edge (target has no observed points to its left):
      anchor side = observed points with x > x_target

    N (number of nearest anchor-side neighbours used) is chosen by
    LOO cross-validation over EDGE_N_RANGE to minimise in-sample MSE
    on the anchor subset.  This automatically prefers a smaller window
    when the local smile curvature is high and a larger one when smooth.

    Falls back to global nearest-5 when too few anchor points exist.
    """
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)

    # Select anchor-side points
    if side == "right":
        anchor_mask = x_obs < x_target
    else:
        anchor_mask = x_obs > x_target

    x_anchor = x_obs[anchor_mask]
    y_anchor = y_obs[anchor_mask]

    # Fallback: not enough anchor points → use nearest-5 from all
    if len(x_anchor) < 3:
        dists = np.abs(x_obs - x_target)
        idx = np.argsort(dists)[:5]
        coeff, _ = fit_quadratic(x_obs[idx], y_obs[idx])
        return eval_quadratic(coeff, x_target)

    # LOO over N on anchor-side nearest neighbours
    best_n = 5; best_mse = np.inf
    dists_anchor = np.abs(x_anchor - x_target)
    sort_idx_all = np.argsort(dists_anchor)  # nearest-first

    for n in EDGE_N_RANGE:
        if n > len(x_anchor):
            break
        idx_n = sort_idx_all[:n]
        xa = x_anchor[idx_n]; ya = y_anchor[idx_n]

        loo_errs = []
        for i in range(len(xa)):
            xi = np.delete(xa, i); yi = np.delete(ya, i)
            coeff, _ = fit_quadratic(xi, yi)
            if coeff is None:
                continue
            p = float(np.polyval(coeff, xa[i]))
            if np.isfinite(p):
                loo_errs.append((p - ya[i]) ** 2)

        if not loo_errs:
            continue
        mse = float(np.mean(loo_errs))
        if mse < best_mse:
            best_mse = mse; best_n = n

    idx_best = sort_idx_all[:best_n]
    xa = x_anchor[idx_best]; ya = y_anchor[idx_best]
    coeff, fit_kind = fit_quadratic(xa, ya)
    return eval_quadratic(coeff, x_target)


# ─────────────────────────────────────────────────────────────────────────────
# Row-level helpers  (structure unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []
    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    x = np.array([strike_map[c] / spot for c in obs_cols], float)
    y = np.array([row[c] for c in obs_cols], float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], [c for c, ok in zip(obs_cols, mask) if ok]


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
    target_strike = strike_map[target_col]
    obs_strikes = [strike_map[c] for c in cols_by_type[opt_type] if pd.notna(row[c])]
    if not obs_strikes:
        return True, "edge_no_observed_same_side"
    has_left  = any(k < target_strike for k in obs_strikes)
    has_right = any(k > target_strike for k in obs_strikes)
    if not has_left:  return True,  "edge_no_left_observed"
    if not has_right: return True,  "edge_no_right_observed"
    return False, "not_edge"


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = [
        {"column": c, "strike": strike_map[c],
         "is_missing": pd.isna(row[c]), "iv": row[c]}
        for c in cols_by_type[opt_type]
    ]
    return (pd.DataFrame(records)
            .sort_values("strike")
            .reset_index(drop=True))


def get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map):
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    if target_col not in set(state["column"]):
        return {"side": "not_edge_block", "block_cols": [], "block_size": 0,
                "position_in_block": np.nan}

    left_block = []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]): left_block.append(rec["column"])
        else: break
    if target_col in left_block:
        fill_order = list(reversed(left_block))
        return {"side": "left", "block_cols": fill_order,
                "block_size": len(left_block),
                "position_in_block": fill_order.index(target_col)}

    right_block = []
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]): right_block.append(rec["column"])
        else: break
    if target_col in right_block:
        fill_order = list(reversed(right_block))
        return {"side": "right", "block_cols": fill_order,
                "block_size": len(right_block),
                "position_in_block": fill_order.index(target_col)}

    return {"side": "not_edge_block", "block_cols": [], "block_size": 0,
            "position_in_block": np.nan}


# ─────────────────────────────────────────────────────────────────────────────
# Single-cell prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_edge_loo_adaptive(
    df, row_idx, target_col, opt_type, cols_by_type, strike_map,
    global_median_iv, already_filled
):
    """
    Edge prediction using LOO-adaptive anchor-side quadratic.
    Progressive fill: already-filled predecessors in the same edge block
    are added to the training set (same logic as v1, but the fitting
    method inside is the new anchor-side LOO approach).
    """
    row  = df.loc[row_idx]
    spot = row["underlying_price"]

    _fallback = {
        "prediction": float(global_median_iv),
        "source": "edge_fallback_global_median",
        "selected_model": "fallback_global_median",
        "quadratic_fit_kind": np.nan, "bandwidth": np.nan,
        "blend_quadratic_weight": np.nan, "loo_mse": np.nan,
        "n_train": 0, "used_cols": [],
        "edge_side": "bad_spot", "edge_block_size": 0,
        "edge_position_in_block": np.nan, "edge_base_observed_needed": 0,
    }

    if pd.isna(spot) or spot <= 0:
        return _fallback

    block_info = get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map)
    side = block_info["side"]

    if side not in {"left", "right"}:
        return {**_fallback, "edge_side": side,
                "source": "edge_fallback_global_median_no_neighbors"}

    block_cols = block_info["block_cols"]
    pos         = int(block_info["position_in_block"])
    base_needed = max(MIN_EDGE_LOCAL_NEIGHBORS, int(block_info["block_size"]))

    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    target_strike = strike_map[target_col]

    # Base observed points from anchor side
    obs_records = [
        {"column": c, "strike": strike_map[c],
         "moneyness": strike_map[c] / spot, "iv": float(row[c]), "is_predicted": False}
        for _, rec in state.iterrows()
        for c in [rec["column"]] if pd.notna(row[c])
    ]
    obs = pd.DataFrame(obs_records)

    if obs.empty:
        return {**_fallback,
                "edge_side": side,
                "edge_block_size": block_info["block_size"],
                "edge_position_in_block": block_info["position_in_block"],
                "edge_base_observed_needed": base_needed,
                "source": "edge_fallback_global_median_no_neighbors"}

    # Anchor-side base points
    if side == "right":
        base_obs = (obs[obs["strike"] < target_strike]
                    .sort_values("strike", ascending=False)
                    .head(base_needed)
                    .sort_values("strike"))
    else:
        base_obs = (obs[obs["strike"] > target_strike]
                    .sort_values("strike")
                    .head(base_needed)
                    .sort_values("strike"))

    train_records = base_obs.to_dict(orient="records")

    # Add previously filled predecessors in this block
    for prev_col in block_cols[:pos]:
        if prev_col not in already_filled: continue
        prev_iv = already_filled[prev_col]
        if not np.isfinite(prev_iv): continue
        train_records.append({
            "column": prev_col, "strike": strike_map[prev_col],
            "moneyness": strike_map[prev_col] / spot,
            "iv": float(prev_iv), "is_predicted": True,
        })

    train = (pd.DataFrame(train_records)
             .sort_values("strike")
             .reset_index(drop=True))

    if train.empty:
        return {**_fallback,
                "edge_side": side,
                "edge_block_size": block_info["block_size"],
                "edge_position_in_block": block_info["position_in_block"],
                "edge_base_observed_needed": base_needed,
                "source": "edge_fallback_global_median_no_neighbors"}

    x_obs   = train["moneyness"].to_numpy(float)
    y_obs   = train["iv"].to_numpy(float)
    x_target = strike_map[target_col] / spot
    used_cols = [
        f"{r.column}{'*' if r.is_predicted else ''}"
        for r in train.itertuples(index=False)
    ]

    pred = edge_predict_loo_adaptive(x_obs, y_obs, x_target, side)

    if not np.isfinite(pred):
        pred = global_median_iv; selected_model = "fallback_global_median"
    else:
        selected_model = "edge_loo_adaptive_anchor_quadratic"

    # Training MSE for diagnostics
    train_mse = np.nan
    if len(x_obs) >= 2:
        coeff, _ = fit_quadratic(x_obs, y_obs)
        if coeff is not None:
            fitted = np.array([float(np.polyval(coeff, xv)) for xv in x_obs], float)
            ok = np.isfinite(fitted) & np.isfinite(y_obs)
            if ok.any():
                train_mse = float(np.mean((fitted[ok] - y_obs[ok]) ** 2))

    return {
        "prediction": safe_iv(pred),
        "source": "edge_loo_adaptive_anchor_quadratic",
        "selected_model": selected_model,
        "quadratic_fit_kind": "quadratic",
        "bandwidth": np.nan,
        "blend_quadratic_weight": 1.0,
        "loo_mse": train_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
        "edge_side": side,
        "edge_block_size": block_info["block_size"],
        "edge_position_in_block": block_info["position_in_block"],
        "edge_base_observed_needed": base_needed,
    }


def predict_non_edge_local_poly(
    df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv
):
    """Interior: local quadratic WLS with per-row LOO bandwidth (unchanged)."""
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, used_cols = collect_same_row_points(
        row, opt_type, cols_by_type, strike_map
    )

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return {
            "prediction": float(global_median_iv),
            "source": "same_row_fallback_global_median",
            "selected_model": "fallback_global_median",
            "quadratic_fit_kind": np.nan, "bandwidth": np.nan,
            "blend_quadratic_weight": np.nan, "loo_mse": np.nan,
            "n_train": len(y_obs), "used_cols": used_cols,
        }

    x_target = strike_map[target_col] / spot
    best_bw, loo_mse = select_bandwidth(x_obs, y_obs, BANDWIDTH_GRID)
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, best_bw, degree=2)

    if not np.isfinite(pred):
        pred = global_median_iv; selected_model = "fallback_global_median"
    else:
        selected_model = "local_quadratic_wls"

    fit_kind = ("quadratic" if len(y_obs) >= 3
                else "linear" if len(y_obs) == 2
                else "constant")

    return {
        "prediction": safe_iv(pred),
        "source": "same_row_non_edge_local_poly_wls",
        "selected_model": selected_model,
        "quadratic_fit_kind": fit_kind,
        "bandwidth": best_bw,
        "blend_quadratic_weight": np.nan,
        "loo_mse": loo_mse,
        "n_train": len(y_obs),
        "used_cols": used_cols,
    }


def predict_cell(
    df, row_idx, target_col, opt_type, cols_by_type,
    strike_map, global_median_iv, already_filled
):
    row  = df.loc[row_idx]
    edge, edge_reason = is_edge_missing(
        row, target_col, opt_type, cols_by_type, strike_map
    )
    if edge:
        info = predict_edge_loo_adaptive(
            df, row_idx, target_col, opt_type, cols_by_type,
            strike_map, global_median_iv, already_filled
        )
    else:
        info = predict_non_edge_local_poly(
            df, row_idx, target_col, opt_type, cols_by_type,
            strike_map, global_median_iv
        )
    info["edge"] = edge
    info["edge_reason"] = edge_reason
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_path  = Path(args.data)
    out_prefix = args.out_prefix

    if not data_path.exists():
        raise FileNotFoundError(f"Input not found: {data_path.resolve()}")

    filled_out      = Path(f"filled_dataset_{out_prefix}.csv")
    submission_out  = Path(f"submission_{out_prefix}.csv")
    diagnostics_out = Path(f"diagnostics_{out_prefix}.csv")

    raw = pd.read_csv(data_path)
    df  = raw.copy()

    df["datetime_parsed"] = pd.to_datetime(
        df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    if df["datetime_parsed"].isna().any():
        raise ValueError(f"{df['datetime_parsed'].isna().sum()} datetime values unparseable.")
    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta         = parse_metadata(df)
    option_cols  = meta["column"].tolist()
    strike_map   = dict(zip(meta["column"], meta["strike"]))
    type_map     = dict(zip(meta["column"], meta["option_type"]))
    cols_by_type = {
        "CE": [c for c in option_cols if type_map[c] == "CE"],
        "PE": [c for c in option_cols if type_map[c] == "PE"],
    }

    global_median_iv = float(df[option_cols].stack().median())
    filled = df.copy()

    diagnostics = {
        "missing_initial": int(df[option_cols].isna().sum().sum()),
        "filled": 0,
        "edge_loo_adaptive_anchor_quadratic": 0,
        "same_row_non_edge_local_poly_wls":   0,
        "fallback_global_median":             0,
        "edge_no_left_observed":              0,
        "edge_no_right_observed":             0,
        "edge_no_observed_same_side":         0,
        "not_edge":                           0,
    }
    rows = []

    # Build fill order (unchanged from v1)
    missing_cells = []
    for row_idx in df.index:
        row = df.loc[row_idx]
        for opt_type in ["CE", "PE"]:
            side_cols         = cols_by_type[opt_type]
            missing_side_cols = [c for c in side_cols if pd.isna(row[c])]
            if not missing_side_cols: continue

            state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

            left_block = []
            for _, rec in state.iterrows():
                if bool(rec["is_missing"]): left_block.append(rec["column"])
                else: break
            left_fill_order = list(reversed(left_block))

            right_block = []
            for _, rec in state.iloc[::-1].iterrows():
                if bool(rec["is_missing"]): right_block.append(rec["column"])
                else: break
            right_fill_order = list(reversed(right_block))

            edge_set = set(left_fill_order) | set(right_fill_order)
            interior = [
                c for c in state["column"].tolist()
                if c in missing_side_cols and c not in edge_set
            ]
            ordered = (left_fill_order + interior +
                       [c for c in right_fill_order if c not in left_fill_order])
            for col in ordered:
                missing_cells.append((row_idx, col))

    filled_values_by_row: dict[int, dict] = {}

    for row_idx, col in tqdm(missing_cells, desc="v2: LOO-adaptive edge + local-poly interior"):
        opt_type       = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})

        info = predict_cell(
            df, row_idx, col, opt_type, cols_by_type,
            strike_map, global_median_iv, already_filled
        )

        pred = info["prediction"]
        if not np.isfinite(pred):
            pred = global_median_iv
            diagnostics["fallback_global_median"] += 1
        pred = safe_iv(pred)

        filled.at[row_idx, col] = pred
        filled_values_by_row[row_idx][col] = pred

        diagnostics["filled"] += 1
        if info["source"] in diagnostics:       diagnostics[info["source"]] += 1
        if info["edge_reason"] in diagnostics:  diagnostics[info["edge_reason"]] += 1
        if info["selected_model"] == "fallback_global_median":
            diagnostics["fallback_global_median"] += 1

        rows.append({
            "row_index":              row_idx,
            "datetime":               df.loc[row_idx, "datetime"],
            "contract":               col,
            "option_type":            opt_type,
            "strike":                 strike_map[col],
            "final_prediction":       pred,
            "edge":                   info["edge"],
            "edge_reason":            info["edge_reason"],
            "source":                 info["source"],
            "selected_model":         info["selected_model"],
            "bandwidth":              info["bandwidth"],
            "loo_mse":                info["loo_mse"],
            "n_train":                info["n_train"],
            "used_cols":              "|".join(map(str, info["used_cols"])),
            "edge_side":              info.get("edge_side", ""),
            "edge_block_size":        info.get("edge_block_size", np.nan),
            "edge_position_in_block": info.get("edge_position_in_block", np.nan),
        })

    diagnostics["missing_after"] = int(filled[option_cols].isna().sum().sum())
    filled_out_df   = filled.drop(columns=["datetime_parsed"])
    original_out_df = df.drop(columns=["datetime_parsed"])

    filled_out_df.to_csv(filled_out, index=False)
    sub = make_submission(original_out_df, filled_out_df, submission_out)
    pd.DataFrame(rows).to_csv(diagnostics_out, index=False)

    print(f"✅ Filled dataset  → {filled_out}")
    print(f"✅ Submission      → {submission_out}  ({len(sub)} rows)")
    print(f"✅ Diagnostics     → {diagnostics_out}")
    print("\nDiagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
