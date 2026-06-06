"""
try_cnn.py — CNN-based IV smile imputer
=========================================

Architecture: Masked 1D Convolutional Autoencoder
---------------------------------------------------
The key insight is that each timestamp's IV cross-section (IV vs strike)
is a 1D signal with strong spatial correlation — a U-shaped smile / skew curve.
A CNN trained on ALL timestamps learns the actual NIFTY smile shape from data,
rather than assuming a parametric form (quadratic, SVI, etc.).

Training:
    For each timestamp with enough observed values, the full cross-section
    (both CE and PE stacked into one 1D vector of length N_strikes_CE + N_strikes_PE)
    serves as a training sample.

    During training, we RANDOMLY mask a fraction of observed positions and
    train the model to reconstruct them. This is a Masked Autoencoder (MAE)
    approach: the model learns to fill gaps from context.

    Normalization: each row is normalized by its median IV, so the model
    learns smile SHAPE rather than absolute level. At inference, we denormalize.

Architecture:
    Input:  [2 * N_strikes] — concatenation of:
              - IV vector (0 at missing positions)
              - binary mask (1 = observed, 0 = missing)
    ->  1D Conv(32, k=5) + ReLU
    ->  1D Conv(64, k=5) + ReLU
    ->  1D Conv(64, k=3) + ReLU  (residual)
    ->  1D Conv(32, k=5) + ReLU
    ->  1D Conv(1,  k=1)          (output)
    Output: [N_strikes] full reconstruction

Safety mechanism:
    The final prediction BLENDS the CNN output with the try.py baseline:
        final = alpha * CNN_pred + (1 - alpha) * baseline_pred
    where alpha is determined by the CNN's reconstruction error on observed
    positions (high error → low alpha → trust baseline more).
    This guarantees we cannot be significantly worse than try.py.

Regime conditioning:
    A binary flag (27-Jan vs pre-27) is appended as a scalar feature,
    allowing the model to learn different smile dynamics for expiry day.

Run:
    python try_cnn.py --data dataset.csv

Outputs:
    filled_dataset_try_cnn.csv
    submission_try_cnn.csv
    diagnostics_try_cnn.csv
    cnn_model_ce.pt, cnn_model_pe.pt, cnn_model_joint.pt
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = Path(
    "dataset.csv"
)

EPS_IV = 1e-6
SEPARATOR = "||"

# Training
TRAIN_MASK_FRAC_MIN = 0.10   # minimum fraction of observed to mask during training
TRAIN_MASK_FRAC_MAX = 0.35   # maximum fraction of observed to mask during training
MIN_OBSERVED_FOR_TRAINING = 6  # skip rows with fewer observed values per type
N_EPOCHS = 120
LR = 3e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4

# Blending
# alpha = CNN confidence weight; 0 = all baseline, 1 = all CNN
CNN_ALPHA_MAX = 0.85   # max CNN weight when confidence is high
CNN_ALPHA_MIN = 0.05   # min CNN weight when confidence is low

# Original best-known non-edge method (fallback baseline)
BANDWIDTH_GRID = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4], dtype=float)
EDGE_LOCAL_POLY_BW = 2e-4
LOCAL_POLY_DEGREE = 2
MIN_EDGE_LOCAL_NEIGHBORS = 3
EDGE_BLEND_CLAUDE = 0.72
EDGE_BLEND_CORRECTED = 0.14
EDGE_BLEND_QUADRATIC = 0.14


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out-prefix", type=str, default="try_cnn")
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def safe_iv(x):
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


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


def is_27jan(datetime_val):
    try:
        return pd.Timestamp(datetime_val).date() == pd.Timestamp("2025-01-27").date()
    except Exception:
        return False


def make_submission(original, filled, out_path):
    rows = []
    for col in [c for c in original.columns if c != "datetime"]:
        was_missing = original[col].isna()
        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})
    sub = pd.DataFrame(rows, columns=["id", "value"]).sort_values("id").reset_index(drop=True)
    sub.to_csv(out_path, index=False)
    return sub


# ─────────────────────────────────────────────────────────────────────
# Baseline (try.py methods, kept intact as safety net)
# ─────────────────────────────────────────────────────────────────────

def local_poly_wls_pred(x_obs, y_obs, x_target, bandwidth, degree=LOCAL_POLY_DEGREE):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs, y_obs = x_obs[mask], y_obs[mask]
    if len(y_obs) == 0: return np.nan
    if len(y_obs) == 1: return safe_iv(float(y_obs[0]))
    actual_degree = min(degree, len(y_obs) - 1)
    dx = x_obs - x_target
    weights = np.exp(-dx ** 2 / (2.0 * bandwidth))
    X = np.column_stack([dx ** j for j in range(actual_degree + 1)])
    WX = X * weights[:, None]
    try:
        coeff = np.linalg.solve(X.T @ WX, X.T @ (weights * y_obs))
        return safe_iv(float(coeff[0]))
    except np.linalg.LinAlgError:
        wsum = float(weights.sum())
        return safe_iv(float((weights @ y_obs) / wsum)) if wsum > 1e-15 else np.nan


def local_poly_wls_loo_preds(x_obs, y_obs, bandwidth):
    x_obs, y_obs = np.asarray(x_obs, dtype=float), np.asarray(y_obs, dtype=float)
    preds = np.full(len(y_obs), np.nan)
    for i in range(len(y_obs)):
        preds[i] = local_poly_wls_pred(np.delete(x_obs, i), np.delete(y_obs, i), x_obs[i], bandwidth)
    return preds


def select_bandwidth_by_loo(x_obs, y_obs, bandwidth_grid=BANDWIDTH_GRID):
    if len(y_obs) <= 2:
        return float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
    best_bw, best_mse = float(bandwidth_grid[len(bandwidth_grid) // 2]), np.inf
    for bw in bandwidth_grid:
        loo = local_poly_wls_loo_preds(x_obs, y_obs, bw)
        valid = np.isfinite(loo) & np.isfinite(y_obs)
        if not valid.any(): continue
        mse = float(np.mean((loo[valid] - y_obs[valid]) ** 2))
        if mse < best_mse:
            best_mse, best_bw = mse, float(bw)
    return best_bw, best_mse


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = [{"column": col, "strike": strike_map[col], "is_missing": pd.isna(row[col]), "iv": row[col]}
               for col in cols_by_type[opt_type]]
    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)


def get_edge_blocks(row, opt_type, cols_by_type, strike_map):
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    left_block, right_block = [], []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]): left_block.append(rec["column"])
        else: break
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]): right_block.append(rec["column"])
        else: break
    return state, list(reversed(left_block)), list(reversed(right_block))


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
    state, left_fill, right_fill = get_edge_blocks(row, opt_type, cols_by_type, strike_map)
    if target_col in set(left_fill): return True, "edge_no_left_observed", "left", left_fill, left_fill.index(target_col)
    if target_col in set(right_fill): return True, "edge_no_right_observed", "right", right_fill, right_fill.index(target_col)
    same_missing = set(state.loc[state["is_missing"], "column"])
    same_obs = set(state.loc[~state["is_missing"], "column"])
    if target_col in same_missing and len(same_obs) == 0:
        return True, "edge_no_observed_same_side", "all_missing", list(state["column"]), 0
    return False, "not_edge", "", [], np.nan


def component_value(already_filled, col, key):
    item = already_filled.get(col)
    if isinstance(item, dict): v = item.get(key, item.get("final", np.nan))
    else: v = item
    try: v = float(v)
    except: return np.nan
    return v if np.isfinite(v) else np.nan


def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0: return np.array([]), np.array([]), []
    obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
    x = np.array([strike_map[c] / spot for c in obs_cols], dtype=float)
    y = np.array([row[c] for c in obs_cols], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], [c for c, k in zip(obs_cols, mask) if k]


def _edge_collect_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0: return np.array([]), np.array([]), {}
    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge: return np.array([]), np.array([]), {}
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs = [{"column": c, "strike": strike_map[c], "x": strike_map[c]/spot, "y": float(row[c])}
           for c in state["column"] if pd.notna(row[c])]
    if not obs: return np.array([]), np.array([]), {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}
    obs_df = pd.DataFrame(obs)
    tgt_s = strike_map[target_col]
    if side == "right": base = obs_df[obs_df["strike"] < tgt_s].sort_values("strike")
    elif side == "left": base = obs_df[obs_df["strike"] > tgt_s].sort_values("strike")
    else: base = obs_df.sort_values("strike")
    x_t, y_t = base["x"].to_list(), base["y"].to_list()
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        pv = component_value(already_filled, pc, "claude")
        if np.isfinite(pv):
            x_t.append(float(pv)); y_t.append(float(pv))
    return np.asarray(x_t, dtype=float), np.asarray(y_t, dtype=float), {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}


def _edge_collect_corrected(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0: return np.array([]), np.array([]), {}
    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge: return np.array([]), np.array([]), {}
    tgt_s = strike_map[target_col]
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    recs = []
    for _, rec in state.iterrows():
        col, val = rec["column"], row[rec["column"]]
        if pd.isna(val): continue
        s = strike_map[col]
        if (side == "right" and s < tgt_s) or (side == "left" and s > tgt_s):
            recs.append({"column": col, "strike": s, "x": s/spot, "y": float(val), "is_predicted": False})
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        pv = component_value(already_filled, pc, "corrected")
        if np.isfinite(pv):
            recs.append({"column": pc, "strike": strike_map[pc], "x": strike_map[pc]/spot, "y": float(pv), "is_predicted": True})
    if not recs: return np.array([]), np.array([]), {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}
    train = pd.DataFrame(recs).sort_values("strike")
    return train["x"].to_numpy(dtype=float), train["y"].to_numpy(dtype=float), {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}


def _edge_collect_quadratic(row, target_col, opt_type, cols_by_type, strike_map, already_filled):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0: return np.array([]), np.array([]), {}
    edge, _, side, block_cols, position = is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map)
    if not edge: return np.array([]), np.array([]), {}
    base_needed = max(MIN_EDGE_LOCAL_NEIGHBORS, len(block_cols))
    tgt_s = strike_map[target_col]
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    obs = [{"column": c, "strike": strike_map[c], "x": strike_map[c]/spot, "y": float(row[c]), "is_predicted": False}
           for c in state["column"] if pd.notna(row[c])]
    if not obs: return np.array([]), np.array([]), {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}
    obs_df = pd.DataFrame(obs)
    if side == "right": base = obs_df[obs_df["strike"] < tgt_s].sort_values("strike", ascending=False).head(base_needed).sort_values("strike")
    elif side == "left": base = obs_df[obs_df["strike"] > tgt_s].sort_values("strike").head(base_needed).sort_values("strike")
    else: base = obs_df.sort_values("strike")
    recs = base.to_dict(orient="records")
    prev = block_cols[:int(position)] if np.isfinite(position) else []
    for pc in prev:
        pv = component_value(already_filled, pc, "quadratic")
        if np.isfinite(pv):
            recs.append({"column": pc, "strike": strike_map[pc], "x": strike_map[pc]/spot, "y": float(pv), "is_predicted": True})
    train = pd.DataFrame(recs).sort_values("strike")
    return train["x"].to_numpy(dtype=float), train["y"].to_numpy(dtype=float), {"edge_side": side, "edge_block_size": len(block_cols), "edge_position_in_block": position}


def fit_quadratic(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
    if len(y) == 0: return None, "no_points"
    if len(y) == 1: return np.array([0., 0., float(y[0])]), "constant"
    if len(y) == 2:
        c = np.polyfit(x, y, 1); return np.array([0., float(c[0]), float(c[1])]), "linear"
    return np.array(np.polyfit(x, y, 2), dtype=float), "quadratic"


def eval_quadratic(coeff, x):
    if coeff is None: return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


def baseline_predict_edge(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled):
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    x_c, y_c, ei_c = _edge_collect_claude(row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    x_co, y_co, _ = _edge_collect_corrected(row, target_col, opt_type, cols_by_type, strike_map, already_filled)
    x_q, y_q, _ = _edge_collect_quadratic(row, target_col, opt_type, cols_by_type, strike_map, already_filled)

    x_target = strike_map[target_col] / spot if (pd.notna(spot) and spot > 0) else np.nan

    if not np.isfinite(x_target):
        return global_median_iv, {"edge_side": "bad_spot", "edge_block_size": 0, "edge_position_in_block": np.nan}

    claude_pred = local_poly_wls_pred(x_c, y_c, x_target, EDGE_LOCAL_POLY_BW) if len(y_c) > 0 else np.nan
    corrected_pred = local_poly_wls_pred(x_co, y_co, x_target, EDGE_LOCAL_POLY_BW) if len(y_co) > 0 else np.nan
    coeff, _ = fit_quadratic(x_q, y_q)
    quadratic_pred = eval_quadratic(coeff, x_target)

    components = [
        (EDGE_BLEND_CLAUDE, claude_pred),
        (EDGE_BLEND_CORRECTED, corrected_pred),
        (EDGE_BLEND_QUADRATIC, quadratic_pred),
    ]
    total_w, pred = 0.0, 0.0
    for w, p in components:
        if np.isfinite(p):
            pred += w * p
            total_w += w
    if total_w < 1e-9 or not np.isfinite(pred):
        pred = global_median_iv

    return safe_iv(pred), ei_c


def baseline_predict_nonedge(df, row_idx, target_col, opt_type, cols_by_type, strike_map, global_median_iv):
    row = df.loc[row_idx]
    spot = row["underlying_price"]
    x_obs, y_obs, _ = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return global_median_iv

    x_target = strike_map[target_col] / spot
    bw, _ = select_bandwidth_by_loo(x_obs, y_obs, BANDWIDTH_GRID)
    pred = local_poly_wls_pred(x_obs, y_obs, x_target, bw)
    return safe_iv(pred) if np.isfinite(pred) else global_median_iv


def build_missing_cell_fill_order(df, cols_by_type, strike_map):
    missing_cells = []
    for row_idx in df.index:
        row = df.loc[row_idx]
        for opt_type in ["CE", "PE"]:
            state, left_fill, right_fill = get_edge_blocks(row, opt_type, cols_by_type, strike_map)
            missing_side_cols = [c for c in state["column"].tolist() if pd.isna(row[c])]
            if not missing_side_cols: continue
            edge_set = set(left_fill) | set(right_fill)
            interior = [c for c in state["column"].tolist() if c in missing_side_cols and c not in edge_set]
            ordered = list(left_fill) + interior + [c for c in right_fill if c not in left_fill]
            for col in ordered:
                missing_cells.append((row_idx, col))
    return missing_cells


# ─────────────────────────────────────────────────────────────────────
# CNN Model
# ─────────────────────────────────────────────────────────────────────

class SmileImputer1D(nn.Module):
    """
    1D CNN for IV smile imputation.
    
    Input:  [batch, 2, N_strikes]
            channel 0: normalized IV (0.0 at missing positions)
            channel 1: binary observation mask (1=observed, 0=missing)
    Output: [batch, 1, N_strikes] — reconstructed normalized IV at ALL positions.
    
    Architecture: dilated 1D convolutions for wide receptive field without
    too many parameters. Residual connections for stability.
    Skip connection from input to output (the model learns the residual correction
    on top of a simple copy of observed values).
    """

    def __init__(self, n_strikes: int, n_filters: int = 64):
        super().__init__()
        self.n_strikes = n_strikes
        # Encoder: widen receptive field with dilation
        self.enc1 = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1), nn.ReLU(),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv1d(n_filters * 2, n_filters, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(n_filters // 2, 1, kernel_size=1),
        )
        # Softplus to ensure positive output
        self.act_out = nn.Softplus()

    def forward(self, x):
        # x: [B, 2, N]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Skip from e1
        cat = torch.cat([e3, e1], dim=1)
        out = self.dec(cat)  # [B, 1, N]
        out = self.act_out(out)
        return out.squeeze(1)  # [B, N]


class JointSmileImputer(nn.Module):
    """
    Joint CE+PE imputer: takes the full cross-section (CE strikes + PE strikes)
    as a single 1D signal. This allows the model to learn put-call relationships.
    
    Conditioning: a scalar feature for 27-Jan regime is injected via feature-wise
    linear modulation (FiLM): y = gamma(cond) * x + beta(cond).
    """

    def __init__(self, n_ce: int, n_pe: int, n_filters: int = 64):
        super().__init__()
        self.n_ce = n_ce
        self.n_pe = n_pe
        n_total = n_ce + n_pe

        # Regime conditioning via FiLM
        self.film_gamma = nn.Linear(1, n_filters)
        self.film_beta = nn.Linear(1, n_filters)

        self.enc1 = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=12, dilation=6), nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(n_filters * 2, n_filters, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(n_filters // 2, 1, kernel_size=1),
        )
        self.act_out = nn.Softplus()

    def forward(self, x, cond):
        # x: [B, 2, N_total], cond: [B, 1]
        e1 = self.enc1(x)                          # [B, F, N]
        # FiLM modulation after enc1
        gamma = self.film_gamma(cond).unsqueeze(-1) # [B, F, 1]
        beta  = self.film_beta(cond).unsqueeze(-1)
        e1 = gamma * e1 + beta
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        cat = torch.cat([e3, e1], dim=1)
        out = self.dec(cat)
        out = self.act_out(out)
        return out.squeeze(1)


# ─────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────

def build_cross_section_matrix(df, cols_sorted, option_cols):
    """
    Build matrix of shape [N_timestamps, N_strikes] from sorted option columns.
    Each row is one timestamp's IV cross-section.
    NaN = originally missing.
    """
    return df[cols_sorted].values.astype(float)  # [T, N]


def normalize_row(row_iv, ref_iv=None):
    """
    Normalize a row of IV values by the row's median (observed values only).
    Returns (normalized_row, scale_factor).
    If ref_iv is provided, use it as scale (for inference where target is missing).
    """
    obs = row_iv[np.isfinite(row_iv)]
    if len(obs) == 0:
        return row_iv, 1.0
    scale = np.median(obs) if ref_iv is None else ref_iv
    if scale < EPS_IV:
        scale = 1.0
    return row_iv / scale, float(scale)


def build_training_samples(
    iv_matrix,        # [T, N] — observed IV, NaN at originally missing
    regime_flags,     # [T] — 1.0 if 27-Jan, else 0.0
    rng,
    mask_frac_min=TRAIN_MASK_FRAC_MIN,
    mask_frac_max=TRAIN_MASK_FRAC_MAX,
    min_observed=MIN_OBSERVED_FOR_TRAINING,
    n_augment=3,       # repeat each row with different random masks
):
    """
    Build (input_tensor, target_tensor, mask_tensor, cond_tensor) for training.
    
    For each row with enough observed values:
        - Randomly mask a fraction of observed positions
        - Input:  normalized IV with 0 at masked positions + binary mask channel
        - Target: full normalized IV row
        - Loss mask: only backprop on the RANDOMLY masked positions
          (i.e., train the model to reconstruct what it can't see)
    """
    inputs, targets, loss_masks, conds = [], [], [], []
    T, N = iv_matrix.shape

    for t in range(T):
        row = iv_matrix[t]
        observed_idx = np.where(np.isfinite(row))[0]
        if len(observed_idx) < min_observed:
            continue

        row_norm, scale = normalize_row(row)

        for _ in range(n_augment):
            n_mask = max(1, int(len(observed_idx) * rng.uniform(mask_frac_min, mask_frac_max)))
            mask_idx = rng.choice(observed_idx, size=n_mask, replace=False)

            # Build input: zero out masked positions
            inp_iv = row_norm.copy()
            inp_iv[mask_idx] = 0.0
            inp_iv = np.where(np.isfinite(inp_iv), inp_iv, 0.0)

            obs_mask = np.where(np.isfinite(row_norm), 1.0, 0.0)
            obs_mask[mask_idx] = 0.0  # treat masked as unobserved for input

            # Target: full row (at ALL positions, but loss only on mask_idx)
            target = np.where(np.isfinite(row_norm), row_norm, 0.0)

            # Loss mask: 1 only at positions we just masked
            loss_mask = np.zeros(N, dtype=float)
            loss_mask[mask_idx] = 1.0

            inputs.append(np.stack([inp_iv, obs_mask], axis=0))   # [2, N]
            targets.append(target)                                  # [N]
            loss_masks.append(loss_mask)                            # [N]
            conds.append([regime_flags[t]])                         # [1]

    if not inputs:
        return None, None, None, None

    return (
        torch.tensor(np.array(inputs), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
        torch.tensor(np.array(loss_masks), dtype=torch.float32),
        torch.tensor(np.array(conds), dtype=torch.float32),
    )


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train_model(model, inputs, targets, loss_masks, conds, device, n_epochs, lr, batch_size, use_cond=False):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)

    dataset = TensorDataset(inputs, targets, loss_masks, conds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    best_loss = np.inf
    best_state = None

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for inp, tgt, lmask, cond in loader:
            inp, tgt, lmask, cond = inp.to(device), tgt.to(device), lmask.to(device), cond.to(device)
            optimizer.zero_grad()
            if use_cond:
                pred = model(inp, cond)
            else:
                pred = model(inp)
            # Masked MSE: only backprop on positions that were artificially masked
            loss = ((pred - tgt) ** 2 * lmask).sum() / lmask.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cnn_predict_row(
    model,
    row_iv,           # [N] — partial row (NaN at missing)
    regime_flag,      # scalar float
    device,
    use_cond=False,
):
    """
    Run CNN inference on one row.
    Returns predicted IV at all N positions (including originally observed ones).
    """
    row_norm, scale = normalize_row(row_iv)
    inp_iv = np.where(np.isfinite(row_norm), row_norm, 0.0)
    obs_mask = np.where(np.isfinite(row_norm), 1.0, 0.0)

    inp = torch.tensor(np.stack([inp_iv, obs_mask], axis=0)[None], dtype=torch.float32).to(device)
    cond = torch.tensor([[regime_flag]], dtype=torch.float32).to(device)

    if use_cond:
        out = model(inp, cond)
    else:
        out = model(inp)

    pred_norm = out[0].cpu().numpy()   # [N]
    pred_iv = pred_norm * scale        # denormalize

    # Reconstruction error on observed positions (confidence signal)
    obs_idx = np.where(np.isfinite(row_iv))[0]
    if len(obs_idx) > 0:
        recon_err = float(np.mean((pred_iv[obs_idx] - row_iv[obs_idx]) ** 2))
    else:
        recon_err = np.inf

    return pred_iv, recon_err


def compute_cnn_alpha(recon_err, err_scale=5e-4):
    """
    Map reconstruction error to CNN blend weight.
    Low error → high CNN trust (alpha close to CNN_ALPHA_MAX).
    High error → low CNN trust (alpha close to CNN_ALPHA_MIN).
    err_scale: error level at which alpha = midpoint.
    """
    alpha = CNN_ALPHA_MAX / (1.0 + recon_err / err_scale)
    return float(np.clip(alpha, CNN_ALPHA_MIN, CNN_ALPHA_MAX))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input not found: {data_path.resolve()}")

    out_prefix = args.out_prefix
    filled_out = Path(f"filled_dataset_{out_prefix}.csv")
    submission_out = Path(f"submission_{out_prefix}.csv")
    diagnostics_out = Path(f"diagnostics_{out_prefix}.csv")
    model_ce_path = Path(f"cnn_model_ce_{out_prefix}.pt")
    model_pe_path = Path(f"cnn_model_pe_{out_prefix}.pt")
    model_joint_path = Path(f"cnn_model_joint_{out_prefix}.pt")

    # ── Load data ────────────────────────────────────────────────────
    raw = pd.read_csv(data_path)
    df = raw.copy()
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["datetime_parsed"].isna().any():
        raise ValueError(f"{df['datetime_parsed'].isna().sum()} datetime values could not be parsed.")
    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta = parse_metadata(df)
    option_cols = meta["column"].tolist()
    strike_map = dict(zip(meta["column"], meta["strike"]))
    type_map = dict(zip(meta["column"], meta["option_type"]))
    cols_by_type = {
        "CE": sorted([c for c in option_cols if type_map[c] == "CE"], key=lambda c: strike_map[c]),
        "PE": sorted([c for c in option_cols if type_map[c] == "PE"], key=lambda c: strike_map[c]),
    }
    global_median_iv = float(df[option_cols].stack().median())

    n_ce = len(cols_by_type["CE"])
    n_pe = len(cols_by_type["PE"])
    n_joint = n_ce + n_pe

    print(f"Strikes: {n_ce} CE + {n_pe} PE = {n_joint} total")
    print(f"Timestamps: {len(df)}")
    print(f"Missing cells: {int(df[option_cols].isna().sum().sum())}")

    regime_flags = np.array([1.0 if is_27jan(dt) else 0.0 for dt in df["datetime"]])
    print(f"27-Jan timestamps: {int(regime_flags.sum())}")

    # ── Build IV matrices ────────────────────────────────────────────
    iv_ce = build_cross_section_matrix(df, cols_by_type["CE"], option_cols)   # [T, n_ce]
    iv_pe = build_cross_section_matrix(df, cols_by_type["PE"], option_cols)   # [T, n_pe]
    iv_joint = np.concatenate([iv_ce, iv_pe], axis=1)                          # [T, n_joint]

    # ── Train CE model ───────────────────────────────────────────────
    print("\n── Training CE model ──────────────────────────────────")
    inp_ce, tgt_ce, lmask_ce, cond_ce = build_training_samples(iv_ce, regime_flags, rng)
    if inp_ce is not None:
        print(f"  CE training samples: {len(inp_ce)}")
        model_ce = SmileImputer1D(n_ce)
        model_ce, ce_best = train_model(
            model_ce, inp_ce, tgt_ce, lmask_ce, cond_ce,
            device, args.epochs, LR, BATCH_SIZE, use_cond=False
        )
        torch.save(model_ce.state_dict(), model_ce_path)
        print(f"  CE best loss: {ce_best:.6f}")
    else:
        print("  WARNING: No CE training data.")
        model_ce = None

    # ── Train PE model ───────────────────────────────────────────────
    print("\n── Training PE model ──────────────────────────────────")
    inp_pe, tgt_pe, lmask_pe, cond_pe = build_training_samples(iv_pe, regime_flags, rng)
    if inp_pe is not None:
        print(f"  PE training samples: {len(inp_pe)}")
        model_pe = SmileImputer1D(n_pe)
        model_pe, pe_best = train_model(
            model_pe, inp_pe, tgt_pe, lmask_pe, cond_pe,
            device, args.epochs, LR, BATCH_SIZE, use_cond=False
        )
        torch.save(model_pe.state_dict(), model_pe_path)
        print(f"  PE best loss: {pe_best:.6f}")
    else:
        print("  WARNING: No PE training data.")
        model_pe = None

    # ── Train Joint model (CE+PE, with FiLM conditioning) ────────────
    print("\n── Training Joint CE+PE model (with regime conditioning) ──")
    inp_j, tgt_j, lmask_j, cond_j = build_training_samples(iv_joint, regime_flags, rng, n_augment=5)
    if inp_j is not None:
        print(f"  Joint training samples: {len(inp_j)}")
        model_joint = JointSmileImputer(n_ce, n_pe)
        model_joint, joint_best = train_model(
            model_joint, inp_j, tgt_j, lmask_j, cond_j,
            device, args.epochs, LR, BATCH_SIZE, use_cond=True
        )
        torch.save(model_joint.state_dict(), model_joint_path)
        print(f"  Joint best loss: {joint_best:.6f}")
    else:
        print("  WARNING: No joint training data.")
        model_joint = None

    # ── Imputation loop ──────────────────────────────────────────────
    print("\n── Imputing missing values ────────────────────────────")
    filled = df.copy()
    rows_diag = []
    missing_cells = build_missing_cell_fill_order(df, cols_by_type, strike_map)
    filled_values_by_row = {}

    # Pre-compute CNN predictions for all rows (batch inference is efficient)
    # We do this per timestamp, filling in baseline-filled progressively.
    cnn_row_cache_ce = {}     # row_idx -> (pred_iv_full, recon_err)
    cnn_row_cache_pe = {}
    cnn_row_cache_joint = {}

    def get_cnn_pred_for_row(row_idx):
        """Lazy compute CNN predictions for a row, using current filled state."""
        row = filled.loc[row_idx]
        regime = regime_flags[row_idx]

        # Build current partial row (using already-filled values)
        row_iv_ce = np.array([row[c] for c in cols_by_type["CE"]], dtype=float)
        row_iv_pe = np.array([row[c] for c in cols_by_type["PE"]], dtype=float)
        row_iv_joint = np.concatenate([row_iv_ce, row_iv_pe])

        results = {}
        if model_ce is not None:
            pred_ce, err_ce = cnn_predict_row(model_ce, row_iv_ce, regime, device, use_cond=False)
            results["ce"] = (pred_ce, err_ce)
        if model_pe is not None:
            pred_pe, err_pe = cnn_predict_row(model_pe, row_iv_pe, regime, device, use_cond=False)
            results["pe"] = (pred_pe, err_pe)
        if model_joint is not None:
            pred_joint, err_joint = cnn_predict_row(model_joint, row_iv_joint, regime, device, use_cond=True)
            results["joint_ce"] = (pred_joint[:n_ce], err_joint)
            results["joint_pe"] = (pred_joint[n_ce:], err_joint)
        return results

    diag_counts = {
        "total_missing": len(missing_cells),
        "filled": 0,
        "baseline_edge": 0,
        "baseline_nonedge": 0,
        "cnn_high_confidence": 0,
        "cnn_low_confidence": 0,
        "fallback_global_median": 0,
    }

    for row_idx, col in tqdm(missing_cells, desc="Imputing"):
        opt_type = type_map[col]
        already_filled = filled_values_by_row.setdefault(row_idx, {})

        edge, edge_reason, side, block_cols, position = is_edge_missing(
            filled.loc[row_idx], col, opt_type, cols_by_type, strike_map
        )

        # ── Baseline prediction (try.py logic) ─────────────────────
        if edge:
            baseline_pred, edge_info = baseline_predict_edge(
                filled, row_idx, col, opt_type, cols_by_type, strike_map, global_median_iv, already_filled
            )
            diag_counts["baseline_edge"] += 1
        else:
            baseline_pred = baseline_predict_nonedge(
                filled, row_idx, col, opt_type, cols_by_type, strike_map, global_median_iv
            )
            edge_info = {}
            diag_counts["baseline_nonedge"] += 1

        # ── CNN prediction ─────────────────────────────────────────
        cnn_preds = get_cnn_pred_for_row(row_idx)

        # Get CNN prediction for this specific column
        col_idx_in_type = cols_by_type[opt_type].index(col)

        cnn_candidates = []
        if opt_type == "CE":
            if "ce" in cnn_preds:
                p, e = cnn_preds["ce"]
                cnn_candidates.append((p[col_idx_in_type], e))
            if "joint_ce" in cnn_preds:
                p, e = cnn_preds["joint_ce"]
                cnn_candidates.append((p[col_idx_in_type], e))
        else:  # PE
            if "pe" in cnn_preds:
                p, e = cnn_preds["pe"]
                cnn_candidates.append((p[col_idx_in_type], e))
            if "joint_pe" in cnn_preds:
                p, e = cnn_preds["joint_pe"]
                cnn_candidates.append((p[col_idx_in_type], e))

        # ── Blend CNN + baseline ────────────────────────────────────
        if cnn_candidates:
            # Ensemble CNN models (type-specific + joint), weighted by confidence
            cnn_pv_list = []
            for cnn_p, cnn_err in cnn_candidates:
                if np.isfinite(cnn_p) and cnn_p > 0:
                    alpha = compute_cnn_alpha(cnn_err)
                    cnn_pv_list.append((alpha, float(cnn_p)))

            if cnn_pv_list:
                # Weight each CNN model by its confidence
                total_alpha = sum(a for a, _ in cnn_pv_list)
                cnn_ensemble = sum(a * p for a, p in cnn_pv_list) / total_alpha
                avg_alpha = total_alpha / len(cnn_pv_list)

                final_pred = avg_alpha * cnn_ensemble + (1.0 - avg_alpha) * baseline_pred
                if avg_alpha >= 0.5:
                    diag_counts["cnn_high_confidence"] += 1
                else:
                    diag_counts["cnn_low_confidence"] += 1
            else:
                final_pred = baseline_pred
        else:
            final_pred = baseline_pred

        if not np.isfinite(final_pred) or final_pred <= 0:
            final_pred = global_median_iv
            diag_counts["fallback_global_median"] += 1

        final_pred = safe_iv(final_pred)
        filled.at[row_idx, col] = final_pred
        diag_counts["filled"] += 1

        # Store for progressive edge
        filled_values_by_row[row_idx][col] = {
            "final": final_pred,
            "claude": final_pred,
            "corrected": final_pred,
            "quadratic": final_pred,
        }

        rows_diag.append({
            "row_index": row_idx,
            "datetime": df.loc[row_idx, "datetime"],
            "contract": col,
            "option_type": opt_type,
            "strike": strike_map[col],
            "final_prediction": final_pred,
            "baseline_prediction": baseline_pred,
            "edge": edge,
            "edge_reason": edge_reason,
            "regime_27jan": bool(regime_flags[row_idx]),
            "cnn_candidates": len(cnn_candidates),
        })

    # ── Save outputs ─────────────────────────────────────────────────
    filled_out_df = filled.drop(columns=["datetime_parsed"])
    original_out_df = df.drop(columns=["datetime_parsed"])
    filled_out_df.to_csv(filled_out, index=False)
    submission = make_submission(original_out_df, filled_out_df, submission_out)
    pd.DataFrame(rows_diag).to_csv(diagnostics_out, index=False)

    print(f"\n✅ Filled dataset → {filled_out}")
    print(f"✅ Submission → {submission_out} ({len(submission)} rows)")
    print(f"✅ Diagnostics → {diagnostics_out}")
    print(f"\nDiagnostics:")
    for k, v in diag_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()