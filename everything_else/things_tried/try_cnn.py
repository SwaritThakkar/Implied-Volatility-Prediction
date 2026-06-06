"""
try_cnn.py  —  Walk-forward CNN IV imputer with no-arbitrage penalties
=======================================================================

Two issues fixed vs original:
  1. Forward bias removed via strict walk-forward training split.
  2. Architecture and training improved using ideas from Hui (2023).

─────────────────────────────────────────────────────────────────────
Forward-bias fix
─────────────────────────────────────────────────────────────────────
The original trained on ALL timestamps and then predicted all of them.
That means when predicting timestamp t the model had already seen the
smile shapes at t+1, t+2, … — a pure look-ahead leak.

Fix: for every timestamp t that has at least one missing cell, the CNN
is trained ONLY on timestamps s < t (strictly causal). Because
retraining per-missing-timestamp is too expensive, we do a
WALK-FORWARD EXPANDING WINDOW with a small number of fixed cutpoints:

  - Collect all unique timestamps that have ≥1 missing cell.
  - Sort them chronologically → prediction set P = [p0, p1, …, pM].
  - Divide P into K folds.
  - For fold k (covers timestamps p_{k*B} … p_{(k+1)*B-1}):
      training data = all timestamps STRICTLY BEFORE p_{k*B}
      inference     = timestamps in this fold

This guarantees zero look-ahead. We train K models total (K=4 default),
each on a growing prefix of the timeline.

─────────────────────────────────────────────────────────────────────
Hui (2023) improvements applied
─────────────────────────────────────────────────────────────────────
1. SVI synthetic pretraining
   Hui trains on Heston-generated surfaces to bootstrap with a large
   synthetic dataset before seeing real data. We use SVI (Gatheral)
   parametric surfaces instead — same idea, no numerical integration,
   faster and more stable. Generate 20,000 random SVI surfaces that
   span the real data's parameter range, pretrain the model on these,
   then fine-tune on real past data.

2. No-arbitrage butterfly penalty (Hui Eq. 3.1 / Algorithm 3)
   Add soft constraint to training loss:
       L = L_MSE + λ_bf * L_butterfly
   where L_butterfly = mean(ReLU(−∂²w/∂k²)) penalises non-convex
   total-variance profiles (butterfly spread violations).

3. Iterative inpainting inference (Hui Algorithm 3)
   At inference time, instead of one forward pass, run N_REFINE steps:
       pred = model(input)
       input.iv[observed] = true_iv[observed]   ← clamp back to truth
       repeat
   This enforces consistency with observed data and lets the model
   self-correct across iterations.

4. Architecture: dilated 1D U-Net with FiLM regime conditioning.
   Same as before but with the padding bug fixed (kernel=3 → padding=1).

─────────────────────────────────────────────────────────────────────
Run
─────────────────────────────────────────────────────────────────────
    python try_cnn.py --data dataset.csv

Outputs:  filled_dataset_try_cnn.csv  submission_try_cnn.csv
          diagnostics_try_cnn.csv     cnn_model_fold*.pt
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
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = Path(
    "dataset.csv"
)

EPS_IV   = 1e-6
SEPARATOR = "||"

# Walk-forward folds (number of CNN models trained)
N_FOLDS = 4
# Minimum training timestamps before first fold (skip fold if not enough history)
MIN_TRAIN_TIMESTAMPS = 30

# SVI synthetic pretraining
N_SVI_PRETRAIN   = 20_000   # synthetic surfaces
N_PRETRAIN_EPOCHS = 40
PRETRAIN_LR       = 3e-3

# Real-data fine-tuning per fold
N_FINETUNE_EPOCHS = 80
FINETUNE_LR       = 1e-3
BATCH_SIZE        = 64
WEIGHT_DECAY      = 1e-4

# Training augmentation
TRAIN_MASK_FRAC_MIN = 0.10
TRAIN_MASK_FRAC_MAX = 0.40
MIN_OBSERVED        = 5      # skip rows with fewer observed strikes

# No-arbitrage penalty weight (Hui 2023)
LAMBDA_BUTTERFLY = 0.05

# Iterative inpainting refinement steps (Hui Algorithm 3)
N_REFINE_STEPS = 8

# Blending: alpha=1 → pure CNN, alpha=0 → pure baseline
CNN_ALPHA_MAX = 0.85
CNN_ALPHA_MIN = 0.05

# Baseline (try.py)
BANDWIDTH_GRID       = np.array([5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4])
EDGE_LOCAL_POLY_BW   = 2e-4
LOCAL_POLY_DEGREE    = 2
MIN_EDGE_NEIGHBORS   = 3
EDGE_W_CLAUDE        = 0.72
EDGE_W_CORRECTED     = 0.14
EDGE_W_QUADRATIC     = 0.14


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out-prefix", type=str, default="try_cnn")
    p.add_argument("--epochs",     type=int, default=N_FINETUNE_EPOCHS)
    p.add_argument("--folds",      type=int, default=N_FOLDS)
    p.add_argument("--no-cuda",    action="store_true")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────

def safe_iv(x):
    if not np.isfinite(x): return np.nan
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
        if col in {"datetime", "datetime_parsed", "underlying_price"}: continue
        m = pattern.match(col)
        if m:
            item = m.groupdict()
            item["column"] = col
            item["strike"] = int(item["strike"])
            item["expiry_date"] = pd.to_datetime(item["expiry"], format="%d%b%y", errors="coerce")
            records.append(item)
    meta = pd.DataFrame(records)
    if meta.empty: raise ValueError("No option columns found.")
    return meta.sort_values(["option_type", "strike"]).reset_index(drop=True)


def is_27jan(dt_val):
    try:
        return pd.Timestamp(dt_val).date() == pd.Timestamp("2025-01-27").date()
    except: return False


def make_submission(original, filled, out_path):
    rows = []
    for col in [c for c in original.columns if c != "datetime"]:
        for idx in original.index[original[col].isna()]:
            uid = f"{original.loc[idx,'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})
    sub = pd.DataFrame(rows, columns=["id","value"]).sort_values("id").reset_index(drop=True)
    sub.to_csv(out_path, index=False)
    return sub


def normalize_row(row_iv):
    """Divide by median of observed values. Returns (normed, scale)."""
    obs = row_iv[np.isfinite(row_iv)]
    if len(obs) == 0: return row_iv.copy(), 1.0
    scale = float(np.median(obs))
    if scale < EPS_IV: scale = 1.0
    return row_iv / scale, scale


# ─────────────────────────────────────────────────────────────────────
# Baseline (try.py logic — unchanged, used as safety net)
# ─────────────────────────────────────────────────────────────────────

def local_poly_wls(x_obs, y_obs, x_tgt, bw, degree=LOCAL_POLY_DEGREE):
    x_obs = np.asarray(x_obs, float); y_obs = np.asarray(y_obs, float)
    m = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs, y_obs = x_obs[m], y_obs[m]
    if len(y_obs) == 0: return np.nan
    if len(y_obs) == 1: return safe_iv(y_obs[0])
    d = min(degree, len(y_obs)-1)
    dx = x_obs - x_tgt
    w  = np.exp(-dx**2 / (2*bw))
    X  = np.column_stack([dx**j for j in range(d+1)])
    try:
        coeff = np.linalg.solve(X.T @ (X*w[:,None]), X.T @ (w*y_obs))
        return safe_iv(float(coeff[0]))
    except np.linalg.LinAlgError:
        ws = w.sum()
        return safe_iv(float((w@y_obs)/ws)) if ws > 1e-15 else np.nan


def loo_mse(x_obs, y_obs, bw):
    preds = [local_poly_wls(np.delete(x_obs,i), np.delete(y_obs,i), x_obs[i], bw)
             for i in range(len(x_obs))]
    p = np.array(preds)
    ok = np.isfinite(p) & np.isfinite(y_obs)
    return float(np.mean((p[ok]-y_obs[ok])**2)) if ok.any() else np.inf


def best_bw(x_obs, y_obs):
    if len(y_obs) <= 2: return BANDWIDTH_GRID[len(BANDWIDTH_GRID)//2], np.inf
    best, bmse = BANDWIDTH_GRID[len(BANDWIDTH_GRID)//2], np.inf
    for bw in BANDWIDTH_GRID:
        mse = loo_mse(x_obs, y_obs, bw)
        if mse < bmse: bmse, best = mse, bw
    return best, bmse


def get_state(row, opt_type, cols_by_type, strike_map):
    recs = [{"col": c, "strike": strike_map[c],
             "missing": pd.isna(row[c]), "iv": row[c]}
            for c in cols_by_type[opt_type]]
    return pd.DataFrame(recs).sort_values("strike").reset_index(drop=True)


def edge_blocks(row, opt_type, cols_by_type, strike_map):
    st = get_state(row, opt_type, cols_by_type, strike_map)
    left, right = [], []
    for _, r in st.iterrows():
        if r["missing"]: left.append(r["col"])
        else: break
    for _, r in st.iloc[::-1].iterrows():
        if r["missing"]: right.append(r["col"])
        else: break
    return st, list(reversed(left)), list(reversed(right))


def is_edge(row, col, opt_type, cols_by_type, strike_map):
    st, lf, rf = edge_blocks(row, opt_type, cols_by_type, strike_map)
    if col in set(lf): return True, "left",  lf, lf.index(col)
    if col in set(rf): return True, "right", rf, rf.index(col)
    miss = set(st.loc[st["missing"],"col"])
    obs  = set(st.loc[~st["missing"],"col"])
    if col in miss and not obs: return True, "all_missing", list(st["col"]), 0
    return False, "interior", [], np.nan


def comp_val(already_filled, col, key):
    it = already_filled.get(col)
    v  = it.get(key, it.get("final", np.nan)) if isinstance(it, dict) else it
    try: v = float(v)
    except: return np.nan
    return v if np.isfinite(v) else np.nan


def _edge_pts(row, col, opt_type, cols_by_type, strike_map, already_filled, mode):
    """Collect training points for edge prediction (three modes: claude/corrected/quad)."""
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0: return np.array([]), np.array([])
    is_e, side, block, pos = is_edge(row, col, opt_type, cols_by_type, strike_map)
    if not is_e: return np.array([]), np.array([])

    st = get_state(row, opt_type, cols_by_type, strike_map)
    tgt_s = strike_map[col]

    if mode == "claude":
        obs = [{"x": strike_map[c]/spot, "y": float(row[c])}
               for c in st["col"] if pd.notna(row[c])]
        if not obs: return np.array([]), np.array([])
        df_obs = pd.DataFrame(obs)
        x_t, y_t = df_obs["x"].tolist(), df_obs["y"].tolist()
        prev = block[:int(pos)] if np.isfinite(pos) else []
        for pc in prev:
            pv = comp_val(already_filled, pc, "claude")
            if np.isfinite(pv): x_t.append(pv); y_t.append(pv)
        return np.array(x_t), np.array(y_t)

    if mode == "corrected":
        recs = []
        for _, r in st.iterrows():
            c, val = r["col"], row[r["col"]]
            if pd.isna(val): continue
            s = strike_map[c]
            if (side=="right" and s < tgt_s) or (side=="left" and s > tgt_s):
                recs.append({"x": s/spot, "y": float(val)})
        prev = block[:int(pos)] if np.isfinite(pos) else []
        for pc in prev:
            pv = comp_val(already_filled, pc, "corrected")
            if np.isfinite(pv): recs.append({"x": strike_map[pc]/spot, "y": float(pv)})
        if not recs: return np.array([]), np.array([])
        df_r = pd.DataFrame(recs).sort_values("x")
        return df_r["x"].values, df_r["y"].values

    if mode == "quad":
        needed = max(MIN_EDGE_NEIGHBORS, len(block))
        obs = [{"col": c, "strike": strike_map[c], "x": strike_map[c]/spot, "y": float(row[c])}
               for c in st["col"] if pd.notna(row[c])]
        if not obs: return np.array([]), np.array([])
        df_obs = pd.DataFrame(obs)
        if side == "right":
            base = df_obs[df_obs["strike"]<tgt_s].sort_values("strike",ascending=False).head(needed).sort_values("strike")
        elif side == "left":
            base = df_obs[df_obs["strike"]>tgt_s].sort_values("strike").head(needed).sort_values("strike")
        else:
            base = df_obs.sort_values("strike")
        recs = base.to_dict("records")
        prev = block[:int(pos)] if np.isfinite(pos) else []
        for pc in prev:
            pv = comp_val(already_filled, pc, "quad")
            if np.isfinite(pv): recs.append({"x": strike_map[pc]/spot, "y": float(pv)})
        df_r = pd.DataFrame(recs).sort_values("x") if recs else pd.DataFrame()
        if df_r.empty: return np.array([]), np.array([])
        return df_r["x"].values, df_r["y"].values

    return np.array([]), np.array([])


def baseline_predict(df, row_idx, col, opt_type, cols_by_type, strike_map, gm, already_filled):
    row  = df.loc[row_idx]
    spot = row["underlying_price"]
    is_e, side, block, pos = is_edge(row, col, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0: return gm

    x_tgt = strike_map[col] / spot

    if is_e:
        preds = []
        for mode, w in [("claude", EDGE_W_CLAUDE), ("corrected", EDGE_W_CORRECTED), ("quad", EDGE_W_QUADRATIC)]:
            xo, yo = _edge_pts(row, col, opt_type, cols_by_type, strike_map, already_filled, mode)
            if len(yo) == 0: continue
            if mode == "quad":
                mask = np.isfinite(xo) & np.isfinite(yo)
                xo, yo = xo[mask], yo[mask]
                if len(yo) == 0: continue
                try:
                    coeff = np.polyfit(xo, yo, min(2, len(yo)-1))
                    p = safe_iv(float(np.polyval(coeff, x_tgt)))
                except: p = np.nan
            else:
                p = local_poly_wls(xo, yo, x_tgt, EDGE_LOCAL_POLY_BW)
            if np.isfinite(p): preds.append((w, p))
        if not preds: return gm
        tw = sum(w for w,_ in preds)
        return safe_iv(sum(w*p for w,p in preds) / tw)
    else:
        obs_cols = [c for c in cols_by_type[opt_type] if pd.notna(row[c])]
        xo = np.array([strike_map[c]/spot for c in obs_cols])
        yo = np.array([float(row[c]) for c in obs_cols])
        m  = np.isfinite(xo) & np.isfinite(yo)
        if not m.any(): return gm
        bw, _ = best_bw(xo[m], yo[m])
        p = local_poly_wls(xo[m], yo[m], x_tgt, bw)
        return safe_iv(p) if np.isfinite(p) else gm


def fill_order(df, cols_by_type, strike_map):
    cells = []
    for idx in df.index:
        row = df.loc[idx]
        for ot in ["CE","PE"]:
            st, lf, rf = edge_blocks(row, ot, cols_by_type, strike_map)
            miss = [c for c in st["col"] if pd.isna(row[c])]
            if not miss: continue
            eset = set(lf)|set(rf)
            interior = [c for c in st["col"] if c in miss and c not in eset]
            for c in lf + interior + [c for c in rf if c not in lf]:
                cells.append((idx, c))
    return cells


# ─────────────────────────────────────────────────────────────────────
# SVI synthetic surface generator (for pretraining, from Hui 2023 §3)
# ─────────────────────────────────────────────────────────────────────

def svi_surface(log_moneyness, a, b, rho, m, sigma, T):
    """Gatheral SVI total-variance → IV surface. Always positive."""
    k   = log_moneyness - m
    w   = a + b * (rho*k + np.sqrt(k**2 + sigma**2))
    w   = np.maximum(w, 1e-8)
    return np.sqrt(w / max(T, 1e-4))


def generate_svi_surfaces(n_surfaces, n_strikes, rng, T_range=(5/365, 60/365)):
    """
    Sample random SVI parameters, return matrix [n_surfaces, n_strikes].
    Parameters sampled to match typical NIFTY smile range.
    """
    # Log-moneyness grid: covers ±15% of spot
    log_m = np.linspace(-0.15, 0.15, n_strikes)

    # SVI parameter ranges calibrated to typical NIFTY short-term smiles
    a_      = rng.uniform(0.005, 0.12,  n_surfaces)   # ATM total variance
    b_      = rng.uniform(0.01,  0.25,  n_surfaces)   # wing slope
    rho_    = rng.uniform(-0.95, 0.0,   n_surfaces)   # skew (equity: negative)
    m_      = rng.uniform(-0.04, 0.04,  n_surfaces)   # smile centre shift
    sigma_  = rng.uniform(0.03,  0.35,  n_surfaces)   # smile curvature
    T_      = rng.uniform(*T_range,     n_surfaces)

    surfaces = np.zeros((n_surfaces, n_strikes), dtype=np.float32)
    for i in range(n_surfaces):
        ivs = svi_surface(log_m, a_[i], b_[i], rho_[i], m_[i], sigma_[i], T_[i])
        # Clip to sane range [0.01, 3.0]
        surfaces[i] = np.clip(ivs, 0.01, 3.0).astype(np.float32)
    return surfaces


# ─────────────────────────────────────────────────────────────────────
# CNN model
# ─────────────────────────────────────────────────────────────────────

class SmileImputer(nn.Module):
    """
    Dilated 1D U-Net for IV smile imputation.

    Input:  [B, 2, N]  — channel 0: normalized IV (0 at missing)
                          channel 1: binary obs mask
    Cond:   [B, 1]     — 0/1 regime flag (injected via FiLM)
    Output: [B, N]     — reconstructed normalized IV (all positions)

    All convolutions use padding = dilation*(kernel-1)//2 to preserve
    sequence length exactly (no truncation or padding artefacts).
    """

    def __init__(self, n: int, f: int = 64):
        super().__init__()
        # FiLM conditioning (regime flag → scale + shift of bottleneck)
        self.film_g = nn.Linear(1, f)
        self.film_b = nn.Linear(1, f)

        # Encoder  (preserves length at every layer)
        self.e1 = self._block(2,  f, k=5, d=1)   # receptive field: 5
        self.e2 = self._block(f,  f, k=5, d=2)   # rf: 9
        self.e3 = self._block(f,  f, k=3, d=4)   # rf: 17
        self.e4 = self._block(f,  f, k=3, d=8)   # rf: 33
        # Decoder  (skip connections from encoder)
        self.d3 = self._block(f*2, f, k=3, d=4)
        self.d2 = self._block(f*2, f, k=5, d=2)
        self.d1 = self._block(f*2, f, k=5, d=1)
        self.out = nn.Conv1d(f, 1, kernel_size=1)
        self.act = nn.Softplus()

    @staticmethod
    def _block(cin, cout, k, d):
        p = d * (k - 1) // 2   # same-length padding
        return nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=p, dilation=d),
            nn.GroupNorm(min(8, cout), cout),
            nn.ReLU(),
        )

    def forward(self, x, cond):
        e1 = self.e1(x)                              # [B, f, N]
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        # FiLM modulation at bottleneck
        g  = self.film_g(cond).unsqueeze(-1)         # [B, f, 1]
        b  = self.film_b(cond).unsqueeze(-1)
        e4 = g * e4 + b
        # Decoder with skip connections
        d3 = self.d3(torch.cat([e4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        d1 = self.d1(torch.cat([d2, e1], dim=1))
        return self.act(self.out(d1)).squeeze(1)     # [B, N]


# ─────────────────────────────────────────────────────────────────────
# No-arbitrage butterfly penalty (Hui 2023 §3.1)
# ─────────────────────────────────────────────────────────────────────

def butterfly_loss(pred_iv, T: float = 30/365):
    """
    Penalise violation of convexity of total variance in strike space.
    TV(k) = IV(k)^2 * T must be convex in log-moneyness k.
    Discrete approximation: second difference of TV must be ≥ 0.
    pred_iv: [B, N]
    """
    tv   = pred_iv ** 2 * T
    d2   = tv[:, :-2] - 2 * tv[:, 1:-1] + tv[:, 2:]
    return torch.mean(torch.relu(-d2))


# ─────────────────────────────────────────────────────────────────────
# Dataset builder (shared for both pretraining and fine-tuning)
# ─────────────────────────────────────────────────────────────────────

def make_dataset(iv_matrix, regime_flags, rng, n_aug=3,
                 mask_min=TRAIN_MASK_FRAC_MIN, mask_max=TRAIN_MASK_FRAC_MAX,
                 min_obs=MIN_OBSERVED):
    """
    iv_matrix: [T, N]   — real IV rows (NaN at originally missing)
                          OR synthetic surfaces (all finite)
    regime_flags: [T]   — 0/1
    Returns TensorDataset(inputs[B,2,N], targets[B,N], loss_masks[B,N], conds[B,1])
    """
    inps, tgts, lmasks, conds = [], [], [], []
    T, N = iv_matrix.shape

    for t in range(T):
        row = iv_matrix[t]
        obs_idx = np.where(np.isfinite(row))[0]
        if len(obs_idx) < min_obs: continue
        rn, _ = normalize_row(row)

        for _ in range(n_aug):
            n_mask  = max(1, int(len(obs_idx) * rng.uniform(mask_min, mask_max)))
            mid     = rng.choice(obs_idx, n_mask, replace=False)
            iv_in   = np.where(np.isfinite(rn), rn, 0.0).copy()
            iv_in[mid] = 0.0
            obs_m   = np.where(np.isfinite(rn), 1.0, 0.0)
            obs_m[mid] = 0.0
            tgt     = np.where(np.isfinite(rn), rn, 0.0)
            lmask   = np.zeros(N, float)
            lmask[mid] = 1.0
            inps.append(np.stack([iv_in, obs_m]))   # [2, N]
            tgts.append(tgt)
            lmasks.append(lmask)
            conds.append([float(regime_flags[t])])

    if not inps: return None
    return TensorDataset(
        torch.tensor(np.array(inps),   dtype=torch.float32),
        torch.tensor(np.array(tgts),   dtype=torch.float32),
        torch.tensor(np.array(lmasks), dtype=torch.float32),
        torch.tensor(np.array(conds),  dtype=torch.float32),
    )


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────

def train(model, dataset, device, n_epochs, lr, batch_size=BATCH_SIZE, tag=""):
    if dataset is None or len(dataset) == 0:
        print(f"    [{tag}] no data — skipping"); return model, np.inf

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.05)
    model.to(device).train()

    best_loss, best_sd = np.inf, None

    for ep in range(1, n_epochs+1):
        epoch_loss = 0.0
        for inp, tgt, lmask, cond in loader:
            inp, tgt, lmask, cond = (t.to(device) for t in (inp, tgt, lmask, cond))
            pred = model(inp, cond)
            # Masked MSE (only on artificially hidden positions)
            mse  = ((pred - tgt)**2 * lmask).sum() / lmask.sum().clamp(min=1)
            # No-arbitrage butterfly penalty (Hui 2023)
            bf   = butterfly_loss(pred)
            loss = mse + LAMBDA_BUTTERFLY * bf
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        avg = epoch_loss / max(len(loader), 1)
        if avg < best_loss:
            best_loss = avg
            best_sd   = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"    [{tag}] ep {ep:3d}/{n_epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    if best_sd: model.load_state_dict(best_sd)
    model.eval()
    return model, best_loss


# ─────────────────────────────────────────────────────────────────────
# Iterative inpainting inference (Hui 2023 Algorithm 3)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def inpaint_row(model, row_iv, regime_flag, device, n_refine=N_REFINE_STEPS):
    """
    Run iterative inpainting on one row.

    At each step:
      1. Forward pass → pred
      2. Clamp observed positions back to their true values
      3. Update input with clamped prediction as the new IV channel
    This enforces exact consistency with observations across all iterations.

    Returns (pred_iv_full [N], recon_err scalar).
    """
    row_n, scale = normalize_row(row_iv)
    obs_mask = np.where(np.isfinite(row_n), 1.0, 0.0)
    iv_in    = np.where(np.isfinite(row_n), row_n, 0.0)
    true_iv  = iv_in.copy()   # observed values (normalized)

    iv_cur = iv_in.copy()
    obs_t  = torch.tensor(obs_mask,  dtype=torch.float32, device=device)
    true_t = torch.tensor(true_iv,   dtype=torch.float32, device=device)
    cond   = torch.tensor([[regime_flag]], dtype=torch.float32, device=device)

    for _ in range(n_refine):
        inp_t = torch.tensor(
            np.stack([iv_cur, obs_mask])[None], dtype=torch.float32, device=device
        )
        pred  = model(inp_t, cond)[0]              # [N]
        # Clamp: keep observed values exact, use model prediction for missing
        merged = obs_t * true_t + (1 - obs_t) * pred
        iv_cur = merged.cpu().numpy()

    pred_n = iv_cur
    pred_iv = pred_n * scale   # denormalize

    # Reconstruction error on observed positions
    obs_idx = np.where(np.isfinite(row_iv))[0]
    if len(obs_idx) > 0:
        recon_err = float(np.mean((pred_iv[obs_idx] - row_iv[obs_idx])**2))
    else:
        recon_err = np.inf

    return pred_iv, recon_err


def cnn_alpha(recon_err, scale=5e-4):
    """High reconstruction error → low CNN trust."""
    return float(np.clip(CNN_ALPHA_MAX / (1 + recon_err / scale),
                         CNN_ALPHA_MIN, CNN_ALPHA_MAX))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng    = np.random.default_rng(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path.resolve())

    pfx  = args.out_prefix
    filled_out     = Path(f"filled_dataset_{pfx}.csv")
    submission_out = Path(f"submission_{pfx}.csv")
    diag_out       = Path(f"diagnostics_{pfx}.csv")

    # ── Load & sort ──────────────────────────────────────────────────
    raw = pd.read_csv(data_path)
    df  = raw.copy()
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M",
                                           errors="coerce")
    if df["datetime_parsed"].isna().any():
        raise ValueError(f"{df['datetime_parsed'].isna().sum()} unparseable datetimes")
    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta        = parse_metadata(df)
    option_cols = meta["column"].tolist()
    strike_map  = dict(zip(meta["column"], meta["strike"]))
    type_map    = dict(zip(meta["column"], meta["option_type"]))
    cols_by_type = {
        "CE": sorted([c for c in option_cols if type_map[c]=="CE"], key=lambda c: strike_map[c]),
        "PE": sorted([c for c in option_cols if type_map[c]=="PE"], key=lambda c: strike_map[c]),
    }
    n_ce    = len(cols_by_type["CE"])
    n_pe    = len(cols_by_type["PE"])
    n_joint = n_ce + n_pe
    gm      = float(df[option_cols].stack().median())

    regime_flags = np.array([1.0 if is_27jan(dt) else 0.0 for dt in df["datetime"]])

    print(f"Strikes: {n_ce} CE + {n_pe} PE = {n_joint}")
    print(f"Timestamps: {len(df)}  |  Missing cells: {int(df[option_cols].isna().sum().sum())}")
    print(f"27-Jan timestamps: {int(regime_flags.sum())}")

    # ── IV matrices ──────────────────────────────────────────────────
    iv_ce    = df[cols_by_type["CE"]].values.astype(float)
    iv_pe    = df[cols_by_type["PE"]].values.astype(float)
    iv_joint = np.concatenate([iv_ce, iv_pe], axis=1)

    # ── Walk-forward fold construction ───────────────────────────────
    # Find all timestamps that have at least one missing cell
    has_missing = np.array([
        df[option_cols].iloc[t].isna().any() for t in range(len(df))
    ])
    pred_timestamps = np.where(has_missing)[0]   # indices into df

    n_folds = min(args.folds, len(pred_timestamps))
    # Divide prediction timestamps into n_folds consecutive groups
    fold_splits = np.array_split(pred_timestamps, n_folds)

    print(f"\nWalk-forward folds: {n_folds}")
    for k, fold in enumerate(fold_splits):
        first_pred = fold[0]
        n_train    = int(np.sum(~has_missing[:first_pred]))
        print(f"  Fold {k}: predict t=[{fold[0]}..{fold[-1]}]  "
              f"train on {first_pred} timestamps before it "
              f"({n_train} fully observed)")

    # ── SVI pretraining dataset (joint CE+PE) ────────────────────────
    print(f"\nGenerating {N_SVI_PRETRAIN:,} SVI synthetic surfaces for pretraining...")
    svi_surfaces_ce    = generate_svi_surfaces(N_SVI_PRETRAIN, n_ce, rng)
    svi_surfaces_pe    = generate_svi_surfaces(N_SVI_PRETRAIN, n_pe, rng)
    svi_joint          = np.concatenate([svi_surfaces_ce, svi_surfaces_pe], axis=1)
    svi_regime         = np.zeros(N_SVI_PRETRAIN)   # all "normal" regime
    print("  Synthetic surfaces generated.")

    # ── Imputation state ─────────────────────────────────────────────
    filled             = df.copy()
    filled_values_by_row = {}
    diag_rows          = []

    miss_cells         = fill_order(df, cols_by_type, strike_map)
    # Map each missing cell to its fold index
    cell_to_fold       = {}
    for k, fold in enumerate(fold_splits):
        for t in fold:
            cell_to_fold[int(t)] = k

    # ── Train one model per fold, then predict that fold ─────────────
    fold_models = {}   # fold_k -> trained SmileImputer

    for k, fold in enumerate(fold_splits):
        first_pred_t = int(fold[0])
        print(f"\n{'='*60}")
        print(f"FOLD {k}  |  first prediction timestamp = {first_pred_t}")

        # ── Build training data: all timestamps STRICTLY before fold ─
        train_mask = np.arange(len(df)) < first_pred_t
        train_idx  = np.where(train_mask)[0]

        if len(train_idx) < MIN_TRAIN_TIMESTAMPS:
            print(f"  Only {len(train_idx)} training timestamps — using SVI pretrain only")
            real_ds = None
        else:
            print(f"  Real training timestamps: {len(train_idx)}")
            real_iv  = iv_joint[train_idx]
            real_reg = regime_flags[train_idx]
            real_ds  = make_dataset(real_iv, real_reg, rng, n_aug=4)
            if real_ds:
                print(f"  Real training samples: {len(real_ds)}")

        # ── Pretrain on SVI synthetic data ───────────────────────────
        print(f"  Pretraining on {N_SVI_PRETRAIN:,} SVI surfaces...")
        model = SmileImputer(n_joint)
        svi_ds = make_dataset(svi_joint, svi_regime, rng, n_aug=1,
                              mask_min=0.05, mask_max=0.50)
        model, pre_loss = train(model, svi_ds, device,
                                N_PRETRAIN_EPOCHS, PRETRAIN_LR,
                                tag=f"fold{k}-pretrain")

        # ── Fine-tune on real past data ──────────────────────────────
        if real_ds:
            print(f"  Fine-tuning on real past data...")
            model, ft_loss = train(model, real_ds, device,
                                   args.epochs, FINETUNE_LR,
                                   tag=f"fold{k}-finetune")

        torch.save(model.state_dict(), Path(f"cnn_model_{pfx}_fold{k}.pt"))
        fold_models[k] = model

        # ── Predict all cells in this fold ───────────────────────────
        fold_cells = [(t, c) for t, c in miss_cells if cell_to_fold.get(int(t)) == k]
        print(f"  Imputing {len(fold_cells)} missing cells in fold {k}...")

        # Cache CNN row predictions per timestamp (avoid re-running for same row)
        cnn_cache = {}

        def get_cnn(row_idx):
            if row_idx not in cnn_cache:
                row     = filled.loc[row_idx]
                regime  = regime_flags[row_idx]
                row_iv  = np.array([row[c] for c in cols_by_type["CE"]] +
                                   [row[c] for c in cols_by_type["PE"]], dtype=float)
                pred_iv, err = inpaint_row(model, row_iv, regime, device)
                cnn_cache[row_idx] = (pred_iv, err)
            return cnn_cache[row_idx]

        for row_idx, col in tqdm(fold_cells, desc=f"  fold {k}"):
            ot = type_map[col]
            af = filled_values_by_row.setdefault(row_idx, {})

            # Baseline prediction
            bp = baseline_predict(filled, row_idx, col, ot, cols_by_type,
                                  strike_map, gm, af)

            # CNN prediction — invalidate cache for this row when newly filled
            if row_idx in cnn_cache: del cnn_cache[row_idx]
            pred_all, recon_err = get_cnn(row_idx)

            # Get per-column CNN prediction
            all_cols = cols_by_type["CE"] + cols_by_type["PE"]
            col_idx  = all_cols.index(col)
            cp = float(pred_all[col_idx])
            cp = safe_iv(cp) if np.isfinite(cp) and cp > 0 else np.nan

            # Blend
            if np.isfinite(cp):
                alpha = cnn_alpha(recon_err)
                fp    = alpha * cp + (1 - alpha) * bp
            else:
                fp    = bp
                alpha = 0.0

            if not np.isfinite(fp) or fp <= 0:
                fp = gm

            fp = safe_iv(fp)
            filled.at[row_idx, col] = fp
            filled_values_by_row[row_idx][col] = {
                "final": fp, "claude": fp, "corrected": fp, "quadratic": fp
            }

            is_e, side, _, _ = is_edge(filled.loc[row_idx], col, ot,
                                       cols_by_type, strike_map)
            diag_rows.append({
                "row_index": row_idx,
                "datetime":  df.loc[row_idx, "datetime"],
                "contract":  col,
                "option_type": ot,
                "strike":    strike_map[col],
                "fold":      k,
                "final":     fp,
                "baseline":  bp,
                "cnn_pred":  cp,
                "alpha":     alpha,
                "recon_err": recon_err,
                "edge":      is_e,
                "edge_side": side,
                "regime_27jan": bool(regime_flags[row_idx]),
            })

    # ── Save ─────────────────────────────────────────────────────────
    filled_df = filled.drop(columns=["datetime_parsed"])
    orig_df   = df.drop(columns=["datetime_parsed"])
    filled_df.to_csv(filled_out, index=False)
    sub = make_submission(orig_df, filled_df, submission_out)
    pd.DataFrame(diag_rows).to_csv(diag_out, index=False)

    n_missing_after = int(filled_df[option_cols].isna().sum().sum())
    print(f"\n{'='*60}")
    print(f"✅  Filled dataset  → {filled_out}")
    print(f"✅  Submission      → {submission_out} ({len(sub)} rows)")
    print(f"✅  Diagnostics     → {diag_out}")
    print(f"    Missing after:   {n_missing_after}")
    high_cnn = sum(1 for r in diag_rows if r["alpha"] >= 0.5)
    print(f"    High-CNN cells:  {high_cnn} / {len(diag_rows)}")


if __name__ == "__main__":
    main()
