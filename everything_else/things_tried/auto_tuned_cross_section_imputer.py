"""
Auto-Tuned Pure Cross-Section IV Imputer

One-file workflow:
    1. Creates a synthetic validation split from the input dataset.
    2. Runs staged hyperparameter optimization on that validation split.
    3. Uses the best hyperparameters immediately in the same script.
    4. Fills the original input dataset.
    5. Saves the final filled dataset, submission file, diagnostics, and tuning results.

This is NOT the earlier two-file setup.
You do not need to copy-paste best_config_arrays.py manually.

Model family
------------
For every missing IV cell:

    A. Edge missing block:
        progressive same-row edge quadratic

        Example right edge with 4 consecutive missing values:
            first missing  -> fit on 4 observed values to the left
            second missing -> fit on 4 observed + first predicted
            third missing  -> fit on 4 observed + first predicted + second predicted
            fourth missing -> fit on 4 observed + first + second + third predicted

        Symmetric logic for left edge, filling from the observed boundary outward.

    B. Non-edge missing cell:
        same-row CE/PE quadratic + Gaussian kernel smoother

        final = blend * quadratic + (1 - blend) * kernel

        bandwidth and blend grids are tuned globally by synthetic CV.

What gets optimized
-------------------
    1. CE bandwidth grid range
    2. PE bandwidth grid range
    3. CE blend grid preset
    4. PE blend grid preset
    5. MIN_EDGE_LOCAL_NEIGHBORS

Run
---
    python auto_tuned_cross_section_imputer.py --data dataset.csv

Fast smoke test:
    python auto_tuned_cross_section_imputer.py --data dataset.csv --quick

Heavier search:
    python auto_tuned_cross_section_imputer.py --data dataset.csv --full

Outputs
-------
    auto_tuned_results/
        tuning_all_results.csv
        tuning_stage1_results.csv
        tuning_stage2_results.csv
        tuning_stage3_results.csv
        tuning_best_config.json
        tuning_best_config_readable.txt
        tuning_best_cv_errors.csv
        tuning_best_cv_worst_250_errors.csv

        auto_holdout_truth.csv
        auto_not_dataset.csv

        filled_dataset_auto_tuned_cross_section.csv
        submission_auto_tuned_cross_section.csv
        diagnostics_auto_tuned_cross_section.csv
"""

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


EPS_IV = 1e-6
SEPARATOR = "||"
DEFAULT_RANDOM_SEED = 42


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-tune and run pure cross-section IV imputer.")
    parser.add_argument("--data", type=str, default="/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv", help="Path to dataset.csv.")
    parser.add_argument("--out-dir", type=str, default="auto_tuned_results", help="Output directory.")
    parser.add_argument("--holdout-frac", type=float, default=0.12, help="Synthetic validation holdout fraction.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--quick", action="store_true", help="Small search for testing.")
    parser.add_argument("--full", action="store_true", help="Larger search. Can take a while.")
    parser.add_argument("--top-k-stage1", type=int, default=6)
    parser.add_argument("--top-k-stage2", type=int, default=4)
    return parser.parse_args()


# ---------------------------------------------------------------------
# Metadata / helpers
# ---------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path.resolve()}")

    df = pd.read_csv(path)

    required = {"datetime", "underlying_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["datetime_parsed"] = pd.to_datetime(
        df["datetime"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if df["datetime_parsed"].isna().any():
        bad = int(df["datetime_parsed"].isna().sum())
        raise ValueError(f"{bad} datetime values could not be parsed.")

    return df.sort_values("datetime_parsed").reset_index(drop=True)


def parse_metadata(df: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError("No option columns parsed. Check option column names.")

    return meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)


def safe_iv(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return max(float(x), EPS_IV)


def moneyness_bucket(m: float) -> str:
    if not np.isfinite(m):
        return "bad"
    if m < 0.94:
        return "<0.94"
    if m < 0.97:
        return "0.94-0.97"
    if m < 1.00:
        return "0.97-1.00"
    if m < 1.03:
        return "1.00-1.03"
    return ">=1.03"


def regime_label(ts) -> str:
    try:
        dt = pd.Timestamp(ts)
        if dt.date() == pd.Timestamp("2026-01-27").date():
            return "expiry_27jan"
    except Exception:
        pass
    return "pre27"


def make_submission(original: pd.DataFrame, filled: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []

    for col in [c for c in original.columns if c != "datetime"]:
        was_missing = original[col].isna()

        for idx in original.index[was_missing]:
            uid = f"{original.loc[idx, 'datetime']}{SEPARATOR}{col}"
            rows.append({"id": uid, "value": filled.loc[idx, col]})

    submission = pd.DataFrame(rows, columns=["id", "value"]).sort_values("id").reset_index(drop=True)
    submission.to_csv(out_path, index=False)
    return submission


# ---------------------------------------------------------------------
# Synthetic holdout split
# ---------------------------------------------------------------------

def create_synthetic_holdout(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    holdout_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hide observed values in a stratified way.

    The validation set includes:
        - pre-27 and 27-Jan points
        - CE and PE
        - different moneyness buckets
        - extra pressure on edge-like cells by including low/high strikes
    """
    rng = np.random.default_rng(seed)

    option_cols = meta["column"].tolist()
    type_map = dict(zip(meta["column"], meta["option_type"]))
    strike_map = dict(zip(meta["column"], meta["strike"]))

    observed_rows = []

    for row_idx, row in df.iterrows():
        spot = row["underlying_price"]
        if pd.isna(spot) or spot <= 0:
            continue

        for col in option_cols:
            if pd.isna(row[col]):
                continue

            m = strike_map[col] / spot
            observed_rows.append({
                "row_index": int(row_idx),
                "datetime": row["datetime"],
                "datetime_parsed": row["datetime_parsed"],
                "contract": col,
                "actual_iv": float(row[col]),
                "option_type": type_map[col],
                "strike": strike_map[col],
                "moneyness": float(m),
                "moneyness_bucket": moneyness_bucket(float(m)),
                "regime": regime_label(row["datetime_parsed"]),
                "date": str(row["datetime_parsed"].date()),
            })

    observed = pd.DataFrame(observed_rows)
    if observed.empty:
        raise ValueError("No observed values available for synthetic holdout.")

    hidden_parts = []

    # Stratified random hide.
    group_cols = ["regime", "option_type", "moneyness_bucket"]
    for _, g in observed.groupby(group_cols, dropna=False):
        n_hide = max(1, int(round(len(g) * holdout_frac)))
        n_hide = min(n_hide, len(g))
        hidden_parts.append(g.sample(n=n_hide, random_state=int(seed + len(g))))

    # Extra edge stress: for each timestamp and type, sometimes hide lowest/highest observed contract.
    edge_candidates = []
    for (row_idx, opt_type), g in observed.groupby(["row_index", "option_type"]):
        g_sorted = g.sort_values("strike")
        if len(g_sorted) >= 6:
            edge_candidates.append(g_sorted.head(1))
            edge_candidates.append(g_sorted.tail(1))

    if edge_candidates:
        edge_df = pd.concat(edge_candidates, ignore_index=True)
        n_edge = max(1, int(0.15 * len(edge_df)))
        n_edge = min(n_edge, len(edge_df))
        hidden_parts.append(edge_df.sample(n=n_edge, random_state=seed + 999))

    truth = (
        pd.concat(hidden_parts, ignore_index=True)
        .drop_duplicates(subset=["row_index", "contract"])
        .reset_index(drop=True)
    )

    not_df = df.copy()
    for _, rec in truth.iterrows():
        not_df.at[int(rec["row_index"]), rec["contract"]] = np.nan

    return not_df, truth


# ---------------------------------------------------------------------
# Hyperparameter candidates
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    stage: str
    config_id: str
    ce_low_exp: float
    ce_high_exp: float
    ce_n: int
    pe_low_exp: float
    pe_high_exp: float
    pe_n: int
    ce_blend_name: str
    pe_blend_name: str
    min_edge_neighbors: int


def bandwidth_grid(low_exp: float, high_exp: float, n: int) -> np.ndarray:
    return np.logspace(float(low_exp), float(high_exp), int(n), dtype=float)


def blend_grid(name: str) -> np.ndarray:
    presets = {
        "full_0p10": np.round(np.arange(0.00, 1.0001, 0.10), 3),
        "full_0p05": np.round(np.arange(0.00, 1.0001, 0.05), 3),
        "ce_default": np.array([
            0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90, 0.95, 1.00,
        ], dtype=float),
        "pe_default": np.array([
            0.40, 0.50, 0.55, 0.60, 0.65,
            0.70, 0.725, 0.75, 0.775,
            0.80, 0.825, 0.85, 0.875,
            0.90, 0.925, 0.95, 0.975, 1.00,
        ], dtype=float),
        "quad_heavy": np.array([
            0.50, 0.60, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.925, 0.95, 0.975, 1.00,
        ], dtype=float),
        "very_quad_heavy": np.array([
            0.75, 0.80, 0.825, 0.85, 0.875,
            0.90, 0.925, 0.95, 0.975, 1.00,
        ], dtype=float),
        "kernel_heavy": np.array([
            0.00, 0.05, 0.10, 0.15, 0.20,
            0.25, 0.30, 0.35, 0.40, 0.50, 0.60,
        ], dtype=float),
        "mid_fine": np.array([
            0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90, 0.95, 1.00,
        ], dtype=float),
    }

    if name not in presets:
        raise ValueError(f"Unknown blend preset: {name}")

    return presets[name]


def candidate_runtime(c: Candidate):
    bw_by_type = {
        "CE": bandwidth_grid(c.ce_low_exp, c.ce_high_exp, c.ce_n),
        "PE": bandwidth_grid(c.pe_low_exp, c.pe_high_exp, c.pe_n),
    }
    blend_by_type = {
        "CE": blend_grid(c.ce_blend_name),
        "PE": blend_grid(c.pe_blend_name),
    }
    return bw_by_type, blend_by_type, int(c.min_edge_neighbors)


def make_stage1_candidates(quick: bool, full: bool) -> List[Candidate]:
    if quick:
        ce_ranges = [(-7.0, -4.0), (-6.5, -4.0)]
        pe_ranges = [(-7.0, -4.0), (-6.5, -4.0)]
        ce_blends = ["ce_default", "quad_heavy"]
        pe_blends = ["pe_default", "quad_heavy"]
        edge_ns = [3, 4, 5]
        n = 12
    elif full:
        ce_ranges = [
            (-9.0, -3.0), (-8.5, -3.5), (-8.0, -4.0),
            (-7.5, -4.0), (-7.0, -4.0), (-6.5, -4.0),
            (-6.0, -4.0), (-5.5, -3.5), (-5.0, -3.0),
        ]
        pe_ranges = [
            (-9.0, -3.0), (-8.5, -3.5), (-8.0, -4.0),
            (-7.5, -4.0), (-7.0, -4.0), (-6.5, -4.0),
            (-6.0, -4.0), (-5.5, -3.5), (-5.0, -3.0),
        ]
        ce_blends = ["full_0p10", "ce_default", "quad_heavy", "kernel_heavy", "mid_fine"]
        pe_blends = ["full_0p10", "pe_default", "quad_heavy", "very_quad_heavy", "mid_fine"]
        edge_ns = [3, 4, 5, 6, 7, 8]
        n = 22
    else:
        ce_ranges = [
            (-8.0, -3.5), (-7.5, -4.0), (-7.0, -4.0),
            (-6.5, -4.0), (-6.0, -4.0), (-5.5, -3.5),
        ]
        pe_ranges = [
            (-8.0, -3.5), (-7.5, -4.0), (-7.0, -4.0),
            (-6.5, -4.0), (-6.0, -4.0), (-5.5, -3.5),
        ]
        ce_blends = ["ce_default", "full_0p10", "quad_heavy", "kernel_heavy", "mid_fine"]
        pe_blends = ["pe_default", "full_0p10", "quad_heavy", "very_quad_heavy", "mid_fine"]
        edge_ns = [3, 4, 5, 6, 7]
        n = 18

    out = []
    idx = 0
    for ce_range in ce_ranges:
        for pe_range in pe_ranges:
            for ce_b in ce_blends:
                for pe_b in pe_blends:
                    for edge_n in edge_ns:
                        idx += 1
                        out.append(Candidate(
                            stage="stage1_wide",
                            config_id=f"s1_{idx:05d}",
                            ce_low_exp=ce_range[0],
                            ce_high_exp=ce_range[1],
                            ce_n=n,
                            pe_low_exp=pe_range[0],
                            pe_high_exp=pe_range[1],
                            pe_n=n,
                            ce_blend_name=ce_b,
                            pe_blend_name=pe_b,
                            min_edge_neighbors=edge_n,
                        ))
    return out


def shrink_range(low: float, high: float, factor: float):
    center = 0.5 * (float(low) + float(high))
    half = 0.5 * (float(high) - float(low)) * factor
    return center - half, center + half


def dedupe_candidates(candidates: List[Candidate]) -> List[Candidate]:
    seen = set()
    out = []

    for c in candidates:
        key = (
            round(c.ce_low_exp, 5), round(c.ce_high_exp, 5), c.ce_n,
            round(c.pe_low_exp, 5), round(c.pe_high_exp, 5), c.pe_n,
            c.ce_blend_name, c.pe_blend_name, c.min_edge_neighbors,
        )
        if key not in seen:
            seen.add(key)
            out.append(c)

    return out


def make_stage2_candidates(best_rows: pd.DataFrame) -> List[Candidate]:
    out = []
    idx = 0

    for _, row in best_rows.iterrows():
        ce_low, ce_high = shrink_range(row["ce_low_exp"], row["ce_high_exp"], 0.55)
        pe_low, pe_high = shrink_range(row["pe_low_exp"], row["pe_high_exp"], 0.55)

        ce_blends = sorted(set([row["ce_blend_name"], "full_0p05", "ce_default", "quad_heavy", "mid_fine"]))
        pe_blends = sorted(set([row["pe_blend_name"], "full_0p05", "pe_default", "quad_heavy", "very_quad_heavy", "mid_fine"]))

        edge = int(row["min_edge_neighbors"])
        edge_ns = sorted(set([max(3, edge - 1), edge, edge + 1]))

        for ce_shift in [-0.30, 0.0, 0.30]:
            for pe_shift in [-0.30, 0.0, 0.30]:
                for ce_b in ce_blends:
                    for pe_b in pe_blends:
                        for edge_n in edge_ns:
                            idx += 1
                            out.append(Candidate(
                                stage="stage2_narrow",
                                config_id=f"s2_{idx:05d}",
                                ce_low_exp=ce_low + ce_shift,
                                ce_high_exp=ce_high + ce_shift,
                                ce_n=26,
                                pe_low_exp=pe_low + pe_shift,
                                pe_high_exp=pe_high + pe_shift,
                                pe_n=26,
                                ce_blend_name=ce_b,
                                pe_blend_name=pe_b,
                                min_edge_neighbors=edge_n,
                            ))

    return dedupe_candidates(out)


def make_stage3_candidates(best_rows: pd.DataFrame) -> List[Candidate]:
    out = []
    idx = 0

    for _, row in best_rows.iterrows():
        ce_low, ce_high = shrink_range(row["ce_low_exp"], row["ce_high_exp"], 0.40)
        pe_low, pe_high = shrink_range(row["pe_low_exp"], row["pe_high_exp"], 0.40)

        ce_blends = sorted(set([row["ce_blend_name"], "full_0p05", "ce_default", "quad_heavy", "mid_fine"]))
        pe_blends = sorted(set([row["pe_blend_name"], "full_0p05", "pe_default", "quad_heavy", "very_quad_heavy", "mid_fine"]))

        edge = int(row["min_edge_neighbors"])
        edge_ns = sorted(set([max(3, edge - 1), edge, edge + 1]))

        for ce_shift in [-0.12, 0.0, 0.12]:
            for pe_shift in [-0.12, 0.0, 0.12]:
                for ce_b in ce_blends:
                    for pe_b in pe_blends:
                        for edge_n in edge_ns:
                            idx += 1
                            out.append(Candidate(
                                stage="stage3_fine",
                                config_id=f"s3_{idx:05d}",
                                ce_low_exp=ce_low + ce_shift,
                                ce_high_exp=ce_high + ce_shift,
                                ce_n=34,
                                pe_low_exp=pe_low + pe_shift,
                                pe_high_exp=pe_high + pe_shift,
                                pe_n=34,
                                ce_blend_name=ce_b,
                                pe_blend_name=pe_b,
                                min_edge_neighbors=edge_n,
                            ))

    return dedupe_candidates(out)


# ---------------------------------------------------------------------
# Model internals
# ---------------------------------------------------------------------

def fit_quadratic(x, y):
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
    if coeff is None:
        return np.nan
    return safe_iv(float(np.polyval(coeff, x)))


def loo_quadratic_preds(x, y):
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        coeff, _ = fit_quadratic(np.delete(x, i), np.delete(y, i))
        preds[i] = eval_quadratic(coeff, x[i])
    return preds


def kernel_predict_many(x_obs, y_obs, x_targets, bandwidth):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_targets = np.asarray(x_targets, dtype=float)

    if len(y_obs) == 0:
        return np.full(len(x_targets), np.nan)
    if len(y_obs) == 1:
        return np.full(len(x_targets), safe_iv(y_obs[0]))

    dist2 = (x_targets[:, None] - x_obs[None, :]) ** 2
    weights = np.exp(-dist2 / (2.0 * bandwidth))
    sums = weights.sum(axis=1)

    preds = np.empty(len(x_targets), dtype=float)
    good = sums > 1e-15

    preds[good] = (weights[good] @ y_obs) / sums[good]

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

    sums = weights.sum(axis=1)
    preds = np.empty(n, dtype=float)

    good = sums > 1e-15
    preds[good] = (weights[good] @ y) / sums[good]

    if (~good).any():
        dist2_no_self = dist2.copy()
        np.fill_diagonal(dist2_no_self, np.inf)
        nearest = np.argmin(dist2_no_self[~good], axis=1)
        preds[~good] = y[nearest]

    return np.maximum(preds, EPS_IV)


def choose_blend_and_bandwidth(x, y, opt_type, bw_by_type, blend_by_type):
    if len(y) <= 1:
        return float(bw_by_type[opt_type][0]), 1.0, "constant", np.inf

    q_loo = loo_quadratic_preds(x, y)
    valid_q = np.isfinite(q_loo)
    best_mse = float(np.mean((q_loo[valid_q] - y[valid_q]) ** 2)) if valid_q.any() else np.inf

    best_bw = float(bw_by_type[opt_type][len(bw_by_type[opt_type]) // 2])
    best_blend = 1.0
    best_model = "pure_quadratic"

    for bw in bw_by_type[opt_type]:
        k_loo = loo_kernel_preds(x, y, bw)

        for blend in blend_by_type[opt_type]:
            pred = blend * q_loo + (1.0 - blend) * k_loo
            mask = np.isfinite(pred)
            if not mask.any():
                continue

            mse = float(np.mean((pred[mask] - y[mask]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_bw = float(bw)
                best_blend = float(blend)

                if blend == 0:
                    best_model = "pure_kernel"
                elif blend == 1:
                    best_model = "pure_quadratic"
                else:
                    best_model = "quadratic_kernel_blend"

    return best_bw, best_blend, best_model, best_mse


def collect_same_row_points(row, opt_type, cols_by_type, strike_map):
    spot = row["underlying_price"]
    if pd.isna(spot) or spot <= 0:
        return np.array([]), np.array([]), []

    obs_cols = [col for col in cols_by_type[opt_type] if pd.notna(row[col])]
    x = np.array([strike_map[col] / spot for col in obs_cols], dtype=float)
    y = np.array([row[col] for col in obs_cols], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    kept = [col for col, keep in zip(obs_cols, mask) if keep]

    return x[mask], y[mask], kept


def get_same_side_state(row, opt_type, cols_by_type, strike_map):
    records = []
    for col in cols_by_type[opt_type]:
        records.append({
            "column": col,
            "strike": strike_map[col],
            "is_missing": pd.isna(row[col]),
            "iv": row[col],
        })
    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)


def is_edge_missing(row, target_col, opt_type, cols_by_type, strike_map):
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


def get_edge_block_info(row, target_col, opt_type, cols_by_type, strike_map):
    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)

    left_block = []
    for _, rec in state.iterrows():
        if bool(rec["is_missing"]):
            left_block.append(rec["column"])
        else:
            break

    if target_col in left_block:
        fill_order = list(reversed(left_block))
        return {
            "side": "left",
            "block_cols": fill_order,
            "block_size": len(left_block),
            "position_in_block": fill_order.index(target_col),
        }

    right_block = []
    for _, rec in state.iloc[::-1].iterrows():
        if bool(rec["is_missing"]):
            right_block.append(rec["column"])
        else:
            break

    if target_col in right_block:
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
    min_edge_neighbors,
):
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

    base_needed = max(int(min_edge_neighbors), block_size)

    state = get_same_side_state(row, opt_type, cols_by_type, strike_map)
    observed_records = []

    for _, rec in state.iterrows():
        col = rec["column"]
        val = row[col]
        if pd.notna(val):
            observed_records.append({
                "column": col,
                "strike": strike_map[col],
                "moneyness": strike_map[col] / spot,
                "iv": float(val),
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
        base_obs = obs[obs["strike"] < target_strike].sort_values("strike", ascending=False).head(base_needed)
        base_obs = base_obs.sort_values("strike")
    else:
        base_obs = obs[obs["strike"] > target_strike].sort_values("strike", ascending=True).head(base_needed)
        base_obs = base_obs.sort_values("strike")

    train_records = base_obs.to_dict(orient="records")

    for prev_col in block_cols[:pos]:
        if prev_col not in already_filled_row_values:
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
        f"{r.column}{'*' if r.is_predicted else ''}"
        for r in train.itertuples(index=False)
    ]

    return x, y, used_cols, {
        **block_info,
        "base_observed_needed": base_needed,
    }


def predict_edge(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    already_filled_row_values,
    min_edge_neighbors,
):
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    if pd.isna(spot) or spot <= 0:
        return float(global_median_iv), {
            "source": "edge_bad_spot",
            "n_train": 0,
            "selected_model": "fallback_global_median",
        }

    x_obs, y_obs, used_cols, block_info = collect_progressive_edge_training_points(
        row=row,
        target_col=target_col,
        opt_type=opt_type,
        cols_by_type=cols_by_type,
        strike_map=strike_map,
        already_filled_row_values=already_filled_row_values,
        min_edge_neighbors=min_edge_neighbors,
    )

    if len(y_obs) == 0:
        return float(global_median_iv), {
            "source": "edge_no_train",
            "n_train": 0,
            "selected_model": "fallback_global_median",
        }

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    x_target = strike_map[target_col] / spot
    pred = eval_quadratic(coeff, x_target)

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"
    else:
        selected_model = "edge_progressive_local_quadratic"

    return safe_iv(pred), {
        "source": "edge_progressive_same_row_quadratic",
        "n_train": len(y_obs),
        "selected_model": selected_model,
        "fit_kind": fit_kind,
        "used_cols": "|".join(map(str, used_cols)),
        "edge_side": block_info.get("side", ""),
        "edge_block_size": block_info.get("block_size", np.nan),
        "edge_position_in_block": block_info.get("position_in_block", np.nan),
        "edge_base_observed_needed": block_info.get("base_observed_needed", np.nan),
    }


def predict_non_edge(
    df,
    row_idx,
    target_col,
    opt_type,
    cols_by_type,
    strike_map,
    global_median_iv,
    bw_by_type,
    blend_by_type,
):
    row = df.loc[row_idx]
    spot = row["underlying_price"]

    x_obs, y_obs, used_cols = collect_same_row_points(row, opt_type, cols_by_type, strike_map)

    if pd.isna(spot) or spot <= 0 or len(y_obs) == 0:
        return float(global_median_iv), {
            "source": "same_row_fallback_global_median",
            "n_train": len(y_obs),
            "selected_model": "fallback_global_median",
        }

    x_target = np.array([strike_map[target_col] / spot], dtype=float)

    bandwidth, blend, selected_model, loo_mse = choose_blend_and_bandwidth(
        x_obs,
        y_obs,
        opt_type,
        bw_by_type,
        blend_by_type,
    )

    coeff, fit_kind = fit_quadratic(x_obs, y_obs)
    pred_quad = eval_quadratic(coeff, x_target[0])
    pred_kernel = kernel_predict_many(x_obs, y_obs, x_target, bandwidth)[0]

    pred = blend * pred_quad + (1.0 - blend) * pred_kernel

    if not np.isfinite(pred):
        pred = global_median_iv
        selected_model = "fallback_global_median"

    return safe_iv(pred), {
        "source": "same_row_non_edge_quad_kernel",
        "n_train": len(y_obs),
        "selected_model": selected_model,
        "fit_kind": fit_kind,
        "bandwidth": bandwidth,
        "blend": blend,
        "loo_mse": loo_mse,
        "used_cols": "|".join(map(str, used_cols)),
    }


def build_ordered_missing_cells(df, meta, cols_by_type, strike_map):
    option_cols = meta["column"].tolist()
    type_map = dict(zip(meta["column"], meta["option_type"]))

    missing_cells = []

    for row_idx in df.index:
        row = df.loc[row_idx]

        for opt_type in ["CE", "PE"]:
            side_cols = cols_by_type[opt_type]
            missing_side_cols = [c for c in side_cols if pd.isna(row[c])]

            if not missing_side_cols:
                continue

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

            edge_set = set(left_fill_order) | set(right_fill_order)
            interior = [
                c for c in state["column"].tolist()
                if c in missing_side_cols and c not in edge_set
            ]
            ordered.extend(interior)

            ordered.extend([c for c in right_fill_order if c not in ordered])

            for col in ordered:
                missing_cells.append((row_idx, col))

    return missing_cells


def fill_dataset(df_input: pd.DataFrame, meta: pd.DataFrame, candidate: Candidate, diagnostics: bool = False):
    df = df_input.copy()
    option_cols = meta["column"].tolist()
    type_map = dict(zip(meta["column"], meta["option_type"]))
    strike_map = dict(zip(meta["column"], meta["strike"]))

    cols_by_type = {
        "CE": [c for c in option_cols if type_map[c] == "CE"],
        "PE": [c for c in option_cols if type_map[c] == "PE"],
    }

    bw_by_type, blend_by_type, min_edge_neighbors = candidate_runtime(candidate)

    global_median_iv = float(df[option_cols].stack().median())
    filled = df.copy()

    missing_cells = build_ordered_missing_cells(df, meta, cols_by_type, strike_map)
    filled_values_by_row = {}
    diag_rows = []

    iterator = missing_cells
    if diagnostics:
        iterator = tqdm(missing_cells, desc="Final filling with best config")

    for row_idx, col in iterator:
        opt_type = type_map[col]
        already = filled_values_by_row.setdefault(row_idx, {})
        row = df.loc[row_idx]

        edge, edge_reason = is_edge_missing(row, col, opt_type, cols_by_type, strike_map)

        if edge:
            pred, info = predict_edge(
                df=df,
                row_idx=row_idx,
                target_col=col,
                opt_type=opt_type,
                cols_by_type=cols_by_type,
                strike_map=strike_map,
                global_median_iv=global_median_iv,
                already_filled_row_values=already,
                min_edge_neighbors=min_edge_neighbors,
            )
        else:
            pred, info = predict_non_edge(
                df=df,
                row_idx=row_idx,
                target_col=col,
                opt_type=opt_type,
                cols_by_type=cols_by_type,
                strike_map=strike_map,
                global_median_iv=global_median_iv,
                bw_by_type=bw_by_type,
                blend_by_type=blend_by_type,
            )

        if not np.isfinite(pred):
            pred = global_median_iv

        pred = safe_iv(pred)
        filled.at[row_idx, col] = pred
        filled_values_by_row.setdefault(row_idx, {})[col] = pred

        if diagnostics:
            rec = {
                "row_index": row_idx,
                "datetime": df.loc[row_idx, "datetime"],
                "contract": col,
                "option_type": opt_type,
                "strike": strike_map[col],
                "final_prediction": pred,
                "edge": edge,
                "edge_reason": edge_reason,
            }
            rec.update(info)
            diag_rows.append(rec)

    return filled, pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------
# Scoring and tuning
# ---------------------------------------------------------------------

def score_predictions(filled: pd.DataFrame, truth: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    rows = []

    for _, rec in truth.iterrows():
        row_idx = int(rec["row_index"])
        col = rec["contract"]
        actual = float(rec["actual_iv"])

        if row_idx >= len(filled) or col not in filled.columns:
            pred = np.nan
        else:
            pred = filled.at[row_idx, col]

        rows.append({
            "row_index": row_idx,
            "datetime": rec.get("datetime", filled.at[row_idx, "datetime"] if row_idx < len(filled) else ""),
            "contract": col,
            "actual_iv": actual,
            "predicted_iv": pred,
            "option_type": rec.get("option_type", ""),
            "strike": rec.get("strike", np.nan),
            "moneyness": rec.get("moneyness", np.nan),
            "moneyness_bucket": rec.get("moneyness_bucket", ""),
            "regime": rec.get("regime", ""),
        })

    scored = pd.DataFrame(rows)
    scored["actual_iv"] = pd.to_numeric(scored["actual_iv"], errors="coerce")
    scored["predicted_iv"] = pd.to_numeric(scored["predicted_iv"], errors="coerce")
    scored = scored.dropna(subset=["actual_iv", "predicted_iv"]).copy()

    if scored.empty:
        return {
            "n": 0,
            "mse": np.inf,
            "rmse": np.inf,
            "mae": np.inf,
            "p95_abs_error": np.inf,
            "max_abs_error": np.inf,
            "bias": np.nan,
        }, scored

    scored["error"] = scored["predicted_iv"] - scored["actual_iv"]
    scored["abs_error"] = scored["error"].abs()
    scored["sq_error"] = scored["error"] ** 2

    metrics = {
        "n": int(len(scored)),
        "mse": float(scored["sq_error"].mean()),
        "rmse": float(np.sqrt(scored["sq_error"].mean())),
        "mae": float(scored["abs_error"].mean()),
        "p95_abs_error": float(scored["abs_error"].quantile(0.95)),
        "max_abs_error": float(scored["abs_error"].max()),
        "bias": float(scored["error"].mean()),
    }

    for col in ["option_type", "regime", "moneyness_bucket"]:
        if col in scored.columns:
            for key, g in scored.groupby(col, dropna=False):
                safe_key = str(key).replace(" ", "_").replace(".", "p").replace("-", "_").replace("<", "lt").replace(">", "gt")
                metrics[f"mse_{col}_{safe_key}"] = float(g["sq_error"].mean())

    return metrics, scored


def evaluate_candidate(candidate: Candidate, cv_df: pd.DataFrame, truth: pd.DataFrame, meta: pd.DataFrame):
    filled, _ = fill_dataset(cv_df, meta, candidate, diagnostics=False)
    metrics, scored = score_predictions(filled, truth)

    result = {
        **asdict(candidate),
        **metrics,
    }

    return result, scored


def run_stage(stage_name: str, candidates: List[Candidate], cv_df, truth, meta, out_dir: Path):
    results = []
    best_result = None
    best_scored = None

    for candidate in tqdm(candidates, desc=stage_name):
        result, scored = evaluate_candidate(candidate, cv_df, truth, meta)
        results.append(result)

        if best_result is None or result["mse"] < best_result["mse"]:
            best_result = result
            best_scored = scored

    results_df = pd.DataFrame(results).sort_values("mse").reset_index(drop=True)
    results_df.to_csv(out_dir / f"tuning_{stage_name}_results.csv", index=False)

    return results_df, best_result, best_scored


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else str(val)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    return obj


def candidate_from_row(row, config_id_override=None):
    return Candidate(
        stage=str(row["stage"]),
        config_id=config_id_override or str(row["config_id"]),
        ce_low_exp=float(row["ce_low_exp"]),
        ce_high_exp=float(row["ce_high_exp"]),
        ce_n=int(row["ce_n"]),
        pe_low_exp=float(row["pe_low_exp"]),
        pe_high_exp=float(row["pe_high_exp"]),
        pe_n=int(row["pe_n"]),
        ce_blend_name=str(row["ce_blend_name"]),
        pe_blend_name=str(row["pe_blend_name"]),
        min_edge_neighbors=int(row["min_edge_neighbors"]),
    )


def save_best_config(best_row, out_dir: Path):
    best = best_row.to_dict()
    with open(out_dir / "tuning_best_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(best), f, indent=2)

    best_candidate = candidate_from_row(best_row, "best")
    bw_by_type, blend_by_type, min_edge_neighbors = candidate_runtime(best_candidate)

    readable_lines = [
        "BEST CONFIG",
        "===========",
        f"MSE: {best_row['mse']}",
        f"RMSE: {best_row['rmse']}",
        f"MAE: {best_row['mae']}",
        "",
        f"CE bandwidth exp range: {best_candidate.ce_low_exp} to {best_candidate.ce_high_exp}, n={best_candidate.ce_n}",
        f"PE bandwidth exp range: {best_candidate.pe_low_exp} to {best_candidate.pe_high_exp}, n={best_candidate.pe_n}",
        f"CE blend preset: {best_candidate.ce_blend_name}",
        f"PE blend preset: {best_candidate.pe_blend_name}",
        f"MIN_EDGE_LOCAL_NEIGHBORS: {min_edge_neighbors}",
        "",
        "CE bandwidth grid:",
        np.array2string(bw_by_type["CE"], precision=12, separator=", "),
        "",
        "PE bandwidth grid:",
        np.array2string(bw_by_type["PE"], precision=12, separator=", "),
        "",
        "CE blend grid:",
        np.array2string(blend_by_type["CE"], precision=4, separator=", "),
        "",
        "PE blend grid:",
        np.array2string(blend_by_type["PE"], precision=4, separator=", "),
    ]

    (out_dir / "tuning_best_config_readable.txt").write_text("\n".join(readable_lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    original_df = load_dataset(data_path)
    meta = parse_metadata(original_df)
    option_cols = meta["column"].tolist()

    print(f"Rows: {len(original_df)}")
    print(f"Parsed option columns: {len(option_cols)}")
    print(f"Output directory: {out_dir.resolve()}")

    print("\nCreating synthetic validation split...")
    cv_df, truth = create_synthetic_holdout(
        df=original_df,
        meta=meta,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
    )

    cv_out = cv_df.drop(columns=["datetime_parsed"])
    cv_out.to_csv(out_dir / "auto_not_dataset.csv", index=False)
    truth.to_csv(out_dir / "auto_holdout_truth.csv", index=False)

    print(f"Hidden validation cells: {len(truth)}")
    print(f"CV file saved: {out_dir / 'auto_not_dataset.csv'}")
    print(f"Truth file saved: {out_dir / 'auto_holdout_truth.csv'}")

    # Stage 1
    stage1_candidates = make_stage1_candidates(quick=args.quick, full=args.full)
    print(f"\nStage 1 wide search candidates: {len(stage1_candidates)}")
    s1, _, _ = run_stage("stage1", stage1_candidates, cv_df, truth, meta, out_dir)

    # Stage 2
    top_s1 = s1.head(args.top_k_stage1)
    stage2_candidates = make_stage2_candidates(top_s1)
    print(f"\nStage 2 narrow search candidates: {len(stage2_candidates)}")
    s2, _, _ = run_stage("stage2", stage2_candidates, cv_df, truth, meta, out_dir)

    # Stage 3
    top_s2 = s2.head(args.top_k_stage2)
    stage3_candidates = make_stage3_candidates(top_s2)
    print(f"\nStage 3 fine search candidates: {len(stage3_candidates)}")
    s3, _, _ = run_stage("stage3", stage3_candidates, cv_df, truth, meta, out_dir)

    all_results = pd.concat([s1, s2, s3], ignore_index=True)
    all_results = all_results.sort_values("mse").reset_index(drop=True)
    all_results.to_csv(out_dir / "tuning_all_results.csv", index=False)

    best_row = all_results.iloc[0]
    save_best_config(best_row, out_dir)
    best_candidate = candidate_from_row(best_row, "best_final")

    # Save best CV errors.
    _, best_cv_scored = evaluate_candidate(best_candidate, cv_df, truth, meta)
    best_cv_scored.to_csv(out_dir / "tuning_best_cv_errors.csv", index=False)
    best_cv_scored.sort_values("abs_error", ascending=False).head(250).to_csv(
        out_dir / "tuning_best_cv_worst_250_errors.csv",
        index=False,
    )

    print("\nBest hyperparameters found:")
    print(f"  Stage/config: {best_row['stage']} / {best_row['config_id']}")
    print(f"  CV MSE: {best_row['mse']}")
    print(f"  CV RMSE: {best_row['rmse']}")
    print(f"  CV MAE: {best_row['mae']}")
    print(f"  CE bw exp range: {best_row['ce_low_exp']} to {best_row['ce_high_exp']} n={best_row['ce_n']}")
    print(f"  PE bw exp range: {best_row['pe_low_exp']} to {best_row['pe_high_exp']} n={best_row['pe_n']}")
    print(f"  CE blend: {best_row['ce_blend_name']}")
    print(f"  PE blend: {best_row['pe_blend_name']}")
    print(f"  MIN_EDGE_LOCAL_NEIGHBORS: {best_row['min_edge_neighbors']}")

    # Final application on original dataset.
    print("\nFilling ORIGINAL dataset using the best hyperparameters...")
    final_filled, final_diag = fill_dataset(original_df, meta, best_candidate, diagnostics=True)

    final_filled_out = final_filled.drop(columns=["datetime_parsed"])
    original_out = original_df.drop(columns=["datetime_parsed"])

    filled_path = out_dir / "filled_dataset_auto_tuned_cross_section.csv"
    submission_path = out_dir / "submission_auto_tuned_cross_section.csv"
    diagnostics_path = out_dir / "diagnostics_auto_tuned_cross_section.csv"

    final_filled_out.to_csv(filled_path, index=False)
    submission = make_submission(original_out, final_filled_out, submission_path)
    final_diag.to_csv(diagnostics_path, index=False)

    print("\nDone.")
    print(f"Filled dataset: {filled_path}")
    print(f"Submission: {submission_path} ({len(submission)} rows)")
    print(f"Diagnostics: {diagnostics_path}")
    print(f"Tuning results: {out_dir / 'tuning_all_results.csv'}")
    print(f"Best config: {out_dir / 'tuning_best_config_readable.txt'}")


if __name__ == "__main__":
    main()
