"""
Create Synthetic Cross-Validation Dataset for IV Imputation

This script takes the original competition dataset and creates a synthetic
validation split:

Input:
    dataset.csv

Outputs:
    not_dataset.csv
        Same as dataset.csv, but with extra observed IV values hidden as NaN.

    holdout_truth.csv
        The true values of the extra hidden cells. Use this only for evaluation.

    holdout_mask.csv
        Boolean-style metadata showing which cells were synthetically hidden.

    holdout_summary.csv
        Summary of how many values were hidden by regime, option type, moneyness bucket.

    cv_config.json
        Parameters and random seed used to create the split.

Why this is useful:
    You can run any imputation method on not_dataset.csv.
    Then evaluate only on the cells hidden by this script, not on original missing cells.

Important:
    Original missing cells in dataset.csv remain missing in not_dataset.csv.
    They are NOT part of the validation target, because their true values are unknown.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 42
DEFAULT_HOLDOUT_FRAC = 0.12
DEFAULT_MIN_REMAINING_PER_ROW_TYPE = 6
DEFAULT_MIN_HOLDOUT_27JAN = 350
EXPIRY_DAY = pd.Timestamp("2026-01-27").date()
SEPARATOR = "||"


def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic CV holdout for IV imputation.")
    parser.add_argument("--input", type=str, default="dataset.csv", help="Path to original dataset.csv.")
    parser.add_argument("--out-dir", type=str, default="cv_split", help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC, help="Fraction of observed IV cells to hide.")
    parser.add_argument("--min-remaining-per-row-type", type=int, default=DEFAULT_MIN_REMAINING_PER_ROW_TYPE,
                        help="Minimum observed cells to leave unhidden for each timestamp and CE/PE side.")
    parser.add_argument("--min-holdout-27jan", type=int, default=DEFAULT_MIN_HOLDOUT_27JAN,
                        help="Minimum synthetic holdout cells from 27 Jan if enough eligible points exist.")
    parser.add_argument("--n-moneyness-buckets", type=int, default=5, help="Number of moneyness buckets.")
    return parser.parse_args()


def parse_metadata(df):
    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )

    records = []
    for col in df.columns:
        if col in ["datetime", "datetime_parsed", "underlying_price"]:
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


def add_candidate_records(df, meta, n_moneyness_buckets):
    option_cols = meta["column"].tolist()
    strike_map = dict(zip(meta["column"], meta["strike"]))
    type_map = dict(zip(meta["column"], meta["option_type"]))

    records = []

    for row_idx, row in df.iterrows():
        spot = row["underlying_price"]
        dt = row["datetime_parsed"]
        date = dt.date()

        if pd.isna(spot) or spot <= 0:
            continue

        regime = "jan27" if date == EXPIRY_DAY else "pre27"

        for col in option_cols:
            iv = row[col]

            # Only observed cells can be hidden.
            if pd.isna(iv):
                continue

            strike = strike_map[col]
            opt_type = type_map[col]
            moneyness = strike / spot

            records.append({
                "row_index": row_idx,
                "datetime": row["datetime"],
                "datetime_parsed": dt,
                "date": str(date),
                "contract": col,
                "option_type": opt_type,
                "strike": strike,
                "underlying_price": spot,
                "moneyness": moneyness,
                "regime": regime,
                "actual_iv": float(iv),
            })

    candidates = pd.DataFrame(records)
    if candidates.empty:
        raise ValueError("No observed IV cells found to hide.")

    # Global moneyness buckets by quantile, robust to duplicate edges.
    try:
        candidates["moneyness_bucket"] = pd.qcut(
            candidates["moneyness"],
            q=n_moneyness_buckets,
            labels=False,
            duplicates="drop",
        ).astype(int)
    except Exception:
        candidates["moneyness_bucket"] = 0

    return candidates


def choose_holdout(candidates, df, meta, seed, holdout_frac, min_remaining_per_row_type, min_holdout_27jan):
    rng = np.random.default_rng(seed)

    option_cols = meta["column"].tolist()
    type_map = dict(zip(meta["column"], meta["option_type"]))

    target_total = int(round(len(candidates) * holdout_frac))
    target_total = max(1, target_total)

    # Ensure 27 Jan is represented strongly because user explicitly asked for it.
    jan27_candidates = candidates[candidates["regime"] == "jan27"]
    target_27 = min(len(jan27_candidates), max(min_holdout_27jan, int(0.20 * target_total)))
    target_pre = max(0, target_total - target_27)

    # Track how many observed cells are still available per row and type.
    remaining_by_row_type = {}
    for row_idx, row in df.iterrows():
        for opt_type in ["CE", "PE"]:
            cols = [c for c in option_cols if type_map[c] == opt_type]
            remaining_by_row_type[(row_idx, opt_type)] = int(row[cols].notna().sum())

    chosen_indices = []

    def sample_groupwise(pool, target_n):
        if target_n <= 0 or pool.empty:
            return []

        selected = []

        # Stratify by option type and moneyness bucket.
        # Shuffle groups so we don't always favor a fixed side.
        groups = list(pool.groupby(["option_type", "moneyness_bucket"], dropna=False))
        rng.shuffle(groups)

        # Allocate roughly proportional sample counts.
        pool_size = len(pool)
        group_targets = []
        for key, g in groups:
            n_g = max(1, int(round(target_n * len(g) / pool_size)))
            group_targets.append((key, g, n_g))

        # Correct possible over-allocation later.
        for key, g, n_g in group_targets:
            if len(selected) >= target_n:
                break

            g_indices = g.index.to_numpy()
            rng.shuffle(g_indices)

            taken_in_group = 0
            for cand_idx in g_indices:
                if len(selected) >= target_n or taken_in_group >= n_g:
                    break

                rec = candidates.loc[cand_idx]
                row_idx = int(rec["row_index"])
                opt_type = rec["option_type"]
                key_rt = (row_idx, opt_type)

                # Leave enough cross-sectional points behind so imputation methods
                # still have a meaningful row/type smile to fit.
                if remaining_by_row_type[key_rt] <= min_remaining_per_row_type:
                    continue

                remaining_by_row_type[key_rt] -= 1
                selected.append(cand_idx)
                taken_in_group += 1

        # If still short, fill from the whole pool randomly.
        if len(selected) < target_n:
            already = set(selected)
            rest = pool.index.to_numpy()
            rng.shuffle(rest)

            for cand_idx in rest:
                if len(selected) >= target_n:
                    break
                if cand_idx in already:
                    continue

                rec = candidates.loc[cand_idx]
                row_idx = int(rec["row_index"])
                opt_type = rec["option_type"]
                key_rt = (row_idx, opt_type)

                if remaining_by_row_type[key_rt] <= min_remaining_per_row_type:
                    continue

                remaining_by_row_type[key_rt] -= 1
                selected.append(cand_idx)
                already.add(cand_idx)

        return selected

    chosen_indices.extend(sample_groupwise(candidates[candidates["regime"] == "jan27"], target_27))
    chosen_indices.extend(sample_groupwise(candidates[candidates["regime"] == "pre27"], target_pre))

    # If total is still short, add from all candidates.
    if len(chosen_indices) < target_total:
        already = set(chosen_indices)
        rest_pool = candidates.loc[[i for i in candidates.index if i not in already]]
        chosen_indices.extend(sample_groupwise(rest_pool, target_total - len(chosen_indices)))

    chosen = candidates.loc[chosen_indices].copy()
    chosen = chosen.drop_duplicates(subset=["row_index", "contract"]).reset_index(drop=True)

    chosen["id"] = chosen["datetime"] + SEPARATOR + chosen["contract"]

    return chosen


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")

    if df["datetime_parsed"].isna().any():
        bad = int(df["datetime_parsed"].isna().sum())
        raise ValueError(f"{bad} datetime values could not be parsed.")

    df = df.sort_values("datetime_parsed").reset_index(drop=True)

    meta = parse_metadata(df)
    option_cols = meta["column"].tolist()

    candidates = add_candidate_records(df, meta, args.n_moneyness_buckets)

    holdout = choose_holdout(
        candidates=candidates,
        df=df,
        meta=meta,
        seed=args.seed,
        holdout_frac=args.holdout_frac,
        min_remaining_per_row_type=args.min_remaining_per_row_type,
        min_holdout_27jan=args.min_holdout_27jan,
    )

    not_df = df.copy()

    for _, rec in holdout.iterrows():
        not_df.at[int(rec["row_index"]), rec["contract"]] = np.nan

    not_df_out = not_df.drop(columns=["datetime_parsed"])
    original_out = df.drop(columns=["datetime_parsed"])

    # Build a wide boolean mask with True at synthetic holdout cells.
    mask_wide = pd.DataFrame(False, index=df.index, columns=option_cols)
    for _, rec in holdout.iterrows():
        mask_wide.at[int(rec["row_index"]), rec["contract"]] = True

    mask_out = pd.concat([df[["datetime"]], mask_wide], axis=1)

    # Summaries.
    summary = (
        holdout
        .groupby(["regime", "option_type", "moneyness_bucket"], dropna=False)
        .size()
        .reset_index(name="n_hidden")
        .sort_values(["regime", "option_type", "moneyness_bucket"])
    )

    contract_summary = (
        holdout
        .groupby(["contract", "option_type", "strike"], dropna=False)
        .agg(
            n_hidden=("actual_iv", "size"),
            mean_hidden_iv=("actual_iv", "mean"),
            min_hidden_iv=("actual_iv", "min"),
            max_hidden_iv=("actual_iv", "max"),
        )
        .reset_index()
        .sort_values(["option_type", "strike"])
    )

    config = {
        "input": str(input_path),
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "min_remaining_per_row_type": args.min_remaining_per_row_type,
        "min_holdout_27jan": args.min_holdout_27jan,
        "n_moneyness_buckets": args.n_moneyness_buckets,
        "n_original_rows": int(len(df)),
        "n_option_cols": int(len(option_cols)),
        "n_observed_candidates": int(len(candidates)),
        "n_hidden": int(len(holdout)),
        "n_hidden_pre27": int((holdout["regime"] == "pre27").sum()),
        "n_hidden_jan27": int((holdout["regime"] == "jan27").sum()),
        "note": "Original missing cells are not part of validation. Only holdout_truth.csv cells are scored.",
    }

    original_out.to_csv(out_dir / "dataset_original_sorted.csv", index=False)
    not_df_out.to_csv(out_dir / "not_dataset.csv", index=False)
    holdout.to_csv(out_dir / "holdout_truth.csv", index=False)
    mask_out.to_csv(out_dir / "holdout_mask.csv", index=False)
    summary.to_csv(out_dir / "holdout_summary.csv", index=False)
    contract_summary.to_csv(out_dir / "holdout_contract_summary.csv", index=False)

    with open(out_dir / "cv_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("✅ Synthetic CV dataset created")
    print(f"Output directory: {out_dir.resolve()}")
    print("")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("")
    print("Files:")
    print(f"  not_dataset.csv")
    print(f"  holdout_truth.csv")
    print(f"  holdout_mask.csv")
    print(f"  holdout_summary.csv")
    print(f"  holdout_contract_summary.csv")
    print(f"  cv_config.json")


if __name__ == "__main__":
    main()
