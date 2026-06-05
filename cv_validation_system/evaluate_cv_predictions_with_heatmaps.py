"""
Evaluate IV Imputation Predictions on Synthetic CV Holdout

Use this after creating not_dataset.csv and holdout_truth.csv.

Typical workflow:
    1. Run create_synthetic_cv_dataset.py
    2. Run your imputer on cv_split/not_dataset.csv
    3. Save your completed file, e.g. cv_split/my_filled.csv
    4. Run:

       python evaluate_cv_predictions.py \
           --truth cv_split/holdout_truth.csv \
           --pred cv_split/my_filled.csv \
           --base cv_split/not_dataset.csv \
           --out-dir cv_eval_results

This evaluates ONLY the cells synthetically hidden by the CV generator.

Outputs:
    metrics_summary.json
    metrics_summary.csv
    error_rows.csv
    worst_errors.csv
    group_metrics_by_*.csv
    plots/*.png

Extra visual diagnostics:
    1. Timestamp-wise IV smile plots:
       plots/smile_by_timestamp_top_errors/
       plots/smile_by_timestamp_sample/

    2. Error heatmaps:
       plots/error_heatmaps/

The smile plots show:
    - observed available values from not_dataset.csv
    - correct hidden values from holdout_truth.csv
    - predicted hidden values from your filled file

The heatmaps show:
    x-axis = moneyness = strike / underlying_price
    y-axis = timestamp row index
    marker color = error size / direction
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEPARATOR = "||"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate synthetic CV imputation predictions.")
    parser.add_argument("--truth", type=str, default="cv_split/holdout_truth.csv", help="Path to holdout_truth.csv.")
    parser.add_argument("--pred", type=str, required=True, help="Path to your filled dataset prediction CSV.")
    parser.add_argument("--base", type=str, default=None,
                        help="Path to not_dataset.csv. If omitted, tries truth_dir/not_dataset.csv.")
    parser.add_argument("--out-dir", type=str, default="cv_eval_results", help="Output directory for analytics.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of worst individual errors to save.")
    parser.add_argument("--top-smile-plots", type=int, default=40,
                        help="Number of worst timestamp/option_type smile plots to save.")
    parser.add_argument("--sample-smile-plots", type=int, default=20,
                        help="Number of sampled timestamp/option_type smile plots to save.")
    parser.add_argument("--plot-all-smiles", action="store_true",
                        help="If set, save smile plots for every timestamp/option_type group with holdouts. Can create many files.")
    parser.add_argument("--heatmap-time-bins", type=int, default=80,
                        help="Number of timestamp bins for binned heatmaps.")
    parser.add_argument("--heatmap-moneyness-bins", type=int, default=30,
                        help="Number of moneyness bins for binned heatmaps.")
    return parser.parse_args()


def parse_option_metadata(df):
    """Parse option columns so smile plots know strike and CE/PE side."""
    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )

    records = []
    for col in df.columns:
        if col in ["datetime", "underlying_price"]:
            continue
        match = pattern.match(col)
        if match:
            item = match.groupdict()
            item["column"] = col
            item["strike"] = int(item["strike"])
            records.append(item)

    meta = pd.DataFrame(records)
    if meta.empty:
        raise ValueError("No option columns parsed from base/pred file.")

    return meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    return {
        "n": int(len(y_true)),
        "mse": float(np.mean(sq_err)) if len(y_true) else np.nan,
        "rmse": float(np.sqrt(np.mean(sq_err))) if len(y_true) else np.nan,
        "mae": float(np.mean(abs_err)) if len(y_true) else np.nan,
        "median_abs_error": float(np.median(abs_err)) if len(y_true) else np.nan,
        "bias_mean_error": float(np.mean(err)) if len(y_true) else np.nan,
        "error_std": float(np.std(err)) if len(y_true) else np.nan,
        "max_abs_error": float(np.max(abs_err)) if len(y_true) else np.nan,
        "p90_abs_error": float(np.quantile(abs_err, 0.90)) if len(y_true) else np.nan,
        "p95_abs_error": float(np.quantile(abs_err, 0.95)) if len(y_true) else np.nan,
        "p99_abs_error": float(np.quantile(abs_err, 0.99)) if len(y_true) else np.nan,
    }


def group_metrics(df, group_cols):
    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        m = compute_metrics(g["actual_iv"], g["predicted_iv"])
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(m)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mse", ascending=False)

    return out


def safe_save_plot(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def filename_safe(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def load_base_file(args, truth_path):
    """Load not_dataset.csv so smile plots can show the actually available observed points."""
    if args.base is not None:
        base_path = Path(args.base)
    else:
        base_path = truth_path.parent / "not_dataset.csv"

    if not base_path.exists():
        raise FileNotFoundError(
            f"Could not find base not_dataset.csv at {base_path.resolve()}.\n"
            "Pass it explicitly with --base path/to/not_dataset.csv."
        )

    return base_path, pd.read_csv(base_path)


def make_basic_plots(errors, out_dir):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(errors["actual_iv"], errors["predicted_iv"], alpha=0.45, s=14)
    lo = min(errors["actual_iv"].min(), errors["predicted_iv"].min())
    hi = max(errors["actual_iv"].max(), errors["predicted_iv"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("Actual IV")
    ax.set_ylabel("Predicted IV")
    ax.set_title("Predicted vs Actual IV on Synthetic Holdout")
    ax.grid(True, alpha=0.3)
    safe_save_plot(fig, plot_dir / "predicted_vs_actual.png")

    # 2. Error histogram.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors["error"], bins=60, alpha=0.8)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error = predicted - actual")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.grid(True, alpha=0.3)
    safe_save_plot(fig, plot_dir / "error_histogram.png")

    # 3. Abs error by moneyness.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(errors["moneyness"], errors["abs_error"], alpha=0.45, s=14)
    ax.set_xlabel("Moneyness = strike / underlying")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Error vs Moneyness")
    ax.grid(True, alpha=0.3)
    safe_save_plot(fig, plot_dir / "abs_error_vs_moneyness.png")

    # 4. Abs error by timestamp row.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(errors["row_index"], errors["abs_error"], alpha=0.45, s=14)
    ax.set_xlabel("Timestamp Row Index")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Error Over Time")
    ax.grid(True, alpha=0.3)
    safe_save_plot(fig, plot_dir / "abs_error_over_time.png")

    # 5. MSE by regime bar.
    gm = group_metrics(errors, ["regime"])
    if not gm.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(gm["regime"].astype(str), gm["mse"])
        ax.set_xlabel("Regime")
        ax.set_ylabel("MSE")
        ax.set_title("MSE by Regime")
        ax.grid(True, axis="y", alpha=0.3)
        safe_save_plot(fig, plot_dir / "mse_by_regime.png")

    # 6. MSE by option type.
    gm = group_metrics(errors, ["option_type"])
    if not gm.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(gm["option_type"].astype(str), gm["mse"])
        ax.set_xlabel("Option Type")
        ax.set_ylabel("MSE")
        ax.set_title("MSE by Option Type")
        ax.grid(True, axis="y", alpha=0.3)
        safe_save_plot(fig, plot_dir / "mse_by_option_type.png")


def plot_timestamp_smile(row_index, option_type, base_df, pred_df, meta, errors_for_group, out_path):
    """
    Save one timestamp-wise IV smile plot.

    The plot shows:
        circles: observed available values from not_dataset.csv
        diamonds: correct hidden values from holdout_truth.csv
        x marks: model predictions from the filled prediction file
        vertical connectors: prediction error for each hidden point

    x-axis:
        moneyness = strike / underlying_price
    """
    row_index = int(row_index)
    if row_index < 0 or row_index >= len(base_df):
        return False

    row_base = base_df.loc[row_index]
    row_pred = pred_df.loc[row_index]
    spot = row_base.get("underlying_price", np.nan)
    if pd.isna(spot) or spot <= 0:
        return False

    meta_side = meta[meta["option_type"] == option_type].copy()
    if meta_side.empty or errors_for_group.empty:
        return False

    obs_records = []
    for _, rec in meta_side.iterrows():
        col = rec["column"]
        if col not in base_df.columns:
            continue
        val = row_base[col]
        if pd.notna(val):
            obs_records.append({
                "contract": col,
                "strike": rec["strike"],
                "moneyness": rec["strike"] / spot,
                "iv": float(val),
            })
    obs = pd.DataFrame(obs_records)

    hidden = errors_for_group.copy()
    hidden["moneyness"] = hidden["strike"] / spot

    pred_curve_records = []
    for _, rec in meta_side.iterrows():
        col = rec["column"]
        if col not in pred_df.columns:
            continue
        val = row_pred[col]
        if pd.notna(val):
            pred_curve_records.append({
                "contract": col,
                "strike": rec["strike"],
                "moneyness": rec["strike"] / spot,
                "iv": float(val),
            })
    pred_curve = pd.DataFrame(pred_curve_records)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not pred_curve.empty:
        pred_curve = pred_curve.sort_values("moneyness")
        ax.plot(
            pred_curve["moneyness"], pred_curve["iv"],
            linewidth=1, alpha=0.35, linestyle="--",
            label="Filled IV curve from prediction file",
        )

    if not obs.empty:
        obs = obs.sort_values("moneyness")
        ax.scatter(
            obs["moneyness"], obs["iv"],
            marker="o", s=45, alpha=0.85,
            label="Observed available in not_dataset",
        )

    ax.scatter(
        hidden["moneyness"], hidden["actual_iv"],
        marker="D", s=75, alpha=0.95,
        label="Correct hidden IV",
    )
    ax.scatter(
        hidden["moneyness"], hidden["predicted_iv"],
        marker="x", s=95, alpha=0.95,
        label="Predicted hidden IV",
    )

    for _, rec in hidden.iterrows():
        ax.plot(
            [rec["moneyness"], rec["moneyness"]],
            [rec["actual_iv"], rec["predicted_iv"]],
            linewidth=1, alpha=0.55,
        )

    worst = hidden.sort_values("abs_error", ascending=False).head(4)
    for _, rec in worst.iterrows():
        suffix = str(rec["contract"]).split("JAN26")[-1]
        ax.annotate(
            f"{suffix}\nerr={rec['error']:+.4g}",
            xy=(rec["moneyness"], rec["predicted_iv"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, alpha=0.9,
        )

    group_mse = float(np.mean(hidden["sq_error"]))
    group_mae = float(np.mean(hidden["abs_error"]))
    group_max = float(np.max(hidden["abs_error"]))
    dt_str = str(row_base["datetime"])

    ax.set_xlabel("Moneyness = strike / underlying_price")
    ax.set_ylabel("IV")
    ax.set_title(
        f"Smile Check | row={row_index} | {dt_str} | {option_type}\n"
        f"hidden={len(hidden)} | MSE={group_mse:.8g} | MAE={group_mae:.8g} | max abs err={group_max:.8g}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    safe_save_plot(fig, out_path)
    return True


def make_smile_plots(scored, base_df, pred_df, out_dir, top_n=40, sample_n=20, plot_all=False):
    """Save timestamp-wise IV smile plots, especially for worst timestamp/side groups."""
    plot_root = out_dir / "plots"
    top_dir = plot_root / "smile_by_timestamp_top_errors"
    sample_dir = plot_root / "smile_by_timestamp_sample"
    all_dir = plot_root / "smile_by_timestamp_all"

    top_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    if plot_all:
        all_dir.mkdir(parents=True, exist_ok=True)

    meta = parse_option_metadata(base_df)

    group_cols = ["row_index", "datetime", "regime", "option_type"]
    smile_metrics = (
        scored
        .groupby(group_cols, dropna=False)
        .agg(
            n_hidden=("sq_error", "size"),
            mse=("sq_error", "mean"),
            mae=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            bias=("error", "mean"),
        )
        .reset_index()
        .sort_values(["mse", "max_abs_error"], ascending=False)
    )
    smile_metrics.to_csv(out_dir / "group_metrics_by_timestamp_smile.csv", index=False)

    saved_top = 0
    for _, g in smile_metrics.head(top_n).iterrows():
        row_index = int(g["row_index"])
        opt_type = g["option_type"]
        group_errors = scored[(scored["row_index"] == row_index) & (scored["option_type"] == opt_type)].copy()
        file_name = (
            f"rank_{saved_top+1:03d}_row_{row_index}_"
            f"{filename_safe(g['datetime'])}_{opt_type}_mse_{g['mse']:.2e}.png"
        )
        ok = plot_timestamp_smile(row_index, opt_type, base_df, pred_df, meta, group_errors, top_dir / file_name)
        if ok:
            saved_top += 1

    sample_candidates = []
    for _, gm in smile_metrics.groupby(["regime", "option_type"], dropna=False):
        sample_candidates.append(gm.head(max(1, sample_n // 4)))
        if len(gm) > 5:
            sample_candidates.append(gm.sample(min(max(1, sample_n // 8), len(gm)), random_state=42))
    sample_metrics = pd.concat(sample_candidates).drop_duplicates(subset=["row_index", "option_type"]) if sample_candidates else smile_metrics.head(sample_n)

    saved_sample = 0
    for _, g in sample_metrics.head(sample_n).iterrows():
        row_index = int(g["row_index"])
        opt_type = g["option_type"]
        group_errors = scored[(scored["row_index"] == row_index) & (scored["option_type"] == opt_type)].copy()
        file_name = f"sample_{saved_sample+1:03d}_row_{row_index}_{filename_safe(g['datetime'])}_{opt_type}.png"
        ok = plot_timestamp_smile(row_index, opt_type, base_df, pred_df, meta, group_errors, sample_dir / file_name)
        if ok:
            saved_sample += 1

    saved_all = 0
    if plot_all:
        for _, g in smile_metrics.iterrows():
            row_index = int(g["row_index"])
            opt_type = g["option_type"]
            group_errors = scored[(scored["row_index"] == row_index) & (scored["option_type"] == opt_type)].copy()
            file_name = f"row_{row_index}_{filename_safe(g['datetime'])}_{opt_type}_mse_{g['mse']:.2e}.png"
            ok = plot_timestamp_smile(row_index, opt_type, base_df, pred_df, meta, group_errors, all_dir / file_name)
            if ok:
                saved_all += 1

    return {
        "timestamp_smile_metrics_file": str(out_dir / "group_metrics_by_timestamp_smile.csv"),
        "saved_top_smile_plots": int(saved_top),
        "saved_sample_smile_plots": int(saved_sample),
        "saved_all_smile_plots": int(saved_all),
        "top_smile_plot_dir": str(top_dir),
        "sample_smile_plot_dir": str(sample_dir),
        "all_smile_plot_dir": str(all_dir) if plot_all else None,
    }


def make_error_heatmaps(scored, out_dir, time_bins=80, moneyness_bins=30):
    """
    Save heatmaps showing where the method was wrong.

    Scatter heatmaps:
        x = moneyness
        y = timestamp row
        marker color = abs_error or signed error

    Binned heatmaps:
        x-bin = moneyness bucket
        y-bin = timestamp bucket
        cell value = mean abs error or mean signed error
    """
    heatmap_dir = out_dir / "plots" / "error_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    heatmap_outputs = []

    subsets = [("all", scored)]
    for opt_type in sorted(scored["option_type"].dropna().unique()):
        subsets.append((f"option_{opt_type}", scored[scored["option_type"] == opt_type]))
    for regime in sorted(scored["regime"].dropna().unique()):
        subsets.append((f"regime_{regime}", scored[scored["regime"] == regime]))
    for regime in sorted(scored["regime"].dropna().unique()):
        for opt_type in sorted(scored["option_type"].dropna().unique()):
            sub = scored[(scored["regime"] == regime) & (scored["option_type"] == opt_type)]
            subsets.append((f"{regime}_{opt_type}", sub))

    # Summary table for binned heatmap cells.
    binned_tables = []

    for name, sub in subsets:
        if sub.empty:
            continue

        sub = sub.copy()
        name_safe = filename_safe(name)

        # Scatter heatmap: absolute error.
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            sub["moneyness"],
            sub["row_index"],
            c=sub["abs_error"],
            s=24,
            alpha=0.85,
        )
        fig.colorbar(sc, ax=ax, label="Absolute Error")
        ax.set_xlabel("Moneyness = strike / underlying_price")
        ax.set_ylabel("Timestamp Row Index")
        ax.set_title(f"Absolute Error Heatmap | {name} | n={len(sub)}")
        ax.grid(True, alpha=0.25)
        path = heatmap_dir / f"scatter_abs_error_{name_safe}.png"
        safe_save_plot(fig, path)
        heatmap_outputs.append(str(path))

        # Scatter heatmap: signed error.
        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = np.nanquantile(np.abs(sub["error"]), 0.98)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(np.abs(sub["error"]))) if len(sub) else 1.0
        sc = ax.scatter(
            sub["moneyness"],
            sub["row_index"],
            c=sub["error"],
            s=24,
            alpha=0.85,
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(sc, ax=ax, label="Signed Error = predicted - actual")
        ax.axvline(1.0, linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Moneyness = strike / underlying_price")
        ax.set_ylabel("Timestamp Row Index")
        ax.set_title(f"Signed Error Heatmap | {name} | n={len(sub)}")
        ax.grid(True, alpha=0.25)
        path = heatmap_dir / f"scatter_signed_error_{name_safe}.png"
        safe_save_plot(fig, path)
        heatmap_outputs.append(str(path))

        # Binned heatmaps.
        if len(sub) >= 5:
            # Make stable bins over this subset.
            try:
                sub["time_bin"] = pd.cut(sub["row_index"], bins=min(time_bins, max(2, sub["row_index"].nunique())), labels=False)
            except Exception:
                sub["time_bin"] = 0

            try:
                sub["moneyness_bin"] = pd.cut(sub["moneyness"], bins=min(moneyness_bins, max(2, sub["moneyness"].nunique())), labels=False)
            except Exception:
                sub["moneyness_bin"] = 0

            binned = (
                sub
                .groupby(["time_bin", "moneyness_bin"], dropna=False)
                .agg(
                    n=("abs_error", "size"),
                    mean_abs_error=("abs_error", "mean"),
                    mean_signed_error=("error", "mean"),
                    mean_moneyness=("moneyness", "mean"),
                    mean_row_index=("row_index", "mean"),
                )
                .reset_index()
            )
            binned["subset"] = name
            binned_tables.append(binned)

            pivot_abs = binned.pivot(index="time_bin", columns="moneyness_bin", values="mean_abs_error")
            if not pivot_abs.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(pivot_abs.to_numpy(), aspect="auto", origin="lower")
                fig.colorbar(im, ax=ax, label="Mean Absolute Error")
                ax.set_xlabel("Moneyness Bin")
                ax.set_ylabel("Timestamp Bin")
                ax.set_title(f"Binned Mean Absolute Error | {name}")
                path = heatmap_dir / f"binned_abs_error_{name_safe}.png"
                safe_save_plot(fig, path)
                heatmap_outputs.append(str(path))

            pivot_signed = binned.pivot(index="time_bin", columns="moneyness_bin", values="mean_signed_error")
            if not pivot_signed.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                vmax = np.nanquantile(np.abs(pivot_signed.to_numpy()), 0.98)
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                im = ax.imshow(
                    pivot_signed.to_numpy(),
                    aspect="auto",
                    origin="lower",
                    vmin=-vmax,
                    vmax=vmax,
                )
                fig.colorbar(im, ax=ax, label="Mean Signed Error")
                ax.set_xlabel("Moneyness Bin")
                ax.set_ylabel("Timestamp Bin")
                ax.set_title(f"Binned Mean Signed Error | {name}")
                path = heatmap_dir / f"binned_signed_error_{name_safe}.png"
                safe_save_plot(fig, path)
                heatmap_outputs.append(str(path))

    if binned_tables:
        binned_all = pd.concat(binned_tables, ignore_index=True)
        binned_all.to_csv(out_dir / "heatmap_binned_error_cells.csv", index=False)

    return {
        "heatmap_dir": str(heatmap_dir),
        "n_heatmap_files": len(heatmap_outputs),
        "binned_error_cells_file": str(out_dir / "heatmap_binned_error_cells.csv") if binned_tables else None,
        "heatmap_files_sample": heatmap_outputs[:20],
    }


def main():
    args = parse_args()

    truth_path = Path(args.truth)
    pred_path = Path(args.pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(truth_path)
    pred = pd.read_csv(pred_path)
    base_path, base_df = load_base_file(args, truth_path)

    required_truth_cols = {"row_index", "datetime", "contract", "actual_iv"}
    missing = required_truth_cols - set(truth.columns)
    if missing:
        raise ValueError(f"holdout_truth.csv missing columns: {missing}")

    errors = truth.copy()

    predicted_values = []
    missing_pred_count = 0

    for _, rec in truth.iterrows():
        row_idx = int(rec["row_index"])
        col = rec["contract"]

        if col not in pred.columns or row_idx >= len(pred):
            predicted_values.append(np.nan)
            missing_pred_count += 1
            continue

        predicted_values.append(pred.at[row_idx, col])

    errors["predicted_iv"] = pd.to_numeric(pd.Series(predicted_values), errors="coerce")
    errors["actual_iv"] = pd.to_numeric(errors["actual_iv"], errors="coerce")

    bad = errors["predicted_iv"].isna() | errors["actual_iv"].isna()
    if bad.any():
        print(f"Warning: {int(bad.sum())} holdout rows have missing/non-numeric predictions or truths. They will be excluded.")

    scored = errors[~bad].copy()

    scored["error"] = scored["predicted_iv"] - scored["actual_iv"]
    scored["abs_error"] = scored["error"].abs()
    scored["sq_error"] = scored["error"] ** 2
    scored["relative_abs_error"] = scored["abs_error"] / scored["actual_iv"].abs().replace(0, np.nan)

    overall = compute_metrics(scored["actual_iv"], scored["predicted_iv"])
    overall["n_unscored_bad_rows"] = int(bad.sum())
    overall["prediction_file"] = str(pred_path)
    overall["truth_file"] = str(truth_path)
    overall["base_file"] = str(base_path)

    # Save detailed rows.
    scored.to_csv(out_dir / "error_rows.csv", index=False)
    scored.sort_values("abs_error", ascending=False).head(args.top_n).to_csv(out_dir / "worst_errors.csv", index=False)

    # Group analytics.
    group_specs = {
        "by_regime": ["regime"],
        "by_option_type": ["option_type"],
        "by_regime_option_type": ["regime", "option_type"],
        "by_moneyness_bucket": ["moneyness_bucket"],
        "by_regime_moneyness_bucket": ["regime", "moneyness_bucket"],
        "by_contract": ["contract", "option_type", "strike"],
        "by_date": ["date"],
        "by_row": ["row_index", "datetime"],
    }

    group_summaries = {}
    for name, cols in group_specs.items():
        cols = [c for c in cols if c in scored.columns]
        if not cols:
            continue

        gm = group_metrics(scored, cols)
        gm.to_csv(out_dir / f"group_metrics_{name}.csv", index=False)
        group_summaries[name] = gm.head(10).to_dict(orient="records") if not gm.empty else []

    pd.DataFrame([overall]).to_csv(out_dir / "metrics_summary.csv", index=False)

    make_basic_plots(scored, out_dir)

    smile_plot_info = make_smile_plots(
        scored=scored,
        base_df=base_df,
        pred_df=pred,
        out_dir=out_dir,
        top_n=args.top_smile_plots,
        sample_n=args.sample_smile_plots,
        plot_all=args.plot_all_smiles,
    )

    heatmap_info = make_error_heatmaps(
        scored=scored,
        out_dir=out_dir,
        time_bins=args.heatmap_time_bins,
        moneyness_bins=args.heatmap_moneyness_bins,
    )

    report = {
        "overall": overall,
        "smile_plot_info": smile_plot_info,
        "heatmap_info": heatmap_info,
        "worst_group_examples": group_summaries,
        "notes": [
            "MSE is computed only on synthetic holdout cells from holdout_truth.csv.",
            "Original dataset missing cells are not scored.",
            "worst_errors.csv shows individual cells with largest absolute errors.",
            "group_metrics_by_contract.csv helps identify contracts where the method failed most.",
            "group_metrics_by_timestamp_smile.csv ranks timestamp-wise CE/PE smiles by error.",
            "plots/smile_by_timestamp_top_errors contains IV-vs-moneyness plots for the worst timestamp smiles.",
            "plots/error_heatmaps contains moneyness-vs-time heatmaps showing where errors concentrate.",
            "Heatmap x-axis is moneyness; y-axis is timestamp row index or binned timestamp index.",
        ],
    }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Evaluation complete")
    print(f"Output directory: {out_dir.resolve()}")
    print("")
    print("Overall metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    print("")
    print("Smile plots:")
    for k, v in smile_plot_info.items():
        print(f"  {k}: {v}")

    print("")
    print("Error heatmaps:")
    for k, v in heatmap_info.items():
        print(f"  {k}: {v}")

    print("")
    print("Most useful files:")
    print("  metrics_summary.json")
    print("  metrics_summary.csv")
    print("  error_rows.csv")
    print("  worst_errors.csv")
    print("  group_metrics_by_timestamp_smile.csv")
    print("  group_metrics_by_regime_option_type.csv")
    print("  group_metrics_by_contract.csv")
    print("  heatmap_binned_error_cells.csv")
    print("  plots/smile_by_timestamp_top_errors/")
    print("  plots/smile_by_timestamp_sample/")
    print("  plots/error_heatmaps/")


if __name__ == "__main__":
    main()
