"""
Lag-1 Correlation + Calendar-Gap EDA for NIFTY IV Time Dynamics
===============================================================

This EDA helps validate whether IV can be treated as a smooth time process when
real calendar gaps are preserved.

It creates two groups of visuals:

1. Lag-1 timestamp analysis
   For each available timestamp t, compare IV(option, t) with IV(option, previous
   available timestamp), independently by contract. Then summarize across CE,
   PE, and ALL options.

2. Calendar-gap contract path analysis
   For one selected contract at a time, plot IV against real calendar time on a
   5-minute scale. Missing calendar days are left as blank space, not compressed.

Run:
    python eda_lag1_calendar_gap_iv.py --data dataset.csv

Optional interactive Matplotlib dashboard:
    python eda_lag1_calendar_gap_iv.py --data dataset.csv --show

Outputs:
    eda_lag1_calendar_gap/
        lag1_pair_rows.csv
        lag1_timestamp_metrics.csv
        lag1_option_metrics.csv
        lag1_summary_dashboard.png
        lag1_corr_by_timestamp.png
        lag1_abs_change_by_timestamp.png
        lag1_gap_minutes_by_timestamp.png
        lag1_heatmap_CE_abs_change.png
        lag1_heatmap_PE_abs_change.png
        lag1_heatmap_ALL_abs_change.png
        lag1_heatmap_CE_signed_change.png
        lag1_heatmap_PE_signed_change.png
        lag1_heatmap_ALL_signed_change.png
        calendar_gap_contract_dashboard.png
        calendar_gap_selected_contract.csv
        interactive_lag1_calendar_dashboard.html
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None


DEFAULT_DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv")
FIVE_MINUTES = pd.Timedelta(minutes=5)


def parse_args():
    parser = argparse.ArgumentParser(description="Lag-1 correlation and calendar-gap IV EDA.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to dataset.csv.")
    parser.add_argument("--out-dir", type=str, default="eda_lag1_calendar_gap", help="Output directory.")
    parser.add_argument("--selected-contract", type=str, default=None,
                        help="Optional contract column for the static calendar-gap plot.")
    parser.add_argument("--max-xticks", type=int, default=14, help="Max x-axis tick labels for static plots.")
    parser.add_argument("--show", action="store_true",
                        help="Open the interactive Matplotlib dashboard after saving outputs.")
    return parser.parse_args()


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
        raise ValueError(f"{bad} datetime values could not be parsed. Expected DD-MM-YYYY HH:MM.")

    return df.sort_values("datetime_parsed").reset_index(drop=True)


def parse_option_metadata(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )

    records = []
    unparsed = []

    for col in df.columns:
        if col in {"datetime", "datetime_parsed", "underlying_price"}:
            continue

        match = pattern.match(col)
        if not match:
            unparsed.append(col)
            continue

        item = match.groupdict()
        item["column"] = col
        item["strike"] = int(item["strike"])
        item["expiry_date"] = pd.to_datetime(item["expiry"], format="%d%b%y", errors="coerce")
        records.append(item)

    meta = pd.DataFrame(records)
    if meta.empty:
        raise ValueError("No option columns parsed. Check column names.")

    meta = meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)

    if unparsed:
        print("Warning: some columns were not parsed as option columns:")
        for col in unparsed:
            print("  ", col)

    return meta


def contract_label(row: pd.Series) -> str:
    return f"{int(row['strike'])} {row['option_type']} ({row['column']})"


def build_lag1_pair_rows(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build long pair-level lag-1 data.

    Each row corresponds to one option contract at timestamp t, compared to the
    previous available dataset timestamp t-1. A row exists only when both IVs
    are observed.
    """
    option_cols = meta["column"].tolist()
    type_map = dict(zip(meta["column"], meta["option_type"]))
    strike_map = dict(zip(meta["column"], meta["strike"]))

    rows = []

    for i in range(1, len(df)):
        now = df.loc[i]
        prev = df.loc[i - 1]

        dt_now = now["datetime_parsed"]
        dt_prev = prev["datetime_parsed"]
        gap_minutes = (dt_now - dt_prev).total_seconds() / 60.0

        spot_now = now["underlying_price"]
        spot_prev = prev["underlying_price"]

        for col in option_cols:
            y_now = now[col]
            y_prev = prev[col]

            if pd.isna(y_now) or pd.isna(y_prev):
                continue

            m_now = np.nan
            m_prev = np.nan
            if pd.notna(spot_now) and spot_now > 0:
                m_now = strike_map[col] / spot_now
            if pd.notna(spot_prev) and spot_prev > 0:
                m_prev = strike_map[col] / spot_prev

            change = float(y_now) - float(y_prev)

            rows.append({
                "row_index": i,
                "prev_row_index": i - 1,
                "datetime": now["datetime"],
                "datetime_parsed": dt_now,
                "prev_datetime": prev["datetime"],
                "prev_datetime_parsed": dt_prev,
                "date": str(dt_now.date()),
                "prev_date": str(dt_prev.date()),
                "is_new_calendar_day": dt_now.date() != dt_prev.date(),
                "gap_minutes": float(gap_minutes),
                "gap_5min_steps": float(gap_minutes / 5.0),
                "contract": col,
                "option_type": type_map[col],
                "strike": strike_map[col],
                "moneyness_now": float(m_now) if np.isfinite(m_now) else np.nan,
                "moneyness_prev": float(m_prev) if np.isfinite(m_prev) else np.nan,
                "iv_now": float(y_now),
                "iv_prev": float(y_prev),
                "iv_change": change,
                "abs_iv_change": abs(change),
                "pct_iv_change": change / float(y_prev) if abs(float(y_prev)) > 1e-12 else np.nan,
            })

    return pd.DataFrame(rows)


def compute_timestamp_metrics(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (idx, opt_type), g in pair_rows.groupby(["row_index", "option_type"], dropna=False):
        corr = float(g["iv_now"].corr(g["iv_prev"])) if len(g) >= 2 else np.nan
        rows.append({
            "row_index": int(idx),
            "datetime": g["datetime"].iloc[0],
            "datetime_parsed": g["datetime_parsed"].iloc[0],
            "date": g["date"].iloc[0],
            "option_type": opt_type,
            "n_pairs": int(len(g)),
            "lag1_corr_cross_options": corr,
            "mean_abs_iv_change": float(g["abs_iv_change"].mean()),
            "median_abs_iv_change": float(g["abs_iv_change"].median()),
            "mean_signed_iv_change": float(g["iv_change"].mean()),
            "mean_pct_iv_change": float(g["pct_iv_change"].replace([np.inf, -np.inf], np.nan).mean()),
            "gap_minutes": float(g["gap_minutes"].iloc[0]),
            "gap_5min_steps": float(g["gap_5min_steps"].iloc[0]),
            "is_new_calendar_day": bool(g["is_new_calendar_day"].iloc[0]),
            "prev_datetime": g["prev_datetime"].iloc[0],
        })

    for idx, g in pair_rows.groupby("row_index", dropna=False):
        corr = float(g["iv_now"].corr(g["iv_prev"])) if len(g) >= 2 else np.nan
        rows.append({
            "row_index": int(idx),
            "datetime": g["datetime"].iloc[0],
            "datetime_parsed": g["datetime_parsed"].iloc[0],
            "date": g["date"].iloc[0],
            "option_type": "ALL",
            "n_pairs": int(len(g)),
            "lag1_corr_cross_options": corr,
            "mean_abs_iv_change": float(g["abs_iv_change"].mean()),
            "median_abs_iv_change": float(g["abs_iv_change"].median()),
            "mean_signed_iv_change": float(g["iv_change"].mean()),
            "mean_pct_iv_change": float(g["pct_iv_change"].replace([np.inf, -np.inf], np.nan).mean()),
            "gap_minutes": float(g["gap_minutes"].iloc[0]),
            "gap_5min_steps": float(g["gap_5min_steps"].iloc[0]),
            "is_new_calendar_day": bool(g["is_new_calendar_day"].iloc[0]),
            "prev_datetime": g["prev_datetime"].iloc[0],
        })

    return pd.DataFrame(rows).sort_values(["option_type", "row_index"]).reset_index(drop=True)


def compute_option_metrics(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for contract, g in pair_rows.groupby("contract"):
        corr = float(g["iv_now"].corr(g["iv_prev"])) if len(g) >= 2 else np.nan
        rows.append({
            "contract": contract,
            "option_type": g["option_type"].iloc[0],
            "strike": int(g["strike"].iloc[0]),
            "n_pairs": int(len(g)),
            "lag1_autocorr_over_time": corr,
            "mean_abs_iv_change": float(g["abs_iv_change"].mean()),
            "median_abs_iv_change": float(g["abs_iv_change"].median()),
            "mean_signed_iv_change": float(g["iv_change"].mean()),
            "mean_abs_change_normal_gap": float(g.loc[g["gap_minutes"] <= 5.01, "abs_iv_change"].mean()),
            "mean_abs_change_after_gap": float(g.loc[g["gap_minutes"] > 5.01, "abs_iv_change"].mean()),
        })

    return pd.DataFrame(rows).sort_values(["option_type", "strike"]).reset_index(drop=True)


def choose_xticks(datetimes: pd.Series, max_ticks: int):
    n = len(datetimes)
    if n == 0:
        return [], []
    pos = np.linspace(0, n - 1, min(max_ticks, n)).astype(int)
    labels = [pd.Timestamp(datetimes.iloc[i]).strftime("%d-%m\n%H:%M") for i in pos]
    return pos, labels


def save_lag1_summary_plots(metrics: pd.DataFrame, out_dir: Path, max_xticks: int):
    all_m = metrics[metrics["option_type"] == "ALL"].copy()
    if all_m.empty:
        return

    x = np.arange(len(all_m))
    tick_pos, tick_labels = choose_xticks(all_m["datetime_parsed"], max_xticks)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(x, all_m["lag1_corr_cross_options"], marker="o", linewidth=1)
    axes[0].set_ylabel("Lag-1 corr")
    axes[0].set_title("Lag-1 cross-option correlation by timestamp (ALL)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, all_m["mean_abs_iv_change"], marker="o", linewidth=1)
    axes[1].set_ylabel("Mean abs ΔIV")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, all_m["mean_signed_iv_change"], marker="o", linewidth=1)
    axes[2].axhline(0, linestyle="--", linewidth=1)
    axes[2].set_ylabel("Mean signed ΔIV")
    axes[2].grid(True, alpha=0.3)

    axes[3].bar(x, all_m["gap_minutes"])
    axes[3].axhline(5, linestyle="--", linewidth=1)
    axes[3].set_ylabel("Gap minutes")
    axes[3].set_xlabel("Timestamp")
    axes[3].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        for xi, is_new in zip(x, all_m["is_new_calendar_day"]):
            if is_new:
                ax.axvline(xi, alpha=0.12, linewidth=2)

    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labels, rotation=0)

    fig.tight_layout()
    fig.savefig(out_dir / "lag1_summary_dashboard.png", dpi=170)
    plt.close(fig)

    for metric_col, y_label, file_name, title in [
        ("lag1_corr_cross_options", "Lag-1 correlation", "lag1_corr_by_timestamp.png", "Lag-1 Cross-Option Correlation"),
        ("mean_abs_iv_change", "Mean abs ΔIV", "lag1_abs_change_by_timestamp.png", "Mean Absolute IV Change"),
        ("gap_minutes", "Minutes", "lag1_gap_minutes_by_timestamp.png", "Real Time Gap Since Previous Available Timestamp"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        for opt_type in ["CE", "PE", "ALL"]:
            sub = metrics[metrics["option_type"] == opt_type].copy()
            if sub.empty:
                continue
            xx = np.arange(len(sub))
            ax.plot(xx, sub[metric_col], marker="o", linewidth=1, label=opt_type)

        if metric_col == "gap_minutes":
            ax.axhline(5, linestyle="--", linewidth=1, label="5 min normal gap")

        ax.set_ylabel(y_label)
        ax.set_xlabel("Available timestamp index")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / file_name, dpi=170)
        plt.close(fig)


def pivot_heatmap(pair_rows: pd.DataFrame, option_type: str, value_col: str):
    sub = pair_rows if option_type == "ALL" else pair_rows[pair_rows["option_type"] == option_type].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.copy()
    sub["contract_label"] = sub["strike"].astype(str) + " " + sub["option_type"]

    pivot = sub.pivot_table(
        index="contract_label",
        columns="row_index",
        values=value_col,
        aggfunc="mean",
    )

    idx_df = (
        sub[["contract_label", "option_type", "strike"]]
        .drop_duplicates()
        .sort_values(["option_type", "strike"])
    )
    ordered_index = [x for x in idx_df["contract_label"] if x in pivot.index]
    return pivot.loc[ordered_index]


def save_lag1_heatmaps(pair_rows: pd.DataFrame, df: pd.DataFrame, out_dir: Path):
    for value_col, name in [
        ("abs_iv_change", "abs_change"),
        ("iv_change", "signed_change"),
    ]:
        for opt_type in ["CE", "PE", "ALL"]:
            pivot = pivot_heatmap(pair_rows, opt_type, value_col)
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(15, max(5, 0.35 * len(pivot))))
            data = pivot.to_numpy(dtype=float)
            im = ax.imshow(data, aspect="auto", interpolation="nearest")

            ax.set_title(f"Lag-1 {value_col.replace('_', ' ')} heatmap — {opt_type}")
            ax.set_ylabel("Option")
            ax.set_xlabel("Timestamp row index")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)

            cols = list(pivot.columns)
            if cols:
                n_ticks = min(14, len(cols))
                positions = np.linspace(0, len(cols) - 1, n_ticks).astype(int)
                labels = []
                for pos in positions:
                    row_idx = int(cols[pos])
                    labels.append(pd.Timestamp(df.loc[row_idx, "datetime_parsed"]).strftime("%d-%m\n%H:%M"))
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=0)

                for j, row_idx in enumerate(cols):
                    if row_idx > 0:
                        gap = (df.loc[row_idx, "datetime_parsed"] - df.loc[row_idx - 1, "datetime_parsed"]).total_seconds() / 60
                        if gap > 5.01:
                            ax.axvline(j - 0.5, color="white", alpha=0.8, linewidth=1.5)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(value_col)
            fig.tight_layout()
            fig.savefig(out_dir / f"lag1_heatmap_{opt_type}_{name}.png", dpi=170)
            plt.close(fig)


def build_calendar_grid(df: pd.DataFrame) -> pd.DatetimeIndex:
    start = df["datetime_parsed"].min()
    end = df["datetime_parsed"].max()
    return pd.date_range(start=start, end=end, freq="5min")


def build_contract_calendar_series(df: pd.DataFrame, contract: str) -> pd.DataFrame:
    grid = pd.DataFrame({"datetime_parsed": build_calendar_grid(df)})
    tmp = df[["datetime_parsed", "datetime", "underlying_price", contract]].copy()
    tmp = tmp.rename(columns={contract: "iv"})
    merged = grid.merge(tmp, on="datetime_parsed", how="left")
    merged["is_dataset_timestamp"] = merged["datetime"].notna()
    merged["is_missing_iv_at_dataset_timestamp"] = merged["is_dataset_timestamp"] & merged["iv"].isna()
    return merged


def save_static_calendar_contract_plot(df: pd.DataFrame, meta: pd.DataFrame, out_dir: Path, selected_contract: str | None):
    if selected_contract is None:
        ce = meta[meta["option_type"] == "CE"].copy()
        selected_contract = ce.iloc[len(ce) // 2]["column"] if not ce.empty else meta.iloc[0]["column"]

    if selected_contract not in df.columns:
        raise ValueError(f"Selected contract {selected_contract} not found in dataset columns.")

    series = build_contract_calendar_series(df, selected_contract)
    series.to_csv(out_dir / "calendar_gap_selected_contract.csv", index=False)

    meta_row = meta[meta["column"] == selected_contract].iloc[0]
    label = contract_label(meta_row)

    observed = series[series["iv"].notna()].copy()
    dataset_missing = series[series["is_missing_iv_at_dataset_timestamp"]].copy()

    fig, ax = plt.subplots(figsize=(16, 6))

    if not observed.empty:
        ax.plot(observed["datetime_parsed"], observed["iv"], linewidth=1, alpha=0.6)
        ax.scatter(observed["datetime_parsed"], observed["iv"], s=24, alpha=0.9, label="Observed IV")
        y_min = max(0, observed["iv"].quantile(0.001) - 0.02)
        y_max = observed["iv"].quantile(0.999) + 0.02
    else:
        y_min, y_max = 0, 1

    missing_y = max(0, y_min - 0.015)

    if not dataset_missing.empty:
        ax.scatter(
            dataset_missing["datetime_parsed"],
            np.full(len(dataset_missing), missing_y),
            marker="x",
            s=55,
            label="Missing IV at dataset timestamp",
        )

    for d in pd.Series(df["datetime_parsed"].dt.normalize().unique()).sort_values():
        ax.axvline(pd.Timestamp(d), alpha=0.12, linewidth=2)

    ax.set_title(f"Calendar-spaced IV path with real blank gaps: {label}")
    ax.set_xlabel("Real calendar time, 5-minute scale")
    ax.set_ylabel("IV")
    ax.set_ylim(max(0, missing_y - 0.01), y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "calendar_gap_contract_dashboard.png", dpi=170)
    plt.close(fig)


def create_plotly_html(df: pd.DataFrame, meta: pd.DataFrame, pair_rows: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path):
    if go is None or make_subplots is None:
        print("Plotly not installed; skipping interactive HTML. Install with: pip install plotly")
        return

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            "Lag-1 cross-option correlation by available timestamp",
            "Mean absolute IV change by available timestamp",
            "Real time gap since previous available timestamp",
        ],
    )

    for opt_type in ["CE", "PE", "ALL"]:
        sub = metrics[metrics["option_type"] == opt_type].copy()
        if sub.empty:
            continue

        visible = True if opt_type == "ALL" else "legendonly"
        fig.add_trace(
            go.Scatter(
                x=sub["datetime_parsed"],
                y=sub["lag1_corr_cross_options"],
                mode="lines+markers",
                name=f"{opt_type} corr",
                visible=visible,
                text=[
                    f"prev={p}<br>gap={g:.0f} min<br>n={n}"
                    for p, g, n in zip(sub["prev_datetime"], sub["gap_minutes"], sub["n_pairs"])
                ],
                hovertemplate="%{x}<br>corr=%{y:.4f}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=sub["datetime_parsed"],
                y=sub["mean_abs_iv_change"],
                mode="lines+markers",
                name=f"{opt_type} mean abs ΔIV",
                visible=visible,
                hovertemplate="%{x}<br>mean abs ΔIV=%{y:.6f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=sub["datetime_parsed"],
                y=sub["gap_minutes"],
                name=f"{opt_type} gap minutes",
                visible=visible,
                hovertemplate="%{x}<br>gap=%{y:.0f} minutes<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title="Lag-1 IV dynamics with real calendar gaps retained",
        height=1000,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Mean abs ΔIV", row=2, col=1)
    fig.update_yaxes(title_text="Gap minutes", row=3, col=1)
    fig.write_html(out_dir / "interactive_lag1_calendar_dashboard.html")


def make_interactive_matplotlib_dashboard(df: pd.DataFrame, meta: pd.DataFrame, pair_rows: pd.DataFrame, metrics: pd.DataFrame):
    option_type_state = {"type": "CE"}
    contract_indices_state = {"indices": list(meta.index[meta["option_type"] == "CE"])}
    selected_pos_state = {"pos": 0}

    fig = plt.figure(figsize=(15, 9))
    plt.subplots_adjust(bottom=0.22, right=0.82)

    ax_corr = fig.add_subplot(2, 2, 1)
    ax_heat = fig.add_subplot(2, 2, 3)
    ax_path = fig.add_subplot(1, 2, 2)

    radio_ax = fig.add_axes([0.85, 0.55, 0.10, 0.18])
    radio = RadioButtons(radio_ax, ("CE", "PE", "ALL"), active=0)

    slider_ax = fig.add_axes([0.15, 0.10, 0.55, 0.035])
    contract_slider = Slider(
        ax=slider_ax,
        label="Contract Index",
        valmin=0,
        valmax=max(0, len(contract_indices_state["indices"]) - 1),
        valinit=0,
        valstep=1,
    )

    prev_ax = fig.add_axes([0.15, 0.04, 0.10, 0.04])
    next_ax = fig.add_axes([0.27, 0.04, 0.10, 0.04])
    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")

    info_text = fig.text(0.42, 0.045, "", fontsize=9, ha="left", va="center")

    def reset_contract_indices(opt_type):
        if opt_type == "ALL":
            indices = list(meta.index)
        else:
            indices = list(meta.index[meta["option_type"] == opt_type])
        contract_indices_state["indices"] = indices
        selected_pos_state["pos"] = 0
        contract_slider.valmax = max(0, len(indices) - 1)
        contract_slider.ax.set_xlim(0, max(0, len(indices) - 1))
        contract_slider.set_val(0)

    def current_contract_row():
        indices = contract_indices_state["indices"]
        if not indices:
            return meta.iloc[0]
        pos = int(selected_pos_state["pos"])
        pos = max(0, min(pos, len(indices) - 1))
        selected_pos_state["pos"] = pos
        return meta.loc[indices[pos]]

    def draw(_=None):
        opt_type = option_type_state["type"]

        ax_corr.clear()
        subm = metrics[metrics["option_type"] == opt_type].copy()
        if not subm.empty:
            x = np.arange(len(subm))
            ax_corr.plot(x, subm["lag1_corr_cross_options"], marker="o", linewidth=1)
            for xi, is_new in zip(x, subm["is_new_calendar_day"]):
                if is_new:
                    ax_corr.axvline(xi, alpha=0.12, linewidth=2)
            ax_corr.set_title(f"Lag-1 correlation by timestamp — {opt_type}")
            ax_corr.set_ylabel("Corr(IV_t, IV_prev) across options")
            ax_corr.set_xlabel("Available timestamp index")
            ax_corr.grid(True, alpha=0.3)

        ax_heat.clear()
        pivot = pivot_heatmap(pair_rows, opt_type, "abs_iv_change")
        if not pivot.empty:
            ax_heat.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
            ax_heat.set_title(f"Heatmap: abs ΔIV from previous timestamp — {opt_type}")
            ax_heat.set_ylabel("Option")
            ax_heat.set_xlabel("Timestamp")
            ax_heat.set_yticks(np.arange(len(pivot.index)))
            ax_heat.set_yticklabels(pivot.index, fontsize=7)
        else:
            ax_heat.set_title("No heatmap data")

        ax_path.clear()
        row = current_contract_row()
        contract = row["column"]
        label = contract_label(row)

        series = build_contract_calendar_series(df, contract)
        observed = series[series["iv"].notna()].copy()
        missing = series[series["is_missing_iv_at_dataset_timestamp"]].copy()

        if not observed.empty:
            ax_path.plot(observed["datetime_parsed"], observed["iv"], linewidth=1, alpha=0.6)
            ax_path.scatter(observed["datetime_parsed"], observed["iv"], s=22, alpha=0.9, label="Observed IV")
            y_min = max(0, observed["iv"].quantile(0.001) - 0.02)
            y_max = observed["iv"].quantile(0.999) + 0.02
        else:
            y_min, y_max = 0, 1

        missing_y = max(0, y_min - 0.015)
        if not missing.empty:
            ax_path.scatter(
                missing["datetime_parsed"],
                np.full(len(missing), missing_y),
                marker="x",
                s=50,
                label="Missing at dataset timestamp",
            )

        for d in pd.Series(df["datetime_parsed"].dt.normalize().unique()).sort_values():
            ax_path.axvline(pd.Timestamp(d), alpha=0.12, linewidth=2)

        ax_path.set_title(f"Real calendar-time IV path\n{label}")
        ax_path.set_xlabel("Calendar time, true 5-minute spacing")
        ax_path.set_ylabel("IV")
        ax_path.set_ylim(max(0, missing_y - 0.01), y_max)
        ax_path.grid(True, alpha=0.3)
        ax_path.legend(fontsize=8)
        fig.autofmt_xdate()

        indices = contract_indices_state["indices"]
        info_text.set_text(
            f"Option type filter: {opt_type} | "
            f"Contract {selected_pos_state['pos'] + 1}/{max(1, len(indices))}: {contract}"
        )
        fig.canvas.draw_idle()

    def on_radio(label):
        option_type_state["type"] = label
        reset_contract_indices(label)
        draw()

    def on_slider(value):
        selected_pos_state["pos"] = int(value)
        draw()

    def on_prev(_):
        selected_pos_state["pos"] = max(0, selected_pos_state["pos"] - 1)
        contract_slider.set_val(selected_pos_state["pos"])

    def on_next(_):
        n = len(contract_indices_state["indices"])
        selected_pos_state["pos"] = min(max(0, n - 1), selected_pos_state["pos"] + 1)
        contract_slider.set_val(selected_pos_state["pos"])

    radio.on_clicked(on_radio)
    contract_slider.on_changed(on_slider)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)

    draw()
    plt.show()


def main():
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(data_path)
    meta = parse_option_metadata(df)

    print(f"Rows: {len(df)}")
    print(f"Parsed option contracts: {len(meta)}")
    print(f"Date range: {df['datetime_parsed'].min()} to {df['datetime_parsed'].max()}")
    present_dates = [pd.Timestamp(x).date() for x in sorted(df["datetime_parsed"].dt.normalize().unique())]
    print("Dates present:", ", ".join(str(x) for x in present_dates))
    print(f"Output directory: {out_dir.resolve()}")

    print("\nBuilding lag-1 pair rows...")
    pair_rows = build_lag1_pair_rows(df, meta)
    metrics = compute_timestamp_metrics(pair_rows)
    option_metrics = compute_option_metrics(pair_rows)

    pair_rows.to_csv(out_dir / "lag1_pair_rows.csv", index=False)
    metrics.to_csv(out_dir / "lag1_timestamp_metrics.csv", index=False)
    option_metrics.to_csv(out_dir / "lag1_option_metrics.csv", index=False)

    print(f"Lag-1 pair rows: {len(pair_rows)}")
    print(f"Timestamp metric rows: {len(metrics)}")
    print(f"Option metric rows: {len(option_metrics)}")

    print("\nSaving static plots...")
    save_lag1_summary_plots(metrics, out_dir, args.max_xticks)
    save_lag1_heatmaps(pair_rows, df, out_dir)
    save_static_calendar_contract_plot(df, meta, out_dir, args.selected_contract)

    print("Saving interactive Plotly HTML...")
    create_plotly_html(df, meta, pair_rows, metrics, out_dir)

    print("\nSaved files:")
    for name in [
        "lag1_timestamp_metrics.csv",
        "lag1_pair_rows.csv",
        "lag1_option_metrics.csv",
        "lag1_summary_dashboard.png",
        "lag1_corr_by_timestamp.png",
        "lag1_abs_change_by_timestamp.png",
        "lag1_gap_minutes_by_timestamp.png",
        "lag1_heatmap_CE_abs_change.png",
        "lag1_heatmap_PE_abs_change.png",
        "lag1_heatmap_ALL_abs_change.png",
        "lag1_heatmap_CE_signed_change.png",
        "lag1_heatmap_PE_signed_change.png",
        "lag1_heatmap_ALL_signed_change.png",
        "calendar_gap_contract_dashboard.png",
        "calendar_gap_selected_contract.csv",
        "interactive_lag1_calendar_dashboard.html",
    ]:
        print("  ", out_dir / name)

    if args.show:
        make_interactive_matplotlib_dashboard(df, meta, pair_rows, metrics)


if __name__ == "__main__":
    main()
