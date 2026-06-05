"""
Matplotlib Contract Time-Series Dashboard: IV vs Equally Spaced Timestamp Index

Run this file from the same folder as dataset.csv, or keep the default path:

    python iv_contract_time_series_dashboard_equal_spacing.py

It opens an interactive matplotlib window.

The dashboard shows:
    x-axis: equally spaced timestamp index, not real datetime spacing
    y-axis: observed implied volatility
    selected contract: one NIFTY option column, e.g. NIFTY27JAN2625200CE
    observed points: actual non-missing IV values through time
    missing points: timestamp indices where that contract's IV is missing

Important:
    Missing IV values do not have a y-value, so this script places missing points
    on a horizontal reference band just below the visible IV region.

No IV prediction or filling is done.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/what_worked/dataset.csv")
# If your file is in the same directory as this script, use:
# DATA_PATH = Path("dataset.csv")


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and sort the competition dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.resolve()}.\n"
            "Put dataset.csv in the expected folder, or edit DATA_PATH."
        )

    df = pd.read_csv(data_path)

    required_cols = {"datetime", "underlying_price"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if df["datetime"].isna().any():
        bad_count = df["datetime"].isna().sum()
        raise ValueError(
            f"{bad_count} datetime values could not be parsed. "
            "Expected format: DD-MM-YYYY HH:MM"
        )

    return df.sort_values("datetime").reset_index(drop=True)


def parse_option_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse option columns like NIFTY27JAN2625200CE into:
        underlying, expiry, strike, option_type, column
    """
    id_cols = {"datetime", "underlying_price"}
    option_cols = [c for c in df.columns if c not in id_cols]

    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )

    records = []
    unparsed = []

    for col in option_cols:
        match = pattern.match(col)

        if match is None:
            unparsed.append(col)
            continue

        item = match.groupdict()
        item["column"] = col
        item["strike"] = int(item["strike"])
        item["expiry"] = pd.to_datetime(
            item["expiry"],
            format="%d%b%y",
            errors="coerce",
        )
        records.append(item)

    meta = pd.DataFrame(records)

    if meta.empty:
        raise ValueError("No option columns could be parsed. Check column names.")

    meta = meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)

    if unparsed:
        print("Warning: some non-ID columns could not be parsed as option columns:")
        for col in unparsed:
            print("   ", col)

    return meta


def contract_label(meta_row: pd.Series) -> str:
    """Readable label for contract selection."""
    return f"{meta_row['strike']} {meta_row['option_type']}  ({meta_row['column']})"


def make_contract_time_dashboard(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Create interactive IV vs equally spaced timestamp index dashboard.
    """
    x_index = np.arange(len(df), dtype=float)
    datetime_labels = df["datetime"].dt.strftime("%d-%m %H:%M").tolist()

    option_type_state = {"type": "ALL"}
    filtered_indices_state = {"indices": list(meta.index)}
    selected_pos_state = {"pos": 0}

    observed_all = df[meta["column"].tolist()].stack().dropna()

    if observed_all.empty:
        raise ValueError("No observed IV values found.")

    observed_y_min = max(0, observed_all.quantile(0.001) - 0.01)
    observed_y_max = observed_all.quantile(0.999) + 0.01

    missing_y = max(0, observed_y_min - 0.015)
    y_min = max(0, missing_y - 0.01)
    y_max = observed_y_max

    fig, ax = plt.subplots(figsize=(13, 7))
    plt.subplots_adjust(bottom=0.25, left=0.10, right=0.82)

    title = ax.set_title("")
    ax.set_xlabel("Equally Spaced Timestamp Index")
    ax.set_ylabel("Observed Implied Volatility")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # Equally spaced x-axis with sparse timestamp labels.
    max_ticks = 10
    tick_positions = np.linspace(0, len(df) - 1, min(max_ticks, len(df))).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([datetime_labels[i] for i in tick_positions], rotation=30, ha="right")

    ax.axhline(
        missing_y,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Missing IV reference level",
    )

    observed_scatter = ax.scatter(
        [],
        [],
        label="Observed IV",
        marker="o",
        alpha=0.85,
        s=35,
    )

    observed_line, = ax.plot(
        [],
        [],
        alpha=0.55,
        linewidth=1,
        label="Observed IV path",
    )

    missing_scatter = ax.scatter(
        [],
        [],
        label="Missing timestamp",
        marker="x",
        alpha=0.95,
        s=70,
    )

    ax.legend(loc="best")

    slider_ax = fig.add_axes([0.15, 0.10, 0.58, 0.035])
    contract_slider = Slider(
        ax=slider_ax,
        label="Contract Index",
        valmin=0,
        valmax=max(0, len(filtered_indices_state["indices"]) - 1),
        valinit=0,
        valstep=1,
    )

    prev_ax = fig.add_axes([0.15, 0.04, 0.10, 0.04])
    next_ax = fig.add_axes([0.27, 0.04, 0.10, 0.04])
    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")

    radio_ax = fig.add_axes([0.85, 0.40, 0.10, 0.18])
    radio = RadioButtons(radio_ax, ("ALL", "CE", "PE"), active=0)

    info_text = fig.text(
        0.42,
        0.045,
        "",
        fontsize=9,
        ha="left",
        va="center",
    )

    def get_current_meta_row():
        indices = filtered_indices_state["indices"]

        if not indices:
            raise ValueError("No contracts available under the current filter.")

        pos = int(selected_pos_state["pos"])
        pos = max(0, min(pos, len(indices) - 1))
        selected_pos_state["pos"] = pos

        return meta.loc[indices[pos]], pos, len(indices)

    def reset_slider_range():
        indices = filtered_indices_state["indices"]
        n = len(indices)

        contract_slider.valmax = max(0, n - 1)
        contract_slider.ax.set_xlim(contract_slider.valmin, contract_slider.valmax)

        if selected_pos_state["pos"] >= n:
            selected_pos_state["pos"] = max(0, n - 1)

        contract_slider.set_val(selected_pos_state["pos"])

    def update_plot(_=None):
        row, pos, n_contracts = get_current_meta_row()
        col = row["column"]

        series = df[col]
        observed_mask = series.notna().to_numpy()
        missing_mask = series.isna().to_numpy()

        obs_x = x_index[observed_mask]
        obs_y = series[observed_mask].to_numpy(dtype=float)

        miss_x = x_index[missing_mask]
        miss_y = np.full(len(miss_x), missing_y)

        if len(obs_x):
            observed_scatter.set_offsets(np.column_stack([obs_x, obs_y]))
            observed_line.set_data(obs_x, obs_y)
        else:
            observed_scatter.set_offsets(np.empty((0, 2)))
            observed_line.set_data([], [])

        if len(miss_x):
            missing_scatter.set_offsets(np.column_stack([miss_x, miss_y]))
        else:
            missing_scatter.set_offsets(np.empty((0, 2)))

        n_obs = int(observed_mask.sum())
        n_missing = int(missing_mask.sum())
        missing_pct = 100 * n_missing / len(series)

        first_ts = df["datetime"].iloc[0].strftime("%d-%m-%Y %H:%M")
        last_ts = df["datetime"].iloc[-1].strftime("%d-%m-%Y %H:%M")

        title.set_text(
            f"IV vs Time Index: {contract_label(row)} | "
            f"Observed: {n_obs} | Missing: {n_missing} ({missing_pct:.1f}%)"
        )

        info_text.set_text(
            f"Filter: {option_type_state['type']} | "
            f"Showing {pos + 1}/{n_contracts} | "
            f"Strike: {row['strike']} | Type: {row['option_type']} | "
            f"Range: {first_ts} to {last_ts}"
        )

        ax.set_xlim(-1, len(df))

        fig.canvas.draw_idle()

    def on_slider_change(value):
        selected_pos_state["pos"] = int(value)
        update_plot()

    def on_prev(_):
        indices = filtered_indices_state["indices"]
        if not indices:
            return

        selected_pos_state["pos"] = max(0, selected_pos_state["pos"] - 1)
        contract_slider.set_val(selected_pos_state["pos"])

    def on_next(_):
        indices = filtered_indices_state["indices"]
        if not indices:
            return

        selected_pos_state["pos"] = min(len(indices) - 1, selected_pos_state["pos"] + 1)
        contract_slider.set_val(selected_pos_state["pos"])

    def on_radio(label):
        option_type_state["type"] = label

        if label == "ALL":
            filtered_indices = list(meta.index)
        else:
            filtered_indices = list(meta.index[meta["option_type"] == label])

        filtered_indices_state["indices"] = filtered_indices
        selected_pos_state["pos"] = 0
        reset_slider_range()
        update_plot()

    contract_slider.on_changed(on_slider_change)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    radio.on_clicked(on_radio)

    update_plot()

    print("Controls:")
    print("  Contract Index slider: choose contract")
    print("  Previous / Next buttons: move one contract")
    print("  Radio buttons: filter ALL / CE / PE")
    print("  X-axis is equally spaced timestamp index, not real datetime distance.")

    plt.show()


def main() -> None:
    df = load_dataset(DATA_PATH)
    meta = parse_option_metadata(df)

    print("Loaded dataset:", DATA_PATH.resolve())
    print("Rows:", len(df))
    print("Parsed option contracts:", len(meta))
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
    print("Available contracts:")
    for i, row in meta.iterrows():
        print(f"  {i}: {contract_label(row)}")

    make_contract_time_dashboard(df, meta)


if __name__ == "__main__":
    main()
