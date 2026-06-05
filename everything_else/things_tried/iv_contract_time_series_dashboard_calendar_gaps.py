"""
Matplotlib Contract Time-Series Dashboard:
IV vs Real Calendar Time with 5-Minute Spacing and Date Gaps Preserved

Run this file from the same folder as dataset.csv, or edit DATA_PATH:

    python iv_contract_time_series_dashboard_calendar_gaps.py

It opens an interactive matplotlib window.

The dashboard shows:
    x-axis: real calendar time compressed into 5-minute slots
    y-axis: observed implied volatility
    selected contract: one NIFTY option column, e.g. NIFTY27JAN2625200CE
    observed points: actual non-missing IV values through time
    missing points: dataset timestamps where that contract's IV is missing

Important:
    This is NOT equally spaced by available rows.

    If the data jumps from 9 Jan to 12 Jan, the x-axis keeps the blank space
    for 10 Jan and 11 Jan. This lets you visually inspect whether IV could be
    modeled as a smooth calendar-time curve and where holidays/weekends break it.

    Missing IV values do not have a y-value, so this script places missing
    points on a horizontal reference band just below the visible IV region.

No IV prediction or filling is done.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/cv_validation_system/dataset.csv")
# Example absolute path:
# DATA_PATH = Path("/Users/swaritthakkar/Documents/IIT R/Second Sem/finclub-open-project-26/what_worked/dataset.csv")

BASE_STEP_MINUTES = 5


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


def build_calendar_x_index(df: pd.DataFrame) -> tuple[np.ndarray, pd.Timestamp, float]:
    """
    Convert real datetimes into 5-minute calendar slots.

    Example:
        first timestamp -> x = 0
        5 minutes later -> x = 1
        1 calendar day later -> x = 288

    This preserves weekend/holiday/date gaps as blank horizontal space.
    """
    start_time = df["datetime"].min()
    minutes_from_start = (df["datetime"] - start_time).dt.total_seconds().to_numpy() / 60.0
    x_calendar = minutes_from_start / BASE_STEP_MINUTES
    return x_calendar.astype(float), start_time, float(BASE_STEP_MINUTES)


def choose_calendar_ticks(df: pd.DataFrame, x_calendar: np.ndarray, max_ticks: int = 14):
    """
    Choose readable x-axis ticks.

    Prefer one tick per available date, plus a few intraday labels if the
    number of dates is small.
    """
    tmp = pd.DataFrame({
        "datetime": df["datetime"],
        "x": x_calendar,
        "date": df["datetime"].dt.date,
    })

    # One tick at first available timestamp of each date.
    date_ticks = tmp.groupby("date", as_index=False).first()

    if len(date_ticks) <= max_ticks:
        tick_positions = date_ticks["x"].to_numpy()
        tick_labels = [
            pd.Timestamp(dt).strftime("%d-%m\n%H:%M")
            for dt in date_ticks["datetime"]
        ]
        return tick_positions, tick_labels

    # If too many dates, spread ticks uniformly across available timestamps.
    pos = np.linspace(0, len(df) - 1, min(max_ticks, len(df))).astype(int)
    tick_positions = x_calendar[pos]
    tick_labels = [
        df["datetime"].iloc[i].strftime("%d-%m\n%H:%M")
        for i in pos
    ]
    return tick_positions, tick_labels


def make_contract_time_dashboard(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Create interactive IV vs real calendar-time dashboard.

    Unlike the earlier equal-spacing version:
        x is not 0,1,2,3,...
        x is number of 5-minute calendar slots since the first timestamp.

    So non-trading days / missing dates remain visually blank.
    """
    x_calendar, start_time, step_minutes = build_calendar_x_index(df)
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

    fig, ax = plt.subplots(figsize=(15, 7))
    plt.subplots_adjust(bottom=0.27, left=0.10, right=0.82)

    title = ax.set_title("")
    ax.set_xlabel(f"Real Calendar Time Index ({BASE_STEP_MINUTES}-minute slots; gaps preserved)")
    ax.set_ylabel("Observed Implied Volatility")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # Calendar-spaced x-axis with date ticks.
    tick_positions, tick_labels = choose_calendar_ticks(df, x_calendar, max_ticks=14)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    # Mark the start of each available date with a faint vertical line.
    day_start_positions = (
        pd.DataFrame({"datetime": df["datetime"], "x": x_calendar, "date": df["datetime"].dt.date})
        .groupby("date", as_index=False)
        .first()
    )
    day_lines = []
    for _, rec in day_start_positions.iterrows():
        line = ax.axvline(
            rec["x"],
            linewidth=1,
            alpha=0.12,
        )
        day_lines.append(line)

    # Mark large gaps between consecutive available timestamps.
    # This makes weekend/holiday jumps visually obvious.
    gap_lines = []
    gap_minutes = df["datetime"].diff().dt.total_seconds().div(60.0)
    large_gap_indices = np.where(gap_minutes.to_numpy() > BASE_STEP_MINUTES + 1e-9)[0]

    for idx in large_gap_indices:
        line = ax.axvline(
            x_calendar[idx],
            linestyle="--",
            linewidth=1.2,
            alpha=0.30,
        )
        gap_lines.append(line)

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

    slider_ax = fig.add_axes([0.15, 0.11, 0.58, 0.035])
    contract_slider = Slider(
        ax=slider_ax,
        label="Contract Index",
        valmin=0,
        valmax=max(0, len(filtered_indices_state["indices"]) - 1),
        valinit=0,
        valstep=1,
    )

    prev_ax = fig.add_axes([0.15, 0.045, 0.10, 0.04])
    next_ax = fig.add_axes([0.27, 0.045, 0.10, 0.04])
    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")

    radio_ax = fig.add_axes([0.85, 0.40, 0.10, 0.18])
    radio = RadioButtons(radio_ax, ("ALL", "CE", "PE"), active=0)

    info_text = fig.text(
        0.42,
        0.055,
        "",
        fontsize=9,
        ha="left",
        va="center",
    )

    gap_info_text = fig.text(
        0.42,
        0.025,
        "",
        fontsize=8,
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

        obs_x = x_calendar[observed_mask]
        obs_y = series[observed_mask].to_numpy(dtype=float)

        miss_x = x_calendar[missing_mask]
        miss_y_values = np.full(len(miss_x), missing_y)

        if len(obs_x):
            observed_scatter.set_offsets(np.column_stack([obs_x, obs_y]))
            observed_line.set_data(obs_x, obs_y)
        else:
            observed_scatter.set_offsets(np.empty((0, 2)))
            observed_line.set_data([], [])

        if len(miss_x):
            missing_scatter.set_offsets(np.column_stack([miss_x, miss_y_values]))
        else:
            missing_scatter.set_offsets(np.empty((0, 2)))

        n_obs = int(observed_mask.sum())
        n_missing = int(missing_mask.sum())
        missing_pct = 100 * n_missing / len(series)

        first_ts = df["datetime"].iloc[0].strftime("%d-%m-%Y %H:%M")
        last_ts = df["datetime"].iloc[-1].strftime("%d-%m-%Y %H:%M")

        total_calendar_steps = int(round(x_calendar[-1] - x_calendar[0]))
        available_rows = len(df)

        title.set_text(
            f"IV vs Real Calendar Time: {contract_label(row)} | "
            f"Observed: {n_obs} | Missing: {n_missing} ({missing_pct:.1f}%)"
        )

        info_text.set_text(
            f"Filter: {option_type_state['type']} | "
            f"Showing {pos + 1}/{n_contracts} | "
            f"Strike: {row['strike']} | Type: {row['option_type']} | "
            f"Range: {first_ts} to {last_ts}"
        )

        gap_info_text.set_text(
            f"X-axis uses real 5-minute calendar slots. "
            f"Available rows: {available_rows}; calendar span: {total_calendar_steps} five-minute slots. "
            f"Dashed vertical lines mark jumps larger than 5 minutes."
        )

        ax.set_xlim(x_calendar.min() - 2, x_calendar.max() + 2)

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
    print("  X-axis is real calendar time in 5-minute slots, so missing dates remain as blank space.")
    print("  Dashed vertical lines mark jumps larger than 5 minutes.")

    plt.show()


def main() -> None:
    df = load_dataset(DATA_PATH)
    meta = parse_option_metadata(df)

    x_calendar, _, _ = build_calendar_x_index(df)

    print("Loaded dataset:", DATA_PATH.resolve())
    print("Rows:", len(df))
    print("Parsed option contracts:", len(meta))
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
    print("Calendar span in 5-minute slots:", int(round(x_calendar[-1] - x_calendar[0])))
    print("Available dates:")
    for d in sorted(df["datetime"].dt.date.unique()):
        print("  ", d)
    print("Available contracts:")
    for i, row in meta.iterrows():
        print(f"  {i}: {contract_label(row)}")

    make_contract_time_dashboard(df, meta)


if __name__ == "__main__":
    main()
