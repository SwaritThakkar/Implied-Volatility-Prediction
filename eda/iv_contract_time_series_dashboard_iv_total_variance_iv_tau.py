"""
Matplotlib Contract Time-Series Dashboard:
IV and Total Variance vs Equally Spaced Timestamp Index

Run this file from the same folder as dataset.csv, or keep the default path:

    python iv_contract_time_series_dashboard_iv_and_total_variance.py

It opens an interactive matplotlib window.

The dashboard shows one selected contract at a time.

You can choose the y-axis display:
    1. IV
    2. Total variance = IV^2 * tau
    3. IV * tau

where:
    tau = time to expiry in years

The x-axis is equally spaced timestamp index, not real datetime spacing.

Missing IV values do not have a y-value, so this script places missing points
on a horizontal reference band just below the visible y-region.

No IV prediction or filling is done.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


DATA_PATH = Path("what_worked/dataset.csv")
# If your file is in the same directory as this script, use:
# DATA_PATH = Path("dataset.csv")

EXPIRY_MARKET_CLOSE_TIME = "15:30"
YEAR_DAYS = 365.0
MIN_TAU_YEARS = 1.0 / (365.0 * 24.0 * 60.0)  # one minute in years


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
        item["expiry_datetime"] = build_expiry_datetime(item["expiry"])
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


def build_expiry_datetime(expiry_date: pd.Timestamp) -> pd.Timestamp:
    """
    Convert expiry date into an expiry datetime at market close.

    For this dataset, option names are like NIFTY27JAN26xxxxxCE/PE,
    so the expiry date is 27 Jan 2026. We use market close rather than
    midnight so that 27 Jan morning still has positive time to expiry.
    """
    if pd.isna(expiry_date):
        return pd.NaT

    date_str = pd.Timestamp(expiry_date).strftime("%Y-%m-%d")
    return pd.Timestamp(f"{date_str} {EXPIRY_MARKET_CLOSE_TIME}")


def compute_tau_years(datetimes: pd.Series, expiry_datetime: pd.Timestamp) -> np.ndarray:
    """
    Compute time to expiry in years.

    tau = max((expiry_datetime - current_datetime) in years, MIN_TAU_YEARS)
    """
    if pd.isna(expiry_datetime):
        return np.full(len(datetimes), MIN_TAU_YEARS, dtype=float)

    seconds = (expiry_datetime - datetimes).dt.total_seconds().to_numpy(dtype=float)
    tau = seconds / (YEAR_DAYS * 24.0 * 3600.0)
    tau = np.maximum(tau, MIN_TAU_YEARS)
    return tau


def contract_label(meta_row: pd.Series) -> str:
    """Readable label for contract selection."""
    expiry = pd.Timestamp(meta_row["expiry"]).strftime("%d-%b-%Y")
    return f"{meta_row['strike']} {meta_row['option_type']}  ({meta_row['column']}) | Expiry: {expiry}"


def make_contract_time_dashboard(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Create interactive dashboard:
        IV vs equally spaced timestamp index
        or total variance = IV^2 * tau vs equally spaced timestamp index
    """
    x_index = np.arange(len(df), dtype=float)
    datetime_labels = df["datetime"].dt.strftime("%d-%m %H:%M").tolist()

    option_type_state = {"type": "ALL"}
    y_mode_state = {"mode": "IV"}
    filtered_indices_state = {"indices": list(meta.index)}
    selected_pos_state = {"pos": 0}

    option_cols = meta["column"].tolist()
    observed_all = df[option_cols].stack().dropna()

    if observed_all.empty:
        raise ValueError("No observed IV values found.")

    fig, ax = plt.subplots(figsize=(13, 7))
    plt.subplots_adjust(bottom=0.25, left=0.10, right=0.82)

    title = ax.set_title("")
    ax.set_xlabel("Equally Spaced Timestamp Index")
    ax.grid(True, alpha=0.3)

    # Equally spaced x-axis with sparse timestamp labels.
    max_ticks = 10
    tick_positions = np.linspace(0, len(df) - 1, min(max_ticks, len(df))).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([datetime_labels[i] for i in tick_positions], rotation=30, ha="right")

    # This reference line is updated because y-scale changes between IV and total variance.
    missing_ref_line = ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Missing value reference level",
    )

    observed_scatter = ax.scatter(
        [],
        [],
        label="Observed value",
        marker="o",
        alpha=0.85,
        s=35,
    )

    observed_line, = ax.plot(
        [],
        [],
        alpha=0.55,
        linewidth=1,
        label="Observed path",
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

    type_radio_ax = fig.add_axes([0.85, 0.47, 0.10, 0.18])
    type_radio = RadioButtons(type_radio_ax, ("ALL", "CE", "PE"), active=0)

    y_radio_ax = fig.add_axes([0.85, 0.20, 0.13, 0.20])
    y_radio = RadioButtons(y_radio_ax, ("IV", "IV² × tau", "IV × tau"), active=0)

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

    def compute_y_values(series: pd.Series, meta_row: pd.Series):
        """
        Return y-values for the selected display mode.

        IV mode:
            y = IV

        Total variance mode:
            y = IV^2 * tau

        IV times tau mode:
            y = IV * tau
        """
        iv = series.to_numpy(dtype=float)

        if y_mode_state["mode"] == "IV":
            y = iv
            ylabel = "Observed Implied Volatility"
            mode_label = "IV"
            tau = None
        elif y_mode_state["mode"] == "IV² × tau":
            tau = compute_tau_years(df["datetime"], meta_row["expiry_datetime"])
            y = (iv ** 2) * tau
            ylabel = "Observed Total Variance = IV² × tau"
            mode_label = "IV² × tau"
        elif y_mode_state["mode"] == "IV × tau":
            tau = compute_tau_years(df["datetime"], meta_row["expiry_datetime"])
            y = iv * tau
            ylabel = "Observed IV × tau"
            mode_label = "IV × tau"
        else:
            raise ValueError(f"Unknown y-axis mode: {y_mode_state['mode']}")

        return y, ylabel, mode_label, tau

    def update_plot(_=None):
        row, pos, n_contracts = get_current_meta_row()
        col = row["column"]

        series = df[col]
        observed_mask = series.notna().to_numpy()
        missing_mask = series.isna().to_numpy()

        y_values, ylabel, mode_label, tau = compute_y_values(series, row)

        observed_y = y_values[observed_mask]
        finite_observed_y = observed_y[np.isfinite(observed_y)]

        if len(finite_observed_y) == 0:
            y_low = 0.0
            y_high = 1.0
            missing_y_level = 0.0
        else:
            observed_y_min = max(0, np.nanquantile(finite_observed_y, 0.001) - 0.05 * np.nanstd(finite_observed_y))
            observed_y_max = np.nanquantile(finite_observed_y, 0.999) + 0.05 * np.nanstd(finite_observed_y)

            if not np.isfinite(observed_y_min):
                observed_y_min = max(0, float(np.nanmin(finite_observed_y)))
            if not np.isfinite(observed_y_max) or observed_y_max <= observed_y_min:
                observed_y_max = float(np.nanmax(finite_observed_y)) + 1e-6

            pad = max(1e-8, 0.08 * (observed_y_max - observed_y_min))
            missing_y_level = max(0, observed_y_min - pad)
            y_low = max(0, missing_y_level - pad)
            y_high = observed_y_max + pad

        obs_x = x_index[observed_mask]
        obs_y = y_values[observed_mask]

        miss_x = x_index[missing_mask]
        miss_y_values = np.full(len(miss_x), missing_y_level)

        if len(obs_x):
            good_obs = np.isfinite(obs_y)
            observed_scatter.set_offsets(np.column_stack([obs_x[good_obs], obs_y[good_obs]]))
            observed_line.set_data(obs_x[good_obs], obs_y[good_obs])
        else:
            observed_scatter.set_offsets(np.empty((0, 2)))
            observed_line.set_data([], [])

        if len(miss_x):
            missing_scatter.set_offsets(np.column_stack([miss_x, miss_y_values]))
        else:
            missing_scatter.set_offsets(np.empty((0, 2)))

        missing_ref_line.set_ydata([missing_y_level, missing_y_level])

        n_obs = int(observed_mask.sum())
        n_missing = int(missing_mask.sum())
        missing_pct = 100 * n_missing / len(series)

        first_ts = df["datetime"].iloc[0].strftime("%d-%m-%Y %H:%M")
        last_ts = df["datetime"].iloc[-1].strftime("%d-%m-%Y %H:%M")

        if tau is not None:
            tau_first = float(tau[0])
            tau_last = float(tau[-1])
            tau_text = f" | tau range: {tau_first:.6f} to {tau_last:.6f} yrs"
        else:
            tau_text = ""

        ax.set_ylabel(ylabel)
        ax.set_ylim(y_low, y_high)
        ax.set_xlim(-1, len(df))

        title.set_text(
            f"{mode_label} vs Time Index: {contract_label(row)} | "
            f"Observed: {n_obs} | Missing: {n_missing} ({missing_pct:.1f}%)"
        )

        info_text.set_text(
            f"Filter: {option_type_state['type']} | "
            f"Y: {mode_label} | "
            f"Showing {pos + 1}/{n_contracts} | "
            f"Strike: {row['strike']} | Type: {row['option_type']} | "
            f"Range: {first_ts} to {last_ts}{tau_text}"
        )

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

    def on_type_radio(label):
        option_type_state["type"] = label

        if label == "ALL":
            filtered_indices = list(meta.index)
        else:
            filtered_indices = list(meta.index[meta["option_type"] == label])

        filtered_indices_state["indices"] = filtered_indices
        selected_pos_state["pos"] = 0
        reset_slider_range()
        update_plot()

    def on_y_radio(label):
        y_mode_state["mode"] = label
        update_plot()

    contract_slider.on_changed(on_slider_change)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    type_radio.on_clicked(on_type_radio)
    y_radio.on_clicked(on_y_radio)

    update_plot()

    print("Controls:")
    print("  Contract Index slider: choose contract")
    print("  Previous / Next buttons: move one contract")
    print("  Type radio buttons: filter ALL / CE / PE")
    print("  Y radio buttons: switch IV vs IV^2 * tau vs IV * tau")
    print("  X-axis is equally spaced timestamp index, not real datetime distance.")
    print("")
    print("Tau calculation:")
    print(f"  expiry timestamp = option expiry date at {EXPIRY_MARKET_CLOSE_TIME}")
    print("  tau = max((expiry_timestamp - current_timestamp) / 365 days, one minute in years)")

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
