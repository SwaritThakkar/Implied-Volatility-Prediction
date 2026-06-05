"""
Matplotlib Time-Slider Dashboard: IV vs Scaled Log-Moneyness

Run this file from the same folder as dataset.csv:

    python iv_moneyness_time_slider_matplotlib.py

It opens an interactive matplotlib window with a timestamp slider.

At the selected timestamp, the plot shows:
    x-axis: scaled log-moneyness = z-score(log(strike / underlying_price))
    y-axis: observed implied volatility
    points: all observed option contracts at that timestamp
    marker/color: CE vs PE

No IV prediction or filling is done.
Missing IV values are simply not plotted.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


DATA_PATH = Path("dataset.csv")


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and sort the competition dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.resolve()}.\n"
            "Put dataset.csv in the same folder as this script, "
            "or edit DATA_PATH at the top of the file."
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

    if unparsed:
        print("Warning: some non-ID columns could not be parsed as option columns:")
        for col in unparsed:
            print("   ", col)

    return meta


def build_long_plot_data(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide IV table into long format and compute log-moneyness.

    log_moneyness = log(strike / underlying_price)

    scaled_log_moneyness is computed globally across all rows and contracts:
        (log_moneyness - mean) / std
    """
    option_cols = meta["column"].tolist()

    long_df = df[["datetime", "underlying_price"] + option_cols].melt(
        id_vars=["datetime", "underlying_price"],
        value_vars=option_cols,
        var_name="contract",
        value_name="iv",
    )

    long_df = long_df.merge(
        meta[["column", "strike", "option_type", "expiry"]],
        left_on="contract",
        right_on="column",
        how="left",
    ).drop(columns=["column"])

    long_df["log_moneyness"] = np.log(
        long_df["strike"] / long_df["underlying_price"]
    )

    mean_log_m = long_df["log_moneyness"].mean()
    std_log_m = long_df["log_moneyness"].std()

    if std_log_m == 0 or np.isnan(std_log_m):
        raise ValueError("Cannot scale log-moneyness because standard deviation is zero/NaN.")

    long_df["scaled_log_moneyness"] = long_df["log_moneyness"]

    # EDA-only: only observed IV values are plotted.
    plot_data = long_df.dropna(subset=["iv"]).copy()

    return plot_data


def make_slider_dashboard(plot_data: pd.DataFrame, use_scaled_moneyness: bool = True) -> None:
    """
    Create the matplotlib slider dashboard.

    Parameters
    ----------
    plot_data:
        Long dataframe containing datetime, IV, moneyness columns, option type.
    use_scaled_moneyness:
        True  -> x-axis is scaled_log_moneyness
        False -> x-axis is raw log_moneyness
    """
    x_col = "scaled_log_moneyness" if use_scaled_moneyness else "log_moneyness"
    x_label = (
        "Scaled Log-Moneyness = z-score(log(strike / underlying_price))"
        if use_scaled_moneyness
        else "Log-Moneyness = log(strike / underlying_price)"
    )

    timestamps = np.array(sorted(plot_data["datetime"].dropna().unique()))

    if len(timestamps) == 0:
        raise ValueError("No timestamps with observed IV values were found.")

    by_time = {
        ts: sub.sort_values(["option_type", x_col])
        for ts, sub in plot_data.groupby("datetime")
    }

    x_margin = 0.25 if use_scaled_moneyness else 0.005
    x_min = plot_data[x_col].min() - x_margin
    x_max = plot_data[x_col].max() + x_margin

    y_min = max(0, plot_data["iv"].quantile(0.001) - 0.01)
    y_max = plot_data["iv"].quantile(0.999) + 0.01

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.18)

    title = ax.set_title("")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Observed Implied Volatility")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    initial_ts = timestamps[0]
    initial_data = by_time.get(initial_ts, pd.DataFrame())

    ce_initial = initial_data[initial_data["option_type"] == "CE"]
    pe_initial = initial_data[initial_data["option_type"] == "PE"]

    ce_scatter = ax.scatter(
        ce_initial[x_col],
        ce_initial["iv"],
        label="CE",
        marker="o",
        alpha=0.8,
    )
    pe_scatter = ax.scatter(
        pe_initial[x_col],
        pe_initial["iv"],
        label="PE",
        marker="x",
        alpha=0.8,
    )

    # Lines help make the smile/skew shape easier to see.
    ce_line, = ax.plot(
        ce_initial[x_col],
        ce_initial["iv"],
        alpha=0.45,
        linewidth=1,
    )
    pe_line, = ax.plot(
        pe_initial[x_col],
        pe_initial["iv"],
        alpha=0.45,
        linewidth=1,
    )

    ax.legend()

    slider_ax = fig.add_axes([0.15, 0.06, 0.70, 0.035])
    time_slider = Slider(
        ax=slider_ax,
        label="Timestamp Index",
        valmin=0,
        valmax=len(timestamps) - 1,
        valinit=0,
        valstep=1,
    )

    def update_plot(_):
        idx = int(time_slider.val)
        ts = timestamps[idx]
        data = by_time.get(ts, pd.DataFrame())

        ce = data[data["option_type"] == "CE"]
        pe = data[data["option_type"] == "PE"]

        ce_xy = (
            np.column_stack([ce[x_col], ce["iv"]])
            if len(ce)
            else np.empty((0, 2))
        )
        pe_xy = (
            np.column_stack([pe[x_col], pe["iv"]])
            if len(pe)
            else np.empty((0, 2))
        )

        ce_scatter.set_offsets(ce_xy)
        pe_scatter.set_offsets(pe_xy)

        ce_line.set_data(ce[x_col], ce["iv"])
        pe_line.set_data(pe[x_col], pe["iv"])

        if len(data):
            spot = data["underlying_price"].iloc[0]
            n_obs = len(data)
        else:
            spot = np.nan
            n_obs = 0

        title.set_text(
            f"IV Smile at {pd.Timestamp(ts).strftime('%d-%m-%Y %H:%M')} | "
            f"Spot: {spot:.2f} | Observed contracts: {n_obs}"
        )

        fig.canvas.draw_idle()

    time_slider.on_changed(update_plot)
    update_plot(None)

    plt.show()


def main() -> None:
    df = load_dataset(DATA_PATH)
    meta = parse_option_metadata(df)
    plot_data = build_long_plot_data(df, meta)

    print("Loaded dataset:", DATA_PATH.resolve())
    print("Rows:", len(df))
    print("Parsed option contracts:", len(meta))
    print("Observed IV points:", len(plot_data))
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())

    make_slider_dashboard(plot_data, use_scaled_moneyness=False)


if __name__ == "__main__":
    main()
