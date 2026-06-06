"""
Matplotlib Time-Slider Dashboard: IV vs Log-Moneyness, with Missing Points

Run this file from the same folder as dataset.csv:

    python iv_moneyness_time_slider_matplotlib_missing.py

It opens an interactive matplotlib window with a timestamp slider.

At the selected timestamp, the plot shows:
    x-axis: log-moneyness = log(strike / underlying_price)
    y-axis: observed implied volatility
    observed points: actual non-missing IV values
    missing points: contracts whose IV is missing at that timestamp

Important:
    Missing IV values do not have a y-value, so this script places missing points
    on a horizontal reference band just below the visible IV region. Their x-position
    is still their true log-moneyness.

No IV prediction or filling is done.
Nothing is connected with lines.
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

    This keeps both observed and missing IV rows.
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

    long_df["is_missing_iv"] = long_df["iv"].isna()

    return long_df


def make_slider_dashboard(plot_data: pd.DataFrame) -> None:
    """
    Create the matplotlib slider dashboard.
    """
    x_col = "log_moneyness"
    x_label = "Log-Moneyness = log(strike / underlying_price)"

    timestamps = np.array(sorted(plot_data["datetime"].dropna().unique()))

    if len(timestamps) == 0:
        raise ValueError("No timestamps were found.")

    by_time = {
        ts: sub.sort_values(["option_type", x_col])
        for ts, sub in plot_data.groupby("datetime")
    }

    observed_all = plot_data.dropna(subset=["iv"]).copy()

    if observed_all.empty:
        raise ValueError("No observed IV values found.")

    x_margin = 0.005
    x_min = plot_data[x_col].min() - x_margin
    x_max = plot_data[x_col].max() + x_margin

    observed_y_min = max(0, observed_all["iv"].quantile(0.001) - 0.01)
    observed_y_max = observed_all["iv"].quantile(0.999) + 0.01

    # Missing points have no IV, so place them on a visible reference band.
    # This is deliberately below the IV cloud, not a fake IV prediction.
    missing_y = max(0, observed_y_min - 0.015)
    y_min = max(0, missing_y - 0.01)
    y_max = observed_y_max

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.18)

    title = ax.set_title("")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Observed Implied Volatility")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # Reference line for missing values.
    ax.axhline(
        missing_y,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Missing IV reference level",
    )

    initial_ts = timestamps[0]
    initial_data = by_time.get(initial_ts, pd.DataFrame())

    def split_data(data: pd.DataFrame):
        obs = data[~data["is_missing_iv"]]
        miss = data[data["is_missing_iv"]]

        obs_ce = obs[obs["option_type"] == "CE"]
        obs_pe = obs[obs["option_type"] == "PE"]

        miss_ce = miss[miss["option_type"] == "CE"]
        miss_pe = miss[miss["option_type"] == "PE"]

        return obs_ce, obs_pe, miss_ce, miss_pe

    obs_ce0, obs_pe0, miss_ce0, miss_pe0 = split_data(initial_data)

    # Observed IV points.
    obs_ce_scatter = ax.scatter(
        obs_ce0[x_col],
        obs_ce0["iv"],
        label="Observed CE",
        marker="o",
        alpha=0.85,
        s=45,
    )

    obs_pe_scatter = ax.scatter(
        obs_pe0[x_col],
        obs_pe0["iv"],
        label="Observed PE",
        marker="x",
        alpha=0.85,
        s=55,
    )

    # Missing IV locations.
    # These points are placed at missing_y because their true IV is unknown.
    miss_ce_scatter = ax.scatter(
        miss_ce0[x_col],
        np.full(len(miss_ce0), missing_y),
        label="Missing CE",
        marker="v",
        alpha=0.95,
        s=70,
    )

    miss_pe_scatter = ax.scatter(
        miss_pe0[x_col],
        np.full(len(miss_pe0), missing_y),
        label="Missing PE",
        marker="^",
        alpha=0.95,
        s=70,
    )

    ax.legend(loc="best")

    slider_ax = fig.add_axes([0.15, 0.06, 0.70, 0.035])
    time_slider = Slider(
        ax=slider_ax,
        label="Timestamp Index",
        valmin=0,
        valmax=len(timestamps) - 1,
        valinit=0,
        valstep=1,
    )

    def xy_observed(data: pd.DataFrame) -> np.ndarray:
        if len(data) == 0:
            return np.empty((0, 2))
        return np.column_stack([data[x_col], data["iv"]])

    def xy_missing(data: pd.DataFrame) -> np.ndarray:
        if len(data) == 0:
            return np.empty((0, 2))
        return np.column_stack([data[x_col], np.full(len(data), missing_y)])

    def update_plot(_):
        idx = int(time_slider.val)
        ts = timestamps[idx]
        data = by_time.get(ts, pd.DataFrame())

        obs_ce, obs_pe, miss_ce, miss_pe = split_data(data)

        obs_ce_scatter.set_offsets(xy_observed(obs_ce))
        obs_pe_scatter.set_offsets(xy_observed(obs_pe))

        miss_ce_scatter.set_offsets(xy_missing(miss_ce))
        miss_pe_scatter.set_offsets(xy_missing(miss_pe))

        if len(data):
            spot = data["underlying_price"].iloc[0]
            n_obs = (~data["is_missing_iv"]).sum()
            n_missing = data["is_missing_iv"].sum()
        else:
            spot = np.nan
            n_obs = 0
            n_missing = 0

        title.set_text(
            f"IV Smile at {pd.Timestamp(ts).strftime('%d-%m-%Y %H:%M')} | "
            f"Spot: {spot:.2f} | Observed: {n_obs} | Missing: {n_missing}"
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
    print("Observed IV points:", plot_data["iv"].notna().sum())
    print("Missing IV points:", plot_data["iv"].isna().sum())
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())

    make_slider_dashboard(plot_data)


if __name__ == "__main__":
    main()
