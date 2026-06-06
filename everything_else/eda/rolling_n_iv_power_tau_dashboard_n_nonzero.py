"""
Matplotlib Contract Dashboard: Rolling n and IV^n * tau

Idea:
    Find n such that:

        IV^n * tau = k = constant

From:
    d/dt [IV^n * tau] = 0

where t is elapsed time in years and tau is time-to-expiry in years.

Since:
    d(tau)/dt = -1

we get:
    n * IV^(n-1) * IV' * tau - IV^n = 0

Divide by IV^(n-1):
    n * IV' * tau = IV

So:
    n = IV / (IV' * tau)

Important:
    IV' must be derivative with respect to elapsed years.

This script computes:
    iv_prime = dIV / dt_years
    raw_n = IV / (iv_prime * tau)
    enforce n != 0 by discarding |raw_n| <= N_EPS
    rolling_n = rolling median(raw_n)
    k_value = IV ** rolling_n * tau

It opens an interactive matplotlib dashboard where you can select one contract.
The x-axis is equally spaced timestamp index.

No IV prediction or filling is done.
Missing IV values are skipped in derivative and k calculations.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


DATA_PATH = Path("dataset.csv")
# If your file is in the same directory as this script, use:
# DATA_PATH = Path("dataset.csv")

EXPIRY_MARKET_CLOSE_TIME = "15:30"
YEAR_DAYS = 365.0
MIN_TAU_YEARS = 1.0 / (365.0 * 24.0 * 60.0)  # one minute in years

# Rolling window over observed points, not raw row count.
ROLLING_WINDOW_OBS = 12

# Very large n values usually happen when IV' is almost zero.
# Clipping makes the plot readable without changing the raw_n column internally.
PLOT_N_CLIP = 50.0

# If abs(iv_prime * tau) is too small, n explodes. Mark as NaN.
DENOM_EPS = 1e-10

# Constraint: n must not be zero.
# Values with |n| <= N_EPS are treated as invalid.
N_EPS = 1e-8


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
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


def build_expiry_datetime(expiry_date: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(expiry_date):
        return pd.NaT

    date_str = pd.Timestamp(expiry_date).strftime("%Y-%m-%d")
    return pd.Timestamp(f"{date_str} {EXPIRY_MARKET_CLOSE_TIME}")


def compute_tau_years(datetimes: pd.Series, expiry_datetime: pd.Timestamp) -> np.ndarray:
    if pd.isna(expiry_datetime):
        return np.full(len(datetimes), MIN_TAU_YEARS, dtype=float)

    seconds = (expiry_datetime - datetimes).dt.total_seconds().to_numpy(dtype=float)
    tau = seconds / (YEAR_DAYS * 24.0 * 3600.0)
    tau = np.maximum(tau, MIN_TAU_YEARS)
    return tau


def parse_option_metadata(df: pd.DataFrame) -> pd.DataFrame:
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


def contract_label(meta_row: pd.Series) -> str:
    expiry = pd.Timestamp(meta_row["expiry"]).strftime("%d-%b-%Y")
    return f"{meta_row['strike']} {meta_row['option_type']} ({meta_row['column']}) | Expiry: {expiry}"


def compute_rolling_n_and_k(df: pd.DataFrame, meta_row: pd.Series) -> pd.DataFrame:
    """
    Compute raw_n, rolling_n, and k = IV^rolling_n * tau for one contract.

    The derivative is computed only across observed IV values.

    For two consecutive observed IV points i-1 and i:
        iv_prime_i = (IV_i - IV_{i-1}) / (t_i - t_{i-1} in years)

    Then:
        raw_n_i = IV_i / (iv_prime_i * tau_i)

    rolling_n_i:
        rolling median of raw_n over observed points.

    k_i:
        IV_i ** rolling_n_i * tau_i

    Notes:
        - The first observed point has no derivative, so raw_n is NaN.
        - If IV' is almost zero, n explodes, so raw_n is set to NaN.
        - k can become extreme when rolling_n is extreme.
    """
    col = meta_row["column"]

    out = pd.DataFrame({
        "datetime": df["datetime"],
        "timestamp_index": np.arange(len(df), dtype=float),
        "iv": df[col].astype(float),
    })

    out["tau"] = compute_tau_years(df["datetime"], meta_row["expiry_datetime"])
    out["is_missing_iv"] = out["iv"].isna()

    observed = out.dropna(subset=["iv"]).copy()

    if observed.empty:
        out["iv_prime_per_year"] = np.nan
        out["raw_n"] = np.nan
        out["rolling_n"] = np.nan
        out["k_iv_pow_n_tau"] = np.nan
        return out

    # Time differences between consecutive observed points, in years.
    dt_years = observed["datetime"].diff().dt.total_seconds() / (YEAR_DAYS * 24.0 * 3600.0)
    d_iv = observed["iv"].diff()

    observed["iv_prime_per_year"] = d_iv / dt_years

    denom = observed["iv_prime_per_year"] * observed["tau"]
    observed["raw_n"] = np.where(
        np.isfinite(denom) & (np.abs(denom) > DENOM_EPS),
        observed["iv"] / denom,
        np.nan,
    )

    # Constraint: n != 0.
    # If n is exactly zero or numerically too close to zero, discard it.
    observed.loc[
        ~np.isfinite(observed["raw_n"]) | (np.abs(observed["raw_n"]) <= N_EPS),
        "raw_n",
    ] = np.nan

    # Rolling median is more robust than rolling mean because raw_n can explode
    # whenever IV' is close to zero.
    observed["rolling_n"] = (
        observed["raw_n"]
        .replace([np.inf, -np.inf], np.nan)
        .rolling(window=ROLLING_WINDOW_OBS, min_periods=max(3, ROLLING_WINDOW_OBS // 3))
        .median()
    )

    # Compute k = IV^n * tau.
    # This is only meaningful when IV > 0 and rolling_n is finite.
    good_k = (
        np.isfinite(observed["iv"])
        & (observed["iv"] > 0)
        & np.isfinite(observed["rolling_n"])
        & np.isfinite(observed["tau"])
    )

    observed["k_iv_pow_n_tau"] = np.nan
    observed.loc[good_k, "k_iv_pow_n_tau"] = (
        np.power(observed.loc[good_k, "iv"], observed.loc[good_k, "rolling_n"])
        * observed.loc[good_k, "tau"]
    )

    # Merge computed columns back to full timestamp table.
    for col_name in ["iv_prime_per_year", "raw_n", "rolling_n", "k_iv_pow_n_tau"]:
        out[col_name] = np.nan
        out.loc[observed.index, col_name] = observed[col_name]

    return out


def robust_ylim(values, pad_frac=0.10, symmetric=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -1.0, 1.0

    lo = np.nanquantile(values, 0.02)
    hi = np.nanquantile(values, 0.98)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(values)
        hi = np.nanmax(values)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(values[0]) if len(values) else 0.0
        return center - 1.0, center + 1.0

    if symmetric:
        m = max(abs(lo), abs(hi))
        return -1.1 * m, 1.1 * m

    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def make_dashboard(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    x_index = np.arange(len(df), dtype=float)
    datetime_labels = df["datetime"].dt.strftime("%d-%m %H:%M").tolist()

    option_type_state = {"type": "ALL"}
    filtered_indices_state = {"indices": list(meta.index)}
    selected_pos_state = {"pos": 0}

    fig, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    plt.subplots_adjust(bottom=0.23, left=0.10, right=0.82, hspace=0.28)

    ax_n.set_ylabel("rolling n")
    ax_k.set_ylabel("IV^n × tau")
    ax_k.set_xlabel("Equally Spaced Timestamp Index")

    for ax in (ax_n, ax_k):
        ax.grid(True, alpha=0.3)

    max_ticks = 10
    tick_positions = np.linspace(0, len(df) - 1, min(max_ticks, len(df))).astype(int)
    ax_k.set_xticks(tick_positions)
    ax_k.set_xticklabels([datetime_labels[i] for i in tick_positions], rotation=30, ha="right")

    n_raw_scatter = ax_n.scatter([], [], s=18, alpha=0.25, label="raw n")
    n_roll_line, = ax_n.plot([], [], linewidth=1.6, alpha=0.90, label=f"rolling median n ({ROLLING_WINDOW_OBS} obs)")
    n_zero_line = ax_n.axhline(0, linewidth=1, linestyle="--", alpha=0.5)

    k_scatter = ax_k.scatter([], [], s=22, alpha=0.75, label="IV^rolling_n × tau")
    missing_scatter = ax_k.scatter([], [], marker="x", s=55, alpha=0.75, label="missing IV timestamp")

    ax_n.legend(loc="best")
    ax_k.legend(loc="best")

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

    radio_ax = fig.add_axes([0.85, 0.42, 0.10, 0.18])
    radio = RadioButtons(radio_ax, ("ALL", "CE", "PE"), active=0)

    info_text = fig.text(
        0.42,
        0.055,
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
        meta_row, pos, n_contracts = get_current_meta_row()
        col = meta_row["column"]

        stats = compute_rolling_n_and_k(df, meta_row)

        observed_mask = ~stats["is_missing_iv"].to_numpy()
        missing_mask = stats["is_missing_iv"].to_numpy()

        raw_n = stats["raw_n"].to_numpy(dtype=float)
        rolling_n = stats["rolling_n"].to_numpy(dtype=float)
        k_val = stats["k_iv_pow_n_tau"].to_numpy(dtype=float)

        # Clip only for plotting raw n, because raw_n can explode when IV' is tiny.
        raw_n_plot = np.clip(raw_n, -PLOT_N_CLIP, PLOT_N_CLIP)
        rolling_n_plot = np.clip(rolling_n, -PLOT_N_CLIP, PLOT_N_CLIP)

        raw_good = observed_mask & np.isfinite(raw_n_plot)
        roll_good = observed_mask & np.isfinite(rolling_n_plot)
        k_good = observed_mask & np.isfinite(k_val)

        if raw_good.any():
            n_raw_scatter.set_offsets(np.column_stack([x_index[raw_good], raw_n_plot[raw_good]]))
        else:
            n_raw_scatter.set_offsets(np.empty((0, 2)))

        if roll_good.any():
            n_roll_line.set_data(x_index[roll_good], rolling_n_plot[roll_good])
        else:
            n_roll_line.set_data([], [])

        if k_good.any():
            k_scatter.set_offsets(np.column_stack([x_index[k_good], k_val[k_good]]))
        else:
            k_scatter.set_offsets(np.empty((0, 2)))

        # Missing IV markers on k subplot.
        k_y_low, k_y_high = robust_ylim(k_val[k_good], pad_frac=0.10)
        missing_y = k_y_low

        miss_x = x_index[missing_mask]
        if len(miss_x):
            missing_scatter.set_offsets(np.column_stack([miss_x, np.full(len(miss_x), missing_y)]))
        else:
            missing_scatter.set_offsets(np.empty((0, 2)))

        n_y_low, n_y_high = robust_ylim(
            np.concatenate([raw_n_plot[np.isfinite(raw_n_plot)], rolling_n_plot[np.isfinite(rolling_n_plot)]]),
            pad_frac=0.10,
            symmetric=True,
        )

        ax_n.set_ylim(n_y_low, n_y_high)
        ax_k.set_ylim(k_y_low, k_y_high)
        ax_k.set_xlim(-1, len(df))

        n_obs = int(observed_mask.sum())
        n_missing = int(missing_mask.sum())
        missing_pct = 100 * n_missing / len(stats)

        tau_first = stats["tau"].iloc[0]
        tau_last = stats["tau"].iloc[-1]

        fig.suptitle(
            f"Rolling n and k = IV^n × tau | {contract_label(meta_row)} | "
            f"Observed: {n_obs} | Missing: {n_missing} ({missing_pct:.1f}%)",
            fontsize=12,
        )

        info_text.set_text(
            f"Filter: {option_type_state['type']} | "
            f"Showing {pos + 1}/{n_contracts} | "
            f"Strike: {meta_row['strike']} | Type: {meta_row['option_type']} | "
            f"tau range: {tau_first:.6f} to {tau_last:.6f} yrs | "
            f"raw n clipped to ±{PLOT_N_CLIP:g} for display"
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
    print("")
    print("Formula:")
    print("  Want IV^n * tau = k")
    print("  n = IV / (IV_prime * tau)")
    print("  IV_prime is computed as ΔIV / Δt_years across observed points.")
    print("")
    print("Notes:")
    print("  raw n is clipped only for display because it explodes when IV_prime ≈ 0.")
    print("  Constraint: n != 0. Values with |n| <= N_EPS are discarded.")
    print("  rolling_n uses a rolling median over observed points after applying n != 0.")
    print("  k uses rolling_n, not raw_n.")

    plt.show()


def main() -> None:
    df = load_dataset(DATA_PATH)
    meta = parse_option_metadata(df)

    print("Loaded dataset:", DATA_PATH.resolve())
    print("Rows:", len(df))
    print("Parsed option contracts:", len(meta))
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
    print("Rolling observed-point window:", ROLLING_WINDOW_OBS)
    print("Available contracts:")
    for i, row in meta.iterrows():
        print(f"  {i}: {contract_label(row)}")

    make_dashboard(df, meta)


if __name__ == "__main__":
    main()
