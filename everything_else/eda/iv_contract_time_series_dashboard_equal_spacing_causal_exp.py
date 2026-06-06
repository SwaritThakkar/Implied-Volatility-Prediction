"""
Matplotlib Contract Time-Series Dashboard:
IV vs Equally Spaced Timestamp Index + Causal Two-Regime Exponential Fits

Run:

    python iv_contract_time_series_dashboard_equal_spacing_causal_exp.py

This dashboard shows one selected option contract at a time.

It plots:
    1. Observed IV points through time.
    2. Missing IV timestamps on a bottom reference band.
    3. A causal exponential fitted curve for the pre-expiry regime.
    4. A causal exponential fitted curve for the expiry-day regime.

Very important causal rule:
    For every fitted value drawn at timestamp index t, the exponential model is fitted
    ONLY using observed IV points from strictly earlier timestamp indices.

    That means:
        allowed training points:     s < t
        forbidden training points:   s >= t

    The plotted fitted lines are therefore not ordinary in-sample smoothers.
    They are walk-forward / causal fitted values.

Regimes:
    Pre-expiry regime:
        all timestamps before 27 Jan 2026

    Expiry-day regime:
        timestamps on 27 Jan 2026

Model form inside each regime:
    IV_hat(x) = anchor_iv + a * (exp(b * x) - 1)

where:
    x = equally spaced index distance from that regime's first timestamp.

The model is anchored at the first observed IV in that regime. If the anchor
does not exist yet for a given causal prediction point, no fitted value is drawn.
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


# Candidate exponential speeds.
# Positive b means upward exponential curvature.
# Negative b means downward exponential curvature.
# b close to 0 behaves approximately linear.
EXP_B_GRID_PRE_EXPIRY = np.linspace(-0.020, 0.020, 81)
EXP_B_GRID_EXPIRY_DAY = np.linspace(-0.150, 0.150, 121)

# Need at least this many previous observed points inside the regime before drawing a fit.
MIN_PREVIOUS_POINTS_FOR_EXP_FIT = 3

# Date that starts the fast expiry-day regime.
EXPIRY_DAY = pd.Timestamp("2026-01-27").date()


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


def exp_basis(x: np.ndarray, b: float) -> np.ndarray:
    """
    Basis g_b(x) = exp(b*x) - 1.

    For b close to zero, exp(b*x)-1 is almost b*x and can be numerically tiny.
    In that case, use x itself as the linear-limit basis.

    This keeps the model stable and lets b≈0 represent an almost-linear curve.
    """
    x = np.asarray(x, dtype=float)

    if abs(b) < 1e-10:
        return x.copy()

    return np.exp(b * x) - 1.0


def fit_anchor_exponential_grid(x_train: np.ndarray, y_train: np.ndarray, b_grid: np.ndarray):
    """
    Fit:
        y = y_anchor + a * (exp(b*x) - 1)

    using a grid search over b. For a fixed b, a is estimated by least squares.

    The anchor is the first available training point inside the regime.
    The x values should already be measured relative to the regime start.

    Returns:
        anchor_x, anchor_y, best_a, best_b

    If fit is not possible, returns None.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mask = np.isfinite(x_train) & np.isfinite(y_train)
    x_train = x_train[mask]
    y_train = y_train[mask]

    if len(y_train) < MIN_PREVIOUS_POINTS_FOR_EXP_FIT:
        return None

    order = np.argsort(x_train)
    x_train = x_train[order]
    y_train = y_train[order]

    anchor_x = float(x_train[0])
    anchor_y = float(y_train[0])

    # Shift x so the anchor is exactly the model origin.
    x_rel = x_train - anchor_x
    y_rel = y_train - anchor_y

    best_mse = np.inf
    best_a = 0.0
    best_b = 0.0

    for b in b_grid:
        g = exp_basis(x_rel, b)

        denom = float(np.dot(g, g))
        if denom <= 1e-15 or not np.isfinite(denom):
            continue

        a = float(np.dot(g, y_rel) / denom)
        pred = anchor_y + a * g
        mse = float(np.mean((pred - y_train) ** 2))

        if np.isfinite(mse) and mse < best_mse:
            best_mse = mse
            best_a = a
            best_b = float(b)

    if not np.isfinite(best_mse):
        return None

    return anchor_x, anchor_y, best_a, best_b


def predict_anchor_exponential(x_target: float, fit_params):
    """
    Predict from fitted anchor exponential parameters.
    """
    if fit_params is None:
        return np.nan

    anchor_x, anchor_y, a, b = fit_params
    x_rel = float(x_target) - float(anchor_x)
    g = exp_basis(np.array([x_rel], dtype=float), b)[0]
    pred = anchor_y + a * g

    if not np.isfinite(pred):
        return np.nan

    return float(pred)


def causal_regime_exponential_fit(
    y_series: pd.Series,
    regime_mask: np.ndarray,
    regime_start_index: int,
    b_grid: np.ndarray,
) -> np.ndarray:
    """
    Build a causal exponential fitted series for one regime.

    IMPORTANT:
    For each target index t, this function uses ONLY observed data points
    with index s < t and inside the same regime.

    This is the key no-lookahead guarantee.

    More explicitly:
        train_mask = regime_mask & observed_mask & (all_indices < t)

    The current point t is NOT used, even if y[t] is observed.
    Future points after t are NOT used.
    """
    n = len(y_series)
    y = y_series.to_numpy(dtype=float)
    all_indices = np.arange(n)

    fitted = np.full(n, np.nan, dtype=float)

    regime_indices = all_indices[regime_mask]

    if len(regime_indices) == 0:
        return fitted

    for t in regime_indices:
        # ------------------------------------------------------------
        # CAUSAL TRAINING SET:
        # only same-regime, observed points with strictly earlier index.
        #
        # DO NOT CHANGE THIS TO <= t.
        # DO NOT USE points after t.
        # This is the line that prevents look-ahead leakage in the plot.
        # ------------------------------------------------------------
        train_mask = regime_mask & np.isfinite(y) & (all_indices < t)

        if train_mask.sum() < MIN_PREVIOUS_POINTS_FOR_EXP_FIT:
            continue

        x_train = all_indices[train_mask] - regime_start_index
        y_train = y[train_mask]

        fit_params = fit_anchor_exponential_grid(x_train, y_train, b_grid)

        if fit_params is None:
            continue

        x_target = t - regime_start_index
        fitted[t] = predict_anchor_exponential(x_target, fit_params)

    return fitted


def make_contract_time_dashboard(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Create interactive IV vs equally spaced timestamp index dashboard.
    """
    x_index = np.arange(len(df), dtype=float)
    datetime_labels = df["datetime"].dt.strftime("%d-%m %H:%M").tolist()

    dates = df["datetime"].dt.date
    pre_expiry_mask = (dates < EXPIRY_DAY).to_numpy()
    expiry_day_mask = (dates >= EXPIRY_DAY).to_numpy()

    if pre_expiry_mask.any():
        pre_start = int(np.where(pre_expiry_mask)[0][0])
    else:
        pre_start = 0

    if expiry_day_mask.any():
        expiry_start = int(np.where(expiry_day_mask)[0][0])
    else:
        expiry_start = len(df) - 1

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

    # Regime separator.
    if expiry_day_mask.any():
        ax.axvline(
            expiry_start,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Expiry-day regime start",
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
        alpha=0.50,
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

    pre_exp_fit_line, = ax.plot(
        [],
        [],
        linewidth=2,
        alpha=0.90,
        label="Causal exp fit: pre-expiry",
    )

    expiry_fit_line, = ax.plot(
        [],
        [],
        linewidth=2,
        alpha=0.90,
        label="Causal exp fit: expiry day",
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

    def set_line_data_from_series(line, fitted_values, mask):
        valid = np.isfinite(fitted_values) & mask

        if valid.any():
            line.set_data(x_index[valid], fitted_values[valid])
        else:
            line.set_data([], [])

    def update_plot(_=None):
        row, pos, n_contracts = get_current_meta_row()
        col = row["column"]

        series = df[col]
        observed_mask = series.notna().to_numpy()
        missing_mask = series.isna().to_numpy()

        obs_x = x_index[observed_mask]
        obs_y = series[observed_mask].to_numpy(dtype=float)

        miss_x = x_index[missing_mask]
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

        # ------------------------------------------------------------
        # CAUSAL EXPONENTIAL FITS:
        # These arrays are computed fresh for the selected contract.
        # Each fitted point t is based only on observed points s < t
        # inside the same regime. See causal_regime_exponential_fit().
        # ------------------------------------------------------------
        pre_fit = causal_regime_exponential_fit(
            y_series=series,
            regime_mask=pre_expiry_mask,
            regime_start_index=pre_start,
            b_grid=EXP_B_GRID_PRE_EXPIRY,
        )

        expiry_fit = causal_regime_exponential_fit(
            y_series=series,
            regime_mask=expiry_day_mask,
            regime_start_index=expiry_start,
            b_grid=EXP_B_GRID_EXPIRY_DAY,
        )

        set_line_data_from_series(pre_exp_fit_line, pre_fit, pre_expiry_mask)
        set_line_data_from_series(expiry_fit_line, expiry_fit, expiry_day_mask)

        n_obs = int(observed_mask.sum())
        n_missing = int(missing_mask.sum())
        missing_pct = 100 * n_missing / len(series)

        first_ts = df["datetime"].iloc[0].strftime("%d-%m-%Y %H:%M")
        last_ts = df["datetime"].iloc[-1].strftime("%d-%m-%Y %H:%M")

        n_pre_fit = int(np.isfinite(pre_fit).sum())
        n_exp_fit = int(np.isfinite(expiry_fit).sum())

        title.set_text(
            f"IV vs Time Index: {contract_label(row)} | "
            f"Observed: {n_obs} | Missing: {n_missing} ({missing_pct:.1f}%)"
        )

        info_text.set_text(
            f"Filter: {option_type_state['type']} | "
            f"Showing {pos + 1}/{n_contracts} | "
            f"Strike: {row['strike']} | Type: {row['option_type']} | "
            f"Causal fit pts: pre={n_pre_fit}, expiry={n_exp_fit} | "
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
    print("")
    print("Causal exponential fit rule:")
    print("  For every fitted point t, the fit uses only observed points s < t.")
    print("  Current and future points are not used.")

    plt.show()


def main() -> None:
    df = load_dataset(DATA_PATH)
    meta = parse_option_metadata(df)

    print("Loaded dataset:", DATA_PATH.resolve())
    print("Rows:", len(df))
    print("Parsed option contracts:", len(meta))
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
    print("Expiry-day regime starts:", EXPIRY_DAY)
    print("Available contracts:")
    for i, row in meta.iterrows():
        print(f"  {i}: {contract_label(row)}")

    make_contract_time_dashboard(df, meta)


if __name__ == "__main__":
    main()
