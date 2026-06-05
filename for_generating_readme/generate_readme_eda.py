import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.interpolate import griddata, PchipInterpolator
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

FILLED_PATH = ROOT / "filled_dataset_try_final_pchip_interior.csv"
DIAG_PATH = ROOT / "diagnostics_try_final_pchip_interior.csv"
DATA_PATHS = [
    ROOT / "everything_else" / "cv_validation_system" / "dataset.csv",
    ROOT / "dataset.csv",
]
NOT_DATASET_PATHS = [
    ROOT / "everything_else" / "cv_validation_system" / "cv_split" / "not_dataset.csv",
    ROOT / "cv_validation_system" / "cv_split" / "not_dataset.csv",
]
CV_METRICS_PATH = OUT / "cv_eval_final_submission" / "metrics_summary.csv"
STRATEGY_METRICS = {
    "final": OUT / "cv_eval_final_submission" / "metrics_summary.csv",
    "linear edge": ROOT / "everything_else" / "strategies_and_results" / "linear_edge" / "cv_eval_results" / "metrics_summary.csv",
    "quadratic progressive": ROOT / "everything_else" / "strategies_and_results" / "quad_edge_case_handled_m2" / "cv_eval_results" / "metrics_summary.csv",
    "quadratic edge fixed": ROOT / "everything_else" / "strategies_and_results" / "quad_edge_case_handled_m1" / "cv_eval_results" / "metrics_summary.csv",
    "raw quadratic": ROOT / "everything_else" / "strategies_and_results" / "quadratic_fit_raw" / "cv_eval_results" / "metrics_summary.csv",
    "jan27 temporal": ROOT / "everything_else" / "strategies_and_results" / "focused_only_on_27th_jan_a_norm" / "cv_eval_results" / "metrics_summary.csv",
}

for path in (ROOT / "everything_else" / "strategies_and_results").glob("m4_tried_by*/cv_eval_results/metrics_summary.csv"):
    STRATEGY_METRICS["local-poly edge"] = path
    break


def option_columns(df):
    return [c for c in df.columns if re.search(r"\d+(CE|PE)$", c)]


def parse_contract(col):
    m = re.match(r"^[A-Z]+\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$", col)
    if not m:
        m = re.search(r"(\d+)(CE|PE)$", col)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def load_first(paths):
    for path in paths:
        if path.exists():
            return pd.read_csv(path), path
    return None, None


def prepare_long(df, opt_type):
    rows = []
    dt = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    expiry = pd.Timestamp("2026-01-27 15:30")
    days_to_expiry = (expiry - dt) / pd.Timedelta(days=1)
    for col in option_columns(df):
        parsed = parse_contract(col)
        if parsed is None:
            continue
        strike, typ = parsed
        if typ != opt_type:
            continue
        x = strike / df["underlying_price"].to_numpy(float)
        rows.append(pd.DataFrame({
            "datetime": dt,
            "time_index": np.arange(len(df)),
            "days_to_expiry": days_to_expiry,
            "strike": strike,
            "moneyness": x,
            "iv": df[col].to_numpy(float),
            "contract": col,
        }))
    return pd.concat(rows, ignore_index=True)


def surface_grid(df, opt_type):
    long = prepare_long(df, opt_type)
    long = long[np.isfinite(long["iv"]) & np.isfinite(long["moneyness"])].copy()
    long = long[long["days_to_expiry"] >= 0].copy()
    lo, hi = long["iv"].quantile([0.005, 0.985])
    long["iv_plot"] = long["iv"].clip(lo, hi)

    d_min, d_max = long["days_to_expiry"].quantile([0.002, 0.998])
    s_min, s_max = long["strike"].min(), long["strike"].max()
    grid_d = np.linspace(float(d_max), float(d_min), 170)
    grid_s = np.linspace(float(s_min), float(s_max), 120)
    D, S = np.meshgrid(grid_d, grid_s)
    pts = long[["days_to_expiry", "strike"]].to_numpy(float)
    vals = long["iv_plot"].to_numpy(float)
    Z = griddata(pts, vals, (D, S), method="linear")
    nearest = griddata(pts, vals, (D, S), method="nearest")
    Z = np.where(np.isfinite(Z), Z, nearest)
    Z = gaussian_filter(Z, sigma=(1.2, 1.2))
    Z = np.clip(Z, lo, hi)
    return long, D, S, Z, (lo, hi)


def observed_surface_points(original, opt_type, lo_hi, step=3):
    if original is None:
        return pd.DataFrame()
    obs = prepare_long(original, opt_type)
    obs = obs[np.isfinite(obs["iv"]) & np.isfinite(obs["days_to_expiry"])].copy()
    obs = obs[obs["days_to_expiry"] >= 0].copy()
    if obs.empty:
        return obs
    lo, hi = lo_hi
    obs["iv_plot"] = obs["iv"].clip(lo, hi)
    return obs.iloc[::step].copy()


def style_3d_axis(ax):
    ax.set_facecolor("#050812")
    ax.tick_params(colors="#d8e1ff", labelsize=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.04, 0.06, 0.12, 0.95))
        axis.pane.set_edgecolor((0.52, 0.64, 0.9, 0.7))
    ax.grid(True, color="#52617f", alpha=0.35)


def save_surface(df, opt_type, filename, title, original=None):
    long, D, S, Z, lo_hi = surface_grid(df, opt_type)
    observed = observed_surface_points(original, opt_type, lo_hi, step=6)

    sample = long.iloc[::4].copy()
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=D,
        y=S,
        z=Z,
        colorscale="Turbo",
        opacity=0.96,
        colorbar={"title": "IV"},
        contours={
            "z": {"show": True, "usecolormap": True, "highlightcolor": "#ffffff", "project_z": True},
            "x": {"show": False},
            "y": {"show": False},
        },
        lighting={"ambient": 0.58, "diffuse": 0.82, "specular": 0.42, "roughness": 0.35},
        lightposition={"x": -140, "y": -80, "z": 220},
        name="interpolated surface",
    ))
    fig.add_trace(go.Scatter3d(
        x=sample["days_to_expiry"],
        y=sample["strike"],
        z=sample["iv_plot"],
        mode="markers",
        marker={"size": 2.2, "color": sample["iv_plot"], "colorscale": "Turbo", "opacity": 0.55},
        name="filled contracts",
        hovertemplate="t=%{x}<br>K/S=%{y:.4f}<br>IV=%{z:.4f}<extra></extra>",
    ))
    if not observed.empty:
        fig.add_trace(go.Scatter3d(
            x=observed["days_to_expiry"],
            y=observed["strike"],
            z=observed["iv_plot"],
            mode="markers",
            marker={
                "size": 2.9,
                "color": "#f8c14a",
                "opacity": 0.68,
                "symbol": "circle",
                "line": {"color": "#ffffff", "width": 0.8},
            },
            name="given IV observations",
            hovertemplate="days=%{x:.2f}<br>strike=%{y}<br>IV=%{z:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 30, "color": "white"}},
        template="plotly_dark",
        width=1700,
        height=1100,
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        paper_bgcolor="#050812",
        scene={
            "xaxis": {"title": "days to expiry", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "yaxis": {"title": "strike", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "zaxis": {"title": "IV, clipped to 1st-99th pct", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "camera": {"eye": {"x": 1.6, "y": -1.55, "z": 0.9}, "center": {"x": 0.03, "y": 0.0, "z": -0.08}},
            "aspectratio": {"x": 1.9, "y": 1.0, "z": 0.78},
        },
        legend={"font": {"color": "#d8e1ff"}},
    )
    html_name = filename.replace(".png", ".html")
    fig.write_html(OUT / html_name, include_plotlyjs="cdn", full_html=True)

    static = plt.figure(figsize=(13, 9.2), dpi=220)
    static.patch.set_facecolor("#050812")
    ax3d = static.add_subplot(111, projection="3d")
    style_3d_axis(ax3d)
    surf = ax3d.plot_surface(
        D, S, Z,
        cmap="turbo",
        linewidth=0.18,
        edgecolor=(0.02, 0.04, 0.08, 0.32),
        antialiased=True,
        alpha=0.98,
        rstride=1,
        cstride=1,
        shade=True,
    )
    z_floor = float(np.nanmin(Z)) - 0.04 * (float(np.nanmax(Z)) - float(np.nanmin(Z)))
    ax3d.contour(D, S, Z, zdir="z", offset=z_floor, levels=18, cmap="turbo", alpha=0.66, linewidths=0.8)
    ax3d.contour(D, S, Z, zdir="x", offset=float(np.nanmax(D)), levels=12, cmap="turbo", alpha=0.32, linewidths=0.55)
    if not observed.empty:
        ax3d.scatter(
            observed["days_to_expiry"],
            observed["strike"],
            observed["iv_plot"],
            s=5,
            c="#f8c14a",
            edgecolors="white",
            linewidths=0.25,
            alpha=0.56,
            depthshade=False,
        )
        ax3d.text2D(0.06, 0.88, "gold dots = given IV observations", transform=ax3d.transAxes, color="#f8c14a", fontsize=11, weight="bold")
    ax3d.view_init(elev=29, azim=-132)
    ax3d.set_box_aspect((1.9, 1.0, 0.78))
    ax3d.set_title(title, color="white", fontsize=19, weight="bold", pad=14)
    ax3d.set_xlabel("days to expiry", color="#d8e1ff", labelpad=10)
    ax3d.set_ylabel("strike", color="#d8e1ff", labelpad=10)
    ax3d.set_zlabel("IV", color="#d8e1ff", labelpad=10)
    ax3d.set_zlim(z_floor, float(np.nanmax(Z)))
    ax3d.invert_xaxis()
    cax = static.add_axes([0.88, 0.22, 0.018, 0.55])
    cbar = static.colorbar(surf, cax=cax)
    cbar.ax.tick_params(colors="#d8e1ff")
    cbar.set_label("IV, clipped to 1st-99th pct", color="#d8e1ff")
    static.subplots_adjust(left=0.0, right=0.86, top=0.94, bottom=0.02)
    static.savefig(OUT / filename, facecolor=static.get_facecolor(), bbox_inches="tight")
    plt.close(static)


def save_combined_surface(df, original=None):
    long_ce, D_ce, S_ce, Z_ce, lo_hi_ce = surface_grid(df, "CE")
    long_pe, D_pe, S_pe, Z_pe, lo_hi_pe = surface_grid(df, "PE")
    observed_ce = observed_surface_points(original, "CE", lo_hi_ce, step=8)
    observed_pe = observed_surface_points(original, "PE", lo_hi_pe, step=8)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=D_ce, y=S_ce, z=Z_ce, colorscale="Blues", opacity=0.92, name="CE surface", showscale=False))
    fig.add_trace(go.Surface(x=D_pe, y=S_pe, z=Z_pe, colorscale="Reds", opacity=0.82, name="PE surface", showscale=True, colorbar={"title": "IV"}))
    if not observed_ce.empty:
        fig.add_trace(go.Scatter3d(
            x=observed_ce["days_to_expiry"],
            y=observed_ce["strike"],
            z=observed_ce["iv_plot"],
            mode="markers",
            marker={"size": 2.8, "color": "#00d4ff", "opacity": 0.68, "line": {"color": "#ffffff", "width": 0.5}},
            name="given CE observations",
            hovertemplate="CE<br>days=%{x:.2f}<br>strike=%{y}<br>IV=%{z:.4f}<extra></extra>",
        ))
    if not observed_pe.empty:
        fig.add_trace(go.Scatter3d(
            x=observed_pe["days_to_expiry"],
            y=observed_pe["strike"],
            z=observed_pe["iv_plot"],
            mode="markers",
            marker={"size": 2.8, "color": "#ff4d6d", "opacity": 0.68, "line": {"color": "#ffffff", "width": 0.5}},
            name="given PE observations",
            hovertemplate="PE<br>days=%{x:.2f}<br>strike=%{y}<br>IV=%{z:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "combined CE and PE implied-volatility surfaces", "x": 0.5, "font": {"size": 30, "color": "white"}},
        template="plotly_dark",
        width=1700,
        height=1100,
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        paper_bgcolor="#050812",
        scene={
            "xaxis": {"title": "days to expiry", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "yaxis": {"title": "strike", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "zaxis": {"title": "IV", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "camera": {"eye": {"x": 1.65, "y": -1.55, "z": 0.9}},
            "aspectratio": {"x": 1.9, "y": 1.0, "z": 0.78},
        },
    )
    fig.write_html(OUT / "iv_surface_combined_3d.html", include_plotlyjs="cdn", full_html=True)

    fig2 = plt.figure(figsize=(13.2, 9.4), dpi=220)
    fig2.patch.set_facecolor("#050812")
    ax = fig2.add_subplot(111, projection="3d")
    style_3d_axis(ax)
    z_min = min(float(np.nanmin(Z_ce)), float(np.nanmin(Z_pe)))
    z_max = max(float(np.nanmax(Z_ce)), float(np.nanmax(Z_pe)))
    z_floor = z_min - 0.04 * (z_max - z_min)
    ax.plot_surface(D_ce, S_ce, Z_ce, cmap="winter", linewidth=0.14, edgecolor=(0.02, 0.04, 0.08, 0.26), alpha=0.92, antialiased=True)
    ax.plot_surface(D_pe, S_pe, Z_pe, cmap="autumn", linewidth=0.14, edgecolor=(0.02, 0.04, 0.08, 0.26), alpha=0.82, antialiased=True)
    ax.contour(D_ce, S_ce, Z_ce, zdir="z", offset=z_floor, levels=16, cmap="winter", alpha=0.52, linewidths=0.7)
    ax.contour(D_pe, S_pe, Z_pe, zdir="z", offset=z_floor, levels=16, cmap="autumn", alpha=0.45, linewidths=0.7)
    if not observed_ce.empty:
        ax.scatter(observed_ce["days_to_expiry"], observed_ce["strike"], observed_ce["iv_plot"], s=5, c="#00d4ff", edgecolors="white", linewidths=0.18, alpha=0.58, depthshade=False)
    if not observed_pe.empty:
        ax.scatter(observed_pe["days_to_expiry"], observed_pe["strike"], observed_pe["iv_plot"], s=5, c="#ff4d6d", edgecolors="white", linewidths=0.18, alpha=0.58, depthshade=False)
    ax.view_init(elev=29, azim=-132)
    ax.set_box_aspect((1.9, 1.0, 0.78))
    ax.set_title("combined CE and PE implied-volatility surfaces", color="white", fontsize=19, weight="bold", pad=14)
    ax.set_xlabel("days to expiry", color="#d8e1ff", labelpad=10)
    ax.set_ylabel("strike", color="#d8e1ff", labelpad=10)
    ax.set_zlabel("IV", color="#d8e1ff", labelpad=10)
    ax.set_zlim(z_floor, z_max)
    ax.invert_xaxis()
    ax.text2D(0.06, 0.90, "CE surface", transform=ax.transAxes, color="#38bdf8", fontsize=13, weight="bold")
    ax.text2D(0.06, 0.855, "PE surface", transform=ax.transAxes, color="#fb7185", fontsize=13, weight="bold")
    ax.text2D(0.06, 0.81, "small bright dots = given IV observations", transform=ax.transAxes, color="#f8c14a", fontsize=11, weight="bold")
    fig2.subplots_adjust(left=0.0, right=0.98, top=0.94, bottom=0.02)
    fig2.savefig(OUT / "iv_surface_combined_3d.png", facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close(fig2)


def save_missingness(original, not_dataset):
    opts = option_columns(original)
    miss = original[opts].isna().astype(int).T
    miss.index = [c[-2:] + " " + re.search(r"(\d+)(CE|PE)$", c).group(1) for c in opts]

    fig, axes = plt.subplots(1, 2 if not_dataset is not None else 1, figsize=(15, 7), dpi=180)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.patch.set_facecolor("#0a0d14")

    sns.heatmap(miss, cmap=["#101722", "#ff4d6d"], cbar=False, ax=axes[0])
    axes[0].set_title("original dataset missing cells", color="white", fontsize=14, weight="bold")
    axes[0].set_xlabel("timestamp index", color="#cdd6f4")
    axes[0].set_ylabel("contract", color="#cdd6f4")
    axes[0].tick_params(colors="#cdd6f4", labelsize=6)
    axes[0].set_facecolor("#0a0d14")

    if not_dataset is not None:
        common = [c for c in opts if c in not_dataset.columns]
        synthetic = not_dataset[common].isna().astype(int).T
        synthetic.index = [c[-2:] + " " + re.search(r"(\d+)(CE|PE)$", c).group(1) for c in common]
        sns.heatmap(synthetic, cmap=["#101722", "#00d4ff"], cbar=False, ax=axes[1])
        axes[1].set_title("synthetic validation mask", color="white", fontsize=14, weight="bold")
        axes[1].set_xlabel("timestamp index", color="#cdd6f4")
        axes[1].set_ylabel("")
        axes[1].tick_params(colors="#cdd6f4", labelsize=6)
        axes[1].set_facecolor("#0a0d14")

    fig.suptitle("where the model had to infer IV", color="white", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "missingness_original_and_validation.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_dataset_regime_eda(original):
    if original is None:
        return
    opts = option_columns(original)
    if not opts:
        return

    df = original.copy()
    dt = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    values = df[opts].astype(float)
    dates = dt.dt.date.astype(str)
    jan27 = dates == "2026-01-27"

    row_summary = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "datetime": dt,
        "date": dates,
        "mean_iv": values.mean(axis=1, skipna=True),
        "median_iv": values.median(axis=1, skipna=True),
        "iv_dispersion": values.std(axis=1, skipna=True),
        "missing_rate": values.isna().mean(axis=1),
        "underlying": df["underlying_price"].astype(float),
    })

    ce_cols = [c for c in opts if parse_contract(c)[1] == "CE"]
    pe_cols = [c for c in opts if parse_contract(c)[1] == "PE"]
    row_summary["ce_mean_iv"] = df[ce_cols].astype(float).mean(axis=1, skipna=True)
    row_summary["pe_mean_iv"] = df[pe_cols].astype(float).mean(axis=1, skipna=True)
    row_summary["ce_pe_gap"] = row_summary["ce_mean_iv"] - row_summary["pe_mean_iv"]

    lag_path = OUT.parent / "everything_else" / "eda" / "eda_lag1_calendar_gap" / "lag1_timestamp_metrics.csv"
    lag = None
    if lag_path.exists():
        lag = pd.read_csv(lag_path)
        lag = lag[lag["option_type"] == "ALL"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    for ax in axes.flat:
        style_axis(ax)
        ax.axvspan(row_summary.loc[jan27, "row_index"].min(), row_summary.loc[jan27, "row_index"].max(), color="#ff4d6d", alpha=0.12, lw=0)

    x = row_summary["row_index"]
    axes[0, 0].plot(x, row_summary["underlying"], color="#f8c14a", lw=1.7, label="underlying")
    ax_iv = axes[0, 0].twinx()
    ax_iv.set_facecolor("none")
    ax_iv.plot(x, row_summary["mean_iv"], color="#00d4ff", lw=2.1, label="mean observed IV")
    ax_iv.tick_params(colors="#d8e1ff")
    for spine in ax_iv.spines.values():
        spine.set_color("#34415f")
    axes[0, 0].set_title("underlying path and observed IV level", color="white", fontsize=14, weight="bold")
    axes[0, 0].set_xlabel("timestamp row index", color="#d8e1ff")
    axes[0, 0].set_ylabel("underlying", color="#f8c14a")
    ax_iv.set_ylabel("mean observed IV", color="#00d4ff")

    axes[0, 1].plot(x, row_summary["iv_dispersion"], color="#6ee7b7", lw=2.0, label="cross-strike IV dispersion")
    axes[0, 1].plot(x, row_summary["missing_rate"], color="#ff4d6d", lw=1.8, alpha=0.78, label="missing rate")
    axes[0, 1].set_title("smile width and missingness by timestamp", color="white", fontsize=14, weight="bold")
    axes[0, 1].set_xlabel("timestamp row index", color="#d8e1ff")
    axes[0, 1].set_ylabel("rate / dispersion", color="#d8e1ff")
    axes[0, 1].legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white", fontsize=8)

    daily = row_summary.groupby("date", as_index=False).agg(
        first_row=("row_index", "min"),
        last_row=("row_index", "max"),
        mean_iv=("mean_iv", "mean"),
        iv_dispersion=("iv_dispersion", "mean"),
        missing_rate=("missing_rate", "mean"),
        row_count=("row_index", "count"),
    )
    daily["mid_row"] = (daily["first_row"] + daily["last_row"]) / 2
    daily_colors = ["#ff4d6d" if d == "2026-01-27" else "#00d4ff" for d in daily["date"]]
    axes[1, 0].bar(daily["mid_row"], daily["mean_iv"], width=46, color=daily_colors, alpha=0.88)
    axes[1, 0].set_title("daily average observed IV: Jan 27 separates", color="white", fontsize=14, weight="bold")
    axes[1, 0].set_xlabel("trading day, placed by row index", color="#d8e1ff")
    axes[1, 0].set_ylabel("daily mean IV", color="#d8e1ff")
    axes[1, 0].set_xticks(daily["mid_row"])
    axes[1, 0].set_xticklabels([pd.Timestamp(d).strftime("%b %d") for d in daily["date"]], rotation=35, ha="right")
    axes[1, 0].set_xlim(-25, len(df) + 25)

    if lag is not None and not lag.empty:
        axes[1, 1].plot(lag["row_index"], lag["lag1_corr_cross_options"], color="#00d4ff", lw=1.7, label="lag-1 cross-option corr")
        ax_change = axes[1, 1].twinx()
        ax_change.set_facecolor("none")
        ax_change.plot(lag["row_index"], lag["mean_abs_iv_change"], color="#ff4d6d", lw=1.7, alpha=0.86, label="mean abs IV change")
        ax_change.tick_params(colors="#d8e1ff")
        for spine in ax_change.spines.values():
            spine.set_color("#34415f")
        axes[1, 1].set_ylabel("lag-1 correlation", color="#00d4ff")
        ax_change.set_ylabel("mean abs IV change", color="#ff4d6d")
    else:
        axes[1, 1].plot(x, row_summary["ce_pe_gap"], color="#a78bfa", lw=1.9)
        axes[1, 1].set_ylabel("CE mean IV - PE mean IV", color="#d8e1ff")
    axes[1, 1].set_title("time continuity weakens around the expiry regime", color="white", fontsize=14, weight="bold")
    axes[1, 1].set_xlabel("timestamp row index", color="#d8e1ff")

    jan_start = int(row_summary.loc[jan27, "row_index"].min())
    jan_end = int(row_summary.loc[jan27, "row_index"].max())
    for ax in axes.flat:
        ax.text(jan_start, ax.get_ylim()[1], "  Jan 27 expiry regime", color="#ff89a0", fontsize=8, weight="bold", va="top")

    fig.suptitle("EDA: the dataset is not one smooth time regime", color="white", fontsize=22, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "dataset_regime_eda.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_missing_given_cross_section_eda(original):
    if original is None:
        return
    opts = option_columns(original)
    if not opts:
        return

    parsed = {c: parse_contract(c) for c in opts}
    display_cols = sorted(opts, key=lambda c: (parsed[c][1], parsed[c][0]))
    given = original[display_cols].notna().astype(int).T
    labels = [f"{parsed[c][1]} {parsed[c][0]}" for c in display_cols]

    fig = plt.figure(figsize=(15.5, 9.2), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.36, wspace=0.22)
    ax_heat = fig.add_subplot(gs[0, :])
    ax_ce = fig.add_subplot(gs[1, 0])
    ax_pe = fig.add_subplot(gs[1, 1])
    for ax in [ax_heat, ax_ce, ax_pe]:
        ax.set_facecolor("#0f1724")
        for spine in ax.spines.values():
            spine.set_color("#34415f")
        ax.tick_params(colors="#d8e1ff")

    sns.heatmap(
        given,
        cmap=["#ff4d6d", "#101722"],
        cbar=False,
        ax=ax_heat,
    )
    ax_heat.set_title("given vs missing IV values across all timestamp cross-sections", color="white", fontsize=15, weight="bold")
    ax_heat.set_xlabel("timestamp row index", color="#d8e1ff")
    ax_heat.set_ylabel("contract ordered by type and strike", color="#d8e1ff")
    ax_heat.set_yticks(np.arange(len(labels)) + 0.5)
    ax_heat.set_yticklabels(labels, fontsize=6)
    ax_heat.text(0.985, 1.05, "dark = given   red = missing", transform=ax_heat.transAxes, ha="right", color="#ffb3c1", fontsize=10, weight="bold")

    for typ, ax, color in [("CE", ax_ce, "#00d4ff"), ("PE", ax_pe, "#ff4d6d")]:
        cols = [c for c in display_cols if parsed[c][1] == typ]
        strikes = np.array([parsed[c][0] for c in cols], dtype=float)
        given_counts = original[cols].notna().sum(axis=0).to_numpy(float)
        missing_counts = original[cols].isna().sum(axis=0).to_numpy(float)
        ax.bar(strikes, given_counts, width=68, color=color, alpha=0.78, label="given values")
        ax.bar(strikes, missing_counts, width=68, bottom=given_counts, color="#f8c14a", alpha=0.92, label="missing values")
        ax.set_title(f"{typ} cross-section coverage measured over all timestamps", color="white", fontsize=14, weight="bold")
        ax.set_xlabel("strike", color="#d8e1ff")
        ax.set_ylabel("timestamp count", color="#d8e1ff")
        ax.set_ylim(0, len(original) * 1.04)
        ax.grid(axis="y", color="#253149", alpha=0.55)
        ax.legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white", fontsize=8)

    fig.suptitle("EDA: where the dataset is given and where it is missing", color="white", fontsize=22, weight="bold")
    fig.savefig(OUT / "missing_given_cross_section_eda.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_decision_charts(diag):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), dpi=180)
    fig.patch.set_facecolor("#0a0d14")
    for ax in axes:
        ax.set_facecolor("#0f1724")
        ax.tick_params(colors="#d8e1ff")
        for spine in ax.spines.values():
            spine.set_color("#34415f")

    edge_counts = diag["edge"].map({True: "edge extrapolation", False: "interior interpolation"}).value_counts()
    axes[0].pie(
        edge_counts.values,
        labels=edge_counts.index,
        autopct="%1.1f%%",
        colors=["#6ee7b7", "#ff4d6d"],
        textprops={"color": "white", "fontsize": 9},
        startangle=120,
    )
    axes[0].set_title("routing decision", color="white", weight="bold")

    pchip_counts = diag["pchip_used"].map({True: "PCHIP blend", False: "edge / no PCHIP"}).value_counts()
    axes[1].bar(pchip_counts.index, pchip_counts.values, color=["#00d4ff", "#f8c14a"])
    axes[1].set_title("interior shape correction", color="white", weight="bold")
    axes[1].set_ylabel("filled cells", color="#d8e1ff")
    axes[1].tick_params(axis="x", rotation=15)

    edge = diag[diag["edge"] == True]
    degree_counts = edge["edge_degree"].dropna().astype(int).value_counts().sort_index()
    axes[2].bar([f"degree {i}" for i in degree_counts.index], degree_counts.values, color=["#a78bfa", "#fb7185"])
    axes[2].set_title("edge model selected by LOO", color="white", weight="bold")
    axes[2].set_ylabel("edge cells", color="#d8e1ff")

    fig.suptitle("final model decisions over 5,460 filled cells", color="white", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "final_model_decisions.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_smile_examples(filled, original):
    opts = option_columns(filled)
    parsed = {c: parse_contract(c) for c in opts}
    dates = pd.to_datetime(filled["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    choices = [0, 120, 300, 600, 850, 930, 960, len(filled) - 1]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.8), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    for ax, idx in zip(axes.flat, choices):
        ax.set_facecolor("#0f1724")
        spot = filled.loc[idx, "underlying_price"]
        for typ, color in [("CE", "#00d4ff"), ("PE", "#ff4d6d")]:
            cols = [c for c in opts if parsed[c][1] == typ]
            cols = sorted(cols, key=lambda c: parsed[c][0])
            x = np.array([parsed[c][0] / spot for c in cols], float)
            y = filled.loc[idx, cols].to_numpy(float)
            missing = original.loc[idx, cols].isna().to_numpy(bool)
            order = np.argsort(x)
            x, y, missing = x[order], y[order], missing[order]
            if len(np.unique(x)) >= 4:
                dense = np.linspace(x.min(), x.max(), 220)
                try:
                    smooth = PchipInterpolator(x, y)(dense)
                    ax.plot(dense, smooth, color=color, lw=2.8, alpha=0.96, label=f"{typ} fitted smile")
                except Exception:
                    ax.plot(x, y, color=color, lw=2.2, alpha=0.96, label=f"{typ} fitted smile")
            else:
                ax.plot(x, y, color=color, lw=2.2, alpha=0.96, label=f"{typ} fitted smile")
            ax.scatter(x[~missing], y[~missing], color=color, s=18, edgecolor="#07101f", linewidth=0.4, alpha=0.82, zorder=3)
            ax.scatter(x[missing], y[missing], color="#f8c14a", marker="D", s=54, edgecolor="black", linewidth=0.55, zorder=5)
        ax.set_title(dates.iloc[idx].strftime("%d %b %Y %H:%M"), color="white", weight="bold")
        ax.set_xlabel("moneyness K/S", color="#d8e1ff")
        ax.set_ylabel("IV", color="#d8e1ff")
        ax.tick_params(colors="#d8e1ff")
        ax.grid(color="#253149", alpha=0.55)
        for spine in ax.spines.values():
            spine.set_color("#34415f")
    axes.flat[0].legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white", fontsize=8)
    fig.suptitle("fitted smiles across the month: inferred cells highlighted in gold", color="white", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "filled_smile_examples.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_error_surface(diag):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    fig.patch.set_facecolor("#0a0d14")
    ax.set_facecolor("#0f1724")
    src = diag["source"].value_counts()
    colors = ["#00d4ff", "#ff4d6d", "#6ee7b7", "#f8c14a"]
    ax.barh(src.index, src.values, color=colors[:len(src)])
    ax.set_title("which engine produced each filled cell", color="white", fontsize=18, weight="bold")
    ax.set_xlabel("filled cells", color="#d8e1ff")
    ax.tick_params(colors="#d8e1ff")
    ax.grid(axis="x", color="#253149", alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#34415f")
    fig.tight_layout()
    fig.savefig(OUT / "prediction_source_counts.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_cv_strategy_comparison():
    rows = []
    for name, path in STRATEGY_METRICS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["method"] = name
        rows.append(row)
    if not rows:
        return
    comp = pd.DataFrame(rows).sort_values("mse")
    comp.to_csv(OUT / "strategy_cv_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    fig.patch.set_facecolor("#0a0d14")
    for ax in axes:
        ax.set_facecolor("#0f1724")
        ax.tick_params(colors="#d8e1ff")
        ax.grid(axis="x", color="#253149", alpha=0.55)
        for spine in ax.spines.values():
            spine.set_color("#34415f")

    colors = ["#6ee7b7" if m == "final" else "#4f7cff" for m in comp["method"]]
    axes[0].barh(comp["method"], comp["mse"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MSE on synthetic CV holdout", color="#d8e1ff")
    axes[0].set_title("final method vs prior attempts", color="white", weight="bold")

    axes[1].barh(comp["method"], comp["p95_abs_error"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("95th percentile absolute error", color="#d8e1ff")
    axes[1].set_title("tail error comparison", color="white", weight="bold")

    fig.suptitle("why the final method was chosen", color="white", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "strategy_cv_comparison.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_cv_metric_cards():
    if not CV_METRICS_PATH.exists():
        return
    row = pd.read_csv(CV_METRICS_PATH).iloc[0]
    metrics = [
        ("MSE", row["mse"]),
        ("RMSE", row["rmse"]),
        ("MAE", row["mae"]),
        ("Median abs err", row["median_abs_error"]),
        ("P95 abs err", row["p95_abs_error"]),
        ("Bias", row["bias_mean_error"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor("#0a0d14")
    for ax, (label, value) in zip(axes.flat, metrics):
        ax.set_facecolor("#0f1724")
        ax.text(0.5, 0.62, label, ha="center", va="center", color="#d8e1ff", fontsize=15, weight="bold")
        ax.text(0.5, 0.34, f"{value:.6f}", ha="center", va="center", color="#6ee7b7", fontsize=24, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#34415f")
    fig.suptitle("synthetic CV holdout results for final_submission.py", color="white", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "cv_metric_cards.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.set_facecolor("#0f1724")
    ax.tick_params(colors="#d8e1ff")
    ax.grid(color="#253149", alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#34415f")


def save_themed_cv_plots():
    cv_dir = OUT / "cv_eval_final_submission"
    errors_path = cv_dir / "error_rows.csv"
    if not errors_path.exists():
        return

    errors = pd.read_csv(errors_path)
    ce_color = "#00d4ff"
    pe_color = "#ff4d6d"
    gold = "#f8c14a"
    green = "#6ee7b7"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    style_axis(ax)
    for typ, color in [("CE", ce_color), ("PE", pe_color)]:
        sub = errors[errors["option_type"] == typ]
        ax.scatter(sub["actual_iv"], sub["predicted_iv"], s=18, alpha=0.62, color=color, label=typ, edgecolors="none")
    lo = min(errors["actual_iv"].min(), errors["predicted_iv"].min())
    hi = max(errors["actual_iv"].max(), errors["predicted_iv"].max())
    ax.plot([lo, hi], [lo, hi], color=gold, lw=2, linestyle="--", label="perfect prediction")
    ax.set_title("synthetic CV: predicted vs actual IV", color="white", fontsize=18, weight="bold")
    ax.set_xlabel("actual IV", color="#d8e1ff")
    ax.set_ylabel("predicted IV", color="#d8e1ff")
    ax.legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white")
    fig.tight_layout()
    fig.savefig(OUT / "cv_predicted_vs_actual_theme.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    style_axis(ax)
    ax.hist(errors["error"], bins=70, color=ce_color, alpha=0.82)
    ax.axvline(0, color=gold, lw=2, linestyle="--")
    ax.set_title("synthetic CV: error distribution", color="white", fontsize=18, weight="bold")
    ax.set_xlabel("prediction error = predicted - actual", color="#d8e1ff")
    ax.set_ylabel("count", color="#d8e1ff")
    fig.tight_layout()
    fig.savefig(OUT / "cv_error_histogram_theme.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    style_axis(ax)
    for typ, color in [("CE", ce_color), ("PE", pe_color)]:
        sub = errors[errors["option_type"] == typ]
        ax.scatter(sub["row_index"], sub["abs_error"], s=14, alpha=0.58, color=color, label=typ, edgecolors="none")
    ax.set_title("synthetic CV: absolute error over time", color="white", fontsize=18, weight="bold")
    ax.set_xlabel("timestamp row index", color="#d8e1ff")
    ax.set_ylabel("absolute error", color="#d8e1ff")
    ax.legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white")
    fig.tight_layout()
    fig.savefig(OUT / "cv_abs_error_over_time_theme.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    style_axis(ax)
    for typ, color in [("CE", ce_color), ("PE", pe_color)]:
        sub = errors[errors["option_type"] == typ]
        ax.scatter(sub["moneyness"], sub["abs_error"], s=16, alpha=0.58, color=color, label=typ, edgecolors="none")
    ax.set_title("synthetic CV: absolute error vs moneyness", color="white", fontsize=18, weight="bold")
    ax.set_xlabel("moneyness K/S", color="#d8e1ff")
    ax.set_ylabel("absolute error", color="#d8e1ff")
    ax.legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white")
    fig.tight_layout()
    fig.savefig(OUT / "cv_abs_error_vs_moneyness_theme.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    for group_file, label_col, out_name, title in [
        ("group_metrics_by_regime.csv", "regime", "cv_mse_by_regime_theme.png", "synthetic CV: MSE by regime"),
        ("group_metrics_by_option_type.csv", "option_type", "cv_mse_by_option_type_theme.png", "synthetic CV: MSE by option type"),
    ]:
        path = cv_dir / group_file
        if not path.exists():
            continue
        gm = pd.read_csv(path).sort_values("mse", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5.4), dpi=190)
        fig.patch.set_facecolor("#0a0d14")
        style_axis(ax)
        colors = [pe_color if str(x).upper() == "PE" or str(x).lower() == "jan27" else ce_color for x in gm[label_col]]
        ax.bar(gm[label_col].astype(str), gm["mse"], color=colors)
        ax.set_title(title, color="white", fontsize=18, weight="bold")
        ax.set_xlabel(label_col.replace("_", " "), color="#d8e1ff")
        ax.set_ylabel("MSE", color="#d8e1ff")
        fig.tight_layout()
        fig.savefig(OUT / out_name, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    make_cv_heatmap(errors, "abs_error", "cv_abs_error_heatmap_theme.png", "synthetic CV: binned absolute error", cmap="mako")
    make_cv_heatmap(errors, "error", "cv_signed_error_heatmap_theme.png", "synthetic CV: binned signed error", cmap="coolwarm", center=0.0)
    save_themed_top_error_smile(errors)


def make_cv_heatmap(errors, value_col, out_name, title, cmap="mako", center=None):
    df = errors.copy()
    df["time_bin"] = pd.cut(df["row_index"], bins=70, labels=False)
    df["money_bin"] = pd.cut(df["moneyness"], bins=34, labels=False)
    agg = df.pivot_table(index="money_bin", columns="time_bin", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    ax.set_facecolor("#0f1724")
    sns.heatmap(
        agg,
        cmap=cmap,
        center=center,
        ax=ax,
        cbar_kws={"label": value_col.replace("_", " ")},
    )
    ax.invert_yaxis()
    ax.set_title(title, color="white", fontsize=18, weight="bold")
    ax.set_xlabel("time bin", color="#d8e1ff")
    ax.set_ylabel("moneyness bin", color="#d8e1ff")
    ax.tick_params(colors="#d8e1ff", labelsize=7)
    ax.collections[0].colorbar.ax.tick_params(colors="#d8e1ff")
    ax.collections[0].colorbar.set_label(value_col.replace("_", " "), color="#d8e1ff")
    fig.tight_layout()
    fig.savefig(OUT / out_name, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_themed_top_error_smile(errors):
    pred_path = ROOT / "filled_dataset_readme_cv_final.csv"
    not_dataset, _ = load_first(NOT_DATASET_PATHS)
    if not pred_path.exists() or not_dataset is None:
        return
    pred = pd.read_csv(pred_path)
    group = (
        errors.groupby(["row_index", "option_type"], as_index=False)
        .agg(mse=("sq_error", "mean"))
        .sort_values("mse", ascending=False)
        .iloc[0]
    )
    row_idx = int(group["row_index"])
    opt_type = group["option_type"]
    gerr = errors[(errors["row_index"] == row_idx) & (errors["option_type"] == opt_type)]
    opts = [c for c in option_columns(pred) if parse_contract(c)[1] == opt_type]
    opts = sorted(opts, key=lambda c: parse_contract(c)[0])
    spot = pred.loc[row_idx, "underlying_price"]
    x = np.array([parse_contract(c)[0] / spot for c in opts], float)
    y = pred.loc[row_idx, opts].to_numpy(float)
    avail = not_dataset.loc[row_idx, opts].notna().to_numpy(bool)
    hidden_cols = set(gerr["contract"])
    hidden = np.array([c in hidden_cols for c in opts], bool)
    truth_map = dict(zip(gerr["contract"], gerr["actual_iv"]))
    truth = np.array([truth_map.get(c, np.nan) for c in opts], float)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    style_axis(ax)
    order = np.argsort(x)
    x, y, avail, hidden, truth = x[order], y[order], avail[order], hidden[order], truth[order]
    dense = np.linspace(x.min(), x.max(), 260)
    try:
        smooth = PchipInterpolator(x, y)(dense)
        ax.plot(dense, smooth, color="#00d4ff" if opt_type == "CE" else "#ff4d6d", lw=3, label="filled smile")
    except Exception:
        ax.plot(x, y, color="#00d4ff" if opt_type == "CE" else "#ff4d6d", lw=3, label="filled smile")
    ax.scatter(x[avail], y[avail], color="#00d4ff", s=34, edgecolor="#07101f", linewidth=0.5, label="available observations", zorder=3)
    ax.scatter(x[hidden], y[hidden], color="#f8c14a", marker="D", s=70, edgecolor="black", linewidth=0.55, label="model prediction", zorder=4)
    mask_truth = hidden & np.isfinite(truth)
    ax.scatter(x[mask_truth], truth[mask_truth], color="white", marker="x", s=70, linewidth=2.0, label="hidden truth", zorder=5)
    for xi, yp, yt in zip(x[mask_truth], y[mask_truth], truth[mask_truth]):
        ax.plot([xi, xi], [yt, yp], color="#f8c14a", lw=1.25, alpha=0.75)
    ax.set_title(f"top-error CV smile: {pred.loc[row_idx, 'datetime']} {opt_type}", color="white", fontsize=18, weight="bold")
    ax.set_xlabel("moneyness K/S", color="#d8e1ff")
    ax.set_ylabel("IV", color="#d8e1ff")
    ax.legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "cv_top_error_smile_theme.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    if not FILLED_PATH.exists():
        raise FileNotFoundError(f"Missing {FILLED_PATH}")
    if not DIAG_PATH.exists():
        raise FileNotFoundError(f"Missing {DIAG_PATH}")

    filled = pd.read_csv(FILLED_PATH)
    diag = pd.read_csv(DIAG_PATH)
    original, _ = load_first(DATA_PATHS)
    not_dataset, _ = load_first(NOT_DATASET_PATHS)
    if original is None:
        original = filled.copy()

    save_dataset_regime_eda(original)
    save_missing_given_cross_section_eda(original)
    save_surface(filled, "CE", "iv_surface_ce_3d.png", "final filled CE implied-volatility surface", original)
    save_surface(filled, "PE", "iv_surface_pe_3d.png", "final filled PE implied-volatility surface", original)
    save_combined_surface(filled, original)
    save_missingness(original, not_dataset)
    save_decision_charts(diag)
    save_smile_examples(filled, original)
    save_error_surface(diag)
    save_cv_strategy_comparison()
    save_cv_metric_cards()
    save_themed_cv_plots()


if __name__ == "__main__":
    main()
