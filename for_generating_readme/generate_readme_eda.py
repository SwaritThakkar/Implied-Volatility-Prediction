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
            "strike": strike,
            "moneyness": x,
            "iv": df[col].to_numpy(float),
            "contract": col,
        }))
    return pd.concat(rows, ignore_index=True)


def save_surface(df, opt_type, filename, title):
    long = prepare_long(df, opt_type)
    long = long[np.isfinite(long["iv"]) & np.isfinite(long["moneyness"])].copy()
    lo, hi = long["iv"].quantile([0.01, 0.99])
    long["iv_plot"] = long["iv"].clip(lo, hi)

    t_min, t_max = long["time_index"].min(), long["time_index"].max()
    m_min, m_max = long["moneyness"].quantile([0.002, 0.998])
    grid_t = np.linspace(t_min, t_max, 180)
    grid_m = np.linspace(m_min, m_max, 90)
    T, M = np.meshgrid(grid_t, grid_m)
    pts = long[["time_index", "moneyness"]].to_numpy(float)
    vals = long["iv_plot"].to_numpy(float)
    Z = griddata(pts, vals, (T, M), method="linear")
    nearest = griddata(pts, vals, (T, M), method="nearest")
    Z = np.where(np.isfinite(Z), Z, nearest)

    sample = long.iloc[::4].copy()
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=T,
        y=M,
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
        x=sample["time_index"],
        y=sample["moneyness"],
        z=sample["iv_plot"],
        mode="markers",
        marker={"size": 2.2, "color": sample["iv_plot"], "colorscale": "Turbo", "opacity": 0.55},
        name="filled contracts",
        hovertemplate="t=%{x}<br>K/S=%{y:.4f}<br>IV=%{z:.4f}<extra></extra>",
    ))
    for idx in np.linspace(t_min, t_max, 9, dtype=int):
        g = long[np.abs(long["time_index"] - idx) <= 2].groupby("moneyness", as_index=False)["iv_plot"].median()
        if len(g) >= 4:
            fig.add_trace(go.Scatter3d(
                x=np.full(len(g), idx),
                y=g["moneyness"],
                z=g["iv_plot"],
                mode="lines",
                line={"color": "white", "width": 5},
                opacity=0.62,
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 30, "color": "white"}},
        template="plotly_dark",
        width=1700,
        height=1100,
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        paper_bgcolor="#050812",
        scene={
            "xaxis": {"title": "timestamp index", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "yaxis": {"title": "moneyness K/S", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "zaxis": {"title": "IV, clipped to 1st-99th pct", "gridcolor": "#3a4766", "backgroundcolor": "#07101f", "color": "#d8e1ff"},
            "camera": {"eye": {"x": 1.65, "y": -1.72, "z": 0.92}, "center": {"x": 0.02, "y": 0.0, "z": -0.08}},
            "aspectratio": {"x": 2.35, "y": 1.0, "z": 0.72},
        },
        legend={"font": {"color": "#d8e1ff"}},
    )
    html_name = filename.replace(".png", ".html")
    fig.write_html(OUT / html_name, include_plotlyjs="cdn", full_html=True)

    static = plt.figure(figsize=(18, 10.5), dpi=190)
    static.patch.set_facecolor("#050812")
    gs = static.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 1.0], height_ratios=[1.0, 0.78])
    ax3d = static.add_subplot(gs[:, 0], projection="3d")
    axh = static.add_subplot(gs[0, 1:])
    axs = static.add_subplot(gs[1, 1:])

    ax3d.set_facecolor("#050812")
    surf = ax3d.plot_surface(
        T, M, Z,
        cmap="turbo",
        linewidth=0,
        antialiased=True,
        alpha=0.98,
        rstride=1,
        cstride=1,
        shade=True,
    )
    ax3d.contour(T, M, Z, zdir="z", offset=float(np.nanmin(Z)), levels=12, cmap="turbo", alpha=0.55, linewidths=0.7)
    ax3d.scatter(sample["time_index"], sample["moneyness"], sample["iv_plot"], c="#ffffff", s=4, alpha=0.16, depthshade=False)
    ax3d.view_init(elev=31, azim=-124)
    ax3d.set_box_aspect((2.3, 1.0, 0.88))
    ax3d.set_title(title, color="white", fontsize=19, weight="bold", pad=14)
    ax3d.set_xlabel("time index", color="#d8e1ff", labelpad=10)
    ax3d.set_ylabel("moneyness K/S", color="#d8e1ff", labelpad=10)
    ax3d.set_zlabel("IV", color="#d8e1ff", labelpad=10)
    ax3d.tick_params(colors="#d8e1ff", labelsize=8)
    ax3d.set_xlim(t_min, t_max)
    ax3d.set_ylim(m_min, m_max)
    ax3d.set_zlim(float(np.nanmin(Z)), float(np.nanmax(Z)))
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.pane.set_facecolor((0.04, 0.06, 0.12, 0.92))
        axis.pane.set_edgecolor((0.52, 0.64, 0.9, 0.7))
    ax3d.grid(True, color="#52617f", alpha=0.35)

    axh.set_facecolor("#07101f")
    heat = axh.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap="turbo",
        extent=[t_min, t_max, m_min, m_max],
        interpolation="bicubic",
    )
    axh.set_title("top-down IV heatmap", color="white", weight="bold", fontsize=14)
    axh.set_xlabel("time index", color="#d8e1ff")
    axh.set_ylabel("moneyness K/S", color="#d8e1ff")
    axh.tick_params(colors="#d8e1ff")
    for spine in axh.spines.values():
        spine.set_color("#34415f")

    axs.set_facecolor("#07101f")
    slice_indices = [0, len(df) // 3, 2 * len(df) // 3, len(df) - 1]
    colors2 = ["#38bdf8", "#a78bfa", "#f8c14a", "#fb7185"]
    for idx, color in zip(slice_indices, colors2):
        g = long[np.abs(long["time_index"] - idx) <= 1].sort_values("moneyness")
        if len(g) < 4:
            continue
        xg = g["moneyness"].to_numpy(float)
        yg = g["iv_plot"].to_numpy(float)
        ux, ix = np.unique(xg, return_index=True)
        uy = yg[ix]
        dense = np.linspace(ux.min(), ux.max(), 200)
        try:
            smooth = PchipInterpolator(ux, uy)(dense)
            axs.plot(dense, smooth, color=color, lw=2.6, label=df.loc[idx, "datetime"])
        except Exception:
            axs.plot(ux, uy, color=color, lw=2.6, label=df.loc[idx, "datetime"])
    axs.set_title("selected fitted smile slices", color="white", weight="bold", fontsize=14)
    axs.set_xlabel("moneyness K/S", color="#d8e1ff")
    axs.set_ylabel("IV", color="#d8e1ff")
    axs.tick_params(colors="#d8e1ff")
    axs.grid(color="#253149", alpha=0.6)
    axs.legend(facecolor="#101722", edgecolor="#34415f", labelcolor="white", fontsize=8, ncol=2)
    for spine in axs.spines.values():
        spine.set_color("#34415f")

    cax = static.add_axes([0.925, 0.18, 0.014, 0.64])
    cbar = static.colorbar(surf, cax=cax)
    cbar.ax.tick_params(colors="#d8e1ff")
    cbar.set_label("IV, clipped to 1st-99th pct", color="#d8e1ff")
    static.subplots_adjust(left=0.035, right=0.895, top=0.93, bottom=0.08, wspace=0.32, hspace=0.34)
    static.savefig(OUT / filename, facecolor=static.get_facecolor(), bbox_inches="tight")
    plt.close(static)


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

    save_surface(filled, "CE", "iv_surface_ce_3d.png", "final filled CE implied-volatility surface")
    save_surface(filled, "PE", "iv_surface_pe_3d.png", "final filled PE implied-volatility surface")
    save_missingness(original, not_dataset)
    save_decision_charts(diag)
    save_smile_examples(filled, original)
    save_error_surface(diag)
    save_cv_strategy_comparison()
    save_cv_metric_cards()


if __name__ == "__main__":
    main()
