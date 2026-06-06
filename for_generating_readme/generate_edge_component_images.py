from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

ORIGINAL_PATH = ROOT / "dataset.csv"
FILLED_PATH = ROOT / "submission_files" / "filled_dataset_final.csv"
DIAG_PATH = ROOT / "submission_files" / "diagnostics_final.csv"

WEIGHTS = {
    "primary": 0.70,
    "secondary": 0.15,
    "nearby": 0.15,
}


def option_columns(df):
    return [c for c in df.columns if c not in {"datetime", "datetime_parsed", "underlying_price"}]


def parse_contract(col):
    m = __import__("re").match(r"^[A-Z]+(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$", col)
    if not m:
        return None
    return int(m.group(2)), m.group(3)


def style_axis(ax):
    ax.set_facecolor("#101722")
    ax.grid(color="#263247", alpha=0.52, linewidth=0.85)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#34415f")
        spine.set_linewidth(1.05)


def load_block(original, filled, diag):
    edge = diag[(diag["edge"] == True) & diag["edge_side"].isin(["left", "right"])].copy()
    pick = edge[
        (edge["datetime"] == "09-01-2026 12:15")
        & (edge["option_type"] == "CE")
        & (edge["edge_side"] == "right")
    ].sort_values("edge_position_in_block")
    if pick.empty:
        first = (
            edge.sort_values(["edge_block_size", "row_index"], ascending=[False, True])
            .groupby(["row_index", "option_type", "edge_side"], as_index=False)
            .head(1)
            .iloc[0]
        )
        pick = edge[
            (edge["row_index"] == first["row_index"])
            & (edge["option_type"] == first["option_type"])
            & (edge["edge_side"] == first["edge_side"])
        ].sort_values("edge_position_in_block")
    row_idx = int(pick.iloc[0]["row_index"])
    typ = pick.iloc[0]["option_type"]
    side = pick.iloc[0]["edge_side"]
    opts = option_columns(filled)
    parsed = {c: parse_contract(c) for c in opts}
    cols = sorted([c for c in opts if parsed[c] and parsed[c][1] == typ], key=lambda c: parsed[c][0])
    spot = float(filled.loc[row_idx, "underlying_price"])
    x = np.array([parsed[c][0] / spot for c in cols], float)
    y0 = original.loc[row_idx, cols].to_numpy(float)
    yf = filled.loc[row_idx, cols].to_numpy(float)
    block_cols = pick["contract"].tolist()
    return row_idx, typ, side, pick, cols, x, y0, yf


def component_values(block, component):
    if component == "secondary":
        col = "edge_secondary_prediction" if "edge_secondary_prediction" in block.columns else "edge_corrected_prediction"
    elif component == "nearby":
        col = "edge_nearby_prediction" if "edge_nearby_prediction" in block.columns else "edge_quadratic_prediction"
    else:
        col = f"edge_{component}_prediction"
    return dict(zip(block["contract"], block[col]))


def filled_series_for_component(y0, cols, revealed, comp_map):
    y = y0.copy()
    for c in revealed:
        if c in comp_map:
            y[cols.index(c)] = comp_map[c]
    return y


def plot_sequence(component, original, filled, diag):
    row_idx, typ, side, block, cols, x, y0, yf = load_block(original, filled, diag)
    block_cols = block["contract"].tolist()
    comp_map = component_values(block, component)
    strikes = np.array([parse_contract(c)[0] for c in cols], float)
    base_n = max(3, len(block_cols))
    max_panels = len(block_cols) + 1
    fig, axes = plt.subplots(1, max_panels, figsize=(3.25 * max_panels, 4.95), dpi=190, sharey=True)
    fig.patch.set_facecolor("#0a0d14")
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    obs = np.isfinite(y0)
    dt = filled.loc[row_idx, "datetime"]

    for step, ax in enumerate(axes):
        style_axis(ax)
        revealed = block_cols[:max(0, step)]
        current = block_cols[step - 1] if step > 0 else None
        y = filled_series_for_component(y0, cols, revealed, comp_map)
        filled_mask = np.array([c in revealed for c in cols])
        future_mask = np.array([(c in set(block_cols)) and (c not in revealed) for c in cols])
        finite = np.isfinite(y)
        ax.plot(x[finite], y[finite], color="#38bdf8", lw=2.1, alpha=0.66)
        ax.scatter(x[obs], y0[obs], color="#00d4ff", s=26, edgecolor="#101722", linewidth=0.4, label="given")
        if future_mask.any() and obs.sum() >= 2:
            ax.scatter(x[future_mask], np.interp(x[future_mask], x[obs], y0[obs]),
                       color="#f8c14a", marker="x", s=46, linewidth=1.6, label="still missing")
        ax.scatter(x[filled_mask], y[filled_mask], color="#f8c14a", marker="D", s=56,
                   edgecolor="#101722", linewidth=0.55, label=f"{component} fill")
        if current is not None:
            ci = cols.index(current)
            ax.scatter([x[ci]], [comp_map[current]], color="#ff4d6d", marker="D", s=84,
                       edgecolor="#d8e1ff", linewidth=0.8)
        if component == "primary" and step >= 2:
            prev = block_cols[:step - 1]
            if prev:
                ax.annotate(
                    f"{len(prev)} prior fill{'s' if len(prev) > 1 else ''}\nstored at x=0",
                    xy=(0.02, comp_map[prev[-1]]),
                    xycoords=("axes fraction", "data"),
                    xytext=(0.26, 0.31),
                    textcoords="axes fraction",
                    color="#ff4d6d",
                    fontsize=8.3,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#101722", ec="#ff4d6d", lw=0.8, alpha=0.86),
                    arrowprops=dict(arrowstyle="->", color="#ff4d6d", lw=1.05, alpha=0.85),
                )
        if component == "nearby" and step > 0 and current is not None:
            target_s = strikes[cols.index(current)]
            if side == "right":
                candidates = np.where(obs & (strikes < target_s))[0]
            elif side == "left":
                candidates = np.where(obs & (strikes > target_s))[0]
            else:
                candidates = np.where(obs)[0]
            if len(candidates):
                order = candidates[np.argsort(np.abs(strikes[candidates] - target_s))][:base_n]
                lo, hi = float(np.min(x[order])), float(np.max(x[order]))
                ax.axvspan(lo, hi, color="#6ee7b7", alpha=0.10, lw=0, zorder=0)
                ax.scatter(x[order], y0[order], facecolors="none", edgecolors="#6ee7b7",
                           s=105, linewidth=1.35, zorder=6, label="selected neighborhood")
                prior = [cols.index(c) for c in revealed[:-1] if c in comp_map]
                if prior:
                    ax.scatter(x[prior], y[prior], facecolors="none", edgecolors="#6ee7b7",
                               s=120, linewidth=1.35, zorder=6)
                ax.text(0.03, 0.92, "selected\nneighborhood", transform=ax.transAxes,
                        color="#6ee7b7", fontsize=8.5, fontweight="bold", va="top")
        elif component == "secondary" and step >= 2:
            ax.text(0.03, 0.92, "uses previous fills\nat real K/S", transform=ax.transAxes,
                    color="#f8c14a", fontsize=8.5, fontweight="bold", va="top")
        ax.set_title("start" if step == 0 else f"fill {step}", color="#d8e1ff", weight="bold")
        ax.set_xlabel(r"moneyness $K/S$", color="#d8e1ff")
        if step == 0:
            ax.set_ylabel("IV", color="#d8e1ff")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", ncol=3,
               facecolor="#101722", edgecolor="#34415f", labelcolor="#d8e1ff")
    subtitles = {
        "primary": "previous primary predictions are tracked, but shifted to x = 0 so they do not pull the local moneyness fit",
        "secondary": "progressive edge filling using all valid-side observed points plus previous secondary fills at real moneyness",
        "nearby": "progressive local-wing filling using nearby observed points plus previous nearby fills at real moneyness",
    }
    fig.suptitle(f"{component} edge sequence: {dt} {typ} {side} wing\n{subtitles[component]}",
                 color="#d8e1ff", fontsize=15.5, weight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.89])
    fig.savefig(OUT / f"edge_{component}_sequence.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_overlay(original, filled, diag):
    row_idx, typ, side, block, cols, x, y0, yf = load_block(original, filled, diag)
    block_cols = block["contract"].tolist()
    maps = {name: component_values(block, name) for name in ["primary", "secondary", "nearby"]}
    blend_map = {
        c: WEIGHTS["primary"] * maps["primary"][c]
        + WEIGHTS["secondary"] * maps["secondary"][c]
        + WEIGHTS["nearby"] * maps["nearby"][c]
        for c in block_cols
    }
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=190)
    fig.patch.set_facecolor("#0a0d14")
    style_axis(ax)
    obs = np.isfinite(y0)
    ax.plot(x[obs], y0[obs], color="#38bdf8", lw=2.3, alpha=0.62, label="given smile")
    ax.scatter(x[obs], y0[obs], color="#00d4ff", s=38, edgecolor="#101722", linewidth=0.5, zorder=4, label="given")
    colors = {"primary": "#ff4d6d", "secondary": "#f8c14a", "nearby": "#a78bfa"}
    markers = {"primary": "o", "secondary": "D", "nearby": "s"}
    for name in ["primary", "secondary", "nearby"]:
        yy = y0.copy()
        for c, val in maps[name].items():
            yy[cols.index(c)] = val
        finite = np.isfinite(yy)
        ax.plot(x[finite], yy[finite], color=colors[name], lw=1.75, alpha=0.7, label=f"{name} path")
        ax.scatter([x[cols.index(c)] for c in block_cols], [maps[name][c] for c in block_cols],
                   color=colors[name], marker=markers[name], s=55, edgecolor="#101722", linewidth=0.5, zorder=5)
    ax.plot([x[cols.index(c)] for c in block_cols], [blend_map[c] for c in block_cols],
            color="#6ee7b7", lw=3.2, alpha=0.95, label="weighted blend")
    ax.scatter([x[cols.index(c)] for c in block_cols], [blend_map[c] for c in block_cols],
               color="#6ee7b7", marker="*", s=175, edgecolor="#101722", linewidth=0.7, zorder=6)
    ax.text(0.02, 0.96, "blend = 0.70 primary + 0.15 secondary + 0.15 nearby",
            transform=ax.transAxes, color="#6ee7b7", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0.02, 0.89, "primary history is shifted to x=0; secondary/nearby are progressive at real K/S",
            transform=ax.transAxes, color="#d8e1ff", fontsize=10.5, va="top")
    ax.set_xlabel(r"moneyness $K/S$", color="#d8e1ff")
    ax.set_ylabel("IV", color="#d8e1ff")
    ax.set_title(f"edge component overlay and weighted imputation: {filled.loc[row_idx, 'datetime']} {typ} {side} wing",
                 color="#d8e1ff", fontsize=16, weight="bold")
    ax.legend(loc="lower left", facecolor="#101722", edgecolor="#34415f", labelcolor="#d8e1ff", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "edge_component_blend_overlay.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    original = pd.read_csv(ORIGINAL_PATH)
    filled = pd.read_csv(FILLED_PATH)
    diag = pd.read_csv(DIAG_PATH)
    for component in ["primary", "secondary", "nearby"]:
        plot_sequence(component, original, filled, diag)
    plot_overlay(original, filled, diag)


if __name__ == "__main__":
    main()
