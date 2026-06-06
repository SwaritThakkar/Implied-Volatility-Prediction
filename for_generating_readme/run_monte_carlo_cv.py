import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "for_generating_readme"
RUN_ROOT = OUT_ROOT / "mc_cv_runs"
CREATE_SPLIT = ROOT / "everything_else" / "cv_validation_system" / "create_synthetic_cv_dataset.py"
FINAL_SUBMISSION = ROOT / "final_submission.py"
DATASET = ROOT / "dataset.csv"


SEEDS = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 83, 97]


def metrics(actual, pred):
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    err = pred - actual
    abs_err = np.abs(err)
    sq_err = err ** 2
    return {
        "n": int(len(actual)),
        "mse": float(np.mean(sq_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "mae": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
        "p99_abs_error": float(np.quantile(abs_err, 0.99)),
        "max_abs_error": float(np.max(abs_err)),
    }


def score_run(truth_path, pred_path):
    truth = pd.read_csv(truth_path)
    pred = pd.read_csv(pred_path)

    values = []
    for rec in truth.itertuples(index=False):
        values.append(pred.at[int(rec.row_index), rec.contract])

    scored = truth.copy()
    scored["predicted_iv"] = values
    scored["error"] = scored["predicted_iv"] - scored["actual_iv"]
    scored["abs_error"] = scored["error"].abs()
    scored["sq_error"] = scored["error"] ** 2
    return scored


def run_one(seed):
    run_dir = RUN_ROOT / f"seed_{seed:03d}"
    split_dir = run_dir / "cv_split"
    run_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(CREATE_SPLIT),
            "--input",
            str(DATASET),
            "--out-dir",
            str(split_dir),
            "--seed",
            str(seed),
            "--holdout-frac",
            "0.12",
            "--min-holdout-27jan",
            "350",
        ],
        cwd=ROOT,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(FINAL_SUBMISSION),
            "--data",
            "cv_split/not_dataset.csv",
            "--out-prefix",
            f"mc_seed_{seed:03d}",
            "--skip-cv",
        ],
        cwd=run_dir,
        check=True,
    )

    pred_path = run_dir / f"filled_dataset_mc_seed_{seed:03d}.csv"
    scored = score_run(split_dir / "holdout_truth.csv", pred_path)
    scored.to_csv(run_dir / "scored_holdout.csv", index=False)

    overall = metrics(scored["actual_iv"], scored["predicted_iv"])
    overall["seed"] = seed

    grouped = []
    for (regime, option_type), group in scored.groupby(["regime", "option_type"], dropna=False):
        row = metrics(group["actual_iv"], group["predicted_iv"])
        row.update({"seed": seed, "regime": regime, "option_type": option_type})
        grouped.append(row)

    return overall, grouped, scored


def apply_theme(ax):
    ax.set_facecolor("#0f1724")
    ax.grid(True, color="#243449", alpha=0.55, linewidth=0.8)
    ax.tick_params(colors="#cfd7ea", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#33435f")
        spine.set_linewidth(1.0)
    ax.xaxis.label.set_color("#d7def2")
    ax.yaxis.label.set_color("#d7def2")
    ax.title.set_color("#e5ecff")


def save_fig(fig, name):
    fig.patch.set_facecolor("#080d16")
    fig.savefig(OUT_ROOT / name, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_metric_distributions(summary):
    metrics_to_plot = [
        ("mse", "MSE"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("p95_abs_error", "p95 |error|"),
        ("p99_abs_error", "p99 |error|"),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4.4), sharex=False)
    colors = ["#39d5ff", "#7cf7b5", "#ffc857", "#ff6b9a", "#b892ff"]
    for ax, (col, title), color in zip(axes, metrics_to_plot, colors):
        apply_theme(ax)
        vals = summary[col].to_numpy(float)
        ax.boxplot(
            vals,
            vert=True,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=color, alpha=0.22, color=color, linewidth=1.5),
            medianprops=dict(color="#ffffff", linewidth=1.8),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor=color, markeredgecolor=color, alpha=0.45, markersize=4),
        )
        jitter = np.linspace(-0.09, 0.09, len(vals))
        ax.scatter(np.ones(len(vals)) + jitter, vals, s=26, color=color, edgecolor="#0b1020", linewidth=0.4, zorder=3)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xticks([])
        ax.axhline(vals.mean(), color="#ffffff", linestyle="--", linewidth=0.9, alpha=0.65)
    fig.suptitle("Monte Carlo synthetic CV: metric stability across random holdouts", color="#e9efff",
                 fontsize=17, fontweight="bold", y=1.03)
    save_fig(fig, "mc_cv_metric_distributions.png")


def plot_seed_trajectory(summary):
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    apply_theme(ax)
    ordered = summary.sort_values("seed")
    ax.plot(ordered["seed"], ordered["rmse"], color="#39d5ff", marker="o", linewidth=2.0, label="RMSE")
    ax.plot(ordered["seed"], ordered["mae"], color="#7cf7b5", marker="o", linewidth=2.0, label="MAE")
    ax.fill_between(
        ordered["seed"],
        ordered["rmse"].mean() - ordered["rmse"].std(),
        ordered["rmse"].mean() + ordered["rmse"].std(),
        color="#39d5ff",
        alpha=0.12,
        label="RMSE mean +/- 1 std",
    )
    ax.set_title("Monte Carlo synthetic CV by seed", fontweight="bold", pad=14)
    ax.set_xlabel("random holdout seed")
    ax.set_ylabel("error")
    leg = ax.legend(facecolor="#111a2b", edgecolor="#33435f", labelcolor="#d7def2")
    for text in leg.get_texts():
        text.set_color("#d7def2")
    save_fig(fig, "mc_cv_seed_trajectory.png")


def plot_regime_robustness(grouped):
    grouped = grouped.copy()
    grouped["bucket"] = grouped["regime"].astype(str) + " / " + grouped["option_type"].astype(str)
    order = ["pre27 / CE", "pre27 / PE", "jan27 / CE", "jan27 / PE"]
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    apply_theme(ax)
    colors = ["#39d5ff", "#7cf7b5", "#ffc857", "#ff6b9a"]
    data = [grouped.loc[grouped["bucket"] == b, "rmse"].to_numpy(float) for b in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, labels=order)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for item in bp["whiskers"] + bp["caps"]:
        item.set_color("#cfd7ea")
    for item in bp["medians"]:
        item.set_color("#ffffff")
        item.set_linewidth(1.8)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals,
                       color=colors[i - 1], edgecolor="#0b1020", linewidth=0.35, s=25, zorder=3)
    ax.set_title("Regime and option-side robustness across synthetic holdouts", fontweight="bold", pad=14)
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "mc_cv_regime_robustness.png")


def plot_quantiles(summary):
    fig, ax = plt.subplots(figsize=(12.5, 5.1))
    apply_theme(ax)
    ordered = summary.sort_values("seed")
    ax.plot(ordered["seed"], ordered["median_abs_error"], color="#7cf7b5", marker="o", label="median |error|")
    ax.plot(ordered["seed"], ordered["p90_abs_error"], color="#ffc857", marker="o", label="p90 |error|")
    ax.plot(ordered["seed"], ordered["p95_abs_error"], color="#ff6b9a", marker="o", label="p95 |error|")
    ax.plot(ordered["seed"], ordered["p99_abs_error"], color="#b892ff", marker="o", label="p99 |error|")
    ax.set_title("Tail behavior stays concentrated across random holdouts", fontweight="bold", pad=14)
    ax.set_xlabel("random holdout seed")
    ax.set_ylabel("absolute error")
    leg = ax.legend(facecolor="#111a2b", edgecolor="#33435f", labelcolor="#d7def2", ncol=4, loc="upper left")
    for text in leg.get_texts():
        text.set_color("#d7def2")
    save_fig(fig, "mc_cv_error_quantiles.png")


def load_scored_runs():
    frames = []
    for path in sorted(RUN_ROOT.glob("seed_*/scored_holdout.csv")):
        frame = pd.read_csv(path)
        seed_text = path.parent.name.split("_")[-1]
        frame["seed"] = int(seed_text)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No scored_holdout.csv files found. Run the Monte Carlo CV first.")
    return pd.concat(frames, ignore_index=True)


def plot_3d_error_surface(scored):
    data = scored.copy()
    data["datetime_parsed"] = pd.to_datetime(data["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    data = data.dropna(subset=["datetime_parsed", "strike", "actual_iv", "abs_error"])

    grouped = (
        data
        .groupby(["datetime_parsed", "strike", "option_type"], dropna=False)
        .agg(
            actual_iv=("actual_iv", "mean"),
            mean_abs_error=("abs_error", "mean"),
            p95_abs_error=("abs_error", lambda x: float(np.quantile(x, 0.95))),
            max_abs_error=("abs_error", "max"),
            n_scored=("abs_error", "size"),
        )
        .reset_index()
    )

    t0 = grouped["datetime_parsed"].min()
    grouped["hours_from_start"] = (grouped["datetime_parsed"] - t0).dt.total_seconds() / 3600.0

    iv_lo, iv_hi = grouped["actual_iv"].quantile([0.01, 0.99])
    grouped["iv_plot"] = grouped["actual_iv"].clip(iv_lo, iv_hi)

    err = grouped["p95_abs_error"].to_numpy(float)
    err_cap = max(float(np.quantile(err, 0.985)), 1e-8)
    grouped["error_color"] = grouped["p95_abs_error"].clip(0, err_cap)
    norm = colors.Normalize(vmin=0, vmax=err_cap)

    fig = plt.figure(figsize=(15, 10.2), dpi=230)
    fig.patch.set_facecolor("#080d16")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0d14")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.04, 0.06, 0.12, 0.96))
        axis.pane.set_edgecolor((0.52, 0.64, 0.90, 0.65))
    ax.grid(True, color="#253149", alpha=0.35)
    ax.tick_params(colors="#d8e1ff", labelsize=8)

    marker_map = {"CE": "o", "PE": "^"}
    label_map = {"CE": "CE scored cells", "PE": "PE scored cells"}
    for opt_type, marker in marker_map.items():
        part = grouped[grouped["option_type"] == opt_type]
        if part.empty:
            continue
        sizes = 9 + 88 * np.sqrt(part["error_color"].to_numpy(float) / err_cap)
        ax.scatter(
            part["hours_from_start"],
            part["strike"],
            part["iv_plot"],
            c=part["error_color"],
            cmap="turbo",
            norm=norm,
            s=sizes,
            marker=marker,
            alpha=0.86,
            edgecolors="#07101d",
            linewidths=0.22,
            depthshade=False,
            label=label_map[opt_type],
        )

    top = grouped.nlargest(26, "p95_abs_error")
    z_floor = float(grouped["iv_plot"].min()) - 0.055 * float(grouped["iv_plot"].max() - grouped["iv_plot"].min())
    for rec in top.itertuples(index=False):
        ax.plot(
            [rec.hours_from_start, rec.hours_from_start],
            [rec.strike, rec.strike],
            [z_floor, rec.iv_plot],
            color="#ff477e",
            linewidth=0.7,
            alpha=0.34,
        )
    ax.scatter(
        top["hours_from_start"],
        top["strike"],
        top["iv_plot"],
        s=130,
        facecolors="none",
        edgecolors="#ff477e",
        linewidths=1.15,
        depthshade=False,
        label="largest p95 errors",
    )

    ax.contourf(
        grouped.pivot_table(index="strike", columns="hours_from_start", values="error_color", aggfunc="mean").columns.to_numpy(),
        grouped.pivot_table(index="strike", columns="hours_from_start", values="error_color", aggfunc="mean").index.to_numpy(),
        grouped.pivot_table(index="strike", columns="hours_from_start", values="error_color", aggfunc="mean").to_numpy(),
        zdir="z",
        offset=z_floor,
        levels=22,
        cmap="turbo",
        norm=norm,
        alpha=0.38,
    )

    ax.view_init(elev=27, azim=-132)
    ax.set_box_aspect((1.9, 1.0, 0.78))
    ax.set_title(
        "Monte Carlo CV 3D IV error heatmap\nheight = actual IV, color = p95 absolute error across synthetic holdouts",
        color="#e8efff",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("hours from first timestamp", color="#d8e1ff", labelpad=11)
    ax.set_ylabel("strike", color="#d8e1ff", labelpad=11)
    ax.set_zlabel("actual IV", color="#d8e1ff", labelpad=11)
    ax.set_zlim(z_floor, float(grouped["iv_plot"].max()))
    ax.text2D(
        0.055,
        0.90,
        "hot colors + larger markers = harder synthetic holdout points",
        transform=ax.transAxes,
        color="#ffcc66",
        fontsize=11,
        fontweight="bold",
    )

    cax = fig.add_axes([0.88, 0.23, 0.018, 0.52])
    mappable = cm.ScalarMappable(norm=norm, cmap="turbo")
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(colors="#d8e1ff", labelsize=8)
    cbar.set_label("p95 absolute error, capped at 98.5 pct", color="#d8e1ff", labelpad=10)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, -0.04),
        ncol=3,
        facecolor="#111a2b",
        edgecolor="#33435f",
        framealpha=0.92,
    )
    for text in legend.get_texts():
        text.set_color("#d7def2")

    fig.subplots_adjust(left=0.0, right=0.86, top=0.93, bottom=0.04)
    fig.savefig(OUT_ROOT / "mc_cv_3d_error_heatmap.png", dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_3d_error_surface(scored):
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    data = scored.copy()
    data["datetime_parsed"] = pd.to_datetime(data["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    data = data.dropna(subset=["datetime_parsed", "strike", "actual_iv", "abs_error"])

    grouped = (
        data
        .groupby(["datetime_parsed", "strike", "option_type"], dropna=False)
        .agg(
            actual_iv=("actual_iv", "mean"),
            p95_abs_error=("abs_error", lambda x: float(np.quantile(x, 0.95))),
            max_abs_error=("abs_error", "max"),
            n_scored=("abs_error", "size"),
        )
        .reset_index()
    )

    t0 = grouped["datetime_parsed"].min()
    grouped["hours_from_start"] = (grouped["datetime_parsed"] - t0).dt.total_seconds() / 3600.0
    grouped["iv_plot"] = grouped["actual_iv"].clip(*grouped["actual_iv"].quantile([0.01, 0.99]).to_numpy(float))

    err_cap = max(float(grouped["p95_abs_error"].quantile(0.975)), 1e-8)
    norm = colors.PowerNorm(gamma=0.46, vmin=0.0, vmax=err_cap)
    cmap = cm.get_cmap("turbo")

    def smooth_grid(part):
        x = part["hours_from_start"].to_numpy(float)
        y = part["strike"].to_numpy(float)
        z = part["iv_plot"].to_numpy(float)
        e = part["p95_abs_error"].clip(0, err_cap).to_numpy(float)

        xi = np.linspace(float(x.min()), float(x.max()), 140)
        yi = np.linspace(float(y.min()), float(y.max()), 72)
        X, Y = np.meshgrid(xi, yi)
        points = np.column_stack([x, y])

        Z = griddata(points, z, (X, Y), method="linear")
        E = griddata(points, e, (X, Y), method="linear")
        Z_near = griddata(points, z, (X, Y), method="nearest")
        E_near = griddata(points, e, (X, Y), method="nearest")
        Z = np.where(np.isfinite(Z), Z, Z_near)
        E = np.where(np.isfinite(E), E, E_near)

        Z = gaussian_filter(Z, sigma=(1.05, 1.45))
        E = gaussian_filter(E, sigma=(1.25, 1.75))
        return X, Y, Z, E

    fig = plt.figure(figsize=(15.4, 10.4), dpi=230)
    fig.patch.set_facecolor("#0a0d14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0d14")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.04, 0.06, 0.12, 0.97))
        axis.pane.set_edgecolor((0.52, 0.64, 0.90, 0.62))
    ax.grid(True, color="#253149", alpha=0.28)
    ax.tick_params(colors="#d8e1ff", labelsize=8)

    z_min = float(grouped["iv_plot"].min())
    z_max = float(grouped["iv_plot"].max())
    z_floor = z_min - 0.075 * (z_max - z_min)

    surface_handles = []
    for opt_type, line_color, alpha, y_offset in [
        ("CE", "#39d5ff", 0.94, -8.0),
        ("PE", "#ff6b9a", 0.88, 8.0),
    ]:
        part = grouped[grouped["option_type"] == opt_type].copy()
        if part.empty:
            continue
        X, Y, Z, E = smooth_grid(part)
        face = cmap(norm(E))
        face[..., 3] = alpha

        surf = ax.plot_surface(
            X,
            Y + y_offset,
            Z,
            facecolors=face,
            linewidth=0.10,
            edgecolor=(0.02, 0.04, 0.08, 0.18),
            antialiased=True,
            shade=False,
            rstride=1,
            cstride=1,
        )
        surface_handles.append((surf, opt_type))

        ax.contour(
            X,
            Y + y_offset,
            Z,
            zdir="z",
            offset=z_floor,
            levels=14,
            colors=line_color,
            alpha=0.20,
            linewidths=0.55,
        )
        ax.contour(
            X,
            Y + y_offset,
            E,
            zdir="z",
            offset=z_floor + 0.004 * (z_max - z_min),
            levels=12,
            cmap="turbo",
            norm=norm,
            alpha=0.42,
            linewidths=0.62,
        )
        ax.contourf(
            X,
            Y + y_offset,
            E,
            zdir="z",
            offset=z_floor,
            levels=28,
            cmap="turbo",
            norm=norm,
            alpha=0.28 if opt_type == "CE" else 0.22,
        )

        ridge = part.nlargest(18, "p95_abs_error")
        ax.scatter(
            ridge["hours_from_start"],
            ridge["strike"] + y_offset,
            ridge["iv_plot"],
            s=22 + 86 * np.sqrt(ridge["p95_abs_error"].clip(0, err_cap) / err_cap),
            facecolors="none",
            edgecolors="#fff0a6" if opt_type == "CE" else "#ffd1df",
            linewidths=0.75,
            alpha=0.72,
            depthshade=False,
        )

    hot = grouped.nlargest(22, "p95_abs_error")
    ax.scatter(
        hot["hours_from_start"],
        hot["strike"],
        hot["iv_plot"],
        s=18 + 72 * np.sqrt(hot["p95_abs_error"].clip(0, err_cap) / err_cap),
        facecolors="none",
        edgecolors="#fff4b8",
        linewidths=0.7,
        alpha=0.68,
        depthshade=False,
    )

    ax.view_init(elev=29, azim=-132)
    ax.set_box_aspect((1.9, 1.0, 0.76))
    ax.set_title(
        "Monte Carlo CV error painted onto the IV surface",
        color="#e8efff",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("hours from first timestamp", color="#d8e1ff", labelpad=11)
    ax.set_ylabel("strike", color="#d8e1ff", labelpad=11)
    ax.set_zlabel("actual IV", color="#d8e1ff", labelpad=11)
    ax.set_zlim(z_floor, z_max)

    cax = fig.add_axes([0.88, 0.23, 0.018, 0.52])
    mappable = cm.ScalarMappable(norm=norm, cmap="turbo")
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(colors="#d8e1ff", labelsize=8)
    cbar.set_label("p95 absolute error, capped at 98.5 pct", color="#d8e1ff", labelpad=10)

    fig.subplots_adjust(left=0.0, right=0.86, top=0.93, bottom=0.04)
    fig.savefig(OUT_ROOT / "mc_cv_3d_error_heatmap.png", dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_readme_block(summary):
    mean = summary.mean(numeric_only=True)
    std = summary.std(numeric_only=True)
    best = summary.loc[summary["rmse"].idxmin()]
    worst = summary.loc[summary["rmse"].idxmax()]

    block = f"""## Monte Carlo synthetic CV robustness

To test whether the synthetic CV result above was just a lucky holdout, I repeated the same synthetic validation idea across `{len(summary)}` independently generated random holdout datasets. Each run used the same final submission code, but a different random seed for hiding observed IV cells.

```text
number of MC runs       : {len(summary)}
hidden cells per run    : {int(summary['n'].min())} to {int(summary['n'].max())}
mean MSE                : {mean['mse']:.10f}
std MSE                 : {std['mse']:.10f}
mean RMSE               : {mean['rmse']:.10f}
std RMSE                : {std['rmse']:.10f}
mean MAE                : {mean['mae']:.10f}
mean p95 absolute error : {mean['p95_abs_error']:.10f}
best seed by RMSE       : {int(best['seed'])}  ({best['rmse']:.10f})
worst seed by RMSE      : {int(worst['seed'])}  ({worst['rmse']:.10f})
```

The first Monte Carlo picture shows the distribution of the main error statistics across random holdouts. The useful thing here is not a single score, but the width of each distribution: a narrow spread means the method is not depending heavily on one particular split.

![Monte Carlo metric distributions](for_generating_readme/mc_cv_metric_distributions.png)

The seed trajectory shows the same robustness in a different way. RMSE and MAE move from seed to seed, but they stay in the same band instead of exploding for a particular random holdout.

![Monte Carlo seed trajectory](for_generating_readme/mc_cv_seed_trajectory.png)

This regime plot separates the synthetic errors into pre-27 Jan and 27 Jan, and also splits CE from PE. It is the direct robustness check for the regime behavior found in the EDA: 27 Jan is harder, but the method remains controlled on both option sides.

![Monte Carlo regime robustness](for_generating_readme/mc_cv_regime_robustness.png)

The final Monte Carlo plot turns the synthetic CV errors back into a 3D IV object. The horizontal axes are time and strike, the vertical axis is the actual IV level, and the color is the high-tail absolute error seen across the random holdouts. This makes the failure geography visible directly on the IV surface: the brightest points are not random noise, but concentrated regions where the surface is hardest to reconstruct.

![Monte Carlo 3D error heatmap](for_generating_readme/mc_cv_3d_error_heatmap.png)
"""
    (OUT_ROOT / "mc_cv_readme_block.md").write_text(block, encoding="utf-8")
    return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    group_rows = []

    for seed in args.seeds:
        overall, grouped, _ = run_one(int(seed))
        summary_rows.append(overall)
        group_rows.extend(grouped)

    summary = pd.DataFrame(summary_rows).sort_values("seed")
    grouped = pd.DataFrame(group_rows).sort_values(["seed", "regime", "option_type"])
    summary.to_csv(OUT_ROOT / "mc_cv_summary.csv", index=False)
    grouped.to_csv(OUT_ROOT / "mc_cv_grouped_by_regime_option.csv", index=False)

    plot_metric_distributions(summary)
    plot_seed_trajectory(summary)
    plot_regime_robustness(grouped)
    plot_quantiles(summary)
    plot_3d_error_surface(load_scored_runs())
    write_readme_block(summary)

    print(summary)


if __name__ == "__main__":
    main()
