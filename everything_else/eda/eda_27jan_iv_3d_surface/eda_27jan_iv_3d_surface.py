"""
27 Jan IV 3D Surface EDA

This script creates interactive and static 3D plots of the 27 Jan IV surface.

For every observed IV value on 27 Jan, it creates a point:

    x = timestamp index on 27 Jan
    y = moneyness = strike / underlying_price
    z = IV

It plots CE and PE separately.

Outputs:
    eda_27jan_iv_3d_surface/
        iv_surface_27jan_interactive.html
        CE_3d_surface.png
        PE_3d_surface.png
        CE_points_3d.png
        PE_points_3d.png
        CE_surface_points.csv
        PE_surface_points.csv

Install requirements:
    pip install pandas numpy matplotlib scipy plotly

Run:
    python eda_27jan_iv_3d_surface.py --data dataset.csv

For CV:
    python eda_27jan_iv_3d_surface.py --data cv_split/not_dataset.csv

Notes:
    - Missing IV values are not used to build the surface.
    - The surface is only an interpolated EDA surface, not a prediction model.
    - If interpolation has holes, the script fills the holes with nearest-neighbor interpolation
      only for visualization.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import griddata
except Exception as exc:
    raise ImportError("This script requires scipy. Install with: pip install scipy") from exc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:
    raise ImportError("This script requires plotly. Install with: pip install plotly") from exc


EXPIRY_DAY = pd.Timestamp("2026-01-27").date()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 27 Jan IV 3D surface.")
    parser.add_argument("--data", type=str, default="dataset.csv", help="Path to dataset.csv or not_dataset.csv.")
    parser.add_argument("--out-dir", type=str, default="eda_27jan_iv_3d_surface", help="Output directory.")
    parser.add_argument("--grid-time", type=int, default=120, help="Grid resolution on timestamp axis.")
    parser.add_argument("--grid-moneyness", type=int, default=120, help="Grid resolution on moneyness axis.")
    parser.add_argument(
        "--method",
        type=str,
        default="linear",
        choices=["linear", "nearest", "cubic"],
        help="Surface interpolation method.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the dataset and parse datetime."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find file: {data_path.resolve()}")

    df = pd.read_csv(data_path)

    required_cols = {"datetime", "underlying_price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["datetime_parsed"] = pd.to_datetime(
        df["datetime"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if df["datetime_parsed"].isna().any():
        bad = int(df["datetime_parsed"].isna().sum())
        raise ValueError(f"{bad} datetime values could not be parsed. Expected DD-MM-YYYY HH:MM.")

    return df.sort_values("datetime_parsed").reset_index(drop=True)


def parse_option_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse option columns such as:
        NIFTY27JAN2625200CE

    into:
        column, strike, option_type, expiry
    """
    pattern = re.compile(
        r"^(?P<underlying>[A-Z]+)"
        r"(?P<expiry>\d{2}[A-Z]{3}\d{2})"
        r"(?P<strike>\d+)"
        r"(?P<option_type>CE|PE)$"
    )

    records = []

    for col in df.columns:
        if col in {"datetime", "datetime_parsed", "underlying_price"}:
            continue

        match = pattern.match(col)
        if match is None:
            continue

        item = match.groupdict()
        item["column"] = col
        item["strike"] = int(item["strike"])
        item["expiry_date"] = pd.to_datetime(item["expiry"], format="%d%b%y", errors="coerce")
        records.append(item)

    meta = pd.DataFrame(records)

    if meta.empty:
        raise ValueError("No option columns could be parsed. Check column names.")

    return meta.sort_values(["option_type", "strike", "column"]).reset_index(drop=True)


def build_27jan_surface_points(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide option-chain data into long 27 Jan surface points.

    Each observed IV becomes one row:
        timestamp_idx_27jan, datetime, option_type, strike, moneyness, iv
    """
    jan27 = df[df["datetime_parsed"].dt.date == EXPIRY_DAY].copy()

    if jan27.empty:
        raise ValueError("No rows found for 27 Jan 2026.")

    jan27 = jan27.reset_index(drop=False).rename(columns={"index": "original_row_index"})
    jan27["timestamp_idx_27jan"] = np.arange(len(jan27), dtype=float)

    records = []

    for _, row in jan27.iterrows():
        spot = row["underlying_price"]

        if pd.isna(spot) or spot <= 0:
            continue

        for _, rec in meta.iterrows():
            col = rec["column"]
            iv = row[col]

            if pd.isna(iv):
                continue

            records.append({
                "original_row_index": int(row["original_row_index"]),
                "timestamp_idx_27jan": float(row["timestamp_idx_27jan"]),
                "datetime": row["datetime"],
                "datetime_parsed": row["datetime_parsed"],
                "underlying_price": float(spot),
                "contract": col,
                "option_type": rec["option_type"],
                "strike": int(rec["strike"]),
                "moneyness": float(rec["strike"] / spot),
                "iv": float(iv),
            })

    points = pd.DataFrame(records)

    if points.empty:
        raise ValueError("No observed IV points found on 27 Jan.")

    return points


def make_interpolated_grid(points_side: pd.DataFrame, grid_time: int, grid_moneyness: int, method: str):
    """
    Create an interpolated surface grid for one option type.

    Returns:
        T_grid, M_grid, IV_grid
    """
    if points_side.empty:
        return None, None, None

    t = points_side["timestamp_idx_27jan"].to_numpy(dtype=float)
    m = points_side["moneyness"].to_numpy(dtype=float)
    iv = points_side["iv"].to_numpy(dtype=float)

    t_grid = np.linspace(t.min(), t.max(), grid_time)
    m_grid = np.linspace(m.min(), m.max(), grid_moneyness)

    T, M = np.meshgrid(t_grid, m_grid)

    IV = griddata(
        points=(t, m),
        values=iv,
        xi=(T, M),
        method=method,
    )

    # For linear/cubic interpolation, points outside the convex hull become NaN.
    # Fill those holes using nearest interpolation so the visual surface is complete.
    if np.isnan(IV).any() and method != "nearest":
        IV_nearest = griddata(
            points=(t, m),
            values=iv,
            xi=(T, M),
            method="nearest",
        )
        IV = np.where(np.isnan(IV), IV_nearest, IV)

    return T, M, IV


def save_static_3d_surface(points_side: pd.DataFrame, opt_type: str, out_dir: Path, grid_time: int, grid_moneyness: int, method: str):
    """Save a Matplotlib 3D interpolated surface plot."""
    T, M, IV = make_interpolated_grid(points_side, grid_time, grid_moneyness, method)

    if T is None:
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        T,
        M,
        IV,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
    )

    ax.scatter(
        points_side["timestamp_idx_27jan"],
        points_side["moneyness"],
        points_side["iv"],
        s=14,
        alpha=0.65,
    )

    ax.set_xlabel("27 Jan Timestamp Index")
    ax.set_ylabel("Moneyness = strike / underlying")
    ax.set_zlabel("Observed IV")
    ax.set_title(f"{opt_type} IV Surface on 27 Jan")

    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label="Interpolated IV")
    fig.tight_layout()
    fig.savefig(out_dir / f"{opt_type}_3d_surface.png", dpi=170)
    plt.close(fig)


def save_static_3d_points(points_side: pd.DataFrame, opt_type: str, out_dir: Path):
    """Save a Matplotlib 3D scatter-only plot of observed IV points."""
    if points_side.empty:
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points_side["timestamp_idx_27jan"],
        points_side["moneyness"],
        points_side["iv"],
        c=points_side["iv"],
        s=22,
        alpha=0.85,
    )

    ax.set_xlabel("27 Jan Timestamp Index")
    ax.set_ylabel("Moneyness = strike / underlying")
    ax.set_zlabel("Observed IV")
    ax.set_title(f"{opt_type} Observed IV Points on 27 Jan")

    fig.colorbar(scatter, ax=ax, shrink=0.55, aspect=12, label="Observed IV")
    fig.tight_layout()
    fig.savefig(out_dir / f"{opt_type}_points_3d.png", dpi=170)
    plt.close(fig)


def create_interactive_plotly(points: pd.DataFrame, out_dir: Path, grid_time: int, grid_moneyness: int, method: str):
    """
    Save one interactive HTML file with CE and PE 3D surface plots.

    Each subplot includes:
        - interpolated surface
        - observed IV scatter points
    """
    option_types = ["CE", "PE"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("CE IV Surface", "PE IV Surface"),
        horizontal_spacing=0.03,
    )

    for col_idx, opt_type in enumerate(option_types, start=1):
        sub = points[points["option_type"] == opt_type].copy()

        if sub.empty:
            continue

        T, M, IV = make_interpolated_grid(sub, grid_time, grid_moneyness, method)

        fig.add_trace(
            go.Surface(
                x=T,
                y=M,
                z=IV,
                colorscale="Viridis",
                opacity=0.80,
                name=f"{opt_type} interpolated surface",
                showscale=(col_idx == 2),
                colorbar=dict(title="IV") if col_idx == 2 else None,
            ),
            row=1,
            col=col_idx,
        )

        fig.add_trace(
            go.Scatter3d(
                x=sub["timestamp_idx_27jan"],
                y=sub["moneyness"],
                z=sub["iv"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=sub["iv"],
                    colorscale="Viridis",
                    opacity=0.95,
                ),
                text=[
                    f"{dt}<br>{contract}<br>m={m:.5f}<br>IV={iv:.6f}"
                    for dt, contract, m, iv in zip(
                        sub["datetime"],
                        sub["contract"],
                        sub["moneyness"],
                        sub["iv"],
                    )
                ],
                hoverinfo="text",
                name=f"{opt_type} observed points",
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title="27 Jan Intraday IV Surface: IV = f(timestamp index, moneyness)",
        width=1500,
        height=750,
        scene=dict(
            xaxis_title="Timestamp Index",
            yaxis_title="Moneyness",
            zaxis_title="IV",
        ),
        scene2=dict(
            xaxis_title="Timestamp Index",
            yaxis_title="Moneyness",
            zaxis_title="IV",
        ),
        legend=dict(x=0.01, y=0.99),
    )

    fig.write_html(out_dir / "iv_surface_27jan_interactive.html")


def main():
    args = parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    meta = parse_option_metadata(df)
    points = build_27jan_surface_points(df, meta)

    print("Loaded:", data_path.resolve())
    print("Rows in dataset:", len(df))
    print("Parsed option columns:", len(meta))
    print("Observed 27 Jan surface points:", len(points))
    print("27 Jan timestamps:", points["timestamp_idx_27jan"].nunique())
    print("Option types:", sorted(points["option_type"].unique()))
    print("Output directory:", out_dir.resolve())

    for opt_type in ["CE", "PE"]:
        sub = points[points["option_type"] == opt_type].copy()

        if sub.empty:
            print(f"Skipping {opt_type}: no observed points.")
            continue

        sub.to_csv(out_dir / f"{opt_type}_surface_points.csv", index=False)
        save_static_3d_surface(
            points_side=sub,
            opt_type=opt_type,
            out_dir=out_dir,
            grid_time=args.grid_time,
            grid_moneyness=args.grid_moneyness,
            method=args.method,
        )
        save_static_3d_points(sub, opt_type, out_dir)

    create_interactive_plotly(
        points=points,
        out_dir=out_dir,
        grid_time=args.grid_time,
        grid_moneyness=args.grid_moneyness,
        method=args.method,
    )

    print("")
    print("Saved:")
    print("  iv_surface_27jan_interactive.html")
    print("  CE_3d_surface.png")
    print("  PE_3d_surface.png")
    print("  CE_points_3d.png")
    print("  PE_points_3d.png")
    print("  CE_surface_points.csv")
    print("  PE_surface_points.csv")


if __name__ == "__main__":
    main()
