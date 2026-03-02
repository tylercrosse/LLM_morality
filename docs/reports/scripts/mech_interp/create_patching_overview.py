#!/usr/bin/env python3
"""Create bird's-eye-view figures for activation patching outputs."""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_component(component: str) -> tuple[int, str]:
    """Return (layer, component_type) where type in {'head','mlp'}."""
    if component.endswith("_MLP"):
        layer = int(component.split("_")[0][1:])
        return layer, "mlp"
    m = re.match(r"L(\d+)H\d+$", component)
    if m:
        return int(m.group(1)), "head"
    raise ValueError(f"Unrecognized component name: {component}")


def load_patch_results(patching_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(patching_dir / "patch_results_*.csv")))
    if not files:
        raise FileNotFoundError(f"No patch_results_*.csv found in {patching_dir}")
    frames = []
    for f in files:
        exp = os.path.basename(f).replace("patch_results_", "").replace(".csv", "")
        df = pd.read_csv(f)
        df["experiment"] = exp
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_experiment_scenario_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    g = (
        df.groupby(["experiment", "scenario"])["delta_change"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .reset_index(name="mean_abs_delta_change")
    )
    pivot = g.pivot(index="experiment", columns="scenario", values="mean_abs_delta_change")

    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Patching Effect Magnitude by Experiment and Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Experiment")
    plt.tight_layout()
    plt.savefig(outdir / "overview_experiment_scenario_heatmap.png", dpi=300)
    plt.close()


def plot_flip_rates(df: pd.DataFrame, outdir: Path) -> None:
    g = (
        df.groupby("experiment")["action_flipped"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "n_flips", "count": "n_total"})
    )
    g["flip_rate"] = g["n_flips"] / g["n_total"]

    plt.figure(figsize=(8, 4))
    sns.barplot(data=g, x="experiment", y="flip_rate", color="#4C78A8")
    plt.title("Action Flip Rate by Patching Experiment")
    plt.xlabel("Experiment")
    plt.ylabel("Flip rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "overview_flip_rates.png", dpi=300)
    plt.close()


def plot_layer_component_overview(df: pd.DataFrame, outdir: Path) -> None:
    parsed = df["component"].apply(parse_component)
    df = df.copy()
    df["layer"] = parsed.apply(lambda x: x[0])
    df["component_type"] = parsed.apply(lambda x: x[1])

    g = (
        df.groupby(["layer", "component_type"])["delta_change"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .reset_index(name="mean_abs_delta_change")
    )
    pivot = g.pivot(index="layer", columns="component_type", values="mean_abs_delta_change")
    pivot = pivot.reindex(columns=["head", "mlp"])

    plt.figure(figsize=(5, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Mean |Delta Sequence Preference| by Layer and Type")
    plt.xlabel("Component type")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(outdir / "overview_layer_type_heatmap.png", dpi=300)
    plt.close()


def plot_component_recurrence(patching_dir: Path, outdir: Path) -> None:
    path = patching_dir / "component_recurrence_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="experiments_present",
        y="mean_of_mean_abs_delta",
        alpha=0.7,
        s=40,
        color="#F28E2B",
    )
    top = df.sort_values("mean_of_mean_abs_delta", ascending=False).head(12)
    for _, row in top.iterrows():
        plt.text(row["experiments_present"] + 0.02, row["mean_of_mean_abs_delta"], row["component"], fontsize=8)
    plt.title("Component Recurrence vs Effect Strength")
    plt.xlabel("Experiments present")
    plt.ylabel("Mean of mean |delta change|")
    plt.tight_layout()
    plt.savefig(outdir / "overview_component_recurrence_scatter.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patching-dir", default="mech_interp_outputs/patching")
    parser.add_argument("--output-dir", default="mech_interp_outputs/patching/overview")
    args = parser.parse_args()

    patching_dir = Path(args.patching_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_patch_results(patching_dir)
    plot_experiment_scenario_heatmap(df, outdir)
    plot_flip_rates(df, outdir)
    plot_layer_component_overview(df, outdir)
    plot_component_recurrence(patching_dir, outdir)

    print(f"Saved overview figures to: {outdir}")


if __name__ == "__main__":
    main()
