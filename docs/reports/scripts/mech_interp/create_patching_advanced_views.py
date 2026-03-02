#!/usr/bin/env python3
"""Create additional bird's-eye visualizations for patching outputs."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    p for p in _THIS_FILE.parents if (p / "mech_interp" / "__init__.py").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.activation_patching import ActivationPatcher
from mech_interp.model_loader import LoRAModelLoader
from mech_interp.prompt_generator import load_prompt_dataset
from mech_interp.utils import get_action_token_ids


def parse_component(component: str) -> Tuple[int, str]:
    if component.endswith("_MLP"):
        layer = int(component.split("_")[0][1:])
        return layer, "mlp"
    m = re.match(r"L(\d+)H\d+$", component)
    if m:
        return int(m.group(1)), "head"
    raise ValueError(f"Unrecognized component: {component}")


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


def component_experiment_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["component", "experiment"])["delta_change"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .reset_index(name="mean_abs_delta_change")
    )
    pivot = g.pivot(index="component", columns="experiment", values="mean_abs_delta_change").fillna(0.0)
    return pivot


def plot_cross_experiment_stability(df: pd.DataFrame, outdir: Path, top_n: int) -> None:
    pivot = component_experiment_table(df)
    score = pivot.mean(axis=1).sort_values(ascending=False)
    top_components = score.head(top_n).index.tolist()
    plot_df = pivot.loc[top_components].reset_index().melt(
        id_vars="component", var_name="experiment", value_name="mean_abs_delta_change"
    )

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=plot_df,
        x="experiment",
        y="mean_abs_delta_change",
        hue="component",
        marker="o",
        linewidth=1.5,
    )
    plt.title(f"Top-{top_n} Component Stability Across Patching Experiments")
    plt.xlabel("Experiment")
    plt.ylabel("Mean |Delta Change|")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "advanced_cross_experiment_component_stability.png", dpi=300)
    plt.close()


def plot_direction_consistency(df: pd.DataFrame, outdir: Path, top_n: int) -> None:
    # Sign consistency: |mean(sign(delta_change))| in [0,1]
    g = (
        df.groupby(["component", "experiment", "scenario"])["delta_change"]
        .apply(lambda s: float(np.abs(np.mean(np.sign(s)))))
        .reset_index(name="sign_consistency")
    )
    component_rank = (
        df.groupby("component")["delta_change"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
    )
    top_components = component_rank.head(top_n).index.tolist()
    g = g[g["component"].isin(top_components)].copy()

    # Combine experiment + scenario for a compact matrix.
    g["exp_scenario"] = g["experiment"] + " | " + g["scenario"]
    pivot = g.pivot(index="component", columns="exp_scenario", values="sign_consistency").fillna(0.0)

    plt.figure(figsize=(14, max(4, int(0.45 * len(top_components)))))
    sns.heatmap(pivot, cmap="magma", vmin=0, vmax=1)
    plt.title("Direction Consistency by Component (|mean sign(delta_change)|)")
    plt.xlabel("Experiment | Scenario")
    plt.ylabel("Component")
    plt.tight_layout()
    plt.savefig(outdir / "advanced_direction_consistency_heatmap.png", dpi=300)
    plt.close()


def plot_variant_robustness(df: pd.DataFrame, outdir: Path, top_n: int) -> None:
    component_rank = (
        df.groupby("component")["delta_change"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
    )
    top_components = component_rank.head(top_n).index.tolist()
    plot_df = df[df["component"].isin(top_components)].copy()

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=plot_df,
        x="component",
        y="delta_change",
        hue="experiment",
        showfliers=False,
    )
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.title(f"Variant Robustness of Top-{top_n} Components")
    plt.xlabel("Component")
    plt.ylabel("Delta Change (patched - baseline)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "advanced_variant_robustness_boxplot.png", dpi=300)
    plt.close()


def plot_cumulative_topk(df: pd.DataFrame, outdir: Path, max_k: int) -> None:
    rows = []
    for (experiment, scenario), g in df.groupby(["experiment", "scenario"]):
        comp = (
            g.groupby("component")["delta_change"]
            .apply(lambda s: float(np.mean(np.abs(s))))
            .sort_values(ascending=False)
        )
        vals = comp.values
        total = float(np.sum(vals)) + 1e-12
        cum = np.cumsum(vals) / total
        k_values = np.arange(1, min(max_k, len(cum)) + 1)
        for k in k_values:
            rows.append(
                {
                    "experiment": experiment,
                    "scenario": scenario,
                    "k": int(k),
                    "cumulative_fraction": float(cum[k - 1]),
                }
            )
    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_df,
        x="k",
        y="cumulative_fraction",
        hue="scenario",
        style="experiment",
        linewidth=1.7,
    )
    plt.title("Cumulative Top-k Share of Total Effect Magnitude")
    plt.xlabel("Top-k components")
    plt.ylabel("Cumulative fraction of total |delta_change|")
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig(outdir / "advanced_cumulative_topk_curves.png", dpi=300)
    plt.close()


def parse_experiment(exp: str) -> Tuple[str, str]:
    if "_to_" not in exp:
        raise ValueError(f"Invalid experiment format: {exp}")
    source, target = exp.split("_to_", maxsplit=1)
    return source, target


def effective_patch_key(component: str) -> str:
    """Map component to effective patched module key.

    Head patches are implemented at layer attention-output module granularity,
    so all heads in the same layer share one effective key.
    """
    if component.endswith("_MLP"):
        layer = int(component.split("_")[0][1:])
        return f"L{layer}_MLP"
    m = re.match(r"L(\d+)H\d+$", component)
    if m:
        return f"L{int(m.group(1))}_ATTN"
    raise ValueError(f"Unrecognized component: {component}")


def compute_pairwise_non_additivity(
    df: pd.DataFrame,
    *,
    interaction_experiment: str,
    interaction_scenario: str,
    top_n: int,
    output_dir: Path,
    device: str,
) -> None:
    source_model_id, target_model_id = parse_experiment(interaction_experiment)
    subset = df[
        (df["experiment"] == interaction_experiment) & (df["scenario"] == interaction_scenario)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No rows for experiment={interaction_experiment}, scenario={interaction_scenario}"
        )

    ranked_components = (
        subset.groupby("component")["delta_change"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Deduplicate by effective patch module to avoid same-layer head aliases.
    top_components: List[str] = []
    seen_keys = set()
    for c in ranked_components:
        k = effective_patch_key(c)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        top_components.append(c)
        if len(top_components) >= top_n:
            break

    prompts = [
        p["prompt"]
        for p in load_prompt_dataset(str(PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json"))
        if p["scenario"] == interaction_scenario
    ]
    if not prompts:
        raise ValueError(f"No prompts found for scenario: {interaction_scenario}")

    source_model = LoRAModelLoader.load_hooked_model(
        source_model_id, device=device, merge_lora=False, use_4bit=True
    )
    source_model.name = source_model_id
    target_model = LoRAModelLoader.load_hooked_model(
        target_model_id, device=device, merge_lora=False, use_4bit=True
    )
    target_model.name = target_model_id
    action_tokens = get_action_token_ids(target_model.tokenizer)
    patcher = ActivationPatcher(source_model, target_model, action_tokens, device=device)

    # Compute prompt-level single effects for selected components.
    single_effects: Dict[Tuple[int, str], float] = {}
    for pi, prompt in enumerate(prompts):
        baseline_delta, _ = patcher.get_baseline_behavior(prompt)
        patcher.cache_source_activations(prompt)
        for c in top_components:
            patched_delta, _, _, _ = patcher.patch_component(prompt, c)
            single_effects[(pi, c)] = float(patched_delta - baseline_delta)

    # Compute non-additivity matrix.
    n = len(top_components)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            a = top_components[i]
            b = top_components[j]
            vals = []
            for pi, prompt in enumerate(prompts):
                baseline_delta, _ = patcher.get_baseline_behavior(prompt)
                patcher.cache_source_activations(prompt)
                joint_delta = patcher._patch_multiple_components(prompt, [a, b])
                joint_change = float(joint_delta - baseline_delta)
                additive = single_effects[(pi, a)] + single_effects[(pi, b)]
                vals.append(joint_change - additive)
            mean_non_add = float(np.mean(vals))
            matrix[i, j] = mean_non_add
            matrix[j, i] = mean_non_add

    plt.figure(figsize=(max(7, n * 0.55), max(6, n * 0.50)))
    sns.heatmap(
        matrix,
        xticklabels=top_components,
        yticklabels=top_components,
        cmap="RdBu_r",
        center=0.0,
    )
    plt.title(
        "Layer/Component Pair Non-Additivity\n"
        f"{interaction_experiment} | {interaction_scenario} | top-{top_n}"
    )
    plt.xlabel("Component B")
    plt.ylabel("Component A")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "advanced_pair_non_additivity_heatmap.png", dpi=300)
    plt.close()

    # Save matrix as CSV for writeups.
    mat_df = pd.DataFrame(matrix, index=top_components, columns=top_components)
    mat_df.to_csv(output_dir / "advanced_pair_non_additivity_matrix.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patching-dir", default="mech_interp_outputs/patching")
    parser.add_argument("--output-dir", default="mech_interp_outputs/patching/overview_plus")
    parser.add_argument("--top-n", type=int, default=12, help="Top components for component-centric plots.")
    parser.add_argument("--max-k", type=int, default=60, help="Max k for cumulative top-k curves.")
    parser.add_argument("--compute-pair-interactions", action="store_true")
    parser.add_argument("--interaction-experiment", default="PT2_COREDe_to_PT3_COREDe")
    parser.add_argument("--interaction-scenario", default="DC_exploited")
    parser.add_argument("--interaction-top-n", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    patching_dir = Path(args.patching_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_patch_results(patching_dir)
    plot_cross_experiment_stability(df, output_dir, top_n=args.top_n)
    plot_direction_consistency(df, output_dir, top_n=args.top_n)
    plot_variant_robustness(df, output_dir, top_n=args.top_n)
    plot_cumulative_topk(df, output_dir, max_k=args.max_k)

    if args.compute_pair_interactions:
        compute_pairwise_non_additivity(
            df,
            interaction_experiment=args.interaction_experiment,
            interaction_scenario=args.interaction_scenario,
            top_n=args.interaction_top_n,
            output_dir=output_dir,
            device=args.device,
        )

    print(f"Saved advanced patching views to: {output_dir}")


if __name__ == "__main__":
    main()
