#!/usr/bin/env python3
"""
Create cross-adapter comparison visualizations from weight norm outputs.

Usage:
    python scripts/mech_interp/visualize_lora_adapter_comparison.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mech_interp.utils import MODEL_LABELS


MODEL_ORDER = ["PT2_COREDe", "PT3_COREDe", "PT3_COREUt", "PT4_COREDe"]
PLOT_COLORS = {
    "PT2_COREDe": "#3B82F6",  # blue
    "PT3_COREDe": "#16A34A",  # green
    "PT3_COREUt": "#EA580C",  # orange
    "PT4_COREDe": "#DC2626",  # red
}


def _load_data(weight_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_models_path = weight_dir / "weight_norms_all_models.csv"
    comparison_path = weight_dir / "weight_norms_comparison.csv"

    if not all_models_path.exists():
        raise FileNotFoundError(
            f"Missing {all_models_path}. Run run_weight_analysis.py first."
        )
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing {comparison_path}. Run run_weight_analysis.py first."
        )

    df_all = pd.read_csv(all_models_path)
    df_summary = pd.read_csv(comparison_path)

    # Keep only the 4 adapters in a fixed order.
    df_all = df_all[df_all["model_id"].isin(MODEL_ORDER)].copy()
    df_summary = df_summary[df_summary["model_id"].isin(MODEL_ORDER)].copy()

    return df_all, df_summary


def _component_matrix(df_all: pd.DataFrame) -> pd.DataFrame:
    matrix = df_all.pivot_table(
        index="component_name",
        columns="model_id",
        values="frobenius_norm",
        aggfunc="mean",
    )
    return matrix[MODEL_ORDER]


def plot_adapter_similarity(
    component_matrix: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    mat = component_matrix.values

    # Cosine similarity between model vectors.
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    normalized = mat / np.clip(norms, 1e-12, None)
    cosine = normalized.T @ normalized
    cosine_df = pd.DataFrame(cosine, index=MODEL_ORDER, columns=MODEL_ORDER)

    # Human-readable labels.
    label_map = {m: MODEL_LABELS.get(m, m) for m in MODEL_ORDER}
    cosine_df = cosine_df.rename(index=label_map, columns=label_map)

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cosine_df,
        cmap="YlOrBr",
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Cosine Similarity"},
        vmin=0.90,
        vmax=1.0,
    )
    plt.title("Adapter Similarity (Component-Wise Cosine)")
    plt.tight_layout()
    out_path = output_dir / "adapter_similarity_heatmap.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    csv_path = output_dir / "adapter_similarity_cosine.csv"
    cosine_df.to_csv(csv_path)
    return cosine_df


def plot_adapter_distance(component_matrix: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    mat = component_matrix.values
    n = mat.shape[1]
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(mat[:, i] - mat[:, j])

    dist_df = pd.DataFrame(dist, index=MODEL_ORDER, columns=MODEL_ORDER)
    label_map = {m: MODEL_LABELS.get(m, m) for m in MODEL_ORDER}
    dist_df = dist_df.rename(index=label_map, columns=label_map)

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        dist_df,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "L2 Distance"},
    )
    plt.title("Adapter Distance (Raw Component Norm Vectors)")
    plt.tight_layout()
    out_path = output_dir / "adapter_distance_heatmap.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    csv_path = output_dir / "adapter_distance_l2.csv"
    dist_df.to_csv(csv_path)
    return dist_df


def plot_top_variable_components(
    component_matrix: pd.DataFrame, output_dir: Path, top_n: int = 30
):
    variability = component_matrix.std(axis=1).sort_values(ascending=False)
    top_components = variability.head(top_n).index
    sub = component_matrix.loc[top_components]

    # Z-score by component so pattern differences are easier to see.
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-12, axis=0)
    z = z.rename(columns={m: MODEL_LABELS.get(m, m) for m in MODEL_ORDER})

    plt.figure(figsize=(9, max(8, top_n * 0.3)))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.2,
        cbar_kws={"label": "Per-Component Z-Score"},
        yticklabels=True,
    )
    plt.title(f"Most Divergent Components Across 4 Adapters (Top {top_n})")
    plt.xlabel("Adapter")
    plt.ylabel("Component")
    plt.tight_layout()
    out_path = output_dir / "adapter_top_variable_components_heatmap.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_layerwise_deltas(df_all: pd.DataFrame, output_dir: Path):
    # Compare layer-wise MLP totals against Strategic adapter.
    mlp = df_all[df_all["module_type"].str.startswith("mlp.")].copy()
    layer_totals = (
        mlp.groupby(["model_id", "layer"], as_index=False)["frobenius_norm"].sum()
    )
    pivot = layer_totals.pivot(index="layer", columns="model_id", values="frobenius_norm")
    pivot = pivot[MODEL_ORDER]

    baseline = pivot["PT2_COREDe"]

    plt.figure(figsize=(10, 5.5))
    for model_id in MODEL_ORDER[1:]:
        delta = pivot[model_id] - baseline
        plt.plot(
            pivot.index,
            delta,
            label=f"{MODEL_LABELS.get(model_id, model_id)} - Strategic",
            color=PLOT_COLORS[model_id],
            linewidth=2,
        )

    plt.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Delta MLP Norm")
    plt.title("Layer-Wise MLP Delta vs Strategic Adapter")
    plt.grid(alpha=0.25, linewidth=0.7)
    plt.legend(frameon=True)
    plt.tight_layout()
    out_path = output_dir / "adapter_layerwise_mlp_delta_vs_strategic.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_summary_bars(df_summary: pd.DataFrame, output_dir: Path):
    ordered = df_summary.set_index("model_id").loc[MODEL_ORDER].reset_index()
    labels = [MODEL_LABELS.get(mid, mid) for mid in ordered["model_id"]]
    x = np.arange(len(labels))

    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        x - width / 2,
        ordered["mlp_mean_norm"],
        width,
        label="MLP Mean Norm",
        color="#0EA5E9",
    )
    ax.bar(
        x + width / 2,
        ordered["attn_mean_norm"],
        width,
        label="Attention Mean Norm",
        color="#FB7185",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mean Frobenius Norm")
    ax.set_title("MLP vs Attention Update Magnitude by Adapter")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / "adapter_mlp_vs_attn_summary.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_quick_summary(
    df_summary: pd.DataFrame,
    cosine_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    output_dir: Path,
):
    ordered = df_summary.set_index("model_id").loc[MODEL_ORDER].reset_index()

    strongest = ordered.sort_values("mean_norm", ascending=False).iloc[0]
    weakest = ordered.sort_values("mean_norm", ascending=True).iloc[0]

    similarity_flat = cosine_df.where(~np.eye(len(cosine_df), dtype=bool)).stack()
    cosine_closest_pair = similarity_flat.idxmax()
    cosine_farthest_pair = similarity_flat.idxmin()

    dist_flat = dist_df.where(~np.eye(len(dist_df), dtype=bool)).stack()
    l2_closest_pair = dist_flat.idxmin()
    l2_farthest_pair = dist_flat.idxmax()

    lines = [
        "# 4-Adapter LoRA Comparison Summary",
        "",
        "## Overall strength (mean norm)",
        *(f"- {MODEL_LABELS.get(r.model_id, r.model_id)}: {r.mean_norm:.4f}" for r in ordered.itertuples()),
        "",
        f"- Strongest adapter: {MODEL_LABELS.get(strongest.model_id, strongest.model_id)}",
        f"- Weakest adapter: {MODEL_LABELS.get(weakest.model_id, weakest.model_id)}",
        "",
        "## Similarity",
        f"- Closest by cosine (shape): {cosine_closest_pair[0]} vs {cosine_closest_pair[1]} (cosine={similarity_flat.max():.4f})",
        f"- Most distinct by cosine (shape): {cosine_farthest_pair[0]} vs {cosine_farthest_pair[1]} (cosine={similarity_flat.min():.4f})",
        f"- Closest by L2 (magnitude): {l2_closest_pair[0]} vs {l2_closest_pair[1]} (L2={dist_flat.min():.4f})",
        f"- Most distinct by L2 (magnitude): {l2_farthest_pair[0]} vs {l2_farthest_pair[1]} (L2={dist_flat.max():.4f})",
    ]

    out_path = output_dir / "adapter_comparison_summary.md"
    out_path.write_text("\n".join(lines))


def main():
    weight_dir = project_root / "mech_interp_outputs" / "weight_analysis"
    df_all, df_summary = _load_data(weight_dir)

    sns.set_theme(style="whitegrid")
    component_matrix = _component_matrix(df_all)

    cosine_df = plot_adapter_similarity(component_matrix, weight_dir)
    dist_df = plot_adapter_distance(component_matrix, weight_dir)
    plot_top_variable_components(component_matrix, weight_dir, top_n=30)
    plot_layerwise_deltas(df_all, weight_dir)
    plot_summary_bars(df_summary, weight_dir)
    save_quick_summary(df_summary, cosine_df, dist_df, weight_dir)

    print("=" * 80)
    print("4-adapter comparison complete")
    print("=" * 80)
    print(f"Output directory: {weight_dir}")
    print("Created:")
    print("  - adapter_similarity_heatmap.png")
    print("  - adapter_similarity_cosine.csv")
    print("  - adapter_distance_heatmap.png")
    print("  - adapter_distance_l2.csv")
    print("  - adapter_top_variable_components_heatmap.png")
    print("  - adapter_layerwise_mlp_delta_vs_strategic.png")
    print("  - adapter_mlp_vs_attn_summary.png")
    print("  - adapter_comparison_summary.md")


if __name__ == "__main__":
    main()
