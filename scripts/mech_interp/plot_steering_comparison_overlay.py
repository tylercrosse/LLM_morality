#!/usr/bin/env python3
"""
Create overlay plots comparing steering effectiveness across all layers.

Generates plots with all steering sweep curves on the same axes to enable
easy visual comparison of which layers are most steerable.

Usage:
    python scripts/mech_interp/plot_steering_comparison_overlay.py

Outputs:
    - mech_interp_outputs/causal_routing/steering_comparison_all_layers_*.png
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from mech_interp.utils import MODEL_LABELS


def load_all_steering_sweeps(output_dir: Path) -> Dict:
    """
    Load all steering sweep CSV files.

    Returns:
        Dict with structure: {model_id: {layer_component: sweep_df}}
    """
    all_sweeps = {}

    # Find all steering sweep CSV files (not matrix files)
    sweep_files = sorted(output_dir.glob("steering_sweep_*.csv"))
    sweep_files = [f for f in sweep_files if "matrix" not in f.name]

    for sweep_file in sweep_files:
        # Parse filename: steering_sweep_PT2_COREDe_L16_mlp.csv
        # Format: steering_sweep_{model_id}_{layer}_{component}.csv
        parts = sweep_file.stem.split("_")

        # Extract model_id (e.g., PT2_COREDe from parts[2] and parts[3])
        model_id = f"{parts[2]}_{parts[3]}"

        # Extract layer and component (e.g., L16_mlp from parts[4] and parts[5])
        layer_component = f"{parts[4]}_{parts[5]}"

        # Load data
        df = pd.read_csv(sweep_file)

        # Store
        if model_id not in all_sweeps:
            all_sweeps[model_id] = {}

        all_sweeps[model_id][layer_component] = df

    return all_sweeps


def plot_all_layers_overlay(
    all_sweeps: Dict,
    model_id: str,
    output_dir: Path,
    metric: str = "mean_coop_rate",
):
    """
    Plot all steering sweeps for a single model on one plot.

    Args:
        all_sweeps: Dict of all sweep data
        model_id: Model to plot
        output_dir: Output directory
        metric: Metric to plot ("mean_coop_rate" or "mean_p_action2")
    """
    if model_id not in all_sweeps:
        print(f"⚠ No data found for model {model_id}")
        return

    sweeps = all_sweeps[model_id]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors and markers for different layers
    layer_styles = {
        "L2_mlp": {"color": "gray", "marker": "x", "alpha": 0.6, "label": "L2 MLP (baseline)"},
        "L8_mlp": {"color": "#E74C3C", "marker": "o", "alpha": 0.9, "label": "L8 MLP"},
        "L9_mlp": {"color": "#3498DB", "marker": "s", "alpha": 0.9, "label": "L9 MLP"},
        "L11_mlp": {"color": "#9B59B6", "marker": "^", "alpha": 0.8, "label": "L11 MLP"},
        "L16_mlp": {"color": "#2ECC71", "marker": "D", "alpha": 0.9, "label": "L16 MLP"},
        "L17_mlp": {"color": "#1ABC9C", "marker": "v", "alpha": 0.9, "label": "L17 MLP"},
        "L19_attn": {"color": "#F39C12", "marker": "p", "alpha": 0.9, "label": "L19 Attention"},
    }

    # Sort layers for consistent ordering
    sorted_layers = sorted(sweeps.keys(), key=lambda x: (
        int(x.split("_")[0][1:]),  # Layer number
        x.split("_")[1]  # Component (attn before mlp)
    ))

    # Plot each layer
    for layer_component in sorted_layers:
        df = sweeps[layer_component]

        style = layer_styles.get(layer_component, {
            "color": "black",
            "marker": "o",
            "alpha": 0.7,
            "label": layer_component
        })

        ax.plot(
            df['strength'],
            df[metric],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linewidth=2.5,
            alpha=style["alpha"],
        )

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='Chance (50%)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='No steering')

    # Labels and formatting
    ax.set_xlabel('Steering Strength', fontsize=14, fontweight='bold')

    if metric == "mean_coop_rate":
        ax.set_ylabel('Cooperation Rate', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        title = f'Steering Effectiveness by Layer: {MODEL_LABELS.get(model_id, model_id)}'
    else:
        ax.set_ylabel('p(action2) - Defection Probability', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        title = f'Defection Probability vs Steering: {MODEL_LABELS.get(model_id, model_id)}'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    # Add text annotation
    if metric == "mean_coop_rate":
        ax.text(
            0.02, 0.02,
            'Higher is more cooperative\nPositive steering → toward moral behavior',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.tight_layout()

    # Save
    filename = f"steering_comparison_all_layers_{model_id}_{metric}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_dual_panel_comparison(
    all_sweeps: Dict,
    output_dir: Path,
):
    """
    Plot Strategic and Deontological models side-by-side.

    Args:
        all_sweeps: Dict of all sweep data
        output_dir: Output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Define layer styles (consistent across both panels)
    layer_styles = {
        "L2_mlp": {"color": "gray", "marker": "x", "alpha": 0.6, "linestyle": "--"},
        "L8_mlp": {"color": "#E74C3C", "marker": "o", "alpha": 0.9, "linestyle": "-"},
        "L9_mlp": {"color": "#3498DB", "marker": "s", "alpha": 0.9, "linestyle": "-"},
        "L11_mlp": {"color": "#9B59B6", "marker": "^", "alpha": 0.8, "linestyle": "-"},
        "L16_mlp": {"color": "#2ECC71", "marker": "D", "alpha": 0.9, "linestyle": "-"},
        "L17_mlp": {"color": "#1ABC9C", "marker": "v", "alpha": 0.9, "linestyle": "-"},
        "L19_attn": {"color": "#F39C12", "marker": "p", "alpha": 0.9, "linestyle": "-"},
    }

    models = ["PT2_COREDe", "PT3_COREDe"]
    axes = [ax1, ax2]

    for model_id, ax in zip(models, axes):
        if model_id not in all_sweeps:
            continue

        sweeps = all_sweeps[model_id]

        # Sort layers
        sorted_layers = sorted(sweeps.keys(), key=lambda x: (
            int(x.split("_")[0][1:]),
            x.split("_")[1]
        ))

        # Plot each layer
        for layer_component in sorted_layers:
            df = sweeps[layer_component]

            style = layer_styles.get(layer_component, {
                "color": "black",
                "marker": "o",
                "alpha": 0.7,
                "linestyle": "-"
            })

            # Create label (only show for first panel to avoid duplicate legend)
            label = layer_component.replace("_", " ").upper() if ax == ax1 else None

            ax.plot(
                df['strength'],
                df['mean_coop_rate'],
                label=label,
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=2.5,
                alpha=style["alpha"],
                linestyle=style["linestyle"],
            )

        # Reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=1)

        # Labels
        ax.set_xlabel('Steering Strength', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cooperation Rate', fontsize=13, fontweight='bold')
        ax.set_title(MODEL_LABELS.get(model_id, model_id), fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

    # Add legend to first panel only
    ax1.legend(fontsize=10, loc='best', framealpha=0.95, ncol=1)

    # Overall title
    fig.suptitle('Steering Effectiveness Comparison: All Layers and Models',
                 fontsize=17, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    filename = "steering_comparison_dual_panel.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def create_effectiveness_ranking(
    all_sweeps: Dict,
    output_dir: Path,
):
    """
    Create a ranking table showing steering effectiveness by layer.

    Args:
        all_sweeps: Dict of all sweep data
        output_dir: Output directory
    """
    results = []

    for model_id, sweeps in all_sweeps.items():
        for layer_component, df in sweeps.items():
            # Get baseline cooperation (strength=0)
            baseline_idx = df[df['strength'] == 0.0].index[0]
            baseline_coop = df.loc[baseline_idx, 'mean_coop_rate']

            # Get max positive steering cooperation (strength=+2.0)
            max_pos_idx = df[df['strength'] == 2.0].index[0]
            max_pos_coop = df.loc[max_pos_idx, 'mean_coop_rate']

            # Compute delta
            delta_coop = max_pos_coop - baseline_coop

            # Check monotonicity (for positive strengths)
            positive_df = df[df['strength'] >= 0.0].sort_values('strength')
            coop_values = positive_df['mean_coop_rate'].values
            is_monotonic = all(coop_values[i] <= coop_values[i+1]
                             for i in range(len(coop_values)-1))

            results.append({
                'model': model_id,
                'layer_component': layer_component,
                'layer': int(layer_component.split("_")[0][1:]),
                'component': layer_component.split("_")[1],
                'baseline_coop': baseline_coop,
                'steered_coop': max_pos_coop,
                'delta_coop': delta_coop,
                'delta_pct': delta_coop * 100,
                'is_monotonic': is_monotonic,
            })

    # Create DataFrame and sort by effectiveness
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('delta_pct', ascending=False)

    # Save to CSV
    csv_path = output_dir / "steering_effectiveness_ranking.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved effectiveness ranking: {csv_path.name}")

    # Print top results
    print("\n" + "=" * 80)
    print("STEERING EFFECTIVENESS RANKING (by Δ cooperation at strength +2.0)")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Layer':<15} {'Model':<15} {'Baseline':<10} {'Steered':<10} {'Δ Coop':<10} {'Monotonic':<10}")
    print("-" * 80)

    for rank, (idx, row) in enumerate(df_results.head(10).iterrows(), 1):
        print(f"{rank:<5} {row['layer_component']:<15} {row['model']:<15} "
              f"{row['baseline_coop']:.2%}    {row['steered_coop']:.2%}    "
              f"{row['delta_pct']:+6.1f}%    {'✓' if row['is_monotonic'] else '✗'}")

    print("\n" + "=" * 80)


def main():
    """Generate overlay comparison plots."""

    output_dir = project_root / "mech_interp_outputs" / "causal_routing"

    print("=" * 80)
    print("STEERING COMPARISON OVERLAY PLOTS")
    print("=" * 80)
    print("\nGenerating overlay plots with all layers on same axes")
    print(f"Output directory: {output_dir}\n")

    # Load all steering sweep data
    print("Loading steering sweep data...")
    all_sweeps = load_all_steering_sweeps(output_dir)

    if not all_sweeps:
        print("⚠ No steering sweep data found!")
        print("  Run comprehensive experiments first:")
        print("    python scripts/mech_interp/run_activation_steering_comprehensive.py")
        return

    print(f"✓ Loaded {sum(len(sweeps) for sweeps in all_sweeps.values())} steering sweeps")
    print(f"  Models: {', '.join(all_sweeps.keys())}")
    print()

    # Generate plots for each model
    print("-" * 80)
    print("Generating individual model plots...")
    print("-" * 80)

    for model_id in all_sweeps.keys():
        print(f"\n{model_id}:")
        plot_all_layers_overlay(all_sweeps, model_id, output_dir, metric="mean_coop_rate")
        plot_all_layers_overlay(all_sweeps, model_id, output_dir, metric="mean_p_action2")

    # Generate dual-panel comparison
    print("\n" + "-" * 80)
    print("Generating dual-panel comparison...")
    print("-" * 80 + "\n")
    plot_dual_panel_comparison(all_sweeps, output_dir)

    # Create effectiveness ranking
    create_effectiveness_ranking(all_sweeps, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("STEERING COMPARISON PLOTS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - steering_comparison_all_layers_PT2_COREDe_mean_coop_rate.png")
    print(f"  - steering_comparison_all_layers_PT3_COREDe_mean_coop_rate.png")
    print(f"  - steering_comparison_all_layers_PT2_COREDe_mean_p_action2.png")
    print(f"  - steering_comparison_all_layers_PT3_COREDe_mean_p_action2.png")
    print(f"  - steering_comparison_dual_panel.png  ← MAIN COMPARISON")
    print(f"  - steering_effectiveness_ranking.csv")
    print(f"\nKey Insights:")
    print(f"  - Overlay plots make it easy to compare layer effectiveness")
    print(f"  - Steeper curves = more effective steering")
    print(f"  - Monotonic curves = consistent steering control")
    print(f"  - Ranking table shows which layers work best")
    print("=" * 80)


if __name__ == "__main__":
    main()
