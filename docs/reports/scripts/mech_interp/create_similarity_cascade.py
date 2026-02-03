#!/usr/bin/env python3
"""
Create a "similarity cascade" visualization showing where moral models differ.

This script creates a synthesis visualization showing three levels of analysis:
1. Component activations: 99.9999% similar
2. Attention patterns: 99.99% similar
3. Component interactions: Significantly different

The visualization tells the story of systematically ruling out mechanisms
until finding where differences actually emerge.

Usage:
    python docs/reports/scripts/mech_interp/create_similarity_cascade.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


def load_component_similarity():
    """
    Compute component activation similarity between De and Ut models.

    Uses Pearson correlation as the primary metric, which captures how
    similar the overall component activation patterns are.

    Returns:
        float: Percentage similarity (0-100)
    """
    dla_path = PROJECT_ROOT / "mech_interp_outputs" / "dla" / "dla_full_results.csv"
    df = pd.read_csv(dla_path)

    # Filter to De and Ut models
    de_data = df[df['model'] == 'PT3_COREDe'].copy()
    ut_data = df[df['model'] == 'PT3_COREUt'].copy()

    # Merge on component, scenario to get paired comparisons
    merged = pd.merge(
        de_data[['scenario', 'component', 'contribution']],
        ut_data[['scenario', 'component', 'contribution']],
        on=['scenario', 'component'],
        suffixes=('_de', '_ut')
    )

    # Compute Pearson correlation
    correlation = np.corrcoef(merged['contribution_de'], merged['contribution_ut'])[0, 1]

    # Convert correlation to percentage similarity
    # r=1.0 -> 100% similar, r=0 -> 50% similar, r=-1 -> 0% similar
    similarity = 50 + (correlation * 50)

    # Also compute max difference for reference
    merged['abs_diff'] = np.abs(merged['contribution_de'] - merged['contribution_ut'])
    max_diff = merged['abs_diff'].max()
    mean_abs_diff = merged['abs_diff'].mean()

    return similarity, correlation, max_diff, mean_abs_diff


def load_attention_similarity():
    """
    Compute attention pattern similarity between De and Ut models.

    Returns:
        float: Percentage similarity (0-100)
    """
    attn_path = PROJECT_ROOT / "mech_interp_outputs" / "attention_analysis" / "attention_comparison_De_vs_Ut.csv"
    df = pd.read_csv(attn_path)

    # Compute mean absolute differences across all token categories
    diff_cols = ['action_attn_diff', 'opponent_attn_diff', 'payoff_attn_diff']
    mean_abs_diff = df[[col for col in diff_cols]].abs().mean().mean()

    # Attention weights are in [0, 1] range, so differences in [0, 1]
    # Convert to similarity percentage
    similarity = 100 * (1 - mean_abs_diff)

    return similarity, mean_abs_diff


def load_interaction_differences():
    """
    Count significantly different interaction pathways.

    Returns:
        tuple: (num_significant, total_pathways, percentage_different)
    """
    interaction_path = PROJECT_ROOT / "mech_interp_outputs" / "component_interactions" / "significant_pathways_De_vs_Ut.csv"
    df = pd.read_csv(interaction_path)

    # Count pathways with |diff| > 0.3 (threshold for "significant")
    num_significant = len(df)

    # Total possible pathways: 52 components (26 ATTN + 26 MLP)
    # Unique pairs: (52 * 51) / 2 = 1,326
    total_pathways = (52 * 51) // 2

    percentage_different = 100 * (num_significant / total_pathways)

    return num_significant, total_pathways, percentage_different


def create_cascade_visualization(
    component_sim,
    attention_sim,
    interaction_pct_diff,
    output_path
):
    """
    Create the similarity cascade visualization.

    Args:
        component_sim: Component similarity percentage
        attention_sim: Attention similarity percentage
        interaction_pct_diff: Interaction difference percentage
        output_path: Path to save the figure
    """
    # Set publication-quality defaults
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    rcParams['font.size'] = 11

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data for the three levels
    levels = [
        'Component\nActivations',
        'Attention\nPatterns',
        'Component\nInteractions'
    ]
    similarities = [component_sim, attention_sim, 100 - interaction_pct_diff]
    labels = [
        f'{component_sim:.4f}% Similar',
        f'{attention_sim:.2f}% Similar',
        f'{interaction_pct_diff:.1f}% Significantly Different'
    ]
    descriptions = [
        'Same components activate with similar strengths',
        'Models attend to the same tokens in prompts',
        'Different wiring: same parts, different connections'
    ]

    # Colors: blue for similar, orange for different
    colors = []
    for sim in similarities:
        if sim > 95:
            colors.append('#2E86AB')  # Blue for high similarity
        elif sim > 80:
            colors.append('#A23B72')  # Purple for moderate
        else:
            colors.append('#F18F01')  # Orange for different

    # Reverse order so Component is at top
    levels = levels[::-1]
    similarities = similarities[::-1]
    labels = labels[::-1]
    descriptions = descriptions[::-1]
    colors = colors[::-1]

    # Create horizontal bars
    y_positions = np.arange(len(levels))
    bars = ax.barh(y_positions, similarities, color=colors, alpha=0.8, height=0.6)

    # Add reference line at 100%
    ax.axvline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Configure axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(levels, fontsize=13, fontweight='bold')
    ax.set_xlabel('Similarity (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.7, len(levels) - 0.3)

    # Add percentage labels on bars
    for i, (bar, label, desc) in enumerate(zip(bars, labels, descriptions)):
        # Label on bar
        x_pos = bar.get_width() + 1
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                label,
                va='center', ha='left',
                fontsize=11, fontweight='bold',
                color=colors[i])

        # Description below
        ax.text(52.5, bar.get_y() + bar.get_height()/2 - 0.25,
                desc,
                va='top', ha='center',
                fontsize=9, style='italic',
                color='#333333')

    # Title
    ax.set_title('Where Do Moral Models Differ?\nA Multi-Level Investigation',
                fontsize=15, fontweight='bold', pad=20)

    # Add subtle grid
    ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add interpretation box
    interpretation = (
        "Systematic investigation reveals: Models use nearly identical components and attend to the same information,\n"
        "but differ in how components are wired together—same Lego bricks, different structure."
    )
    fig.text(0.5, 0.02, interpretation,
            ha='center', fontsize=9, style='italic',
            color='#555555', wrap=True)

    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PNG: {output_path}")

    # Also save SVG
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved SVG: {svg_path}")

    plt.close()


def main():
    print("\n" + "="*60)
    print("SIMILARITY CASCADE VISUALIZATION")
    print("="*60 + "\n")

    print("Computing metrics...\n")

    # Load data
    print("1. Component activation similarity...")
    comp_sim, correlation, max_diff, mean_diff = load_component_similarity()
    print(f"   Similarity: {comp_sim:.4f}% (r={correlation:.6f})")
    print(f"   Max difference: {max_diff:.4f}")
    print(f"   Mean abs difference: {mean_diff:.4f}\n")

    print("2. Attention pattern similarity...")
    attn_sim, mean_diff = load_attention_similarity()
    print(f"   Similarity: {attn_sim:.4f}%")
    print(f"   Mean abs difference: {mean_diff:.6f}\n")

    print("3. Component interaction differences...")
    num_sig, total, pct_diff = load_interaction_differences()
    print(f"   Significant pathways: {num_sig}/{total}")
    print(f"   Percentage different: {pct_diff:.1f}%\n")

    # Create visualization
    print("Creating visualization...")
    output_path = PROJECT_ROOT / "mech_interp_outputs" / "synthesis" / "similarity_cascade.png"

    create_cascade_visualization(
        component_sim=comp_sim,
        attention_sim=attn_sim,
        interaction_pct_diff=pct_diff,
        output_path=output_path
    )

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60 + "\n")
    print(f"Output: {output_path}")
    print(f"Output: {output_path.with_suffix('.svg')}")
    print("\nNext steps:")
    print("  1. Review the visualization")
    print("  2. Add to WRITE_UP.md after attention analysis section")
    print("  3. Update PRESENTATION.md if desired\n")


if __name__ == "__main__":
    main()
