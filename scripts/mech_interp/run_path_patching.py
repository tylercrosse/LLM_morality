#!/usr/bin/env python3
"""
Run Path Patching experiments.

Tests whether information flows causally through specific pathways (e.g., L2→L9)
by replacing residual stream activations from source to target model. This goes
beyond single-component patching to test pathway-level causality.

Key experiments:
    1. Full path: L2→L9 residual stream replacement
    2. Progressive: L2→L2, L2→L3, ..., L2→L9 to find critical range
    3. MLP-only path: L2_MLP → ... → L9_MLP
    4. Attention-only path: L2_ATTN → ... → L9_ATTN
    5. Compare: Full path vs MLP-only vs ATTN-only contributions

Usage:
    python scripts/mech_interp/run_path_patching.py

Outputs:
    - mech_interp_outputs/causal_routing/path_patch_*.csv
    - mech_interp_outputs/causal_routing/progressive_patch_*.csv
    - mech_interp_outputs/causal_routing/path_patch_comparison.png
    - mech_interp_outputs/causal_routing/progressive_patch_*.png
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mech_interp.path_patching import (
    PathPatcher,
    save_path_patch_result,
    save_progressive_result,
)
from mech_interp.utils import MODEL_LABELS


def plot_progressive_patching(progressive_results, output_dir):
    """Create visualization of progressive patching results."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for result in progressive_results:
        label = f"{result.source_model_id} → {result.target_model_id} ({result.component_type})"

        # Plot 1: Cooperation rate vs end layer
        ax1.plot(result.end_layers, result.mean_coop_rate,
                marker='o', linewidth=2, markersize=6, label=label)

        # Plot 2: p(action2) vs end layer
        ax2.plot(result.end_layers, result.mean_p_action2,
                marker='o', linewidth=2, markersize=6, label=label)

    ax1.set_xlabel('End Layer', fontsize=12)
    ax1.set_ylabel('Cooperation Rate', fontsize=12)
    ax1.set_title('Progressive Path Patching: Cooperation Rate', fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xlabel('End Layer', fontsize=12)
    ax2.set_ylabel('p(action2) - Defection', fontsize=12)
    ax2.set_title('Progressive Path Patching: Defection Probability', fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()

    # Save
    plot_path = output_dir / "progressive_patch_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved progressive patching plot: {plot_path}")
    plt.close()


def plot_component_comparison(full_path_result, mlp_only_result, attn_only_result, output_dir):
    """Compare full path vs MLP-only vs attention-only contributions."""

    fig, ax = plt.subplots(figsize=(10, 6))

    results = [
        ("Full Residual", full_path_result),
        ("MLP Only", mlp_only_result),
        ("Attention Only", attn_only_result),
    ]

    labels = []
    baseline_coop = []
    patched_coop = []
    delta_coop = []

    for label, result in results:
        labels.append(label)
        baseline_coop.append(1 - result.baseline_p_action2)
        patched_coop.append(1 - result.patched_p_action2)
        delta_coop.append((1 - result.patched_p_action2) - (1 - result.baseline_p_action2))

    x_pos = np.arange(len(labels))
    width = 0.25

    ax.bar(x_pos - width, baseline_coop, width, label='Baseline', alpha=0.8, color='steelblue')
    ax.bar(x_pos, patched_coop, width, label='Patched', alpha=0.8, color='coral')
    ax.bar(x_pos + width, delta_coop, width, label='Δ Cooperation', alpha=0.8, color='green')

    ax.set_xlabel('Pathway Type', fontsize=12)
    ax.set_ylabel('Cooperation Rate / Change', fontsize=12)
    ax.set_title(f'Pathway Contribution Comparison\n{full_path_result.source_model_id} → {full_path_result.target_model_id}', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()

    # Save
    plot_path = output_dir / f"component_comparison_{full_path_result.source_model_id}_to_{full_path_result.target_model_id}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved component comparison plot: {plot_path}")
    plt.close()


def main():
    """Run path patching experiments."""

    # Output directory
    output_dir = project_root / "mech_interp_outputs" / "causal_routing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PATH PATCHING EXPERIMENTS")
    print("=" * 80)
    print("\nTesting hypothesis: L2→L9 pathway causally mediates moral behavior")
    print("Method: Replace residual stream activations from source to target")
    print("\nExperiments:")
    print("  1. Full path (L2→L9): Replace residual stream")
    print("  2. Progressive: Find critical layer range")
    print("  3. MLP-only: Isolate MLP pathway contribution")
    print("  4. Attention-only: Isolate attention pathway contribution")
    print("=" * 80)

    # Initialize patcher
    patcher = PathPatcher(device="cuda")

    # Experiment 1: Full path patching (L2→L9)
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: Full Path Patching (L2→L9)")
    print(f"{'='*80}")
    print("\nHypothesis: Replacing L2→L9 residual stream from Deontological to Strategic increases cooperation")

    result_full_De_to_St = patcher.patch_residual_path(
        source_model_id="PT3_COREDe",  # Deontological (cooperative)
        target_model_id="PT2_COREDe",  # Strategic (selfish)
        start_layer=2,
        end_layer=9,
        component_type="residual",
        scenarios=None,
    )

    save_path_patch_result(result_full_De_to_St, output_dir)

    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Baseline cooperation: {1 - result_full_De_to_St.baseline_p_action2:.2%}")
    print(f"  Patched cooperation: {1 - result_full_De_to_St.patched_p_action2:.2%}")
    print(f"  Δ cooperation: {(1 - result_full_De_to_St.patched_p_action2) - (1 - result_full_De_to_St.baseline_p_action2):+.2%}")
    print(f"{'='*80}")

    # Check hypothesis
    hypothesis_supported = result_full_De_to_St.delta_p_action2 < -0.3  # >30% cooperation increase
    print(f"\nHypothesis (>30% cooperation increase): {'✓ SUPPORTED' if hypothesis_supported else '✗ NOT SUPPORTED'}")

    # Experiment 2: Reverse direction (Strategic to Deontological)
    print(f"\n{'='*80}")
    print("EXPERIMENT 2: Reverse Path Patching (L2→L9)")
    print(f"{'='*80}")
    print("\nHypothesis: Replacing L2→L9 from Strategic to Deontological decreases cooperation")

    result_full_St_to_De = patcher.patch_residual_path(
        source_model_id="PT2_COREDe",  # Strategic (selfish)
        target_model_id="PT3_COREDe",  # Deontological (cooperative)
        start_layer=2,
        end_layer=9,
        component_type="residual",
        scenarios=None,
    )

    save_path_patch_result(result_full_St_to_De, output_dir)

    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Baseline cooperation: {1 - result_full_St_to_De.baseline_p_action2:.2%}")
    print(f"  Patched cooperation: {1 - result_full_St_to_De.patched_p_action2:.2%}")
    print(f"  Δ cooperation: {(1 - result_full_St_to_De.patched_p_action2) - (1 - result_full_St_to_De.baseline_p_action2):+.2%}")
    print(f"{'='*80}")

    hypothesis_supported_reverse = result_full_St_to_De.delta_p_action2 > 0.3  # >30% cooperation decrease
    print(f"\nHypothesis (>30% cooperation decrease): {'✓ SUPPORTED' if hypothesis_supported_reverse else '✗ NOT SUPPORTED'}")

    # Experiment 3: Progressive path patching (Deontological → Strategic)
    print(f"\n{'='*80}")
    print("EXPERIMENT 3: Progressive Path Patching")
    print(f"{'='*80}")
    print("\nFinding critical layer range...")

    progressive_result_De_to_St = patcher.progressive_path_patching(
        source_model_id="PT3_COREDe",
        target_model_id="PT2_COREDe",
        start_layer=2,
        max_layer=9,
        component_type="residual",
        scenarios=None,
    )

    save_progressive_result(progressive_result_De_to_St, output_dir)

    # Experiment 4: MLP-only pathway
    print(f"\n{'='*80}")
    print("EXPERIMENT 4: MLP-Only Pathway (L2→L9)")
    print(f"{'='*80}")

    result_mlp_only_De_to_St = patcher.patch_residual_path(
        source_model_id="PT3_COREDe",
        target_model_id="PT2_COREDe",
        start_layer=2,
        end_layer=9,
        component_type="mlp_only",
        scenarios=None,
    )

    save_path_patch_result(result_mlp_only_De_to_St, output_dir)

    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Δ cooperation (MLP-only): {(1 - result_mlp_only_De_to_St.patched_p_action2) - (1 - result_mlp_only_De_to_St.baseline_p_action2):+.2%}")
    print(f"{'='*80}")

    # Experiment 5: Attention-only pathway
    print(f"\n{'='*80}")
    print("EXPERIMENT 5: Attention-Only Pathway (L2→L9)")
    print(f"{'='*80}")

    result_attn_only_De_to_St = patcher.patch_residual_path(
        source_model_id="PT3_COREDe",
        target_model_id="PT2_COREDe",
        start_layer=2,
        end_layer=9,
        component_type="attn_only",
        scenarios=None,
    )

    save_path_patch_result(result_attn_only_De_to_St, output_dir)

    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Δ cooperation (Attention-only): {(1 - result_attn_only_De_to_St.patched_p_action2) - (1 - result_attn_only_De_to_St.baseline_p_action2):+.2%}")
    print(f"{'='*80}")

    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}\n")

    plot_progressive_patching([progressive_result_De_to_St], output_dir)
    plot_component_comparison(
        result_full_De_to_St,
        result_mlp_only_De_to_St,
        result_attn_only_De_to_St,
        output_dir,
    )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Path Patching Experiments")
    print(f"{'='*80}\n")

    print(f"Full Path (Deontological → Strategic, L2→L9):")
    print(f"  Δ cooperation: {(1 - result_full_De_to_St.patched_p_action2) - (1 - result_full_De_to_St.baseline_p_action2):+.2%}")
    print(f"  Hypothesis: {'✓ SUPPORTED' if hypothesis_supported else '✗ NOT SUPPORTED'}")

    print(f"\nFull Path (Strategic → Deontological, L2→L9):")
    print(f"  Δ cooperation: {(1 - result_full_St_to_De.patched_p_action2) - (1 - result_full_St_to_De.baseline_p_action2):+.2%}")
    print(f"  Hypothesis: {'✓ SUPPORTED' if hypothesis_supported_reverse else '✗ NOT SUPPORTED'}")

    print(f"\nProgressive Patching:")
    if progressive_result_De_to_St.saturation_layer:
        print(f"  Effect saturates at layer: {progressive_result_De_to_St.saturation_layer}")
    else:
        print(f"  Effect continues to grow through L9")

    print(f"\nComponent Decomposition (Deontological → Strategic):")
    delta_full = (1 - result_full_De_to_St.patched_p_action2) - (1 - result_full_De_to_St.baseline_p_action2)
    delta_mlp = (1 - result_mlp_only_De_to_St.patched_p_action2) - (1 - result_mlp_only_De_to_St.baseline_p_action2)
    delta_attn = (1 - result_attn_only_De_to_St.patched_p_action2) - (1 - result_attn_only_De_to_St.baseline_p_action2)

    print(f"  Full residual: {delta_full:+.2%}")
    print(f"  MLP-only: {delta_mlp:+.2%}")
    print(f"  Attention-only: {delta_attn:+.2%}")

    if abs(delta_mlp) > abs(delta_attn):
        print(f"  → MLPs dominate pathway effect ({abs(delta_mlp) / abs(delta_full):.1%} of total)")
    else:
        print(f"  → Attention dominates pathway effect ({abs(delta_attn) / abs(delta_full):.1%} of total)")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")

    if hypothesis_supported and hypothesis_supported_reverse:
        print("✓ STRONG EVIDENCE for L2→L9 pathway causality")
        print("  → Bidirectional path patching shows large behavioral shifts")
        print("  → Effect size (>30%) much larger than single-component patching (0%)")
        print("  → Information flows causally through this pathway")
    elif hypothesis_supported or hypothesis_supported_reverse:
        print("⚠ MODERATE EVIDENCE for L2→L9 pathway causality")
        print("  → Effect observed in one direction but not both")
        print("  → May be asymmetric or context-dependent")
    else:
        print("✗ WEAK EVIDENCE for L2→L9 pathway causality")
        print("  → Effect size smaller than expected")
        print("  → Routing may be more distributed than hypothesized")

    # Compare to single-component patching
    print(f"\nComparison to Single-Component Patching:")
    print(f"  Single-component (21,060 patches): 0 behavioral flips")
    print(f"  Path patching (L2→L9): {abs(delta_full):.1%} cooperation change")
    print(f"  → Path-level effects are {abs(delta_full) / 0.01 if abs(delta_full) > 0.01 else '>0'}x larger")
    print(f"  → Supports distributed encoding + pathway-specific routing")

    print(f"\n{'='*80}")
    print("Path Patching Experiments Complete!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    for f in sorted(output_dir.glob("path_patch_*")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("progressive_patch_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
