#!/usr/bin/env python3
"""
Run Activation Steering experiments.

Tests whether steering L2_MLP activations can control moral behavior by adding
directional vectors to activations. This provides evidence for L2_MLP's role in
routing information relevant to moral decisions.

Key experiments:
    1. Compute steering vector (moral - strategic direction)
    2. Apply steering to Strategic model → expect cooperation increase
    3. Apply steering to Deontological model → expect cooperation decrease
    4. Steering sweep to show continuous control
    5. Downstream effect analysis (L8/L9 MLP changes)

Usage:
    python scripts/mech_interp/run_activation_steering.py

Outputs:
    - mech_interp_outputs/causal_routing/steering_vector_*.pt
    - mech_interp_outputs/causal_routing/steering_*_L*_*.csv
    - mech_interp_outputs/causal_routing/steering_sweep_*.csv
    - mech_interp_outputs/causal_routing/steering_sweep_*.png
    - mech_interp_outputs/causal_routing/downstream_effects_*.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mech_interp.activation_steering import (
    ActivationSteerer,
    save_steering_results,
    save_sweep_results,
)
from mech_interp.utils import MODEL_LABELS


def plot_steering_sweep(sweep_result, output_dir):
    """Create visualization of steering sweep results."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cooperation rate vs steering strength
    ax1.plot(sweep_result.strengths, sweep_result.mean_coop_rate,
             marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No steering')
    ax1.set_xlabel('Steering Strength', fontsize=12)
    ax1.set_ylabel('Cooperation Rate', fontsize=12)
    ax1.set_title(f'Cooperation Rate vs Steering Strength\n{MODEL_LABELS.get(sweep_result.model_id, sweep_result.model_id)}', fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot 2: p(action2) vs steering strength
    ax2.plot(sweep_result.strengths, sweep_result.mean_p_action2,
             marker='o', linewidth=2, markersize=8, color='coral')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No steering')
    ax2.set_xlabel('Steering Strength', fontsize=12)
    ax2.set_ylabel('p(action2) - Defection', fontsize=12)
    ax2.set_title(f'Defection Probability vs Steering Strength\n{MODEL_LABELS.get(sweep_result.model_id, sweep_result.model_id)}', fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save
    filename = f"steering_sweep_{sweep_result.model_id}_L{sweep_result.layer}_{sweep_result.component}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sweep plot: {plot_path}")
    plt.close()


def plot_per_scenario_heatmap(sweep_result, output_dir):
    """Create heatmap showing per-scenario steering effects."""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert p_action2 to cooperation rate for easier interpretation
    coop_matrix = 1 - sweep_result.p_action2_matrix

    sns.heatmap(
        coop_matrix,
        xticklabels=sweep_result.scenarios,
        yticklabels=[f"{s:+.1f}" for s in sweep_result.strengths],
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Cooperation Rate'},
        ax=ax,
    )

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Steering Strength', fontsize=12)
    ax.set_title(f'Cooperation Rate by Steering Strength and Scenario\n{MODEL_LABELS.get(sweep_result.model_id, sweep_result.model_id)}', fontsize=13)

    plt.tight_layout()

    # Save
    filename = f"steering_heatmap_{sweep_result.model_id}_L{sweep_result.layer}_{sweep_result.component}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved per-scenario heatmap: {plot_path}")
    plt.close()


def plot_downstream_effects(downstream_result, output_dir):
    """Create visualization of downstream activation changes."""

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        [f"L{layer}" for layer in downstream_result.layer_indices],
        downstream_result.activation_changes,
        color='steelblue',
        alpha=0.8,
    )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Absolute Activation Change', fontsize=12)
    ax.set_title(f'Downstream Effects of Steering L{downstream_result.steering_layer}_MLP\nStrength: {downstream_result.steering_strength:+.1f}', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    filename = f"downstream_effects_L{downstream_result.steering_layer}_strength{downstream_result.steering_strength:+.1f}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved downstream effects plot: {plot_path}")
    plt.close()


def main():
    """Run activation steering experiments."""

    # Output directory
    output_dir = project_root / "mech_interp_outputs" / "causal_routing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ACTIVATION STEERING EXPERIMENTS")
    print("=" * 80)
    print("\nTesting hypothesis: L2_MLP controls moral routing")
    print("Method: Add directional vectors to L2_MLP activations")
    print("\nExperiments:")
    print("  1. Compute steering vector (Deontological - Strategic)")
    print("  2. Steer Strategic model → Expect cooperation increase")
    print("  3. Steer Deontological model → Expect cooperation decrease")
    print("  4. Steering sweep → Test continuous control")
    print("  5. Downstream effects → Measure L8/L9 MLP changes")
    print("=" * 80)

    # Initialize steerer
    steerer = ActivationSteerer(device="cuda")

    # Experiment 1: Compute steering vector
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: Compute Steering Vector")
    print(f"{'='*80}")

    steering_vector, metadata = steerer.find_steering_vector(
        moral_model_id="PT3_COREDe",  # Deontological (cooperative)
        strategic_model_id="PT2_COREDe",  # Strategic (selfish)
        layer=2,
        component="mlp",
        scenarios=None,  # Use all IPD scenarios
    )

    # Save steering vector
    vector_path = output_dir / "steering_vector_L2_mlp_De_minus_Strategic.pt"
    torch.save({
        'vector': steering_vector,
        'metadata': metadata,
    }, vector_path)
    print(f"\n✓ Saved steering vector: {vector_path}")

    # Experiment 2: Steer Strategic model (expect cooperation increase)
    print(f"\n{'='*80}")
    print("EXPERIMENT 2: Steer Strategic Model")
    print(f"{'='*80}")
    print("\nHypothesis: Steering Strategic in 'moral' direction increases cooperation")
    print("Testing strength: +1.0")

    result_strategic = steerer.steer_and_evaluate(
        model_id="PT2_COREDe",
        layer=2,
        component="mlp",
        steering_vector=steering_vector,
        strength=+1.0,  # Positive = toward moral
        scenarios=None,
    )

    print(f"\nResults:")
    print(f"  Baseline p(action2): {result_strategic.baseline_p_action2:.4f}")
    print(f"  Steered p(action2): {result_strategic.steered_p_action2:.4f}")
    print(f"  Δ p(action2): {result_strategic.delta_p_action2:+.4f}")
    print(f"  Baseline cooperation: {1 - result_strategic.baseline_p_action2:.2%}")
    print(f"  Steered cooperation: {1 - result_strategic.steered_p_action2:.2%}")
    print(f"  Δ cooperation: {(1 - result_strategic.steered_p_action2) - (1 - result_strategic.baseline_p_action2):+.2%}")

    save_steering_results(result_strategic, output_dir)

    # Check hypothesis
    hypothesis_supported_strategic = result_strategic.delta_p_action2 < -0.05  # More cooperation
    print(f"\nHypothesis: {'✓ SUPPORTED' if hypothesis_supported_strategic else '✗ NOT SUPPORTED'}")

    # Experiment 3: Steer Deontological model (expect cooperation decrease)
    print(f"\n{'='*80}")
    print("EXPERIMENT 3: Steer Deontological Model")
    print(f"{'='*80}")
    print("\nHypothesis: Steering Deontological in 'strategic' direction decreases cooperation")
    print("Testing strength: -1.0")

    result_deontological = steerer.steer_and_evaluate(
        model_id="PT3_COREDe",
        layer=2,
        component="mlp",
        steering_vector=steering_vector,
        strength=-1.0,  # Negative = toward strategic
        scenarios=None,
    )

    print(f"\nResults:")
    print(f"  Baseline p(action2): {result_deontological.baseline_p_action2:.4f}")
    print(f"  Steered p(action2): {result_deontological.steered_p_action2:.4f}")
    print(f"  Δ p(action2): {result_deontological.delta_p_action2:+.4f}")
    print(f"  Baseline cooperation: {1 - result_deontological.baseline_p_action2:.2%}")
    print(f"  Steered cooperation: {1 - result_deontological.steered_p_action2:.2%}")
    print(f"  Δ cooperation: {(1 - result_deontological.steered_p_action2) - (1 - result_deontological.baseline_p_action2):+.2%}")

    save_steering_results(result_deontological, output_dir)

    # Check hypothesis
    hypothesis_supported_deontological = result_deontological.delta_p_action2 > 0.05  # Less cooperation
    print(f"\nHypothesis: {'✓ SUPPORTED' if hypothesis_supported_deontological else '✗ NOT SUPPORTED'}")

    # Experiment 4a: Steering sweep on Strategic model
    print(f"\n{'='*80}")
    print("EXPERIMENT 4a: Steering Sweep (Strategic Model)")
    print(f"{'='*80}")

    sweep_strategic = steerer.steering_sweep(
        model_id="PT2_COREDe",
        layer=2,
        component="mlp",
        steering_vector=steering_vector,
        strengths=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        scenarios=None,
    )

    save_sweep_results(sweep_strategic, output_dir)
    plot_steering_sweep(sweep_strategic, output_dir)
    plot_per_scenario_heatmap(sweep_strategic, output_dir)

    # Check for monotonicity
    is_monotonic = all(
        sweep_strategic.mean_p_action2[i] >= sweep_strategic.mean_p_action2[i + 1]
        for i in range(len(sweep_strategic.mean_p_action2) - 1)
    )
    print(f"\nMonotonic relationship: {'✓ YES' if is_monotonic else '✗ NO'}")

    # Experiment 4b: Steering sweep on Deontological model
    print(f"\n{'='*80}")
    print("EXPERIMENT 4b: Steering Sweep (Deontological Model)")
    print(f"{'='*80}")

    sweep_deontological = steerer.steering_sweep(
        model_id="PT3_COREDe",
        layer=2,
        component="mlp",
        steering_vector=steering_vector,
        strengths=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        scenarios=None,
    )

    save_sweep_results(sweep_deontological, output_dir)
    plot_steering_sweep(sweep_deontological, output_dir)
    plot_per_scenario_heatmap(sweep_deontological, output_dir)

    # Experiment 5: Downstream effects (Strategic model)
    print(f"\n{'='*80}")
    print("EXPERIMENT 5: Downstream Effect Analysis")
    print(f"{'='*80}")
    print("\nMeasuring how steering L2_MLP affects L8 and L9 MLPs")

    downstream_result = steerer.downstream_effect_analysis(
        model_id="PT2_COREDe",
        steering_layer=2,
        steering_component="mlp",
        steering_vector=steering_vector,
        strength=1.0,
        downstream_layers=[8, 9],
        scenarios=None,
    )

    # Save results
    downstream_df = pd.DataFrame({
        'layer': downstream_result.layer_indices,
        'component': downstream_result.top_affected_components,
        'mean_abs_change': downstream_result.activation_changes,
    })

    downstream_csv = output_dir / f"downstream_effects_L{downstream_result.steering_layer}_strength{downstream_result.steering_strength:+.1f}.csv"
    downstream_df.to_csv(downstream_csv, index=False)
    print(f"\n✓ Saved downstream effects: {downstream_csv}")

    plot_downstream_effects(downstream_result, output_dir)

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY: Activation Steering Experiments")
    print(f"{'='*80}\n")

    print(f"Steering Vector:")
    print(f"  Source: Deontological - Strategic")
    print(f"  Layer: L2_MLP")
    print(f"  Cosine similarity: {metadata['cosine_similarity']:.4f}")
    print(f"  Euclidean distance: {metadata['euclidean_distance']:.2f}")

    print(f"\nSteering Strategic Model (+1.0):")
    print(f"  Δ cooperation: {(1 - result_strategic.steered_p_action2) - (1 - result_strategic.baseline_p_action2):+.2%}")
    print(f"  Hypothesis: {'✓ SUPPORTED' if hypothesis_supported_strategic else '✗ NOT SUPPORTED'}")

    print(f"\nSteering Deontological Model (-1.0):")
    print(f"  Δ cooperation: {(1 - result_deontological.steered_p_action2) - (1 - result_deontological.baseline_p_action2):+.2%}")
    print(f"  Hypothesis: {'✓ SUPPORTED' if hypothesis_supported_deontological else '✗ NOT SUPPORTED'}")

    print(f"\nSteering Sweep:")
    print(f"  Monotonic relationship: {'✓ YES' if is_monotonic else '✗ NO'}")
    print(f"  Range (Strategic): {min(sweep_strategic.mean_coop_rate):.2%} to {max(sweep_strategic.mean_coop_rate):.2%}")
    print(f"  Range (Deontological): {min(sweep_deontological.mean_coop_rate):.2%} to {max(sweep_deontological.mean_coop_rate):.2%}")

    print(f"\nDownstream Effects:")
    for layer, change in zip(downstream_result.layer_indices, downstream_result.activation_changes):
        print(f"  L{layer}_MLP: {change:.4f} mean absolute change")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")

    if hypothesis_supported_strategic and hypothesis_supported_deontological and is_monotonic:
        print("✓ STRONG EVIDENCE for L2_MLP routing control")
        print("  → Steering L2_MLP bidirectionally shifts behavior")
        print("  → Monotonic relationship shows continuous control")
        print("  → L2_MLP acts as a causal routing switch")
    elif hypothesis_supported_strategic or hypothesis_supported_deontological:
        print("⚠ MODERATE EVIDENCE for L2_MLP routing control")
        print("  → Some steering effect observed")
        print("  → May be context-dependent or require larger strengths")
    else:
        print("✗ WEAK EVIDENCE for L2_MLP routing control")
        print("  → Steering L2_MLP alone may not be sufficient")
        print("  → Consider path patching to test pathway-level effects")

    print(f"\n{'='*80}")
    print("Activation Steering Experiments Complete!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    for f in sorted(output_dir.glob("steering_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
