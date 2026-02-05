#!/usr/bin/env python3
"""
Comprehensive Activation Steering Experiments

Tests which layers and components can effectively steer moral behavior in LLM agents.
L2_MLP steering FAILED (cooperation barely changed), so this tests evidence-based
alternatives: L16-17 (final decision), L8-9 (cooperation encoders), L19 (attention hub).

Usage:
    python scripts/mech_interp/run_activation_steering_comprehensive.py

Outputs:
    - mech_interp_outputs/causal_routing/steering_vector_*.pt (6 vectors)
    - mech_interp_outputs/causal_routing/steering_sweep_*.csv (24 CSVs)
    - mech_interp_outputs/causal_routing/summary_all_layers.csv (PRIMARY OUTPUT)
    - mech_interp_outputs/causal_routing/comparison_*.png (visualizations)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from mech_interp.activation_steering import (
    ActivationSteerer,
    SteeringSweepResult,
    save_steering_results,
    save_sweep_results,
)
from mech_interp.utils import MODEL_LABELS


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

TIER1_EXPERIMENTS = [
    {
        "layer": 16,
        "component": "mlp",
        "name": "L16_MLP",
        "priority": "Tier1",
        "rationale": "Final decision layer (logit lens divergence)"
    },
    {
        "layer": 17,
        "component": "mlp",
        "name": "L17_MLP",
        "priority": "Tier1",
        "rationale": "Final decision layer (logit lens divergence)"
    },
    {
        "layer": 8,
        "component": "mlp",
        "name": "L8_MLP",
        "priority": "Tier1",
        "rationale": "Pro-Defect encoder (DLA +7-10)"
    },
    {
        "layer": 9,
        "component": "mlp",
        "name": "L9_MLP",
        "priority": "Tier1",
        "rationale": "Pro-Cooperate encoder (DLA -8 to -9)"
    },
    {
        "layer": 19,
        "component": "attn",
        "name": "L19_ATTN",
        "priority": "Tier1",
        "rationale": "Late routing hub (pathway differences)"
    },
]

TIER2_EXPERIMENTS = [
    {
        "layer": 11,
        "component": "mlp",
        "name": "L11_MLP",
        "priority": "Tier2",
        "rationale": "Highest DLA contributor (but similar across models)"
    },
]

MODELS_TO_TEST = ["PT2_COREDe", "PT3_COREDe"]  # Strategic, Deontological
STEERING_STRENGTHS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_monotonicity(sweep_result: SteeringSweepResult) -> Tuple[bool, int]:
    """
    Check if cooperation increases monotonically with positive steering strength.

    Returns:
        (is_monotonic, num_violations)
    """
    # Only check positive strengths (0.0 → +2.0)
    positive_indices = [i for i, s in enumerate(sweep_result.strengths) if s >= 0.0]
    positive_coop_rates = [sweep_result.mean_coop_rate[i] for i in positive_indices]

    violations = 0
    for i in range(len(positive_coop_rates) - 1):
        if positive_coop_rates[i] > positive_coop_rates[i + 1]:
            violations += 1

    return (violations == 0, violations)


def compute_consistency(sweep_result: SteeringSweepResult, strength_idx: int) -> float:
    """
    Compute cross-scenario consistency (std dev of cooperation rates).

    Args:
        sweep_result: Steering sweep result
        strength_idx: Index of strength to analyze (typically max positive strength)

    Returns:
        Standard deviation across scenarios
    """
    scenario_coop_rates = 1 - sweep_result.p_action2_matrix[strength_idx, :]
    return float(np.std(scenario_coop_rates))


def compute_effect_size(delta_coop: float, consistency_std: float) -> float:
    """
    Compute effect size (Cohen's d style).

    Args:
        delta_coop: Change in cooperation rate
        consistency_std: Standard deviation across scenarios

    Returns:
        Effect size (delta / std)
    """
    if consistency_std < 0.001:  # Avoid division by zero
        return float('inf') if abs(delta_coop) > 0.001 else 0.0
    return delta_coop / consistency_std


def check_success_criteria(
    delta_coop: float,
    is_monotonic: bool,
    consistency_std: float
) -> bool:
    """
    Check if experiment meets success criteria.

    Success criteria:
    1. |Δ cooperation| > 5%
    2. Monotonic relationship
    3. Cross-scenario consistency (std < 0.3)

    Returns:
        True if all criteria met
    """
    return (
        abs(delta_coop) > 0.05 and
        is_monotonic and
        consistency_std < 0.3
    )


# ==============================================================================
# PHASE 1: COMPUTE STEERING VECTORS
# ==============================================================================

def compute_all_steering_vectors(
    steerer: ActivationSteerer,
    output_dir: Path
) -> Dict[str, Tuple[torch.Tensor, Dict]]:
    """
    Compute steering vectors for all experiments.

    Args:
        steerer: ActivationSteerer instance
        output_dir: Output directory for saving vectors

    Returns:
        Dict mapping experiment name to (steering_vector, metadata)
    """
    print(f"\n{'='*80}")
    print("PHASE 1: COMPUTE ALL STEERING VECTORS")
    print(f"{'='*80}\n")

    all_experiments = TIER1_EXPERIMENTS + TIER2_EXPERIMENTS
    vectors = {}

    for exp in all_experiments:
        print(f"\n{'-'*80}")
        print(f"Computing: {exp['name']} (Layer {exp['layer']}, {exp['component'].upper()})")
        print(f"Rationale: {exp['rationale']}")
        print(f"{'-'*80}")

        vector, metadata = steerer.find_steering_vector(
            moral_model_id="PT3_COREDe",  # Deontological
            strategic_model_id="PT2_COREDe",  # Strategic
            layer=exp["layer"],
            component=exp["component"],
            scenarios=None,  # All IPD scenarios
        )

        # Save vector
        vector_path = output_dir / f"steering_vector_{exp['name']}_De_minus_Strategic.pt"
        torch.save({
            'vector': vector,
            'metadata': metadata,
            'layer': exp['layer'],
            'component': exp['component'],
            'name': exp['name'],
        }, vector_path)
        print(f"✓ Saved: {vector_path.name}")

        vectors[exp['name']] = (vector, metadata)

    print(f"\n{'='*80}")
    print(f"✓ Computed {len(vectors)} steering vectors")
    print(f"{'='*80}\n")

    return vectors


# ==============================================================================
# PHASE 2-3: RUN TIER EXPERIMENTS
# ==============================================================================

def run_single_layer_experiment(
    steerer: ActivationSteerer,
    exp: Dict,
    model_id: str,
    steering_vector: torch.Tensor,
    output_dir: Path
) -> Dict:
    """
    Run steering sweep experiment for a single layer and model.

    Args:
        steerer: ActivationSteerer instance
        exp: Experiment configuration dict
        model_id: Model ID to test
        steering_vector: Precomputed steering vector
        output_dir: Output directory

    Returns:
        Dict with experiment results and validation metrics
    """
    print(f"\n{'-'*80}")
    print(f"Experiment: {exp['name']} | Model: {MODEL_LABELS.get(model_id, model_id)}")
    print(f"{'-'*80}")

    # Run steering sweep
    sweep_result = steerer.steering_sweep(
        model_id=model_id,
        layer=exp["layer"],
        component=exp["component"],
        steering_vector=steering_vector,
        strengths=STEERING_STRENGTHS,
        scenarios=None,
    )

    # Save results
    save_sweep_results(sweep_result, output_dir)

    # Plot (reuse existing plotting from activation_steering.py)
    from scripts.mech_interp.run_activation_steering import (
        plot_steering_sweep,
        plot_per_scenario_heatmap,
    )
    plot_steering_sweep(sweep_result, output_dir)
    plot_per_scenario_heatmap(sweep_result, output_dir)

    # Compute validation metrics
    baseline_coop = sweep_result.mean_coop_rate[STEERING_STRENGTHS.index(0.0)]
    max_strength_idx = STEERING_STRENGTHS.index(max(STEERING_STRENGTHS))
    steered_coop = sweep_result.mean_coop_rate[max_strength_idx]
    delta_coop = steered_coop - baseline_coop

    is_monotonic, violations = validate_monotonicity(sweep_result)
    consistency_std = compute_consistency(sweep_result, max_strength_idx)
    effect_size = compute_effect_size(delta_coop, consistency_std)
    success = check_success_criteria(delta_coop, is_monotonic, consistency_std)

    # Print summary
    print(f"\nResults:")
    print(f"  Baseline cooperation: {baseline_coop:.1%}")
    print(f"  Steered cooperation (+{max(STEERING_STRENGTHS)}): {steered_coop:.1%}")
    print(f"  Δ cooperation: {delta_coop:+.1%}")
    print(f"  Monotonic: {'✓ YES' if is_monotonic else f'✗ NO ({violations} violations)'}")
    print(f"  Consistency (std): {consistency_std:.3f}")
    print(f"  Effect size: {effect_size:.2f}")
    print(f"  SUCCESS: {'✓ PASSED' if success else '✗ FAILED'}")

    return {
        "layer": exp["layer"],
        "component": exp["component"],
        "name": exp["name"],
        "priority": exp["priority"],
        "rationale": exp["rationale"],
        "model": model_id,
        "baseline_coop": baseline_coop,
        "steered_coop": steered_coop,
        "delta_coop_pct": delta_coop * 100,
        "is_monotonic": is_monotonic,
        "monotonic_violations": violations,
        "consistency_std": consistency_std,
        "effect_size": effect_size,
        "success": success,
    }


def run_tier_experiments(
    steerer: ActivationSteerer,
    tier_experiments: List[Dict],
    vectors: Dict,
    output_dir: Path,
    tier_name: str
) -> List[Dict]:
    """
    Run all experiments for a given tier.

    Args:
        steerer: ActivationSteerer instance
        tier_experiments: List of experiment configs
        vectors: Dict of precomputed steering vectors
        output_dir: Output directory
        tier_name: Tier name for logging

    Returns:
        List of result dicts
    """
    print(f"\n{'='*80}")
    print(f"PHASE {tier_name}: RUN EXPERIMENTS")
    print(f"{'='*80}\n")

    results = []

    for exp in tier_experiments:
        steering_vector, _ = vectors[exp['name']]

        for model_id in MODELS_TO_TEST:
            result = run_single_layer_experiment(
                steerer, exp, model_id, steering_vector, output_dir
            )
            results.append(result)

    print(f"\n{'='*80}")
    print(f"✓ Completed {len(results)} {tier_name} experiments")
    print(f"{'='*80}\n")

    return results


# ==============================================================================
# PHASE 4: MULTI-LAYER STEERING
# ==============================================================================

def run_multilayer_steering(
    steerer: ActivationSteerer,
    vectors: Dict,
    output_dir: Path
) -> List[Dict]:
    """
    Test multi-layer steering (L8 + L16).

    Args:
        steerer: ActivationSteerer instance
        vectors: Dict of precomputed steering vectors
        output_dir: Output directory

    Returns:
        List of result dicts
    """
    print(f"\n{'='*80}")
    print("PHASE 4: MULTI-LAYER STEERING (L8 + L16)")
    print(f"{'='*80}\n")
    print("Testing pathway-level steering:")
    print("  - L8_MLP: Fixed at +1.0 (cooperation encoding)")
    print("  - L16_MLP: Sweep [-2.0 to +2.0] (final decision)")
    print()

    # TODO: Implement multi-layer steering in activation_steering.py
    # For now, return empty results
    print("⚠ Multi-layer steering not yet implemented")
    print("  This requires extending ActivationSteerer to support multiple hooks")
    print()

    return []


# ==============================================================================
# PHASE 5: COMPARATIVE ANALYSIS
# ==============================================================================

def generate_comparative_analysis(
    all_results: List[Dict],
    output_dir: Path
):
    """
    Generate comparative analysis across all experiments.

    Args:
        all_results: Combined results from all experiments
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print("PHASE 5: COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    # 1. Create summary CSV
    df = pd.DataFrame(all_results)
    summary_path = output_dir / "summary_all_layers.csv"
    df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")

    # 2. Print summary table
    print(f"\n{'-'*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'-'*80}\n")

    # Group by layer and show best result
    for (layer, component, name), group in df.groupby(['layer', 'component', 'name']):
        best_result = group.loc[group['delta_coop_pct'].abs().idxmax()]
        print(f"{name:12} | {best_result['model']:12} | "
              f"Δ coop: {best_result['delta_coop_pct']:+6.1f}% | "
              f"Monotonic: {'✓' if best_result['is_monotonic'] else '✗'} | "
              f"Effect: {best_result['effect_size']:5.2f} | "
              f"{'✓ SUCCESS' if best_result['success'] else '✗ FAILED'}")

    print()

    # 3. Generate comparison bar plot
    plot_comparison_bar(df, output_dir)

    # 4. Generate sweep overlay plot
    plot_sweep_overlay(df, output_dir)

    # 5. Generate effect size heatmap
    plot_effect_heatmap(df, output_dir)

    # 6. Print success summary
    num_successful = df['success'].sum()
    print(f"\n{'='*80}")
    print(f"SUCCESS SUMMARY: {num_successful}/{len(df)} experiments passed")
    print(f"{'='*80}\n")

    successful = df[df['success']]
    if len(successful) > 0:
        print("Successful experiments:")
        for _, row in successful.iterrows():
            print(f"  - {row['name']} ({row['model']}): {row['delta_coop_pct']:+.1f}% cooperation change")
    else:
        print("⚠ No experiments met success criteria (>5% change + monotonic + consistent)")

    print()


def plot_comparison_bar(df: pd.DataFrame, output_dir: Path):
    """Generate bar plot comparing cooperation changes across layers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot data for grouped bar chart
    pivot = df.pivot_table(
        index='name',
        columns='model',
        values='delta_coop_pct',
        aggfunc='first'
    )

    pivot.plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.3, label='Success threshold (+5%)')
    ax.axhline(y=-5, color='green', linestyle='--', alpha=0.3)

    ax.set_xlabel('Layer / Component', fontsize=12)
    ax.set_ylabel('Cooperation Change (%)', fontsize=12)
    ax.set_title('Steering Effectiveness by Layer and Model', fontsize=14)
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_path = output_dir / "comparison_bar_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison bar plot: {plot_path}")
    plt.close()


def plot_sweep_overlay(df: pd.DataFrame, output_dir: Path):
    """Generate overlay plot of all steering sweeps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by layer and model
    for (name, model), group in df.groupby(['name', 'model']):
        # Load sweep results
        component = group.iloc[0]['component']
        layer = group.iloc[0]['layer']

        sweep_path = output_dir / f"steering_sweep_{model}_L{layer}_{component}.csv"
        if sweep_path.exists():
            sweep_df = pd.read_csv(sweep_path)

            label = f"{name} ({MODEL_LABELS.get(model, model)})"
            linestyle = '-' if model == 'PT2_COREDe' else '--'
            ax.plot(sweep_df['strength'], sweep_df['mean_coop_rate'],
                   marker='o', linestyle=linestyle, label=label, alpha=0.7)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No steering')

    ax.set_xlabel('Steering Strength', fontsize=12)
    ax.set_ylabel('Cooperation Rate', fontsize=12)
    ax.set_title('All Steering Sweeps Overlay', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / "comparison_sweep_overlay.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sweep overlay plot: {plot_path}")
    plt.close()


def plot_effect_heatmap(df: pd.DataFrame, output_dir: Path):
    """Generate heatmap of effect sizes across layers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot data for heatmap
    pivot = df.pivot_table(
        index='name',
        columns='model',
        values='effect_size',
        aggfunc='first'
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Effect Size'},
        ax=ax,
    )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Layer / Component', fontsize=12)
    ax.set_title('Effect Size by Layer and Model', fontsize=14)

    plt.tight_layout()

    plot_path = output_dir / "effect_size_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved effect size heatmap: {plot_path}")
    plt.close()


# ==============================================================================
# PHASE 6: DOWNSTREAM EFFECTS (CONDITIONAL)
# ==============================================================================

def run_downstream_effects(
    steerer: ActivationSteerer,
    successful_experiments: List[Dict],
    vectors: Dict,
    output_dir: Path
):
    """
    Run downstream effect analysis for successful experiments.

    Args:
        steerer: ActivationSteerer instance
        successful_experiments: List of successful experiment results
        vectors: Dict of precomputed steering vectors
        output_dir: Output directory
    """
    if len(successful_experiments) == 0:
        print(f"\n{'='*80}")
        print("PHASE 6: DOWNSTREAM EFFECTS - SKIPPED")
        print(f"{'='*80}\n")
        print("⚠ No successful experiments to analyze\n")
        return

    print(f"\n{'='*80}")
    print(f"PHASE 6: DOWNSTREAM EFFECTS ({len(successful_experiments)} experiments)")
    print(f"{'='*80}\n")

    for exp_result in successful_experiments:
        print(f"\n{'-'*80}")
        print(f"Analyzing: {exp_result['name']} ({exp_result['model']})")
        print(f"{'-'*80}")

        steering_vector, _ = vectors[exp_result['name']]

        downstream_result = steerer.downstream_effect_analysis(
            model_id=exp_result['model'],
            steering_layer=exp_result['layer'],
            steering_component=exp_result['component'],
            steering_vector=steering_vector,
            strength=1.0,
            downstream_layers=[8, 9, 16, 17],  # Measure all key layers
            scenarios=None,
        )

        # Save results
        downstream_df = pd.DataFrame({
            'layer': downstream_result.layer_indices,
            'component': downstream_result.top_affected_components,
            'mean_abs_change': downstream_result.activation_changes,
        })

        csv_path = output_dir / f"downstream_effects_{exp_result['name']}_strength+1.0.csv"
        downstream_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

        # Plot
        from scripts.mech_interp.run_activation_steering import plot_downstream_effects
        plot_downstream_effects(downstream_result, output_dir)

    print()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """Run comprehensive activation steering experiments."""

    # Output directory
    output_dir = project_root / "mech_interp_outputs" / "causal_routing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE ACTIVATION STEERING EXPERIMENTS")
    print("=" * 80)
    print("\nObjective: Test which layers can effectively steer moral behavior")
    print("Background: L2_MLP steering FAILED (cooperation: 15.8% → 14.8%)")
    print("\nHypotheses to Test:")
    print("  Tier 1 (High Priority):")
    print("    - L16-17 MLP: Final decision layers")
    print("    - L8-9 MLP: Cooperation/defection encoders")
    print("    - L19 Attention: Late routing hub")
    print("  Tier 2:")
    print("    - L11 MLP: Highest DLA contributor")
    print("    - Multi-layer: L8+L16 pathway intervention")
    print("\nModels: Strategic (PT2_COREDe), Deontological (PT3_COREDe)")
    print(f"Strengths: {STEERING_STRENGTHS}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)

    # Initialize steerer
    steerer = ActivationSteerer(device="cuda")

    # Phase 1: Compute steering vectors
    vectors = compute_all_steering_vectors(steerer, output_dir)

    # Phase 2: Run Tier 1 experiments
    tier1_results = run_tier_experiments(
        steerer, TIER1_EXPERIMENTS, vectors, output_dir, "2 (TIER 1)"
    )

    # Phase 3: Run Tier 2 experiments
    tier2_results = run_tier_experiments(
        steerer, TIER2_EXPERIMENTS, vectors, output_dir, "3 (TIER 2)"
    )

    # Phase 4: Multi-layer steering
    multilayer_results = run_multilayer_steering(steerer, vectors, output_dir)

    # Combine all results
    all_results = tier1_results + tier2_results + multilayer_results

    # Phase 5: Comparative analysis
    generate_comparative_analysis(all_results, output_dir)

    # Phase 6: Downstream effects (conditional)
    successful_experiments = [r for r in all_results if r['success']]
    run_downstream_effects(steerer, successful_experiments, vectors, output_dir)

    # Final summary
    print("=" * 80)
    print("COMPREHENSIVE STEERING EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nPrimary outputs:")
    print(f"  - {output_dir / 'summary_all_layers.csv'}")
    print(f"  - {output_dir / 'comparison_bar_plot.png'}")
    print(f"  - {output_dir / 'comparison_sweep_overlay.png'}")
    print(f"  - {output_dir / 'effect_size_heatmap.png'}")
    print("\nNext steps:")
    print("  1. Review summary_all_layers.csv to see which layers succeeded")
    print("  2. Examine comparison plots for visual ranking")
    print("  3. If layers succeeded, run path patching to trace information flow")
    print("  4. If all failed, consider residual stream steering or pathway interventions")
    print("=" * 80)


if __name__ == "__main__":
    main()
