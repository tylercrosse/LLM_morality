#!/usr/bin/env python3
"""
Run Frankenstein (LoRA Weight Transplant) experiments.

Tests the hypothesis that L2_MLP acts as a routing switch by transplanting its
LoRA weights from one model to another. If L2_MLP weights are sufficient to
shift behavior, this provides direct causal evidence for the routing hypothesis.

Example experiments:
    - Strategic model + Deontological L2_MLP → Increased cooperation?
    - Deontological model + Strategic L2_MLP → Increased defection?

Usage:
    python scripts/mech_interp/run_frankenstein.py

Outputs:
    - mech_interp_outputs/causal_routing/frankenstein_*.csv (per-scenario results)
    - mech_interp_outputs/causal_routing/frankenstein_*_summary.json (summary stats)
    - mech_interp_outputs/causal_routing/frankenstein_summary_all.csv (all experiments)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mech_interp.lora_weight_transplant import (
    LoRAWeightTransplanter,
    save_transplant_results,
)
from mech_interp.utils import MODEL_LABELS


def plot_transplant_comparison(all_results, output_dir):
    """Create visualization comparing all transplant experiments."""

    # Prepare data
    rows = []
    for result in all_results:
        rows.append({
            'experiment': f"{result.source_model_id} → {result.target_model_id}",
            'baseline_coop': result.baseline_coop_rate,
            'transplanted_coop': result.transplanted_coop_rate,
            'delta_coop': result.delta_coop_rate,
        })

    df = pd.DataFrame(rows)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Before vs After cooperation rates
    x_pos = range(len(df))
    width = 0.35

    ax1.bar([x - width/2 for x in x_pos], df['baseline_coop'], width,
            label='Baseline', alpha=0.8, color='steelblue')
    ax1.bar([x + width/2 for x in x_pos], df['transplanted_coop'], width,
            label='Transplanted', alpha=0.8, color='coral')

    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation Rate: Baseline vs Transplanted')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['experiment'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Delta cooperation rate
    colors = ['green' if d > 0 else 'red' for d in df['delta_coop']]
    ax2.bar(x_pos, df['delta_coop'], color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Δ Cooperation Rate')
    ax2.set_title('Change in Cooperation Rate')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df['experiment'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    plot_path = output_dir / "frankenstein_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {plot_path}")
    plt.close()


def plot_per_scenario_effects(all_results, output_dir):
    """Create heatmap showing per-scenario effects."""

    # Prepare data for heatmap
    heatmap_data = []
    scenario_names = all_results[0].scenarios

    for result in all_results:
        experiment_name = f"{result.source_model_id} → {result.target_model_id}"
        delta_coop_rates = [
            (1 - t) - (1 - b)
            for t, b in zip(result.transplanted_p_action2, result.baseline_p_action2)
        ]
        heatmap_data.append({
            'experiment': experiment_name,
            **{scenario: delta for scenario, delta in zip(scenario_names, delta_coop_rates)}
        })

    df = pd.DataFrame(heatmap_data)
    df = df.set_index('experiment')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df,
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Δ Cooperation Rate'},
        ax=ax,
        vmin=-0.3,
        vmax=0.3,
    )
    ax.set_title('Per-Scenario Cooperation Rate Changes')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Experiment')

    plt.tight_layout()

    # Save
    plot_path = output_dir / "frankenstein_per_scenario_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved per-scenario heatmap: {plot_path}")
    plt.close()


def main():
    """Run Frankenstein experiments."""

    # Output directory
    output_dir = project_root / "mech_interp_outputs" / "causal_routing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FRANKENSTEIN EXPERIMENTS: LoRA Weight Transplant")
    print("=" * 80)
    print("\nTesting hypothesis: L2_MLP acts as routing switch")
    print("Method: Transplant L2_MLP LoRA weights between models")
    print("\nExperiments:")
    print("  1. Strategic + Deontological_L2 → Expect cooperation increase")
    print("  2. Deontological + Strategic_L2 → Expect cooperation decrease")
    print("  3. Utilitarian + Deontological_L2 → Expect cooperation increase")
    print("  4. Deontological + Utilitarian_L2 → Expect cooperation decrease")
    print("=" * 80)

    # Initialize transplanter
    transplanter = LoRAWeightTransplanter(device="cuda")

    # Define experiments
    experiments = [
        {
            "source": "PT3_COREDe",  # Deontological (cooperative)
            "target": "PT2_COREDe",  # Strategic (selfish)
            "hypothesis": "Adding Deontological L2_MLP to Strategic should increase cooperation",
            "expected_direction": "positive",
        },
        {
            "source": "PT2_COREDe",  # Strategic (selfish)
            "target": "PT3_COREDe",  # Deontological (cooperative)
            "hypothesis": "Adding Strategic L2_MLP to Deontological should decrease cooperation",
            "expected_direction": "negative",
        },
        {
            "source": "PT3_COREDe",  # Deontological (cooperative)
            "target": "PT3_COREUt",  # Utilitarian (mostly cooperative)
            "hypothesis": "Adding Deontological L2_MLP to Utilitarian should increase cooperation",
            "expected_direction": "positive",
        },
        {
            "source": "PT3_COREUt",  # Utilitarian (mostly cooperative)
            "target": "PT3_COREDe",  # Deontological (cooperative)
            "hypothesis": "Adding Utilitarian L2_MLP to Deontological should slightly decrease cooperation",
            "expected_direction": "negative",
        },
    ]

    all_results = []

    # Run each experiment
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}")
        print(f"{'='*80}")
        print(f"\nSource: {MODEL_LABELS.get(exp['source'], exp['source'])} ({exp['source']})")
        print(f"Target: {MODEL_LABELS.get(exp['target'], exp['target'])} ({exp['target']})")
        print(f"\nHypothesis: {exp['hypothesis']}")
        print(f"Expected direction: {exp['expected_direction']}")

        try:
            # Run transplant experiment
            result = transplanter.run_transplant_experiment(
                source_model_id=exp['source'],
                target_model_id=exp['target'],
                layer=2,  # L2_MLP
                module_types=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
                scenarios=None,  # Use all IPD scenarios
            )

            all_results.append(result)

            # Save individual result
            save_transplant_results(result, output_dir)

            # Check if hypothesis supported
            if exp['expected_direction'] == 'positive':
                supported = result.delta_coop_rate > 0.05  # >5% increase
                threshold_desc = ">5% cooperation increase"
            else:
                supported = result.delta_coop_rate < -0.05  # >5% decrease
                threshold_desc = ">5% cooperation decrease"

            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i} RESULT:")
            print(f"  Hypothesis: {'✓ SUPPORTED' if supported else '✗ NOT SUPPORTED'}")
            print(f"  Expected: {threshold_desc}")
            print(f"  Observed: {result.delta_coop_rate:+.2%}")
            print(f"{'='*80}")

        except Exception as e:
            print(f"\n⚠ Experiment {i} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY: All Frankenstein Experiments")
    print(f"{'='*80}\n")

    summary_rows = []
    for i, (exp, result) in enumerate(zip(experiments, all_results), 1):
        summary_rows.append({
            'experiment_id': i,
            'source_model': exp['source'],
            'source_name': MODEL_LABELS.get(exp['source'], exp['source']),
            'target_model': exp['target'],
            'target_name': MODEL_LABELS.get(exp['target'], exp['target']),
            'baseline_p_action2': result.baseline_p_action2_mean,
            'transplanted_p_action2': result.transplanted_p_action2_mean,
            'delta_p_action2': result.delta_p_action2,
            'baseline_coop_rate': result.baseline_coop_rate,
            'transplanted_coop_rate': result.transplanted_coop_rate,
            'delta_coop_rate': result.delta_coop_rate,
            'expected_direction': exp['expected_direction'],
            'hypothesis_supported': (
                (exp['expected_direction'] == 'positive' and result.delta_coop_rate > 0.05) or
                (exp['expected_direction'] == 'negative' and result.delta_coop_rate < -0.05)
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "frankenstein_summary_all.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary: {summary_csv}\n")

    # Print table
    print(summary_df.to_string(index=False))

    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}\n")

    plot_transplant_comparison(all_results, output_dir)
    plot_per_scenario_effects(all_results, output_dir)

    # Final interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")

    supported_count = sum(row['hypothesis_supported'] for row in summary_rows)
    total_count = len(summary_rows)

    print(f"Hypotheses supported: {supported_count}/{total_count}")

    if supported_count >= 3:
        print("\n✓ STRONG EVIDENCE for L2_MLP routing hypothesis")
        print("  → Transplanting L2_MLP weights alone is sufficient to shift behavior")
        print("  → L2_MLP acts as a causal routing switch")
    elif supported_count >= 2:
        print("\n⚠ MODERATE EVIDENCE for L2_MLP routing hypothesis")
        print("  → Some effect observed, but not consistent across all experiments")
    else:
        print("\n✗ WEAK EVIDENCE for L2_MLP routing hypothesis")
        print("  → L2_MLP weights alone may not be sufficient to explain routing")
        print("  → Consider path patching or activation steering for further investigation")

    print(f"\n{'='*80}")
    print("Frankenstein Experiments Complete!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    for f in sorted(output_dir.glob("frankenstein_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
