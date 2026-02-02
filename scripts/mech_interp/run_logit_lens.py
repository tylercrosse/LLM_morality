#!/usr/bin/env python3
"""
Run complete logit lens analysis on all models and scenarios.

Generates:
- Individual trajectory plots for each model/scenario
- Comparison plots across models for each scenario
- Grid visualization of all scenarios
- Heatmap of final layer preferences
- Summary statistics CSV
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
from typing import Dict
import json

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.prompt_generator import IPDPromptGenerator
from mech_interp.logit_lens import LogitLensAnalyzer, LogitLensVisualizer
from mech_interp.utils import MODEL_LABELS


def run_full_analysis(
    models_to_analyze: list = None,
    scenarios_to_analyze: list = None,
    output_dir: str = str(PROJECT_ROOT / "mech_interp_outputs" / "logit_lens"),
):
    """
    Run complete logit lens analysis.

    Args:
        models_to_analyze: List of model IDs (default: all models)
        scenarios_to_analyze: List of scenario keys (default: all scenarios)
        output_dir: Output directory for results
    """
    if models_to_analyze is None:
        models_to_analyze = ["base", "PT2_COREDe", "PT3_COREDe", "PT3_COREUt", "PT4_COREDe"]

    if scenarios_to_analyze is None:
        scenarios_to_analyze = [
            "CC_continue", "CC_temptation", "CD_punished",
            "DC_exploited", "DD_trapped"
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("LOGIT LENS ANALYSIS")
    print("="*70)
    print(f"\nModels: {len(models_to_analyze)}")
    print(f"Scenarios: {len(scenarios_to_analyze)}")
    print(f"Output: {output_dir}\n")

    # Load prompt dataset
    prompt_path = PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json"
    dataset = IPDPromptGenerator.load_dataset(prompt_path)

    # Load tokenizer once
    tokenizer = LoRAModelLoader.load_tokenizer()

    # Storage for results
    all_trajectories = {}  # scenario -> model_id -> trajectory
    final_deltas = {}  # scenario -> model_id -> final_delta
    decision_stats = []  # Per-variant stats
    aggregate_stats = []  # Per (scenario, model) stats
    all_variant_trajectories = {}  # scenario -> model_id -> variant_id -> trajectory

    # Analyze each scenario
    for scenario in scenarios_to_analyze:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*70}\n")

        all_trajectories[scenario] = {}
        final_deltas[scenario] = {}
        all_variant_trajectories[scenario] = {}

        # Get all prompt variants for this scenario
        prompts = IPDPromptGenerator.get_prompts_by_scenario(dataset, scenario)
        if not prompts:
            print(f"Warning: No prompts found for {scenario}")
            continue

        prompts = sorted(prompts, key=lambda p: p.get("variant", 0))
        print(f"  Prompt variants: {len(prompts)}")

        # Analyze each model
        for model_id in models_to_analyze:
            print(f"  Analyzing {MODEL_LABELS[model_id]}...")

            # Load model
            model = LoRAModelLoader.load_hooked_model(model_id)

            # Analyze
            analyzer = LogitLensAnalyzer(model, tokenizer)
            all_variant_trajectories[scenario][model_id] = {}
            variant_trajectories = []
            variant_final_deltas = []

            for prompt_data in prompts:
                trajectory = analyzer.compute_action_trajectory(prompt_data["prompt"])
                decision_info = analyzer.analyze_decision_layer(trajectory)

                variant_id = prompt_data.get("variant", -1)
                all_variant_trajectories[scenario][model_id][f"v{variant_id}"] = trajectory
                variant_trajectories.append(trajectory)
                variant_final_deltas.append(float(trajectory[-1]))

                decision_stats.append({
                    "scenario": scenario,
                    "variant": variant_id,
                    "seed": prompt_data.get("seed"),
                    "prompt_id": prompt_data.get("id"),
                    "model_id": model_id,
                    "model_name": MODEL_LABELS[model_id],
                    "final_delta": float(trajectory[-1]),
                    "first_decision_layer": decision_info['first_decision_layer'],
                    "stabilization_layer": decision_info['stabilization_layer'],
                    "max_abs_delta": float(decision_info['max_delta']),
                    "mean_delta": float(decision_info['mean_delta']),
                })

            mean_trajectory = np.mean(np.stack(variant_trajectories, axis=0), axis=0)
            mean_decision_info = analyzer.analyze_decision_layer(mean_trajectory)

            # Store aggregate results for visualization and summary
            all_trajectories[scenario][model_id] = mean_trajectory
            final_deltas[scenario][model_id] = float(mean_trajectory[-1])

            aggregate_stats.append({
                "scenario": scenario,
                "model_id": model_id,
                "model_name": MODEL_LABELS[model_id],
                "n_variants": len(variant_final_deltas),
                "final_delta_mean": float(np.mean(variant_final_deltas)),
                "final_delta_std": float(np.std(variant_final_deltas)),
                "final_delta_min": float(np.min(variant_final_deltas)),
                "final_delta_max": float(np.max(variant_final_deltas)),
                "first_decision_layer": mean_decision_info['first_decision_layer'],
                "stabilization_layer": mean_decision_info['stabilization_layer'],
                "max_abs_delta": float(mean_decision_info['max_delta']),
                "mean_delta": float(mean_decision_info['mean_delta']),
            })

            print(
                f"    Final Δ (mean±std over variants): "
                f"{np.mean(variant_final_deltas):.2f}±{np.std(variant_final_deltas):.2f} "
                f"({'Defect' if mean_trajectory[-1] > 0 else 'Cooperate'})"
            )

            # Cleanup
            del model, analyzer
            torch.cuda.empty_cache()

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    # 1. Comparison plots for each scenario
    print("1. Creating scenario comparison plots...")
    for scenario in scenarios_to_analyze:
        if scenario in all_trajectories:
            output_path = output_dir / f"comparison_{scenario}.png"
            LogitLensVisualizer.plot_model_comparison(
                all_trajectories[scenario],
                scenario,
                output_path,
            )
            print(f"   ✓ {scenario}")

    # 2. Grid of all scenarios
    print("\n2. Creating scenario grid...")
    LogitLensVisualizer.plot_scenario_grid(
        all_trajectories,
        scenarios_to_analyze,
        output_dir / "all_scenarios_grid.png"
    )
    print("   ✓ Grid complete")

    # 3. Heatmap of final preferences
    print("\n3. Creating final preferences heatmap...")
    LogitLensVisualizer.plot_final_comparison_heatmap(
        final_deltas,
        output_dir / "final_preferences_heatmap.png"
    )
    print("   ✓ Heatmap complete")

    # Save statistics
    print("\n4. Saving statistics...")
    stats_df = pd.DataFrame(aggregate_stats)
    stats_path = output_dir / "decision_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"   ✓ Aggregate statistics saved to: {stats_path}")

    variant_stats_df = pd.DataFrame(decision_stats)
    variant_stats_path = output_dir / "decision_statistics_by_variant.csv"
    variant_stats_df.to_csv(variant_stats_path, index=False)
    print(f"   ✓ Per-variant statistics saved to: {variant_stats_path}")

    # Save raw trajectories
    trajectories_path = output_dir / "trajectories.json"
    trajectories_serializable = {}
    for scenario, models in all_trajectories.items():
        trajectories_serializable[scenario] = {}
        for model_id, traj in models.items():
            trajectories_serializable[scenario][model_id] = traj.tolist()

    with open(trajectories_path, 'w') as f:
        json.dump(trajectories_serializable, f, indent=2)
    print(f"   ✓ Trajectories saved to: {trajectories_path}")

    variant_trajectories_path = output_dir / "trajectories_by_variant.json"
    variant_trajectories_serializable = {}
    for scenario, models in all_variant_trajectories.items():
        variant_trajectories_serializable[scenario] = {}
        for model_id, variants in models.items():
            variant_trajectories_serializable[scenario][model_id] = {
                variant_id: traj.tolist() for variant_id, traj in variants.items()
            }

    with open(variant_trajectories_path, 'w') as f:
        json.dump(variant_trajectories_serializable, f, indent=2)
    print(f"   ✓ Variant trajectories saved to: {variant_trajectories_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    # Group by scenario and show statistics
    summary = stats_df.groupby('scenario').agg({
        'final_delta_mean': ['mean', 'std', 'min', 'max']
    }).round(3)

    print("Final Δ Logit by Scenario (across all models):")
    print(summary)
    print()

    # Group by model and show statistics
    model_summary = stats_df.groupby('model_name').agg({
        'final_delta_mean': ['mean', 'std']
    }).round(3)

    print("\nFinal Δ Logit by Model (across all scenarios):")
    print(model_summary)

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  • comparison_*.png - Per-scenario model comparisons (mean over variants)")
    print(f"  • all_scenarios_grid.png - Grid of all scenarios (mean over variants)")
    print(f"  • final_preferences_heatmap.png - Heatmap across models/scenarios")
    print(f"  • decision_statistics.csv - Aggregate statistics")
    print(f"  • decision_statistics_by_variant.csv - Per-variant statistics")
    print(f"  • trajectories.json - Mean trajectories")
    print(f"  • trajectories_by_variant.json - Raw per-variant trajectories")
    print()


def main():
    """Run full analysis."""
    run_full_analysis()


if __name__ == "__main__":
    main()
