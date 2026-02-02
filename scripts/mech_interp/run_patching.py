"""
Run Activation Patching analysis to discover causal circuits.

This script performs key patching experiments:
1. PT2 → PT3_De: Patch strategic into deontological (find "selfishness circuit")
2. PT2 → PT3_Ut: Patch strategic into utilitarian
3. PT3_De → PT3_Ut: Cross-patch deontological into utilitarian
4. PT3_Ut → PT3_De: Cross-patch utilitarian into deontological

For each experiment, we:
- Systematically patch all components (heads + MLPs)
- Identify components that cause behavioral flips
- Discover minimal circuits
- Generate visualizations and export results
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.activation_patching import (
    ActivationPatcher,
    PatchingVisualizer,
    export_patching_results
)
from mech_interp.utils import get_action_token_ids, MODEL_LABELS, MODEL_COLORS
from mech_interp.prompt_generator import load_prompt_dataset


def run_patching_experiment(
    source_model_name: str,
    target_model_name: str,
    prompts: List[Dict],
    output_dir: str,
    device: str = "cuda",
    discover_circuits: bool = True
) -> Dict:
    """
    Run patching experiment: source → target.

    Args:
        source_model_name: Model to take activations from
        target_model_name: Model to patch activations into
        prompts: List of prompt dictionaries
        output_dir: Directory to save results
        device: Device to run on
        discover_circuits: Whether to run circuit discovery (slow)

    Returns:
        Dictionary with results and statistics
    """
    experiment_name = f"{source_model_name}_to_{target_model_name}"

    print(f"\n{'='*60}")
    print(f"Patching Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"  Source: {MODEL_LABELS.get(source_model_name, source_model_name)}")
    print(f"  Target: {MODEL_LABELS.get(target_model_name, target_model_name)}")

    # Load models
    print("\nLoading models...")
    source_model = LoRAModelLoader.load_hooked_model(source_model_name, device=device)
    source_model.name = source_model_name

    target_model = LoRAModelLoader.load_hooked_model(target_model_name, device=device)
    target_model.name = target_model_name

    # Get action tokens
    action_tokens = get_action_token_ids(target_model.tokenizer)

    # Create patcher
    patcher = ActivationPatcher(
        source_model,
        target_model,
        action_tokens,
        device=device
    )

    # Run patching for each prompt
    all_results = []
    circuit_discoveries = []

    for i, prompt_data in enumerate(prompts):
        scenario = prompt_data['scenario']
        variant = prompt_data['variant']
        prompt_text = prompt_data['prompt']

        print(f"\n[{i+1}/{len(prompts)}] {scenario} (variant {variant})")

        # Get baseline
        baseline_delta, baseline_action = patcher.get_baseline_behavior(prompt_text)
        print(f"  Baseline: {baseline_action} (Δ={baseline_delta:.3f})")

        # Systematic patching
        print(f"  Patching all components...")
        results = patcher.systematic_patch(prompt_text, scenario)

        # Count flips
        n_flips = sum(r.action_flipped for r in results)
        print(f"  Components that flip action: {n_flips}/{len(results)}")

        # Top effects
        top_5 = sorted(results, key=lambda x: abs(x.delta_change), reverse=True)[:5]
        print(f"  Top component: {top_5[0].patched_component} (Δ change={top_5[0].delta_change:.3f})")

        all_results.extend(results)

        # Circuit discovery (optional, slow)
        if discover_circuits:
            print(f"  Discovering minimal circuit...")
            discovery = patcher.discover_minimal_circuit(prompt_text, scenario, max_components=10)
            print(f"  Minimal circuit: {discovery.minimal_circuit} ({len(discovery.minimal_circuit)} components)")
            circuit_discoveries.append({
                'scenario': scenario,
                'variant': variant,
                'minimal_circuit': discovery.minimal_circuit,
                'discovery': discovery
            })

    # Cleanup models
    del source_model, target_model
    torch.cuda.empty_cache()

    return {
        'experiment_name': experiment_name,
        'all_results': all_results,
        'circuit_discoveries': circuit_discoveries,
        'source_model': source_model_name,
        'target_model': target_model_name
    }


def generate_visualizations(
    experiment_results: Dict,
    output_dir: str
):
    """
    Generate visualizations for patching experiment.

    Args:
        experiment_results: Results from run_patching_experiment
        output_dir: Directory to save plots
    """
    experiment_name = experiment_results['experiment_name']
    all_results = experiment_results['all_results']
    circuit_discoveries = experiment_results['circuit_discoveries']

    print(f"\n{'='*60}")
    print(f"Generating Visualizations: {experiment_name}")
    print(f"{'='*60}")

    visualizer = PatchingVisualizer()

    # Group results by scenario
    scenarios = sorted(set(r.scenario for r in all_results))

    # 1. Heatmaps for each scenario
    print("\n1. Creating patch effect heatmaps...")
    for scenario in scenarios:
        scenario_results = [r for r in all_results if r.scenario == scenario]

        if not scenario_results:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Delta change heatmap
        visualizer.plot_patch_heatmap(
            scenario_results,
            metric="delta_change",
            ax=axes[0],
            title=f"{scenario}: Δ Logit Change"
        )

        # Effect size heatmap
        visualizer.plot_patch_heatmap(
            scenario_results,
            metric="effect_size",
            ax=axes[1],
            title=f"{scenario}: Effect Size"
        )

        fig.suptitle(f"{experiment_name}: {scenario}", fontsize=14, y=0.98)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"patch_heatmap_{experiment_name}_{scenario}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    # 2. Top components for each scenario
    print("\n2. Creating top component plots...")
    for scenario in scenarios:
        scenario_results = [r for r in all_results if r.scenario == scenario]

        if not scenario_results:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))

        visualizer.plot_top_components(
            scenario_results,
            top_k=30,
            ax=ax,
            title=f"{experiment_name}: {scenario}"
        )

        save_path = os.path.join(output_dir, f"patch_top_components_{experiment_name}_{scenario}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    # 3. Circuit discovery visualizations
    if circuit_discoveries:
        print("\n3. Creating circuit discovery plots...")
        for disc_data in circuit_discoveries:
            scenario = disc_data['scenario']
            variant = disc_data['variant']
            discovery = disc_data['discovery']

            fig, ax = plt.subplots(figsize=(12, 8))

            visualizer.plot_circuit_discovery(
                discovery,
                ax=ax,
                title=f"{experiment_name}: {scenario} (v{variant})"
            )

            save_path = os.path.join(
                output_dir,
                f"circuit_discovery_{experiment_name}_{scenario}_v{variant}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
            plt.close()

    # 4. Overall summary: components that consistently flip behavior
    print("\n4. Creating consistency analysis...")

    # Count how many scenarios each component flips
    component_flip_counts = {}
    for r in all_results:
        if r.action_flipped:
            comp = r.patched_component
            component_flip_counts[comp] = component_flip_counts.get(comp, 0) + 1

    if component_flip_counts:
        # Sort by consistency
        sorted_comps = sorted(
            component_flip_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:30]

        fig, ax = plt.subplots(figsize=(10, 10))

        components = [c[0] for c in sorted_comps]
        counts = [c[1] for c in sorted_comps]

        y_pos = np.arange(len(components))
        ax.barh(y_pos, counts, color='#d62728', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.set_xlabel('Number of Scenarios Flipped')
        ax.set_title(f"{experiment_name}: Components by Flip Consistency")
        ax.grid(True, alpha=0.3, axis='x')

        save_path = os.path.join(output_dir, f"patch_consistency_{experiment_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    print("\nVisualization complete!")


def export_results(
    experiment_results: Dict,
    output_dir: str
):
    """
    Export patching results to CSV files.

    Args:
        experiment_results: Results from run_patching_experiment
        output_dir: Directory to save CSV files
    """
    experiment_name = experiment_results['experiment_name']
    all_results = experiment_results['all_results']
    circuit_discoveries = experiment_results['circuit_discoveries']

    print(f"\n{'='*60}")
    print(f"Exporting Results: {experiment_name}")
    print(f"{'='*60}")

    # Export full results
    csv_path = os.path.join(output_dir, f"patch_results_{experiment_name}.csv")
    df = export_patching_results(all_results, csv_path)

    # Create summary statistics
    summary_stats = df.groupby(['scenario', 'component']).agg({
        'delta_change': ['mean', 'std'],
        'effect_size': ['mean', 'std'],
        'action_flipped': 'sum'
    }).reset_index()

    summary_path = os.path.join(output_dir, f"patch_summary_{experiment_name}.csv")
    summary_stats.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Export circuit discoveries
    if circuit_discoveries:
        circuits_data = []
        for disc_data in circuit_discoveries:
            circuits_data.append({
                'scenario': disc_data['scenario'],
                'variant': disc_data['variant'],
                'minimal_circuit': json.dumps(disc_data['minimal_circuit']),
                'circuit_size': len(disc_data['minimal_circuit'])
            })

        circuits_df = pd.DataFrame(circuits_data)
        circuits_path = os.path.join(output_dir, f"circuits_{experiment_name}.csv")
        circuits_df.to_csv(circuits_path, index=False)
        print(f"  Saved: {circuits_path}")

    # Identify top components across all scenarios
    top_components = df.groupby('component').agg({
        'delta_change': 'mean',
        'effect_size': 'mean',
        'action_flipped': 'sum'
    }).reset_index()
    top_components = top_components.sort_values('effect_size', ascending=False)

    top_path = os.path.join(output_dir, f"top_components_{experiment_name}.csv")
    top_components.to_csv(top_path, index=False)
    print(f"  Saved: {top_path}")

    print("\nExport complete!")


def run_full_analysis(
    experiments: List[Tuple[str, str]] = None,
    scenarios_to_analyze: List[str] = None,
    output_dir: str = str(PROJECT_ROOT / "mech_interp_outputs" / "patching"),
    device: str = "cuda",
    discover_circuits: bool = True
):
    """
    Run complete activation patching analysis.

    Args:
        experiments: List of (source, target) model pairs
        scenarios_to_analyze: List of scenario names (None = all)
        output_dir: Output directory
        device: Device to run on
        discover_circuits: Whether to run circuit discovery (slow)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    print("Loading prompt dataset...")
    prompts = load_prompt_dataset(
        str(PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json")
    )
    print(f"  Loaded {len(prompts)} prompts")

    # Filter scenarios if specified
    if scenarios_to_analyze:
        prompts = [p for p in prompts if p['scenario'] in scenarios_to_analyze]
        print(f"  Filtered to {len(prompts)} prompts for scenarios: {scenarios_to_analyze}")

    # Define experiments if not provided
    if experiments is None:
        experiments = [
            ("PT2_COREDe", "PT3_COREDe"),  # Strategic → Deontological
            ("PT2_COREDe", "PT3_COREUt"),  # Strategic → Utilitarian
            ("PT3_COREDe", "PT3_COREUt"),  # Deontological → Utilitarian
            ("PT3_COREUt", "PT3_COREDe"),  # Utilitarian → Deontological
        ]

    # Run each experiment
    all_experiment_results = []

    for source, target in experiments:
        results = run_patching_experiment(
            source,
            target,
            prompts,
            output_dir,
            device=device,
            discover_circuits=discover_circuits
        )

        # Generate visualizations
        generate_visualizations(results, output_dir)

        # Export results
        export_results(results, output_dir)

        all_experiment_results.append(results)

    # Create cross-experiment summary
    print(f"\n{'='*60}")
    print("Creating Cross-Experiment Summary")
    print(f"{'='*60}")

    # Compare minimal circuits across experiments
    if discover_circuits:
        summary_data = []
        for exp_results in all_experiment_results:
            exp_name = exp_results['experiment_name']
            for disc_data in exp_results['circuit_discoveries']:
                summary_data.append({
                    'experiment': exp_name,
                    'scenario': disc_data['scenario'],
                    'circuit_size': len(disc_data['minimal_circuit']),
                    'circuit': json.dumps(disc_data['minimal_circuit'])
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, "cross_experiment_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\n  Saved: {summary_path}")

    print(f"\n{'='*60}")
    print("Activation Patching Analysis Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - patch_heatmap_<experiment>_<scenario>.png: Patching effect heatmaps")
    print("  - patch_top_components_<experiment>_<scenario>.png: Top components")
    print("  - circuit_discovery_<experiment>_<scenario>.png: Minimal circuits")
    print("  - patch_consistency_<experiment>.png: Consistent flip components")
    print("  - patch_results_<experiment>.csv: Full patching data")
    print("  - patch_summary_<experiment>.csv: Statistical summary")
    print("  - circuits_<experiment>.csv: Discovered circuits")
    print("  - top_components_<experiment>.csv: Top components ranked")
    print("  - cross_experiment_summary.csv: Cross-experiment comparison")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Activation Patching analysis")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenarios to analyze (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "mech_interp_outputs" / "patching"),
        help="Output directory"
    )
    parser.add_argument(
        "--no-circuits",
        action="store_true",
        help="Skip circuit discovery (faster)"
    )

    args = parser.parse_args()

    run_full_analysis(
        scenarios_to_analyze=args.scenarios,
        output_dir=args.output_dir,
        discover_circuits=not args.no_circuits
    )
