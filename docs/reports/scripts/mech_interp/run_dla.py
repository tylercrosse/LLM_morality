"""
Run Direct Logit Attribution (DLA) analysis across all models and scenarios.

This script:
1. Loads all 5 models (base + 4 fine-tuned)
2. Analyzes component-level contributions to action logits
3. Identifies top pro-Defect and pro-Cooperate heads/MLPs
4. Generates heatmaps and comparison visualizations
5. Exports results to CSV for further analysis
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.direct_logit_attribution import DirectLogitAttributor, DLAVisualizer
from mech_interp.utils import get_action_token_ids, MODEL_LABELS, MODEL_COLORS, load_prompt_dataset


def run_dla_for_model(
    model_name: str,
    prompts: List[Dict],
    output_dir: str,
    device: str = "cuda"
) -> List:
    """
    Run DLA analysis for a single model across all prompts.

    Args:
        model_name: Name of model to analyze
        prompts: List of prompt dictionaries
        output_dir: Directory to save results
        device: Device to run on

    Returns:
        List of DLAResult objects
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {MODEL_LABELS.get(model_name, model_name)}")
    print(f"{'='*60}")

    # Load model
    hooked_model = LoRAModelLoader.load_hooked_model(
        model_name,
        device=device,
        merge_lora=False,
        use_4bit=True,
    )
    hooked_model.name = model_name

    # Get action tokens
    action_tokens = get_action_token_ids(hooked_model.tokenizer)

    # Create attributor
    attributor = DirectLogitAttributor(hooked_model, action_tokens)

    # Analyze each prompt
    results = []
    for i, prompt_data in enumerate(prompts):
        scenario = prompt_data['scenario']
        variant = prompt_data['variant']
        prompt_text = prompt_data['prompt']

        print(f"\n  [{i+1}/{len(prompts)}] {scenario} (variant {variant})")

        # Run DLA
        result = attributor.decompose_logits(prompt_text)
        result.scenario = scenario
        result.model_name = model_name

        # Print summary
        print(
            f"    Final Δseq(a2-a1): {result.final_delta:.3f} "
            f"(pref={result.final_preferred_action}, p_action2={result.final_p_action2:.3f})"
        )
        print(f"    Legacy final Δ(Dtoken-Ctoken): {result.final_delta_legacy:.3f}")

        # Get top components
        top_comps = attributor.identify_top_components(result, top_k=5)
        print(f"    Top pro-Defect: {top_comps['pro_defect'][0][0]} ({top_comps['pro_defect'][0][1]:.3f})")
        print(f"    Top pro-Coop:   {top_comps['pro_cooperate'][0][0]} ({top_comps['pro_cooperate'][0][1]:.3f})")

        results.append(result)

    # Cleanup
    del hooked_model
    torch.cuda.empty_cache()

    return results


def generate_visualizations(
    all_results: Dict[str, List],
    scenarios: List[str],
    output_dir: str
):
    """
    Generate comparison visualizations across models and scenarios.

    Args:
        all_results: Dictionary mapping model_name -> list of DLAResults
        scenarios: List of scenario names
        output_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")

    # Create visualizer
    dummy_tokens = {'C': 0, 'D': 1}  # Tokens not needed for plotting
    visualizer = DLAVisualizer(dummy_tokens)

    # 1. Head heatmaps for each scenario (comparing all models)
    print("\n1. Creating head heatmap comparisons...")
    for scenario in scenarios:
        # Get results for this scenario from all models
        scenario_results = []
        for model_name in all_results.keys():
            model_results = all_results[model_name]
            # Find first result matching this scenario
            for result in model_results:
                if result.scenario == scenario:
                    scenario_results.append(result)
                    break

        if scenario_results:
            save_path = os.path.join(output_dir, f"dla_heads_{scenario}.png")
            visualizer.plot_model_comparison(
                scenario_results,
                MODEL_LABELS,
                scenario,
                save_path=save_path
            )

    # 2. MLP contributions comparison
    print("\n2. Creating MLP contribution plots...")
    for scenario in scenarios:
        fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4))
        if len(all_results) == 1:
            axes = [axes]

        for idx, (model_name, model_results) in enumerate(all_results.items()):
            # Find result for this scenario
            result = None
            for r in model_results:
                if r.scenario == scenario:
                    result = r
                    break

            if result:
                label = MODEL_LABELS.get(model_name, model_name)
                visualizer.plot_mlp_contributions(
                    result,
                    ax=axes[idx],
                    title=f"{label}\nΔseq={result.final_delta:.2f}"
                )

        fig.suptitle(f"MLP Contributions: {scenario}", fontsize=14, y=0.98)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"dla_mlps_{scenario}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    # 3. Top components for each model
    print("\n3. Creating top component plots...")
    for model_name, model_results in all_results.items():
        # Use first scenario as representative
        if model_results:
            result = model_results[0]
            fig, ax = plt.subplots(figsize=(12, 10))
            label = MODEL_LABELS.get(model_name, model_name)

            visualizer.plot_top_components(
                result,
                top_k=20,
                ax=ax,
                title=f"Top-20 Components: {label} ({result.scenario})"
            )

            save_path = os.path.join(output_dir, f"dla_top_components_{model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
            plt.close()

    print("\nVisualization complete!")


def export_results(
    all_results: Dict[str, List],
    output_dir: str
):
    """
    Export DLA results to CSV files.

    Args:
        all_results: Dictionary mapping model_name -> list of DLAResults
        output_dir: Directory to save CSV files
    """
    print(f"\n{'='*60}")
    print("Exporting Results")
    print(f"{'='*60}")

    # Combine all results into single DataFrame
    all_dfs = []

    for model_name, model_results in all_results.items():
        for result in model_results:
            rows = []
            for layer_idx in range(result.head_contributions.shape[0]):
                for head_idx in range(result.head_contributions.shape[1]):
                    rows.append({
                        'model': result.model_name,
                        'scenario': result.scenario,
                        'final_delta_seq': result.final_delta,
                        'final_delta_legacy': result.final_delta_legacy,
                        'final_p_action2': result.final_p_action2,
                        'final_preferred_action': result.final_preferred_action,
                        'component_type': 'head',
                        'layer': layer_idx,
                        'head': head_idx,
                        'component': f"L{layer_idx}H{head_idx}",
                        'contribution': result.head_contributions[layer_idx, head_idx]
                    })
            for layer_idx in range(result.mlp_contributions.shape[0]):
                rows.append({
                    'model': result.model_name,
                    'scenario': result.scenario,
                    'final_delta_seq': result.final_delta,
                    'final_delta_legacy': result.final_delta_legacy,
                    'final_p_action2': result.final_p_action2,
                    'final_preferred_action': result.final_preferred_action,
                    'component_type': 'mlp',
                    'layer': layer_idx,
                    'head': -1,
                    'component': f"L{layer_idx}_MLP",
                    'contribution': result.mlp_contributions[layer_idx]
                })
            all_dfs.append(pd.DataFrame(rows))

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save full results
    csv_path = os.path.join(output_dir, "dla_full_results.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Total rows: {len(combined_df)}")

    # Create summary statistics
    summary = combined_df.groupby(['model', 'scenario', 'component_type'])['contribution'].agg([
        'mean', 'std', 'min', 'max'
    ]).reset_index()

    summary_path = os.path.join(output_dir, "dla_summary_stats.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Identify top components across all scenarios
    top_components = combined_df.groupby(['model', 'component'])['contribution'].mean().reset_index()
    top_components = top_components.sort_values('contribution', ascending=False)

    top_path = os.path.join(output_dir, "dla_top_components.csv")
    top_components.to_csv(top_path, index=False)
    print(f"  Saved: {top_path}")

    print("\nExport complete!")


def run_full_analysis(
    models_to_analyze: List[str] = None,
    scenarios_to_analyze: List[str] = None,
    output_dir: str = str(PROJECT_ROOT / "mech_interp_outputs" / "dla"),
    device: str = "cuda"
):
    """
    Run complete DLA analysis pipeline.

    Args:
        models_to_analyze: List of model names (None = all models)
        scenarios_to_analyze: List of scenario names (None = all scenarios)
        output_dir: Output directory
        device: Device to run on
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

    # Get unique scenarios
    scenarios = sorted(set(p['scenario'] for p in prompts))
    print(f"  Scenarios: {scenarios}")

    # Define models to analyze
    if models_to_analyze is None:
        models_to_analyze = ['base', 'PT2_COREDe', 'PT3_COREDe', 'PT3_COREUt', 'PT4_COREDe']

    # Run analysis for each model
    all_results = {}

    for model_name in models_to_analyze:
        results = run_dla_for_model(
            model_name,
            prompts,
            output_dir,
            device=device
        )
        all_results[model_name] = results

    # Generate visualizations
    generate_visualizations(all_results, scenarios, output_dir)

    # Export results
    export_results(all_results, output_dir)

    print(f"\n{'='*60}")
    print("DLA Analysis Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - dla_heads_<scenario>.png: Head contribution heatmaps")
    print("  - dla_mlps_<scenario>.png: MLP contribution plots")
    print("  - dla_top_components_<model>.png: Top-20 components per model")
    print("  - dla_full_results.csv: Complete attribution data")
    print("  - dla_summary_stats.csv: Statistical summary")
    print("  - dla_top_components.csv: Top components ranked by contribution")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Direct Logit Attribution analysis")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to analyze (default: all)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenarios to analyze (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "mech_interp_outputs" / "dla"),
        help="Output directory"
    )

    args = parser.parse_args()

    run_full_analysis(
        models_to_analyze=args.models,
        scenarios_to_analyze=args.scenarios,
        output_dir=args.output_dir
    )
