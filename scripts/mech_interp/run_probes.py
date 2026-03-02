#!/usr/bin/env python3
"""
Runner script for linear probe analysis.

Trains linear probes on residual stream activations to detect:
1. Betrayal concepts (CD/DC scenarios)
2. Joint payoff representations (cooperative outcomes)

This maps the "Geography of Judgment" across model layers.

Expected findings:
- Deontological models: High betrayal accuracy early (L10-15)
- Utilitarian models: High payoff R² mid-layers (L12-18)
- Strategic models: Lower probe performance overall

Usage:
    python scripts/mech_interp/run_probes.py

Output:
    mech_interp_outputs/linear_probes/
        probe_results_{model_id}.csv       - Detailed probe results per model
        probe_trajectories_{model_id}.png  - Performance vs layer plots
        betrayal_probe_comparison.png      - Cross-model betrayal comparison
        payoff_probe_comparison.png        - Cross-model payoff comparison
        peak_layer_heatmap.png             - Peak performance layers
        probe_summary.csv                  - Summary statistics
"""

import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    p for p in _THIS_FILE.parents if (p / "mech_interp" / "__init__.py").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.utils import load_prompt_dataset, MODEL_LABELS
from mech_interp.model_loader import LoRAModelLoader
from mech_interp.linear_probes import (
    LabelGenerator,
    ActivationExtractor,
    LinearProbeTrainer,
    LinearProbeVisualizer,
    export_results,
    compute_summary,
)


def main():
    print("\n" + "=" * 70)
    print("LINEAR PROBE ANALYSIS")
    print("=" * 70 + "\n")

    # Configuration
    output_dir = Path("/root/LLM_morality/mech_interp_outputs/linear_probes")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Models to analyze
    model_ids = ["base", "PT2_COREDe", "PT3_COREDe", "PT3_COREUt", "PT4_COREDe"]
    print(f"Models to analyze: {', '.join(model_ids)}\n")

    # Load prompts and generate labels
    print("Loading prompts and generating labels...")
    prompts_data = load_prompt_dataset()
    print(f"Loaded {len(prompts_data)} prompts")

    labels = LabelGenerator.generate_labels(prompts_data)
    print(f"Generated {len(labels)} labels")

    # Validate label distribution
    n_betrayal = sum(1 for label in labels if label.is_betrayal)
    n_non_betrayal = len(labels) - n_betrayal
    print(f"  Betrayal scenarios: {n_betrayal} ({n_betrayal/len(labels)*100:.1f}%)")
    print(f"  Non-betrayal scenarios: {n_non_betrayal} ({n_non_betrayal/len(labels)*100:.1f}%)")
    print()

    # Initialize trainer and visualizer
    trainer = LinearProbeTrainer(random_state=42)
    visualizer = LinearProbeVisualizer(output_dir)

    # Store all results for cross-model comparison
    all_results = []

    # Process each model
    for model_id in model_ids:
        model_name = MODEL_LABELS.get(model_id, model_id)
        print("-" * 70)
        print(f"Processing: {model_name} ({model_id})")
        print("-" * 70)

        try:
            # Load model with 4-bit quantization for memory efficiency
            print(f"\nLoading model: {model_id}...")
            hooked_model = LoRAModelLoader.load_hooked_model(
                model_id,
                use_4bit=True,
                attn_implementation="eager",
            )
            print(f"Model loaded: {hooked_model.n_layers} layers, {hooked_model.d_model} dim\n")

            # Extract activations
            print("Extracting activations from residual stream...")
            extractor = ActivationExtractor(hooked_model, use_chat_template=True)
            prompts = [p["prompt"] for p in prompts_data]

            activations = extractor.extract_layer_activations(prompts)
            print(f"Extracted activations: {activations.shape}")
            print(f"  Shape: (n_prompts={activations.shape[0]}, n_layers={activations.shape[1]}, d_model={activations.shape[2]})\n")

            # Train probes for all layers
            print("Training linear probes for all layers...")
            results = trainer.train_all_layers(model_id, activations, labels)
            print(f"Trained {len(results)} probes (26 layers × 2 types)\n")

            # Save per-model results
            output_path = output_dir / f"probe_results_{model_id}.csv"
            export_results(results, output_path)
            print(f"Saved results: {output_path}")

            # Generate per-model trajectory plot
            visualizer.plot_probe_trajectories(results, model_id, model_name)
            print(f"Generated trajectory plot: probe_trajectories_{model_id}.png\n")

            # Store for cross-model comparison
            all_results.extend(results)

            # Clean up GPU memory
            del hooked_model
            torch.cuda.empty_cache()
            print(f"Completed {model_id}\n")

        except Exception as e:
            print(f"ERROR processing {model_id}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Cross-model visualizations
    print("-" * 70)
    print("Generating cross-model comparisons...")
    print("-" * 70)

    if all_results:
        # Betrayal probe comparison
        print("\nGenerating betrayal probe comparison...")
        visualizer.plot_model_comparison(
            all_results,
            probe_type="betrayal",
            title_suffix="Betrayal Detection Across Models",
        )
        print("Saved: betrayal_probe_comparison.png")

        # Payoff probe comparison
        print("\nGenerating payoff probe comparison...")
        visualizer.plot_model_comparison(
            all_results,
            probe_type="payoff",
            title_suffix="Joint Payoff Prediction Across Models",
        )
        print("Saved: payoff_probe_comparison.png")

        # Compute and save summary statistics
        print("\nComputing summary statistics...")
        summary_df = compute_summary(all_results)
        summary_path = output_dir / "probe_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

        # Peak layer heatmap
        print("\nGenerating peak layer heatmap...")
        visualizer.plot_peak_layer_heatmap(summary_df)
        print("Saved: peak_layer_heatmap.png")

        # Print summary to console
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print()

    else:
        print("WARNING: No results to visualize")

    print("\n" + "=" * 70)
    print("LINEAR PROBE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - probe_results_{model_id}.csv (5 files)")
    print("  - probe_trajectories_{model_id}.png (5 files)")
    print("  - betrayal_probe_comparison.png")
    print("  - payoff_probe_comparison.png")
    print("  - peak_layer_heatmap.png")
    print("  - probe_summary.csv")
    print()


if __name__ == "__main__":
    main()
