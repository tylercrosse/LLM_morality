#!/usr/bin/env python3
"""
Run LoRA weight analysis across all fine-tuned models.

Computes Frobenius norms ||B @ A|| for all LoRA modules to identify which
components were most heavily modified during fine-tuning. This helps test
whether L2_MLP "switching" behavior comes from direct weight modifications
or upstream routing changes.

Usage:
    python scripts/mech_interp/run_weight_analysis.py

Outputs:
    - mech_interp_outputs/weight_analysis/weight_norms_{model_id}.csv
    - mech_interp_outputs/weight_analysis/norm_heatmap_{model_id}.png
    - mech_interp_outputs/weight_analysis/top_components_{model_id}.png
    - mech_interp_outputs/weight_analysis/l2_mlp_comparison.png
    - mech_interp_outputs/weight_analysis/layer_profile_mlp.png
    - mech_interp_outputs/weight_analysis/weight_norms_all_models.csv
    - mech_interp_outputs/weight_analysis/weight_norms_comparison.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from dataclasses import asdict

from mech_interp.weight_analysis import (
    WeightAnalyzer,
    WeightAnalysisVisualizer,
    WeightNormResult,
)
from mech_interp.utils import MODEL_LABELS


def main():
    """Run weight analysis for all models."""

    # Output directory
    output_dir = project_root / "mech_interp_outputs" / "weight_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LoRA Weight Analysis")
    print("=" * 80)

    # Models to analyze
    model_ids = ["PT2_COREDe", "PT3_COREDe", "PT3_COREUt", "PT4_COREDe"]

    # Store all results for cross-model comparisons
    all_results = {}

    # Analyze each model
    for model_id in model_ids:
        print(f"\n{'='*80}")
        print(f"Analyzing: {MODEL_LABELS.get(model_id, model_id)} ({model_id})")
        print(f"{'='*80}\n")

        # Create analyzer
        analyzer = WeightAnalyzer(model_id)

        # Analyze all modules
        print(f"Computing Frobenius norms for 182 modules (26 layers × 7 types)...")
        results = analyzer.analyze_all_modules()
        all_results[model_id] = results

        # Validation
        assert len(results) == 182, f"Expected 182 modules, got {len(results)}"
        print(f"✓ Analyzed {len(results)} modules")

        # Compute L2_MLP statistics
        l2_mlp_results = [
            r for r in results
            if r.layer == 2 and r.module_type.startswith("mlp.")
        ]
        l2_mlp_total_norm = sum(r.frobenius_norm for r in l2_mlp_results)
        l2_mlp_percentile = analyzer.compute_percentile_rank(results, target_layer=2, target_prefix="mlp")

        print(f"\nL2_MLP Statistics:")
        print(f"  Total Frobenius norm: {l2_mlp_total_norm:.3f}")
        print(f"  Percentile rank: {l2_mlp_percentile:.1f}%")
        print(f"  Components: {len(l2_mlp_results)}")

        # Show top-5 components
        sorted_results = sorted(results, key=lambda r: r.frobenius_norm, reverse=True)
        print(f"\nTop 5 Components:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r.component_name}: {r.frobenius_norm:.3f}")

        # Save detailed results to CSV
        df = pd.DataFrame([asdict(r) for r in results])
        csv_path = output_dir / f"weight_norms_{model_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved detailed results: {csv_path}")

        # Generate visualizations
        print(f"\nGenerating visualizations...")
        visualizer = WeightAnalysisVisualizer(output_dir)

        # Heatmap (normalized)
        visualizer.plot_norm_heatmap(results, model_id, use_normalized=True)

        # Heatmap (raw)
        visualizer.plot_norm_heatmap(results, model_id, use_normalized=False)

        # Top components (highlight L2_MLP in red)
        visualizer.plot_top_components(results, model_id, top_n=30, highlight_layer=2)

    # Cross-model comparisons
    print(f"\n{'='*80}")
    print("Cross-Model Comparisons")
    print(f"{'='*80}\n")

    visualizer = WeightAnalysisVisualizer(output_dir)

    # L2_MLP comparison
    print("Generating L2_MLP comparison...")
    visualizer.plot_l2_mlp_comparison(all_results, target_layer=2)

    # Layer profiles
    print("Generating MLP layer profiles...")
    visualizer.plot_layer_profiles(all_results, module_prefix="mlp")

    print("Generating Attention layer profiles...")
    visualizer.plot_layer_profiles(all_results, module_prefix="self_attn")

    # Combine all results into single CSV
    print("\nCombining results...")
    all_dfs = []
    for model_id, results in all_results.items():
        df = pd.DataFrame([asdict(r) for r in results])
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv_path = output_dir / "weight_norms_all_models.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Saved combined results: {combined_csv_path}")

    # Create comparison summary
    print("\nCreating comparison summary...")
    comparison_rows = []

    for model_id, results in all_results.items():
        # Aggregate statistics
        all_norms = [r.frobenius_norm for r in results]
        mlp_results = [r for r in results if r.module_type.startswith("mlp.")]
        attn_results = [r for r in results if r.module_type.startswith("self_attn.")]

        # L2_MLP specific
        l2_mlp_results = [
            r for r in results
            if r.layer == 2 and r.module_type.startswith("mlp.")
        ]
        l2_mlp_total = sum(r.frobenius_norm for r in l2_mlp_results)

        # Early layers (0-8), middle (9-17), late (18-25)
        early_results = [r for r in mlp_results if r.layer < 9]
        middle_results = [r for r in mlp_results if 9 <= r.layer < 18]
        late_results = [r for r in mlp_results if r.layer >= 18]

        analyzer = WeightAnalyzer(model_id)
        l2_mlp_percentile = analyzer.compute_percentile_rank(results, target_layer=2, target_prefix="mlp")

        comparison_rows.append({
            'model_id': model_id,
            'model_name': MODEL_LABELS.get(model_id, model_id),
            'total_modules': len(results),
            'mean_norm': np.mean(all_norms),
            'median_norm': np.median(all_norms),
            'max_norm': np.max(all_norms),
            'mlp_mean_norm': np.mean([r.frobenius_norm for r in mlp_results]),
            'attn_mean_norm': np.mean([r.frobenius_norm for r in attn_results]),
            'l2_mlp_total_norm': l2_mlp_total,
            'l2_mlp_percentile': l2_mlp_percentile,
            'early_mlp_mean': np.mean([r.frobenius_norm for r in early_results]),
            'middle_mlp_mean': np.mean([r.frobenius_norm for r in middle_results]),
            'late_mlp_mean': np.mean([r.frobenius_norm for r in late_results]),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = output_dir / "weight_norms_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"✓ Saved comparison summary: {comparison_csv_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(comparison_df.to_string(index=False))

    # Key findings
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['model_name']}:")
        print(f"  L2_MLP total norm: {row['l2_mlp_total_norm']:.3f}")
        print(f"  L2_MLP percentile: {row['l2_mlp_percentile']:.1f}%")
        print(f"  MLP vs Attn mean: {row['mlp_mean_norm']:.3f} vs {row['attn_mean_norm']:.3f}")
        print(f"  Layer distribution: Early={row['early_mlp_mean']:.3f}, Mid={row['middle_mlp_mean']:.3f}, Late={row['late_mlp_mean']:.3f}")

        # Interpretation
        if row['l2_mlp_percentile'] > 90:
            print(f"  → L2_MLP heavily modified (>90th percentile) - supports 'switching' hypothesis")
        elif row['l2_mlp_percentile'] < 50:
            print(f"  → L2_MLP lightly modified (<50th percentile) - refutes 'switching' hypothesis")
        else:
            print(f"  → L2_MLP moderately modified (50-90th percentile)")

    print("\n" + "="*80)
    print("Weight Analysis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total files created: {len(list(output_dir.glob('*')))}")


if __name__ == "__main__":
    main()
