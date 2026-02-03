#!/usr/bin/env python3
"""
Run component interaction analysis for Deontological vs Utilitarian models.

This script analyzes how components (attention heads and MLPs) correlate with each other,
testing the hypothesis that De and Ut models may wire similar components together differently.

Usage:
    python scripts/mech_interp/run_component_interactions.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.component_interactions import (
    run_component_interaction_analysis,
    compare_deontological_vs_utilitarian,
    analyze_key_pathways,
)


def main():
    print("\n" + "="*60)
    print("COMPONENT INTERACTION ANALYSIS")
    print("="*60 + "\n")

    # Models to analyze
    model_ids = ["PT3_COREDe", "PT3_COREUt"]

    # Output directory
    output_dir = str(PROJECT_ROOT / "mech_interp_outputs" / "component_interactions")

    print("This analysis will:")
    print("  1. Extract component activation patterns for all scenarios")
    print("  2. Compute correlation matrices between components")
    print("  3. Identify pathways with different correlation strengths")
    print("  4. Focus on connections involving key components (L8_MLP, L9_MLP, L25 heads)")
    print()

    # Run analysis
    print("Running component interaction analysis...")
    all_correlation_matrices = run_component_interaction_analysis(
        model_ids=model_ids,
        prompt_dataset_path=str(
            PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json"
        ),
        output_dir=output_dir,
        device="cuda"
    )

    print("\n" + "="*60)
    print("COMPARING DEONTOLOGICAL VS UTILITARIAN")
    print("="*60 + "\n")

    # Compare
    comparison_df, significant_pathways = compare_deontological_vs_utilitarian(output_dir)

    print("\n" + "="*60)
    print("ANALYZING KEY PATHWAYS")
    print("="*60 + "\n")

    # Focus on key components from DLA analysis
    key_components = ["L8_MLP", "L9_MLP", "L25H0", "L25H1", "L25H2", "L25H3",
                      "L13H0", "L13H1", "L6_MLP"]

    key_pathways = analyze_key_pathways(comparison_df, key_components)

    # Save key pathways
    key_pathways_file = Path(output_dir) / "key_component_pathways_De_vs_Ut.csv"
    key_pathways.to_csv(key_pathways_file, index=False)
    print(f"\nSaved key pathways to {key_pathways_file}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")

    print(f"Results saved to: {output_dir}/")
    print("\nKey outputs:")
    print(f"  - component_activations_PT3_COREDe.json")
    print(f"  - component_activations_PT3_COREUt.json")
    print(f"  - correlation_matrix_PT3_COREDe.npz")
    print(f"  - correlation_matrix_PT3_COREUt.npz")
    print(f"  - correlation_matrix_PT3_COREDe.png")
    print(f"  - correlation_matrix_PT3_COREUt.png")
    print(f"  - interaction_comparison_De_vs_Ut.csv")
    print(f"  - interaction_diff_Deontological_vs_Utilitarian.png")
    print(f"  - significant_pathways_De_vs_Ut.csv")
    print(f"  - key_component_pathways_De_vs_Ut.csv")


if __name__ == "__main__":
    main()
