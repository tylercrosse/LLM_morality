#!/usr/bin/env python3
"""
Run attention pattern analysis for Deontological vs Utilitarian models.

This script analyzes what tokens each model attends to, testing the hypothesis
that Deontological models focus more on opponent actions while Utilitarian
models focus more on payoff information.

Usage:
    python scripts/mech_interp/run_attention_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.attention_analysis import (
    run_attention_analysis,
    compare_deontological_vs_utilitarian,
)


def main():
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60 + "\n")

    # Models to analyze
    model_ids = ["PT3_COREDe", "PT3_COREUt"]

    # Output directory
    output_dir = str(PROJECT_ROOT / "mech_interp_outputs" / "attention_analysis")

    print("This analysis will:")
    print("  1. Extract attention patterns for all scenarios")
    print("  2. Identify which tokens each model attends to")
    print("  3. Compare Deontological vs Utilitarian attention strategies")
    print()

    # Run analysis
    print("Running attention analysis...")
    all_results = run_attention_analysis(
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
    comparison_df = compare_deontological_vs_utilitarian(output_dir)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")

    print(f"Results saved to: {output_dir}/")
    print("\nKey outputs:")
    print(f"  - attention_results_PT3_COREDe.json")
    print(f"  - attention_results_PT3_COREUt.json")
    print(f"  - attention_summary_PT3_COREDe.csv")
    print(f"  - attention_summary_PT3_COREUt.csv")
    print(f"  - attention_comparison_De_vs_Ut.csv")
    print(f"  - attention_comparison_Deontological_vs_Utilitarian.png")


if __name__ == "__main__":
    main()
