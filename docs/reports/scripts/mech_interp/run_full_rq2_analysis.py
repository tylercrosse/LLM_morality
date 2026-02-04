#!/usr/bin/env python3
"""
Run complete RQ2 deep-dive analysis: Attention patterns + Component interactions.

This script executes both analyses in sequence to investigate how Deontological
and Utilitarian models produce different behaviors despite nearly identical
component-level representations.

Usage:
    python scripts/mech_interp/run_full_rq2_analysis.py
"""

import sys
from pathlib import Path

# Add project root to path
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    p for p in _THIS_FILE.parents if (p / "mech_interp" / "__init__.py").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.attention_analysis import (
    run_attention_analysis,
    compare_deontological_vs_utilitarian as compare_attention,
)
from mech_interp.component_interactions import (
    run_component_interaction_analysis,
    compare_deontological_vs_utilitarian as compare_interactions,
    analyze_key_pathways,
)


def main():
    print("\n" + "="*80)
    print("COMPLETE RQ2 DEEP-DIVE ANALYSIS")
    print("="*80 + "\n")

    print("This script will run two analyses to investigate:")
    print("  1. ATTENTION PATTERNS: What information do De vs Ut models attend to?")
    print("  2. COMPONENT INTERACTIONS: How do models wire components together?")
    print()
    print("Expected runtime: ~10 minutes total")
    print()

    # Models to analyze
    model_ids = ["PT3_COREDe", "PT3_COREUt"]

    # ============================================================================
    # PART 1: ATTENTION ANALYSIS
    # ============================================================================

    print("\n" + "="*80)
    print("PART 1: ATTENTION PATTERN ANALYSIS")
    print("="*80 + "\n")

    attention_dir = str(PROJECT_ROOT / "mech_interp_outputs" / "attention_analysis")

    print("Running attention analysis...")
    print("  - Extracting attention weights from all layers and heads")
    print("  - Identifying what tokens each model attends to")
    print("  - Comparing Deontological vs Utilitarian attention strategies")
    print()

    attention_results = run_attention_analysis(
        model_ids=model_ids,
        prompt_dataset_path=str(
            PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json"
        ),
        output_dir=attention_dir,
        device="cuda"
    )

    print("\nComparing attention patterns...")
    attention_comparison = compare_attention(attention_dir)

    print("\n✓ Attention analysis complete!")
    print(f"  Results saved to: {attention_dir}/")

    # ============================================================================
    # PART 2: COMPONENT INTERACTION ANALYSIS
    # ============================================================================

    print("\n" + "="*80)
    print("PART 2: COMPONENT INTERACTION ANALYSIS")
    print("="*80 + "\n")

    interaction_dir = str(PROJECT_ROOT / "mech_interp_outputs" / "component_interactions")

    print("Running component interaction analysis...")
    print("  - Extracting component activation patterns")
    print("  - Computing correlation matrices between components")
    print("  - Identifying pathways with different correlation strengths")
    print()

    interaction_results = run_component_interaction_analysis(
        model_ids=model_ids,
        prompt_dataset_path=str(
            PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json"
        ),
        output_dir=interaction_dir,
        device="cuda"
    )

    print("\nComparing interaction patterns...")
    interaction_comparison, significant_pathways = compare_interactions(interaction_dir)

    print("\nAnalyzing key component pathways...")
    key_components = ["L8_MLP", "L9_MLP", "L25H0", "L25H1", "L25H2", "L25H3",
                      "L13H0", "L13H1", "L6_MLP"]
    key_pathways = analyze_key_pathways(interaction_comparison, key_components)

    # Save key pathways
    key_pathways_file = Path(interaction_dir) / "key_component_pathways_De_vs_Ut.csv"
    key_pathways.to_csv(key_pathways_file, index=False)

    print("\n✓ Component interaction analysis complete!")
    print(f"  Results saved to: {interaction_dir}/")

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80 + "\n")

    print("ATTENTION ANALYSIS:")
    print(f"  • {len(attention_comparison)} scenario comparisons")
    print(f"  • Results: {attention_dir}/attention_comparison_De_vs_Ut.csv")
    print(f"  • Plot: {attention_dir}/attention_comparison_Deontological_vs_Utilitarian.png")

    print("\nCOMPONENT INTERACTION ANALYSIS:")
    print(f"  • {len(interaction_comparison):,} component pair comparisons")
    print(f"  • {len(significant_pathways)} significant pathways (|diff| > 0.3)")
    print(f"  • Results: {interaction_dir}/interaction_comparison_De_vs_Ut.csv")
    print(f"  • Significant pathways: {interaction_dir}/significant_pathways_De_vs_Ut.csv")
    print(f"  • Key pathways: {key_pathways_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")

    print("1. Review attention comparison CSV:")
    print(f"   cat {attention_dir}/attention_comparison_De_vs_Ut.csv")

    print("\n2. Review significant interaction pathways:")
    print(f"   head -20 {interaction_dir}/significant_pathways_De_vs_Ut.csv")

    print("\n3. Check visualizations:")
    print(f"   - Attention: {attention_dir}/*.png")
    print(f"   - Interactions: {interaction_dir}/*.png")

    print("\n4. Statistical testing:")
    print("   - Run paired t-tests on attention differences")
    print("   - Compute effect sizes (Cohen's d)")
    print("   - Apply multiple comparison corrections")

    print("\n5. Update RQ2 analysis document:")
    print("   - Add findings to RQ2_ANALYSIS_RESULTS.md")
    print("   - Integrate with existing DLA and patching results")
    print("   - Create final answer to 'How do similar circuits produce different behaviors?'")

    print("\nFor more details, see: ATTENTION_AND_INTERACTION_ANALYSIS.md")
    print()


if __name__ == "__main__":
    main()
