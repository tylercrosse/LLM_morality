"""
Experiment 6: Direct Logit Attribution Decomposition at Layer 9

Decomposes Layer 9 contributions to understand whether L9_MLP or L9_ATTN
drives the cooperative dip.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.direct_logit_attribution import DirectLogitAttributor
from mech_interp.utils import load_prompt_dataset, get_action_token_ids

OUTPUT_DIR = REPO_ROOT / "mech_interp_outputs" / "l9_investigation" / "dla_decomposition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def decompose_layer_contributions(
    attributor: DirectLogitAttributor,
    prompt: str,
    layer_idx: int = 9
) -> Dict[str, float]:
    """
    Decompose a specific layer's contribution to the final logit.

    This extends the DLA analyzer to extract per-component contributions
    at a specific layer.

    Returns:
        Dict with 'mlp_contribution' and 'attn_contribution' to cooperative signal
    """
    # Run attribution analysis
    result = attributor.decompose_logits(prompt)

    # Get contributions from the specified layer
    # DLAResult has:
    #   mlp_contributions: [n_layers] array
    #   head_contributions: [n_layers, n_heads] array

    # Extract Layer contributions
    mlp_contrib = result.mlp_contributions[layer_idx]
    attn_contrib = result.head_contributions[layer_idx, :].sum()

    return {
        'mlp_contribution': float(mlp_contrib),
        'attn_contribution': float(attn_contrib),
        'total_layer_contribution': float(mlp_contrib + attn_contrib),
    }


def analyze_l9_decomposition(
    model_id: str,
    scenario: str = "CC_continue",
    variant: int = 0
) -> pd.DataFrame:
    """
    Analyze L9 component contributions.

    Args:
        model_id: Model to analyze
        scenario: Scenario name
        variant: Variant number

    Returns:
        DataFrame with component contributions
    """
    print(f"\nAnalyzing L9 decomposition for {model_id}")
    print(f"Scenario: {scenario}, Variant: {variant}")

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LoRAModelLoader.load_hooked_model(
        model_id,
        device=device,
        merge_lora=False,
        use_4bit=True,
    )

    # Get action tokens
    action_tokens = get_action_token_ids(model.tokenizer)

    # Initialize attributor
    attributor = DirectLogitAttributor(
        hooked_model=model,
        action_tokens=action_tokens,
        use_chat_template=True,
    )

    # Load prompt
    prompts = load_prompt_dataset()
    prompt_data = None
    for p in prompts:
        if p['scenario'] == scenario and p['variant'] == variant:
            prompt_data = p
            break

    if prompt_data is None:
        raise ValueError(f"No prompt found for scenario={scenario}, variant={variant}")

    prompt = prompt_data['prompt']

    # Run full attribution
    print("\nComputing full DLA...")
    result = attributor.decompose_logits(prompt)

    # Extract Layer 9 specific decomposition
    print("\nDecomposing Layer 9...")
    l9_decomp = decompose_layer_contributions(attributor, prompt, layer_idx=9)

    # Create results dataframe
    results = []

    # Add Layer 9 decomposition
    results.append({
        'model': model_id,
        'scenario': scenario,
        'variant': variant,
        'component': 'L9_MLP',
        'contribution': l9_decomp['mlp_contribution'],
        'contribution_type': 'cooperative' if l9_decomp['mlp_contribution'] < 0 else 'defective',
    })

    results.append({
        'model': model_id,
        'scenario': scenario,
        'variant': variant,
        'component': 'L9_ATTN',
        'contribution': l9_decomp['attn_contribution'],
        'contribution_type': 'cooperative' if l9_decomp['attn_contribution'] < 0 else 'defective',
    })

    # Add all other components for context
    # Iterate through MLP contributions
    for layer_idx in range(result.mlp_contributions.shape[0]):
        if layer_idx != 9:  # Skip L9 (already added)
            results.append({
                'model': model_id,
                'scenario': scenario,
                'variant': variant,
                'component': f'L{layer_idx}_MLP',
                'contribution': float(result.mlp_contributions[layer_idx]),
                'contribution_type': 'cooperative' if result.mlp_contributions[layer_idx] < 0 else 'defective',
            })

    # Iterate through head contributions
    for layer_idx in range(result.head_contributions.shape[0]):
        if layer_idx != 9:  # Skip L9 (already added)
            for head_idx in range(result.head_contributions.shape[1]):
                results.append({
                    'model': model_id,
                    'scenario': scenario,
                    'variant': variant,
                    'component': f'L{layer_idx}H{head_idx}',
                    'contribution': float(result.head_contributions[layer_idx, head_idx]),
                    'contribution_type': 'cooperative' if result.head_contributions[layer_idx, head_idx] < 0 else 'defective',
                })

    return pd.DataFrame(results)


def plot_component_contributions(
    df: pd.DataFrame,
    output_path: Path,
    top_k: int = 20
):
    """
    Plot top component contributions with L9 highlighted.
    """
    # Sort by absolute contribution
    df_sorted = df.copy()
    df_sorted['abs_contribution'] = df_sorted['contribution'].abs()
    df_sorted = df_sorted.sort_values('abs_contribution', ascending=False).head(top_k)

    # Highlight L9 components
    colors = ['red' if 'L9' in comp else 'blue' for comp in df_sorted['component']]

    plt.figure(figsize=(12, 8))

    plt.barh(range(len(df_sorted)), df_sorted['contribution'], color=colors, alpha=0.7)

    plt.yticks(range(len(df_sorted)), df_sorted['component'], fontsize=9)
    plt.xlabel('Contribution to Final Logit (D - C)', fontsize=12)
    plt.title(f'Top {top_k} Component Contributions (L9 in red)\n{df["model"].iloc[0]} - {df["scenario"].iloc[0]}', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved component contributions plot to {output_path}")
    plt.close()


def plot_l9_decomposition(
    df: pd.DataFrame,
    output_path: Path
):
    """
    Plot L9_MLP vs L9_ATTN contributions for all models.
    """
    # Filter to L9 components only
    l9_df = df[df['component'].str.contains('L9')].copy()

    # Pivot for grouped bar plot
    models = l9_df['model'].unique()
    components = ['L9_MLP', 'L9_ATTN']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    mlp_values = []
    attn_values = []

    for model in models:
        model_df = l9_df[l9_df['model'] == model]
        mlp_val = model_df[model_df['component'] == 'L9_MLP']['contribution'].iloc[0] if len(model_df[model_df['component'] == 'L9_MLP']) > 0 else 0
        attn_val = model_df[model_df['component'] == 'L9_ATTN']['contribution'].iloc[0] if len(model_df[model_df['component'] == 'L9_ATTN']) > 0 else 0

        mlp_values.append(mlp_val)
        attn_values.append(attn_val)

    ax.bar(x - width/2, mlp_values, width, label='L9_MLP', color='blue', alpha=0.7)
    ax.bar(x + width/2, attn_values, width, label='L9_ATTN', color='green', alpha=0.7)

    ax.set_ylabel('Contribution to Final Logit (D - C)', fontsize=12)
    ax.set_title('Layer 9 Component Contributions', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved L9 decomposition plot to {output_path}")
    plt.close()


def main():
    """Run Experiment 6: DLA Decomposition at Layer 9."""
    print("="*80)
    print("Experiment 6: Direct Logit Attribution Decomposition at Layer 9")
    print("="*80)

    # Models to test
    models = ['PT2_COREDe', 'PT3_COREDe']
    scenario = 'CC_continue'
    variant = 0

    all_results = []

    for model_id in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_id}")
        print('='*60)

        try:
            df = analyze_l9_decomposition(
                model_id=model_id,
                scenario=scenario,
                variant=variant
            )

            # Save results
            csv_path = OUTPUT_DIR / f"dla_decomposition_{model_id}_{scenario}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved results to {csv_path}")

            # Generate plots
            contrib_path = OUTPUT_DIR / f"component_contributions_{model_id}_{scenario}.png"
            plot_component_contributions(df, contrib_path, top_k=20)

            # Print L9 analysis
            l9_df = df[df['component'].str.contains('L9')]
            print(f"\nLayer 9 Decomposition:")
            for _, row in l9_df.iterrows():
                print(f"  {row['component']}: {row['contribution']:.4f} ({row['contribution_type']})")

            all_results.append(df)

        except Exception as e:
            print(f"\nError with model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combined plot
    if len(all_results) > 0:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = OUTPUT_DIR / f"l9_decomposition_comparison_{scenario}.png"
        plot_l9_decomposition(combined_df, combined_path)

    print("\n" + "="*80)
    print("EXPERIMENT 6 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
