"""
Experiment 2: Layer 9 Attention Pattern Analysis

Compares attention patterns at Layer 3 vs Layer 9 to understand
what information Layer 9 is re-integrating after early-layer debiasing.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import torch

from mech_interp.attention_analysis import AttentionAnalyzer
from mech_interp.utils import load_prompt_dataset

OUTPUT_DIR = REPO_ROOT / "mech_interp_outputs" / "l9_investigation" / "attention_patterns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_layer_attention(
    model_id: str,
    scenario: str,
    variant: int = 0,
    layers_to_compare: List[int] = [3, 9]
):
    """
    Analyze attention patterns at specific layers.

    Args:
        model_id: Model identifier (e.g., "PT2_COREDe")
        scenario: Scenario name (e.g., "CC_continue")
        variant: Prompt variant number
        layers_to_compare: List of layer indices to analyze

    Returns:
        Dict with attention patterns and token information
    """
    print(f"\nAnalyzing {model_id} on {scenario} variant {variant}...")

    # Load prompt
    prompts = load_prompt_dataset()

    # Find the specific prompt
    prompt_data = None
    for p in prompts:
        if p['scenario'] == scenario and p['variant'] == variant:
            prompt_data = p
            break

    if prompt_data is None:
        raise ValueError(f"No prompt found for scenario={scenario}, variant={variant}")

    prompt = prompt_data['prompt']

    # Initialize analyzer
    analyzer = AttentionAnalyzer(
        model_id=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run analysis
    result = analyzer.analyze_prompt(prompt, scenario, variant)

    # Extract layer-specific attention
    layer_attention = {}
    for layer_idx in layers_to_compare:
        # Get attention from this layer, averaged across heads
        # Shape: (num_heads, seq_len, seq_len)
        layer_attn = result.attention_patterns[layer_idx]

        # Average across heads: (seq_len, seq_len)
        layer_attn_avg = layer_attn.mean(axis=0)

        # Get what the final token attends to from this layer
        final_token_attn = layer_attn_avg[-1, :]

        layer_attention[layer_idx] = {
            'full_attention': layer_attn_avg,
            'final_token_attention': final_token_attn,
        }

    return {
        'model_id': model_id,
        'scenario': scenario,
        'variant': variant,
        'layer_attention': layer_attention,
        'token_texts': result.token_texts,
        'token_ids': result.token_ids,
        'action_keyword_positions': result.action_keyword_positions,
        'opponent_action_positions': result.opponent_action_positions,
        'payoff_positions': result.payoff_positions,
    }


def compare_layers(data: Dict, layer1: int, layer2: int) -> pd.DataFrame:
    """
    Compare attention patterns between two layers.

    Returns DataFrame with token-level comparison.
    """
    tokens = data['token_texts']
    l1_attn = data['layer_attention'][layer1]['final_token_attention']
    l2_attn = data['layer_attention'][layer2]['final_token_attention']

    records = []
    for i, token in enumerate(tokens):
        records.append({
            'position': i,
            'token': token,
            f'L{layer1}_attention': l1_attn[i],
            f'L{layer2}_attention': l2_attn[i],
            'difference': l2_attn[i] - l1_attn[i],
            'is_action_keyword': i in data['action_keyword_positions'],
            'is_opponent_action': i in data['opponent_action_positions'],
            'is_payoff': i in data['payoff_positions'],
        })

    return pd.DataFrame(records)


def plot_attention_comparison(
    data: Dict,
    layer1: int,
    layer2: int,
    output_path: Path
):
    """
    Plot attention patterns for Layer 3 vs Layer 9 side-by-side.
    """
    tokens = data['token_texts']
    l1_attn = data['layer_attention'][layer1]['final_token_attention']
    l2_attn = data['layer_attention'][layer2]['final_token_attention']

    # Truncate long token lists for readability
    max_tokens = min(50, len(tokens))
    tokens_display = tokens[:max_tokens]
    l1_attn_display = l1_attn[:max_tokens]
    l2_attn_display = l2_attn[:max_tokens]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Layer 1 attention
    axes[0].bar(range(len(tokens_display)), l1_attn_display, alpha=0.7, color='blue')
    axes[0].set_ylabel(f'Attention Weight (Layer {layer1})', fontsize=12)
    axes[0].set_title(f'Layer {layer1} Attention Pattern (Final Token)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Highlight special tokens
    for pos in data['action_keyword_positions']:
        if pos < max_tokens:
            axes[0].axvline(x=pos, color='red', linestyle='--', alpha=0.5)

    # Layer 2 attention
    axes[1].bar(range(len(tokens_display)), l2_attn_display, alpha=0.7, color='green')
    axes[1].set_ylabel(f'Attention Weight (Layer {layer2})', fontsize=12)
    axes[1].set_xlabel('Token Position', fontsize=12)
    axes[1].set_title(f'Layer {layer2} Attention Pattern (Final Token)', fontsize=14)
    axes[1].set_xticks(range(len(tokens_display)))
    axes[1].set_xticklabels(tokens_display, rotation=90, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Highlight special tokens
    for pos in data['action_keyword_positions']:
        if pos < max_tokens:
            axes[1].axvline(x=pos, color='red', linestyle='--', alpha=0.5, label='Action keyword')
    for pos in data['opponent_action_positions']:
        if pos < max_tokens:
            axes[1].axvline(x=pos, color='orange', linestyle=':', alpha=0.5, label='Opponent action')

    plt.suptitle(f'{data["model_id"]} - {data["scenario"]}\nAttention Pattern Comparison', fontsize=16)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention comparison plot to {output_path}")
    plt.close()


def plot_attention_difference_heatmap(
    data: Dict,
    layer1: int,
    layer2: int,
    output_path: Path
):
    """
    Plot difference heatmap showing which tokens Layer 9 attends to MORE than Layer 3.
    """
    tokens = data['token_texts']
    l1_attn = data['layer_attention'][layer1]['final_token_attention']
    l2_attn = data['layer_attention'][layer2]['final_token_attention']

    difference = l2_attn - l1_attn

    # Truncate for readability
    max_tokens = min(50, len(tokens))
    tokens_display = tokens[:max_tokens]
    difference_display = difference[:max_tokens]

    plt.figure(figsize=(16, 4))

    # Create bar plot colored by sign
    colors = ['green' if d > 0 else 'red' for d in difference_display]
    plt.bar(range(len(tokens_display)), difference_display, color=colors, alpha=0.7)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel(f'Attention Difference (L{layer2} - L{layer1})', fontsize=12)
    plt.title(f'Attention Difference: Layer {layer2} vs Layer {layer1}\n(Green = L{layer2} attends MORE, Red = L{layer2} attends LESS)', fontsize=14)
    plt.xticks(range(len(tokens_display)), tokens_display, rotation=90, ha='right', fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention difference plot to {output_path}")
    plt.close()


def analyze_token_category_attention(
    comparison_df: pd.DataFrame,
    layer1: int,
    layer2: int
) -> pd.DataFrame:
    """
    Analyze attention by token category (action keywords, opponent actions, payoffs).
    """
    categories = {
        'action_keywords': comparison_df['is_action_keyword'],
        'opponent_actions': comparison_df['is_opponent_action'],
        'payoffs': comparison_df['is_payoff'],
        'other': ~(comparison_df['is_action_keyword'] | comparison_df['is_opponent_action'] | comparison_df['is_payoff'])
    }

    records = []
    for category_name, mask in categories.items():
        category_df = comparison_df[mask]

        if len(category_df) > 0:
            records.append({
                'category': category_name,
                f'L{layer1}_mean_attention': category_df[f'L{layer1}_attention'].mean(),
                f'L{layer2}_mean_attention': category_df[f'L{layer2}_attention'].mean(),
                'attention_difference': category_df['difference'].mean(),
                'num_tokens': len(category_df),
            })

    return pd.DataFrame(records)


def main():
    """Run Experiment 2: Layer 9 Attention Pattern Analysis."""
    print("="*80)
    print("Experiment 2: Layer 9 Attention Pattern Analysis")
    print("="*80)

    # Models to analyze
    models = ['PT2_COREDe', 'PT3_COREDe']
    scenario = 'CC_continue'
    variant = 0
    layers = [3, 9]

    all_comparisons = []

    for model_id in models:
        print(f"\n{'='*60}")
        print(f"Analyzing model: {model_id}")
        print('='*60)

        # Analyze attention
        data = analyze_layer_attention(model_id, scenario, variant, layers)

        # Compare layers
        comparison_df = compare_layers(data, layers[0], layers[1])

        # Save detailed comparison
        comparison_csv = OUTPUT_DIR / f"attention_comparison_{model_id}_{scenario}.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Saved detailed comparison to {comparison_csv}")

        # Analyze by category
        category_analysis = analyze_token_category_attention(comparison_df, layers[0], layers[1])
        print(f"\nToken Category Analysis for {model_id}:")
        print(category_analysis.to_string(index=False))

        category_csv = OUTPUT_DIR / f"category_analysis_{model_id}_{scenario}.csv"
        category_analysis.to_csv(category_csv, index=False)

        all_comparisons.append({
            'model': model_id,
            'category_analysis': category_analysis,
            'comparison_df': comparison_df,
        })

        # Generate plots
        plot_attention_comparison(
            data,
            layers[0],
            layers[1],
            OUTPUT_DIR / f"attention_comparison_{model_id}_{scenario}.png"
        )

        plot_attention_difference_heatmap(
            data,
            layers[0],
            layers[1],
            OUTPUT_DIR / f"attention_difference_{model_id}_{scenario}.png"
        )

    # Cross-model comparison
    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON")
    print("="*80)

    for i, comp1 in enumerate(all_comparisons):
        for comp2 in all_comparisons[i+1:]:
            model1 = comp1['model']
            model2 = comp2['model']

            print(f"\n{model1} vs {model2}:")

            cat1 = comp1['category_analysis']
            cat2 = comp2['category_analysis']

            # Merge on category
            merged = cat1.merge(cat2, on='category', suffixes=(f'_{model1}', f'_{model2}'))
            print(merged.to_string(index=False))

            merged_csv = OUTPUT_DIR / f"cross_model_comparison_{model1}_vs_{model2}.csv"
            merged.to_csv(merged_csv, index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 2 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
