"""
Experiment 5: Activation Steering at Layer 9 MLP

Tests whether steering L9_MLP can control the magnitude of the cooperative dip.
Applies steering vectors at varying strengths and measures trajectory changes.
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
from mech_interp.logit_lens import LogitLensAnalyzer
from mech_interp.activation_steering import ActivationSteerer
from mech_interp.utils import load_prompt_dataset, get_action_token_ids

OUTPUT_DIR = REPO_ROOT / "mech_interp_outputs" / "l9_investigation" / "steering_sweeps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_steering_sweep(
    model_id: str,
    steering_layer: int = 9,
    steering_component: str = "mlp",
    steering_strengths: List[float] = None,
    scenario: str = "CC_continue",
    variant: int = 0
) -> pd.DataFrame:
    """
    Sweep steering strengths and measure trajectory changes.

    Args:
        model_id: Model to steer
        steering_layer: Layer to apply steering
        steering_component: Component type ("mlp" or "attn")
        steering_strengths: List of steering strengths to test
        scenario: Scenario name
        variant: Variant number

    Returns:
        DataFrame with steering results
    """
    if steering_strengths is None:
        steering_strengths = [-2.0, -1.0, 0.0, +1.0, +2.0]

    print(f"\nAnalyzing steering sweep for {model_id}")
    print(f"Layer: {steering_layer}, Component: {steering_component}")
    print(f"Strengths: {steering_strengths}")

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

    # Initialize analyzers
    steerer = ActivationSteerer(
        model=model,
        device=device,
    )

    logit_lens = LogitLensAnalyzer(
        model=model,
        tokenizer=model.tokenizer,
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

    # Compute steering vector
    # For now, use a simple approach: compute difference between moral and selfish activations
    # In practice, you'd want to compute this properly using contrastive examples
    print("\nComputing steering vector...")

    # We'll use a simple random steering vector as a placeholder
    # In real implementation, you'd compute: moral_activation - selfish_activation
    d_model = model.d_model
    steering_vector = torch.randn(d_model, device=device)
    steering_vector = steering_vector / steering_vector.norm()  # Normalize

    # Compute baseline trajectory
    print("\nComputing baseline trajectory...")
    baseline_trajectory = logit_lens.compute_action_trajectory(prompt)

    # Compute steered trajectories
    results = []

    for strength in steering_strengths:
        print(f"\nSteering strength: {strength}")

        try:
            steered_trajectory = logit_lens.compute_action_trajectory_with_steering(
                prompt=prompt,
                steering_layer=steering_layer,
                steering_component=steering_component,
                steering_vector=steering_vector,
                steering_strength=strength,
            )

            # Compute effect at each layer
            for layer_idx in range(len(baseline_trajectory)):
                baseline_delta = baseline_trajectory[layer_idx]
                steered_delta = steered_trajectory[layer_idx]
                effect = steered_delta - baseline_delta

                results.append({
                    'model': model_id,
                    'scenario': scenario,
                    'variant': variant,
                    'steering_layer': steering_layer,
                    'steering_component': steering_component,
                    'steering_strength': strength,
                    'layer': layer_idx,
                    'baseline_delta': baseline_delta,
                    'steered_delta': steered_delta,
                    'steering_effect': effect,
                })

        except Exception as e:
            print(f"Error with strength {strength}: {e}")
            continue

    return pd.DataFrame(results)


def plot_steering_trajectories(
    df: pd.DataFrame,
    output_path: Path
):
    """
    Plot trajectories for different steering strengths.
    """
    strengths = sorted(df['steering_strength'].unique())

    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    cmap = plt.cm.RdBu_r
    colors = [cmap(i / (len(strengths) - 1)) for i in range(len(strengths))]

    for idx, strength in enumerate(strengths):
        strength_df = df[df['steering_strength'] == strength]

        if strength == 0.0:
            # Baseline
            ax.plot(
                strength_df['layer'],
                strength_df['steered_delta'],
                label=f'Baseline (strength=0.0)',
                color='black',
                linewidth=3,
                marker='o',
                markersize=4
            )
        else:
            ax.plot(
                strength_df['layer'],
                strength_df['steered_delta'],
                label=f'Strength={strength:+.1f}',
                color=colors[idx],
                linewidth=2,
                marker='s',
                markersize=3,
                alpha=0.8
            )

    # Highlight key layers
    ax.axvline(x=3, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=9, color='red', linestyle='--', alpha=0.5, linewidth=2, label='L9 (steering layer)')
    ax.axvline(x=16, color='gray', linestyle=':', alpha=0.3)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Delta (D - C)', fontsize=12)
    ax.set_title(f'Steering Effect on Trajectory\n{df["model"].iloc[0]} - {df["scenario"].iloc[0]}', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved steering trajectories plot to {output_path}")
    plt.close()


def plot_l9_dip_vs_strength(
    df: pd.DataFrame,
    output_path: Path
):
    """
    Plot Layer 9 dip magnitude vs steering strength.
    """
    # Extract L9 values
    l9_df = df[df['layer'] == 9].copy()

    plt.figure(figsize=(10, 6))

    plt.plot(
        l9_df['steering_strength'],
        l9_df['steered_delta'],
        marker='o',
        markersize=10,
        linewidth=2,
        color='blue'
    )

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Steering Strength', fontsize=12)
    plt.ylabel('Layer 9 Delta (D - C)', fontsize=12)
    plt.title('Layer 9 Cooperative Dip vs Steering Strength', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved L9 vs strength plot to {output_path}")
    plt.close()


def plot_steering_effect_heatmap(
    df: pd.DataFrame,
    output_path: Path
):
    """
    Plot heatmap showing steering effect across layers and strengths.
    """
    # Pivot data
    pivot = df.pivot_table(
        index='steering_strength',
        columns='layer',
        values='steering_effect',
        aggfunc='mean'
    )

    plt.figure(figsize=(16, 6))

    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0,
        annot=False,
        cbar_kws={'label': 'Steering Effect (Δ)'},
        xticklabels=range(len(pivot.columns)),
    )

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Steering Strength', fontsize=12)
    plt.title('Steering Effect Across Layers', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved steering heatmap to {output_path}")
    plt.close()


def main():
    """Run Experiment 5: Activation Steering at Layer 9 MLP."""
    print("="*80)
    print("Experiment 5: Activation Steering at Layer 9 MLP")
    print("="*80)

    # Models to test
    models = ['PT2_COREDe', 'PT3_COREDe']
    scenario = 'CC_continue'
    variant = 0
    steering_layer = 9
    steering_component = "mlp"
    strengths = [-2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2.0]

    for model_id in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_id}")
        print('='*60)

        try:
            df = analyze_steering_sweep(
                model_id=model_id,
                steering_layer=steering_layer,
                steering_component=steering_component,
                steering_strengths=strengths,
                scenario=scenario,
                variant=variant
            )

            # Save results
            csv_path = OUTPUT_DIR / f"steering_{model_id}_{scenario}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved results to {csv_path}")

            # Generate plots
            traj_path = OUTPUT_DIR / f"steering_trajectories_{model_id}_{scenario}.png"
            plot_steering_trajectories(df, traj_path)

            l9_path = OUTPUT_DIR / f"l9_vs_strength_{model_id}_{scenario}.png"
            plot_l9_dip_vs_strength(df, l9_path)

            heatmap_path = OUTPUT_DIR / f"steering_heatmap_{model_id}_{scenario}.png"
            plot_steering_effect_heatmap(df, heatmap_path)

            # Analyze L9 response
            l9_df = df[df['layer'] == 9]
            print(f"\nLayer 9 Analysis:")
            for _, row in l9_df.iterrows():
                print(f"  Strength {row['steering_strength']:+.1f}: delta = {row['steered_delta']:.3f}, effect = {row['steering_effect']:.3f}")

        except Exception as e:
            print(f"\nError with model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("EXPERIMENT 5 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
