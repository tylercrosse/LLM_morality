"""
Experiment 4: Activation Patching at Layer 9 MLP with Trajectory Analysis

Tests whether L9_MLP is causally responsible for the cooperative dip by
patching it between models and measuring full logit lens trajectories.
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
from mech_interp.activation_patching import ActivationPatcher
from mech_interp.utils import load_prompt_dataset, get_action_token_ids

OUTPUT_DIR = REPO_ROOT / "mech_interp_outputs" / "l9_investigation" / "patching_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_trajectory_with_patch(
    patcher: ActivationPatcher,
    logit_lens: LogitLensAnalyzer,
    prompt: str,
    component: str
) -> np.ndarray:
    """
    Compute logit lens trajectory with a component patched.

    This uses the same patching approach as ActivationPatcher but captures
    the full trajectory during the patched forward pass.

    Returns:
        trajectory: (n_layers,) array of delta logits
    """
    # Ensure source activations are cached
    if patcher.source_cache is None:
        patcher.cache_source_activations(prompt)

    # Parse component to get layer index
    if "_MLP" in component:
        layer_idx = int(component.split("_")[0][1:])
        component_type = "mlp"
    else:
        # Head format: L5H3
        parts = component[1:].split("H")
        layer_idx = int(parts[0])
        component_type = "head"

    # Get source activation to patch in (using correct cache keys)
    if component_type == "mlp":
        cache_key = f"blocks.{layer_idx}.hook_mlp_out"
        source_activation_last = patcher.source_cache[cache_key][:, -1:, :]
    else:
        cache_key = f"blocks.{layer_idx}.hook_attn_out"
        source_activation_last = patcher.source_cache[cache_key][:, -1:, :]

    # Prepare input
    input_ids = patcher._prepare_input_ids(patcher.target_model, prompt)
    target_model = patcher.target_model

    # Register patching hook (following activation_patching.py pattern)
    hook_handles = []

    if component_type == "mlp":
        layer = target_model.transformer_stack.layers[layer_idx]

        def patch_hook(module, input, output):
            patched = output.clone()
            patched[:, -1:, :] = source_activation_last.to(output.device, dtype=output.dtype)
            return patched

        handle = layer.mlp.register_forward_hook(patch_hook)
        hook_handles.append(handle)
    else:
        layer = target_model.transformer_stack.layers[layer_idx]

        def patch_hook(module, input, output):
            attn_out = output[0]
            patched = attn_out.clone()
            patched[:, -1:, :] = source_activation_last.to(attn_out.device, dtype=attn_out.dtype)
            return (patched,) + output[1:]

        handle = layer.self_attn.register_forward_hook(patch_hook)
        hook_handles.append(handle)

    # Run with cache to get trajectory
    with torch.no_grad():
        _, cache = target_model.run_with_cache(input_ids)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Compute logit lens trajectory from cached activations
    trajectory = []
    c_token = logit_lens.action_tokens['C']
    d_token = logit_lens.action_tokens['D']

    for layer in range(target_model.n_layers):
        # Get residual stream at this layer (using correct cache key)
        resid_key = f"blocks.{layer}.hook_resid_post"

        if resid_key in cache:
            hidden_state = cache[resid_key][0, -1, :]  # [d_model]

            # Apply final layer norm
            hidden_state = target_model.ln_final(hidden_state.unsqueeze(0)).squeeze(0)

            # Project through unembed
            logits = target_model.W_U @ hidden_state  # [vocab_size]

            # Compute delta
            delta = logits[d_token].item() - logits[c_token].item()
            trajectory.append(delta)
        else:
            trajectory.append(np.nan)

    return np.array(trajectory)


def analyze_patching_effect(
    source_model_id: str,
    target_model_id: str,
    scenario: str = "CC_continue",
    variant: int = 0,
    components_to_patch: List[str] = None
) -> pd.DataFrame:
    """
    Analyze effect of patching various components on logit lens trajectory.

    Args:
        source_model_id: Model to take activations from
        target_model_id: Model to patch activations into
        scenario: Scenario name
        variant: Variant number
        components_to_patch: List of component names (e.g., ["L9_MLP", "L8_MLP", "L10_MLP"])

    Returns:
        DataFrame with patching results
    """
    if components_to_patch is None:
        components_to_patch = ["L9_MLP"]

    print(f"\nAnalyzing patching: {source_model_id} → {target_model_id}")
    print(f"Scenario: {scenario}, Variant: {variant}")
    print(f"Components to patch: {components_to_patch}")

    # Load models
    print("\nLoading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_model = LoRAModelLoader.load_hooked_model(
        source_model_id,
        device=device,
        merge_lora=False,
        use_4bit=True,
    )

    target_model = LoRAModelLoader.load_hooked_model(
        target_model_id,
        device=device,
        merge_lora=False,
        use_4bit=True,
    )

    # Get action tokens
    action_tokens = get_action_token_ids(target_model.tokenizer)

    # Initialize analyzers
    patcher = ActivationPatcher(
        source_model=source_model,
        target_model=target_model,
        action_tokens=action_tokens,
        device=device,
    )

    logit_lens = LogitLensAnalyzer(
        model=target_model,
        tokenizer=target_model.tokenizer,
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

    # Compute baseline trajectory (no patching)
    print("\nComputing baseline trajectory...")
    baseline_trajectory = logit_lens.compute_action_trajectory(prompt)

    # Compute patched trajectories
    results = []

    for component in components_to_patch:
        print(f"\nPatching {component}...")

        try:
            patched_trajectory = compute_trajectory_with_patch(
                patcher,
                logit_lens,
                prompt,
                component
            )

            # Compute effect at each layer
            for layer_idx in range(len(baseline_trajectory)):
                baseline_delta = baseline_trajectory[layer_idx]
                patched_delta = patched_trajectory[layer_idx]
                effect = patched_delta - baseline_delta

                results.append({
                    'source_model': source_model_id,
                    'target_model': target_model_id,
                    'scenario': scenario,
                    'variant': variant,
                    'patched_component': component,
                    'layer': layer_idx,
                    'baseline_delta': baseline_delta,
                    'patched_delta': patched_delta,
                    'patch_effect': effect,
                })

        except Exception as e:
            print(f"Error patching {component}: {e}")
            continue

    return pd.DataFrame(results)


def plot_patching_effect(
    df: pd.DataFrame,
    output_path: Path
):
    """
    Plot baseline vs patched trajectories.
    """
    # Get unique patched components
    components = df['patched_component'].unique()

    fig, axes = plt.subplots(len(components), 1, figsize=(14, 4 * len(components)), sharex=True)

    if len(components) == 1:
        axes = [axes]

    for idx, component in enumerate(components):
        ax = axes[idx]
        comp_df = df[df['patched_component'] == component]

        # Plot baseline
        ax.plot(
            comp_df['layer'],
            comp_df['baseline_delta'],
            label='Baseline (no patch)',
            color='blue',
            linewidth=2,
            marker='o',
            markersize=3
        )

        # Plot patched
        ax.plot(
            comp_df['layer'],
            comp_df['patched_delta'],
            label=f'Patched ({component})',
            color='red',
            linewidth=2,
            marker='s',
            markersize=3
        )

        # Highlight key layers
        ax.axvline(x=3, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=9, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=16, color='gray', linestyle=':', alpha=0.3)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        ax.set_ylabel('Delta (D - C)', fontsize=11)
        ax.set_title(f'Patching Effect: {component}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Layer', fontsize=12)

    source = df['source_model'].iloc[0]
    target = df['target_model'].iloc[0]
    scenario = df['scenario'].iloc[0]

    plt.suptitle(
        f'Activation Patching: {source} → {target}\nScenario: {scenario}',
        fontsize=14,
        y=1.00
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved patching plot to {output_path}")
    plt.close()


def plot_patch_effect_heatmap(
    df: pd.DataFrame,
    output_path: Path
):
    """
    Plot heatmap showing patch effect across layers.
    """
    # Pivot data
    pivot = df.pivot_table(
        index='patched_component',
        columns='layer',
        values='patch_effect',
        aggfunc='mean'
    )

    plt.figure(figsize=(16, max(4, len(pivot) * 0.5)))

    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0,
        annot=False,
        cbar_kws={'label': 'Patch Effect (Δ)'},
        xticklabels=range(len(pivot.columns)),
    )

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Patched Component', fontsize=12)
    plt.title('Activation Patching Effect Across Layers', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def main():
    """Run Experiment 4: Activation Patching at Layer 9 MLP."""
    print("="*80)
    print("Experiment 4: Activation Patching at Layer 9 MLP")
    print("="*80)

    # Experiment configurations
    experiments = [
        {
            'source': 'PT3_COREDe',  # Deontological
            'target': 'PT2_COREDe',  # Strategic
            'name': 'De_to_Strategic'
        },
        {
            'source': 'PT2_COREDe',  # Strategic
            'target': 'PT3_COREDe',  # Deontological
            'name': 'Strategic_to_De'
        },
    ]

    scenario = 'CC_continue'
    variant = 0

    # Components to patch (L9_MLP and neighbors)
    components = ["L8_MLP", "L9_MLP", "L10_MLP"]

    all_results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp['name']}")
        print('='*60)

        try:
            df = analyze_patching_effect(
                source_model_id=exp['source'],
                target_model_id=exp['target'],
                scenario=scenario,
                variant=variant,
                components_to_patch=components
            )

            # Save results
            csv_path = OUTPUT_DIR / f"patching_{exp['name']}_{scenario}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved results to {csv_path}")

            # Generate plots
            plot_path = OUTPUT_DIR / f"patching_trajectory_{exp['name']}_{scenario}.png"
            plot_patching_effect(df, plot_path)

            heatmap_path = OUTPUT_DIR / f"patching_heatmap_{exp['name']}_{scenario}.png"
            plot_patch_effect_heatmap(df, heatmap_path)

            # Compute L9 dip change
            l9_baseline = df[(df['patched_component'] == 'L9_MLP') & (df['layer'] == 9)]['baseline_delta'].iloc[0]
            l9_patched = df[(df['patched_component'] == 'L9_MLP') & (df['layer'] == 9)]['patched_delta'].iloc[0]
            l9_effect = l9_patched - l9_baseline

            print(f"\nLayer 9 Analysis:")
            print(f"  Baseline delta: {l9_baseline:.3f}")
            print(f"  Patched delta: {l9_patched:.3f}")
            print(f"  Patch effect: {l9_effect:.3f}")

            all_results.append(df)

        except Exception as e:
            print(f"\nError in experiment {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("EXPERIMENT 4 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
