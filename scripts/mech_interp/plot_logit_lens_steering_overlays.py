#!/usr/bin/env python3
"""
Create overlay plots for logit lens steering visualizations.

Generates comparison plots with multiple curves on the same axes:
- All layers for a given model/scenario/strength
- Both models for a given layer/scenario/strength
- All scenarios for a given layer/model/strength

Usage:
    python scripts/mech_interp/plot_logit_lens_steering_overlays.py

Outputs:
    - mech_interp_outputs/causal_routing/logit_lens_steering/overlays/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.logit_lens import LogitLensAnalyzer
from mech_interp.utils import load_prompt_dataset, MODEL_LABELS


COMPONENT_SORT_ORDER = {"attn": 0, "mlp": 1}


def _layer_sort_key(layer_name: str) -> Tuple[int, int, str]:
    """Sort labels like L2_MLP, L19_ATTN by numeric layer then component."""
    parts = layer_name.split("_")
    if len(parts) >= 2 and parts[0].startswith("L") and parts[0][1:].isdigit():
        layer = int(parts[0][1:])
        component_rank = COMPONENT_SORT_ORDER.get(parts[1].lower(), 99)
        return (layer, component_rank, layer_name)
    return (10**9, 99, layer_name)


def compute_all_trajectories(output_dir: Path) -> Dict:
    """
    Compute all baseline and steered trajectories.

    Returns:
        Dict with structure: {
            'baselines': {model_id: {scenario: trajectory}},
            'steered': {model_id: {layer_name: {scenario: {strength: trajectory}}}}
        }
    """
    print("Computing all trajectories...")

    # Configuration
    experiments = [
        {"layer": 17, "component": "mlp", "name": "L17_MLP"},
        {"layer": 16, "component": "mlp", "name": "L16_MLP"},
        {"layer": 8, "component": "mlp", "name": "L8_MLP"},
        {"layer": 9, "component": "mlp", "name": "L9_MLP"},
        {"layer": 11, "component": "mlp", "name": "L11_MLP"},
        {"layer": 19, "component": "attn", "name": "L19_ATTN"},
    ]

    models = ["PT2_COREDe", "PT3_COREDe"]
    scenarios = ["CC_continue", "CC_temptation", "CD_punished", "DC_exploited", "DD_trapped"]
    strengths = [-2.0, 2.0]

    # Load prompts
    prompts_data = load_prompt_dataset()
    prompts_dict = {p["scenario"]: p["prompt"] for p in prompts_data}

    # Storage
    baselines = {}
    steered = {}
    vectors_dir = project_root / "mech_interp_outputs" / "causal_routing"

    # Iterate over models
    for model_id in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        # Load model
        print(f"Loading model...")
        model = LoRAModelLoader.load_hooked_model(model_id, device="cuda")
        analyzer = LogitLensAnalyzer(model, model.tokenizer)

        # Compute baselines for all scenarios
        baselines[model_id] = {}
        for scenario in scenarios:
            prompt = prompts_dict[scenario]
            print(f"  Computing baseline: {scenario}...")
            baselines[model_id][scenario] = analyzer.compute_action_trajectory(prompt)

        # Compute steered trajectories
        steered[model_id] = {}
        for exp in experiments:
            layer_name = exp['name']
            print(f"\n  Layer: {layer_name}")

            # Load steering vector
            vector_path = vectors_dir / f"steering_vector_{layer_name}_De_minus_Strategic.pt"
            if not vector_path.exists():
                print(f"    ⚠ Steering vector not found, skipping")
                continue

            vector_data = torch.load(vector_path)
            steering_vector = vector_data['vector']

            steered[model_id][layer_name] = {}

            for scenario in scenarios:
                prompt = prompts_dict[scenario]
                steered[model_id][layer_name][scenario] = {}

                for strength in strengths:
                    print(f"    {scenario}, strength={strength:+.1f}...", end=" ")
                    traj = analyzer.compute_action_trajectory_with_steering(
                        prompt=prompt,
                        steering_layer=exp['layer'],
                        steering_component=exp['component'],
                        steering_vector=steering_vector,
                        steering_strength=strength,
                    )
                    steered[model_id][layer_name][scenario][strength] = traj
                    print("✓")

        # Clean up
        del model, analyzer
        torch.cuda.empty_cache()

    return {'baselines': baselines, 'steered': steered}


def plot_all_layers_overlay(
    data: Dict,
    model_id: str,
    scenario: str,
    strength: float,
    output_dir: Path,
):
    """Plot all layers for a single model/scenario/strength."""

    layer_styles = {
        "L17_MLP": {"color": "#1ABC9C", "marker": "v", "linestyle": "-", "linewidth": 2.5},
        "L16_MLP": {"color": "#2ECC71", "marker": "D", "linestyle": "-", "linewidth": 2.5},
        "L11_MLP": {"color": "#9B59B6", "marker": "^", "linestyle": "-", "linewidth": 2.0},
        "L9_MLP": {"color": "#3498DB", "marker": "s", "linestyle": "-", "linewidth": 2.0},
        "L8_MLP": {"color": "#E74C3C", "marker": "o", "linestyle": "-", "linewidth": 2.0},
        "L19_ATTN": {"color": "#F39C12", "marker": "p", "linestyle": "--", "linewidth": 2.0},
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot baseline
    baseline = data['baselines'][model_id][scenario]
    layers = np.arange(len(baseline))
    ax.plot(layers, baseline, color='gray', linestyle=':', linewidth=2.5,
            alpha=0.6, label='Baseline (no steering)', zorder=1)

    # Plot steered trajectories for each layer
    available_layers = sorted(data['steered'][model_id].keys(), key=_layer_sort_key)
    for layer_name in available_layers:
        style = layer_styles.get(layer_name, {
            "color": "black",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.0,
        })
        if layer_name not in data['steered'][model_id]:
            continue
        if scenario not in data['steered'][model_id][layer_name]:
            continue
        if strength not in data['steered'][model_id][layer_name][scenario]:
            continue

        traj = data['steered'][model_id][layer_name][scenario][strength]

        ax.plot(layers, traj,
                label=layer_name.replace('_', ' '),
                color=style['color'],
                marker=style['marker'],
                markersize=6,
                markevery=2,
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                alpha=0.85,
                zorder=2)

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1, zorder=0)

    # Labels
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=13, fontweight='bold')

    strength_sign = "+" if strength > 0 else ""
    title = f'{MODEL_LABELS.get(model_id, model_id)} | {scenario.replace("_", " ").title()}\nSteering Strength: {strength_sign}{strength:.1f} (All Layers)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)

    # Annotation
    direction = "toward cooperation" if strength > 0 else "toward defection"
    ax.text(0.02, 0.98, f'Steering direction: {direction}\nPositive = prefers Defect\nNegative = prefers Cooperate',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    filename = f"overlay_all_layers_{model_id}_{scenario}_str{strength:+.1f}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def plot_both_models_overlay(
    data: Dict,
    layer_name: str,
    scenario: str,
    strength: float,
    output_dir: Path,
):
    """Plot both models for a single layer/scenario/strength."""

    fig, ax = plt.subplots(figsize=(12, 7))

    model_styles = {
        "PT2_COREDe": {"color": "#E74C3C", "linestyle": "-", "marker": "o"},
        "PT3_COREDe": {"color": "#3498DB", "linestyle": "-", "marker": "s"},
    }

    layers = np.arange(26)  # Gemma-2-2b has 26 layers

    for model_id, style in model_styles.items():
        # Check if data exists
        if model_id not in data['steered']:
            continue
        if layer_name not in data['steered'][model_id]:
            continue
        if scenario not in data['steered'][model_id][layer_name]:
            continue
        if strength not in data['steered'][model_id][layer_name][scenario]:
            continue

        # Plot baseline (dotted)
        baseline = data['baselines'][model_id][scenario]
        ax.plot(layers, baseline,
                color=style['color'],
                linestyle=':',
                linewidth=2.0,
                alpha=0.5,
                label=f"{MODEL_LABELS.get(model_id, model_id)} (baseline)")

        # Plot steered (solid)
        traj = data['steered'][model_id][layer_name][scenario][strength]
        ax.plot(layers, traj,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=6,
                markevery=2,
                linewidth=2.5,
                alpha=0.9,
                label=f"{MODEL_LABELS.get(model_id, model_id)} (steered)")

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Labels
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=13, fontweight='bold')

    strength_sign = "+" if strength > 0 else ""
    title = f'{layer_name.replace("_", " ")} | {scenario.replace("_", " ").title()}\nSteering Strength: {strength_sign}{strength:.1f} (Model Comparison)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # Save
    filename = f"overlay_both_models_{layer_name}_{scenario}_str{strength:+.1f}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def plot_all_scenarios_overlay(
    data: Dict,
    model_id: str,
    layer_name: str,
    strength: float,
    output_dir: Path,
):
    """Plot all scenarios for a single model/layer/strength."""

    scenario_styles = {
        "CC_continue": {"color": "#2ECC71", "marker": "o", "linestyle": "-"},
        "CC_temptation": {"color": "#E67E22", "marker": "s", "linestyle": "-"},
        "CD_punished": {"color": "#E74C3C", "marker": "^", "linestyle": "-"},
        "DC_exploited": {"color": "#9B59B6", "marker": "v", "linestyle": "-"},
        "DD_trapped": {"color": "#34495E", "marker": "D", "linestyle": "-"},
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    layers = np.arange(26)

    for scenario, style in scenario_styles.items():
        # Check if data exists
        if layer_name not in data['steered'][model_id]:
            continue
        if scenario not in data['steered'][model_id][layer_name]:
            continue
        if strength not in data['steered'][model_id][layer_name][scenario]:
            continue

        traj = data['steered'][model_id][layer_name][scenario][strength]

        ax.plot(layers, traj,
                label=scenario.replace('_', ' ').title(),
                color=style['color'],
                marker=style['marker'],
                markersize=5,
                markevery=3,
                linestyle=style['linestyle'],
                linewidth=2.0,
                alpha=0.85)

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Labels
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=13, fontweight='bold')

    strength_sign = "+" if strength > 0 else ""
    title = f'{MODEL_LABELS.get(model_id, model_id)} | {layer_name.replace("_", " ")}\nSteering Strength: {strength_sign}{strength:.1f} (All Scenarios)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # Save
    filename = f"overlay_all_scenarios_{model_id}_{layer_name}_str{strength:+.1f}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def plot_bidirectional_steering_overlay(
    data: Dict,
    model_id: str,
    scenario: str,
    output_dir: Path,
):
    """
    Plot all layers with both positive and negative steering on same plot.

    Shows +2.0 and -2.0 strength for each layer with color coding:
    - Darker shade = positive steering (toward cooperation)
    - Lighter shade = negative steering (toward defection)
    """

    layer_configs = {
        "L17_MLP": {"color_pos": "#1ABC9C", "color_neg": "#7DCEA0", "marker": "v"},
        "L16_MLP": {"color_pos": "#2ECC71", "color_neg": "#82E0AA", "marker": "D"},
        "L11_MLP": {"color_pos": "#9B59B6", "color_neg": "#D7BDE2", "marker": "^"},
        "L9_MLP": {"color_pos": "#3498DB", "color_neg": "#85C1E2", "marker": "s"},
        "L8_MLP": {"color_pos": "#E74C3C", "color_neg": "#F1948A", "marker": "o"},
        "L19_ATTN": {"color_pos": "#F39C12", "color_neg": "#F8C471", "marker": "p"},
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    layers = np.arange(26)

    # Plot baseline first (make it prominent)
    if scenario in data['baselines'][model_id]:
        baseline = data['baselines'][model_id][scenario]
        ax.plot(layers, baseline, color='#2C3E50', linestyle='-', linewidth=3.5,
                alpha=0.8, label='Baseline (no steering)', zorder=10)

    # Plot each layer with both positive and negative steering
    available_layers = sorted(data['steered'][model_id].keys(), key=_layer_sort_key)

    for layer_name in available_layers:
        if layer_name not in layer_configs:
            continue
        if layer_name not in data['steered'][model_id]:
            continue
        if scenario not in data['steered'][model_id][layer_name]:
            continue

        config = layer_configs[layer_name]

        # Plot negative steering (lighter color, dashed)
        if -2.0 in data['steered'][model_id][layer_name][scenario]:
            traj_neg = data['steered'][model_id][layer_name][scenario][-2.0]
            ax.plot(layers, traj_neg,
                    color=config['color_neg'],
                    marker=config['marker'],
                    markersize=6,
                    markevery=3,
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.75,
                    label=f"{layer_name.replace('_', ' ')} (str=-2.0)")

        # Plot positive steering (darker color, solid)
        if 2.0 in data['steered'][model_id][layer_name][scenario]:
            traj_pos = data['steered'][model_id][layer_name][scenario][2.0]
            ax.plot(layers, traj_pos,
                    color=config['color_pos'],
                    marker=config['marker'],
                    markersize=6,
                    markevery=3,
                    linestyle='-',
                    linewidth=2.5,
                    alpha=0.9,
                    label=f"{layer_name.replace('_', ' ')} (str=+2.0)")

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Labels
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=14, fontweight='bold')

    title = f'{MODEL_LABELS.get(model_id, model_id)} | {scenario.replace("_", " ").title()}\nBidirectional Steering Comparison (±2.0)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Legend with two columns
    ax.legend(fontsize=9, loc='best', framealpha=0.95, ncol=2)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)

    # Annotation
    ax.text(0.02, 0.98,
            'Dark/solid = +2.0 (toward cooperation)\nLight/dashed = -2.0 (toward defection)\nNegative = prefers Cooperate\nPositive = prefers Defect',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    filename = f"overlay_bidirectional_{model_id}_{scenario}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def plot_bidirectional_both_models_overlay(
    data: Dict,
    layer_name: str,
    scenario: str,
    output_dir: Path,
):
    """
    Plot both models with both positive and negative steering on same plot.

    Shows Strategic and Deontological models, each with +2.0 and -2.0 strength:
    - Strategic: Red shades (dark=+2.0, light=-2.0)
    - Deontological: Blue shades (dark=+2.0, light=-2.0)
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    layers = np.arange(26)

    model_configs = {
        "PT2_COREDe": {
            "color_pos": "#E74C3C",  # Dark red
            "color_neg": "#F1948A",  # Light red
            "color_baseline": "#A93226",  # Darker red for baseline
            "marker": "o",
            "label": "Strategic"
        },
        "PT3_COREDe": {
            "color_pos": "#3498DB",  # Dark blue
            "color_neg": "#85C1E2",  # Light blue
            "color_baseline": "#1F618D",  # Darker blue for baseline
            "marker": "s",
            "label": "Deontological"
        },
    }

    for model_id, config in model_configs.items():
        if model_id not in data['steered']:
            continue
        if layer_name not in data['steered'][model_id]:
            continue
        if scenario not in data['steered'][model_id][layer_name]:
            continue

        # Plot baseline (solid, dark, prominent, model-specific color)
        if scenario in data['baselines'][model_id]:
            baseline = data['baselines'][model_id][scenario]
            ax.plot(layers, baseline,
                    color=config['color_baseline'],
                    linestyle='-',
                    linewidth=3.5,
                    alpha=0.8,
                    label=f"{config['label']} (baseline)",
                    zorder=10)

        # Plot negative steering (lighter color, dashed)
        if -2.0 in data['steered'][model_id][layer_name][scenario]:
            traj_neg = data['steered'][model_id][layer_name][scenario][-2.0]
            ax.plot(layers, traj_neg,
                    color=config['color_neg'],
                    marker=config['marker'],
                    markersize=6,
                    markevery=3,
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.75,
                    label=f"{config['label']} (str=-2.0)")

        # Plot positive steering (darker color, solid)
        if 2.0 in data['steered'][model_id][layer_name][scenario]:
            traj_pos = data['steered'][model_id][layer_name][scenario][2.0]
            ax.plot(layers, traj_pos,
                    color=config['color_pos'],
                    marker=config['marker'],
                    markersize=6,
                    markevery=3,
                    linestyle='-',
                    linewidth=2.5,
                    alpha=0.9,
                    label=f"{config['label']} (str=+2.0)")

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Labels
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=14, fontweight='bold')

    title = f'{layer_name.replace("_", " ")} | {scenario.replace("_", " ").title()}\nModel Comparison with Bidirectional Steering (±2.0)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Legend
    ax.legend(fontsize=9, loc='best', framealpha=0.95, ncol=2)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)

    # Annotation
    ax.text(0.02, 0.98,
            'Dark/solid = +2.0 (toward cooperation)\nLight/dashed = -2.0 (toward defection)\nRed = Strategic | Blue = Deontological',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    filename = f"overlay_bidirectional_both_models_{layer_name}_{scenario}.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def plot_key_comparisons(data: Dict, output_dir: Path):
    """Generate key comparison plots for paper/presentation."""

    print("\n" + "="*80)
    print("GENERATING KEY COMPARISON PLOTS")
    print("="*80)

    # 1. L17 vs L16 head-to-head (Strategic model, positive steering, CC_temptation)
    print("\n1. L17 vs L16 comparison (top performers)...")
    fig, ax = plt.subplots(figsize=(12, 7))

    layers = np.arange(26)
    model_id = "PT2_COREDe"
    scenario = "CC_temptation"
    strength = 2.0

    # Baseline
    baseline = data['baselines'][model_id][scenario]
    ax.plot(layers, baseline, color='gray', linestyle=':', linewidth=2.5,
            alpha=0.6, label='Baseline', zorder=1)

    # L17
    if "L17_MLP" in data['steered'][model_id]:
        traj_l17 = data['steered'][model_id]["L17_MLP"][scenario][strength]
        ax.plot(layers, traj_l17, color='#1ABC9C', marker='v', markersize=7,
                markevery=2, linewidth=3.0, label='L17 MLP (best)', alpha=0.9, zorder=3)
        ax.axvline(x=17, color='#1ABC9C', linestyle='--', alpha=0.4, linewidth=1.5)

    # L16
    if "L16_MLP" in data['steered'][model_id]:
        traj_l16 = data['steered'][model_id]["L16_MLP"][scenario][strength]
        ax.plot(layers, traj_l16, color='#2ECC71', marker='D', markersize=7,
                markevery=2, linewidth=3.0, label='L16 MLP (2nd best)', alpha=0.9, zorder=2)
        ax.axvline(x=16, color='#2ECC71', linestyle='--', alpha=0.4, linewidth=1.5)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=14, fontweight='bold')
    ax.set_title('Top Performers: L17 vs L16 Steering Comparison\nStrategic Model, CC Temptation Scenario, +2.0 Strength',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "KEY_l17_vs_l16_comparison.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: KEY_l17_vs_l16_comparison.png")
    plt.close()

    # 2. Early vs Late layer comparison (show washout)
    print("\n2. Early vs Late layer comparison (washout effect)...")
    fig, ax = plt.subplots(figsize=(12, 7))

    # L8 (early, washes out)
    if "L8_MLP" in data['steered'][model_id]:
        traj_l8 = data['steered'][model_id]["L8_MLP"][scenario][strength]
        ax.plot(layers, traj_l8, color='#E74C3C', marker='o', markersize=6,
                markevery=2, linewidth=2.5, label='L8 MLP (early, washes out)', alpha=0.9)
        ax.axvline(x=8, color='#E74C3C', linestyle='--', alpha=0.4, linewidth=1.5)

    # L17 (late, persists)
    if "L17_MLP" in data['steered'][model_id]:
        traj_l17 = data['steered'][model_id]["L17_MLP"][scenario][strength]
        ax.plot(layers, traj_l17, color='#1ABC9C', marker='v', markersize=6,
                markevery=2, linewidth=2.5, label='L17 MLP (late, persists)', alpha=0.9)
        ax.axvline(x=17, color='#1ABC9C', linestyle='--', alpha=0.4, linewidth=1.5)

    # Baseline
    ax.plot(layers, baseline, color='gray', linestyle=':', linewidth=2.5,
            alpha=0.6, label='Baseline')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=14, fontweight='bold')
    ax.set_title('Signal Washout: Early Layer (L8) vs Late Layer (L17)\nStrategic Model, CC Temptation Scenario, +2.0 Strength',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)

    # Add annotation
    ax.annotate('L8 effect\nwashes out', xy=(15, traj_l8[15]), xytext=(18, traj_l8[15] + 1.5),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
                fontsize=10, color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.annotate('L17 effect\npersists', xy=(23, traj_l17[23]), xytext=(20, traj_l17[23] - 1.5),
                arrowprops=dict(arrowstyle='->', color='#1ABC9C', lw=2),
                fontsize=10, color='#1ABC9C', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "KEY_early_vs_late_washout.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: KEY_early_vs_late_washout.png")
    plt.close()

    # 3. Model universality: Strategic vs Deontological on L17
    print("\n3. Model universality comparison...")
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_id, color, marker, label in [
        ("PT2_COREDe", "#E74C3C", "o", "Strategic"),
        ("PT3_COREDe", "#3498DB", "s", "Deontological")
    ]:
        if "L17_MLP" in data['steered'][model_id]:
            traj = data['steered'][model_id]["L17_MLP"][scenario][strength]
            ax.plot(layers, traj, color=color, marker=marker, markersize=6,
                    markevery=2, linewidth=2.5, label=f'{label} (steered)', alpha=0.9)

            baseline = data['baselines'][model_id][scenario]
            ax.plot(layers, baseline, color=color, linestyle=':', linewidth=2.0,
                    label=f'{label} (baseline)', alpha=0.5)

    ax.axvline(x=17, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Logit Difference (Defect - Cooperate)', fontsize=14, fontweight='bold')
    ax.set_title('Model Universality: Strategic vs Deontological\nL17 MLP Steering, CC Temptation Scenario, +2.0 Strength',
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "KEY_model_universality_l17.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: KEY_model_universality_l17.png")
    plt.close()


def main():
    """Generate overlay comparison plots."""

    output_base = project_root / "mech_interp_outputs" / "causal_routing" / "logit_lens_steering"
    output_dir = output_base / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LOGIT LENS STEERING OVERLAY PLOTS")
    print("="*80)
    print(f"\nOutput directory: {output_dir}\n")

    # Compute or load trajectories
    cache_file = output_base / "trajectory_cache.pt"

    if cache_file.exists():
        print("Loading cached trajectories...")
        data = torch.load(cache_file, weights_only=False)
        print("✓ Loaded from cache\n")
    else:
        data = compute_all_trajectories(output_base)
        print(f"\nSaving trajectory cache to {cache_file.name}...")
        torch.save(data, cache_file)
        print("✓ Cached for future use\n")

    # Generate overlay plots
    models = ["PT2_COREDe", "PT3_COREDe"]
    layers = sorted(["L17_MLP", "L16_MLP", "L8_MLP", "L9_MLP", "L11_MLP", "L19_ATTN"], key=_layer_sort_key)
    scenarios = ["CC_continue", "CC_temptation", "CD_punished", "DC_exploited", "DD_trapped"]
    strengths = [-2.0, 2.0]

    # 1. All layers overlay (for each model/scenario/strength)
    print("\n" + "="*80)
    print("GENERATING ALL-LAYERS OVERLAY PLOTS")
    print("="*80)

    for model_id in models:
        print(f"\n{model_id}:")
        for scenario in scenarios:
            for strength in strengths:
                plot_all_layers_overlay(data, model_id, scenario, strength, output_dir)

    # 2. Both models overlay (for each layer/scenario/strength)
    print("\n" + "="*80)
    print("GENERATING BOTH-MODELS OVERLAY PLOTS")
    print("="*80)

    for layer_name in sorted(["L17_MLP", "L16_MLP", "L11_MLP"], key=_layer_sort_key):  # Focus on key layers
        print(f"\n{layer_name}:")
        for scenario in scenarios:
            for strength in strengths:
                plot_both_models_overlay(data, layer_name, scenario, strength, output_dir)

    # 3. All scenarios overlay (for each model/layer/strength)
    print("\n" + "="*80)
    print("GENERATING ALL-SCENARIOS OVERLAY PLOTS")
    print("="*80)

    for model_id in models:
        print(f"\n{model_id}:")
        for layer_name in sorted(["L17_MLP", "L16_MLP", "L8_MLP"], key=_layer_sort_key):  # Focus on key layers
            for strength in strengths:
                plot_all_scenarios_overlay(data, model_id, layer_name, strength, output_dir)

    # 4. Bidirectional steering overlay (positive and negative on same plot)
    print("\n" + "="*80)
    print("GENERATING BIDIRECTIONAL STEERING OVERLAY PLOTS")
    print("="*80)

    for model_id in models:
        print(f"\n{model_id}:")
        for scenario in scenarios:
            plot_bidirectional_steering_overlay(data, model_id, scenario, output_dir)

    # 5. Bidirectional both-models overlay (both models with +/- steering)
    print("\n" + "="*80)
    print("GENERATING BIDIRECTIONAL BOTH-MODELS OVERLAY PLOTS")
    print("="*80)

    for layer_name in sorted(["L17_MLP", "L16_MLP", "L11_MLP"], key=_layer_sort_key):
        print(f"\n{layer_name}:")
        for scenario in scenarios:
            plot_bidirectional_both_models_overlay(data, layer_name, scenario, output_dir)

    # 6. Key comparison plots
    plot_key_comparisons(data, output_dir)

    # Summary
    plot_count = len(list(output_dir.glob("*.png")))

    print("\n" + "="*80)
    print("OVERLAY PLOTS COMPLETE")
    print("="*80)
    print(f"\nGenerated {plot_count} overlay plots in: {output_dir}")
    print(f"\nKey plots to examine:")
    print(f"  1. KEY_l17_vs_l16_comparison.png")
    print(f"     - Direct comparison of top two performers")
    print(f"  2. KEY_early_vs_late_washout.png")
    print(f"     - Shows why early layer steering fails (signal washout)")
    print(f"  3. KEY_model_universality_l17.png")
    print(f"     - Tests if steering patterns are universal across models")
    print(f"  4. overlay_bidirectional_*.png ← NEW!")
    print(f"     - Shows +2.0 and -2.0 steering on same plot for single model")
    print(f"     - Makes effect size and symmetry obvious")
    print(f"  5. overlay_bidirectional_both_models_*.png ← NEW!")
    print(f"     - Shows both models with +2.0 and -2.0 steering")
    print(f"     - Tests model universality and asymmetry together")
    print(f"\nAdditional plots:")
    print(f"  - overlay_all_layers_*.png - Compare all layers for same model/scenario")
    print(f"  - overlay_both_models_*.png - Compare Strategic vs Deontological")
    print(f"  - overlay_all_scenarios_*.png - Compare scenarios for same layer/model")
    print("="*80)


if __name__ == "__main__":
    main()
