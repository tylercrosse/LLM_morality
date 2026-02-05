#!/usr/bin/env python3
"""
Generate Logit Lens plots with steering applied.

Shows how steering at a specific layer affects the layer-by-layer evolution
of action preferences (Cooperate vs Defect).

Usage:
    python scripts/mech_interp/generate_steering_logit_lens.py

Outputs:
    - mech_interp_outputs/causal_routing/logit_lens_steering/*.png
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.logit_lens import LogitLensAnalyzer, LogitLensVisualizer
from mech_interp.utils import load_prompt_dataset


def main():
    """Generate steering-aware logit lens plots."""

    # Configuration
    output_dir = project_root / "mech_interp_outputs" / "causal_routing" / "logit_lens_steering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOGIT LENS WITH STEERING VISUALIZATION")
    print("=" * 80)
    print("\nGenerating layer-by-layer logit evolution plots with steering applied")
    print(f"Output directory: {output_dir}\n")

    # Load steering vectors (from comprehensive experiments)
    vectors_dir = project_root / "mech_interp_outputs" / "causal_routing"

    # Define experiments to visualize
    experiments = [
        # Top performers (L16-17)
        {
            "layer": 17,
            "component": "mlp",
            "name": "L17_MLP",
            "strengths": [-2.0, +2.0],
            "priority": "TOP",
            "rationale": "Best performer (+29.6%)",
        },
        {
            "layer": 16,
            "component": "mlp",
            "name": "L16_MLP",
            "strengths": [-2.0, +2.0],
            "priority": "TOP",
            "rationale": "Second-best performer (+25.7%)",
        },
        # Failed layers for comparison
        {
            "layer": 8,
            "component": "mlp",
            "name": "L8_MLP",
            "strengths": [-2.0, +2.0],
            "priority": "CONTRAST",
            "rationale": "Failed layer (-0.9%), shows washout",
        },
        {
            "layer": 9,
            "component": "mlp",
            "name": "L9_MLP",
            "strengths": [-2.0, +2.0],
            "priority": "CONTRAST",
            "rationale": "Failed layer (-2.8%), encodes cooperation but not steerable",
        },
        # Paradoxical finding
        {
            "layer": 11,
            "component": "mlp",
            "name": "L11_MLP",
            "strengths": [-2.0, +2.0],
            "priority": "INVESTIGATE",
            "rationale": "Paradox: Makes cooperation WORSE (-3.9%)",
        },
        # Attention layer
        {
            "layer": 19,
            "component": "attn",
            "name": "L19_ATTN",
            "strengths": [-2.0, +2.0],
            "priority": "MODERATE",
            "rationale": "Attention routing (+4.3%)",
        },
    ]

    # Select scenarios to visualize
    scenarios = ["CC_continue", "CC_temptation", "CD_punished", "DC_exploited", "DD_trapped"]

    # Models to test
    models = [
        {"id": "PT2_COREDe", "label": "Strategic (selfish baseline)"},
        {"id": "PT3_COREDe", "label": "Deontological (cooperative baseline)"},
    ]

    print(f"Testing models: {[m['id'] for m in models]}")
    print(f"Scenarios: {scenarios}")
    print(f"Experiments: {len(experiments)} layer/component combinations")
    print(f"  - TOP performers: L17, L16 (test success)")
    print(f"  - CONTRAST layers: L8, L9 (test failure)")
    print(f"  - INVESTIGATE: L11 (paradoxical worsening)")
    print(f"  - MODERATE: L19 (attention routing)")
    print()

    # Load prompts once
    prompts_data = load_prompt_dataset()
    prompts_dict = {p["scenario"]: p["prompt"] for p in prompts_data}

    # Iterate over models
    for model_info in models:
        model_id = model_info["id"]

        print(f"\n{'='*80}")
        print(f"MODEL: {model_id} ({model_info['label']})")
        print(f"{'='*80}\n")

        # Load model
        print(f"Loading model {model_id}...")
        model = LoRAModelLoader.load_hooked_model(model_id, device="cuda")
        analyzer = LogitLensAnalyzer(model, model.tokenizer)

        # Generate plots for each experiment
        for exp in experiments:
            print(f"\n{'-' * 80}")
            print(f"Experiment: {exp['name']} ({exp['priority']}) - {exp['rationale']}")
            print(f"Layer {exp['layer']}, {exp['component'].upper()}")
            print(f"{'-' * 80}\n")

            # Load steering vector
            vector_path = vectors_dir / f"steering_vector_{exp['name']}_De_minus_Strategic.pt"

            if not vector_path.exists():
                print(f"⚠ Steering vector not found: {vector_path}")
                print(f"  Run comprehensive experiments first to generate steering vectors")
                continue

            vector_data = torch.load(vector_path)
            steering_vector = vector_data['vector']

            print(f"✓ Loaded steering vector: {vector_path.name}")

            # Generate plots for each scenario
            for scenario in scenarios:
                if scenario not in prompts_dict:
                    print(f"⚠ Scenario {scenario} not found in prompts")
                    continue

                prompt = prompts_dict[scenario]

                print(f"\n  Scenario: {scenario}")

                # Compute baseline trajectory (no steering)
                print(f"    Computing baseline trajectory...")
                baseline_traj = analyzer.compute_action_trajectory(prompt)

                # Generate plot for each steering strength
                for strength in exp['strengths']:
                    if strength == 0.0:
                        # Skip zero strength (same as baseline)
                        continue

                    print(f"    Computing steered trajectory (strength={strength:+.1f})...")

                    # Compute steered trajectory
                    steered_traj = analyzer.compute_action_trajectory_with_steering(
                        prompt=prompt,
                        steering_layer=exp['layer'],
                        steering_component=exp['component'],
                        steering_vector=steering_vector,
                        steering_strength=strength,
                    )

                    # Generate comparison plot
                    plot_filename = f"steering_logit_lens_{model_id}_{exp['name']}_str{strength:+.1f}_{scenario}.png"
                    plot_path = output_dir / plot_filename

                    LogitLensVisualizer.plot_steering_comparison(
                        baseline_trajectory=baseline_traj,
                        steered_trajectory=steered_traj,
                        model_id=model_id,
                        scenario=scenario,
                        steering_layer=exp['layer'],
                        steering_component=exp['component'],
                        steering_strength=strength,
                        output_path=plot_path,
                    )

                    print(f"    ✓ Saved: {plot_filename}")

        # Clean up model to free memory
        del model, analyzer
        torch.cuda.empty_cache()
        print(f"\n✓ Completed all experiments for {model_id}")

    # Count generated plots
    plot_count = len(list(output_dir.glob("*.png")))

    # Summary
    print(f"\n{'=' * 80}")
    print("LOGIT LENS STEERING VISUALIZATION COMPLETE")
    print(f"{'=' * 80}\n")
    print(f"Generated {plot_count} plots in: {output_dir}")
    print(f"\nExperiments completed:")
    print(f"  Models: {len(models)} (Strategic, Deontological)")
    print(f"  Layers: {len(experiments)} (L17, L16, L8, L9, L11, L19)")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Steering strengths: 2 per layer (-2.0, +2.0)")
    print(f"\nKey plots to examine:")
    print(f"  1. L17_MLP (TOP performer):")
    print(f"     - steering_logit_lens_PT2_COREDe_L17_MLP_str+2.0_*.png")
    print(f"     - Compare to L16 - even stronger effect?")
    print(f"  2. L11_MLP (PARADOX):")
    print(f"     - steering_logit_lens_*_L11_MLP_str+2.0_*.png")
    print(f"     - Why does positive steering make cooperation worse?")
    print(f"  3. L8 vs L9 (CONTRAST):")
    print(f"     - steering_logit_lens_*_L8_MLP_* vs L9_MLP_*")
    print(f"     - Both fail - washout at different rates?")
    print(f"  4. Strategic vs Deontological:")
    print(f"     - PT2_COREDe vs PT3_COREDe for same layer")
    print(f"     - Universal patterns or model-specific?")
    print(f"  5. Negative steering (str-2.0):")
    print(f"     - Does steering toward 'strategic' work on Deontological model?")
    print(f"\nInterpretation Guide:")
    print(f"  - Blue line: Baseline trajectory (no steering)")
    print(f"  - Orange line: Steered trajectory")
    print(f"  - Red dashed line: Layer where steering was applied")
    print(f"  - Positive values: Prefers Defect")
    print(f"  - Negative values: Prefers Cooperate")
    print(f"\nKey Questions to Answer:")
    print(f"  1. When does steering take effect? (immediate vs gradual)")
    print(f"  2. Does the effect persist through remaining layers?")
    print(f"  3. Is L17 even better than L16? (both ~+28%)")
    print(f"  4. Why does L11 steering worsen cooperation?")
    print(f"  5. Are patterns universal across Strategic/Deontological models?")
    print(f"  6. Does negative steering work symmetrically?")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
