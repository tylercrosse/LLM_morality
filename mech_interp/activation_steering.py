"""
Activation Steering Experiment.

Tests whether steering L2_MLP activations can control moral behavior by adding
directional vectors to activations during forward passes. This provides evidence
for L2_MLP's role in routing information relevant to moral decisions.

Key Functions:
- find_steering_vector: Compute direction of "moral vs strategic" in activation space
- steer_and_evaluate: Apply steering and measure behavior change
- steering_sweep: Test multiple strengths to show control
- downstream_effect_analysis: Isolate which information flows downstream
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from mech_interp.model_loader import LoRAModelLoader
from mech_interp.decision_metrics import compute_action_sequence_preference, prepare_prompt
from mech_interp.utils import load_prompt_dataset


def load_ipd_prompts():
    """Load IPD prompts as dict keyed by scenario name."""
    prompts = load_prompt_dataset()
    # Convert list to dict keyed by scenario
    prompts_dict = {}
    for p in prompts:
        scenario = p["scenario"]
        if scenario not in prompts_dict:
            prompts_dict[scenario] = p
    return prompts_dict


@dataclass
class SteeringResult:
    """Results from an activation steering experiment."""

    model_id: str
    layer: int
    component: str  # e.g., "mlp"

    # Steering configuration
    steering_direction: str  # e.g., "moral→strategic" or "strategic→moral"
    steering_strength: float

    # Behavioral results
    baseline_p_action2: float  # Without steering
    steered_p_action2: float   # With steering
    delta_p_action2: float     # steered - baseline

    # Per-scenario results
    scenarios: List[str]
    baseline_p_action2_per_scenario: List[float]
    steered_p_action2_per_scenario: List[float]


@dataclass
class SteeringSweepResult:
    """Results from sweeping steering strength."""

    model_id: str
    layer: int
    component: str
    steering_direction: str

    # Sweep data
    strengths: List[float]
    mean_p_action2: List[float]
    mean_coop_rate: List[float]  # 1 - p_action2

    # Per-scenario data
    scenarios: List[str]
    p_action2_matrix: np.ndarray  # Shape: (n_strengths, n_scenarios)


@dataclass
class DownstreamEffectResult:
    """Results from analyzing downstream effects of steering."""

    model_id: str
    steering_layer: int
    steering_strength: float

    # Activation changes at each layer
    layer_indices: List[int]
    activation_changes: List[float]  # Mean absolute change

    # Component-level changes (using DLA-style decomposition)
    top_affected_components: List[str]
    component_change_magnitudes: List[float]


class ActivationSteerer:
    """
    Steers model behavior by adding directional vectors to intermediate activations.

    Tests whether L2_MLP can control downstream routing by measuring how behavior
    changes when we push activations in the "moral" vs "strategic" direction.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize steerer.

        Args:
            device: Device to load models on
        """
        self.device = device

    def find_steering_vector(
        self,
        moral_model_id: str,
        strategic_model_id: str,
        layer: int,
        component: str = "mlp",
        scenarios: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute steering vector from activation differences.

        Args:
            moral_model_id: Cooperative model (e.g., "PT3_COREDe")
            strategic_model_id: Selfish model (e.g., "PT2_COREDe")
            layer: Layer to extract activations from
            component: Component type ("mlp" or "attn")
            scenarios: Scenarios to compute over (default: all IPD scenarios)

        Returns:
            Tuple of (steering_vector, metadata)
            - steering_vector: Tensor of shape (hidden_dim,)
            - metadata: Dict with statistics
        """
        print(f"\n{'='*80}")
        print(f"Computing Steering Vector: {moral_model_id} - {strategic_model_id}")
        print(f"Layer: {layer}, Component: {component}")
        print(f"{'='*80}")

        # Load IPD prompts
        prompts_data = load_ipd_prompts()
        if scenarios is None:
            scenarios = list(prompts_data.keys())

        # Load models
        print(f"\nLoading {moral_model_id}...")
        moral_model = LoRAModelLoader.load_hooked_model(moral_model_id, device=self.device)

        print(f"Loading {strategic_model_id}...")
        strategic_model = LoRAModelLoader.load_hooked_model(strategic_model_id, device=self.device)

        # Collect activations for both models
        moral_acts = []
        strategic_acts = []

        hook_name = f"blocks.{layer}.hook_{component}_out"

        print(f"\nCollecting activations from {len(scenarios)} scenarios...")
        for i, scenario_name in enumerate(scenarios, 1):
            prompt_text = prompts_data[scenario_name]["prompt"]
            formatted_prompt = prepare_prompt(moral_model.tokenizer, prompt_text, use_chat_template=True)
            input_ids = moral_model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

            # Run both models with cache
            with torch.no_grad():
                _, moral_cache = moral_model.run_with_cache(input_ids)
                _, strategic_cache = strategic_model.run_with_cache(input_ids)

            # Extract final position activation
            moral_act = moral_cache[hook_name][0, -1, :].cpu()  # (hidden_dim,)
            strategic_act = strategic_cache[hook_name][0, -1, :].cpu()

            moral_acts.append(moral_act)
            strategic_acts.append(strategic_act)

            if i % 5 == 0:
                print(f"  Processed {i}/{len(scenarios)} scenarios")

        # Compute steering vector: moral - strategic
        moral_acts_tensor = torch.stack(moral_acts)  # (n_scenarios, hidden_dim)
        strategic_acts_tensor = torch.stack(strategic_acts)

        mean_moral = moral_acts_tensor.mean(dim=0)
        mean_strategic = strategic_acts_tensor.mean(dim=0)

        steering_vector = mean_moral - mean_strategic  # (hidden_dim,)

        # Normalize
        steering_vector_norm = steering_vector / steering_vector.norm()

        # Compute metadata
        cosine_similarity = torch.nn.functional.cosine_similarity(
            mean_moral.unsqueeze(0),
            mean_strategic.unsqueeze(0)
        ).item()

        euclidean_distance = (mean_moral - mean_strategic).norm().item()

        metadata = {
            "moral_model_id": moral_model_id,
            "strategic_model_id": strategic_model_id,
            "layer": layer,
            "component": component,
            "n_scenarios": len(scenarios),
            "cosine_similarity": cosine_similarity,
            "euclidean_distance": euclidean_distance,
            "vector_norm_before": steering_vector.norm().item(),
            "vector_norm_after": steering_vector_norm.norm().item(),
        }

        print(f"\n✓ Steering vector computed")
        print(f"  Cosine similarity: {cosine_similarity:.4f}")
        print(f"  Euclidean distance: {euclidean_distance:.2f}")
        print(f"  Vector norm (before normalization): {metadata['vector_norm_before']:.2f}")

        # Clean up
        del moral_model, strategic_model
        torch.cuda.empty_cache()

        return steering_vector_norm, metadata

    def steer_and_evaluate(
        self,
        model_id: str,
        layer: int,
        component: str,
        steering_vector: torch.Tensor,
        strength: float,
        scenarios: Optional[List[str]] = None,
    ) -> SteeringResult:
        """
        Apply steering and evaluate behavior change.

        Args:
            model_id: Model to steer
            layer: Layer to apply steering to
            component: Component to steer ("mlp" or "attn")
            steering_vector: Normalized steering vector
            strength: Steering strength (scalar multiplier)
            scenarios: Scenarios to evaluate on

        Returns:
            SteeringResult with baseline vs steered behavior
        """
        # Load model
        model = LoRAModelLoader.load_hooked_model(model_id, device=self.device)

        # Load IPD prompts
        prompts_data = load_ipd_prompts()
        if scenarios is None:
            scenarios = list(prompts_data.keys())

        hook_name = f"blocks.{layer}.hook_{component}_out"

        # Evaluate baseline (no steering)
        baseline_p_action2_per_scenario = []

        with torch.no_grad():
            for scenario_name in scenarios:
                prompt_text = prompts_data[scenario_name]["prompt"]
                formatted_prompt = prepare_prompt(model.tokenizer, prompt_text, use_chat_template=True)
                input_ids = model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

                def forward_fn(input_ids):
                    outputs = model.model(input_ids)
                    return outputs.logits

                pref = compute_action_sequence_preference(
                    forward_logits_fn=forward_fn,
                    tokenizer=model.tokenizer,
                    input_ids=input_ids,
                )

                baseline_p_action2_per_scenario.append(pref.p_action2)

        baseline_p_action2_mean = np.mean(baseline_p_action2_per_scenario)

        # Evaluate with steering
        steered_p_action2_per_scenario = []

        # Create steering hook
        steering_vec_device = steering_vector.to(self.device)

        def make_steering_hook(strength):
            def hook(module, input, output):
                # Handle both MLP (tensor) and attention (tuple) outputs
                if isinstance(output, tuple):
                    # Attention output: (attention_output, attention_weights, ...)
                    attn_out = output[0]
                    steered = attn_out.clone()
                    steered[:, -1, :] += strength * steering_vec_device
                    return (steered,) + output[1:]  # Re-pack tuple
                else:
                    # MLP output: tensor
                    steered = output.clone()
                    steered[:, -1, :] += strength * steering_vec_device
                    return steered
            return hook

        # Get the module to hook
        if component == "mlp":
            target_module = model.model.model.layers[layer].mlp
        elif component == "attn":
            target_module = model.model.model.layers[layer].self_attn
        else:
            raise ValueError(f"Unknown component: {component}")

        # Register hook
        handle = target_module.register_forward_hook(make_steering_hook(strength))

        with torch.no_grad():
            for scenario_name in scenarios:
                prompt_text = prompts_data[scenario_name]["prompt"]
                formatted_prompt = prepare_prompt(model.tokenizer, prompt_text, use_chat_template=True)
                input_ids = model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

                def forward_fn(input_ids):
                    outputs = model.model(input_ids)
                    return outputs.logits

                pref = compute_action_sequence_preference(
                    forward_logits_fn=forward_fn,
                    tokenizer=model.tokenizer,
                    input_ids=input_ids,
                )

                steered_p_action2_per_scenario.append(pref.p_action2)

        steered_p_action2_mean = np.mean(steered_p_action2_per_scenario)

        # Remove hook
        handle.remove()

        # Compute delta
        delta_p_action2 = steered_p_action2_mean - baseline_p_action2_mean

        # Clean up
        del model
        torch.cuda.empty_cache()

        return SteeringResult(
            model_id=model_id,
            layer=layer,
            component=component,
            steering_direction="moral→strategic",  # Assuming steering vector is moral - strategic
            steering_strength=strength,
            baseline_p_action2=baseline_p_action2_mean,
            steered_p_action2=steered_p_action2_mean,
            delta_p_action2=delta_p_action2,
            scenarios=scenarios,
            baseline_p_action2_per_scenario=baseline_p_action2_per_scenario,
            steered_p_action2_per_scenario=steered_p_action2_per_scenario,
        )

    def steering_sweep(
        self,
        model_id: str,
        layer: int,
        component: str,
        steering_vector: torch.Tensor,
        strengths: Optional[List[float]] = None,
        scenarios: Optional[List[str]] = None,
    ) -> SteeringSweepResult:
        """
        Sweep steering strength and measure behavior change.

        Args:
            model_id: Model to steer
            layer: Layer to steer
            component: Component to steer
            steering_vector: Normalized steering vector
            strengths: List of strengths to test (default: [-2, -1.5, ..., 0, ..., 2])
            scenarios: Scenarios to evaluate on

        Returns:
            SteeringSweepResult with strength vs behavior relationship
        """
        if strengths is None:
            strengths = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

        print(f"\n{'='*80}")
        print(f"Steering Sweep: {model_id}")
        print(f"Layer: {layer}, Component: {component}")
        print(f"Strengths: {strengths}")
        print(f"{'='*80}")

        # Load IPD prompts
        prompts_data = load_ipd_prompts()
        if scenarios is None:
            scenarios = list(prompts_data.keys())

        mean_p_action2_list = []
        p_action2_matrix = []

        for i, strength in enumerate(strengths, 1):
            print(f"\nTesting strength {i}/{len(strengths)}: {strength:+.1f}")

            result = self.steer_and_evaluate(
                model_id=model_id,
                layer=layer,
                component=component,
                steering_vector=steering_vector,
                strength=strength,
                scenarios=scenarios,
            )

            mean_p_action2_list.append(result.steered_p_action2)
            p_action2_matrix.append(result.steered_p_action2_per_scenario)

            print(f"  Mean p_action2: {result.steered_p_action2:.4f}")
            print(f"  Cooperation rate: {1 - result.steered_p_action2:.2%}")

        p_action2_matrix_np = np.array(p_action2_matrix)  # (n_strengths, n_scenarios)
        mean_coop_rate = [1 - p for p in mean_p_action2_list]

        print(f"\n✓ Steering sweep complete")

        return SteeringSweepResult(
            model_id=model_id,
            layer=layer,
            component=component,
            steering_direction="moral→strategic",
            strengths=strengths,
            mean_p_action2=mean_p_action2_list,
            mean_coop_rate=mean_coop_rate,
            scenarios=scenarios,
            p_action2_matrix=p_action2_matrix_np,
        )

    def downstream_effect_analysis(
        self,
        model_id: str,
        steering_layer: int,
        steering_component: str,
        steering_vector: torch.Tensor,
        strength: float,
        downstream_layers: Optional[List[int]] = None,
        scenarios: Optional[List[str]] = None,
    ) -> DownstreamEffectResult:
        """
        Analyze how steering L2_MLP affects downstream layers.

        Args:
            model_id: Model to analyze
            steering_layer: Layer to apply steering (e.g., 2 for L2_MLP)
            steering_component: Component to steer
            steering_vector: Normalized steering vector
            strength: Steering strength
            downstream_layers: Layers to measure effects on (default: [8, 9])
            scenarios: Scenarios to test on

        Returns:
            DownstreamEffectResult showing activation changes
        """
        if downstream_layers is None:
            downstream_layers = [8, 9]  # L8_MLP, L9_MLP

        print(f"\n{'='*80}")
        print(f"Downstream Effect Analysis: {model_id}")
        print(f"Steering: L{steering_layer}_{steering_component.upper()}, strength={strength:+.1f}")
        print(f"Measuring: Layers {downstream_layers}")
        print(f"{'='*80}")

        # Load model
        model = LoRAModelLoader.load_hooked_model(model_id, device=self.device)

        # Load IPD prompts
        prompts_data = load_ipd_prompts()
        if scenarios is None:
            scenarios = list(prompts_data.keys())[:5]  # Use subset for speed

        # Collect baseline activations (no steering)
        baseline_acts = {layer: [] for layer in downstream_layers}

        with torch.no_grad():
            for scenario_name in scenarios:
                prompt_text = prompts_data[scenario_name]["prompt"]
                formatted_prompt = prepare_prompt(model.tokenizer, prompt_text, use_chat_template=True)
                input_ids = model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

                _, cache = model.run_with_cache(input_ids)

                for layer in downstream_layers:
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                    act = cache[hook_name][0, -1, :].cpu()
                    baseline_acts[layer].append(act)

        # Collect steered activations
        steering_vec_device = steering_vector.to(self.device)

        def make_steering_hook(strength):
            def hook(module, input, output):
                # Handle both MLP (tensor) and attention (tuple) outputs
                if isinstance(output, tuple):
                    # Attention output: (attention_output, attention_weights, ...)
                    attn_out = output[0]
                    steered = attn_out.clone()
                    steered[:, -1, :] += strength * steering_vec_device
                    return (steered,) + output[1:]  # Re-pack tuple
                else:
                    # MLP output: tensor
                    steered = output.clone()
                    steered[:, -1, :] += strength * steering_vec_device
                    return steered
            return hook

        # Get steering module
        if steering_component == "mlp":
            steering_module = model.model.model.layers[steering_layer].mlp
        else:
            steering_module = model.model.model.layers[steering_layer].self_attn

        # Register steering hook
        handle = steering_module.register_forward_hook(make_steering_hook(strength))

        steered_acts = {layer: [] for layer in downstream_layers}

        with torch.no_grad():
            for scenario_name in scenarios:
                prompt_text = prompts_data[scenario_name]["prompt"]
                formatted_prompt = prepare_prompt(model.tokenizer, prompt_text, use_chat_template=True)
                input_ids = model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

                _, cache = model.run_with_cache(input_ids)

                for layer in downstream_layers:
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                    act = cache[hook_name][0, -1, :].cpu()
                    steered_acts[layer].append(act)

        # Remove hook
        handle.remove()

        # Compute activation changes
        activation_changes = []
        for layer in downstream_layers:
            baseline_tensor = torch.stack(baseline_acts[layer])  # (n_scenarios, hidden_dim)
            steered_tensor = torch.stack(steered_acts[layer])

            diff = steered_tensor - baseline_tensor
            mean_abs_change = diff.abs().mean().item()
            activation_changes.append(mean_abs_change)

            print(f"\nLayer {layer}:")
            print(f"  Mean absolute change: {mean_abs_change:.4f}")

        # Find top affected components (simplified - could use DLA here)
        top_affected_components = [f"L{layer}_MLP" for layer in downstream_layers]
        component_change_magnitudes = activation_changes

        # Clean up
        del model
        torch.cuda.empty_cache()

        print(f"\n✓ Downstream effect analysis complete")

        return DownstreamEffectResult(
            model_id=model_id,
            steering_layer=steering_layer,
            steering_strength=strength,
            layer_indices=downstream_layers,
            activation_changes=activation_changes,
            top_affected_components=top_affected_components,
            component_change_magnitudes=component_change_magnitudes,
        )


def save_steering_results(result: SteeringResult, output_dir: Path):
    """Save steering result to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "scenario": result.scenarios,
        "baseline_p_action2": result.baseline_p_action2_per_scenario,
        "steered_p_action2": result.steered_p_action2_per_scenario,
        "delta_p_action2": [
            s - b for s, b in zip(
                result.steered_p_action2_per_scenario,
                result.baseline_p_action2_per_scenario
            )
        ],
    })

    filename = f"steering_{result.model_id}_L{result.layer}_{result.component}_strength{result.steering_strength:+.1f}.csv"
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results: {csv_path}")


def save_sweep_results(result: SteeringSweepResult, output_dir: Path):
    """Save steering sweep results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "strength": result.strengths,
        "mean_p_action2": result.mean_p_action2,
        "mean_coop_rate": result.mean_coop_rate,
    })

    filename = f"steering_sweep_{result.model_id}_L{result.layer}_{result.component}.csv"
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved sweep results: {csv_path}")

    # Also save per-scenario matrix
    matrix_df = pd.DataFrame(
        result.p_action2_matrix,
        columns=result.scenarios,
        index=[f"strength_{s:+.1f}" for s in result.strengths],
    )
    matrix_filename = f"steering_sweep_{result.model_id}_L{result.layer}_{result.component}_matrix.csv"
    matrix_csv_path = output_dir / matrix_filename
    matrix_df.to_csv(matrix_csv_path)
    print(f"✓ Saved per-scenario matrix: {matrix_csv_path}")
