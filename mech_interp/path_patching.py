"""
Path Patching Experiment.

Tests whether specific pathways through the residual stream causally mediate
moral behavior differences. Unlike single-component patching (which showed zero
flips), path patching replaces entire pathways from L2 → L9 to test if
information flows through this route.

Key Functions:
- patch_residual_path: Replace residual activations from start to end layer
- progressive_path_patching: Find critical layer range
- isolate_mlp_vs_attn_path: Decompose pathway contributions
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
class PathPatchResult:
    """Results from patching a single pathway."""

    source_model_id: str
    target_model_id: str
    start_layer: int
    end_layer: int
    component_type: str  # "residual", "mlp_only", or "attn_only"

    # Behavioral results
    baseline_p_action2: float  # Target model without patching
    patched_p_action2: float   # Target model with source pathway
    delta_p_action2: float     # patched - baseline

    # Per-scenario results
    scenarios: List[str]
    baseline_p_action2_per_scenario: List[float]
    patched_p_action2_per_scenario: List[float]


@dataclass
class ProgressivePatchResult:
    """Results from progressive path patching."""

    source_model_id: str
    target_model_id: str
    start_layer: int
    component_type: str

    # Progressive results
    end_layers: List[int]
    mean_p_action2: List[float]
    mean_coop_rate: List[float]

    # Find saturation point
    saturation_layer: Optional[int]


class PathPatcher:
    """
    Patches pathways through the residual stream to test causal information flow.

    Unlike single-component activation patching, path patching replaces multiple
    layers to test if information flows through a specific route (e.g., L2→L9).
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize path patcher.

        Args:
            device: Device to load models on
        """
        self.device = device

    def patch_residual_path(
        self,
        source_model_id: str,
        target_model_id: str,
        start_layer: int,
        end_layer: int,
        component_type: str = "residual",
        scenarios: Optional[List[str]] = None,
    ) -> PathPatchResult:
        """
        Patch pathway from start_layer to end_layer.

        Information flow in transformers:
        - L2.mlp → L2.resid_post → L3.attn → L3.mlp → L3.resid_post → ... → L9

        This function replaces the residual stream activations at each layer
        from start to end with activations from the source model.

        Args:
            source_model_id: Model to copy activations FROM
            target_model_id: Model to copy activations TO
            start_layer: First layer to patch (inclusive)
            end_layer: Last layer to patch (inclusive)
            component_type: "residual" (full path), "mlp_only", or "attn_only"
            scenarios: Scenarios to evaluate on

        Returns:
            PathPatchResult with baseline vs patched behavior
        """
        print(f"\n{'='*80}")
        print(f"Path Patching: {source_model_id} → {target_model_id}")
        print(f"Layers: {start_layer} to {end_layer}, Type: {component_type}")
        print(f"{'='*80}")

        # Load models
        print(f"\nLoading source model: {source_model_id}")
        source_model = LoRAModelLoader.load_hooked_model(source_model_id, device=self.device)

        print(f"Loading target model: {target_model_id}")
        target_model = LoRAModelLoader.load_hooked_model(target_model_id, device=self.device)

        # Load IPD prompts
        prompts_data = load_ipd_prompts()
        if scenarios is None:
            scenarios = list(prompts_data.keys())

        # Evaluate baseline (target model without patching)
        print(f"\nEvaluating baseline (target model)...")
        baseline_p_action2_per_scenario = []

        with torch.no_grad():
            for scenario_name in scenarios:
                prompt_text = prompts_data[scenario_name]["prompt"]
                formatted_prompt = prepare_prompt(target_model.tokenizer, prompt_text, use_chat_template=True)
                input_ids = target_model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

                def forward_fn(input_ids):
                    outputs = target_model.model(input_ids)
                    return outputs.logits

                pref = compute_action_sequence_preference(
                    forward_logits_fn=forward_fn,
                    tokenizer=target_model.tokenizer,
                    input_ids=input_ids,
                )

                baseline_p_action2_per_scenario.append(pref.p_action2)

        baseline_p_action2_mean = np.mean(baseline_p_action2_per_scenario)
        print(f"  Baseline p(action2): {baseline_p_action2_mean:.4f}")

        # Evaluate with path patching
        print(f"\nEvaluating with path patching...")
        patched_p_action2_per_scenario = []

        # Create patch hooks
        handles = []

        for scenario_name in scenarios:
            # Cache source activations
            prompt_text = prompts_data[scenario_name]["prompt"]
            formatted_prompt = prepare_prompt(source_model.tokenizer, prompt_text, use_chat_template=True)
            input_ids = source_model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                _, source_cache = source_model.run_with_cache(input_ids)

            # Register hooks on target model
            for layer_idx in range(start_layer, end_layer + 1):
                if component_type == "residual":
                    # Patch full residual stream
                    hook_name = f"blocks.{layer_idx}.hook_resid_post"
                    source_activation = source_cache[hook_name][:, -1:, :]

                    def make_hook(src_act):
                        def hook(module, input, output):
                            # Layer outputs are tuples (hidden_states,)
                            hidden_states = output[0]
                            patched = hidden_states.clone()
                            patched[:, -1:, :] = src_act.to(patched.device)
                            return (patched,)
                        return hook

                    layer = target_model.model.model.layers[layer_idx]
                    handle = layer.register_forward_hook(make_hook(source_activation))
                    handles.append(handle)

                elif component_type == "mlp_only":
                    # Patch only MLP outputs
                    hook_name = f"blocks.{layer_idx}.hook_mlp_out"
                    source_activation = source_cache[hook_name][:, -1:, :]

                    def make_hook(src_act):
                        def hook(module, input, output):
                            patched = output.clone()
                            patched[:, -1:, :] = src_act.to(output.device)
                            return patched
                        return hook

                    layer = target_model.model.model.layers[layer_idx]
                    handle = layer.mlp.register_forward_hook(make_hook(source_activation))
                    handles.append(handle)

                elif component_type == "attn_only":
                    # Patch only attention outputs
                    hook_name = f"blocks.{layer_idx}.hook_attn_out"
                    source_activation = source_cache[hook_name][:, -1:, :]

                    def make_hook(src_act):
                        def hook(module, input, output):
                            # output is tuple (attn_output, attn_weights, past_key_value)
                            attn_output = output[0].clone()
                            attn_output[:, -1:, :] = src_act.to(attn_output.device)
                            return (attn_output,) + output[1:]
                        return hook

                    layer = target_model.model.model.layers[layer_idx]
                    handle = layer.self_attn.register_forward_hook(make_hook(source_activation))
                    handles.append(handle)

            # Evaluate with hooks active
            input_ids_target = target_model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                def forward_fn(input_ids):
                    outputs = target_model.model(input_ids)
                    return outputs.logits

                pref = compute_action_sequence_preference(
                    forward_logits_fn=forward_fn,
                    tokenizer=target_model.tokenizer,
                    input_ids=input_ids_target,
                )

                patched_p_action2_per_scenario.append(pref.p_action2)

            # Remove hooks
            for handle in handles:
                handle.remove()
            handles = []

        patched_p_action2_mean = np.mean(patched_p_action2_per_scenario)
        delta_p_action2 = patched_p_action2_mean - baseline_p_action2_mean

        print(f"  Patched p(action2): {patched_p_action2_mean:.4f}")
        print(f"  Δ p(action2): {delta_p_action2:+.4f}")

        # Clean up
        del source_model, target_model
        torch.cuda.empty_cache()

        return PathPatchResult(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            start_layer=start_layer,
            end_layer=end_layer,
            component_type=component_type,
            baseline_p_action2=baseline_p_action2_mean,
            patched_p_action2=patched_p_action2_mean,
            delta_p_action2=delta_p_action2,
            scenarios=scenarios,
            baseline_p_action2_per_scenario=baseline_p_action2_per_scenario,
            patched_p_action2_per_scenario=patched_p_action2_per_scenario,
        )

    def progressive_path_patching(
        self,
        source_model_id: str,
        target_model_id: str,
        start_layer: int,
        max_layer: int,
        component_type: str = "residual",
        scenarios: Optional[List[str]] = None,
    ) -> ProgressivePatchResult:
        """
        Progressively extend path length to find critical layer range.

        Tests paths: L2→L2, L2→L3, L2→L4, ..., L2→L9
        to identify where the effect saturates.

        Args:
            source_model_id: Model to copy from
            target_model_id: Model to copy to
            start_layer: Starting layer (e.g., 2 for L2_MLP)
            max_layer: Maximum end layer (e.g., 9 for L9_MLP)
            component_type: "residual", "mlp_only", or "attn_only"
            scenarios: Scenarios to evaluate on

        Returns:
            ProgressivePatchResult showing effect vs path length
        """
        print(f"\n{'='*80}")
        print(f"Progressive Path Patching: {source_model_id} → {target_model_id}")
        print(f"Start: L{start_layer}, Max: L{max_layer}, Type: {component_type}")
        print(f"{'='*80}")

        end_layers = list(range(start_layer, max_layer + 1))
        mean_p_action2_list = []
        mean_coop_rate_list = []

        for i, end_layer in enumerate(end_layers, 1):
            print(f"\n[{i}/{len(end_layers)}] Patching L{start_layer} → L{end_layer}")

            result = self.patch_residual_path(
                source_model_id=source_model_id,
                target_model_id=target_model_id,
                start_layer=start_layer,
                end_layer=end_layer,
                component_type=component_type,
                scenarios=scenarios,
            )

            mean_p_action2_list.append(result.patched_p_action2)
            mean_coop_rate_list.append(1 - result.patched_p_action2)

        # Find saturation point (where effect stops increasing)
        saturation_layer = None
        for i in range(len(mean_p_action2_list) - 1):
            delta = abs(mean_p_action2_list[i + 1] - mean_p_action2_list[i])
            if delta < 0.01:  # Less than 1% change
                saturation_layer = end_layers[i]
                break

        print(f"\n✓ Progressive patching complete")
        if saturation_layer:
            print(f"  Saturation at layer: {saturation_layer}")
        else:
            print(f"  No clear saturation point found")

        return ProgressivePatchResult(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            start_layer=start_layer,
            component_type=component_type,
            end_layers=end_layers,
            mean_p_action2=mean_p_action2_list,
            mean_coop_rate=mean_coop_rate_list,
            saturation_layer=saturation_layer,
        )


def save_path_patch_result(result: PathPatchResult, output_dir: Path):
    """Save path patching result to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "scenario": result.scenarios,
        "baseline_p_action2": result.baseline_p_action2_per_scenario,
        "patched_p_action2": result.patched_p_action2_per_scenario,
        "delta_p_action2": [
            p - b for p, b in zip(
                result.patched_p_action2_per_scenario,
                result.baseline_p_action2_per_scenario
            )
        ],
    })

    filename = f"path_patch_{result.source_model_id}_to_{result.target_model_id}_L{result.start_layer}-L{result.end_layer}_{result.component_type}.csv"
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved result: {csv_path}")


def save_progressive_result(result: ProgressivePatchResult, output_dir: Path):
    """Save progressive patching result to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "end_layer": result.end_layers,
        "mean_p_action2": result.mean_p_action2,
        "mean_coop_rate": result.mean_coop_rate,
    })

    filename = f"progressive_patch_{result.source_model_id}_to_{result.target_model_id}_L{result.start_layer}_{result.component_type}.csv"
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved progressive result: {csv_path}")
