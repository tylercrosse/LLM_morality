"""
Activation Patching for mechanistic interpretability.

Identifies causal components by patching activations from one model into another
and measuring behavioral changes. This reveals which components are necessary
and sufficient for moral vs strategic decision-making.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from mech_interp.decision_metrics import (
    prepare_prompt,
    compute_action_sequence_preference,
)


@dataclass
class PatchResult:
    """Results from a single activation patch."""

    # Patch configuration
    source_model: str
    target_model: str
    patched_component: str  # e.g., "L5H3" or "L12_MLP"
    scenario: str

    # Baseline behavior (primary: sequence preference)
    baseline_delta: float  # ΔlogP(action2-action1)
    baseline_action: str  # "Cooperate" or "Defect" from sequence preference
    baseline_p_action2: float

    # Legacy baseline (single-token digit delta)
    baseline_delta_legacy: float

    # Patched behavior
    patched_delta: float
    patched_action: str
    patched_p_action2: float

    # Legacy patched behavior (single-token digit delta)
    patched_delta_legacy: float

    # Behavioral change
    delta_change: float  # patched_delta - baseline_delta
    delta_change_legacy: float  # patched_delta_legacy - baseline_delta_legacy
    action_flipped: bool  # Did the action choice change?

    # Effect size
    effect_size: float  # Normalized change
    effect_size_legacy: float


@dataclass
class CircuitDiscovery:
    """Results from systematic circuit discovery."""

    # Components ranked by causal effect
    ranked_components: List[Tuple[str, float]]  # (component, effect_size)

    # Minimal circuit (components whose combined patch flips behavior)
    minimal_circuit: List[str]

    # Full results
    all_patches: List[PatchResult]


class ActivationPatcher:
    """
    Performs activation patching between models to identify causal components.

    Key idea: Replace specific components in target model with activations from
    source model, then measure how much the output changes.
    """

    def __init__(
        self,
        source_model,
        target_model,
        action_tokens: Dict,
        device: str = "cuda",
        use_chat_template: bool = True,
        action1_label: str = "action1",
        action2_label: str = "action2",
    ):
        """
        Args:
            source_model: HookedGemmaModel to take activations from
            target_model: HookedGemmaModel to patch activations into
            action_tokens: Dictionary with 'C' and 'D' token IDs
            device: Device to run on
        """
        self.source_model = source_model
        self.target_model = target_model
        self.action_tokens = action_tokens
        self.device = device
        self.use_chat_template = use_chat_template
        self.action1_label = action1_label
        self.action2_label = action2_label

        # Cache for source activations
        self.source_cache = None

    def _prepare_input_ids(self, model, prompt: str) -> torch.Tensor:
        prepared = prepare_prompt(
            model.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
        )
        return model.tokenizer(prepared, return_tensors="pt").input_ids.to(self.device)

    def _compute_legacy_token_delta(self, model, input_ids: torch.Tensor) -> float:
        """Legacy single-token decision metric: logit(action2_digit)-logit(action1_digit)."""
        with torch.no_grad():
            logits = model(input_ids)
        final_logits = logits[0, -1, :]
        c_token = self.action_tokens['C']
        d_token = self.action_tokens['D']
        return float(final_logits[d_token].item() - final_logits[c_token].item())

    def _compute_sequence_behavior(self, model, input_ids: torch.Tensor) -> Tuple[float, str, float]:
        pref = compute_action_sequence_preference(
            forward_logits_fn=model,
            tokenizer=self.target_model.tokenizer,
            input_ids=input_ids,
            action1_label=self.action1_label,
            action2_label=self.action2_label,
        )
        action = "Defect" if pref.preferred_action == self.action2_label else "Cooperate"
        return (
            float(pref.delta_logp_action2_minus_action1),
            action,
            float(pref.p_action2),
        )

    def get_baseline_behavior(self, prompt: str) -> Tuple[float, str]:
        """
        Get target model's baseline behavior (no patching).

        Returns:
            (delta_sequence_preference, action_choice)
        """
        input_ids = self._prepare_input_ids(self.target_model, prompt)
        delta, action, _ = self._compute_sequence_behavior(self.target_model, input_ids)
        return delta, action

    def get_baseline_behavior_full(self, prompt: str) -> Dict[str, float]:
        """Get baseline sequence and legacy token-level metrics."""
        input_ids = self._prepare_input_ids(self.target_model, prompt)
        delta, action, p_action2 = self._compute_sequence_behavior(self.target_model, input_ids)
        legacy_delta = self._compute_legacy_token_delta(self.target_model, input_ids)
        return {
            "baseline_delta": delta,
            "baseline_action": action,
            "baseline_p_action2": p_action2,
            "baseline_delta_legacy": legacy_delta,
        }

    def cache_source_activations(self, prompt: str):
        """Run source model and cache all activations."""
        input_ids = self._prepare_input_ids(self.source_model, prompt)

        with torch.no_grad():
            _, cache = self.source_model.run_with_cache(input_ids)

        self.source_cache = cache

    def patch_component(
        self,
        prompt: str,
        component: str
    ) -> Tuple[float, str, float, float]:
        """
        Patch a single component and measure output.

        Args:
            prompt: Input prompt
            component: Component to patch (e.g., "L5H3" or "L12_MLP")

        Returns:
            (patched_delta_seq, patched_action, patched_p_action2, patched_delta_legacy)
        """
        # Parse component specification
        if "_MLP" in component:
            layer_idx = int(component.split("_")[0][1:])
            component_type = "mlp"
            head_idx = None
        else:
            # Format: "L5H3"
            parts = component[1:].split("H")
            layer_idx = int(parts[0])
            head_idx = int(parts[1])
            component_type = "head"

        # Get source activation to patch in
        if component_type == "mlp":
            cache_key = f"blocks.{layer_idx}.hook_mlp_out"
            source_activation_last = self.source_cache[cache_key][:, -1:, :]
        else:
            cache_key = f"blocks.{layer_idx}.hook_attn_out"
            source_activation_last = self.source_cache[cache_key][:, -1:, :]

        # Run target model with patching hook
        input_ids = self._prepare_input_ids(self.target_model, prompt)

        # Register patching hook
        hook_handles = []

        if component_type == "mlp":
            layer = self.target_model.transformer_stack.layers[layer_idx]

            def patch_hook(module, input, output):
                # Patch only the final position to keep sequence-length compatibility.
                patched = output.clone()
                patched[:, -1:, :] = source_activation_last.to(output.device, dtype=output.dtype)
                return patched

            # Hook the MLP module
            handle = layer.mlp.register_forward_hook(patch_hook)
            hook_handles.append(handle)

        else:
            # For attention heads, we need to patch at the attention output level
            # This is a simplification - ideally we'd patch individual head outputs
            layer = self.target_model.transformer_stack.layers[layer_idx]

            def patch_hook(module, input, output):
                # Replace attention output at final position only.
                attn_out = output[0]
                patched = attn_out.clone()
                patched[:, -1:, :] = source_activation_last.to(attn_out.device, dtype=attn_out.dtype)
                return (patched,) + output[1:]

            handle = layer.self_attn.register_forward_hook(patch_hook)
            hook_handles.append(handle)

        # Run forward passes with patching active (sequence + legacy metrics)
        patched_delta, patched_action, patched_p_action2 = self._compute_sequence_behavior(
            self.target_model,
            input_ids,
        )
        patched_delta_legacy = self._compute_legacy_token_delta(self.target_model, input_ids)

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        return patched_delta, patched_action, patched_p_action2, patched_delta_legacy

    def systematic_patch(
        self,
        prompt: str,
        scenario: str,
        components: List[str] = None
    ) -> List[PatchResult]:
        """
        Systematically patch all components and measure effects.

        Args:
            prompt: Input prompt
            scenario: Scenario name
            components: List of components to patch (None = all)

        Returns:
            List of PatchResult objects
        """
        # Get baseline
        baseline = self.get_baseline_behavior_full(prompt)
        baseline_delta = baseline["baseline_delta"]
        baseline_action = baseline["baseline_action"]
        baseline_p_action2 = baseline["baseline_p_action2"]
        baseline_delta_legacy = baseline["baseline_delta_legacy"]

        # Cache source activations
        self.cache_source_activations(prompt)

        # Generate component list if not provided
        if components is None:
            components = []
            # Add all heads
            for layer in range(self.target_model.n_layers):
                for head in range(self.target_model.n_heads):
                    components.append(f"L{layer}H{head}")
            # Add all MLPs
            for layer in range(self.target_model.n_layers):
                components.append(f"L{layer}_MLP")

        # Patch each component
        results = []

        for component in tqdm(components, desc="Patching components", leave=False):
            try:
                patched_delta, patched_action, patched_p_action2, patched_delta_legacy = self.patch_component(prompt, component)

                delta_change = patched_delta - baseline_delta
                delta_change_legacy = patched_delta_legacy - baseline_delta_legacy
                action_flipped = (patched_action != baseline_action)

                # Compute effect size (normalized by baseline)
                effect_size = abs(delta_change) / (abs(baseline_delta) + 1e-6)
                effect_size_legacy = abs(delta_change_legacy) / (abs(baseline_delta_legacy) + 1e-6)

                result = PatchResult(
                    source_model=getattr(self.source_model, 'name', 'unknown'),
                    target_model=getattr(self.target_model, 'name', 'unknown'),
                    patched_component=component,
                    scenario=scenario,
                    baseline_delta=baseline_delta,
                    baseline_action=baseline_action,
                    baseline_p_action2=baseline_p_action2,
                    baseline_delta_legacy=baseline_delta_legacy,
                    patched_delta=patched_delta,
                    patched_action=patched_action,
                    patched_p_action2=patched_p_action2,
                    patched_delta_legacy=patched_delta_legacy,
                    delta_change=delta_change,
                    delta_change_legacy=delta_change_legacy,
                    action_flipped=action_flipped,
                    effect_size=effect_size,
                    effect_size_legacy=effect_size_legacy,
                )

                results.append(result)

            except Exception as e:
                print(f"  Warning: Failed to patch {component}: {e}")
                continue

        return results

    def discover_minimal_circuit(
        self,
        prompt: str,
        scenario: str,
        max_components: int = 10
    ) -> CircuitDiscovery:
        """
        Discover minimal circuit of components that flip behavior.

        Uses greedy search: iteratively add the component with largest effect
        until behavior flips or max_components reached.

        Args:
            prompt: Input prompt
            scenario: Scenario name
            max_components: Maximum circuit size

        Returns:
            CircuitDiscovery with minimal circuit and ranked components
        """
        # Get systematic patching results
        all_results = self.systematic_patch(prompt, scenario)

        # Rank components by effect size
        ranked = sorted(
            all_results,
            key=lambda x: abs(x.delta_change),
            reverse=True
        )
        ranked_components = [(r.patched_component, r.delta_change) for r in ranked]

        # Greedy search for minimal circuit
        baseline_delta, baseline_action = self.get_baseline_behavior(prompt)
        self.cache_source_activations(prompt)

        minimal_circuit = []
        current_delta = baseline_delta

        for component, _ in ranked_components[:max_components]:
            # Try adding this component
            test_circuit = minimal_circuit + [component]

            # Patch all components in circuit simultaneously
            patched_delta = self._patch_multiple_components(prompt, test_circuit)

            # Check if behavior flipped
            patched_action = "Defect" if patched_delta > 0 else "Cooperate"

            if patched_action != baseline_action:
                # Found minimal circuit!
                minimal_circuit = test_circuit
                break
            else:
                # Add to circuit and continue
                minimal_circuit = test_circuit
                current_delta = patched_delta

        return CircuitDiscovery(
            ranked_components=ranked_components,
            minimal_circuit=minimal_circuit,
            all_patches=all_results
        )

    def _patch_multiple_components(
        self,
        prompt: str,
        components: List[str]
    ) -> float:
        """
        Patch multiple components simultaneously.

        Args:
            prompt: Input prompt
            components: List of components to patch

        Returns:
            Patched sequence preference delta
        """
        input_ids = self._prepare_input_ids(self.target_model, prompt)

        hook_handles = []

        # Register hooks for all components
        for component in components:
            if "_MLP" in component:
                layer_idx = int(component.split("_")[0][1:])
                cache_key = f"blocks.{layer_idx}.hook_mlp_out"
                source_activation_last = self.source_cache[cache_key][:, -1:, :]

                layer = self.target_model.transformer_stack.layers[layer_idx]

                def make_patch_hook(src_act):
                    def hook(module, input, output):
                        patched = output.clone()
                        patched[:, -1:, :] = src_act.to(output.device, dtype=output.dtype)
                        return patched
                    return hook

                handle = layer.mlp.register_forward_hook(make_patch_hook(source_activation_last))
                hook_handles.append(handle)

            else:
                parts = component[1:].split("H")
                layer_idx = int(parts[0])
                cache_key = f"blocks.{layer_idx}.hook_attn_out"
                source_activation_last = self.source_cache[cache_key][:, -1:, :]

                layer = self.target_model.transformer_stack.layers[layer_idx]

                def make_patch_hook(src_act):
                    def hook(module, input, output):
                        attn_out = output[0]
                        patched = attn_out.clone()
                        patched[:, -1:, :] = src_act.to(attn_out.device, dtype=attn_out.dtype)
                        return (patched,) + output[1:]
                    return hook

                handle = layer.self_attn.register_forward_hook(make_patch_hook(source_activation_last))
                hook_handles.append(handle)

        # Run forward pass via sequence preference metric
        patched_delta, _, _ = self._compute_sequence_behavior(self.target_model, input_ids)

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        return patched_delta


class PatchingVisualizer:
    """Visualization utilities for activation patching results."""

    def plot_patch_heatmap(
        self,
        results: List[PatchResult],
        metric: str = "delta_change",
        ax=None,
        title: str = None
    ):
        """
        Plot heatmap of patching effects.

        Args:
            results: List of PatchResult objects
            metric: Metric to visualize ('delta_change', 'effect_size')
            ax: Matplotlib axis
            title: Plot title
        """
        import matplotlib.pyplot as plt

        # Extract head results only (for clean heatmap)
        head_results = [r for r in results if "H" in r.patched_component and "_MLP" not in r.patched_component]

        if not head_results:
            print("No head results to plot")
            return None

        # Parse components into (layer, head) coordinates
        coords = []
        values = []
        for r in head_results:
            parts = r.patched_component[1:].split("H")
            layer = int(parts[0])
            head = int(parts[1])
            coords.append((layer, head))

            if metric == "delta_change":
                values.append(r.delta_change)
            elif metric == "effect_size":
                values.append(r.effect_size)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Get dimensions
        max_layer = max(c[0] for c in coords)
        max_head = max(c[1] for c in coords)

        # Create matrix
        matrix = np.zeros((max_layer + 1, max_head + 1))
        for (layer, head), value in zip(coords, values):
            matrix[layer, head] = value

        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax)

        if metric == "delta_change":
            cbar.set_label('Δ Sequence Preference (patched - baseline)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Effect Size', rotation=270, labelpad=20)

        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        ax.set_title(title or f"Patching Effects: {metric}")

        return ax

    def plot_top_components(
        self,
        results: List[PatchResult],
        top_k: int = 20,
        ax=None,
        title: str = None
    ):
        """Plot top-k components by effect size."""
        import matplotlib.pyplot as plt

        # Sort by absolute effect
        sorted_results = sorted(
            results,
            key=lambda x: abs(x.delta_change),
            reverse=True
        )[:top_k]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Extract data
        components = [r.patched_component for r in sorted_results]
        effects = [r.delta_change for r in sorted_results]
        flipped = [r.action_flipped for r in sorted_results]

        # Plot
        colors = ['red' if f else 'blue' for f in flipped]
        y_pos = np.arange(len(components))

        ax.barh(y_pos, effects, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.set_xlabel('Δ Sequence Preference')
        ax.set_title(title or f"Top-{top_k} Components by Patching Effect")
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Action flipped'),
            Patch(facecolor='blue', alpha=0.7, label='No flip')
        ]
        ax.legend(handles=legend_elements, loc='best')

        return ax

    def plot_circuit_discovery(
        self,
        discovery: CircuitDiscovery,
        ax=None,
        title: str = None
    ):
        """Visualize minimal circuit discovery."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Plot ranked components
        components = [c[0] for c in discovery.ranked_components[:30]]
        effects = [c[1] for c in discovery.ranked_components[:30]]

        # Highlight minimal circuit components
        colors = ['red' if c in discovery.minimal_circuit else 'gray' for c in components]

        y_pos = np.arange(len(components))
        ax.barh(y_pos, effects, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components, fontsize=8)
        ax.set_xlabel('Δ Sequence Preference')
        ax.set_title(title or f"Circuit Discovery (Minimal: {len(discovery.minimal_circuit)} components)")
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Minimal circuit'),
            Patch(facecolor='gray', alpha=0.7, label='Other components')
        ]
        ax.legend(handles=legend_elements, loc='best')

        return ax


def export_patching_results(
    results: List[PatchResult],
    output_path: str
):
    """Export patching results to CSV."""
    rows = []
    for r in results:
        rows.append({
            'source_model': r.source_model,
            'target_model': r.target_model,
            'component': r.patched_component,
            'scenario': r.scenario,
            'baseline_delta': r.baseline_delta,
            'baseline_action': r.baseline_action,
            'baseline_p_action2': r.baseline_p_action2,
            'baseline_delta_legacy': r.baseline_delta_legacy,
            'patched_delta': r.patched_delta,
            'patched_action': r.patched_action,
            'patched_p_action2': r.patched_p_action2,
            'patched_delta_legacy': r.patched_delta_legacy,
            'delta_change': r.delta_change,
            'delta_change_legacy': r.delta_change_legacy,
            'action_flipped': r.action_flipped,
            'effect_size': r.effect_size,
            'effect_size_legacy': r.effect_size_legacy,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return df
