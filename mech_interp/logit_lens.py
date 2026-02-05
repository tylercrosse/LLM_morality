"""
Logit Lens analysis for IPD models.

Analyzes how action preference (Cooperate vs Defect) evolves across layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from mech_interp.model_loader import HookedGemmaModel
from mech_interp.utils import get_action_token_ids, MODEL_LABELS, MODEL_COLORS
from mech_interp.decision_metrics import (
    prepare_prompt,
    compute_action_sequence_preference as compute_action_sequence_preference_shared,
)


class LogitLensAnalyzer:
    """Logit lens analysis for mechanistic interpretability."""

    def __init__(self, model: HookedGemmaModel, tokenizer, use_chat_template: bool = True):
        """
        Initialize logit lens analyzer.

        Args:
            model: HookedGemmaModel with activation caching
            tokenizer: HuggingFace tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.action_tokens = get_action_token_ids(tokenizer)

    def _prepare_prompt(self, prompt: str) -> str:
        """Mirror inference formatting by using Gemma chat templating."""
        return prepare_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
        )

    def compute_action_sequence_preference(
        self,
        prompt: str,
        action1_label: str = "action1",
        action2_label: str = "action2",
    ) -> Dict[str, float]:
        """
        Compare full sequence probabilities for action1 vs action2.

        Returns:
            Dictionary with log-probabilities and normalized two-way probabilities.
        """
        prepared_prompt = self._prepare_prompt(prompt)
        input_ids = self.tokenizer(prepared_prompt, return_tensors="pt").input_ids.to(self.model.device)

        pref = compute_action_sequence_preference_shared(
            forward_logits_fn=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            action1_label=action1_label,
            action2_label=action2_label,
        )

        return {
            "logp_action1": pref.logp_action1,
            "logp_action2": pref.logp_action2,
            "delta_logp_action2_minus_action1": pref.delta_logp_action2_minus_action1,
            "p_action1": pref.p_action1,
            "p_action2": pref.p_action2,
            "preferred_action": pref.preferred_action,
        }

    def compute_layer_logits(self, prompt: str) -> Tuple[torch.Tensor, Dict]:
        """
        Run model and extract logits at each layer.

        Args:
            prompt: Input prompt

        Returns:
            final_logits: Final layer logits (vocab_size,)
            layer_logits: Dict mapping layer_idx -> logits tensor
        """
        # Tokenize (optionally with chat template, matching inference pipeline)
        prepared_prompt = self._prepare_prompt(prompt)
        input_ids = self.tokenizer(prepared_prompt, return_tensors="pt").input_ids.to(self.model.device)

        # Run with cache
        final_logits, cache = self.model.run_with_cache(input_ids)

        # Extract logits at each layer
        layer_logits = {}
        for layer_idx in range(self.model.n_layers):
            cache_key = f"blocks.{layer_idx}.hook_resid_post"

            if cache_key not in cache:
                print(f"Warning: {cache_key} not in cache")
                continue

            # Get hidden state at last token position
            hidden = cache[cache_key][0, -1, :]  # (d_model,)

            # Apply unembed (with layer norm)
            logits = self.model.unembed(hidden)  # (vocab_size,)

            layer_logits[layer_idx] = logits.cpu()

        return final_logits[0, -1, :].cpu(), layer_logits

    def compute_action_trajectory(self, prompt: str) -> np.ndarray:
        """
        Compute action logit difference (Defect - Cooperate) across layers.

        Args:
            prompt: Input prompt

        Returns:
            delta_logits: Array of shape (n_layers,)
                Positive = prefers Defect, Negative = prefers Cooperate
        """
        final_logits, layer_logits = self.compute_layer_logits(prompt)

        c_token_id = self.action_tokens['C']
        d_token_id = self.action_tokens['D']

        delta_logits = []
        for layer_idx in range(self.model.n_layers):
            if layer_idx not in layer_logits:
                delta_logits.append(np.nan)
                continue

            logits = layer_logits[layer_idx]
            c_logit = logits[c_token_id].item()
            d_logit = logits[d_token_id].item()
            delta = d_logit - c_logit

            delta_logits.append(delta)
            
        return np.array(delta_logits)

    def compute_action_trajectory_with_steering(
        self,
        prompt: str,
        steering_layer: int,
        steering_component: str,
        steering_vector: torch.Tensor,
        steering_strength: float,
    ) -> np.ndarray:
        """
        Compute action logit difference across layers WITH steering applied.

        Args:
            prompt: Input prompt
            steering_layer: Layer to apply steering at
            steering_component: Component to steer ("mlp" or "attn")
            steering_vector: Normalized steering vector
            steering_strength: Steering strength scalar

        Returns:
            delta_logits: Array of shape (n_layers,)
                Positive = prefers Defect, Negative = prefers Cooperate
        """
        # Tokenize
        prepared_prompt = self._prepare_prompt(prompt)
        input_ids = self.tokenizer(prepared_prompt, return_tensors="pt").input_ids.to(self.model.device)

        # Create steering hook
        steering_vec_device = steering_vector.to(self.model.device)

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
            steering_module = self.model.model.model.layers[steering_layer].mlp
        elif steering_component == "attn":
            steering_module = self.model.model.model.layers[steering_layer].self_attn
        else:
            raise ValueError(f"Unknown component: {steering_component}")

        # Register steering hook
        handle = steering_module.register_forward_hook(make_steering_hook(steering_strength))

        # Run with cache (steering hook will be active)
        with torch.no_grad():
            final_logits, cache = self.model.run_with_cache(input_ids)

        # Remove hook
        handle.remove()

        # Extract logits at each layer (same as baseline)
        c_token_id = self.action_tokens['C']
        d_token_id = self.action_tokens['D']

        delta_logits = []
        for layer_idx in range(self.model.n_layers):
            cache_key = f"blocks.{layer_idx}.hook_resid_post"

            if cache_key not in cache:
                delta_logits.append(np.nan)
                continue

            # Get hidden state at last token position
            hidden = cache[cache_key][0, -1, :]  # (d_model,)

            # Apply unembed (with layer norm)
            logits = self.model.unembed(hidden)  # (vocab_size,)

            c_logit = logits[c_token_id].item()
            d_logit = logits[d_token_id].item()
            delta = d_logit - c_logit

            delta_logits.append(delta)

        return np.array(delta_logits)

    def analyze_decision_layer(self, trajectory: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Identify where the decision becomes stable.

        Args:
            trajectory: Delta logit trajectory across layers
            threshold: Minimum absolute value to consider decision made

        Returns:
            Dictionary with decision layer analysis
        """
        n_layers = len(trajectory)

        # Find first layer where |delta| > threshold
        decision_layers = np.where(np.abs(trajectory) > threshold)[0]
        first_decision_layer = decision_layers[0] if len(decision_layers) > 0 else None

        # Find where decision stabilizes (last major change)
        delta_changes = np.abs(np.diff(trajectory))
        if len(delta_changes) > 0:
            # Look for last significant change (> 20% of max change)
            significant_threshold = 0.2 * np.max(delta_changes)
            significant_changes = np.where(delta_changes > significant_threshold)[0]
            stabilization_layer = significant_changes[-1] + 1 if len(significant_changes) > 0 else n_layers - 1
        else:
            stabilization_layer = n_layers - 1

        return {
            "first_decision_layer": first_decision_layer,
            "stabilization_layer": stabilization_layer,
            "final_delta": trajectory[-1],
            "max_delta": np.max(np.abs(trajectory)),
            "mean_delta": np.mean(trajectory),
        }


class LogitLensVisualizer:
    """Visualization utilities for logit lens analysis."""

    @staticmethod
    def plot_single_trajectory(
        trajectory: np.ndarray,
        model_id: str,
        scenario: str,
        output_path: Path = None,
    ):
        """
        Plot trajectory for a single model and scenario.

        Args:
            trajectory: Delta logit trajectory (n_layers,)
            model_id: Model identifier
            scenario: Scenario name
            output_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            range(len(trajectory)),
            trajectory,
            color=MODEL_COLORS[model_id],
            linewidth=2.5,
            marker='o',
            markersize=4,
        )

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel("Logit(Defect) - Logit(Cooperate)", fontsize=14)
        ax.set_title(
            f"Logit Lens: {MODEL_LABELS[model_id]} - {scenario}",
            fontsize=16,
        )
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_model_comparison(
        trajectories: Dict[str, np.ndarray],
        scenario: str,
        output_path: Path = None,
        title_suffix: str = "",
    ):
        """
        Plot trajectories for all models on one scenario.

        Args:
            trajectories: Dict mapping model_id -> trajectory
            scenario: Scenario name
            output_path: Path to save plot (optional)
            title_suffix: Additional text for title
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        for model_id, trajectory in trajectories.items():
            ax.plot(
                range(len(trajectory)),
                trajectory,
                label=MODEL_LABELS[model_id],
                color=MODEL_COLORS[model_id],
                linewidth=2.5,
                marker='o',
                markersize=3,
                alpha=0.8,
            )

        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel("Logit(Defect) - Logit(Cooperate)", fontsize=14)

        title = f"Logit Lens: {scenario}"
        if title_suffix:
            title += f" {title_suffix}"
        ax.set_title(title, fontsize=16, pad=15)

        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_scenario_grid(
        all_trajectories: Dict[str, Dict[str, np.ndarray]],
        scenarios: List[str],
        output_path: Path = None,
    ):
        """
        Create grid of plots for multiple scenarios.

        Args:
            all_trajectories: Dict[scenario -> Dict[model_id -> trajectory]]
            scenarios: List of scenario names to plot
            output_path: Path to save plot (optional)
        """
        n_scenarios = len(scenarios)
        ncols = 2
        nrows = (n_scenarios + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
        axes = axes.flatten() if n_scenarios > 1 else [axes]

        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]

            if scenario not in all_trajectories:
                ax.text(0.5, 0.5, f"No data for {scenario}", ha='center', va='center')
                continue

            trajectories = all_trajectories[scenario]

            for model_id, trajectory in trajectories.items():
                ax.plot(
                    range(len(trajectory)),
                    trajectory,
                    label=MODEL_LABELS[model_id],
                    color=MODEL_COLORS[model_id],
                    linewidth=2,
                    alpha=0.8,
                )

            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Δ Logit (D - C)", fontsize=12)
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=14)
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)

        # Hide extra subplots
        for idx in range(n_scenarios, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_final_comparison_heatmap(
        final_deltas: Dict[str, Dict[str, float]],
        output_path: Path = None,
        *,
        title: str = "Final Layer Preferences (Heatmap)",
        colorbar_label: str = "Logit(Defect) - Logit(Cooperate)",
    ):
        """
        Create heatmap of final layer preferences across models and scenarios.

        Args:
            final_deltas: Dict[scenario -> Dict[model_id -> final_delta]]
            output_path: Path to save plot (optional)
        """
        # Convert to matrix
        scenarios = list(final_deltas.keys())
        models = list(MODEL_LABELS.keys())

        matrix = np.zeros((len(scenarios), len(models)))
        for i, scenario in enumerate(scenarios):
            for j, model_id in enumerate(models):
                if model_id in final_deltas[scenario]:
                    matrix[i, j] = final_deltas[scenario][model_id]
                else:
                    matrix[i, j] = np.nan

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            matrix,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": colorbar_label},
            xticklabels=[MODEL_LABELS[m] for m in models],
            yticklabels=[s.replace('_', ' ').title() for s in scenarios],
            ax=ax,
        )

        ax.set_title(title, fontsize=16, pad=15)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_steering_comparison(
        baseline_trajectory: np.ndarray,
        steered_trajectory: np.ndarray,
        model_id: str,
        scenario: str,
        steering_layer: int,
        steering_component: str,
        steering_strength: float,
        output_path: Path = None,
    ):
        """
        Plot baseline vs steered trajectories side-by-side.

        Args:
            baseline_trajectory: Delta logit trajectory without steering (n_layers,)
            steered_trajectory: Delta logit trajectory with steering (n_layers,)
            model_id: Model identifier
            scenario: Scenario name
            steering_layer: Layer where steering was applied
            steering_component: Component steered ("mlp" or "attn")
            steering_strength: Steering strength used
            output_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        n_layers = len(baseline_trajectory)
        layers = np.arange(n_layers)

        # Plot baseline
        ax.plot(
            layers,
            baseline_trajectory,
            label='Baseline (no steering)',
            color='steelblue',
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.8,
        )

        # Plot steered
        ax.plot(
            layers,
            steered_trajectory,
            label=f'Steered (L{steering_layer}_{steering_component.upper()}, strength={steering_strength:+.1f})',
            color='coral',
            linewidth=2.5,
            marker='s',
            markersize=4,
            alpha=0.8,
        )

        # Mark steering layer with vertical line
        ax.axvline(
            steering_layer,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label=f'Steering applied at L{steering_layer}'
        )

        # Zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Labels and formatting
        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel("Logit(Defect) - Logit(Cooperate)", fontsize=14)
        ax.set_title(
            f"Logit Lens with Steering: {MODEL_LABELS.get(model_id, model_id)} - {scenario}",
            fontsize=16,
        )
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)

        # Add text annotation showing final delta
        baseline_final = baseline_trajectory[-1]
        steered_final = steered_trajectory[-1]
        delta_effect = steered_final - baseline_final

        ax.text(
            0.02, 0.98,
            f'Final Δ: {baseline_final:.2f} → {steered_final:.2f} (Δ = {delta_effect:+.2f})',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Example usage of logit lens analyzer."""
    from mech_interp.model_loader import LoRAModelLoader
    from mech_interp.prompt_generator import IPDPromptGenerator

    # Load prompt dataset
    prompt_path = "/root/LLM_morality/mech_interp_outputs/prompt_datasets/ipd_eval_prompts.json"
    dataset = IPDPromptGenerator.load_dataset(prompt_path)

    # Test with one model and one scenario
    print("Testing logit lens with base model on CC_continue scenario...\n")

    model_id = "base"
    scenario = "CC_continue"

    # Get first prompt for this scenario
    prompts = IPDPromptGenerator.get_prompts_by_scenario(dataset, scenario)
    test_prompt = prompts[0]['prompt']

    # Load model
    model = LoRAModelLoader.load_hooked_model(model_id)
    tokenizer = LoRAModelLoader.load_tokenizer()

    # Analyze
    analyzer = LogitLensAnalyzer(model, tokenizer)
    trajectory = analyzer.compute_action_trajectory(test_prompt)

    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Final delta: {trajectory[-1]:.2f}")

    # Analyze decision layer
    decision_info = analyzer.analyze_decision_layer(trajectory)
    print(f"\nDecision analysis:")
    print(f"  First decision layer: {decision_info['first_decision_layer']}")
    print(f"  Stabilization layer: {decision_info['stabilization_layer']}")
    print(f"  Max |delta|: {decision_info['max_delta']:.2f}")

    # Plot
    output_dir = Path("/root/LLM_morality/mech_interp_outputs/logit_lens")
    output_dir.mkdir(parents=True, exist_ok=True)

    LogitLensVisualizer.plot_single_trajectory(
        trajectory,
        model_id,
        scenario,
        output_dir / f"test_{model_id}_{scenario}.png"
    )

    print(f"\n✓ Test plot saved to: {output_dir}/test_{model_id}_{scenario}.png")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
