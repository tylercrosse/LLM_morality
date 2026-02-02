"""
Direct Logit Attribution (DLA) for mechanistic interpretability.

Decomposes final action logits into per-component (attention head, MLP) contributions
to identify which parts of the model drive cooperation vs defection decisions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class DLAResult:
    """Results from Direct Logit Attribution analysis."""

    # Per-head contributions: [n_layers, n_heads]
    head_contributions: np.ndarray

    # Per-MLP contributions: [n_layers]
    mlp_contributions: np.ndarray

    # Residual/other contributions
    residual_contribution: float

    # Final logit difference (Defect - Cooperate)
    final_delta: float

    # Metadata
    model_name: str
    prompt_text: str
    scenario: str


class DirectLogitAttributor:
    """
    Performs direct logit attribution to decompose final action logits.

    Key idea: Project each component's output through the unembedding matrix
    to measure its contribution to the final Cooperate vs Defect decision.
    """

    def __init__(self, hooked_model, action_tokens: Dict):
        """
        Args:
            hooked_model: HookedGemmaModel instance
            action_tokens: Dictionary with 'C' and 'D' token IDs
        """
        self.model = hooked_model
        self.action_tokens = action_tokens
        self.device = hooked_model.device if hooked_model is not None else "cpu"

    def decompose_logits(self, prompt: str) -> DLAResult:
        """
        Decompose final action logits into per-component contributions.

        Returns:
            DLAResult with per-head and per-MLP attributions
        """
        # Run forward pass with caching
        input_ids = self.model.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(self.device)

        final_logits, cache = self.model.run_with_cache(input_ids)

        # Get final position logits
        final_pos_logits = final_logits[0, -1, :]

        # Extract action token logits
        c_token = self.action_tokens['C']
        d_token = self.action_tokens['D']

        c_logit = final_pos_logits[c_token].item()
        d_logit = final_pos_logits[d_token].item()
        final_delta = d_logit - c_logit

        # Decompose into per-component contributions
        head_contribs, mlp_contribs, residual = self._attribute_components(
            cache, c_token, d_token
        )

        return DLAResult(
            head_contributions=head_contribs,
            mlp_contributions=mlp_contribs,
            residual_contribution=residual,
            final_delta=final_delta,
            model_name=getattr(self.model, 'name', 'unknown'),
            prompt_text=prompt,
            scenario='unknown'
        )

    def _attribute_components(
        self,
        cache: Dict,
        c_token: int,
        d_token: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Attribute logit difference to individual components.

        For Gemma-2:
        - Attention has 8 heads per layer
        - We need to decompose attention output into per-head contributions

        Returns:
            head_contributions: [n_layers, n_heads] - delta logit per head
            mlp_contributions: [n_layers] - delta logit per MLP
            residual: Unexplained contribution (should be small)
        """
        n_layers = self.model.n_layers
        n_heads = self.model.n_heads

        head_contribs = np.zeros((n_layers, n_heads))
        mlp_contribs = np.zeros(n_layers)

        # Get unembedding matrix and final layer norm
        W_U = self.model.W_U  # [vocab_size, d_model]
        ln_final = self.model.ln_final

        # Extract action direction in unembed space
        c_direction = W_U[c_token, :]  # [d_model]
        d_direction = W_U[d_token, :]  # [d_model]
        action_direction = d_direction - c_direction  # Defect - Cooperate

        for layer_idx in range(n_layers):
            # Get cached activations (at final token position)
            attn_out_key = f"blocks.{layer_idx}.hook_attn_out"
            mlp_out_key = f"blocks.{layer_idx}.hook_mlp_out"

            if attn_out_key in cache:
                attn_out = cache[attn_out_key][0, -1, :]  # [d_model]

                # Decompose attention into per-head contributions
                # This requires accessing individual head outputs
                # For now, we'll attribute the full attention output
                # (per-head decomposition requires deeper model access)
                attn_normalized = ln_final(attn_out.unsqueeze(0)).squeeze(0)
                attn_contrib = torch.dot(attn_normalized, action_direction).item()

                # Distribute evenly across heads (approximation)
                # TODO: Implement per-head decomposition
                head_contribs[layer_idx, :] = attn_contrib / n_heads

            if mlp_out_key in cache:
                mlp_out = cache[mlp_out_key][0, -1, :]  # [d_model]
                mlp_normalized = ln_final(mlp_out.unsqueeze(0)).squeeze(0)
                mlp_contrib = torch.dot(mlp_normalized, action_direction).item()
                mlp_contribs[layer_idx] = mlp_contrib

        # Compute residual (should be close to 0 if attribution is complete)
        total_attributed = head_contribs.sum() + mlp_contribs.sum()

        # Get actual final delta for comparison
        # Note: We need to compute this from the full residual stream
        residual = 0.0  # Placeholder

        return head_contribs, mlp_contribs, residual

    def identify_top_components(
        self,
        result: DLAResult,
        top_k: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify top-k components driving Cooperate vs Defect.

        Returns:
            Dictionary with 'pro_defect' and 'pro_cooperate' lists
            Each list contains (component_name, contribution) tuples
        """
        components = []

        # Add head contributions
        for layer_idx in range(result.head_contributions.shape[0]):
            for head_idx in range(result.head_contributions.shape[1]):
                contrib = result.head_contributions[layer_idx, head_idx]
                components.append((f"L{layer_idx}H{head_idx}", contrib))

        # Add MLP contributions
        for layer_idx in range(result.mlp_contributions.shape[0]):
            contrib = result.mlp_contributions[layer_idx]
            components.append((f"L{layer_idx}_MLP", contrib))

        # Sort by contribution (positive = pro-Defect, negative = pro-Cooperate)
        components.sort(key=lambda x: x[1], reverse=True)

        return {
            'pro_defect': components[:top_k],
            'pro_cooperate': components[-top_k:][::-1]  # Reverse to show most negative first
        }

    def export_to_dataframe(self, result: DLAResult) -> pd.DataFrame:
        """Export DLA results to a pandas DataFrame for analysis."""
        rows = []

        # Add head contributions
        for layer_idx in range(result.head_contributions.shape[0]):
            for head_idx in range(result.head_contributions.shape[1]):
                rows.append({
                    'model': result.model_name,
                    'scenario': result.scenario,
                    'component_type': 'head',
                    'layer': layer_idx,
                    'head': head_idx,
                    'component': f"L{layer_idx}H{head_idx}",
                    'contribution': result.head_contributions[layer_idx, head_idx]
                })

        # Add MLP contributions
        for layer_idx in range(result.mlp_contributions.shape[0]):
            rows.append({
                'model': result.model_name,
                'scenario': result.scenario,
                'component_type': 'mlp',
                'layer': layer_idx,
                'head': -1,
                'component': f"L{layer_idx}_MLP",
                'contribution': result.mlp_contributions[layer_idx]
            })

        return pd.DataFrame(rows)


class DLAVisualizer:
    """Visualization utilities for Direct Logit Attribution results."""

    def __init__(self, action_tokens: Dict):
        self.action_tokens = action_tokens

    def plot_head_heatmap(
        self,
        result: DLAResult,
        ax=None,
        cmap='RdBu_r',
        title=None
    ):
        """
        Plot heatmap of per-head contributions.

        Args:
            result: DLAResult instance
            ax: Matplotlib axis (creates new figure if None)
            cmap: Colormap (default: RdBu_r, red=pro-Defect, blue=pro-Cooperate)
            title: Plot title
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(
            result.head_contributions,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Logit Contribution (D - C)', rotation=270, labelpad=20)

        # Labels
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        ax.set_title(title or f"Head Contributions: {result.model_name}")

        # Set ticks
        ax.set_xticks(range(result.head_contributions.shape[1]))
        ax.set_yticks(range(0, result.head_contributions.shape[0], 2))

        return ax

    def plot_mlp_contributions(
        self,
        result: DLAResult,
        ax=None,
        title=None
    ):
        """Plot MLP contributions across layers."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        layers = range(len(result.mlp_contributions))
        ax.bar(
            layers,
            result.mlp_contributions,
            color=['red' if x > 0 else 'blue' for x in result.mlp_contributions],
            alpha=0.7
        )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('MLP Contribution (D - C)')
        ax.set_title(title or f"MLP Contributions: {result.model_name}")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_top_components(
        self,
        result: DLAResult,
        top_k: int = 20,
        ax=None,
        title=None
    ):
        """Plot top-k components driving each action."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Get top components
        top_comps = DirectLogitAttributor(
            hooked_model=None,
            action_tokens=self.action_tokens
        ).identify_top_components(result, top_k)

        # Plot pro-Defect components
        pro_d_names = [x[0] for x in top_comps['pro_defect']]
        pro_d_values = [x[1] for x in top_comps['pro_defect']]

        # Plot pro-Cooperate components
        pro_c_names = [x[0] for x in top_comps['pro_cooperate']]
        pro_c_values = [abs(x[1]) for x in top_comps['pro_cooperate']]  # Take abs for visual clarity

        # Create horizontal bar chart
        y_pos = np.arange(top_k)

        ax.barh(y_pos, pro_d_values[::-1], color='#d62728', alpha=0.7, label='Pro-Defect')
        ax.barh(y_pos + top_k + 1, pro_c_values[::-1], color='#1f77b4', alpha=0.7, label='Pro-Cooperate')

        ax.set_yticks(list(y_pos) + list(y_pos + top_k + 1))
        ax.set_yticklabels(pro_d_names[::-1] + pro_c_names[::-1])
        ax.set_xlabel('Contribution Magnitude')
        ax.set_title(title or f"Top-{top_k} Components: {result.model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        return ax

    def plot_model_comparison(
        self,
        results: List[DLAResult],
        model_labels: Dict[str, str],
        scenario: str,
        save_path: str = None
    ):
        """
        Compare head heatmaps across multiple models for a given scenario.

        Args:
            results: List of DLAResult instances
            model_labels: Dictionary mapping model names to display labels
            scenario: Scenario name
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 8))

        if n_models == 1:
            axes = [axes]

        for idx, result in enumerate(results):
            label = model_labels.get(result.model_name, result.model_name)
            self.plot_head_heatmap(
                result,
                ax=axes[idx],
                title=f"{label}\nΔ={result.final_delta:.2f}"
            )

        fig.suptitle(f"Head Contributions: {scenario}", fontsize=14, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig
