"""
LoRA Weight Analysis for mechanistic interpretability.

Computes Frobenius norms of LoRA adapter weights to quantify which components
were most heavily modified during fine-tuning. This helps distinguish between
components that were directly retrained vs. passive responders to upstream changes.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import safetensors.torch
import json

from mech_interp.utils import get_model_path, MODEL_LABELS, MODEL_COLORS


@dataclass
class WeightNormResult:
    """Results for a single LoRA module."""

    model_id: str
    layer: int
    module_type: str  # e.g., "mlp.gate_proj", "self_attn.q_proj"
    component_name: str  # e.g., "L2_MLP_GATE", "L5_ATTN_Q"

    # LoRA configuration
    lora_rank: int
    lora_alpha: int

    # Weight norms
    frobenius_norm: float  # ||B @ A||_F
    frobenius_norm_normalized: float  # ||B @ A||_F / sqrt(output_dim)
    lora_A_norm: float  # ||A||_F
    lora_B_norm: float  # ||B||_F
    effective_scale: float  # (lora_alpha / lora_rank) * frobenius_norm


class WeightAnalyzer:
    """
    Analyzes LoRA adapter weights to quantify component-level modifications.

    Key idea: Components with high ||B @ A||_F were heavily modified during
    fine-tuning, while low-norm components may be passive responders.
    """

    def __init__(self, model_id: str):
        """
        Initialize analyzer for a specific model.

        Args:
            model_id: Model identifier (e.g., 'PT3_COREDe')
        """
        self.model_id = model_id
        self.model_path = Path(get_model_path(model_id))

        # Load adapter config
        config_path = self.model_path / "adapter_config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        self.lora_rank = self.config["r"]
        self.lora_alpha = self.config["lora_alpha"]
        self.target_modules = self.config["target_modules"]

        # Load adapter weights
        self.weights = self.load_adapter_weights()

    def load_adapter_weights(self) -> Dict[str, torch.Tensor]:
        """Load LoRA adapter weights from safetensors."""
        safetensors_path = self.model_path / "adapter_model.safetensors"

        if not safetensors_path.exists():
            raise FileNotFoundError(f"No adapter weights found at {safetensors_path}")

        weights = safetensors.torch.load_file(str(safetensors_path))
        return weights

    def compute_module_norm(self, layer: int, module_type: str) -> WeightNormResult:
        """
        Compute Frobenius norm ||B @ A|| for a single LoRA module.

        Args:
            layer: Layer index (0-25 for Gemma-2-2b)
            module_type: Module type (e.g., "mlp.gate_proj", "self_attn.q_proj")

        Returns:
            WeightNormResult with computed norms
        """
        # Construct weight keys
        base_key = f"base_model.model.model.layers.{layer}.{module_type}"
        a_key = f"{base_key}.lora_A.weight"
        b_key = f"{base_key}.lora_B.weight"

        if a_key not in self.weights or b_key not in self.weights:
            raise KeyError(f"Missing weights for {base_key}")

        # Load weights
        lora_A = self.weights[a_key]  # Shape: (rank, input_dim)
        lora_B = self.weights[b_key]  # Shape: (output_dim, rank)

        # Compute ||B @ A||_F
        # Note: B @ A gives the effective weight update (output_dim, input_dim)
        BA = torch.matmul(lora_B, lora_A)  # (output_dim, input_dim)
        frobenius_norm = torch.norm(BA, p='fro').item()

        # Normalize by sqrt(output_dim) for fair comparison across module types
        output_dim = lora_B.shape[0]
        frobenius_norm_normalized = frobenius_norm / np.sqrt(output_dim)

        # Individual norms
        lora_A_norm = torch.norm(lora_A, p='fro').item()
        lora_B_norm = torch.norm(lora_B, p='fro').item()

        # Effective scale (accounts for LoRA scaling factor)
        effective_scale = (self.lora_alpha / self.lora_rank) * frobenius_norm

        # Component naming (matches existing conventions)
        component_name = self._get_component_name(layer, module_type)

        return WeightNormResult(
            model_id=self.model_id,
            layer=layer,
            module_type=module_type,
            component_name=component_name,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            frobenius_norm=frobenius_norm,
            frobenius_norm_normalized=frobenius_norm_normalized,
            lora_A_norm=lora_A_norm,
            lora_B_norm=lora_B_norm,
            effective_scale=effective_scale,
        )

    def _get_component_name(self, layer: int, module_type: str) -> str:
        """
        Generate component name following existing conventions.

        Examples:
            - L2_MLP_GATE, L2_MLP_UP, L2_MLP_DOWN
            - L5_ATTN_Q, L5_ATTN_K, L5_ATTN_V, L5_ATTN_O
        """
        if module_type.startswith("mlp."):
            proj_type = module_type.split(".")[-1]  # gate_proj -> gate
            proj_short = proj_type.replace("_proj", "").upper()  # gate -> GATE
            return f"L{layer}_MLP_{proj_short}"
        elif module_type.startswith("self_attn."):
            proj_type = module_type.split(".")[-1]  # q_proj -> q
            proj_short = proj_type.replace("_proj", "").upper()  # q -> Q
            return f"L{layer}_ATTN_{proj_short}"
        else:
            return f"L{layer}_{module_type.upper()}"

    def analyze_all_modules(self) -> List[WeightNormResult]:
        """
        Analyze all 182 LoRA modules (26 layers × 7 modules).

        Returns:
            List of WeightNormResult for all modules
        """
        results = []

        # Gemma-2-2b has 26 layers
        n_layers = 26

        # Module types (from adapter_config.json target_modules)
        # MLP: gate_proj, up_proj, down_proj
        # Attention: q_proj, k_proj, v_proj, o_proj
        module_types = [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

        for layer in range(n_layers):
            for module_type in module_types:
                result = self.compute_module_norm(layer, module_type)
                results.append(result)

        return results

    def aggregate_mlp_norms(self, results: List[WeightNormResult]) -> Dict[int, float]:
        """
        Aggregate gate_proj + up_proj + down_proj norms per layer.

        This gives a single "MLP modification strength" per layer.

        Args:
            results: List of WeightNormResult from analyze_all_modules()

        Returns:
            Dictionary mapping layer -> total MLP norm
        """
        mlp_norms = {}

        for layer in range(26):
            layer_mlp_results = [
                r for r in results
                if r.layer == layer and r.module_type.startswith("mlp.")
            ]
            total_norm = sum(r.frobenius_norm for r in layer_mlp_results)
            mlp_norms[layer] = total_norm

        return mlp_norms

    def compute_percentile_rank(
        self,
        results: List[WeightNormResult],
        target_layer: int,
        target_prefix: str = "mlp"
    ) -> float:
        """
        Compute percentile rank for a specific layer's modules.

        Args:
            results: List of WeightNormResult
            target_layer: Layer to analyze (e.g., 2 for L2)
            target_prefix: Module type prefix (e.g., "mlp" for MLP modules)

        Returns:
            Percentile rank (0-100)
        """
        # Get target modules
        target_results = [
            r for r in results
            if r.layer == target_layer and r.module_type.startswith(target_prefix)
        ]
        target_norm = sum(r.frobenius_norm for r in target_results)

        # Get all layer-level norms
        all_layer_norms = []
        for layer in range(26):
            layer_results = [
                r for r in results
                if r.layer == layer and r.module_type.startswith(target_prefix)
            ]
            layer_norm = sum(r.frobenius_norm for r in layer_results)
            all_layer_norms.append(layer_norm)

        # Compute percentile
        percentile = (np.sum(np.array(all_layer_norms) <= target_norm) / len(all_layer_norms)) * 100
        return percentile


class WeightAnalysisVisualizer:
    """Generate visualizations for weight analysis results."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100

    def plot_norm_heatmap(
        self,
        results: List[WeightNormResult],
        model_id: str,
        use_normalized: bool = True
    ):
        """
        Generate heatmap of Frobenius norms (26 layers × 7 modules).

        Args:
            results: List of WeightNormResult
            model_id: Model identifier for title
            use_normalized: Use normalized norms (True) or raw norms (False)
        """
        # Organize into matrix (26 layers × 7 modules)
        n_layers = 26
        module_order = [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

        matrix = np.zeros((n_layers, len(module_order)))

        for result in results:
            layer_idx = result.layer
            module_idx = module_order.index(result.module_type)
            norm = result.frobenius_norm_normalized if use_normalized else result.frobenius_norm
            matrix[layer_idx, module_idx] = norm

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 12))

        # Short labels
        module_labels = ['Gate', 'Up', 'Down', 'Q', 'K', 'V', 'O']

        sns.heatmap(
            matrix,
            ax=ax,
            cmap='YlOrRd',
            cbar_kws={'label': 'Normalized ||B @ A||_F' if use_normalized else '||B @ A||_F'},
            yticklabels=[f"L{i}" for i in range(n_layers)],
            xticklabels=module_labels,
            annot=False,
        )

        ax.set_xlabel('Module Type')
        ax.set_ylabel('Layer')
        ax.set_title(f'LoRA Weight Norms - {MODEL_LABELS.get(model_id, model_id)}')

        plt.tight_layout()

        suffix = "normalized" if use_normalized else "raw"
        output_path = self.output_dir / f"norm_heatmap_{model_id}_{suffix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap: {output_path}")

    def plot_top_components(
        self,
        results: List[WeightNormResult],
        model_id: str,
        top_n: int = 30,
        highlight_layer: Optional[int] = None
    ):
        """
        Generate bar chart of top-N components by Frobenius norm.

        Args:
            results: List of WeightNormResult
            model_id: Model identifier
            top_n: Number of top components to show
            highlight_layer: Layer to highlight (e.g., 2 for L2)
        """
        # Sort by frobenius_norm
        sorted_results = sorted(results, key=lambda r: r.frobenius_norm, reverse=True)
        top_results = sorted_results[:top_n]

        # Prepare data
        component_names = [r.component_name for r in top_results]
        norms = [r.frobenius_norm for r in top_results]

        # Color code: highlight specific layer if requested
        colors = []
        for r in top_results:
            if highlight_layer is not None and r.layer == highlight_layer and r.module_type.startswith("mlp"):
                colors.append('red')
            else:
                colors.append(MODEL_COLORS.get(model_id, '#4C78A8'))

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(component_names))
        ax.barh(y_pos, norms, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(component_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Frobenius Norm ||B @ A||_F')
        ax.set_title(f'Top {top_n} Components by Weight Modification - {MODEL_LABELS.get(model_id, model_id)}')

        if highlight_layer is not None:
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label=f'L{highlight_layer} MLP'),
                Patch(facecolor=MODEL_COLORS.get(model_id, '#4C78A8'), label='Other'),
            ]
            ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        output_path = self.output_dir / f"top_components_{model_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved top components: {output_path}")

    def plot_l2_mlp_comparison(
        self,
        all_results: Dict[str, List[WeightNormResult]],
        target_layer: int = 2
    ):
        """
        Generate cross-model comparison of L2_MLP (or other layer) total norms.

        Args:
            all_results: Dictionary mapping model_id -> List[WeightNormResult]
            target_layer: Layer to compare
        """
        model_ids = list(all_results.keys())
        mlp_norms = []

        for model_id in model_ids:
            results = all_results[model_id]

            # Sum MLP modules for target layer
            layer_mlp_results = [
                r for r in results
                if r.layer == target_layer and r.module_type.startswith("mlp.")
            ]
            total_norm = sum(r.frobenius_norm for r in layer_mlp_results)
            mlp_norms.append(total_norm)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [MODEL_COLORS.get(m, '#4C78A8') for m in model_ids]
        labels = [MODEL_LABELS.get(m, m) for m in model_ids]

        x_pos = np.arange(len(model_ids))
        ax.bar(x_pos, mlp_norms, color=colors)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Total Frobenius Norm')
        ax.set_title(f'L{target_layer} MLP Weight Modification Across Models')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"l{target_layer}_mlp_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved L{target_layer} MLP comparison: {output_path}")

    def plot_layer_profiles(
        self,
        all_results: Dict[str, List[WeightNormResult]],
        module_prefix: str = "mlp"
    ):
        """
        Plot layer-wise norm profiles for all models.

        Args:
            all_results: Dictionary mapping model_id -> List[WeightNormResult]
            module_prefix: Module type to aggregate (e.g., "mlp", "self_attn")
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for model_id, results in all_results.items():
            # Aggregate norms per layer
            layer_norms = []
            for layer in range(26):
                layer_results = [
                    r for r in results
                    if r.layer == layer and r.module_type.startswith(module_prefix)
                ]
                total_norm = sum(r.frobenius_norm for r in layer_results)
                layer_norms.append(total_norm)

            # Plot
            ax.plot(
                range(26),
                layer_norms,
                marker='o',
                label=MODEL_LABELS.get(model_id, model_id),
                color=MODEL_COLORS.get(model_id, '#4C78A8'),
                linewidth=2,
                markersize=4
            )

        ax.set_xlabel('Layer')
        ax.set_ylabel('Total Frobenius Norm')
        ax.set_title(f'{module_prefix.upper()} Weight Modification Profile')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"layer_profile_{module_prefix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved layer profile: {output_path}")
