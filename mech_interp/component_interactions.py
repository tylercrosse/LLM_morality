"""
Component Interaction Analysis for Mechanistic Interpretability

This module analyzes how components (attention heads and MLPs) interact with each other
through correlation analysis of their activations. This reveals whether different models
use similar components but wire them together differently.

Key hypothesis: De and Ut models may have similar individual component strengths
but differ in how components coordinate (e.g., L8_MLP → L25 head connections).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from mech_interp.decision_metrics import (
    compute_action_sequence_preference,
    prepare_prompt,
)
from mech_interp.model_loader import LoRAModelLoader
from mech_interp.utils import load_prompt_dataset, get_action_token_ids


@dataclass
class ComponentInteractionResult:
    """Results from component interaction analysis."""
    model_id: str
    scenario: str
    variant: int

    # Sequence-level decision metric for this prompt.
    seq_delta_logp_action2_minus_action1: float
    seq_p_action2: float
    seq_preferred_action: str

    # Component activations: dict mapping component name -> activation value
    # e.g., "L8_MLP" -> scalar, "L25H3" -> scalar
    component_activations: Dict[str, float]

    # Correlation matrix computed across scenarios
    correlation_matrix: Optional[np.ndarray] = None
    component_names: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        base = {
            'model_id': self.model_id,
            'scenario': self.scenario,
            'variant': self.variant,
            'seq_delta_logp_action2_minus_action1': self.seq_delta_logp_action2_minus_action1,
            'seq_p_action2': self.seq_p_action2,
            'seq_preferred_action': self.seq_preferred_action,
            'component_activations': self.component_activations,
        }
        if self.correlation_matrix is not None:
            base['correlation_matrix'] = self.correlation_matrix.tolist()
            base['component_names'] = self.component_names
        return base


class ComponentInteractionAnalyzer:
    """Analyzes interactions between model components."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        use_chat_template: bool = True,
        action1_label: str = "action1",
        action2_label: str = "action2",
    ):
        """Initialize analyzer with a model.

        Args:
            model_id: Model identifier (e.g., "PT3_COREDe")
            device: Device to run on
        """
        self.model_id = model_id
        self.device = device
        self.model = LoRAModelLoader.load_hooked_model(
            model_id,
            device=device,
            merge_lora=False,
            use_4bit=True,
        )
        self.tokenizer = self.model.tokenizer
        self.use_chat_template = use_chat_template
        self.action1_label = action1_label
        self.action2_label = action2_label

        # Get action token IDs
        action_tokens = get_action_token_ids(self.tokenizer)
        self.c_token = action_tokens["C"]
        self.d_token = action_tokens["D"]

    def extract_component_activations(
        self,
        prompt: str,
        scenario: str,
        variant: int
    ) -> ComponentInteractionResult:
        """Extract activation magnitudes for all components.

        Args:
            prompt: Input prompt text
            scenario: Scenario name
            variant: Variant number

        Returns:
            ComponentInteractionResult with component activations
        """
        prepared_prompt = prepare_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
        )
        # Tokenize
        inputs = self.tokenizer(prepared_prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.device)

        # Sequence-level decision metric.
        seq_pref = compute_action_sequence_preference(
            forward_logits_fn=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            action1_label=self.action1_label,
            action2_label=self.action2_label,
        )

        # Run forward pass with cache
        with torch.no_grad():
            final_logits, cache = self.model.run_with_cache(input_ids)

        # Extract final token hidden states for each component
        final_pos = input_ids.shape[1] - 1
        component_activations = {}

        # Extract attention and MLP activations for each layer
        # Note: HookedGemmaModel provides full attention output, not per-head
        # We use layer-level granularity for correlation analysis
        for layer in range(self.model.n_layers):
            # Attention output
            attn_key = f"blocks.{layer}.hook_attn_out"
            if attn_key in cache:
                # Shape: (batch, seq_len, d_model)
                attn_out = cache[attn_key][0, final_pos]  # (d_model,)
                magnitude = float(torch.norm(attn_out, p=2).cpu())
                component_activations[f"L{layer}_ATTN"] = magnitude

            # MLP output
            mlp_key = f"blocks.{layer}.hook_mlp_out"
            if mlp_key in cache:
                # Shape: (batch, seq_len, d_model)
                mlp_out = cache[mlp_key][0, final_pos]  # (d_model,)
                magnitude = float(torch.norm(mlp_out, p=2).cpu())
                component_activations[f"L{layer}_MLP"] = magnitude

        return ComponentInteractionResult(
            model_id=self.model_id,
            scenario=scenario,
            variant=variant,
            seq_delta_logp_action2_minus_action1=float(seq_pref.delta_logp_action2_minus_action1),
            seq_p_action2=float(seq_pref.p_action2),
            seq_preferred_action=seq_pref.preferred_action,
            component_activations=component_activations,
        )

    def compute_correlation_matrix(
        self,
        results: List[ComponentInteractionResult],
        method: str = "pearson"
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute correlation matrix across scenarios.

        Args:
            results: List of ComponentInteractionResult from different scenarios
            method: "pearson" or "spearman"

        Returns:
            Tuple of (correlation_matrix, component_names)
        """
        # Build activation matrix: (num_scenarios, num_components)
        component_names = sorted(results[0].component_activations.keys())
        num_scenarios = len(results)
        num_components = len(component_names)

        activation_matrix = np.zeros((num_scenarios, num_components))

        for i, result in enumerate(results):
            for j, comp_name in enumerate(component_names):
                activation_matrix[i, j] = result.component_activations[comp_name]

        # Compute correlation between components (correlation of columns)
        corr_matrix = np.zeros((num_components, num_components))
        if num_scenarios < 2:
            # Degenerate case for smoke tests / tiny datasets.
            np.fill_diagonal(corr_matrix, 1.0)
            return corr_matrix, component_names

        for i in range(num_components):
            for j in range(num_components):
                if method == "pearson":
                    corr, _ = pearsonr(activation_matrix[:, i], activation_matrix[:, j])
                else:  # spearman
                    corr, _ = spearmanr(activation_matrix[:, i], activation_matrix[:, j])
                corr_matrix[i, j] = corr

        return corr_matrix, component_names


class InteractionComparator:
    """Compare component interactions between models."""

    def __init__(self, results_dir: str = "/root/LLM_morality/mech_interp_outputs/component_interactions"):
        """Initialize comparator.

        Args:
            results_dir: Directory for saving results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def compare_correlation_matrices(
        self,
        corr1: np.ndarray,
        corr2: np.ndarray,
        component_names: List[str],
        model1_name: str,
        model2_name: str
    ) -> pd.DataFrame:
        """Compare correlation matrices between two models.

        Args:
            corr1: Correlation matrix from model 1
            corr2: Correlation matrix from model 2
            component_names: List of component names
            model1_name: Name for model 1
            model2_name: Name for model 2

        Returns:
            DataFrame with comparison statistics
        """
        # Compute difference matrix
        diff_matrix = corr1 - corr2

        # Find largest differences
        num_components = len(component_names)
        differences = []

        for i in range(num_components):
            for j in range(i + 1, num_components):  # Upper triangle only
                differences.append({
                    'component_1': component_names[i],
                    'component_2': component_names[j],
                    f'{model1_name}_corr': corr1[i, j],
                    f'{model2_name}_corr': corr2[i, j],
                    'correlation_diff': diff_matrix[i, j],
                    'abs_diff': abs(diff_matrix[i, j]),
                })

        df = pd.DataFrame(differences)
        df = df.sort_values('abs_diff', ascending=False)

        return df

    def plot_correlation_matrix(
        self,
        corr_matrix: np.ndarray,
        component_names: List[str],
        model_name: str,
        output_path: Optional[str] = None,
        top_n: int = 50
    ):
        """Plot correlation matrix heatmap.

        Args:
            corr_matrix: Correlation matrix
            component_names: Component names
            model_name: Model name for title
            output_path: Where to save plot
            top_n: Only plot top N most variable components
        """
        # Select top N components with highest variance
        variances = np.var(corr_matrix, axis=1)
        top_indices = np.argsort(variances)[-top_n:]

        corr_subset = corr_matrix[top_indices][:, top_indices]
        names_subset = [component_names[i] for i in top_indices]

        # Perform hierarchical clustering
        linkage = hierarchy.linkage(corr_subset, method='average')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        order = dendro['leaves']

        corr_ordered = corr_subset[order][:, order]
        names_ordered = [names_subset[i] for i in order]

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(
            corr_ordered,
            xticklabels=names_ordered,
            yticklabels=names_ordered,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )

        ax.set_title(f'Component Interaction Matrix: {model_name}\n(Top {top_n} most variable components)',
                     fontsize=14, pad=20)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {output_path}")
        else:
            save_path = self.results_dir / f"correlation_matrix_{model_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")

        plt.close()

    def plot_difference_matrix(
        self,
        diff_matrix: np.ndarray,
        component_names: List[str],
        model1_name: str,
        model2_name: str,
        output_path: Optional[str] = None,
        top_n: int = 50
    ):
        """Plot correlation difference matrix.

        Args:
            diff_matrix: Difference matrix (corr1 - corr2)
            component_names: Component names
            model1_name: Name for model 1
            model2_name: Name for model 2
            output_path: Where to save plot
            top_n: Only plot top N components with largest differences
        """
        # Select top N components with largest variance in differences
        variances = np.var(diff_matrix, axis=1)
        top_indices = np.argsort(variances)[-top_n:]

        diff_subset = diff_matrix[top_indices][:, top_indices]
        names_subset = [component_names[i] for i in top_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(
            diff_subset,
            xticklabels=names_subset,
            yticklabels=names_subset,
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation Difference'}
        )

        ax.set_title(f'Interaction Differences: {model1_name} - {model2_name}\n(Top {top_n} most different components)',
                     fontsize=14, pad=20)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved difference matrix to {output_path}")
        else:
            save_path = self.results_dir / f"interaction_diff_{model1_name}_vs_{model2_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved difference matrix to {save_path}")

        plt.close()

    def identify_pathway_differences(
        self,
        comparison_df: pd.DataFrame,
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """Identify pathways (component pairs) with large correlation differences.

        Args:
            comparison_df: DataFrame from compare_correlation_matrices()
            threshold: Minimum absolute difference to consider

        Returns:
            DataFrame filtered to significant pathway differences
        """
        significant = comparison_df[comparison_df['abs_diff'] >= threshold].copy()

        # Categorize pathway types
        def categorize_pathway(row):
            c1, c2 = row['component_1'], row['component_2']

            # Extract layer numbers
            l1 = int(c1[1:].split('H')[0].split('_')[0])
            l2 = int(c2[1:].split('H')[0].split('_')[0])

            # Same layer
            if l1 == l2:
                return "intra_layer"
            # Adjacent layers
            elif abs(l1 - l2) == 1:
                return "adjacent_layer"
            # Early to late
            elif l1 < 10 and l2 > 15:
                return "early_to_late"
            # Late to late
            elif l1 > 15 and l2 > 15:
                return "late_stage"
            else:
                return "other"

        significant['pathway_type'] = significant.apply(categorize_pathway, axis=1)

        return significant.sort_values('abs_diff', ascending=False)


def run_component_interaction_analysis(
    model_ids: List[str],
    prompt_dataset_path: Optional[str] = None,
    output_dir: str = "/root/LLM_morality/mech_interp_outputs/component_interactions",
    device: str = "cuda"
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Run component interaction analysis for multiple models.

    Args:
        model_ids: List of model IDs to analyze
        prompt_dataset_path: Path to prompt dataset JSON
        output_dir: Where to save results
        device: Device to run on

    Returns:
        Dictionary mapping model_id to (correlation_matrix, component_names)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompt_dataset(prompt_dataset_path)

    all_correlation_matrices = {}

    for model_id in model_ids:
        print(f"\n{'='*60}")
        print(f"Analyzing component interactions for {model_id}")
        print(f"{'='*60}\n")

        analyzer = ComponentInteractionAnalyzer(model_id, device=device)
        model_results = []

        # Extract activations for all prompts
        for prompt_data in prompts:
            scenario = prompt_data['scenario']
            variant = prompt_data['variant']
            prompt = prompt_data['prompt']

            print(f"  Extracting activations for {scenario} variant {variant}...")

            result = analyzer.extract_component_activations(prompt, scenario, variant)
            model_results.append(result)

        # Compute correlation matrix
        print(f"\n  Computing correlation matrix...")
        corr_matrix, component_names = analyzer.compute_correlation_matrix(model_results)

        all_correlation_matrices[model_id] = (corr_matrix, component_names)

        # Save results
        results_file = output_dir / f"component_activations_{model_id}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in model_results], f, indent=2)

        corr_file = output_dir / f"correlation_matrix_{model_id}.npz"
        np.savez(corr_file, correlation_matrix=corr_matrix, component_names=component_names)

        print(f"  Saved results to {results_file}")
        print(f"  Saved correlation matrix to {corr_file}")

        summary_rows = []
        for r in model_results:
            summary_rows.append({
                "model_id": r.model_id,
                "scenario": r.scenario,
                "variant": r.variant,
                "seq_preferred_action": r.seq_preferred_action,
                "seq_p_action2": r.seq_p_action2,
                "seq_delta_logp_action2_minus_action1": r.seq_delta_logp_action2_minus_action1,
                "mean_component_activation": float(np.mean(list(r.component_activations.values()))),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "variant"])
        summary_file = output_dir / f"component_interaction_summary_{model_id}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"  Saved summary to {summary_file}")

        # Plot correlation matrix
        comparator = InteractionComparator(output_dir)
        comparator.plot_correlation_matrix(corr_matrix, component_names, model_id)

    return all_correlation_matrices


def compare_deontological_vs_utilitarian(
    output_dir: str = "/root/LLM_morality/mech_interp_outputs/component_interactions"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Deontological vs Utilitarian component interactions.

    Args:
        output_dir: Directory containing interaction results

    Returns:
        Tuple of (comparison_df, significant_pathways_df)
    """
    output_dir = Path(output_dir)

    # Load correlation matrices
    de_file = output_dir / "correlation_matrix_PT3_COREDe.npz"
    ut_file = output_dir / "correlation_matrix_PT3_COREUt.npz"

    de_data = np.load(de_file)
    ut_data = np.load(ut_file)

    de_corr = de_data['correlation_matrix']
    de_names = list(de_data['component_names'])

    ut_corr = ut_data['correlation_matrix']
    ut_names = list(ut_data['component_names'])

    assert de_names == ut_names, "Component names must match"

    # Compare
    comparator = InteractionComparator(output_dir)
    comparison_df = comparator.compare_correlation_matrices(
        de_corr, ut_corr, de_names,
        "Deontological", "Utilitarian"
    )

    # Save comparison
    comparison_file = output_dir / "interaction_comparison_De_vs_Ut.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved comparison to {comparison_file}")

    # Plot difference matrix
    diff_matrix = de_corr - ut_corr
    comparator.plot_difference_matrix(diff_matrix, de_names, "Deontological", "Utilitarian")

    # Identify significant pathway differences
    significant_pathways = comparator.identify_pathway_differences(comparison_df, threshold=0.3)

    pathways_file = output_dir / "significant_pathways_De_vs_Ut.csv"
    significant_pathways.to_csv(pathways_file, index=False)
    print(f"Saved significant pathways to {pathways_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("COMPONENT INTERACTION COMPARISON: Deontological vs Utilitarian")
    print("="*60 + "\n")

    print(f"Total component pairs analyzed: {len(comparison_df)}")
    print(f"Pairs with |diff| > 0.1: {(comparison_df['abs_diff'] > 0.1).sum()}")
    print(f"Pairs with |diff| > 0.2: {(comparison_df['abs_diff'] > 0.2).sum()}")
    print(f"Pairs with |diff| > 0.3: {(comparison_df['abs_diff'] > 0.3).sum()}")

    print("\nTop 10 most different pathways:")
    print(comparison_df.head(10)[['component_1', 'component_2', 'Deontological_corr',
                                    'Utilitarian_corr', 'correlation_diff']])

    if len(significant_pathways) > 0:
        print(f"\nPathway types with |diff| > 0.3:")
        print(significant_pathways['pathway_type'].value_counts())

    return comparison_df, significant_pathways


def analyze_key_pathways(
    comparison_df: pd.DataFrame,
    key_components: List[str] = ["L8_MLP", "L9_MLP", "L25H0", "L25H1", "L25H2", "L25H3"]
) -> pd.DataFrame:
    """Analyze pathways involving key components identified in DLA.

    Args:
        comparison_df: DataFrame from compare_correlation_matrices()
        key_components: List of important components from DLA analysis

    Returns:
        DataFrame filtered to pathways involving key components
    """
    mask = (
        comparison_df['component_1'].isin(key_components) |
        comparison_df['component_2'].isin(key_components)
    )

    key_pathways = comparison_df[mask].copy()
    key_pathways = key_pathways.sort_values('abs_diff', ascending=False)

    print("\n" + "="*60)
    print("PATHWAYS INVOLVING KEY COMPONENTS")
    print("="*60 + "\n")

    for comp in key_components:
        comp_pathways = key_pathways[
            (key_pathways['component_1'] == comp) |
            (key_pathways['component_2'] == comp)
        ]
        if len(comp_pathways) > 0:
            print(f"\n{comp} connections (top 5):")
            print(comp_pathways.head(5)[['component_1', 'component_2',
                                          'correlation_diff', 'abs_diff']])

    return key_pathways
