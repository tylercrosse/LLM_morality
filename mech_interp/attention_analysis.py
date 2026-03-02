"""
Attention Pattern Analysis for Mechanistic Interpretability

This module analyzes what tokens different models attend to, helping understand
how Deontological vs Utilitarian models process information differently.

Key hypothesis: De models attend more to opponent's previous actions,
while Ut models attend more to joint payoff information.
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

from mech_interp.decision_metrics import (
    compute_action_sequence_preference,
    prepare_prompt,
)
from mech_interp.model_loader import LoRAModelLoader
from mech_interp.utils import load_prompt_dataset


@dataclass
class AttentionPatternResult:
    """Results from attention pattern analysis."""
    model_id: str
    scenario: str
    variant: int

    # Sequence-level decision metric on this prompt.
    seq_delta_logp_action2_minus_action1: float
    seq_p_action2: float
    seq_preferred_action: str

    # Attention weights: (num_layers, num_heads, seq_len, seq_len)
    attention_patterns: np.ndarray

    # Token-level attention aggregation: (seq_len,)
    # How much attention does the final token pay to each position?
    final_token_attention: np.ndarray

    # Token IDs and text
    token_ids: List[int]
    token_texts: List[str]

    # Special token positions
    action_keyword_positions: List[int]  # Positions of "action1", "action2"
    opponent_action_positions: List[int]  # Positions mentioning opponent's previous move
    payoff_positions: List[int]  # Positions mentioning payoffs/points

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'model_id': self.model_id,
            'scenario': self.scenario,
            'variant': self.variant,
            'seq_delta_logp_action2_minus_action1': self.seq_delta_logp_action2_minus_action1,
            'seq_p_action2': self.seq_p_action2,
            'seq_preferred_action': self.seq_preferred_action,
            'attention_patterns': self.attention_patterns.tolist(),
            'final_token_attention': self.final_token_attention.tolist(),
            'token_ids': self.token_ids,
            'token_texts': self.token_texts,
            'action_keyword_positions': self.action_keyword_positions,
            'opponent_action_positions': self.opponent_action_positions,
            'payoff_positions': self.payoff_positions,
        }


class AttentionAnalyzer:
    """Analyzes attention patterns in models."""

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

    def analyze_prompt(self, prompt: str, scenario: str, variant: int) -> AttentionPatternResult:
        """Analyze attention patterns for a single prompt.

        Args:
            prompt: Input prompt text
            scenario: Scenario name (e.g., "CC_continue")
            variant: Variant number

        Returns:
            AttentionPatternResult with attention patterns and token information
        """
        prepared_prompt = prepare_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
        )

        # Tokenize
        inputs = self.tokenizer(prepared_prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.device)

        # Sequence-level decision metric aligned with logit-lens/patching/DLA.
        seq_pref = compute_action_sequence_preference(
            forward_logits_fn=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            action1_label=self.action1_label,
            action2_label=self.action2_label,
        )

        # Run forward pass with cache to get attention weights
        with torch.no_grad():
            _, cache = self.model.run_with_cache(input_ids, output_attentions=True)

        # Extract attention patterns
        # Cache keys: "model.layers.{layer}.self_attn.attn_weights"
        num_layers = self.model.n_layers
        num_heads = self.model.n_heads
        seq_len = input_ids.shape[1]

        attention_patterns = np.zeros((num_layers, num_heads, seq_len, seq_len))

        for layer in range(num_layers):
            attn_key = f"model.layers.{layer}.self_attn.attn_weights"
            if attn_key in cache:
                # Shape: (batch, num_heads, seq_len, seq_len)
                # Convert to float32 first (numpy doesn't support bfloat16)
                attn_weights = cache[attn_key][0].float().cpu().numpy()  # Remove batch dim
                attention_patterns[layer] = attn_weights

        # Get final token attention (what the last token attends to)
        # Average across all layers and heads
        final_token_attention = attention_patterns[:, :, -1, :].mean(axis=(0, 1))

        # Decode tokens
        token_ids = input_ids[0].cpu().tolist()
        token_texts = [self.tokenizer.decode([tid]) for tid in token_ids]

        # Identify special token positions
        action_keyword_positions = self._find_action_keywords(token_texts)
        opponent_action_positions = self._find_opponent_action_mentions(prompt, token_texts)
        payoff_positions = self._find_payoff_mentions(token_texts)

        return AttentionPatternResult(
            model_id=self.model_id,
            scenario=scenario,
            variant=variant,
            seq_delta_logp_action2_minus_action1=float(seq_pref.delta_logp_action2_minus_action1),
            seq_p_action2=float(seq_pref.p_action2),
            seq_preferred_action=seq_pref.preferred_action,
            attention_patterns=attention_patterns,
            final_token_attention=final_token_attention,
            token_ids=token_ids,
            token_texts=token_texts,
            action_keyword_positions=action_keyword_positions,
            opponent_action_positions=opponent_action_positions,
            payoff_positions=payoff_positions,
        )

    def _find_action_keywords(self, token_texts: List[str]) -> List[int]:
        """Find positions of action keywords (action1, action2)."""
        positions = []
        for i, text in enumerate(token_texts):
            if "action" in text.lower() and any(str(j) in text for j in [1, 2]):
                positions.append(i)
        return positions

    def _find_opponent_action_mentions(self, prompt: str, token_texts: List[str]) -> List[int]:
        """Find positions mentioning opponent's previous action."""
        positions = []
        # Look for phrases like "they played", "opponent chose", etc.
        keywords = ["they played", "opponent", "other player", "their choice"]

        for i, text in enumerate(token_texts):
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                positions.append(i)

        return positions

    def _find_payoff_mentions(self, token_texts: List[str]) -> List[int]:
        """Find positions mentioning payoffs or points."""
        positions = []
        keywords = ["point", "payoff", "reward", "outcome", "receive", "get"]

        for i, text in enumerate(token_texts):
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                positions.append(i)

        return positions


class AttentionComparator:
    """Compare attention patterns between models."""

    def __init__(self, results_dir: str = "/root/LLM_morality/mech_interp_outputs/attention_analysis"):
        """Initialize comparator.

        Args:
            results_dir: Directory containing attention analysis results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def compare_models(
        self,
        model1_results: List[AttentionPatternResult],
        model2_results: List[AttentionPatternResult],
        model1_name: str,
        model2_name: str,
    ) -> pd.DataFrame:
        """Compare attention patterns between two models.

        Args:
            model1_results: Results from first model
            model2_results: Results from second model
            model1_name: Name for first model (e.g., "Deontological")
            model2_name: Name for second model (e.g., "Utilitarian")

        Returns:
            DataFrame with comparison statistics
        """
        comparisons = []

        for r1, r2 in zip(model1_results, model2_results):
            assert r1.scenario == r2.scenario and r1.variant == r2.variant

            # Compute attention to special token types
            m1_action_attn = self._compute_attention_to_positions(
                r1.final_token_attention, r1.action_keyword_positions
            )
            m2_action_attn = self._compute_attention_to_positions(
                r2.final_token_attention, r2.action_keyword_positions
            )

            m1_opponent_attn = self._compute_attention_to_positions(
                r1.final_token_attention, r1.opponent_action_positions
            )
            m2_opponent_attn = self._compute_attention_to_positions(
                r2.final_token_attention, r2.opponent_action_positions
            )

            m1_payoff_attn = self._compute_attention_to_positions(
                r1.final_token_attention, r1.payoff_positions
            )
            m2_payoff_attn = self._compute_attention_to_positions(
                r2.final_token_attention, r2.payoff_positions
            )

            comparisons.append({
                'scenario': r1.scenario,
                'variant': r1.variant,
                f'{model1_name}_seq_delta': r1.seq_delta_logp_action2_minus_action1,
                f'{model2_name}_seq_delta': r2.seq_delta_logp_action2_minus_action1,
                f'{model1_name}_seq_p_action2': r1.seq_p_action2,
                f'{model2_name}_seq_p_action2': r2.seq_p_action2,
                f'{model1_name}_seq_pref': r1.seq_preferred_action,
                f'{model2_name}_seq_pref': r2.seq_preferred_action,
                'seq_delta_diff': (
                    r1.seq_delta_logp_action2_minus_action1
                    - r2.seq_delta_logp_action2_minus_action1
                ),
                'seq_p_action2_diff': r1.seq_p_action2 - r2.seq_p_action2,
                f'{model1_name}_action_attn': m1_action_attn,
                f'{model2_name}_action_attn': m2_action_attn,
                f'{model1_name}_opponent_attn': m1_opponent_attn,
                f'{model2_name}_opponent_attn': m2_opponent_attn,
                f'{model1_name}_payoff_attn': m1_payoff_attn,
                f'{model2_name}_payoff_attn': m2_payoff_attn,
                'action_attn_diff': m1_action_attn - m2_action_attn,
                'opponent_attn_diff': m1_opponent_attn - m2_opponent_attn,
                'payoff_attn_diff': m1_payoff_attn - m2_payoff_attn,
            })

        return pd.DataFrame(comparisons)

    def _compute_attention_to_positions(
        self,
        attention_weights: np.ndarray,
        positions: List[int]
    ) -> float:
        """Compute total attention to a set of positions."""
        if not positions:
            return 0.0
        return float(attention_weights[positions].sum())

    def visualize_comparison(
        self,
        comparison_df: pd.DataFrame,
        model1_name: str,
        model2_name: str,
        output_path: Optional[str] = None
    ):
        """Create visualization comparing attention patterns.

        Args:
            comparison_df: DataFrame from compare_models()
            model1_name: Name for first model
            model2_name: Name for second model
            output_path: Where to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Aggregate by scenario
        # Restrict aggregation to numeric columns; comparison_df includes
        # string labels such as seq preference names.
        scenario_stats = comparison_df.groupby('scenario').mean(numeric_only=True)

        token_types = ['action', 'opponent', 'payoff']
        titles = [
            'Attention to Action Keywords',
            'Attention to Opponent Action Context',
            'Attention to Payoff Information'
        ]

        for ax, token_type, title in zip(axes, token_types, titles):
            m1_col = f'{model1_name}_{token_type}_attn'
            m2_col = f'{model2_name}_{token_type}_attn'

            x = np.arange(len(scenario_stats))
            width = 0.35

            ax.bar(x - width/2, scenario_stats[m1_col], width, label=model1_name, alpha=0.8)
            ax.bar(x + width/2, scenario_stats[m2_col], width, label=model2_name, alpha=0.8)

            ax.set_xlabel('Scenario')
            ax.set_ylabel('Mean Attention Weight')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(scenario_stats.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention comparison plot to {output_path}")
        else:
            plt.savefig(self.results_dir / f"attention_comparison_{model1_name}_vs_{model2_name}.png",
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_attention_heatmap(
        self,
        result: AttentionPatternResult,
        layer: int,
        head: int,
        output_path: Optional[str] = None
    ):
        """Plot attention heatmap for a specific layer and head.

        Args:
            result: AttentionPatternResult to visualize
            layer: Layer index
            head: Head index
            output_path: Where to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        attn_weights = result.attention_patterns[layer, head]

        # Create heatmap
        sns.heatmap(
            attn_weights,
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )

        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'{result.model_id} - Layer {layer}, Head {head}\n{result.scenario} (variant {result.variant})')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.results_dir / f"attention_heatmap_{result.model_id}_L{layer}H{head}_{result.scenario}_v{result.variant}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")

        plt.close()


def run_attention_analysis(
    model_ids: List[str],
    prompt_dataset_path: Optional[str] = None,
    output_dir: str = "/root/LLM_morality/mech_interp_outputs/attention_analysis",
    device: str = "cuda"
) -> Dict[str, List[AttentionPatternResult]]:
    """Run attention analysis for multiple models.

    Args:
        model_ids: List of model IDs to analyze
        prompt_dataset_path: Path to prompt dataset JSON
        output_dir: Where to save results
        device: Device to run on

    Returns:
        Dictionary mapping model_id to list of results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompt_dataset(prompt_dataset_path)

    all_results = {}

    for model_id in model_ids:
        print(f"\n{'='*60}")
        print(f"Analyzing attention patterns for {model_id}")
        print(f"{'='*60}\n")

        analyzer = AttentionAnalyzer(model_id, device=device)
        model_results = []

        for prompt_data in prompts:
            scenario = prompt_data['scenario']
            variant = prompt_data['variant']
            prompt = prompt_data['prompt']

            print(f"  Processing {scenario} variant {variant}...")

            result = analyzer.analyze_prompt(prompt, scenario, variant)
            model_results.append(result)

        all_results[model_id] = model_results

        # Save results
        results_file = output_dir / f"attention_results_{model_id}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in model_results], f, indent=2)
        print(f"\nSaved results to {results_file}")

        # Lightweight tabular export for quick validation.
        summary_rows = []
        for r in model_results:
            summary_rows.append({
                "model_id": r.model_id,
                "scenario": r.scenario,
                "variant": r.variant,
                "seq_preferred_action": r.seq_preferred_action,
                "seq_p_action2": r.seq_p_action2,
                "seq_delta_logp_action2_minus_action1": r.seq_delta_logp_action2_minus_action1,
                "attn_to_action_tokens": float(r.final_token_attention[r.action_keyword_positions].sum())
                if r.action_keyword_positions else 0.0,
                "attn_to_opponent_context": float(r.final_token_attention[r.opponent_action_positions].sum())
                if r.opponent_action_positions else 0.0,
                "attn_to_payoff_tokens": float(r.final_token_attention[r.payoff_positions].sum())
                if r.payoff_positions else 0.0,
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "variant"])
        summary_file = output_dir / f"attention_summary_{model_id}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")

    return all_results


def compare_deontological_vs_utilitarian(
    output_dir: str = "/root/LLM_morality/mech_interp_outputs/attention_analysis"
) -> pd.DataFrame:
    """Compare Deontological vs Utilitarian attention patterns.

    Args:
        output_dir: Directory containing attention results

    Returns:
        DataFrame with comparison statistics
    """
    output_dir = Path(output_dir)

    # Load results
    de_file = output_dir / "attention_results_PT3_COREDe.json"
    ut_file = output_dir / "attention_results_PT3_COREUt.json"

    with open(de_file) as f:
        de_data = json.load(f)
    with open(ut_file) as f:
        ut_data = json.load(f)

    # Convert to AttentionPatternResult objects
    de_results = [AttentionPatternResult(
        model_id=d['model_id'],
        scenario=d['scenario'],
        variant=d['variant'],
        seq_delta_logp_action2_minus_action1=d.get('seq_delta_logp_action2_minus_action1', 0.0),
        seq_p_action2=d.get('seq_p_action2', np.nan),
        seq_preferred_action=d.get('seq_preferred_action', 'unknown'),
        attention_patterns=np.array(d['attention_patterns']),
        final_token_attention=np.array(d['final_token_attention']),
        token_ids=d['token_ids'],
        token_texts=d['token_texts'],
        action_keyword_positions=d['action_keyword_positions'],
        opponent_action_positions=d['opponent_action_positions'],
        payoff_positions=d['payoff_positions'],
    ) for d in de_data]

    ut_results = [AttentionPatternResult(
        model_id=d['model_id'],
        scenario=d['scenario'],
        variant=d['variant'],
        seq_delta_logp_action2_minus_action1=d.get('seq_delta_logp_action2_minus_action1', 0.0),
        seq_p_action2=d.get('seq_p_action2', np.nan),
        seq_preferred_action=d.get('seq_preferred_action', 'unknown'),
        attention_patterns=np.array(d['attention_patterns']),
        final_token_attention=np.array(d['final_token_attention']),
        token_ids=d['token_ids'],
        token_texts=d['token_texts'],
        action_keyword_positions=d['action_keyword_positions'],
        opponent_action_positions=d['opponent_action_positions'],
        payoff_positions=d['payoff_positions'],
    ) for d in ut_data]

    # Compare
    comparator = AttentionComparator(output_dir)
    comparison_df = comparator.compare_models(
        de_results, ut_results,
        "Deontological", "Utilitarian"
    )

    # Save comparison
    comparison_file = output_dir / "attention_comparison_De_vs_Ut.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved comparison to {comparison_file}")

    # Visualize
    comparator.visualize_comparison(comparison_df, "Deontological", "Utilitarian")

    # Print summary statistics
    print("\n" + "="*60)
    print("ATTENTION PATTERN COMPARISON: Deontological vs Utilitarian")
    print("="*60 + "\n")

    print("Mean differences (Deontological - Utilitarian):")
    print(f"  Action keywords:     {comparison_df['action_attn_diff'].mean():+.6f}")
    print(f"  Opponent context:    {comparison_df['opponent_attn_diff'].mean():+.6f}")
    print(f"  Payoff information:  {comparison_df['payoff_attn_diff'].mean():+.6f}")

    print("\nBy scenario:")
    scenario_means = comparison_df.groupby('scenario')[['action_attn_diff', 'opponent_attn_diff', 'payoff_attn_diff']].mean()
    print(scenario_means)
    print("\nSequence metric diffs (Deontological - Utilitarian):")
    print(
        comparison_df.groupby('scenario')[['seq_delta_diff', 'seq_p_action2_diff']].mean()
    )

    return comparison_df
