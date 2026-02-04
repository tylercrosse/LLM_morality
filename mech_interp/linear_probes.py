"""
Linear Probes for detecting moral concepts in residual stream activations.

Trains linear classifiers to detect:
1. Betrayal concepts (CD/DC scenarios)
2. Joint payoff representations (cooperative outcomes)

This maps the "Geography of Judgment" across model layers.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from mech_interp.model_loader import HookedGemmaModel
from mech_interp.decision_metrics import prepare_prompt
from mech_interp.utils import MODEL_LABELS, MODEL_COLORS


@dataclass
class ProbeLabel:
    """Labels for probe training from a single prompt."""

    # Probe targets
    is_betrayal: bool  # True for CD_punished/DC_exploited scenarios
    current_joint_payoff: float  # Payoff from current state (CC=6, DD=2, CD/DC=4)

    # State information (for reference)
    my_last_action: str  # "action1" or "action2"
    opp_last_action: str  # "action1" or "action2"
    scenario: str  # Full scenario name


@dataclass
class ProbeResult:
    """Results from training a single probe."""

    model_id: str
    probe_type: str  # "betrayal" or "payoff"
    layer: int

    # Performance metrics
    train_accuracy: float  # For classification
    test_accuracy: float
    train_r2: float  # For regression
    test_r2: float
    train_mse: float
    test_mse: float

    # Training details
    n_train_samples: int
    n_test_samples: int
    coef_norm: float  # L2 norm of probe coefficients
    intercept: float


class LabelGenerator:
    """Generate probe labels from prompt dataset."""

    # IPD payoff mapping
    PAYOFF_MAP = {
        ("action1", "action1"): (3, 3),  # CC
        ("action1", "action2"): (0, 4),  # CD
        ("action2", "action1"): (4, 0),  # DC
        ("action2", "action2"): (1, 1),  # DD
    }

    @staticmethod
    def generate_labels(prompts: List[Dict]) -> List[ProbeLabel]:
        """
        Generate labels for probe training from prompt dataset.

        Args:
            prompts: List of prompt dictionaries from ipd_eval_prompts.json

        Returns:
            List of ProbeLabel objects
        """
        labels = []

        for prompt_data in prompts:
            scenario = prompt_data["scenario"]
            my_action = prompt_data["state_self"]
            opp_action = prompt_data["state_opp"]

            # Determine if this is a betrayal scenario
            # Betrayal: CD_punished (I cooperated, opponent defected)
            #           DC_exploited (I defected, opponent cooperated)
            is_betrayal = scenario in ["CD_punished", "DC_exploited"]

            # Compute joint payoff from current state
            my_payoff, opp_payoff = LabelGenerator.PAYOFF_MAP[(my_action, opp_action)]
            joint_payoff = my_payoff + opp_payoff

            labels.append(
                ProbeLabel(
                    is_betrayal=is_betrayal,
                    current_joint_payoff=float(joint_payoff),
                    my_last_action=my_action,
                    opp_last_action=opp_action,
                    scenario=scenario,
                )
            )

        return labels


class ActivationExtractor:
    """Extract residual stream activations for probe training."""

    def __init__(self, model: HookedGemmaModel, use_chat_template: bool = True):
        """
        Args:
            model: HookedGemmaModel with activation caching
            use_chat_template: Whether to apply chat template to prompts
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.use_chat_template = use_chat_template

    def extract_layer_activations(self, prompts: List[str]) -> np.ndarray:
        """
        Extract residual stream activations at final token position for all layers.

        Args:
            prompts: List of raw prompt strings

        Returns:
            Activations array of shape (n_prompts, n_layers, d_model)
        """
        n_prompts = len(prompts)
        n_layers = self.model.n_layers
        d_model = self.model.d_model

        # Initialize output array
        activations = np.zeros((n_prompts, n_layers, d_model), dtype=np.float32)

        # Extract activations for each prompt
        for i, prompt in enumerate(prompts):
            # Prepare prompt with chat template
            prepared_prompt = prepare_prompt(
                self.tokenizer, prompt, use_chat_template=self.use_chat_template
            )

            # Tokenize
            input_ids = self.tokenizer(prepared_prompt, return_tensors="pt").input_ids.to(
                self.model.device
            )

            # Run with cache
            _, cache = self.model.run_with_cache(input_ids)

            # Extract residual stream at final token for each layer
            for layer_idx in range(n_layers):
                cache_key = f"blocks.{layer_idx}.hook_resid_post"
                # Get final token: [batch=1, seq_len, d_model] -> [d_model]
                layer_activation = cache[cache_key][0, -1, :].cpu().numpy()
                activations[i, layer_idx, :] = layer_activation

        return activations


class LinearProbeTrainer:
    """Train linear probes on residual stream activations."""

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def train_betrayal_probe(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[LogisticRegression, Dict[str, float]]:
        """
        Train binary classification probe for betrayal detection.

        Args:
            X: Activations (n_samples, d_model)
            y: Binary labels (n_samples,)

        Returns:
            trained_probe: Fitted LogisticRegression model
            metrics: Dict with train/test accuracy
        """
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        # Train probe
        probe = LogisticRegression(random_state=self.random_state, max_iter=1000)
        probe.fit(X_train, y_train)

        # Evaluate
        train_preds = probe.predict(X_train)
        test_preds = probe.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "test_accuracy": accuracy_score(y_test, test_preds),
            "train_r2": 0.0,  # Not applicable for classification
            "test_r2": 0.0,
            "train_mse": 0.0,
            "test_mse": 0.0,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "coef_norm": float(np.linalg.norm(probe.coef_)),
            "intercept": float(probe.intercept_[0]),
        }

        return probe, metrics

    def train_payoff_probe(
        self, X: np.ndarray, y: np.ndarray, y_betrayal: np.ndarray
    ) -> Tuple[Ridge, Dict[str, float]]:
        """
        Train regression probe for joint payoff prediction.

        Args:
            X: Activations (n_samples, d_model)
            y: Continuous payoff labels (n_samples,)
            y_betrayal: Binary betrayal labels for stratified split

        Returns:
            trained_probe: Fitted Ridge regression model
            metrics: Dict with train/test R² and MSE
        """
        # Train-test split (stratified by betrayal to ensure balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y_betrayal
        )

        # Train probe
        probe = Ridge(alpha=1.0, random_state=self.random_state)
        probe.fit(X_train, y_train)

        # Evaluate
        train_preds = probe.predict(X_train)
        test_preds = probe.predict(X_test)

        metrics = {
            "train_accuracy": 0.0,  # Not applicable for regression
            "test_accuracy": 0.0,
            "train_r2": r2_score(y_train, train_preds),
            "test_r2": r2_score(y_test, test_preds),
            "train_mse": mean_squared_error(y_train, train_preds),
            "test_mse": mean_squared_error(y_test, test_preds),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "coef_norm": float(np.linalg.norm(probe.coef_)),
            "intercept": float(probe.intercept_),
        }

        return probe, metrics

    def train_all_layers(
        self,
        model_id: str,
        activations: np.ndarray,
        labels: List[ProbeLabel],
    ) -> List[ProbeResult]:
        """
        Train probes for all layers and both probe types.

        Args:
            model_id: Model identifier
            activations: Array of shape (n_prompts, n_layers, d_model)
            labels: List of ProbeLabel objects

        Returns:
            List of ProbeResult objects (26 layers × 2 types = 52 results)
        """
        n_prompts, n_layers, d_model = activations.shape

        # Extract label arrays
        y_betrayal = np.array([label.is_betrayal for label in labels])
        y_payoff = np.array([label.current_joint_payoff for label in labels])

        results = []

        # Train probes for each layer
        for layer_idx in range(n_layers):
            X = activations[:, layer_idx, :]  # (n_prompts, d_model)

            # Train betrayal probe
            _, betrayal_metrics = self.train_betrayal_probe(X, y_betrayal)
            results.append(
                ProbeResult(
                    model_id=model_id,
                    probe_type="betrayal",
                    layer=layer_idx,
                    **betrayal_metrics,
                )
            )

            # Train payoff probe
            _, payoff_metrics = self.train_payoff_probe(X, y_payoff, y_betrayal)
            results.append(
                ProbeResult(
                    model_id=model_id,
                    probe_type="payoff",
                    layer=layer_idx,
                    **payoff_metrics,
                )
            )

        return results


class LinearProbeVisualizer:
    """Generate visualizations for probe results."""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_probe_trajectories(
        self, results: List[ProbeResult], model_id: str, model_name: str
    ):
        """
        Plot probe performance across layers for a single model.

        Args:
            results: List of ProbeResult objects
            model_id: Model identifier
            model_name: Human-readable model name
        """
        # Separate by probe type
        betrayal_results = [r for r in results if r.probe_type == "betrayal"]
        payoff_results = [r for r in results if r.probe_type == "payoff"]

        # Sort by layer
        betrayal_results.sort(key=lambda r: r.layer)
        payoff_results.sort(key=lambda r: r.layer)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot betrayal probe trajectory
        layers = [r.layer for r in betrayal_results]
        test_acc = [r.test_accuracy for r in betrayal_results]

        axes[0].plot(layers, test_acc, marker="o", linewidth=2, markersize=4)
        axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        axes[0].set_xlabel("Layer", fontsize=11)
        axes[0].set_ylabel("Test Accuracy", fontsize=11)
        axes[0].set_title(f"Betrayal Probe - {model_name}", fontsize=12, fontweight="bold")
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim([0.0, 1.0])
        axes[0].legend()

        # Plot payoff probe trajectory
        layers = [r.layer for r in payoff_results]
        test_r2 = [r.test_r2 for r in payoff_results]

        axes[1].plot(layers, test_r2, marker="o", linewidth=2, markersize=4, color="orange")
        axes[1].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
        axes[1].set_xlabel("Layer", fontsize=11)
        axes[1].set_ylabel("Test R²", fontsize=11)
        axes[1].set_title(f"Payoff Probe - {model_name}", fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"probe_trajectories_{model_id}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_model_comparison(
        self, all_results: List[ProbeResult], probe_type: str, title_suffix: str
    ):
        """
        Plot probe performance comparison across models.

        Args:
            all_results: List of ProbeResult objects from all models
            probe_type: "betrayal" or "payoff"
            title_suffix: Title suffix for plot
        """
        # Filter by probe type
        results = [r for r in all_results if r.probe_type == probe_type]

        # Group by model
        model_results = {}
        for r in results:
            if r.model_id not in model_results:
                model_results[r.model_id] = []
            model_results[r.model_id].append(r)

        # Create plot
        plt.figure(figsize=(10, 5))

        for model_id, model_res in model_results.items():
            # Sort by layer
            model_res.sort(key=lambda r: r.layer)

            layers = [r.layer for r in model_res]
            if probe_type == "betrayal":
                values = [r.test_accuracy for r in model_res]
                ylabel = "Test Accuracy"
            else:
                values = [r.test_r2 for r in model_res]
                ylabel = "Test R²"

            model_name = MODEL_LABELS.get(model_id, model_id)
            color = MODEL_COLORS.get(model_id, "gray")

            plt.plot(layers, values, marker="o", linewidth=2, markersize=3, label=model_name, color=color)

        # Add baseline
        if probe_type == "betrayal":
            plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
            plt.ylim([0.0, 1.0])
        else:
            plt.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")

        plt.xlabel("Layer", fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.title(title_suffix, fontsize=12, fontweight="bold")
        plt.legend(loc="best", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = f"{probe_type}_probe_comparison.png"
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_peak_layer_heatmap(self, summary_df: pd.DataFrame):
        """
        Plot heatmap of peak performance layers for each model × probe type.

        Args:
            summary_df: DataFrame with probe summary statistics
        """
        # Pivot to create heatmap data
        # Rows: model_name, Columns: probe_type, Values: peak_layer
        pivot_data = summary_df.pivot(index="model_name", columns="probe_type", values="peak_layer")

        # Create heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            cbar_kws={"label": "Peak Layer"},
            linewidths=0.5,
        )
        plt.title("Peak Performance Layer by Model and Probe Type", fontsize=12, fontweight="bold")
        plt.ylabel("Model", fontsize=11)
        plt.xlabel("Probe Type", fontsize=11)
        plt.tight_layout()
        plt.savefig(self.output_dir / "peak_layer_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()


def export_results(results: List[ProbeResult], output_path: Path):
    """
    Export probe results to CSV.

    Args:
        results: List of ProbeResult objects
        output_path: Path to save CSV file
    """
    # Convert to DataFrame
    rows = []
    for r in results:
        rows.append(
            {
                "model_id": r.model_id,
                "probe_type": r.probe_type,
                "layer": r.layer,
                "train_accuracy": r.train_accuracy,
                "test_accuracy": r.test_accuracy,
                "train_r2": r.train_r2,
                "test_r2": r.test_r2,
                "train_mse": r.train_mse,
                "test_mse": r.test_mse,
                "n_train_samples": r.n_train_samples,
                "n_test_samples": r.n_test_samples,
                "coef_norm": r.coef_norm,
                "intercept": r.intercept,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def compute_summary(all_results: List[ProbeResult]) -> pd.DataFrame:
    """
    Compute summary statistics across all models and probe types.

    Args:
        all_results: List of all ProbeResult objects

    Returns:
        DataFrame with summary statistics
    """
    summary_rows = []

    # Group by model and probe type
    for model_id in set(r.model_id for r in all_results):
        for probe_type in ["betrayal", "payoff"]:
            # Filter results
            results = [
                r
                for r in all_results
                if r.model_id == model_id and r.probe_type == probe_type
            ]

            if not results:
                continue

            # Compute statistics
            if probe_type == "betrayal":
                test_values = [r.test_accuracy for r in results]
                peak_value = max(test_values)
                mean_value = np.mean(test_values)
                peak_layer = results[np.argmax(test_values)].layer

                summary_rows.append(
                    {
                        "model_id": model_id,
                        "model_name": MODEL_LABELS.get(model_id, model_id),
                        "probe_type": probe_type,
                        "peak_layer": peak_layer,
                        "peak_test_accuracy": peak_value,
                        "peak_test_r2": 0.0,
                        "mean_test_accuracy": mean_value,
                        "mean_test_r2": 0.0,
                    }
                )
            else:  # payoff
                test_values = [r.test_r2 for r in results]
                peak_value = max(test_values)
                mean_value = np.mean(test_values)
                peak_layer = results[np.argmax(test_values)].layer

                summary_rows.append(
                    {
                        "model_id": model_id,
                        "model_name": MODEL_LABELS.get(model_id, model_id),
                        "probe_type": probe_type,
                        "peak_layer": peak_layer,
                        "peak_test_accuracy": 0.0,
                        "peak_test_r2": peak_value,
                        "mean_test_accuracy": 0.0,
                        "mean_test_r2": mean_value,
                    }
                )

    return pd.DataFrame(summary_rows)
