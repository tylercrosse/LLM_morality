"""
LoRA Weight Transplant ("Frankenstein") Experiment.

Tests the hypothesis that L2_MLP acts as a routing switch by transplanting its
LoRA weights from one model to another. If L2_MLP weights are sufficient to
shift behavior, this provides direct causal evidence for the routing hypothesis.

Example:
    Strategic model + Deontological L2_MLP → Increased cooperation?
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import safetensors.torch
import copy

from mech_interp.weight_analysis import WeightAnalyzer
from mech_interp.model_loader import LoRAModelLoader
from mech_interp.decision_metrics import compute_action_sequence_preference, prepare_prompt
from mech_interp.utils import get_model_path, load_prompt_dataset


@dataclass
class TransplantResult:
    """Results from a weight transplant experiment."""

    source_model_id: str
    target_model_id: str
    transplanted_layer: int
    transplanted_modules: List[str]  # e.g., ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    # Behavioral results
    baseline_p_action2_mean: float
    transplanted_p_action2_mean: float
    delta_p_action2: float  # transplanted - baseline

    # Per-scenario results
    scenarios: List[str]
    baseline_p_action2: List[float]
    transplanted_p_action2: List[float]

    # Cooperation rate change (for IPD: 1 - p_action2 = cooperation)
    baseline_coop_rate: float
    transplanted_coop_rate: float
    delta_coop_rate: float


class LoRAWeightTransplanter:
    """
    Transplants LoRA weights from source model to target model.

    The "Frankenstein" experiment: Can we shift model behavior by swapping
    only L2_MLP's LoRA weights while keeping everything else constant?
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize transplanter.

        Args:
            device: Device to load models on
        """
        self.device = device

    def transplant_lora_weights(
        self,
        source_model_id: str,
        target_model_id: str,
        layer: int,
        module_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Transplant LoRA weights from source to target model.

        Args:
            source_model_id: Model to copy weights FROM (e.g., "PT3_COREDe")
            target_model_id: Model to copy weights TO (e.g., "PT2_COREDe")
            layer: Layer index (e.g., 2 for L2_MLP)
            module_types: Modules to transplant (e.g., ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])

        Returns:
            Dict of transplanted weights ready to load into target model
        """
        # Load source weights
        source_analyzer = WeightAnalyzer(source_model_id)
        source_weights = source_analyzer.weights

        # Load target weights (to copy everything else)
        target_analyzer = WeightAnalyzer(target_model_id)
        target_weights = target_analyzer.weights

        # Create new weight dict (copy from target)
        transplanted_weights = {}
        for key, value in target_weights.items():
            transplanted_weights[key] = value.clone()

        # Transplant specified modules from source
        transplanted_keys = []
        for module_type in module_types:
            base_key = f"base_model.model.model.layers.{layer}.{module_type}"
            a_key = f"{base_key}.lora_A.weight"
            b_key = f"{base_key}.lora_B.weight"

            if a_key in source_weights and b_key in source_weights:
                transplanted_weights[a_key] = source_weights[a_key].clone()
                transplanted_weights[b_key] = source_weights[b_key].clone()
                transplanted_keys.extend([a_key, b_key])
                print(f"✓ Transplanted {base_key}")
            else:
                print(f"⚠ Warning: {base_key} not found in source model")

        print(f"\nTransplanted {len(transplanted_keys)} weight tensors")
        return transplanted_weights

    def load_model_with_transplanted_weights(
        self,
        target_model_id: str,
        transplanted_weights: Dict[str, torch.Tensor],
    ):
        """
        Load target model and inject transplanted LoRA weights.

        Args:
            target_model_id: Base model to load
            transplanted_weights: Dict of weight tensors (from transplant_lora_weights)

        Returns:
            Model with transplanted weights loaded
        """
        # Load base model with LoRA (merged)
        # We load merged because we want the base model + new LoRA weights
        # and the easiest way is to: load merged target, then re-apply new weights

        # Actually, we need to load the base HF model without merging,
        # then manually set the LoRA weights, then merge
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig

        # Get paths
        base_model_name = "google/gemma-2-2b-it"
        target_model_path = Path(get_model_path(target_model_id))

        # Load base model
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        # Load PEFT model (this adds LoRA adapters)
        print(f"Loading LoRA adapters from: {target_model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            str(target_model_path),
            is_trainable=False,
        )

        # Now replace the LoRA weights with transplanted weights
        print("Injecting transplanted weights...")
        with torch.no_grad():
            state_dict = model.state_dict()

            # Update only the transplanted weights
            for key, value in transplanted_weights.items():
                if key in state_dict:
                    # Ensure same device and dtype
                    state_dict[key] = value.to(
                        device=state_dict[key].device,
                        dtype=state_dict[key].dtype
                    )

            # Load the modified state dict
            model.load_state_dict(state_dict)

        # Merge LoRA weights into base model for faster inference
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        return model, tokenizer

    def evaluate_model_on_ipd(
        self,
        model,
        tokenizer,
        scenarios: Optional[List[str]] = None,
    ) -> Tuple[float, List[float], List[str]]:
        """
        Evaluate model on IPD scenarios.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            scenarios: List of scenario names (default: all scenarios from dataset)

        Returns:
            Tuple of (mean_p_action2, per_scenario_p_action2, scenario_names)
        """
        # Load IPD prompts
        prompts = load_prompt_dataset()

        # Filter by scenarios if specified
        if scenarios is not None:
            prompts = [p for p in prompts if p["scenario"] in scenarios]

        # Get unique scenario names
        scenario_names = [p["scenario"] for p in prompts]

        p_action2_values = []

        for prompt_item in prompts:
            prompt_text = prompt_item["prompt"]

            # Prepare prompt with chat template
            formatted_prompt = prepare_prompt(tokenizer, prompt_text, use_chat_template=True)
            input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)

            # Compute sequence preference
            def forward_fn(input_ids):
                with torch.no_grad():
                    outputs = model(input_ids)
                    return outputs.logits

            pref = compute_action_sequence_preference(
                forward_logits_fn=forward_fn,
                tokenizer=tokenizer,
                input_ids=input_ids,
            )

            p_action2_values.append(pref.p_action2)

        mean_p_action2 = np.mean(p_action2_values)

        return mean_p_action2, p_action2_values, scenario_names

    def run_transplant_experiment(
        self,
        source_model_id: str,
        target_model_id: str,
        layer: int = 2,
        module_types: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
    ) -> TransplantResult:
        """
        Run complete Frankenstein experiment.

        Args:
            source_model_id: Model to copy L2_MLP FROM
            target_model_id: Model to copy L2_MLP TO
            layer: Layer to transplant (default: 2 for L2_MLP)
            module_types: Modules to transplant (default: all MLP modules)
            scenarios: Scenarios to evaluate on (default: all IPD scenarios)

        Returns:
            TransplantResult with baseline vs transplanted behavior
        """
        if module_types is None:
            module_types = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

        print("="*80)
        print(f"FRANKENSTEIN EXPERIMENT: {source_model_id} L{layer}_MLP → {target_model_id}")
        print("="*80)

        # Step 1: Evaluate baseline (target model without transplant)
        print("\n[Step 1/3] Evaluating baseline (target model)...")
        target_hooked = LoRAModelLoader.load_hooked_model(
            target_model_id,
            device=self.device
        )
        baseline_p_action2_mean, baseline_p_action2, scenario_names = \
            self.evaluate_model_on_ipd(target_hooked.model, target_hooked.tokenizer, scenarios)
        baseline_coop_rate = 1.0 - baseline_p_action2_mean

        print(f"  Baseline p(action2): {baseline_p_action2_mean:.4f}")
        print(f"  Baseline cooperation rate: {baseline_coop_rate:.2%}")

        # Clear memory
        del target_hooked
        torch.cuda.empty_cache()

        # Step 2: Transplant weights
        print(f"\n[Step 2/3] Transplanting weights from {source_model_id} to {target_model_id}...")
        transplanted_weights = self.transplant_lora_weights(
            source_model_id, target_model_id, layer, module_types
        )

        # Step 3: Evaluate transplanted model
        print("\n[Step 3/3] Evaluating transplanted model...")
        transplanted_model, transplanted_tokenizer = \
            self.load_model_with_transplanted_weights(target_model_id, transplanted_weights)

        transplanted_p_action2_mean, transplanted_p_action2, _ = \
            self.evaluate_model_on_ipd(transplanted_model, transplanted_tokenizer, scenario_names)
        transplanted_coop_rate = 1.0 - transplanted_p_action2_mean

        print(f"  Transplanted p(action2): {transplanted_p_action2_mean:.4f}")
        print(f"  Transplanted cooperation rate: {transplanted_coop_rate:.2%}")

        # Compute deltas
        delta_p_action2 = transplanted_p_action2_mean - baseline_p_action2_mean
        delta_coop_rate = transplanted_coop_rate - baseline_coop_rate

        # Summary
        print("\n" + "="*80)
        print("RESULTS:")
        print(f"  Δ p(action2):       {delta_p_action2:+.4f}")
        print(f"  Δ cooperation rate: {delta_coop_rate:+.2%}")
        print("="*80)

        # Clear memory
        del transplanted_model
        torch.cuda.empty_cache()

        return TransplantResult(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            transplanted_layer=layer,
            transplanted_modules=module_types,
            baseline_p_action2_mean=baseline_p_action2_mean,
            transplanted_p_action2_mean=transplanted_p_action2_mean,
            delta_p_action2=delta_p_action2,
            scenarios=scenario_names,
            baseline_p_action2=baseline_p_action2,
            transplanted_p_action2=transplanted_p_action2,
            baseline_coop_rate=baseline_coop_rate,
            transplanted_coop_rate=transplanted_coop_rate,
            delta_coop_rate=delta_coop_rate,
        )


def save_transplant_results(result: TransplantResult, output_dir: Path):
    """Save transplant experiment results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-scenario results
    df_scenarios = pd.DataFrame({
        "scenario": result.scenarios,
        "baseline_p_action2": result.baseline_p_action2,
        "transplanted_p_action2": result.transplanted_p_action2,
        "delta_p_action2": [t - b for t, b in zip(result.transplanted_p_action2, result.baseline_p_action2)],
        "baseline_coop_rate": [1 - p for p in result.baseline_p_action2],
        "transplanted_coop_rate": [1 - p for p in result.transplanted_p_action2],
    })

    # Save per-scenario
    filename = f"frankenstein_{result.source_model_id}_to_{result.target_model_id}_L{result.transplanted_layer}.csv"
    df_scenarios.to_csv(output_dir / filename, index=False)
    print(f"\nSaved per-scenario results to: {output_dir / filename}")

    # Summary
    summary = {
        "experiment": "frankenstein",
        "source_model_id": result.source_model_id,
        "target_model_id": result.target_model_id,
        "transplanted_layer": result.transplanted_layer,
        "transplanted_modules": result.transplanted_modules,
        "baseline_p_action2_mean": float(result.baseline_p_action2_mean),
        "transplanted_p_action2_mean": float(result.transplanted_p_action2_mean),
        "delta_p_action2": float(result.delta_p_action2),
        "baseline_coop_rate": float(result.baseline_coop_rate),
        "transplanted_coop_rate": float(result.transplanted_coop_rate),
        "delta_coop_rate": float(result.delta_coop_rate),
    }

    import json
    summary_file = output_dir / f"frankenstein_{result.source_model_id}_to_{result.target_model_id}_L{result.transplanted_layer}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")
