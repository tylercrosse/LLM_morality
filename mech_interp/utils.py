"""Shared utilities for mechanistic interpretability analysis."""

import json
from pathlib import Path
from typing import Dict

# Model labels for plotting and analysis
MODEL_LABELS = {
    "base": "Base (no FT)",
    "PT2_COREDe": "Strategic",
    "PT3_COREDe": "Deontological",
    "PT3_COREUt": "Utilitarian",
    "PT4_COREDe": "Hybrid",
}

# Model colors for consistent visualization
MODEL_COLORS = {
    "base": "#9AA0A6",
    "PT2_COREDe": "#4C78A8",
    "PT3_COREDe": "#59A14F",
    "PT3_COREUt": "#F28E2B",
    "PT4_COREDe": "#B07AA1",
}


def get_action_token_ids(tokenizer) -> Dict:
    """
    Get token IDs for 'action1' (Cooperate) and 'action2' (Defect) actions.

    The training uses 'action1' and 'action2' as the action tokens.
    These are typically multi-token sequences (e.g., ["action", "1"]).

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary mapping action names to token ID lists
        Also includes 'first_token' field for quick comparison
    """
    # Use the actual tokens from training: action1 (C) and action2 (D)
    action1_tokens = tokenizer.encode("action1", add_special_tokens=False)
    action2_tokens = tokenizer.encode("action2", add_special_tokens=False)

    # For logit lens, we track the first differentiating token
    # Usually: action1 = ["action", "1"] and action2 = ["action", "2"]
    # So we want to compare logits at the position of "1" vs "2"

    return {
        "action1_tokens": action1_tokens,
        "action2_tokens": action2_tokens,
        # First token that's different between action1 and action2
        # (usually the digit)
        "action1_first_diff": action1_tokens[-1],  # Last token (the digit)
        "action2_first_diff": action2_tokens[-1],
        # For backward compatibility
        "Cooperate": action1_tokens[-1],
        "Defect": action2_tokens[-1],
        "C": action1_tokens[-1],
        "D": action2_tokens[-1],
        "action1": action1_tokens[-1],
        "action2": action2_tokens[-1],
    }


def load_prompt_dataset(path: str = None):
    """
    Load JSON prompt dataset.

    Args:
        path: Path to JSON file (default: standard dataset path)

    Returns:
        List of prompt dictionaries
    """
    if path is None:
        path = "/root/LLM_morality/mech_interp_outputs/prompt_datasets/ipd_eval_prompts.json"

    with open(path) as f:
        return json.load(f)


def get_model_path(model_id: str) -> str:
    """
    Get the full path for a model checkpoint.

    Args:
        model_id: Model identifier (e.g., 'PT2_COREDe')

    Returns:
        Full path to model directory
    """
    base_dir = Path("/root/LLM_morality/models")

    if model_id == "base":
        return str(base_dir / "gemma-2-2b-it")

    # Map model IDs to checkpoint directory names
    checkpoint_map = {
        "PT2_COREDe": "gemma-2-2b-it_FT_PT2_oppTFT_run1_1000ep_COREDe",
        "PT3_COREDe": "gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREDe",
        "PT3_COREUt": "gemma-2-2b-it_FT_PT3_oppTFT_run1_1000ep_COREUt",
        "PT4_COREDe": "gemma-2-2b-it_FT_PT4_oppTFT_run1_1000ep_COREDe",
    }

    if model_id not in checkpoint_map:
        raise ValueError(f"Unknown model_id: {model_id}. Must be one of {list(checkpoint_map.keys())}")

    return str(base_dir / checkpoint_map[model_id])
