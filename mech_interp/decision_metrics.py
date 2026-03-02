"""Shared decision metrics for mech-interp analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import torch


@dataclass
class ActionPreference:
    """Preference over action1 vs action2 for a single prompt."""

    logp_action1: float
    logp_action2: float
    delta_logp_action2_minus_action1: float
    p_action1: float
    p_action2: float
    preferred_action: str


def prepare_prompt(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool = True,
) -> str:
    """Apply inference-style chat formatting when requested."""
    if not use_chat_template:
        return prompt
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def encode_action_variants(tokenizer, action_label: str) -> List[List[int]]:
    """Encode both spaced and unspaced forms of an action label."""
    candidates = [f" {action_label}", action_label]
    variants: List[List[int]] = []
    seen = set()
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        variants.append(ids)
    if not variants:
        raise ValueError(f"Could not tokenize action label: {action_label}")
    return variants


def sequence_logprob(
    forward_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    token_ids: Sequence[int],
) -> float:
    """Compute conditional log-probability of token_ids after input_ids."""
    context = input_ids
    total_logprob = 0.0
    for token_id in token_ids:
        with torch.no_grad():
            logits = forward_logits_fn(context)[:, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        total_logprob += float(logprobs[0, token_id].item())
        next_token = torch.tensor([[token_id]], device=context.device, dtype=context.dtype)
        context = torch.cat([context, next_token], dim=1)
    return total_logprob


def compute_action_sequence_preference(
    *,
    forward_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    tokenizer,
    input_ids: torch.Tensor,
    action1_label: str = "action1",
    action2_label: str = "action2",
) -> ActionPreference:
    """Compare sequence probabilities for action1 vs action2."""
    action1_variants = encode_action_variants(tokenizer, action1_label)
    action2_variants = encode_action_variants(tokenizer, action2_label)

    action1_variant_logps = np.array(
        [sequence_logprob(forward_logits_fn, input_ids, seq) for seq in action1_variants],
        dtype=np.float64,
    )
    action2_variant_logps = np.array(
        [sequence_logprob(forward_logits_fn, input_ids, seq) for seq in action2_variants],
        dtype=np.float64,
    )

    logp_action1 = float(np.logaddexp.reduce(action1_variant_logps))
    logp_action2 = float(np.logaddexp.reduce(action2_variant_logps))
    delta_logp = float(logp_action2 - logp_action1)

    max_logp = max(logp_action1, logp_action2)
    p1_unnorm = np.exp(logp_action1 - max_logp)
    p2_unnorm = np.exp(logp_action2 - max_logp)
    z = p1_unnorm + p2_unnorm
    p_action1 = float(p1_unnorm / z)
    p_action2 = float(p2_unnorm / z)

    return ActionPreference(
        logp_action1=logp_action1,
        logp_action2=logp_action2,
        delta_logp_action2_minus_action1=delta_logp,
        p_action1=p_action1,
        p_action2=p_action2,
        preferred_action=action2_label if delta_logp > 0 else action1_label,
    )

