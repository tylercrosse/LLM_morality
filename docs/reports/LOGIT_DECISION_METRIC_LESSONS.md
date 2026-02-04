# Logit/Decision Metric Lessons Learned

This note summarizes what changed in the recent mech-interp refactor and
how to avoid repeating the same decision-metric mistakes in future work.

## The Core Mistake

We were often interpreting behavior from a **single final-token logit
difference** (e.g., `D - C` at one position), while inference behavior is
driven by a **sequence decision** (probability of generating `action1` vs
`action2` continuation).

That mismatch can make models look artificially similar in analyses even
when generation behavior is clearly different.

## What We Changed

### 1) Standardized decision metric across analyses

Added a shared utility:

- `mech_interp/decision_metrics.py`

Key behavior:

- Applies inference-style prompt formatting (`prepare_prompt`)
- Computes sequence log-prob preference:
  - `delta_logp_action2_minus_action1`
  - `p_action1`, `p_action2`
  - `preferred_action`

### 2) Migrated core analysis pipelines to use the shared metric

Updated:

- `mech_interp/logit_lens.py`
- `mech_interp/activation_patching.py`
- `mech_interp/direct_logit_attribution.py`
- `mech_interp/attention_analysis.py`
- `mech_interp/component_interactions.py`

Runner scripts updated accordingly:

- `docs/reports/scripts/mech_interp/run_logit_lens.py`
- `docs/reports/scripts/mech_interp/run_patching.py`
- `docs/reports/scripts/mech_interp/run_dla.py`
- `docs/reports/scripts/mech_interp/run_attention_analysis.py`
- `docs/reports/scripts/mech_interp/run_component_interactions.py`

### 3) Kept legacy token-delta where useful

For compatibility and comparisons with old outputs, some pipelines still
export legacy single-token deltas. These are now secondary diagnostics, not
the primary behavior metric.

### 4) Aligned model loading with inference-style behavior

Hooked-model loading was aligned to use non-merged LoRA + quantized paths
when needed (matching inference behavior more closely).

Relevant file:

- `mech_interp/model_loader.py`

### 5) Added explicit validation harnesses

New scripts to verify that sequence preference matches sampled outputs and
to test model separation statistically:

- `docs/reports/scripts/mech_interp/validate_mech_interp_alignment.py`
- `docs/reports/scripts/mech_interp/analyze_prompt_sensitivity.py`

## Why This Fix Matters

Before:

- “No separation” in some mech-interp plots
- But clear separation in generation experiments
- Confusion about whether prompts or models were the issue

After:

- Behavior metric now reflects the same object inference uses
- Separation is measured on comparable semantics across pipelines
- Prompt sensitivity and alignment checks can catch metric drift early

## Rules to Follow in Future Experiments

1. **Never treat single-token `D-C` as primary behavior evidence.**
2. **Always report sequence preference (`delta_logp_action2_minus_action1`
   and `p_action2`) for action choices.**
3. **Use the shared helper (`decision_metrics.py`) instead of duplicating
   custom token logic in each module.**
4. **Use inference-style prompt formatting (`prepare_prompt`) whenever
   comparing to generation behavior.**
5. **Validate alignment with sampled outputs before interpreting circuits.**
6. **If legacy metrics are kept, clearly label them “legacy” in plots/CSVs.**

## Practical Pre-Flight Checklist

Before trusting a new mechanistic result:

- [ ] Prompt is passed through shared `prepare_prompt(...)`
- [ ] Decision is sequence-level (`action1` vs `action2`)
- [ ] Inference and analysis use consistent model-loading settings
- [ ] Per-scenario/model sequence preference agrees with sampled majority
- [ ] Pairwise separation tests (e.g., Strategic vs De/Ut) are significant
- [ ] Any legacy token metric is explicitly secondary

## Common Failure Modes to Watch

- Looking at the wrong token position for multi-token action strings
- Mixing chat-templated and raw prompts across scripts
- Comparing merged-LoRA analysis runs to unmerged inference behavior
- Assuming top-1 next-token logit is equivalent to sequence-level action

## Quick Debug Path When Results Look Wrong

1. Run:
   - `docs/reports/scripts/mech_interp/validate_mech_interp_alignment.py`
2. Check:
   - `alignment_by_scenario_model.csv`
   - `alignment_confusion_table.csv`
3. If agreement is low:
   - inspect prompt formatting consistency
   - inspect action tokenization variants
   - verify sequence metric is actually used as primary in the module

## Bottom Line

Use **one shared sequence decision metric** across all analysis pipelines,
and always cross-check against sampled inference behavior before making
mechanistic claims.
