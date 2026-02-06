# Sequence-Level Logit Lens Trajectories

## Summary

Implemented sequence-level trajectory computation for logit lens steering experiments to comply with [LOGIT_DECISION_METRIC_LESSONS.md](LOGIT_DECISION_METRIC_LESSONS.md) guidelines.

## What Changed

### 1. New Methods in LogitLensAnalyzer

Added to [mech_interp/logit_lens.py](../../mech_interp/logit_lens.py):

**`compute_action_trajectory_sequence_level()`**
- Uses first-token log-probability of "action1" vs "action2" sequences
- Returns `delta_logp = logp_action2 - logp_action1` at each layer
- **Positive = prefers action2 (Defect)**, Negative = prefers action1 (Cooperate)
- Note: In IPD, action1=Cooperate, action2=Defect

**`compute_action_trajectory_with_steering_sequence_level()`**
- Same as above, but with steering hook applied
- Properly handles both MLP and attention steering

### 2. Legacy Methods Marked

The original methods now have `[LEGACY]` markers:
- `compute_action_trajectory()` - Uses single-char tokens 'C' and 'D'
- `compute_action_trajectory_with_steering()` - Same, with steering

These use the **same sign convention**:
- **Positive = prefers Defect**, Negative = prefers Cooperate

### 3. Overlay Plotting Script Updated

[scripts/mech_interp/plot_logit_lens_steering_overlays.py](../../scripts/mech_interp/plot_logit_lens_steering_overlays.py):

**Configuration flag**:
```python
USE_SEQUENCE_LEVEL = True  # Set to False for legacy behavior
```

**Automatic handling**:
- Cache files: `trajectory_cache_sequence_level.pt` vs `trajectory_cache_legacy.pt`
- Y-axis labels: "Log-Prob Difference (action2 - action1)" vs "Logit Difference (Defect - Cooperate) [LEGACY]"
- Plot annotations: Correctly interpret positive/negative based on metric

## Compliance with Guidelines

✅ **Uses sequence preference** (first-token log-prob) instead of single-token 'C' vs 'D'
✅ **Uses prepare_prompt()** for inference-style formatting
✅ **Legacy metrics clearly labeled** in code and plots
✅ **Behavioral evaluation** (in `activation_steering.py`) already used proper metrics

## Key Differences

| Aspect | Legacy (C vs D) | Sequence-Level (action1 vs action2) |
|--------|----------------|-------------------------------------|
| **Tokens** | Single chars 'C', 'D' | Multi-token sequences "action1", "action2" |
| **Metric** | Logit difference | Log-probability difference (first token) |
| **Positive means** | Prefers Defect | Prefers action2 (Defect) |
| **Negative means** | Prefers Cooperate | Prefers action1 (Cooperate) |
| **Alignment** | May not match behavior | Better aligned with decision metrics |
| **Sign convention** | D - C | action2 - action1 (same as D - C) |

## Usage

### Using Sequence-Level (Default)

```python
from mech_interp.logit_lens import LogitLensAnalyzer

analyzer = LogitLensAnalyzer(model, tokenizer)

# Baseline trajectory
traj = analyzer.compute_action_trajectory_sequence_level(prompt)

# Steered trajectory
traj_steered = analyzer.compute_action_trajectory_with_steering_sequence_level(
    prompt=prompt,
    steering_layer=16,
    steering_component="mlp",
    steering_vector=vector,
    steering_strength=2.0,
)
```

### Using Legacy (For Comparison)

```python
# Legacy single-token method
traj = analyzer.compute_action_trajectory(prompt)
```

### Regenerating Plots

**Sequence-level** (default):
```bash
python scripts/mech_interp/plot_logit_lens_steering_overlays.py
```

**Legacy**:
```python
# In plot_logit_lens_steering_overlays.py, set:
USE_SEQUENCE_LEVEL = False
```

Then run the script.

## Interpretation Guide

### Sequence-Level Plots

**Y-axis**: Log-Prob Difference (action2 - action1)

- **Positive values**: Model prefers action2 (Defect in IPD)
- **Negative values**: Model prefers action1 (Cooperate in IPD)
- **Zero**: Equal preference

**Steering effects**:
- Positive steering (+2.0) → Shifts toward cooperation (more negative)
- Negative steering (-2.0) → Shifts toward defection (more positive)

### Legacy Plots (If Used)

**Y-axis**: Logit Difference (Defect - Cooperate) [LEGACY]

- **Positive values**: Model prefers Defect
- **Negative values**: Model prefers Cooperate
- **Note**: Sign convention is **same** as sequence-level (both are Defect - Cooperate)

## Validation

The sequence-level metrics better align with:
1. Behavioral evaluation in `activation_steering.py` (which uses `p_action2`)
2. Decision metrics guidelines in [LOGIT_DECISION_METRIC_LESSONS.md](LOGIT_DECISION_METRIC_LESSONS.md)
3. Actual model generation behavior

## Performance

**Computational cost**: Similar to legacy method
- Only computes first-token log-probs, not full sequence
- Uses same logit lens infrastructure
- No additional model forward passes

**Cache files**: ~40KB (similar to legacy)

## Recommendations

1. **Use sequence-level by default** for new analyses
2. **Keep legacy available** for backward compatibility and comparison
3. **Document which metric was used** in papers/presentations
4. **Validate alignment** with behavioral metrics (see [steering effectiveness ranking](../../mech_interp_outputs/causal_routing/steering_effectiveness_ranking.csv))

## Related Files

- [mech_interp/logit_lens.py](../../mech_interp/logit_lens.py) - Core implementation
- [mech_interp/decision_metrics.py](../../mech_interp/decision_metrics.py) - Shared metrics
- [scripts/mech_interp/plot_logit_lens_steering_overlays.py](../../scripts/mech_interp/plot_logit_lens_steering_overlays.py) - Plotting
- [docs/reports/LOGIT_DECISION_METRIC_LESSONS.md](LOGIT_DECISION_METRIC_LESSONS.md) - Guidelines
- [docs/reports/ACTIVATION_STEERING_EXPERIMENTS.md](ACTIVATION_STEERING_EXPERIMENTS.md) - Steering docs

## Future Work

**Full sequence-level (expensive)**:
- Compute full multi-token sequence probabilities at each layer
- Requires multiple forward passes per layer
- May provide even better alignment with behavioral metrics

Currently using **first-token proxy** as a good compromise between accuracy and efficiency.
