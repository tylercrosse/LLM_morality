# Logit Lens Investigation: Why Are Final Layer Differences So Small?

**Date**: 2026-02-03
**Issue**: The logit lens shows nearly identical final-layer values across models (differences of 0.005-0.042), but behavioral outputs show clear differences between Strategic, Deontological, and Utilitarian models.

---

## Hypothesis

**The eval prompts may not be capturing the scenarios that elicit different behaviors.**

### Evidence

1. **Eval prompts are limited to IPD only**
   - `ipd_eval_prompts.json` contains 15 prompts, all using standard IPD payoffs (3,3 / 0,4 / 4,0 / 1,1)
   - All prompts test 5 scenarios: CC_continue, CC_temptation, CD_punished, DC_exploited, DD_trapped

2. **Inference code tests multiple game types**
   - `inference_vsRandom.py` evaluates on:
     - IPD (Prisoner's Dilemma)
     - ISH (Stag Hunt)
     - ICN (Iterated Chicken)
     - BOS (Battle of Sexes)
     - IVD (not clear)
     - ICD (not clear)
   - The `cross_game_generalization_publication.png` plot shows different patterns for Strategic vs Deontological/Utilitarian models

3. **Within-model variance >> between-model differences**
   - Final delta differences: 0.005-0.042
   - Within-model std: 0.15-0.19 (5-10x larger!)
   - Signal-to-noise ratio: ~0.05-0.15

### Specific Findings from CSV Analysis

```
CC_temptation scenario:
  Strategic:      -1.5469
  Deontological:  -1.5417 (diff: +0.0052)
  Utilitarian:    -1.5729 (diff: -0.0260)

CD_punished scenario:
  Strategic:      -1.3229
  Deontological:  -1.3125 (diff: +0.0104)
  Utilitarian:    -1.3542 (diff: -0.0312)
```

The largest difference is 0.0312 (3% of the absolute value), which is tiny!

---

## Why This Matters

Even small logit differences can create behavioral differences when:

1. **Sampling is used**: `inference_vsRandom.py` uses `top_p = 1.0`, which means nucleus sampling is active. Small logit differences can shift sampling probabilities.

2. **Multiple rounds amplify effects**: In iterated games, small per-round differences compound over many episodes.

3. **Different games have different equilibria**: The Strategic model might behave similarly to moral models in IPD (where cooperation is stable) but differently in Stag Hunt or Chicken (where equilibria are less clear).

---

## Recommendations

### Short-term: Verify the hypothesis

1. **Check actual behavioral outputs on eval prompts**
   ```python
   # Run actual inference (with sampling) on the eval prompts
   # Measure: what % of time does each model choose Cooperate vs Defect?
   # Expected: differences should be small (matching small logit diffs)
   ```

2. **Test logit lens on other game types**
   ```python
   # Create eval prompts for Stag Hunt, Chicken, Battle of Sexes
   # Run logit lens analysis on these
   # Expected: larger final-layer differences in games where models behave more differently
   ```

3. **Analyze per-variant distributions**
   - Instead of averaging over 3 variants per scenario, look at distributions
   - Check if models show bimodal distributions (some variants very different)

### Medium-term: Enhanced analysis

1. **Probability-space analysis**
   - Convert logit differences to probability differences: `P(Cooperate) = softmax([-delta, delta])[0]`
   - Show that even 0.03 logit difference = ~0.7% probability shift
   - For 100 episodes, this is ~7 different decisions

2. **Layer-wise divergence tracking**
   - Current analysis shows final layer is very similar
   - Check if models diverge more in mid-layers (L10-L20) but converge by final layer
   - This could explain: "different reasoning paths, same conclusion"

3. **Game-specific prompt dataset**
   - Create structured eval prompts for all game types tested in `inference_vsRandom.py`
   - Specifically test scenarios where Strategic differs from moral models

---

## Immediate Next Steps

1. **Verify that eval prompts don't elicit different behavior**
   ```bash
   # Run inference on eval prompts and measure cooperation rates
   python scripts/run_inference_on_eval_prompts.py
   ```

2. **Check the cross_game_generalization plot details**
   - What game types are shown?
   - What are the actual cooperation/defection rates?
   - Are differences statistically significant?

3. **Test logit lens on one non-IPD game**
   - Create 3-5 Stag Hunt prompts
   - Run logit lens
   - Check if final-layer differences are larger

---

## Open Questions

1. **Are the behavioral differences real or noise?**
   - Need to check if the `cross_game_generalization` plot shows statistically significant differences
   - If yes, then logit lens is missing something
   - If no, then models are actually very similar

2. **Is the logit lens measuring the right thing?**
   - Currently: `delta = logit(action2) - logit(action1)` at final token position
   - Should we be looking at probabilities instead?
   - Should we be looking at earlier layers?

3. **Why do Deontological and Utilitarian look identical in the logit lens?**
   - Final deltas differ by only 0.005-0.042
   - But they were trained with very different objectives!
   - Suggests either:
     a) Training didn't produce meaningfully different models
     b) Differences are in reasoning process (mid-layers), not final output
     c) Differences emerge only in specific game types not tested

---

## Conclusion

**Most likely explanation**: The eval prompts test only IPD scenarios where all models converge to similar cooperative behavior. The behavioral differences shown in the publication figures likely come from:
- Different game types (Stag Hunt, Chicken)
- Scenarios with less clear equilibria
- Accumulated differences over many episodes
- Sampling variance amplifying small logit differences

**Action**: Create eval prompts for non-IPD games and re-run logit lens analysis.
