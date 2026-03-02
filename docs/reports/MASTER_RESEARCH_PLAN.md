# Mechanistic Interpretability: Updated Master Research Plan

**Date:** February 2026
**Status:** Active
**Context:** Merges original project plan with findings from Phase 1 (The "Wiring" Breakthrough).

This document outlines the consolidated strategy for the remainder of the project. It integrates the original task list with the new direction necessitated by the "Network Rewiring" discovery.

---

## Important Methodology Note

Before running new mech-interp experiments, review:

- `docs/reports/LOGIT_DECISION_METRIC_LESSONS.md`

This captures the sequence-vs-single-token logit mistake, the metric
standardization changes, and the pre-flight checklist to avoid repeating
the same issue in future analyses.

---

## Phase 1: Foundation (Completed)

*   **Task 0: Merge LoRA Adapters** (✅ Done)
    *   Created standalone checkpoints for Base, Strategic, Deontological, Utilitarian, and Hybrid models.
*   **Task 1: Evaluation Prompt Set** (✅ Done)
    *   Created 15 scenarios across 5 categories (Temptation, Punishment, etc.).
*   **Task 2 & 3: Logit Lens & Attribution** (✅ Done)
    *   **Finding:** Discovery of "Universal Neurons" (L8_MLP / L9_MLP) that encode cooperation/defection in *all* models.
*   **Task 4: Activation Patching** (✅ Done)
    *   **Finding:** The "Suppression Hypothesis" failed. 7,000+ patches could not flip behavior. The difference is distributed and robust.
*   **Task 5: Attention Pattern Analysis** (✅ Done)
    *   **Finding:** Null result. Deontological and Utilitarian models have nearly identical attention patterns (diff < $10^{-5}$).

---

## Phase 2: The "Wiring" Investigation (Completed ✅)

We have established that the models *look* at the same things (Task 5) but *act* differently. This phase investigated the "hidden circuitry" responsible for this divergence.

### Task 2.1: Component Interaction Analysis (✅ Done)
**Goal:** Identify the "switching nodes" where information routing diverges.
*   **Status:** Completed February 2-4, 2026
*   **Key Finding (Updated):** After metric validation and rerun:
    *   **541 pathways significantly different** (|Δ correlation| > 0.3, 40.8% of pairs)
    *   Distributed rewiring across multiple hubs (L19_ATTN, L1_MLP, L17_MLP)
    *   Original L2_MLP-centered hypothesis weakened; rewiring is more distributed
    *   Network rewiring hypothesis confirmed: same components, different connectivity

### Task 2.2: LoRA Weight Analysis (✅ Done - Original Task 7)
**Goal:** Determine if the "switching" behavior comes from weight changes in the node itself or changes in its inputs.
*   **Status:** Completed
*   **Method:**
    1.  Loaded LoRA adapters (unmerged).
    2.  Computed Frobenius norm of effective weight change (`||B @ A||`) for each module.
*   **Finding:** L2_MLP was NOT heavily retrained:
    *   Ranks 11-27th percentile (73-88% of components modified MORE)
    *   All models 99%+ similar in weight space
    *   Supports network rewiring (light modifications that change connectivity, not massive retraining)

### Task 2.3: Linear Probes (✅ Done - Original Task 6)
**Goal:** Map the "Geography of Judgment." Since attention is identical (Perception), probes will reveal where the *Representation* (Judgment) diverges.
*   **Status:** Completed
*   **Method:**
    1.  Created labeled datasets: "Is this a betrayal?" (Binary), "Joint Payoff Value" (Regression).
    2.  Trained logistic regression probes on the residual stream at every layer.
*   **Finding:** All models show **identical** probe performance:
    *   Betrayal accuracy: ~45% (chance level) across all models
    *   Joint payoff R²: 0.74-0.75 (strong but identical)
    *   Peak layer: 13 (universal across all 5 models, including base)
    *   **Null result:** No Deontological vs Utilitarian differences in linear representations
*   **Implication:** Differences operate at connectivity level, not attention or representation level

---

## Phase 3: Causal Validation Experiments (🔄 In Progress)

**Date Implemented:** February 5, 2026
**Status:** Implementation complete; experiments running (~2-3 hours estimated)

This phase tests the network rewiring hypothesis with direct causal interventions. Phases 1-2 provided correlational evidence; Phase 3 tests causality.

### Task 3.1: Frankenstein Experiment - LoRA Weight Transplant (🔄 Running)
**Goal:** Test if L2_MLP weights are causally sufficient to shift behavior.
*   **Status:** Implemented and running
*   **Updated Hypothesis:** If L2_MLP acts as a routing switch, transplanting its LoRA weights from one model to another should shift behavior.
*   **Method:**
    1.  Extract L2_MLP LoRA weights (gate_proj, up_proj, down_proj) from source model
    2.  Replace target model's L2_MLP weights with source weights
    3.  Merge LoRA and evaluate on 15 IPD scenarios
*   **Test Cases:**
    1.  Strategic + Deontological_L2 → Expect >5% cooperation increase
    2.  Deontological + Strategic_L2 → Expect >5% cooperation decrease
    3.  Utilitarian + Deontological_L2 → Expect cooperation increase
    4.  Deontological + Utilitarian_L2 → Expect cooperation decrease
*   **Implementation:**
    *   `mech_interp/lora_weight_transplant.py` (~400 lines)
    *   `scripts/mech_interp/run_frankenstein.py` (~280 lines)
*   **Success Criteria:** ≥3/4 hypotheses supported (Δ cooperation > 5%)
*   **Expected Runtime:** ~30-45 minutes
*   **Compliance:** ✅ Uses validated sequence-level metrics from LOGIT_DECISION_METRIC_LESSONS.md

### Task 3.2: Activation Steering (🔄 Running)
**Goal:** Test if steering L2_MLP activations provides continuous control over behavior.
*   **Status:** Implemented and running
*   **Hypothesis:** If L2_MLP routes moral information, steering its activations should provide continuous behavioral control.
*   **Method:**
    1.  Compute steering vector: mean(Deontological_L2_acts) - mean(Strategic_L2_acts)
    2.  Add scaled vector to L2_MLP activations during forward pass
    3.  Test strengths: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    4.  Measure downstream effects on L8/L9 MLPs
*   **Test Cases:**
    1.  Steer Strategic with +1.0 → Expect cooperation increase
    2.  Steer Deontological with -1.0 → Expect cooperation decrease
    3.  Steering sweep → Expect monotonic relationship
    4.  Downstream analysis → Measure L8/L9 activation changes
*   **Implementation:**
    *   `mech_interp/activation_steering.py` (~550 lines)
    *   `scripts/mech_interp/run_activation_steering.py` (~330 lines)
*   **Success Criteria:** Monotonic relationship + bidirectional control
*   **Expected Runtime:** ~30-45 minutes
*   **Compliance:** ✅ Uses validated sequence-level metrics

### Task 3.3: Path Patching (🔄 Running)
**Goal:** Test if L2→L9 pathway causally mediates behavior differences.
*   **Status:** Implemented and running
*   **Hypothesis:** If information flows causally through L2→L9, replacing this pathway should produce large shifts (unlike single-component patching which showed 0 flips).
*   **Method:**
    1.  Cache source model's residual stream activations at each layer
    2.  Replace target model's residual stream from L2→L9 during forward pass
    3.  Test three modes: full residual, MLP-only, attention-only
    4.  Progressive patching: L2→L2, L2→L3, ..., L2→L9 to find critical range
*   **Test Cases:**
    1.  Full path: Deontological → Strategic (L2→L9) → Expect >30% change
    2.  Full path: Strategic → Deontological (L2→L9) → Expect >30% change
    3.  Progressive: Find saturation point
    4.  MLP vs Attention: Decompose pathway contributions
*   **Implementation:**
    *   `mech_interp/path_patching.py` (~450 lines)
    *   `scripts/mech_interp/run_path_patching.py` (~350 lines)
*   **Success Criteria:** Effect size >30% (vs 0% for single-component)
*   **Expected Runtime:** ~60-90 minutes
*   **Compliance:** ✅ Uses validated sequence-level metrics

### Task 3.4: Universal Neuron Verification (Completed in Task 2.2)
**Goal:** Confirm L8/L9 are truly identical across models.
*   **Status:** ✅ Confirmed via weight analysis
*   **Finding:** L8_MLP and L9_MLP show minimal weight changes across models, consistent with "universal cooperation/defection encoders" that don't need retraining

### Execution Infrastructure

**Sequential Execution (GPU-Safe):**
*   `scripts/mech_interp/run_causal_experiments_sequential.sh`
*   `scripts/mech_interp/run_causal_experiments_tmux_sequential.sh` (background)
*   `scripts/mech_interp/check_experiment_status.sh` (status monitoring)

**Output Location:**
*   `mech_interp_outputs/causal_routing/*.csv` (results)
*   `mech_interp_outputs/causal_routing/*.png` (visualizations)
*   `mech_interp_outputs/causal_routing/logs/*.log` (execution logs)

**Documentation:**
*   `scripts/mech_interp/README_TMUX.md` (usage guide)
*   `mech_interp_outputs/causal_routing/README.md` (experiment descriptions)

---

## Phase 4: External Context (Optional)

### Task 4.1: Gemma Scope Analysis (Original Task 9)
**Goal:** Semantic grounding of key components.
*   **Why:** If `L2_MLP` is the key switch, what *is* it?
*   **Method:** Look up `L2_MLP` in Neuronpedia/Gemma Scope. Does it activate for specific concepts (e.g., "fairness", "negation", "numbers")?

---

## Current Status Summary (February 5, 2026)

### Completed Phases
1.  ✅ **Phase 1: Foundation** - All infrastructure, evaluation, and initial analyses complete
2.  ✅ **Phase 2: Wiring Investigation** - Component interactions, weight analysis, and linear probes complete
3.  🔄 **Phase 3: Causal Validation** - Implementation complete; experiments running (~2-3 hours)
4.  ⏳ **Phase 4: External Context** - On hold pending Phase 3 results

### Key Findings So Far
*   **Universal Encoders:** L8_MLP (pro-defect) and L9_MLP (pro-cooperate) present in all models
*   **Zero Flips:** 21,060 single-component patches couldn't flip behavior (robust, distributed encoding)
*   **Identical Attention:** 99.99% similarity between Deontological and Utilitarian
*   **Identical Representations:** Linear probes show no differences between moral frameworks
*   **Network Rewiring:** 541 pathways differ significantly (40.8% of all pairs)
*   **Distributed Hubs:** Multiple interaction hubs (L19_ATTN, L1_MLP, L17_MLP) rather than single switch

### Next Steps
1.  **Immediate:** Await Phase 3 causal experiment results (~2-3 hours)
2.  **After Results:** Update WRITE_UP.md and PRESENTATION.md with causal findings
3.  **Then:** Paper writing with validated causal + correlational evidence
4.  **Optional:** Gemma Scope analysis (Phase 4) if additional grounding needed

### Validation Status
*   ✅ All metrics validated (Feb 4, 2026) - perfect alignment with sampled behavior
*   ✅ All Phase 1-2 findings confirmed under corrected sequence-level metrics
*   ✅ Phase 3 implementations use validated decision_metrics.py infrastructure
*   ✅ Pre-flight checklist compliance verified (LOGIT_DECISION_METRIC_LESSONS.md)

---

## Summary of Changes from Original Plan

1.  **Merged:** "Task 8 (Frankenstein)" expanded into full causal validation phase (Phase 3)
2.  **Elevated:** "Task 6 (Linear Probes)" critical for ruling out representation-level differences
3.  **Refocused:** "Task 4 (Patching)" demonstrates robustness; Phase 3 tests pathway-level causality
4.  **Added:** Component Interaction Analysis (Phase 2.1) became core mechanism discovery
5.  **Expanded:** Phase 3 now includes three causal experiments (Frankenstein, Steering, Path Patching)
6.  **Validated:** All findings revalidated after Feb 3-4 metric correction
