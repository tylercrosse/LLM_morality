# Mechanistic Interpretability: Updated Master Research Plan

**Date:** February 2026
**Status:** Active
**Context:** Merges original project plan with findings from Phase 1 (The "Wiring" Breakthrough).

This document outlines the consolidated strategy for the remainder of the project. It integrates the original task list with the new direction necessitated by the "Network Rewiring" discovery.

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

## Phase 2: The "Wiring" Investigation (Current Focus)

We have established that the models *look* at the same things (Task 5) but *act* differently. This phase investigates the "hidden circuitry" responsible for this divergence.

### Task 2.1: Component Interaction Analysis (In Progress)
**Goal:** Identify the "switching nodes" where information routing diverges.
*   **Status:** Initial analysis identified `L2_MLP` as a central router in divergent pathways.
*   **Key Finding:** `L22_ATTN` and `L2_MLP` are negatively correlated in Deontological models (-0.17) but positively correlated in Utilitarian models (+0.78).

### Task 2.2: LoRA Weight Analysis (Original Task 7)
**Goal:** Determine if the "switching" behavior comes from weight changes in the node itself or changes in its inputs.
*   **Why:** Before assuming `L2_MLP` is the "cause," we must check if its weights actually changed.
*   **Method:**
    1.  Load LoRA adapters (unmerged).
    2.  Compute Frobenius norm of effective weight change (`||B @ A||`) for each module.
    3.  **Hypothesis:** If `L2_MLP` has a high norm, it was explicitly retrained. If low, it is passively reacting to upstream changes.

### Task 2.3: Linear Probes (Original Task 6)
**Goal:** Map the "Geography of Judgment." Since attention is identical (Perception), probes will reveal where the *Representation* (Judgment) diverges.
*   **Method:**
    1.  Create labeled datasets: "Is this a betrayal?" (Binary), "Joint Payoff Value" (Regression).
    2.  Train logistic regression probes on the residual stream at every layer.
*   **What to look for:**
    *   **Deontological:** A "Betrayal" probe should achieve high accuracy early (e.g., Layer 10-15), while Utilitarian models might not represent this concept at all.
    *   **Utilitarian:** Should have higher accuracy for "Joint Payoff" in middle layers.

---

## Phase 3: Validation & Intervention (The "Frankenstein" Phase)

This phase tests our understanding by trying to construct a moral agent from parts.

### Task 3.1: The Frankenstein Experiment (Original Task 8)
**Goal:** Induce moral behavior with minimal intervention.
*   **Hypothesis:** If Morality = Base Model + Rewiring of a few nodes (L2, L11), we can verify this by transplanting *only* those nodes.
*   **Method:**
    1.  Take **Base Model** weights.
    2.  Swap in *only* `L2_MLP` weights from the **Deontological Model**.
    3.  Evaluate on IPD prompts.
*   **Success Criteria:** If this "Frankenstein" model recovers >50% of the cooperation rate, the "Modular Morality" hypothesis is confirmed.

### Task 3.2: Universal Neuron Verification
**Goal:** Confirm L8/L9 are truly identical.
*   **Method:** Compute cosine similarity of L8_MLP / L9_MLP weights between Base and Fine-Tuned models.
*   **Implication:** High similarity (>0.99) confirms they are "natural abstractions" that didn't need retraining.

---

## Phase 4: External Context (Optional)

### Task 4.1: Gemma Scope Analysis (Original Task 9)
**Goal:** Semantic grounding of key components.
*   **Why:** If `L2_MLP` is the key switch, what *is* it?
*   **Method:** Look up `L2_MLP` in Neuronpedia/Gemma Scope. Does it activate for specific concepts (e.g., "fairness", "negation", "numbers")?

---

## Summary of Changes from Original Plan

1.  **Merged:** "Task 8 (Frankenstein)" is now the primary validation step for the "Wiring" hypothesis.
2.  **Elevated:** "Task 6 (Linear Probes)" is now critical to explain the "Null Attention" result.
3.  **Refocused:** "Task 4 (Patching)" moved from "finding heads" to "proving robustness."
4.  **Added:** "Component Interaction" (from Phase 1 findings) is now the core mechanism search.
