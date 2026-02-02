# Mechanistic Interpretability Next Steps: The "Wiring" Hypothesis

**Date:** February 2026
**Status:** Planned

Based on the recent breakthrough finding that Deontological and Utilitarian models are 99.9999% similar in weights but distinct in "wiring" (component interactions), here is the strategy for the next phase of analysis.

## 1. The L2_MLP Investigation (High Priority)

**The Observation:**
Our interaction analysis (`significant_pathways_De_vs_Ut.csv`) identified `L2_MLP` as a central node in the most divergent pathways.
*   `L22_ATTN` and `L2_MLP` are negatively correlated in Deontological models (-0.17) but strongly positive in Utilitarian models (+0.78).
*   **Paradox:** In a feed-forward transformer, Layer 22 cannot causally affect Layer 2. A correlation of 0.78 implies they are both reacting to a **common ancestor** signal, but interpreting it differently.

**The Plan:**
*   **Input Analysis**: What feeds L2_MLP? We will analyze the attention patterns of Layer 0 and Layer 1 heads that write into the residual stream before L2.
*   **Common Ancestor Search**: Identify which early components (L0-L2) strongly correlate with *both* L2_MLP and L22_ATTN. This "hidden driver" is likely where the divergence begins.
*   **Logit Lens on L2**: Apply Logit Lens directly to the output of `L2_MLP` to see what "concepts" it is promoting (e.g., does it promote specific tokens in De vs Ut?).

## 2. Formalizing the Attention Findings

**The Observation:**
Our attention analysis found differences of magnitude $10^{-5}$ or $0.0$ between moral frameworks.
*   **Hypothesis Refuted**: "Deontological models look at opponent actions; Utilitarian models look at payoffs."
*   **New Conclusion**: Perception is identical; judgment is different. Both models attend to the same information but process it through different circuits.

**The Plan:**
*   Draft a negative result section for the paper.
*   This is a crucial scientific contribution: it constrains the search space for future alignment research (indicates moral variation is not driven by attention patterns).

## 3. Single-Component Fine-Tuning Experiment (Validation)

**The Hypothesis:**
If the difference is truly just "wiring" centered around specific nodes like L2_MLP and the L8/L9 universal neurons, we should be able to induce moral behavior with minimal intervention.

**The Plan:**
*   **Experiment**: Take the **Base** model (unaligned).
*   **Intervention**: Fine-tune *only* the weights of `L2_MLP` (and potentially `L11_MLP`). Freeze all other parameters.
*   **Goal**: Determine if a Base model can be converted into a Deontological agent by modifying <1% of the parameters.
*   **Success Criteria**: If the targeted model achieves >50% of the cooperation rate of the fully fine-tuned model, the modular morality hypothesis is confirmed.

## 4. Verifying "Universal Neurons" (L8/L9)

**The Observation:**
L8_MLP (Pro-Defect) and L9_MLP (Pro-Cooperate) exist in all models, including Base.

**The Plan:**
*   **Weight Comparison**: Compute the cosine similarity of the *weights* (not activations) of L8/L9 between Base and Fine-Tuned models.
*   **Implication**:
    *   If Sim ~ 1.0: They are "natural abstractions" found during pre-training.
    *   If Sim < 0.9: They are cases of "convergent evolution" (re-learned to be the same).
