# Mechanistic Interpretability of Moral Fine-Tuning

## Summary

The primary objective was to determine how moral fine-tuning alters model behavior—specifically, whether it creates new "moral" components, suppresses "selfish" ones, or shifts attention patterns.

The data indicates the primary mechanism is **network rewiring**. The analysis shows no significant formation of new components or changes in attention patterns. Instead, the models appear to repurpose existing components by altering how information is routed between them.

## Key Findings

I analyzed the models across four levels:

1.  **Components**: The models are effectively identical in terms of weights (99.9999% similarity). Fine-tuning did not significantly alter individual component weights.
2.  **Attention**: Attention patterns are nearly indistinguishable (99.99% similarity). Both Deontological and Utilitarian models attend to the same input information.
3.  **Behaviors**: despite the structural similarity, the behavioral outputs differ significantly. There are no direct coding flips, but the outputs are consistently asymmetric.
4.  **Interactions**: This is the primary differentiator. Component correlations show only ~20% overlap between models. The connections (wiring) between components have shifted.

### Specific Mechanism Example

A significant shift appears in `L2_MLP`, which functions as a routing node:
*   In the **Deontological** model, it has a positive connection to `L9_MLP` (+0.27), promoting cooperation.
*   In the **Utilitarian** model, the same connection is negative (-0.49), inhibiting cooperation.

## Addressing the Research Questions

### Research Questions

*   **RQ1**: How are "selfish" attention heads suppressed during moral fine-tuning?
*   **RQ2**: Do Deontological vs. Utilitarian agents develop distinct circuit structures?
*   **RQ3**: Can we identify which parts of the model to fine-tune specifically for more targeted training?

### Findings

*   **RQ1 (Suppression)**: "Selfish" heads are not suppressed. They remain active but are bypassed or routed differently.
*   **RQ2 (Distinct structures?)**: The models possess distinct structures, but these differences lie in the wiring (specifically 29 changed pathways) rather than the nodes themselves.
*   **RQ3 (Fine-tuning targets)**: It appears feasible to fine-tune specific pathways (e.g., L2→L9 or the L6 hub) rather than entire layers.

## Implications

These findings suggest a shift in how we interpret model differences:

*   **Interpretability**: Analyzing components in isolation is insufficient. The functional difference lies in the structure/wiring rather than the individual parts.
*   **Safety**: Models can be nearly identical at the component level yet behave differently. Standard safety evaluations focusing on weights or activation norms may overlook these structural shifts.
*   **Efficiency**: The existence of "universal" moral neurons (L8/L9) suggests we could potentially freeze ~50-70% of the model (early layers + universal MLPs) and fine-tune only the routing layers (L11-L23).
*   **Robustness**: Testing with over 7,000 distinct patches failed to flip a single decision, suggesting that moral fine-tuning creates a distributed and robust defense against manipulation.

## Mini Research Log

*   **Phase 1 (Setup)**: I replicated the training pipeline for the 4 main model variations (Game Payoffs, Deontological, Utilitarian, Hybrid) and generated 15 prompts across 5 standard scenarios to establish a baseline.
*   **Phase 2 (Initial DLA Analysis)**: I ran Direct Logit Attribution to identify suppressed "selfish" neurons. The analysis revealed L8/L9 MLPs as strong "cooperation components" present in all models, including the base model.
*   **Phase 3 (Suppression Check)**: I investigated whether "selfish" components were being suppressed. The data showed these components remain as active in moral models as in strategic ones, but their influence is outweighed or rerouted.
*   **Phase 4 (Model Comparison)**: Comparison of Deontological and Utilitarian models showed them to be 99.9999% similar by cosine distance. Consistency across multiple metrics confirmed this was not an error.
*   **Phase 5 (Interaction Analysis)**: Bidirectional patching revealed that while components are identical, 78% respond differently to context swaps. Subsequent wiring analysis identified specific pathways (such as L2→L9) that reversed polarity.

## Next Steps / In Progress

*   **L2_MLP Investigation**: I am analyzing `L2_MLP` to understand its role as a central router in divergent pathways.
*   **Attention Analysis**: I am documenting the finding that Deontological and Utilitarian models possess nearly identical attention patterns (diff < 10^-5), indicating that differences arise from processing rather than attention.
*   **Single-Component Fine-Tuning**: I plan to test the wiring hypothesis by fine-tuning only the `L2_MLP` component in the Base model to determine if this single modification can induce the target behavior.
