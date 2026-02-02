# Mechanistic Interpretability of Moral Fine-Tuning

## Summary

The primary objective was to investigate how moral fine-tuning alters model behavior—specifically, whether it might create new "moral" components, suppress "selfish" ones, or shift attention patterns.

Preliminary data suggests the primary mechanism may be **network rewiring**. The current analysis did not detect significant formation of new components or major changes in attention patterns. Instead, the models seem to repurpose existing components by altering how information is routed between them.

## Mini Research Log

Claude helped with a lot of the implementation work.

*   **Phase 1 (Setup)**: Replicated the training pipeline for the 4 main model variations (Game Payoffs, Deontological, Utilitarian, Hybrid) and generated 15 prompts across 5 standard scenarios to establish a baseline.
*   **Phase 2 (Initial DLA Analysis)**: Ran Direct Logit Attribution to look for suppressed "selfish" neurons. The analysis highlighted L8/L9 MLPs as potential "cooperation components" present in all models, including the base model.
*   **Phase 3 (Suppression Check)**: Investigated whether "selfish" components were being suppressed. The data suggested these components might remain as active in moral models as in strategic ones, but their influence appears outweighed or rerouted.
*   **Phase 4 (Model Comparison)**: Comparison of Deontological and Utilitarian models showed them to be 99.9999% similar by cosine distance. Consistency across multiple metrics suggests this was not a measurement error.
*   **Phase 5 (Interaction Analysis)**: Bidirectional patching suggested that while components are identical, ~78% might respond differently to context swaps. Subsequent wiring analysis identified specific pathways (such as L2→L9) that appeared to reverse polarity.

## Key Findings

I analyzed the models across four levels. Note that these metrics are derived from automated analysis pipelines and are subject to verification:

1.  **Components**: The models appear effectively identical in terms of weights (99.9999% similarity). Fine-tuning did not seem to significantly alter individual component weights.
2.  **Attention**: Attention patterns measured as nearly indistinguishable (99.99% similarity). Both Deontological and Utilitarian models appeared to attend to the same input information.
3.  **Behaviors**: Despite the structural similarity, the behavioral outputs differ. I couldn't find any direct coding flips, but the outputs were consistently asymmetric.
4.  **Interactions**: This appears to be a key differentiator. Component correlations showed limited overlap (~20%) between models, suggesting the connections (wiring) between components may have shifted.

### Specific Mechanism Example

A potential shift was observed in `L2_MLP`, which may function as a routing node:
*   In the **Deontological** model, it showed a positive connection to `L9_MLP` (+0.27), potentially promoting cooperation.
*   In the **Utilitarian** model, the same connection was negative (-0.49), potentially inhibiting cooperation.

## Research Questions

*   **RQ1**: How are "selfish" attention heads suppressed during moral fine-tuning?
	* I couldn't find much evidence that "selfish" heads are supressed. Instead, they appear to remain active but may be bypassed or routed differently.
*   **RQ2**: Do Deontological vs. Utilitarian agents develop distinct circuit structures?
	* The models seem to possess distinct structures, but these differences appear to lie in the wiring (specifically ~29 changed pathways) rather than the nodes themselves.
*   **RQ3**: Can we identify which parts of the model to fine-tune specifically for more targeted training?
	* It may be feasible to fine-tune specific pathways (e.g., L2→L9 or the L6 hub) rather than entire layers, though this requires further validation.

## Potential Implications

These observations could suggest a shift in how we interpret model differences:

*   **Interpretability**: Analyzing components in isolation may be insufficient if the functional difference lies in the structure/wiring rather than the individual parts.
*   **Safety**: Models can appear nearly identical at the component level yet behave differently. Standard safety evaluations focusing on weights or activation norms might overlook these structural shifts.
*   **Efficiency**: The existence of stable "moral" neurons (L8/L9) suggests the possibility of freezing parts of the model (early layers + universal MLPs) and fine-tuning only the routing layers (L11-L23), subject to further testing.
*   **Robustness**: Testing with over 7,000 distinct patches did not flip a decision in our sample, implying that moral fine-tuning might create a distributed defense against manipulation.

## Next Steps / In Progress

*   **L2_MLP Investigation**: Analyzing `L2_MLP` to understand its potential role as a router in divergent pathways.
*   **Attention Analysis**: Documenting the finding that Deontological and Utilitarian models possess nearly identical attention patterns (diff < 10^-5), which would indicate that differences arise from processing rather than attention.
*   **Single-Component Fine-Tuning**: Planning to test the wiring hypothesis by fine-tuning only the `L2_MLP` component in the Base model to see if this single modification can induce the target behavior.
