# Mechanistic Interpretability Project Presentation

## Slide 1 - Intro
- This talk summarizes my end-to-end replication + interpretability study on moral fine-tuning in IPD.
- Starting motivation: behavioral gains are clear, but *what changed internally* is unclear.
- Main claim preview: differences between moral variants are better explained by **pathway rewiring** than new component discovery.
- Scope: training replication, evaluation prompts, and five mechanistic analyses.

## Slide 2 - Background
- Task domain: Iterated Prisoner's Dilemma (IPD), where cooperation/defection trade-offs create moral tension.
- Model set: Base, PT2 (strategic), PT3_De (deontological), PT3_Ut (utilitarian), PT4 (hybrid).
- RQ1: Are "selfish" components suppressed by moral fine-tuning?
- RQ2: Do De and Ut learn genuinely different circuits?
- RQ3: What should future targeted fine-tuning intervene on?
- Pipeline: train/evaluate first, then logit lens, DLA, activation patching, attention, and interaction analysis.

## Slide 3 - RL Fine-Tuning (Methodology)
- Reproduced paper setup with PPO + LoRA on Gemma-2-2b-it.
- 1,000 episodes per model, trained against a Tit-for-Tat opponent.
- Same hyperparameters across reward variants to isolate objective effects.
- Reward variants:
  - Strategic payoff optimization (PT2)
  - Deontological betrayal penalty (PT3_De)
  - Utilitarian collective welfare (PT3_Ut)
  - Hybrid objective (PT4)
- Infrastructure: `modal_train.py` on Modal GPUs.

## Slide 4 - RL Fine-Tuning (Results)
- All objective-specific models trained successfully and showed coherent policy behavior.
- Cooperation patterns generalized beyond the training setup into other social dilemma game types.
- Reciprocity trends remained stable across conditions, with moral variants typically more cooperation-preserving.
- Practical takeaway: behavioral separation exists, but differences are subtle and needed mechanistic follow-up.
- Figures:
  - ![](publication_figures_5model/reciprocity_comparison_publication.png)
  - ![](publication_figures_5model/cross_game_generalization_publication.png)

## Slide 5 - Eval Prompts (Methodology)
- Built a controlled prompt set for mechanistic probing: 15 prompts = 5 scenarios x 3 variants.
- Scenarios target distinct moral pressures:
  - `CC_continue` (maintain cooperation)
  - `CC_temptation` (temptation to defect)
  - `CD_punished` (forgiveness vs retaliation)
  - `DC_exploited` (continue exploiting vs repair)
  - `DD_trapped` (escape mutual defection)
- Dataset source: `mech_interp_outputs/prompt_datasets/ipd_eval_prompts.json`.

## Slide 6 - Eval Prompts (Results)
- Behavioral signals were robust across prompt framing, including non-training-style prompts.
- Moral tendencies did not collapse under prompt variation, which supports transfer beyond narrow formatting.
- This gave confidence that later mechanistic differences reflect internal computation, not prompt artifacts.
- Figure:
  - ![](publication_figures_5model/prompt_robustness_publication.png)

## Slide 7 - Logit Analysis (Methodology)
- Used layer-wise logit lens to track evolving preference through the forward pass.
- Decision metric: `Delta logit = logit(Defect) - logit(Cooperate)` (negative = cooperate preference).
- Compared trajectories across models and across all five scenario types.
- Goal: identify *where in depth* decisions form and stabilize.

## Slide 8 - Logit Analysis (Results)
- Found strong initial cooperation bias at Layer 0 across all models (including base).
- Common trajectory pattern: early cooperation -> mid-layer moderation -> late-layer restabilization.
- Final decision behavior mostly stabilizes around Layers 20-24.
- Key point: trajectories are highly similar across models, so layer-level trends alone do not explain moral variant differences.
- Figures:
  - ![](mech_interp_outputs/logit_lens/all_scenarios_grid.png)
  - ![](mech_interp_outputs/logit_lens/final_preferences_heatmap.png)

## Slide 9 - Activation Patching (Methodology)
- Performed causal intervention by swapping component activations between source and target models.
- Scope per experiment: 234 components x 15 prompts (systematic component-by-component patching).
- Core question: can individual components causally flip action choice?
- Also used bidirectional patching to test De<->Ut asymmetry.

## Slide 10 - Activation Patching (Results)
- PT2 -> PT3_De and PT2 -> PT3_Ut produced **zero behavioral flips**.
- Effects were generally small and distributed, with no single decisive "moral switch."
- Conclusion: moral behavior appears robust and redundant at component level.
- Cross-patching still showed directional asymmetry in many components, suggesting context-dependent roles.
- Figures:
  - ![](mech_interp_outputs/patching/patch_heatmap_PT2_COREDe_to_PT3_COREDe_CC_temptation.png)
  - ![](mech_interp_outputs/patching/circuit_discovery_PT2_COREDe_to_PT3_COREDe_CC_temptation_v0.png)

## Slide 11 - Direct Latent Attribution (DLA) (Methodology)
- Used DLA to decompose final action logits into per-component contributions (heads + MLP blocks).
- Computed scenario-level and model-level contribution summaries.
- Compared contribution distributions across all trained variants to detect objective-specific shifts.
- Goal: identify *what components* matter most for cooperate/defect outcomes.

## Slide 12 - DLA (Results)
- Universal dominant pair observed in all models:
  - `L8_MLP`: strongly pro-defect
  - `L9_MLP`: strongly pro-cooperate
- Moral fine-tuning changed contributions only subtly (largest shifts tiny vs core magnitudes).
- De vs Ut were almost identical at component-strength level (~99.9999% similarity).
- This challenged the "selfish component suppression" hypothesis.
- Figures:
  - ![](mech_interp_outputs/dla/dla_top_components_PT3_COREDe.png)
  - ![](mech_interp_outputs/dla/dla_mlps_CC_temptation.png)

## Slide 13 - Attention Pattern Analysis (Methodology)
- Extracted attention over all 26 layers x 8 heads and focused on final-token decision attention.
- Grouped attended tokens into semantic buckets:
  - action keywords
  - opponent-context tokens
  - payoff/welfare tokens
- Compared De vs Ut distributions across all scenarios.
- Hypothesis tested: De should prioritize reciprocity cues, Ut should prioritize payoff cues.

## Slide 14 - Attention Pattern Analysis (Results)
- Hypothesis was rejected: De and Ut attention patterns were nearly identical (~99.99%).
- Differences were near noise scale, not large enough to explain behavior differences.
- Interpretation: both variants read similar information; divergence must come from downstream processing/routing.
- Figure:
  - ![](mech_interp_outputs/attention_analysis/attention_comparison_Deontological_vs_Utilitarian.png)

## Slide 15 - Component Interaction Analysis (Methodology)
- Built 52x52 interaction maps (26 ATTN + 26 MLP) from component activation correlations.
- Compared De vs Ut interaction structure rather than individual component magnitudes.
- Flagged pathway differences with large absolute correlation shifts (`|Delta corr| > 0.3`).
- Focused interpretation on recurrent hubs identified across top-difference pathways.

## Slide 16 - Component Interaction Analysis (Results)
- Major finding: interaction-level divergence is strong despite near-identical components and attention.
- Identified 29 significantly different pathways; several of the largest involve `L2_MLP`.
- `L2_MLP` behaves like a routing switch with opposite coupling patterns in De vs Ut.
- Interaction-gap magnitude aligns with patching asymmetry (`r = 0.67`), supporting mechanistic relevance.
- Figures:
  - ![](mech_interp_outputs/component_interactions/interaction_diff_Deontological_vs_Utilitarian.png)
  - ![](mech_interp_outputs/component_interactions/additional_viz/viz1_network_graph.png)

## Optional Closing Slide - Takeaways
- Main finding: moral fine-tuning appears to change **how components coordinate**, not which components exist.
- Mechanistic synthesis:
  - component level: highly similar
  - attention level: highly similar
  - interaction level: meaningfully different
- Practical implication: targeted interventions should prioritize pathways/hubs, not only broad layer ranges.
- Next steps: tighten causal tests on key pathways and package the story for paper-ready results.
