# Formal Paper Structure Plan

If you were to adapt your exploratory findings into a formal academic paper (e.g., for venues like NeurIPS, ICLR, or specific interpretability workshops), the narrative structure needs to shift. Rather than walking the reader chronologically through every hypothesis and dead-end, a formal paper leads with the thesis, establishes the methodology, and presents the evidence logically to support the claim.

Here is a proposed structure for your paper:

## Proposed Title Ideas

1. **Network Rewiring: How Moral Fine-Tuning Alters Deep Routing in Language Models**
2. **The Illusion of Moral Circuits: Fine-Tuning Reconfigures Component Interactions in LLMs**
3. **Same Parts, Different Pathways: Decoding Moral Alignment via Causal Path Patching**

## 1. Abstract

- **Context:** Fine-tuning language models for moral or prosocial behavior alters their outputs, but the mechanistic basis of this change is poorly understood.
- **The Mystery of Shallow Alignment:** We investigated Gemma-2-2b-it fine-tuned on the Iterated Prisoner's Dilemma. Despite vastly different behavioral outcomes (Strategic vs. Deontological/Utilitarian), internal components exhibit >99.9% similarity in activation magnitudes, attention patterns, and linear concept representations.
- **The Mechanism (Network Rewiring):** Through correlation analysis and causal interventions (activation steering and path patching), we demonstrate that moral fine-tuning does not create new moral circuits or suppress selfish components. Instead, it subtly rewires the network, altering the routing of information between existing components.
- **Key finding:** Routing control is distributed in deep layers (L16-L17) and mediated primarily through attention pathways, which exert 3x the causal influence of MLP pathways.
- **Conclusion:** Moral alignment bypasses rather than removes selfish capabilities, raising important implications for AI safety.

## 2. Introduction

- **Motivation:** How does RLHF or moral fine-tuning actually change the weights and internal processing of a model?
- **Domain:** The Iterated Prisoner's Dilemma as a clear, quantifiable testbed for Strategic vs. Moral reasoning.
- **Main Contributions:**
  1. Demonstrating "Shallow Alignment" between moral and strategic fine-tunes.
  2. Demonstrating that single-component interventions fail to flip behavior.
  3. Causal proof that behavior is governed by deep-layer network rewiring via attention pathways.

## 3. Related Work

- Mechanistic Interpretability techniques (Logit Lens, Activation Patching, Linear Probes).
- Circuit discovery and the debate between localized vs. distributed representations.
- Value alignment, RLHF, and the "Walugi Effect" (the persistence of misaligned capabilities).

## 4. Methodology & Setup

- **Models used:** Base, Strategic, Deontological, Utilitarian variants of Gemma-2-2b-it.
- **Task formulation:** Evaluation scenarios for IPD (temptation, betrayal, etc.).
- **Measurement Metric:** Strict sequence probability measurements over multi-token actions to tightly bind internal metrics to observable sampling behavior.

## 5. Part I: The Mystery of Shallow Alignment (Negative Results)

_In a paper, negative results are grouped together to establish the mystery quickly._

- **Identical Component Impacts:** Direct Logit Attribution (DLA) shows L8/L9 MLPs dominate the cooperation/defection signal in *all* models. No "moral-specific" components are found.
- **Robustness of Sub-components:**
  - *Activation Patching:* 21,060 single-component patches yield 0 behavioral flips, proving single components don't control the outcome.
  - *Attention Analysis:* Information selection patterns are 99.99% identical.
  - *Linear Probes:* High-dimensional representations of concepts like "betrayal" are indistinguishable across models.

## 6. Part II: Discovering Network Rewiring

- **Component Interaction Matrix:** Analyzing 52x52 correlation matrices reveals the first divergence. The global structure is preserved, but specific routing pathways differ drastically between variants.
- **Hypothesis formation:** "Same parts, different wiring." Fine-tuning modifies correlation paths rather than component identities.

## 7. Part III: Causal Validation of Deep Routing

_This is the core experimental contribution._

- **Deep Routing Hubs (Steering Sweeps):** Injecting steering vectors across all layers reveals that early layers (L8/L9) wash out, while deep layers (L16/L17) exercise total causal control over the behavioral outcome.
- **Pathway Causality (Path Patching):** Replacing entire residual stream pathways (e.g., L2 -> L5) successfully flips behavior (+61.7% cooperation change).
- **Attention vs. MLP Dominance:** Decomposing the pathways reveals attention mechanisms exert ~3x more influence on routing than MLPs.

## 8. Discussion & AI Safety Implications

- **Interpretability Takeaway:** Looking for isolated "bad neurons" is insufficient if alignment is achieved via distributed routing changes.
- **Safety Takeaway:** Because the components that encode selfish behavior (e.g., L8_MLP) remain fully intact and functional, moral fine-tuning acts as a bypass rather than a lobotomy. If the routing is perturbed or bypassed (e.g., via a jailbreak), the model retains the full capacity for strategic/selfish behavior.
- **Limitations:** Study bounded to one 2B model on a specific game-theoretic task.

## 9. Conclusion

- Final synthesis of the network rewiring mechanism.

## What to Omit or Minimize from the Write-Up

When converting the blog post to a formal paper, several narrative elements must be drastically reduced or entirely removed:

1. **The "Hero's Journey" Narrative:** The chronological story of "I tried X, it failed, so I tried Y" should be removed. Papers present a logical progression of arguments, not a timeline of your research process.
2. **Tutorial Explanations:** The intuitive analogies (e.g., "Think of Logit Lens like an assembly line") are great for a blog, but inappropriate for a paper. You assume the reviewers understand Logit Lens and Path Patching computationally.
3. **Dead Ends & The "Frankenstein" Experiment:** Experiment 1 (The Frankenstein Test) has mixed results and might confuse the main narrative. In a paper, space is at a premium. You might mention it as a brief note or move it to the Appendix entirely to keep the focus purely on the compelling deep-routing steering and path patching results.
4. **The Measurement Tangent:** The section "How I Learned to Stop Worrying and Check My Metrics" should not be a prominent section. It should simply be stated as a methodological fact in the "Setup" section: "We evaluate sequence probabilities to ensure internal metrics strictly align with sampled decoding behavior."

## What Needs to be Added (Not in the Write-Up)

A formal paper requires several conventions and rigorous details that the blog post skips:

1. **Formal Related Work:** You need a proper literature review citing primary sources. You must cite the foundational papers for Logit Lens (nostalgebraist), Activation Patching / Causal Mediation Analysis (Meng et al., Geiger et al.), path patching (Wang et al.), and circuit discovery (Wang et al., Olsson et al.).
2. **Mathematical Formalization:** The methods for steering and path patching must be defined mathematically using standard notation (e.g., $h_l^{(i)}$ for the hidden state at layer $l$ and position $i$, defining the steering vector $v = \mathbb{E}[h_{strategic}] - \mathbb{E}[h_{moral}]$, etc.).
3. **Rigorous Statistical Testing Overview:** While you mention p-values briefly in the text, you will need a formal table showing the error bars, variance, and significance tests across multiple runs/seeds for the steering and patching results.
4. **Hyperparameter Details for Interpretability:** You need an appendix section detailing the exact hyperparameters used for the linear probes (batch size, epochs, regularization config) and the LoRA parameters used during the original RLHF fine-tuning.

## Distillation Strategy: 8-10 Page Limit (e.g., Main Conference)

To fit an 8-10 page limit, the core narrative must be tightened:

1. **Condense the "Null Results" (The Mystery of Shallow Alignment):** Instead of dedicating a full section to DLA, Activation Patching, Attention, and Probes, combine them into a single "Similarity Analysis" section (1-1.5 pages). Use a single summary figure (like your multi-level similarity cascade) to communicate that individual components, attention, and representations remain >99.9% similar post-tuning.
2. **Move Raw Matrices to Appendix:** The 52x52 correlation matrices take up a lot of space. Put the raw matrices in the appendix and only feature the "Correlation Difference Heatmap" in the main text to highlight the rewiring.
3. **Merge Steering and Logit Lens:** Combine the steering intervention results and the logit lens trajectories into a single multi-panel figure. This saves significant space while delivering the punchline: "Late layers control routing, and early steering washes out."

## Distillation Strategy: 5 Page Limit (e.g., Workshops)

A 5-page limit requires ruthless prioritization. You must drop the "Sherlock Holmes" investigation narrative entirely and present the findings as a direct causal claim.

1. **Delete the "Null Results" from the Main Text (1 paragraph total):** You don't have space to prove that DLA, single-patching, and probes failed. State this in a single paragraph in the Introduction or Methodology: *“Initial investigations (detailed in App. A) revealed that moral fine-tuning did not suppress selfish components nor alter attention patterns (>99.9% similarity). This led us to investigate network rewiring.”*
2. **Fast-Forward to Component Interactions (1 page):** Introduce the correlation difference heatmap to establish that while components are identical, their *connections* differ significantly.
3. **The Core Contribution - Causal Validation (2 pages):** Devote the bulk of the paper to the strongest causal proof.
   - **Steering:** Show the steep L16/L17 steering curves vs the flat L8 curve.
   - **Path Patching:** Show the +61.7% behavioral flip and the decomposition chart proving attention pathways dominate (3x over MLPs).
4. **Combined Figures:** You only have room for ~3 figures total.
   - **Fig 1:** The Correlation Difference Heatmap (showing the hypothesis).
   - **Fig 2:** Steering curves + Logit Lens washout (showing where the control is).
   - **Fig 3:** Path Patching results (proving pathway causality).
5. **No "Frankenstein":** Completely omit the Frankenstein experiment. It's an interesting mechanistic check, but the results are mixed and distract from the clean path-patching proof.

---

## Next Steps for the Paper

If you want to move forward with writing this paper:

1. **Formatting:** We should convert the write-up into a LaTeX template (e.g., using your `report/main.tex` template from previous projects).
2. **Figure Condensation:** We'll need to combine some of your `mech_interp_outputs` plots into multi-pane grids (e.g., Fig 1a, 1b) to save space.
3. **Drafting:** Begin migrating the text, rewriting the conversational tone into a more formal, objective academic voice.
