# Editorial Review: WRITE_UP.md (LessWrong Post)

Reviewer: Claude | Date: 2026-02-24

## 1. Narrative Tightness & Redundancy

**The post's biggest issue.** Key findings are restated 2-3 times. Specific cuts:

| Finding | Locations | Recommendation |
|---------|-----------|----------------|
| Zero flips (21,060 patches) | Lines 307-311, 335, 585 | Keep line 307-311. Cut 335 or make it a parenthetical. Remove line 585 restatement. |
| 99.99% attention similarity | Lines 345, 374, 787 | Keep line 345. Reference "as shown above" elsewhere. |
| r=0.67 correlation | Lines 315, 447, 449 | Keep 315 (forward-ref) and 447. Cut 449 (near-duplicate). |
| "Same parts, different wiring" | Lines 412, 443, 451, 611, 745 | Keep 2 instances max. |
| Washout Effect | Lines 579-583, 585-587 | Near-identical paragraphs. Merge into one. |

**End-of-section interpretation paragraphs** (lines 191-193, 246-249, 316-317, 448-450, 606-607, 609-613) often restate what was just demonstrated. LW readers will have followed. Cut most or compress to single-sentence transitions.

**Estimated cuttable**: 20-25% of body text without losing substance.

## 2. Audience Calibration for LessWrong

### Too basic (compress or cut):
- **Lines 7-17**: IPD explanation with payoff matrix. LW readers know IPD. Two sentences + matrix suffices.
- **Lines 19-23**: Defining strategic/deontological/utilitarian as ethical frameworks. LW knows these. Just describe the reward signals.
- **Line 81**: Explaining PPO ("a reinforcement learning algorithm that gradually adjusts..."). Too basic.
- **Line 81**: Explaining LoRA. One clause is enough for this audience.
- **Lines 147-149**: Explaining transformers have layers. Compress.
- **Lines 203-208**: Explaining that models have attention heads and MLPs. Audience knows this.

### Needs MORE precision:
- **Component interaction analysis (Part 4)**: Uses "correlation patterns" but doesn't specify: Pearson? Spearman? Across what -- the 15 scenarios as data points? With only 15 data points, correlation thresholds (0.3, 0.5, 0.7) need justification. This is the weakest methodological link and the most important to nail since the "network rewiring" claim rests on it.
- **Steering vector method (line 515)**: "compute a steering vector (the difference between moral and strategic model activations)" -- at which token position? Averaged over what?
- **Path patching (lines 591-593)**: "replacing entire pathways, consecutive chunks of the residual stream" -- replacing residual stream values or just MLP/attention outputs?
- **Validation check (lines 625-633)**: Uses 9 samples. Can't meaningfully distinguish 92.7% vs 92.6% with n=9. Be upfront about this limitation.

## 3. Claim Strength vs Evidence

### Potentially overstated:
- **"This finally explained our mystery"** (line 443) → "This suggested an answer" (causal evidence comes later in Part 5)
- **"Proves Pathway Causality"** (Part 5 heading) → "Demonstrates" or "Establishes"
- **"Causal Proof"** in mermaid diagram (line 53) → same issue
- **Chen et al. (2025) as "external validation"** (line 447) → note whether their setup is similar enough to be converging evidence
- **"distributed processing in subsequent layers compensates"** (line 585) → this assumes active correction, which is stronger than the evidence supports

### Appropriately hedged:
- Caveats section (lines 724-745) is well done
- Frankenstein appendix (lines 750-769) is admirably honest

### Undersold:
- **DLA finding** (lines 228-231): Pro-Cooperate/Defect components identical across all models *including untrained base* = cooperation/defection are pretrained world knowledge, not fine-tuning artifacts. Deserves more emphasis.
- **78% directional asymmetry** (line 313): Genuinely novel observation, presented as an afterthought.

## 4. Structure and Pacing

### What works:
- Mystery → investigation → discovery arc is effective
- Part 5 (causal experiments) is the strongest section

### What to fix:
- **Part 3 null results** (patching, attention, probes) presented sequentially each with full Setup/Findings treatment. Consider compressing into one "Ruling Out Hypotheses" section. The attention section (lines 339-347) is ~8 lines of content -- doesn't need its own major header.
- **Mermaid diagrams**: Keep first (line 42) and last (line 646). Cut intermediate ones (lines 255, 378, 463) -- reader can follow the narrative without a roadmap every 2 pages.
- **Part 1** could be compressed. Evaluation prompts section (lines 107-133) describes 5 scenarios in detail; most readers only need "I designed 15 controlled test prompts across 5 IPD contexts."
- **Lead with punchline**: Add a brief "here's what I found" after the intro. LW posts that do this get more engagement.

## 5. Figures

### Essential (keep in body): ~6-8
- Figure 2a (CC_temptation logit lens) -- single clearest visual
- Figure 3 OR 3b (DLA components) -- one is enough
- Figure 6c (correlation difference heatmap) -- tells the rewiring story
- Figure 8a (all steering sweeps overlaid)
- Figure 8e (KEY washout) -- most important figure
- Figure 9a (progressive path patching)

### Move to appendix:
- Figure 2b (all scenarios grid) -- same point as 2a, more exhaustive
- Figure 2c (final preferences heatmap) -- minor validation
- Figures 6a-b (raw correlation matrices) -- 6c alone tells the story
- Figure 8b (effect size heatmap) -- overlay already makes the point
- Figures 8c/8d (bidirectional steering) -- supplementary to 8e

### Could cut entirely:
- Figure 7a (betrayal probe) -- "nothing interesting" can be text
- Figure 5a (layer-wise patching sensitivity) -- spatial pattern is secondary to the zero-flip finding

**Current count**: ~20 figures. Target for LW: 8-10 in body, rest in appendix.

## 6. Title Alternatives

"Investigating How Moral Fine-Tuning Changes LLMs" is too generic. Suggestions:

1. **"Same Parts, Different Wiring: Mechanistic Interpretability of Moral Fine-Tuning"**
2. **"Moral Fine-Tuning Doesn't Delete Selfishness, It Reroutes Around It"**
3. **"The Selfish Neuron That Didn't Get Suppressed: Mech Interp of Moral RL"**
4. **"How Moral Training Rewires LLM Decision Routing Without Changing Components"**
5. **"Zero Flips in 21,060 Patches: How Moral Alignment Lives in Wiring, Not Neurons"**

## 7. Safety Implications

### What works:
- Four numbered implications (lines 712-718) are concrete and actionable
- Waluigi Effect connection is interesting

### What's missing:
- **Point 2 (original wiring remains)** deserves more development. Direct relevance to jailbreaking -- if aligned behavior is routing-dependent, targeted activation manipulation at L16/L17 = a jailbreak vector. Connect to existing activation-based attack research.
- **Point 4 (bypass switch is locatable)** is the most provocative claim. Preempt the objection: "locatable in this small model on this specific task" vs "locatable in general."
- **Missing implication**: Cooperation/defection features in the *base* pretrained model (before any fine-tuning) = implications for alignment-by-default and the "alignment tax" discussion.
- **Waluigi Effect thread**: Introduced at line 29, briefly revisited at line 709. Your findings provide concrete mechanistic evidence: the "Waluigi" wasn't *sharpened* by training, it was already there. Training routes around it. That's more nuanced than the original framing. Make this thread more explicit.

## 8. LessWrong Conventions

- [ ] **Add epistemic status header.** E.g.: *"Epistemic status: Moderately confident in the main finding (network rewiring). Less confident in generalizability beyond Gemma-2-2b-it on IPD. This is my first mech interp project, completed during ARENA."*
- [ ] **Add TL;DR.** LW readers expect one for long posts.
- [ ] **`> [!note]` syntax** (line 35) is GitHub markdown, not LW markdown. Check rendering.
- [ ] **Mermaid diagrams** don't render natively on LW. Convert to images.
- [ ] **Closing line** ("If you made it this far, thanks for reading!") reads blog-casual. Replace with forward-looking statement about next steps or open questions.
- [ ] **Image hosting**: Ensure blog_bundle_write_up images are hosted somewhere LW can access (upload directly to LW editor).
