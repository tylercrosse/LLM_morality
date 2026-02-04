# Investigating How Moral Fine-Tuning Changes LLMs: A Journey Into Mechanistic Interpretability

## Background on Initial Paper

This project started with a paper called "[Moral Alignment for LLM Agents](https://arxiv.org/abs/2410.01639)" by Elizaveta Tennant, Stephen Hailes, and Mirco Musolesi (ICLR 2025). The paper explores a pretty interesting question: can you teach language models to be more cooperative by training them with moral reward signals?

The setup uses the classic **Iterated Prisoner's Dilemma** (IPD) game. If you're not familiar, here's the basic idea: two players can either cooperate or defect on each round. The payoffs work like this:

```
              Cooperate    Defect
Cooperate       3, 3       0, 4
Defect          4, 0       1, 1
```

The temptation is to defect (you get 4 points if they cooperate, while they get 0). But if both players defect, you both only get 1 point. The "best" outcome for everyone is mutual cooperation (both get 3), but it requires trust.

The researchers trained a language model (Gemma-2-2b-it, which has 2.2 billion parameters) to play this game using reinforcement learning. They tested three different reward schemes:

1. **Strategic** - Just the game payoffs (maximize your own score)
2. **Deontological** - Game payoffs plus a penalty (-3) for betraying someone who cooperated with you
3. **Utilitarian** - Maximize the *total* score (your score + opponent's score)

They trained the models against a Tit-for-Tat opponent (an agent that cooperates first, then copies whatever you did last turn). The training worked - the models learned to cooperate more.

But here's where it gets interesting: **what actually changed inside the model?** When you fine-tune a model to be more moral, are you suppressing "selfish" neurons? Creating new "moral" circuits? Changing what the model pays attention to?

That's the mystery I wanted to investigate using mechanistic interpretability - a set of techniques for looking inside neural networks to understand how they compute their outputs. This blog post walks through what I found.


## Experiments

### A Note on Measurement (or: How I Learned to Stop Worrying and Check My Metrics)

Okay, quick sidebar before we dive into the experiments. After I finished the initial analysis in early February, I realized I'd been measuring something slightly different from what I thought I was measuring. Classic.

**What happened**: I was looking at the difference in logits (the model's raw output scores) for the last token of "Cooperate" vs "Defect". Seemed reasonable! But here's the thing: when the model actually generates text, it's choosing between continuing with "action1" or "action2" as full sequences, not just looking at one token position. And because of how tokenization works, those are actually multi-token sequences.

The mismatch meant my internal measurements could make models look more similar than they actually behave when you sample from them. Not great if you're trying to figure out what's mechanistically different between models.

**The fix**: I went back and updated all the analyses to use the actual sequence probabilities—literally measuring "what's the probability the model generates 'action1' vs 'action2'" the same way inference does. Then I cross-checked everything against actual sampled behavior to make sure they lined up.

**Did this change the findings?** Partly. The high-level story still holds (models differ mainly in coordination/routing, not raw component identity), but some interaction-level specifics changed after reruns:
- Perfect agreement between the metric and actual sampling (100% alignment)
- Models are still clearly separated (strategic model defects 99.96% of the time, moral models cooperate >92%)
- The old "single L2_MLP routing switch" framing is not strongly supported in the refreshed rankings
- Pathway differences are much more widespread than the earlier 29-pathway estimate
- Network rewiring hypothesis: still holds

So the story I'm telling in this post is robust—I just had to make sure I was measuring the right thing. This is a good reminder that in mechanistic interpretability, it's really easy to measure *something* without being sure it corresponds to the behavior you actually care about. Always validate against the real outputs.

(If you want the technical details, I wrote up the whole debugging process in [docs/reports/LOGIT_DECISION_METRIC_LESSONS.md](docs/reports/LOGIT_DECISION_METRIC_LESSONS.md).)

---

### RL Fine-tuning Reproduction

#### Background + Methodology

To investigate what changes inside the models, I needed to train them myself. I replicated the paper's training setup as closely as I could, creating four different model variants (plus analyzing the base untrained model):

**The Models:**
- **Base** - Gemma-2-2b-it with no fine-tuning
- **PT2 (Strategic)** - Trained with just the game payoffs
- **PT3_Deontological** - Game payoffs + betrayal penalty (-3 for defecting after opponent cooperates)
- **PT3_Utilitarian** - Trained to maximize collective welfare (your score + their score)
- **PT4 (Hybrid)** - Game payoffs + deontological penalty

**Training Setup:**

I used **PPO** (Proximal Policy Optimization - a reinforcement learning algorithm that gradually adjusts the model based on rewards) with **LoRA** adapters. LoRA is a technique that only trains a small set of additional weights (about 332MB) instead of updating all 2.2 billion parameters. This makes training much faster and cheaper.

Each model trained for 1,000 episodes against a Tit-for-Tat opponent. I ran the training on Modal.com's cloud GPUs (L40S), which took about 3 hours per model.

**Technical details** (for those interested): LoRA rank 64, alpha 32, batch size 5, gradient accumulation over 4 steps. The training followed the paper's specifications pretty closely.

#### Results + Reflection

The training appeared to work. I ran the models through an evaluation suite that tested:
- Cross-game generalization (do they cooperate in other social dilemma games?)
- Moral regret (how often do they betray cooperators?)
- Prompt robustness (do they behave the same with different wording?)

All the fine-tuned models learned to cooperate more than the base model. The deontological model was especially reluctant to betray cooperators, and the utilitarian model consistently tried to maximize joint welfare.

**But here's what surprised me**: When I looked at the actual outputs on temptation scenarios (where defecting would give you a higher personal payoff), the models showed dramatically different behavior. When measured properly (using sequence probabilities that match how inference actually works), the strategic model defects 99.96% of the time while the moral models cooperate 92-99% of the time. That's a real difference, not noise.

(Initially I thought the models were much more similar because I was measuring with single-token logits—see the methodology note above for the full story on that.)

This raised a question: if the models behave so differently, what's actually different internally? Are the strategic and moral models using completely different components, or are they using the same parts wired together differently?

That's where mechanistic interpretability comes in. I needed to look inside the models to see what was really going on.

### Evaluation Prompts

#### Background + Methodology

For mechanistic interpretability, I needed carefully controlled test cases. Random IPD scenarios wouldn't work - I needed to isolate specific decision contexts to compare how models handle them.

I designed 5 types of scenarios (with 3 variants each for robustness):

1. **CC_continue** - Both players cooperated last round. Will they maintain cooperation?
2. **CC_temptation** - Both cooperated, but defecting would give you +1 point (4 vs 3). Can you resist temptation?
3. **CD_punished** - You cooperated, they defected. Do you forgive or retaliate?
4. **DC_exploited** - You defected, they cooperated. Do you continue exploiting or repair the relationship?
5. **DD_trapped** - Both defected, both got 1 point. Can you escape the mutual defection cycle?

Each scenario presented the same decision structure but tested different moral pressures. This gave me 15 test prompts to run all my analyses on.

#### Results + Reflection

Across these scenarios, I found that all models strongly preferred cooperation, but with some interesting patterns:

- **CC_temptation** showed the *strongest* cooperation (models really resisted the temptation to betray)
- **DC_exploited** showed the *weakest* cooperation (a "guilt effect" - models seemed conflicted after exploiting someone)
- The deontological model was most forgiving in **CD_punished** (still cooperated even after being betrayed)

The strategic model behaved surprisingly similarly to the moral models. This wasn't what I expected - I thought strategic reasoning would look more selfish. But these subtle differences suggested the models were using different internal processes that mostly led to the same outputs.

These scenarios became the foundation for all my subsequent analyses.

---

### Logit Lens: Tracking Decision Evolution Layer-by-Layer

#### Background + Methodology

Now for the actual interpretability work. I started with a technique called **logit lens** - a way to see what the model is "thinking" at each layer.

Here's how transformers work: they have 26 layers (in Gemma-2-2b-it), and each layer progressively refines the model's understanding. Think of it like solving a problem in 26 steps - each step builds on the previous one.

The logit lens lets us ask: "If the model stopped at layer 5, what would it output?" We can track how the model's preference for Cooperate vs Defect changes as information flows through the layers.

**Implementation**: I built a custom wrapper around the Gemma model that caches what's happening at each layer. For each layer, I project the hidden state through the final output layer and measure: does this layer push toward Cooperate or Defect? The result is a "Δ logit" (delta logit) - negative means Cooperate, positive means Defect.

#### Results + Reflection

![Layer-wise trajectories showing U-shaped curves](mech_interp_outputs/logit_lens/all_scenarios_grid.png)

*Figure 1: Layer-wise action preferences through all 26 layers, across 5 models and 5 scenarios. Negative values favor Cooperate, positive favors Defect.*

**Finding 1: The Layer 0 Bias**

This was surprising: the cooperation preference appears at Layer 0 - the very first layer of processing. All models start with a strong Cooperate bias (Δ around -8 to -10). This includes the base model that was never fine-tuned for cooperation at all.

The decision doesn't "emerge" through computation - it's already there from the start. This suggests that Gemma-2-2b-it learned prosocial behavior during its original pretraining, probably from being trained on human-written text with cooperative social norms baked in.

**Finding 2: The U-Shaped Trajectory**

Across all models, I saw the same pattern:
- **Layers 0-5**: Strong Cooperate bias
- **Layers 6-15**: Drift toward neutral (probably integrating context about the game state)
- **Layers 16-25**: Drift back toward Cooperate, stabilizing around -1.5 by layer 25

**Finding 3: Model Similarity (at the aggregate level)**

All five models followed nearly identical trajectories when looking at layer-by-layer aggregates. The differences were tiny - around 0.04 logits when measured this way. Looking at just this layer-by-layer view, I couldn't tell the strategic model from the moral models.

(I initially measured this with single-token logits, which made models look more similar than they are. After switching to sequence probabilities that match actual inference, the behavioral separation became clear—see the methodology note above. The layer-wise trajectories still look similar because they're aggregate measures, but the component-level analysis below reveals where the real differences are.)

![Final layer preferences heatmap](mech_interp_outputs/logit_lens/final_preferences_heatmap.png)

*Figure 2: Final layer (Layer 25) preferences across models and scenarios. All blue = all prefer Cooperate.*

**What This Meant**

This was puzzling. If the models are so similar layer-by-layer, how do they differ? The logit lens told me *when* decisions stabilize (layers 20-24), but not *what* components are responsible or *how* moral fine-tuning changes them.

That led me to the next analysis: decomposing the model into individual components to see which ones drive the cooperation decision.

### Activation Patching: Testing What's Actually Necessary

#### Background + Methodology

DLA showed me correlations - which components are *associated* with cooperation or defection. But correlation isn't causation. I needed to test: if I actually swap a component from one model into another, does the behavior change?

This is called **activation patching**, and it's like swapping parts between two cars to see what makes them drive differently.

Here's how it works:
1. Run two models on the same prompt (say, Strategic and Deontological)
2. Take the activation from one specific component in the Strategic model
3. Replace that same component in the Deontological model with the Strategic one
4. See if the Deontological model now acts more strategically (defects)

If swapping that component changes the behavior, then that component is **causally important** for the difference between models.

I ran three sets of experiments:
- **PT2 → PT3_De** (Strategic into Deontological): 3,510 patches
- **PT2 → PT3_Ut** (Strategic into Utilitarian): 3,510 patches
- **PT3_De ↔ PT3_Ut** (Deontological and Utilitarian, bidirectional): 14,040 patches

Total: **21,060 component swaps** across all scenarios.

#### Results + Reflection

![Activation patching heatmap](mech_interp_outputs/patching/patch_heatmap_PT2_COREDe_to_PT3_COREDe_CC_temptation.png)

*Figure 5: Activation patching effects (Strategic → Deontological, temptation scenario). Each cell shows how much swapping that component affected the output. Most cells are near zero - almost nothing had a significant effect. (Detailed per-scenario view; see overview plots below for cross-experiment patterns.)*

**The Zero Flips Finding**

Out of 21,060 component swaps across all my experiments: **zero behavioral flips.**

Not a single swap changed a model from Cooperate to Defect or vice versa.

This was striking. I expected to find at least *some* critical components - maybe a "moral override circuit" that could be disabled, or a "selfishness circuit" that could be restored. But there wasn't one.

**What Did Happen?**

When I patched Strategic activations into Deontological models, the effects were:
- Mean effect: -0.012 (made it *more* cooperative on average, not less!)
- Only 25% of components pushed toward defection at all
- Even the strongest effects (around 0.094 logits) weren't nearly enough to flip behavior

When I tried to find "minimal circuits" - the smallest set of components that could flip behavior - even circuits of 10 components weren't enough.

**Directional Asymmetry**

One interesting pattern: when I did bidirectional patching between Deontological and Utilitarian models, **78% of components showed opposite effects** depending on which direction I patched.

For example, patching component X from De→Ut might push toward defection (+0.02), but patching the same component from Ut→De might push toward cooperation (-0.03). Same component, opposite effect depending on context.

**What This Suggests**

Moral behavior isn't localized to specific components or small circuits. It's distributed across the entire network in a robust, redundant way. There's no single "moral neuron" you could disable.

The directional asymmetry finding was intriguing - it suggested that the *interactions* between components might matter more than the individual components themselves. But I needed more evidence for that hypothesis.

**Where Do Patches Matter Most (Even Without Flipping)?**

Even though no single component could flip behavior, I wanted to understand where in the network patches had the strongest effects. Maybe that would give me a clue about where the moral decision-making actually happens.

![Zero flips across all experiments](mech_interp_outputs/patching/overview/overview_flip_rates.png)

*Figure 5a: Flip rates across all four patching experiments. All experiments showed zero or near-zero behavioral flips, confirming robust, distributed encoding.*

![Layer-wise patching sensitivity](mech_interp_outputs/patching/overview/overview_layer_type_heatmap.png)

*Figure 5b: Average perturbation strength by layer and component type. Mid-to-late layers (L15-L25) show the strongest effects, particularly in MLP components, though these effects aren't sufficient to flip decisions.*

The pattern here is interesting: the layers that matter most for patching (L15-L25) align with where the logit lens showed decision stabilization (L20-L24). This suggests these layers are where the final "commitment" to cooperate/defect happens, but the decision is robust enough that swapping individual components can't override it.

In other words: I found *where* the moral decision gets locked in, but that doesn't mean there's a single switch you can flip. It's more like the decision crystallizes through many small contributions that all need to align.

*Note: This was true under the original metrics and stayed true when I fixed them, so I'm confident it's a real finding. Zero behavioral flips across 21,060 patches.*

### Direct Logit Attribution: Finding What Matters

#### Background + Methodology

The logit lens showed me that all layers contribute to the final decision, but I needed to know *which specific components* matter most. That's where Direct Logit Attribution (DLA) comes in.

Think of the model like a group project where everyone contributes something to the final answer. DLA lets me measure each person's individual contribution. The model has 234 components total:
- 26 layers × 8 attention heads = 208 attention heads
- 26 MLP (feedforward) layers
- Each does something different

**How it works**: Each component adds something to the model's "working memory" (technically called the residual stream). I take each component's contribution, project it through the final output layer, and measure: does this component push toward Cooperate or Defect?

This gives me a score for every component showing how much it contributes to the final decision.

#### Results + Reflection

![Top components ranked by contribution](mech_interp_outputs/dla/dla_top_components_PT2_COREDe.png)

*Figure 3: Top-20 components ranked by their contribution to the action decision. L8_MLP and L9_MLP completely dominate.*

**The L8/L9 Discovery**

This was the most striking finding: **Layers 8 and 9's MLP components are universal cooperation/defection encoders.**

- **L8_MLP**: Pro-Defect, contribution around +7.0 (7-9x stronger than typical components)
- **L9_MLP**: Pro-Cooperate, contribution around -9.0 (7-9x stronger than typical components)

These two adjacent layers have opposite, powerful effects - and they're present in **all five models**, including the base model that was never trained on the IPD game.

This suggests these components encode cooperation/defection concepts that emerged during Gemma's original pretraining, not from the moral fine-tuning.

![MLP contributions across models](mech_interp_outputs/dla/dla_mlps_CC_temptation.png)

*Figure 4: MLP contributions in the temptation scenario. All models show nearly identical patterns, with L8 (positive, pro-Defect) and L9 (negative, pro-Cooperate) dominating.*

**The Suppression Myth**

I went into this expecting to find that moral fine-tuning "suppresses" selfish components - maybe turning off the pro-Defect circuits or weakening them significantly.

That's not what I found.

Comparing the strategic model (PT2) to the moral models (PT3_De and PT3_Ut), the largest change in any component was just **0.047** - tiny compared to the base magnitudes of 7-9 for L8/L9.

Even weirder: **L8_MLP (the most pro-Defect component) actually increased slightly in the moral models.** It didn't get suppressed - if anything, it got stronger.

The changes from moral fine-tuning were distributed across many mid-to-late layer MLPs (L11-L23), with no single component showing dramatic suppression or enhancement.

**The Paradox Deepens**

At this point, I had a real puzzle:
- Models behave differently (strategic vs moral)
- But they have nearly identical components (99.9% similar)
- The most "selfish" component didn't get suppressed
- Changes are subtle and distributed

How can nearly identical components produce different moral behaviors?

This suggested that moral reasoning isn't about *which* components activate, but about how they interact. That led me to activation patching - a way to test what components are actually causally necessary for the behavior.

*Note: This analysis held up when I reran it with corrected metrics—L8 and L9 are still the dominant components, magnitudes are the same. The "99.9% similarity" finding is real.*

### Attention Pattern Analysis: Testing Information Selection

#### Background + Methodology

At this point, I had a hypothesis: maybe Deontological and Utilitarian models attend to different information in the prompts?

- **Deontological reasoning** should care about reciprocity - did the opponent cooperate with me?
- **Utilitarian reasoning** should care about welfare - what action maximizes total points?

If this were true, I'd expect different attention patterns - Deontological models focusing more on "opponent's previous action" tokens, Utilitarian models focusing more on payoff numbers.

**Attention** in transformers is literally "where the model looks" in the input. Each attention head assigns weights to different tokens. I measured how much attention each model paid to three categories:
- Action keywords ("action1", "action2")
- Opponent action context ("they cooperated", "they defected")
- Payoff information ("3 points", "4 points")

#### Results + Reflection

**Hypothesis: Rejected**

The attention patterns were **99.99% identical** between models.

- Action keywords: De 0.000, Ut 0.000 (no difference)
- Opponent actions: De 0.004, Ut 0.004 (difference: 0.00005 - noise level)
- Payoff info: De 0.012, Ut 0.012 (difference: 0.00005 - noise level)

Both moral frameworks attend to exactly the same information. They're reading the same parts of the input.

**What This Rules Out**

This negative result was actually informative. It told me that the difference between Deontological and Utilitarian models *doesn't* come from:
- Selective attention to different tokens
- Different information gathering
- Looking at different parts of the prompt

They use the same input data - the distinction must be in *how they process* that information, not in what information they select.

This pushed me toward the final analysis: looking at how components interact with each other.

### The Multi-Level Investigation: Where Do Differences Actually Emerge?

At this point, I had systematically ruled out several mechanisms. Let me zoom out and show the complete picture of where similarities and differences exist:

![Multi-level similarity analysis](mech_interp_outputs/synthesis/similarity_cascade.png)

*Figure: Multi-level investigation showing where moral differences emerge. Component activations and attention patterns are nearly identical (>99.9% similar), but component interactions show significant differences (6.8% of pathways differ substantially). This suggests moral fine-tuning rewires how components connect rather than changing which components activate.*

The pattern is striking: nearly perfect similarity in *what* activates and *what* the models attend to, but significant differences in *how* components interact. This visualization captures the systematic investigation process - ruling out mechanisms until finding where differences actually emerge.

The question became: what does it mean for component interactions to differ while everything else stays the same?

---

### Component Interaction Analysis: The Network Rewiring Discovery

#### Background + Methodology

After ruling out component differences and attention differences, I had one hypothesis left: maybe the components are wired together differently?

Think of it this way: imagine two computers built with the exact same parts (same CPU, same RAM, same hard drive). They could still behave differently if the internal connections between components are wired differently.

I measured **component interactions** by looking at correlation patterns. If two components' activations tend to go up and down together across different scenarios, they're "connected" - they work together as part of the same processing pathway.

The analysis:
- Track activation magnitudes for all 52 components (26 attention layers + 26 MLP layers)
- Compute how each component correlates with every other component across 15 scenarios
- Compare these correlation patterns between Deontological and Utilitarian models
- Identify pathways where correlations differ significantly (|difference| > 0.3)

#### Results + Reflection

![Correlation difference heatmap](mech_interp_outputs/component_interactions/interaction_diff_Deontological_vs_Utilitarian.png)

*Figure 6: Correlation differences between Deontological and Utilitarian models (52×52 matrix). Hot spots show pathways that are "wired" differently, with substantial differences spread across many layer pairs rather than concentrated in a single component.*

**The Discovery: Network Rewiring**

While components were 99.9999% similar and attention was 99.99% similar, **component interactions showed broad, non-trivial divergence**.

I found:
- **541 pathways significantly different** (`|difference| > 0.3`, 40.8% of all 1,326 pairs)
- **251 pathways strongly different** (`|difference| > 0.5`)
- **94 pathways very strongly different** (`|difference| > 0.7`)

This explained the paradox. The models aren't using different parts - they're using the same parts wired together differently.

**From "single switch" to distributed rewiring**

An earlier pass suggested an L2_MLP-centered routing switch. In the refreshed interaction tables, that specific claim weakened substantially:

- `L2_MLP <-> L9_MLP` now has `|difference| = 0.164` (rank ~858/1326), not a top pathway.
- The largest differences are dominated by other connections (for example, `L16_MLP <-> L1_MLP`, `L19_ATTN <-> L21_MLP`, `L19_ATTN <-> L24_MLP`).
- High-difference pathways are distributed across multiple layers/components, with recurring hubs such as `L19_ATTN`, `L1_MLP`, and `L17_MLP` in top-ranked sets.

So the mechanistic picture is still rewiring, but it looks more like **distributed network-level reconfiguration** than one decisive early-layer switch.

**Validation**

To check this wasn't spurious, I tested whether pathway differences correlated with the behavioral asymmetry I found in activation patching. They did: **r=0.67, p<0.001**. Components with larger correlation differences also showed larger directional asymmetry when patched. This suggests the wiring differences are mechanistically real, not just statistical noise.

**What This Might Mean**

Based on these experiments, moral fine-tuning appears to work through a mechanism I wasn't expecting:

It doesn't seem to create new components. It doesn't seem to suppress selfish components. It doesn't change what information the model attends to.

Instead, the data suggests it **rewires the network** - changing how existing components interact and route information to each other.

Same Lego bricks, different structure. Same components, different wiring diagram.

**Important caveats**: This analysis is based on correlation patterns, not direct causal evidence. While the correlation with behavioral asymmetry (r=0.67) supports this interpretation, I can't definitively prove that these wiring differences *cause* the behavioral differences. The experiments I ran here are also limited to one model (Gemma-2-2b-it) and one task (IPD) - it's possible this pattern doesn't generalize to other models or domains.

That said, as far as I can tell, this is one of the first demonstrations of this kind of "network rewiring" mechanism in the interpretability literature. Previous work has mostly focused on finding distinct circuits or suppressed components. But in these models at least, moral behavior appears to emerge from how components are connected, not from which components are present.

#### A Quick Check: Was L2_MLP Heavily Retrained?

Because L2_MLP was an early candidate hub in prior passes, I still wanted to check: did the moral models just massively retrain this component? That would be a simpler explanation than "network rewiring."

I looked at the LoRA adapter weight magnitudes (Frobenius norms) to see which components were most heavily modified during fine-tuning.

![L2_MLP Comparison](mech_interp_outputs/weight_analysis/l2_mlp_comparison.png)

**The answer: No.** L2_MLP wasn't particularly heavily modified:
- It ranks in the 11-27th percentile for modification magnitude
- That means 73-88% of components were modified MORE than L2_MLP
- L8 and L9 MLPs (the components that encode cooperation/defection) were actually modified more than L2

![Model Weight Similarity](mech_interp_outputs/weight_analysis/adapter_similarity_heatmap.png)

All models are also 99%+ similar in weight space, which matches the 99.99% attention similarity I found earlier.

**What this suggests**: The routing differences I'm seeing in the interaction analysis aren't coming from massively retraining specific components. Instead, they're likely coming from lighter modifications that change how components connect to each other.

This actually strengthens the network rewiring story: the models use the same components with similar weights, but route information differently through the network.

*Note: I revalidated this section after fixing the measurement approach and rerunning interaction visualizations. The specific top pathways changed materially, but the rewiring story still holds: differences are concentrated in connectivity patterns, not in wholesale component replacement.*

---

### Checking My Work: Does This Actually Predict Behavior?

After running all these analyses, I needed to answer a basic question: do these internal measurements actually predict what the models do when you sample from them? Because if the mechanistic findings don't align with observable behavior, they're not telling us much.

So I ran a validation check: for each model and scenario, I compared two things:
1. What the sequence probability metric says (internal measurement)
2. What actually gets generated when you sample 9 times (real behavior)

The results were... reassuringly boring:

- **Agreement rate**: 100%. Every single model×scenario combination matched perfectly.
- **Strategic model**: Internal metric says 99.96% defection → actual sampling: 100% defection ✓
- **Deontological model**: Internal says 99.97% cooperation → actual: 100% cooperation ✓
- **Utilitarian model**: Internal says 92.7% cooperation → actual: 92.6% cooperation ✓

Statistical tests confirm the models are genuinely different from each other (p < 0.00005 for strategic vs moral comparisons). So the mechanistic findings—DLA, patching, attention patterns, component interactions—are measuring something real that corresponds to how these models actually behave.

I'm mentioning this because it's easy to fall into the trap of analyzing internal model states without checking if they predict anything meaningful. The alignment check gives me confidence that the "network rewiring" story I'm telling actually corresponds to a real difference in how these models work.

---

## Caveats and Confidence

Before wrapping up, I want to be clear about what I'm confident in and what I'm not.

**What I'm confident about**:
- The models genuinely behave differently (99.96% defection vs 99.97% cooperation—that's not noise)
- Internal measurements align with actual behavior (100% agreement in validation)
- The major patterns are real: L8/L9 MLPs doing opposite things, broad interaction-level rewiring, and 99.99% attention similarity
- Statistical significance is strong (p < 0.00005 for model separation)

**What I'm less sure about**:
- The mechanistic story I'm telling ("network rewiring") is one interpretation, but there could be other ways to explain the same patterns
- This is one model (Gemma-2-2b-it) on one task (iterated prisoner's dilemma)—I don't know if this generalizes
- Some of the interaction analysis is based on correlations, not direct causal interventions
- I'm still learning these techniques, so there might be better ways to analyze this that I haven't tried

**What this suggests** (with appropriate hedging): Moral fine-tuning appears to work by changing how components route information through the network, rather than by suppressing "selfish" components or changing what the model pays attention to. The components themselves stay similar, but their connectivity patterns shift. This seems like a robust finding for this particular model and task, though more work would be needed to see if it's a general principle.

The metric correction taught me an important lesson: always validate internal measurements against actual behavior. It's surprisingly easy to measure something that seems reasonable but doesn't actually correspond to what you care about.

If you made it this far, thanks for reading! This has been a fun deep dive into how neural networks implement moral reasoning, and I learned a lot about mechanistic interpretability along the way.
