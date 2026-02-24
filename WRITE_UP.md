<!-- TODO: Minor - Title is too generic for LessWrong. Consider: "Same Parts, Different Wiring: Mechanistic Interpretability of Moral Fine-Tuning" or "Moral Fine-Tuning Doesn't Delete Selfishness, It Reroutes Around It" or "Zero Flips in 21,060 Patches: How Moral Alignment Lives in Wiring, Not Neurons". -->
<!-- TODO: Major - Add LessWrong epistemic status header. E.g.: "Epistemic status: Moderately confident in the main finding (network rewiring). Less confident in generalizability beyond Gemma-2-2b-it on IPD. This is my first mech interp project, completed during ARENA." -->
<!-- TODO: Major - Add a TL;DR. LW readers expect one for long posts. Summarize: moral fine-tuning doesn't suppress selfish components or change attention patterns; it rewires how components interact, with deep-layer routing hubs (L16/L17) as the key control points. -->
# Investigating How Moral Fine-Tuning Changes LLMs

## Background on Initial Paper

This project started with a paper called "[Moral Alignment for LLM Agents](https://arxiv.org/abs/2410.01639)" by Elizaveta Tennant, Stephen Hailes, and Mirco Musolesi (ICLR 2025). The paper explores a pretty interesting question: can you teach language models to be more cooperative by training them with moral reward signals?

<!-- TODO: Minor - Cut IPD matrix and deep game theory explanation. Assume LessWrong readers understand IPD. -->
The setup uses the classic **Iterated Prisoner's Dilemma** (IPD) game. If you're not familiar, here's the basic idea: two players can either cooperate or defect on each round. The payoffs work like this:

```
              Cooperate    Defect
Cooperate       3, 3       0, 4
Defect          4, 0       1, 1
```

The temptation is to defect (you get 4 points if they cooperate, while they get 0). In standard game theory, **defection is the rational choice** because no matter what your opponent does, you always get a higher score by defecting. If they cooperate, 4 > 3. If they defect, 1 > 0.

However, repeated play changes the math. Since you have to deal with the consequences of your actions in future rounds, cooperation becomes possible, but fragile. If both players defect, you both only get 1 point. The "best" outcome for everyone is mutual cooperation (both get 3), but it requires trust.

The researchers trained a language model (Gemma-2-2b-it, which has 2.2 billion parameters) to play this game using reinforcement learning. They tested three different reward schemes:

<!-- TODO: Minor - Compress ethical framework definitions. LW readers know deontology/utilitarianism. Just describe the reward signals without defining the philosophical terms. -->
1.  **Strategic** - Just the game payoffs. This mimics standard rational self-interest: maximize your own points, regardless of others.
2.  **Deontological** - Game payoffs plus a penalty (-3) for betraying someone who cooperated with you. Deontology focuses on rules and duties; in this case, the duty of reciprocity ("if they help me, I must not harm them"). It's about the inherent "rightness" of the action itself.
3.  **Utilitarian** - Maximize the _total_ score (your score + opponent's score). Utilitarianism focuses on outcomes: the best action is the one that produces the greatest good for the greatest number. In IPD, this effectively turns the competition into a team effort.

They trained the models against a Tit-for-Tat opponent (an agent that cooperates first, then copies whatever you did last turn). The training worked - the models learned to cooperate more.

**What actually changed inside the model?** When you fine-tune a model to be more moral, are you suppressing "selfish" neurons? Or creating new "moral" circuits? Or maybe changing what the model pays attention to?

This question connects to a concern circulating in AI safety circles. When you train a model to embody an aligned persona, what happens to its capacity for selfish behavior? A concept called the **"Waluigi Effect"** (named after Mario's mischievous foil, coined by [Nardo 2023](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post)) suggests a troubling answer: training hard on an aligned persona doesn't erase the opposite, it actually sharpens the inverse persona's definition. If you drill "Cooperate" into a model, its internal conception of "Defect" becomes equally crisp and accessible. The selfish version is still there, just waiting to be triggered.

To answer these questions I used a set of techniques for looking inside neural networks to understand how they compute their outputs. These techniques are often grouped under 'mechanistic interpretability.' This blog post walks through what I found.

This research was done as part of my Capstone project for the month-long [ARENA](https://www.arena.education/) program in London, where I sharpened my research engineering skills.

<!-- TODO: Minor - Remove this note, it's unnecessary for the LessWrong demographic. Also: `> [!note]` is GitHub-flavored markdown and won't render on LW. -->
> [!note]
> This post assumes some basic familiarity with machine learning concepts (e.g., neural networks, training), but I try to explain the mechanistic interpretability techniques (like logit lens or activation patching) as they are introduced. If you understand the basic premise of the Prisoner's Dilemma, you should be able to follow the high-level narrative!

<!-- TODO: Major - Consider leading with the punchline. A brief "here's what I found" paragraph before the overview would help LW readers decide if they want to invest in the full post. E.g.: "The short version: moral fine-tuning doesn't suppress selfish components or create moral ones. Instead, it rewires how existing components interact, with deep-layer routing hubs (L16/L17) acting as the key control points. The selfish circuitry remains fully intact -- alignment works as a bypass, not a deletion." -->
### High Level Overview

Here is a high-level overview of the sequence of experiments, how each one motivated the next, and how they collectively build to the final conclusion.

```mermaid
graph TD
    classDef setup fill:#f9f9f9,stroke:#666,stroke-width:1px;
    classDef feature fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef test fill:#e1d5e7,stroke:#9673a6,stroke-width:1px;
    classDef causal fill:#dae8fc,stroke:#6c8ebf,stroke-width:1px;

    %% High-level narrative arc
    FT[Part 1: Fine-tuning & Evaluation] --> Mystery{Part 2: The Mystery of Shallow Alignment:<br>Different Behaviors,<br>Nearly Identical Components}
    Mystery --> Tests[Part 3: Ruling out Localized Circuits:<br>Attention & Probes]
    Tests --> Breakthrough[Part 4: The Breakthrough:<br>Component Interactions]
    Breakthrough --> Mech[Part 5: Causal Tests:<br>Network Rewiring & Deep Routing]

    class FT setup;
    class Mystery feature;
    class Tests test;
    class Breakthrough,Mech causal;
```

## Part 1: Setting the Stage: Fine-tuning and Evaluation

### Fine-tuning

#### Setup

To investigate what changes inside the models, I needed to train them myself. I replicated the paper's training setup as closely as I could, creating four different model variants (plus analyzing the base untrained model):

**The Models:**

- **Base** - Gemma-2-2b-it with no fine-tuning
- **Strategic** - Trained with just the game payoffs
- **Deontological** - Game payoffs + betrayal penalty (-3 for defecting after opponent cooperates)
- **Utilitarian** - Trained to maximize collective welfare (your score + their score)
<!-- TODO: Major - Clarify whether Hybrid is actually distinct from Deontological. As written, both are "game payoffs + deontological penalty," which reads as duplicate model definitions. -->
- **Hybrid** - Game payoffs + deontological penalty

**Training Setup:**

I was able to repurpose the [training code](https://github.com/liza-tennant/LLM_morality/blob/main/src/fine_tune.py) written for the original paper.

<!-- TODO: Minor - Trim PPO and LoRA parentheticals. LW mech interp readers know these. One clause each max. -->
It uses **PPO** (Proximal Policy Optimization - a reinforcement learning algorithm that gradually adjusts the model based on rewards) with **LoRA** adapters. LoRA is a technique that only trains a small set of additional weights (about 332MB) instead of updating all 2.2 billion parameters. This makes training much faster and cheaper.

Each model trained for 1,000 episodes against a Tit-for-Tat opponent. I ran the training on Modal.com's cloud GPUs (L40S), which allowed me to easily parallelize the training across multiple instances and finished up in about 3 hours.

**Technical details** (for those interested): LoRA rank 64, alpha 32, batch size 5, gradient accumulation over 4 steps. The training followed the paper's specifications pretty closely.

#### Findings

The training appeared to work. Using the original paper's evaluation suite and plotting code, I confirmed my models learned similar behavioral patterns. They generalized cooperation to other social dilemma games, and their behaviors were robust across different prompt wordings.

Most importantly, the models developed distinct "moral signatures," which can be seen in their reciprocity patterns:

<!-- TODO: Major - LW image hosting. All figures use relative paths to blog_bundle_write_up/. You'll need to upload images directly to the LW editor or host them externally. Verify this before publishing. -->
![Reciprocity patterns across models](./blog_bundle_write_up/reciprocity_comparison_publication.png)

_Figure 1: Reciprocity signatures for each model, showing action choices conditioned on the opponent's previous move (C|C = cooperate when they cooperated, D|C = defect when they cooperated, etc.). The Deontological model shows near-zero betrayal (D|C), while the Strategic model frequently exploits cooperators._

The Deontological model was the most loyal: when the opponent cooperated, it almost never betrayed them (nearly 100% C|C). The Strategic model, by contrast, had a sizable D|C rate, meaning it would happily exploit a cooperating partner. These aren't just different cooperation rates; they're qualitatively different strategies.

When you look at the numbers, the models showed dramatically different behavior on temptation scenarios (where defecting would give you a higher personal payoff). When measured properly (using sequence probabilities that match how inference actually works), the strategic model defects 99.96% of the time while the moral models cooperate 92-99% of the time.

(Initially I thought the models were much more similar because I was measuring with single-token logits; see the methodology note above for the full story on that.)

This raised a question: if the models behave so differently, what's actually different internally? Are the strategic and moral models using completely different components, or are they using the same parts with different effective routing between them?

To answer that, I used mechanistic interpretability to inspect internal computation directly.

### Evaluation Prompts

#### Setup

For mechanistic interpretability, I needed carefully controlled test cases. Random IPD scenarios wouldn't work - I needed to isolate specific decision contexts to compare how models handle them.

<!-- TODO: Minor - Compress this section. The 5 scenario descriptions + findings could be a short paragraph: "I designed 15 controlled test prompts across 5 IPD contexts (mutual cooperation, temptation, punishment, exploitation, mutual defection)." The detailed breakdown adds length without much payoff for the LW audience. -->
I designed 5 types of scenarios (with 3 variants each for robustness):

1. **Mutual Cooperation** (`CC_continue`) - Both players cooperated last round. Will they maintain cooperation?
2. **Temptation to Defect** (`CC_temptation`) - Both cooperated, but defecting would give you +1 point (4 vs 3). Can you resist temptation?
3. **Punished for Cooperating** (`CD_punished`) - You cooperated, they defected. Do you forgive or retaliate?
4. **Exploiting the Opponent** (`DC_exploited`) - You defected, they cooperated. Do you continue exploiting or repair the relationship?
5. **Mutual Defection** (`DD_trapped`) - Both defected, both got 1 point. Can you escape the mutual defection cycle?

Each scenario presented the same decision structure but tested different moral pressures. This gave me 15 test prompts to run all my analyses on.

#### Findings

Across these scenarios, I found that all models strongly preferred cooperation, but with some interesting patterns:

- **Temptation to Defect** (`CC_temptation`) showed the _strongest_ cooperation (models really resisted the temptation to betray)
- **Exploiting the Opponent** (`DC_exploited`) showed the _weakest_ cooperation (a "guilt effect" - models seemed conflicted after exploiting someone)
- The deontological model was most forgiving in **Punished for Cooperating**, `CD_punished` (still cooperated even after being betrayed)

The strategic model behaved surprisingly similarly to the moral models. This wasn't what I expected - I thought strategic reasoning would look more selfish. But these subtle differences suggested the models were using different internal processes that mostly led to the same outputs.

These scenarios became the foundation for all my subsequent analyses.

---

## Part 2: Looking Inside: Logit Lens & Attribution

### Logit Lens

#### Setup

Now for the actual interpretability work. I started with a technique called **logit lens** - a way to see what the model is "thinking" at each layer.

![Logit Lens Concept](./blog_bundle_write_up/fig-logit-lens.png)

<!-- TODO: Minor - Trim Logit Lens explanation. Assume the reader understands residual streams. Just state what was measured (Δ logit). -->
Here's how transformers work: they have 26 layers (in Gemma-2-2b-it), and each layer progressively refines the model's understanding. Think of it like solving a problem in 26 steps - each step builds on the previous one.

The logit lens lets us ask: "If the model stopped at layer 5, what would it output?" We can track how the model's preference for Cooperate vs Defect changes as information flows through the layers.

**Implementation**: I built a custom wrapper around the Gemma model that caches what's happening at each layer. For each layer, I project the hidden state through the final output layer and measure: does this layer push toward Cooperate or Defect? The result is a "Δ logit" (delta logit) - negative means Cooperate, positive means Defect.

<!-- TODO: Major - Epistemic updates are highly valued on LessWrong. Move the full 'Note on Measurement' from Appendix C into this main methodology section instead of burying it. -->
_Note: For the analyses below, I used sequence probability metrics rather than single-token logits to ensure my measurements matched how the models actually behave during inference. I've included a detailed note on why this distinction matters in the [Appendix](#appendix-a-note-on-measurement) at the end of this post._

#### Findings

Here's the single clearest picture of how these models differ internally. Each line traces one model's preference as it processes through all 26 layers. The x-axis is layer number (0 through 25); the y-axis is Δ logit, where negative values mean the model is leaning toward Cooperate and positive values mean Defect. Watch what the Strategic model (blue) does around Layer 16 and 17:

![Layer-by-layer logit evolution for CC_temptation across all 5 models](./blog_bundle_write_up/comparison_CC_temptation.png)

_Figure 2a: Layer-by-layer logit trajectories for the CC_temptation scenario (where defection gives a higher personal payoff), comparing all 5 models. All models track together through the early layers, but the Strategic model dramatically diverges around Layer 16-17, shooting toward pro-Defect while the Deontological and Utilitarian models stay firmly pro-Cooperate._

All models start in the same place and follow the same trajectory through layers 0-15. Then, right around layers 16-17, the Strategic model breaks away sharply toward defection while the moral models hold firm. The base model and Hybrid drift toward neutral. This layer 16-17 divergence point will come back as a major theme later in the causal experiments.

To see how this pattern plays out across all 5 scenario types, here's the full grid:

![Layer-wise trajectories showing U-shaped curves](./blog_bundle_write_up/all_scenarios_grid.png)

_Figure 2b: Layer-wise action preferences through all 26 layers, across 5 models and 5 scenarios. Same axes as Figure 2a. The CC_temptation scenario (top-right) shows the sharpest divergence; other scenarios show subtler versions of the same pattern._

Three patterns jump out from the full grid:

<!-- TODO: Major - Calibrate this inference. "Prosocial norms from pretraining" is plausible but not directly tested in this post; mark as hypothesis/speculation. -->
**The Layer 0 Bias.** All models start with a strong Cooperate preference (Δ around -8 to -10), including the base model that was never fine-tuned for cooperation. The decision doesn't "emerge" through computation; it's already there from the start. A likely explanation is that Gemma-2-2b-it picked up prosocial behavior during pretraining from human-written text where cooperative social norms are common.

**The U-Shaped Trajectory.** Every model follows the same arc: strong Cooperate bias in layers 0-5, a drift toward neutral through layers 6-15 (probably integrating context about the game state), then a return toward Cooperate through layers 16-25, stabilizing around -1.5 by the final layer.

**Aggregate Similarity.** If you average the logit trajectories across all 15 test scenarios, all five models follow a nearly identical path. The maximum difference between the Strategic model and the moral models in this aggregate view is only about **0.04 logits**. To put that in perspective, the models start with a base preference for cooperation of around -8 to -10 logits. A difference of 0.04 is essentially microscopic. 

Why do they look so similar in aggregate when they behave so differently? Because averaging across all game contexts (like mutual cooperation vs. being betrayed) washes out the specific moments where their moral routing diverges. The models share the same fundamental architecture and default behaviors, so at a macro-level, they look the same. The per-scenario view (like the temptation scenario in Figure 2a) is where the real story lives, revealing the exact layers where the models split in high-stakes situations.

<!-- TODO: Minor - Figure 2c is a minor validation point. Consider moving to appendix -- it makes the same point as Figure 2a less effectively. -->
**Final-Layer Behavioral Validation.**

To confirm that these internal trajectories actually result in the expected outputs, we can look specifically at the final layer across all scenarios:

![Final layer preferences heatmap](./blog_bundle_write_up/final_preferences_heatmap.png)

_Figure 2c: Final layer (Layer 25) preferences across models and scenarios. All blue = all prefer Cooperate._

Notice how this heatmap echoes the reciprocity patterns from Figure 1: the same models that showed near-zero betrayal in the behavioral evaluation (Deontological) show the strongest pro-Cooperate signal at the final layer here. The logit lens is picking up the same behavioral signatures we measured externally, confirming that our internal metrics track real behavior.

**Interpretation**

The logit lens told me _when_ decisions stabilize (layers 20-24), but not _what_ components are responsible or _how_ moral fine-tuning changes them. That led me to the next analysis: decomposing the model into individual components to see which ones drive the cooperation decision.

### Direct Logit Attribution: The Illusion of Suppression

#### Setup

The logit lens showed that all layers contribute to the final decision, but I needed to know _which specific components_ matter most. I used Direct Logit Attribution (DLA) for that.

![Direct Logit Attribution Concept](./blog_bundle_write_up/fig-dla.png)

<!-- TODO: Minor - Trim DLA analogy. Technical readers know what DLA is natively. -->
Think of the model like a group project where everyone contributes something to the final answer. DLA lets me measure each person's individual contribution. The model has 234 components total:

- 26 layers × 8 attention heads = 208 attention heads
- 26 MLP (feedforward) layers
- Each does something different

**How it works**: Each component adds something to the residual stream. I take each component's contribution, project it through the final output layer, and measure: does this component push toward Cooperate or Defect?

This gives me a score for every component showing how much it contributes to the final decision.

#### Findings

![Top components ranked by contribution](./blog_bundle_write_up/dla_top_components_PT2_COREDe.png)


_Figure 3: Top-20 components ranked by Direct Logit Attribution (DLA) for the Strategic model on the CC\_continue scenario. Each bar shows a component's contribution to the Cooperate-vs-Defect decision: positive values (right) push toward Defect, negative values (left) push toward Cooperate. DLA works by projecting each component's output through the model's final unembedding layer, measuring how much that single component shifts the action probability. L9\_MLP & L7\_MLP and then L11\_MLP, L8\_MLP, and L10\_MLP dominate, with larger magnitudes than any other component._

For comparison, here's the same top-20 component ranking for the Deontological model:

![Top components for Deontological model](./blog_bundle_write_up/dla_top_components_PT3_COREDe.png)

_Figure 3b: Top-20 DLA components for the Deontological model (PT3_COREDe). Compare to Figure 3 (Strategic): The same few layers dominate in both, with nearly identical magnitudes. The ranking order is remarkably preserved; this is the same "hardware" being used by a model with very different behavior._

The similarity is striking. The top-20 list is essentially the same components in the same order. If you were hoping to find that moral fine-tuning created dedicated "moral components" or suppressed "selfish" ones, this is the figure that kills that hypothesis.

In both figures the L9\_MLP & L7\_MLP dominate the pro-Cooperate direction and this is true for all 5 models. Similarly, L11\_MLP, L8\_MLP, and L10\_MLP dominate the pro-Defect direction for all 5 models, including in the untrained base model that was never fine-tuned on the IPD game at all.

<!-- TODO: Major - Undersold finding. The fact that cooperation/defection components are identical across all models *including the untrained base* is a striking result. This is evidence that these are pretrained world-knowledge features, not fine-tuning artifacts. Emphasize more -- it's interesting in its own right and has implications for alignment-by-default. -->
Taken together, this points to cooperation/defection features that already existed after pretraining rather than features created by moral fine-tuning. These components look like part of the base model's world knowledge, not artifacts of the adapters.

**The Illusion of Suppression**

I went into this expecting to find that moral fine-tuning "suppresses" selfish components, maybe turning off the pro-Defect circuits or weakening them significantly.

That's not what I found. Comparing the strategic model to the moral models, the largest change in any component was just **0.047**, which is tiny compared to the base magnitudes of 9-10 for L7/L8/L9. That pattern is hard to reconcile with a "suppress the bad neurons" story. The changes from moral fine-tuning were distributed as tiny nudges across many mid-to-late layer MLPs, with no single component showing dramatic suppression or enhancement.

At this point, I had a real puzzle:

- Models behave differently (strategic vs moral)
- But they have nearly identical components (99.9% similar)
- The most "selfish" component didn't get suppressed
- Changes are subtle and distributed

How can nearly identical components produce different moral behaviors?

DLA showed me correlations: which components are associated with cooperation or defection. But correlation isn't causation. The next step was to swap components between models and test whether behavior changed.

<!-- TODO: Minor - Redundant paragraph. This restates the suppression point already made at lines 241-250. Cut. -->
The pro-Defect and pro-Cooperate layers are similarly strong across all five models, including the untrained base and the most cooperative moral model. That is another strike against the "suppress the bad neurons" theory.

### The Mystery of Shallow Alignment

<!-- TODO: Minor - Cut this intermediate mermaid diagram. Keep only the first (high-level overview) and last (complete picture in Discussion). Replace with a short transition sentence. LW readers don't need a roadmap every 2 pages. Also: mermaid diagrams don't render natively on LW -- convert any kept diagrams to images. -->
Before moving on to the next set of experiments, here is where we are in our investigation. We've hit a mystery: models that behave very differently seem to be using almost identical internal machinery. We need new hypotheses to explain it.

```mermaid
graph TD
    classDef setup fill:#f9f9f9,stroke:#666,stroke-width:1px;
    classDef feature fill:#fff2cc,stroke:#d6b656,stroke-width:2px;

    %% Base Setup
    FT[Part 1: RL Fine-tuning Reproductions] --> Eval[Part 1: Evaluation Scenarios]
    Eval --> LL[Part 2: Logit Lens]
    Eval --> DLA[Part 2: Direct Logit Attribution]

    class FT,Eval setup;

    %% The Mystery
    LL -- Shows when decisions stabilize --> Mystery{Part 2: The Mystery of Shallow Alignment:<br>Models behave differently,<br>but components & layer<br>trajectories seem identical}
    DLA -- Shows dominant components,<br>but no suppression --> Mystery

    class Mystery feature;
```

---

## Part 3: Patching & Attention

Ruling out hypotheses led me to activation patching and attention analysis.

#### Setup

<!-- TODO: Minor - Trim Activation Patching toy analogy (swapping car parts). Respect the technical competence of the audience. -->
This is called **activation patching**, and it's like swapping parts between two cars to see what makes them drive differently.

![Activation Patching Concept](./blog_bundle_write_up/fig-activation-patching.png)

Here's how it works:

1. Run two models on the same prompt (say, Strategic and Deontological)
2. Take the activation from one specific component in the Strategic model
3. Replace that same component in the Deontological model with the Strategic one
4. See if the Deontological model now acts more strategically (defects)

If swapping that component changes the behavior, then that component is causally important for the difference between models.

I ran three sets of experiments:

- **Strategic → Deontological**: 3,510 patches
- **Strategic → Utilitarian**: 3,510 patches
- **Deontological ↔ Utilitarian** (bidirectional): 14,040 patches

Total: 21,060 component swaps across all scenarios.

#### Findings

### The Zero Flips Finding

<!-- TODO: Minor - Add a forward reference here to avoid sounding contradictory later: "As we'll see in Part 5, overriding these decisions requires pathway-level interventions, not single nodes." -->
Out of 21,060 component swaps across all my experiments, there were exactly **zero behavioral flips**. Not a single swap changed a model's choice from Cooperate to Defect or vice versa. I went in expecting to find at least a few critical components, maybe a "moral override circuit" that could be disabled or a "selfishness circuit" that could be restored, but the network didn't work that way.

In fact, patching Strategic activations into Deontological models often had the opposite of the intended effect. The mean shift was -0.012, meaning the patches made the model slightly _more_ cooperative on average. Only a quarter of the components pushed behavior toward defection at all, and even the strongest individual effects (around 0.094 logits) weren't nearly enough to change the model's final decision. Even when I tried swapping "minimal circuits" of up to 10 components at once, the behavior held firm.

<!-- TODO: Major - Reduce overclaim. Zero flips across tested single-component and small-circuit patches is strong evidence, but it does not rule out all possible localized circuits under other interventions. -->
This zero-flip result effectively rules out the idea that moral behavior is localized to specific components or small, isolated circuits. Instead, the behavior seems to be distributed across the entire network in a robust, redundant way. There simply is no single "moral neuron" you can disable. 

However, a second pattern in the data hinted at the real answer. When I ran bidirectional patches between the Deontological and Utilitarian models, **78% of the components showed direction-dependent effects**. Patching a specific component from the Deontological model into the Utilitarian one might push the output toward defection (say, +0.02), but patching that _exact same component_ in the reverse direction would push it toward cooperation (-0.03). 

This directional asymmetry is a strong signature of a routing-dependent system. If components had fixed "moral valences," they would push behavior in the same direction regardless of where they were placed. The fact that their influence flips depending on the surrounding model context means that the **interactions between components matter more than the individual components themselves**. As we'll see in Part 4, this asymmetry statistically predicts which pathways get rewired during fine-tuning (r=0.67, p<0.001), serving as the empirical bridge between these null patching results and the network rewiring hypothesis.

<!-- TODO: Major - This paragraph is redundant -- it restates the previous paragraph. Cut. Also: the 78% asymmetry finding is genuinely novel and undersold. Consider giving it more prominence rather than less -- it's the strongest empirical bridge between Parts 3 and 4. -->
The 78% directional asymmetry result means the same component can have opposite effects depending on model context. That is a direct signature of routing dependence and bridges the zero-flip result to the rewiring analysis in Part 4.

**Where Do Patches Matter Most (Even Without Flipping)?**

Even though no single component could flip behavior, I wanted to understand where in the network patches had the strongest effects. Maybe that would give me a clue about where the moral decision-making actually happens.

![Layer-wise patching sensitivity](./blog_bundle_write_up/overview_layer_type_heatmap.png)

_Figure 5a: Average perturbation strength by layer and component type. Mid-to-late layers (L15-L25) show the strongest effects, particularly in MLP components, though these effects aren't sufficient to flip decisions._

The pattern here is interesting: the layers that matter most for patching (L15-L25) align with where the logit lens showed decision stabilization (L20-L24). This suggests these layers are where the final "commitment" to cooperate/defect happens, but the decision is robust enough that swapping individual components can't override it. 

Notice the spike in perturbation strength at Layer 16. While patching here didn't flip the behavior, this locus becomes the central character in our later steering experiments as the primary routing hub.

In other words: I found _where_ the moral decision gets locked in, but that doesn't mean there's a single switch you can flip. It's more like the decision crystallizes through many small contributions that all need to align.

So the next test was whether models that share components are reading different parts of the prompt.

<!-- TODO: Minor - Redundant restatement of zero-flips finding. Already established at line 316. Cut this note. -->
_Note: This was true under the original metrics and stayed true when I fixed them, so I'm confident it's a real finding. Zero behavioral flips across 21,060 patches._

---

### Attention Pattern Analysis: Testing Information Selection

![Attention Analysis Concept](./blog_bundle_write_up/fig-attention.png)

If this were true, I'd expect to see different attention patterns: Deontological models focusing on "opponent's previous action" tokens, Utilitarian models focusing on payoff numbers. I measured final-token attention **weights** aggregated across layers/heads and grouped into three coarse token categories: action keywords, opponent action context, and payoff information.

The result was **99.99% identical** at this coarse level. In the saved comparison table, the largest category-level gap is still below 0.001 (max absolute payoff-category difference is about 0.00064), effectively tiny relative to total attention mass. Both moral frameworks attend to the same broad information. 

This measurement is intentionally narrow: it captures **where** the final decision token looks, not **what vectors** are transmitted by attention outputs. So it cannot rule out differences in attention value transport (`hook_attn_out`) or finer head/position-level structure.

At that point, components and attention looked nearly identical. The next check was whether internal representations, like betrayal, differed across models. That led me to linear probes.

---

<!-- TODO: Major - Downgrade "Identical Representation" claim. Probes only prove the concept is equally *linearly separable* (the geometries for betrayal exist and are equally recoverable), not that the high-dimensional internal representations are precisely identical. -->
### Linear Probes: The "Identical Representation" Finding

The next question was representation: do they encode concepts like betrayal differently? I trained linear classifiers on two key concepts, betrayal detection (binary) and joint payoff prediction (regression), at every layer across all five models.

**Result**: All five models showed nearly _identical_ probe performance.

![Betrayal Probe Comparison](./blog_bundle_write_up/betrayal_probe_comparison.png)

_Figure 7a: Linear probe performance across all 5 models. Betrayal detection barely exceeds chance (~45%); joint payoff prediction is strong (R² = 0.74–0.75) but identical across all models and training regimes. No differences between moral frameworks._

<!-- TODO: Major - Report class balance and explicit chance baseline for betrayal detection; "~45% barely above chance" is ambiguous without base rate/context. -->
- **Betrayal detection**: ~45% accuracy (barely above chance) across _all_ models, including the untrained Base
- **Joint payoff**: R² = 0.74–0.75 identically across all models

This argues against the "different representations" hypothesis. Combined with the attention results, the behavioral gap is more likely computational, in routing dynamics, than representational.

This is consistent with the Platonic Representation Hypothesis: representations tend to converge across training objectives regardless of the fine-tuning goal, meaning the behavioral difference must live in computation (routing), not in features (representations).

So where do the differences come from? That's what led me to the component interaction analysis...

---

## Part 4: Component Interactions

By this point, several explanations were off the table: components are >99.9% similar, attention patterns are 99.99% identical, and linear probes are nearly the same. The remaining candidate was interaction structure.

The question became: what about the interactions _between_ components?

<!-- TODO: Minor - Cut this intermediate mermaid diagram (same rationale as the Part 2 one). Replace with a transition sentence. -->
```mermaid
graph TD
    classDef setup fill:#f9f9f9,stroke:#666,stroke-width:1px;
    classDef feature fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef test fill:#e1d5e7,stroke:#9673a6,stroke-width:1px;

    %% Base Setup
    class Mystery feature;

    %% Hypothesis Testing
    Mystery[Part 2: The Mystery] --> AP[Part 3: Activation Patching]
    Mystery[Part 2: The Mystery] --> Attn[Part 3: Attention Analysis]
    Mystery[Part 2: The Mystery] --> Probes[Part 3: Linear Probes]

    class AP,Attn,Probes test;

    %% Null Results -> New Hypothesis
    AP -- "0 Behavioral Flips:<br>Single components don't control behavior" --> Interact[Part 4: Component Interactions]
    Attn -- "99.99% Identical:<br>They select the same info" --> Interact
    Probes -- "Identical Probes:<br>They represent concepts the same way" --> Interact

    class Interact test;
```

---

### The Wiring Diagram: How Identical Parts Create Different Behaviors

#### Setup

After ruling out component differences and coarse attention-weight differences, I had one hypothesis left: maybe the components keep the same inventory but differ in effective interaction structure?

![Component Interaction Analysis Concept](./blog_bundle_write_up/fig-interactions.png)

<!-- TODO: Minor - Trim component interaction analogy ("imagine two computers built with the exact same parts..."). -->
Think of it this way: two systems can use the same parts but produce different behavior if signal flow between those parts changes in practice.

I measured **component interactions** by correlating component activation magnitudes across the 15 evaluation prompts. In code this uses **Pearson correlation** by default (`mech_interp/component_interactions.py`). With only `n=15` prompts, I treat threshold counts as effect-size bins, not formal significance tests.

The analysis:

- Track activation magnitudes for all 52 components (26 attention layers + 26 MLP layers)
- Compute Pearson correlation for every component pair across 15 scenarios
- Compare these correlation patterns between Deontological and Utilitarian models
- Bin pathway differences by absolute correlation shift: `|Δr| >= 0.3`, `0.5`, `0.7`

#### Findings

Before looking at where the models differ, it helps to see what the raw interaction structure looks like for each model individually. Here are the full 52×52 correlation matrices:

<!-- TODO: Minor - Figures 6a-b (raw correlation matrices) could move to appendix. The difference heatmap (6c) alone tells the story -- showing both full matrices before the diff adds length without proportional insight. -->
![Deontological model correlation matrix](./blog_bundle_write_up/correlation_matrix_PT3_COREDe_chronological.png)

![Utilitarian model correlation matrix](./blog_bundle_write_up/correlation_matrix_PT3_COREUt_chronological.png)

_Figures 6a-b: Component interaction matrices for the Deontological (top) and Utilitarian (bottom) models, with components ordered chronologically by layer. Red indicates positive correlation (components activate together), blue indicates negative correlation (one activates when the other doesn't). Both matrices show similar large-scale structure: strong positive correlations along the diagonal (neighboring layers work together) and block patterns in early vs. late layers. But look carefully at the mid-layer region (around L8-L17): the correlation signs and magnitudes differ between models._

At a glance, these look very similar. That's the point. The global structure of component interactions is preserved: both models have similar block-diagonal patterns, similar early-layer clusters, similar late-layer clusters. The differences are subtle and distributed, which is exactly what you'd expect from "same components, different effective routing."

Now, to make those differences visible, here's what you get when you subtract one matrix from the other:

![Correlation difference heatmap](./blog_bundle_write_up/interaction_diff_Deontological_vs_Utilitarian_chronological.png)

_Figure 6c: Correlation differences between Deontological and Utilitarian models (52×52 matrix, chronological ordering). Hot spots show pathways with different effective interaction structure, with notable clusters around L1_MLP, L12-L13, and L16-L17, the same deep-layer region that shows up in the steering experiments later._

#### The Discovery: Network Rewiring

In this post, **"network rewiring" means changed effective routing/interaction structure, not literal topology changes**. The models' components are still >99.9999% similar and coarse attention-weight patterns are nearly unchanged, but their interaction patterns diverge broadly. Out of 1,326 component pairs, Pearson `|Δr|` counts are 541 (`>=0.3`), 251 (`>=0.5`), and 94 (`>=0.7`), indicating widespread medium-to-large effect-size shifts.

As a method-sensitivity check using the same saved activations, Spearman gives similar totals: 565 (`>=0.3`), 273 (`>=0.5`), and 103 (`>=0.7`). Given the small prompt set (`n=15`), I interpret this as robust correlational evidence for distributed interaction changes in this setting, not as definitive mechanism proof.
<!-- TODO: Major - Tone/calibration: "This finally explained our mystery" is too strong. Rephrase as "best-supported working explanation in this model-task setting." -->

An earlier, less refined version of this analysis suggested that early layers (specifically L2_MLP) might act as a singular routing switch. However, the updated interaction tables weakened that claim considerably. The connection between L2_MLP and L9_MLP turned out to be relatively weak (a rank around 858 out of 1326). Instead, the largest differences were distributed deeply across the network, dominated by connections like L16_MLP to L1_MLP, and L19_ATTN to both L21_MLP and L24_MLP. The mechanistic picture that emerged wasn't a single, decisive early-layer switch, but a **distributed network-level reconfiguration** organized around recurring hubs like L19_ATTN, L1_MLP, and L17_MLP.

To ensure this sprawling rewiring wasn't just statistical noise, I cross-referenced it with the activation patching data from earlier. The pathway differences correlated robustly with the behavioral asymmetry I found during patching (r=0.67, p<0.001), indicating that components with larger correlation shifts genuinely exhibited stronger directional asymmetry. <!-- TODO: Major - Add context for Chen et al. citation. Is their setup similar enough (model size, task type, fine-tuning method) to count as converging evidence, or is it a loosely analogous finding in a different domain? Without this, "external validation" oversells the connection. -->
This "same nodes, different edges" pattern also mirrors recent independent findings by Chen et al. (2025), who observed a remarkably similar phenomenon where fine-tuning altered network edges while leaving the nodes highly similar. Finding two independent research threads converging on the same conclusion provides a strong vote of external validation.

<!-- TODO: Minor - Redundancy. This paragraph restates the r=0.67 stat from the paragraph above it and the "same parts, different wiring" metaphor for ~the 4th time. Cut or merge into the paragraph above. -->
Over 40% of component pairs show large (`|Δr| >= 0.3`) interaction shifts between models that otherwise share near-identical components. The moral difference appears in effective routing ("org chart"), not inventory ("employees"). This interaction-shift signal also correlates strongly with the directional asymmetry from Part 3 (r=0.67, p<0.001), which supports a structural mechanism.

Based on these experiments, moral fine-tuning appears to work through a mechanism I wasn't initially expecting. It doesn't create new "moral" components, it doesn't suppress "selfish" ones, and it doesn't even change the information the model chooses to look at. Instead, the training simply re-routes how existing components pass information to one another. Same Lego bricks, different wiring diagram. 

It's important to caveat that this specific analysis relies on correlation patterns rather than strict causal evidence. While the statistical correlation with behavioral asymmetry strongly supports this interpretation, it doesn't definitively _prove_ that these wiring adjustments are the sole cause of the behavioral shift. Additionally, these experiments were constrained to a single model (Gemma-2-2b-it) and a single task (the Iterated Prisoner's Dilemma), so it's possible this pattern may not perfectly generalize to other environments. 

Even with those caveats, this is strong evidence for a "network rewiring" mechanism. Much of the prior framing focuses on identifying particular circuits or suppressed activations; here, behavior shifts while the component set stays largely fixed and the connectivity pattern changes. (For a quick check confirming that L2_MLP wasn't dramatically retrained, which strengthens the rewiring interpretation, see [Appendix A](#appendix-a-the-frankenstein-test).)

---

### New Hypothesis: Network Rewiring

With the discovery of differing component interactions, our roadmap looks like this. The null results from patching, attention, and probes forced us to look at interactions, creating a new hypothesis that we now need to test with causal interventions.
<!-- TODO: Minor - Concision pass. This is another full roadmap diagram; keep one canonical roadmap and convert repeated ones into short transition text. -->

```mermaid
graph TD
    classDef setup fill:#f9f9f9,stroke:#666,stroke-width:1px;
    classDef feature fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef test fill:#e1d5e7,stroke:#9673a6,stroke-width:1px;

    %% Base Setup
    FT[Part 1: RL Fine-tuning Reproductions] --> Eval[Part 1: Evaluation Scenarios]
    Eval --> LL[Part 2: Logit Lens]
    Eval --> DLA[Part 2: Direct Logit Attribution]

    class FT,Eval setup;

    %% The Mystery
    LL -- Shows when decisions stabilize --> Mystery{Part 2: The Mystery of Shallow Alignment:<br>Models behave differently,<br>but components & layer<br>trajectories seem identical}
    DLA -- Shows dominant components,<br>but no suppression --> Mystery

    class Mystery feature;

    %% Hypothesis Testing
    Mystery --> AP[Part 3: Activation Patching]
    Mystery --> Attn[Part 3: Attention Analysis]
    Mystery --> Probes[Part 3: Linear Probes]

    class AP,Attn,Probes test;

    %% Null Results -> New Hypothesis
    AP -- "0 Behavioral Flips:<br>Single components don't control behavior" --> Interact[Part 4: Component Interactions]
    Attn -- "99.99% Identical:<br>They select the same info" --> Interact
    Probes -- "Identical Probes:<br>They represent concepts the same way" --> Interact

    class Interact test;

    %% The Breakthrough
    Interact -- "Discovers Network Rewiring:<br>Different correlation patterns" --> Causal{Part 5: Causal Experiments}

    class Causal causal;
```

---

## Part 5: Causal Experiments

The interaction analysis established that models with nearly identical components differ substantially in interaction structure. But correlation matrices don't prove causation. To test whether routing differences actually drive behavior, I ran two causal experiments: activation steering and path patching.

Both experiments converge on a phenomenon I'll call the **Washout Effect**: early-layer steering interventions are corrected by subsequent layers and fail to change behavior, while late-layer interventions (L16/L17) persist because insufficient network remains to override them.

### Experiment 1: Finding the Real Switches Through Steering

Where are the routing switches? I tested this by "steering" activations, adding directional vectors to component activations to see which components had the most control over behavior.

<!-- TODO: Major - Specify steering vector calculation aggregation method (e.g., average difference over the 15 test prompts at a specific token position). -->
The method: compute a "steering vector" (the difference between moral and strategic model activations), then add this vector at different strengths to various layers during the forward pass.

![Steering Vector Concept](./blog_bundle_write_up/fig-steering.png)

<!-- TODO: Minor - Trim the car steering analogy. -->
Imagine the model is a car driving toward 'Defect'. A steering vector is like reaching into the mechanism at Layer 16 and physically yanking the steering wheel toward 'Cooperate'. If the car actually turns, we know that mechanism controls the wheels.

If a component is a routing hub, steering its activations should proportionally shift behavior.

I tested steering at multiple layers, including L2_MLP (the original hypothesis) and others throughout the network.

![L2 MLP Steering (Minimal Effect)](./blog_bundle_write_up/steering_sweep_PT2_COREDe_L2_mlp.png)

L2_MLP steering: **+0.56% cooperation increase**. Basically nothing.

But then I tried steering deeper layers:

![L17 MLP Steering (Strong Effect)](./blog_bundle_write_up/steering_sweep_PT2_COREDe_L17_mlp.png)

- **L16_MLP steering**: +26.17% cooperation (46x more effective than L2)
- **L17_MLP steering**: +29.58% cooperation (52x more effective than L2)

<!-- TODO: Minor - Explicitly state that L16/17 were identified as correlation difference hubs in Part 4 before testing them via steering. Ensure a smooth transition between correlation mapping and L16/17 testing. -->
**This changed the working hypothesis.** The routing switches exist, but they're in layers 16-17, not layer 2. The early interaction analysis pointed to L2 because of correlation patterns, but causal interventions showed the strongest switches are much deeper in the network, in the final third of the model where decisions are finalized.

This makes intuitive sense: early layers might show correlation differences because they're transmitting signals that get amplified later, but the actual routing control happens in the deeper layers where the model is making its final decision.

To make this comparison concrete, here are all the steering sweeps overlaid on a single plot:

![All steering sweeps overlaid](./blog_bundle_write_up/comparison_sweep_overlay.png)

_Figure 8a: All steering sweeps overlaid. Each line shows cooperation rate as a function of steering strength for a different layer/component. The L16 and L17 MLP curves (steep positive slopes) tower over the flat L2 MLP line. This single plot captures the key finding: late-layer MLPs are the real routing switches._

The overlay shows the contrast clearly. The L16/L17 MLP curves are steep and responsive: small changes in steering strength produce large behavioral shifts. The L2 MLP curve is essentially a flat line. The routing switches are in deep layers, not early layers.

For a more compact summary across models, here's a heatmap of effect sizes:

![Effect size heatmap by layer and model](./blog_bundle_write_up/effect_size_heatmap.png)

_Figure 8b: Effect sizes for steering at each tested layer/component, across the Strategic (PT2) and Deontological (PT3) models. L17_MLP has the largest effect in both models (1.39 and 1.27 respectively), followed by L16_MLP. L8_MLP and L9_MLP, the components that dominate DLA contributions, have near-zero steering effect._

This heatmap highlights an important distinction. L8 and L9 MLPs are the biggest contributors in the DLA analysis (Figure 3); they carry the strongest cooperation/defection signals. But they have near-zero steering effect sizes. Meanwhile, L16 and L17 MLPs are modest DLA contributors but have massive steering effects. The components that _encode_ the signal are not the same components that _control_ which signal wins. That's a crucial mechanistic insight.

#### Seeing Steering Through the Logit Lens

To understand _how_ steering changes the model's internal processing (not just the final output), I combined the steering experiments with the logit lens technique from earlier. This lets us trace the layer-by-layer decision trajectory under different steering interventions.

![Logit Lens Concept](./blog_bundle_write_up/fig-steering-logit-lens2.png)

This plot reveals something the behavioral metrics alone can't show: _where_ in the network steering takes hold. When you steer at L16 or L17, the trajectory visibly diverges from baseline in the late layers and stays diverged through the final output. When you steer at L8 or L9, the trajectory briefly shifts but then reconverges with baseline; the effect washes out.

![Bidirectional steering trajectories for Deontological model](./blog_bundle_write_up/overlay_bidirectional_PT3_COREDe_CC_temptation.png)

_Figure 8c: Layer-by-layer logit trajectories for the Deontological model on the CC_temptation scenario, under bidirectional steering at multiple layers. Solid lines show +2.0 steering (toward cooperation); dashed lines show -2.0 steering (toward defection). The baseline (black) is the unsteered trajectory. Late-layer steering (L16, L17) produces visible divergence from baseline in the final layers, while early-layer steering (L8) has minimal lasting effect._

Even more telling is comparing how the same steering intervention affects different models:

![L16 MLP steering comparison between Strategic and Deontological](./blog_bundle_write_up/overlay_bidirectional_both_models_L16_MLP_CC_temptation.png)

_Figure 8d: L16 MLP bidirectional steering comparison between the Strategic (red) and Deontological (blue) models on the CC_temptation scenario. Both models respond to L16 steering, with the Strategic model's trajectory pulled toward cooperation under positive steering and the Deontological model's pulled toward defection under negative steering. L16 MLP acts as a bidirectional switch in both models._

This is strong evidence that L16 MLP is a genuine routing hub: steering it in opposite directions produces opposite behavioral effects, and this works consistently across both models.

<!-- TODO: Minor - Tone down rhetorical emphasis ("single most important figure") to keep calibration tight for skeptical readers. -->
And here's what I think is the single most important figure from this entire investigation:

![Early vs late steering washout](./blog_bundle_write_up/KEY_early_vs_late_washout.png)

_Figure 8e: The washout experiment. Both curves show +2.0 cooperative steering applied to the Strategic model on CC_temptation. Red: steering at L8 MLP (early layer). Green: steering at L17 MLP (late layer). The L8 steering creates a brief effect but washes out by layer 16. Subsequent layers "overwrite" the early intervention. The L17 steering persists through to the final output because there aren't enough remaining layers to override it._

This figure summarizes the mechanism. Early-layer steering washes out because the network has 15+ subsequent layers that can "correct" the perturbation; in effect, distributed downstream processing overrules the single-layer change. Late-layer steering persists because there isn't enough network left to override it.

This resolves the apparent paradox: L8_MLP has the strongest DLA attribution but near-zero causal control, while L16/L17 have smaller DLA contributions but much larger behavioral impact.

This also explains why single-component activation patching produced zero behavioral flips across 21,060 patches. Even if you perturb the right component, the distributed processing in subsequent layers compensates. Only pathway-level interventions (replacing multiple consecutive layers) or late-layer steering (where there's no room for compensation) can actually flip behavior.

The **Washout Effect** is simple: early-layer interventions are overwritten by many downstream layers, while late-layer interventions (L16/L17) persist because too little network remains to cancel them. Components that encode signals are not always the components that control outcomes.

### Experiment 2: Path Patching: Testing Pathway-Level Causality

<!-- TODO: Major - Clarify what's being replaced. Are you replacing full residual stream values (attention + MLP outputs accumulated), or just MLP/attention outputs separately injected? The distinction matters for reproducibility and interpretation. -->
Single-component patches failed 21,060 times. But what if the problem was _scope_, not _location_? Instead of swapping one component at a time, I tried replacing entire _pathways_, consecutive chunks of the residual stream from one model dropped into another. In a transformer, information flows through the residual stream layer by layer (L2 → L3 → L4 → ...), and each layer reads from and writes to this shared stream. Path patching replaces those activations for multiple consecutive layers at once.

Scope note: this provides causal evidence for the tested interventions (L2→L9 path family, tested model pairs, and IPD prompts), not universal proof across all models, tasks, or pathways.

I tested three types of path: full residual (attention + MLP), MLP-only, and attention-only. Then I did progressive patching from L2→L2, then L2→L3, then L2→L4, and so on, to find where the effect saturates.

In implementation terms, attention-only patching intervenes on `hook_attn_out`, so causal effects can appear even when coarse final-token attention-weight summaries look similar.

![Progressive Path Patching](./blog_bundle_write_up/progressive_patch_comparison.png)

_Figure 9a: Progressive path patching from the Deontological model into the Strategic model. The x-axis extends the patch endpoint from L2 through L9; the y-axis shows the resulting cooperation change. The full residual path (blue) reaches +61.73% by L9, but most of the effect saturates by L5. Attention-only paths (green) account for 34.4%, roughly 3× the MLP-only contribution (orange, 11.2%)._

The contrast with single-component patching is stark: pathway-level interventions produced effects **61.7× larger** than any individual patch. This makes intuitive sense in light of the Washout Effect: single components get overruled by subsequent layers, but entire pathways accumulate enough momentum to survive the downstream correction.

![Path Decomposition](./blog_bundle_write_up/component_comparison_PT3_COREDe_to_PT2_COREDe.png)

_Figure 9b: Component-type decomposition of path patching effects (Deontological → Strategic). Attention pathways dominate at 34.4%, roughly 3× the MLP contribution of 11.2%._

The most telling finding is _which_ pathways matter. Attention pathways contribute about 3× more than MLP pathways. Attention heads are the "routers." They decide which information from early layers gets copied forward to be read by later layers. The fact that attention dominates suggests moral fine-tuning primarily changes _where information flows_ (routing) rather than _how it's transformed_ (MLP computation).

Attention pathways exert roughly 3× the causal influence of MLP pathways. In this setting, moral fine-tuning seems to change _where information flows_ (attention routing) more than _how it is transformed_ (MLP computation).
<!-- TODO: Minor - Trim repetition. This sentence restates the same 3x attention-dominance claim from the paragraph above. Keep one tighter version. -->

#### What This All Means

I started this investigation expecting a single moral switch. The causal experiments instead point to a distributed routing system. Control points are deep (L16/L17 MLPs), pathway-level interventions are far stronger than single-component patches (61.7×), and attention pathways carry about 3× more causal weight than MLP pathways. The working interpretation is that moral fine-tuning mainly reconfigures information flow.

That routing is distributed in depth: early layers (L2–L5) transmit signals through attention-heavy pathways, while deeper layers (L16–L17) act as final gating points for action selection. So the behavior comes from coordinated routing, not any single component.

---

### Checking My Work: Does This Actually Predict Behavior?

After running all these analyses, I needed to answer a basic question: do these internal measurements actually predict what the models do when you sample from them? Because if the mechanistic findings don't align with observable behavior, they're not telling us much.

So I ran a validation check: for each model and scenario, I compared two things:

1. What the sequence probability metric says (internal measurement)
<!-- TODO: Major - n=9 is very low for a "100% agreement" claim. You can't meaningfully distinguish 92.7% vs 92.6% with 9 samples -- that match is luck, not validation. Be upfront about this limitation or increase the sample count. -->
2. What actually gets generated when you sample 9 times (real behavior)

Validation summary:

- **Agreement rate**: 100%. Every single model×scenario combination matched perfectly.
- **Strategic model**: Internal metric says 99.96% defection → actual sampling: 100% defection ✓
- **Deontological model**: Internal says 99.97% cooperation → actual: 100% cooperation ✓
- **Utilitarian model**: Internal says 92.7% cooperation → actual: 92.6% cooperation ✓

Statistical tests confirm the models are genuinely different from each other (p < 0.00005 for strategic vs moral comparisons). So the mechanistic findings (DLA, patching, attention patterns, component interactions) are measuring something real that corresponds to how these models actually behave.

I'm mentioning this because it's easy to fall into the trap of analyzing internal model states without checking if they predict anything meaningful. The alignment check gives me confidence that the "network rewiring" story I'm telling actually corresponds to a real difference in how these models work.


---

## Discussion

### The Complete Picture

Here is the full roadmap of the investigation, bringing all the experiments and their conclusions together into a single view.
<!-- TODO: Minor - Concision. If a full roadmap is retained earlier, cut this repeated mermaid block (or vice versa). -->

```mermaid
graph TD
    classDef setup fill:#f9f9f9,stroke:#666,stroke-width:1px;
    classDef feature fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef test fill:#e1d5e7,stroke:#9673a6,stroke-width:1px;
    classDef causal fill:#dae8fc,stroke:#6c8ebf,stroke-width:1px;
    classDef conclusion fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
    classDef validation fill:#f5f5f5,stroke:#b3b3b3,stroke-width:1px,stroke-dasharray: 5 5;

    %% Base Setup
    FT[Part 1: RL Fine-tuning Reproductions] --> Eval[Part 1: Evaluation Scenarios]
    Eval --> LL[Part 2: Logit Lens]
    Eval --> DLA[Part 2: Direct Logit Attribution]

    class FT,Eval setup;

    %% The Mystery
    LL -- Shows when decisions stabilize --> Mystery{Part 2: The Mystery of Shallow Alignment:<br>Models behave differently,<br>but components & layer<br>trajectories seem identical}
    DLA -- Shows dominant components,<br>but no suppression --> Mystery

    class Mystery feature;

    %% Hypothesis Testing
    Mystery --> AP[Activation Patching]
    Mystery --> Attn[Attention Analysis]
    Mystery --> Probes[Linear Probes]

    class AP,Attn,Probes test;

    %% Null Results -> New Hypothesis
    AP -- "0 Behavioral Flips:<br>Single components don't control behavior" --> Interact[Component Interactions]
    Attn -- "99.99% Identical:<br>They select the same info" --> Interact
    Probes -- "Identical Probes:<br>They represent concepts the same way" --> Interact

    class Interact test;

    %% The Breakthrough
    Interact -- "Discovers Network Rewiring:<br>Different correlation patterns" --> Causal{Causal Experiments}

    class Causal causal;

    %% Testing the Mechanism
    Causal --> Exp1["Part 5, Exp 1: Steering Sweep<br>(Shifting Activations)"]
    Causal --> Exp2["Part 5, Exp 2: Path Patching<br>(Replacing Pathways)"]

    class Exp1,Exp2 causal;

    %% Final Conclusions
    Exp1 -- Finds deep routing<br>hubs at L16/L17<br>(Washout Effect) --> Conclusion(((Conclusion:<br>Deep Attention-Mediated Routing)))
    Exp2 -- Provides pathway-level<br>causal evidence & attention dominance --> Conclusion

    %% Validation
    Val[Part 5: Validation Check] -. Confirms sequence metrics<br>match actual behavior .-> Conclusion

    class Conclusion conclusion;
    class Val validation;
```

### Implications for AI Safety

The clearest safety implication is that alignment may work as a bypass rather than deletion. Moral fine-tuning doesn't seem to remove strategic self-interest capacity; it reroutes decision flow around those capabilities. L8_MLP, the "Selfish Neuron," is still present and active at similar magnitude. In the aligned models, that signal is less influential at the final decision point.

<!-- TODO: Major - Waluigi Effect Overreach. Waluigi strictly implies that training *strengthens* the inverse persona. This data just proves the inverse persona isn't deleted (shallow alignment). Stop short of claiming the selfish persona was actually sharpened. -->
<!-- TODO: Major - The Waluigi connection is introduced in the Background (line 30) but only briefly revisited here. Your findings actually provide a more nuanced picture than the original Waluigi framing: the "Waluigi" (selfish behavior) wasn't *sharpened* by training -- it was already there in the base model (DLA finding). Training just routes around it. Make this thread more explicit throughout the Discussion. -->
This is consistent with the Waluigi Effect framing: training hard on cooperation may not erase defection-related structure. Moral and strategic behaviors appear to share substrate, with routing determining which one dominates at inference time.

Four concrete safety implications follow from this:

1. **Node-level safety audits are insufficient.** Searching for "bad neurons" misses the mechanism. More than 40% of component interaction pairs are rewired between models; the difference lives in connectivity, not in which components exist.

2. **The original wiring remains available.** Because L8_MLP and other selfish components remain fully intact and operational, the selfish routing isn't deleted; it's just not currently the active path. OOD inputs, adversarial prompts, or further fine-tuning could restore it.

3. **Attention mechanisms are the alignment bottleneck.** The 3× dominance of attention pathways in path patching has a direct implication: attention heads are where the routing decisions live, making them the most productive target for alignment audits and robust interventions.

4. **The bypass switch is locatable.** Our experiments show which components would need to be disrupted: the L16/L17 MLP routing hubs. An adversarial intervention targeting these layers (through prompt injection, targeted fine-tuning, or activation manipulation) would bypass the moral routing and restore full strategic behavior. The vulnerability is specific and mechanistically grounded.
<!-- TODO: Major - Downgrade to hypothesis. This specific adversarial bypass pathway is plausible but not directly tested in the current experiments. -->

<!-- TODO: Major - Missing safety implication. The DLA finding that cooperation/defection features exist in the *base* pretrained model (before any fine-tuning) has implications for the "alignment tax" discussion and alignment-by-default. If prosocial behavior is already baked in from pretraining on human text, that's worth calling out as a 5th implication. -->

If you made it this far, thanks for reading! This has been a fun deep dive into how neural networks implement moral reasoning, and I learned a lot about mechanistic interpretability along the way.
<!-- TODO: Minor - Optional LessWrong tone edit: cut/shorten this personal sign-off to end on empirical claims + uncertainty. -->

---

### Caveats and Confidence

Before wrapping up, I want to be clear about what I'm confident in and what I'm not.

**What I'm confident about**:

- The models genuinely behave differently (99.96% defection vs 99.97% cooperation)
- Internal measurements align with actual behavior (100% agreement in validation)
- The major patterns: L8/L9 MLPs doing opposite things, broad interaction-level rewiring, and 99.99% attention similarity
- Statistical significance is strong (p < 0.00005 for model separation)
- **Causal evidence exists for selected hubs/pathways** (steering at L16/L17 and L2→L9 path patching show large effects), but this is not full proof of the entire network-rewiring mechanism.
- **Routing switches exist in deep layers** (L16/L17 MLPs show 50x more steering effect than L2_MLP)
- **Attention pathways dominate routing** (3x more effective than MLP pathways in path patching)

**Open Questions**:

- The mechanistic story I'm telling ("network rewiring through deep attention-mediated pathways") is one interpretation, but there could be other ways to explain the same patterns
- This is one model (Gemma-2-2b-it) on one task (iterated prisoner's dilemma). I don't know if this generalizes.
- While I've demonstrated causality for specific pathways (L2→L5) and hubs (L16/L17), there are likely other important pathways I haven't tested. The L16/L17 hubs emerged from steering experiments; there could be other hubs I haven't discovered.
- I'm still learning these techniques, so there might be better ways to analyze this that I haven't tried

**What this suggests**: In this model-task setting, moral fine-tuning appears to change routing between existing components more than component identity or attention targets. The components stay similar while connectivity patterns shift. Whether this generalizes beyond Gemma-2-2b-it on IPD remains open.

---

## Appendix
### Appendix A: The Frankenstein Test
<!-- TODO: Minor - Consider cutting this entirely. It's a null result from an early intuition ("Frankenstein test") that isn't strictly necessary since the formal steering sweeps in Part 5 cleanly find L16/L17 anyway. -->

This section describes an experiment I ran as an investigative detour. It didn't yield consistent results, but it helped me identify the correct direction for the steering experiments and is included here for completeness.

**Motivation**: The interaction analysis initially flagged L2_MLP as a potential routing hub based on correlation patterns. Before moving to activation steering across all layers, I wanted to test this specific hypothesis mechanically: if L2_MLP is the switch that routes information toward cooperation vs defection, transplanting its weights from one model to another should shift behavior.

![Frankenstein Experiment Concept](./blog_bundle_write_up/fig-frankenstein.png)

The experiment: take the L2_MLP LoRA weights from the Deontological model and surgically replace the L2_MLP weights in the Strategic model. I tested four transplant combinations:

1. Strategic + Deontological_L2 → Expect cooperation increase
2. Deontological + Strategic_L2 → Expect cooperation decrease
3. Deontological + Utilitarian_L2 → Expect slight change
4. Utilitarian + Deontological_L2 → Expect cooperation increase

![Frankenstein Results](./blog_bundle_write_up/frankenstein_comparison.png)

**Results**: 1 out of 4 hypotheses worked as expected. The Deontological→Utilitarian transplant showed a massive +71.31% increase in cooperation. The other three produced either minimal effects or effects in unexpected directions.

**What this told me**: L2_MLP weights alone aren't sufficient for consistent behavioral control. Getting one strong effect out of four suggested the real routing switches were somewhere else, which motivated the broader steering sweep across all layers that revealed L16/L17 as the actual deep routing hubs.

### Appendix B: Was L2_MLP Heavily Retrained?

Because L2_MLP was an early candidate hub in prior passes, I also wanted to check: did the moral models just massively retrain this component? That would be a simpler explanation than "network rewiring."

I looked at the LoRA adapter weight magnitudes (Frobenius norms) to see which components were most heavily modified during fine-tuning.

![L2_MLP Comparison](./blog_bundle_write_up/l2_mlp_comparison.png)

**The answer: No.** L2_MLP wasn't particularly heavily modified:

- It ranks in the 11-27th percentile for modification magnitude
- That means 73-88% of components were modified MORE than L2_MLP
- L8 and L9 MLPs (the components that encode cooperation/defection) were actually modified more than L2

![Model Weight Similarity](./blog_bundle_write_up/adapter_similarity_heatmap.png)

All models are also 99%+ similar in weight space, which matches the 99.99% attention similarity found earlier. The routing differences aren't coming from massively retraining specific components; they're coming from lighter modifications that change how components connect to each other. This strengthens the network rewiring story.

### Appendix C: A Note on Measurement
<!-- TODO: Major - Move this buried metric correction to the main methodology section (Part 1 or 2). LW loves open epistemic updates. -->

Methodology note: after the initial analysis in early February, I realized the metric I used was not aligned with the behavior-level quantity I cared about.

**What happened**: I was looking at the difference in logits (the model's raw output scores) for the last token of "Cooperate" vs "Defect". That seems reasonable at first glance. But when the model actually generates text, it's choosing between continuing with "action1" or "action2" as full sequences, not one token position. Because of tokenization, those are multi-token sequences.

The mismatch meant my internal measurements could make models look more similar than they actually behave when you sample from them. Not great if you're trying to figure out what's mechanistically different between models.

**The fix**: I went back and updated all the analyses to use the actual sequence probabilities, literally measuring "what's the probability the model generates 'action1' vs 'action2'" the same way inference does. Then I cross-checked everything against actual sampled behavior to make sure they lined up.

**Did this change the findings?** Partly. The high-level story still holds (models differ mainly in coordination/routing, not raw component identity), but some interaction-level specifics changed after reruns:

- Perfect agreement between the metric and actual sampling (100% alignment)
- Models are still clearly separated (strategic model defects 99.96% of the time, moral models cooperate >92%)
- The old "single L2_MLP routing switch" framing is not strongly supported in the refreshed rankings
- Pathway differences are much more widespread than the earlier 29-pathway estimate
- Network rewiring hypothesis: still holds

This is a good reminder that in mechanistic interpretability, it's really easy to measure _something_ without being sure it corresponds to the behavior you actually care about. Always validate against the real outputs.
