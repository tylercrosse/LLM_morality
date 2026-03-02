# Original Research Plan

## Task 0: Merge LoRA Adapters into Base Model

**Goal:** Create standalone model checkpoints that TransformerLens can hook into cleanly.

**Why:** TransformerLens hooks into specific named modules (`blocks.attn.hook_q`, etc.). When LoRA adapters are active, the forward pass goes through `lora_A` and `lora_B` matrices that sit alongside the base weights. This can cause hook points to miss the actual computation, or give you activations that don't reflect the full adapted behavior. Merging bakes the LoRA deltas directly into the weight matrices, giving you a "normal" model that TransformerLens handles correctly.

**Deliverable:** Four saved models:

- `gemma-2b-base` (unchanged, for reference)
- `gemma-2b-strategic` (game payoffs only)
- `gemma-2b-deont` (deontological reward)
- `gemma-2b-util` (utilitarian reward)
- `gemma-2b-hybrid` (game + deontological)

**Code sketch:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

def merge_and_save(base_path, lora_path, output_path):
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, lora_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_path)
    return merged

# Repeat for each variant
merge_and_save("google/gemma-2-2b-it", "./checkpoints/strategic", "./merged/strategic")
```

---

## Task 1: Build Evaluation Prompt Set

**Goal:** Create a controlled set of IPD scenarios that isolate the behavioral differences you want to study.

**Why:** Mechanistic interpretability requires comparing model internals across carefully matched inputs. If your prompts vary in irrelevant ways (length, phrasing, context), you'll see activation differences that reflect surface variation rather than the moral reasoning you care about. A clean prompt set lets you attribute internal differences to the actual decision-making.

**Deliverable:** A dataset of 10-20 prompts covering:

- **Baseline cooperation scenario:** Opponent cooperated last round, cooperation is "obviously correct" for all moral frameworks
- **Temptation scenario:** Opponent cooperated, defection would yield higher personal payoff (this is where strategic vs. moral models should diverge)
- **Retaliation scenario:** Opponent defected last round (tests whether deontological "no betrayal" rule only applies to cooperators)
- **First move scenario:** No history yet (tests default disposition)

**Structure suggestion:**

```python
prompts = [
    {
        "id": "temptation_01",
        "text": "You are playing Iterated Prisoner's Dilemma...[payoff matrix]...Last round: You cooperated, they cooperated. What do you play?",
        "expected_strategic": "action2",  # defect (higher payoff)
        "expected_deont": "action1",       # cooperate (no betrayal)
        "expected_util": "action1",        # cooperate (higher joint payoff)
        "scenario_type": "temptation"
    },
    # ...
]
```

---

## Task 2: Logit Lens — Identify the Decision Layer

**Goal:** Find the layer where the model's "mind is made up" about cooperate vs. defect.

**Why:** Transformer computation is iterative — early layers build representations, later layers make decisions. The logit lens decodes the residual stream at each layer using the final unembedding matrix, showing you how the output distribution evolves. Finding the "critical layer" tells you where to focus your circuit analysis. If the decision is made at layer 12, you don't need to spend time analyzing layers 20-26.

**Method:**

1. Run each model on a temptation scenario
2. At each layer, project the residual stream to vocabulary space
3. Track the logit difference: `logit(action2) - logit(action1)`
4. Plot this across layers for each model variant

**What to look for:**

- **Strategic model:** The defection logit should rise and stabilize at some layer (call this the "decision layer")
- **Moral models:** Either (a) the cooperation logit dominates throughout, or (b) defection rises early but gets "corrected" at a later layer

**Interpretation:** If moral models show a correction pattern (defection rises then falls), this suggests the moral fine-tuning added a "veto circuit" rather than removing selfish computation entirely. This directly addresses RQ1.

**Code sketch:**

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained_no_processing("./merged/strategic")
logits, cache = model.run_with_cache(prompt)

# Get residual stream at each layer
per_layer_logits = []
for layer in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]  # last token position
    layer_logits = model.unembed(model.ln_final(resid))
    per_layer_logits.append(layer_logits)

# Extract logit difference for action tokens
action1_id = model.tokenizer.encode("action1")[0]
action2_id = model.tokenizer.encode("action2")[0]
logit_diff = [l[0, action2_id] - l[0, action1_id] for l in per_layer_logits]
```

---

## Task 3: Direct Logit Attribution by Head

**Goal:** Identify which attention heads contribute most to the cooperate/defect decision.

**Why:** The logit lens tells you _when_ the decision happens but not _what components_ are responsible. Direct logit attribution decomposes the final logit into additive contributions from each attention head and MLP layer. This gives you a ranked list of "who's responsible for this output" — your candidate selfish/moral heads.

**Method:**

1. Run the model with caching
2. For each attention head, compute its contribution to the final residual stream
3. Project each contribution through the unembedding to get its effect on the action token logits
4. Compare these attributions across model variants

**What to look for:**

- Heads that strongly push toward defection in the strategic model but are weakened/reversed in moral models → candidate "selfish heads"
- Heads that are neutral in the strategic model but strongly push toward cooperation in moral models → candidate "moral heads" (possibly newly developed circuits)

**Code sketch:**

```python
# Get per-head contributions to residual stream
head_results = cache.stack_head_results(layer=-1, pos_slice=-1)  
# Shape: [n_layers, n_heads, d_model]

# Project each head's output to logit space
head_logit_contribs = torch.einsum(
    "lhd,dv->lhv", 
    head_results, 
    model.W_U
)

# Extract contribution to action logit difference
head_action_diff = head_logit_contribs[:, :, action2_id] - head_logit_contribs[:, :, action1_id]
# Shape: [n_layers, n_heads] — heatmap of "selfish contribution" per head
```

---

## Task 4: Activation Patching — Causal Verification

**Goal:** Verify that your candidate heads are _causally_ responsible for the behavioral difference, not just correlated with it.

**Why:** Attribution tells you what components have large effects on the output, but correlation isn't causation. A head might have a large attribution simply because it's in the residual stream path, not because it's doing the relevant computation. Activation patching is the intervention that establishes causality: if swapping this head's activation changes the behavior, it's genuinely load-bearing.

**Method:**

1. **Clean run:** Strategic model on temptation prompt → outputs defect
2. **Corrupted run:** Deontological model on same prompt → outputs cooperate
3. **Patch:** Replace one head's activation in the deontological model with the activation from the strategic model
4. **Measure:** Does the patched model now defect?

**Variants to test:**

- **Noising:** Patch strategic → moral (does injecting "selfish" activations break moral behavior?)
- **Denoising:** Patch moral → strategic (does injecting "moral" activations break selfish behavior?)

**Interpretation:**

- If patching a single head from strategic → moral restores defection, that head is a "selfish head" that the moral model learned to suppress
- If no single head is sufficient but a layer's full attention output is, the circuit is distributed across heads

**Code pattern:**

```python
def patch_head_activation(corrupted_cache, clean_cache, layer, head):
    """Returns a hook function that patches one head's output."""
    def hook(activation, hook):
        # activation shape: [batch, pos, n_heads, d_head]
        activation[:, :, head, :] = clean_cache[hook.name][:, :, head, :]
        return activation
    return hook

# Run with patching
patched_logits = moral_model.run_with_hooks(
    prompt,
    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_head_activation(moral_cache, strategic_cache, layer, head))]
)
```

---

## Task 5: Attention Pattern Analysis — Deontological vs. Utilitarian

**Goal:** Test whether the two moral frameworks implement mechanistically distinct strategies.

**Why:** Deontological and utilitarian reasoning should, in principle, attend to different parts of the prompt:

- **Deontological:** "Did they cooperate? → Don't betray" — should attend heavily to the opponent's previous action
- **Utilitarian:** "What maximizes joint payoff?" — should attend to the payoff matrix numbers

If this hypothesis is correct, you'll see different attention patterns even when both models output the same action. This addresses RQ2 directly.

**Method:**

1. Extract attention patterns for all heads across both moral model variants
2. Identify which tokens in the prompt correspond to:
    - Opponent's last action ("they cooperated" / "they defected")
    - Payoff values ("3 points", "4 points", etc.)
    - Player's last action (for Tit-for-Tat-like reasoning)
3. Compare average attention to these token groups across model variants

**What to look for:**

- Utilitarian models should have heads that attend more to payoff numbers
- Deontological models should have heads that attend more to opponent action history
- The heads identified in Task 4 should show this differential attention pattern

---

## Task 6: Linear Probes — Representation Analysis

**Goal:** Test whether specific concepts (betrayal, joint payoff) are represented differently across model variants.

**Why:** Attention patterns tell you what information flows where, but not how it's _represented_. A linear probe asks: "Can I predict concept X from the activations at layer Y?" If deontological models develop strong "this would be a betrayal" representations that utilitarian models lack, probes will reveal this even if both models attend to similar tokens.

**Method:**

1. Create labeled data:
    - For each prompt, label whether defecting would constitute "betrayal" (opponent cooperated last round)
    - Label the joint payoff for each action
2. Train simple logistic regression probes at each layer
3. Compare probe accuracy curves across model variants

**Probes to train:**

- **Betrayal probe:** Binary classification — "would defecting here betray the opponent?"
- **Opponent action probe:** Binary — "did opponent cooperate or defect last round?"
- **Joint payoff probe:** Regression — "what's the total payoff if we cooperate?"

**What to look for:**

- Deontological models should have higher betrayal probe accuracy, especially in middle layers
- Utilitarian models should have higher joint payoff probe accuracy
- Accuracy emergence layer should correlate with the decision layer from Task 2

---

## Task 7: LoRA Weight Analysis — Where Did Fine-Tuning Push?

**Goal:** Identify which model components changed most during fine-tuning.

**Why:** This is the "cheap" version of RQ3. Before doing expensive interventions, you can simply look at the LoRA adapter weights and see where the optimization pressure was highest. Large weight norms indicate components that needed to change to achieve the reward signal.

**Method:**

1. Load the LoRA adapters (before merging)
2. Compute the Frobenius norm of the effective weight change: `||B @ A||` for each adapted module
3. Rank modules by change magnitude
4. Compare which modules changed most for deontological vs. utilitarian training

**What to look for:**

- If the same layers/heads changed for both moral variants, the "moral circuit" may be shared
- If different components changed, the moral frameworks have distinct implementations
- Concentration in specific layers suggests targeted fine-tuning is feasible

**Code sketch:**

```python
from peft import PeftModel

peft_model = PeftModel.from_pretrained(base, "./checkpoints/deont")

weight_changes = {}
for name, param in peft_model.named_parameters():
    if "lora_A" in name:
        # Find corresponding lora_B
        b_name = name.replace("lora_A", "lora_B")
        lora_a = param
        lora_b = dict(peft_model.named_parameters())[b_name]
        # Effective change is B @ A
        delta = lora_b @ lora_a
        weight_changes[name] = torch.norm(delta).item()
```

---

## Task 8: Frankenstein Validation — Targeted Weight Transfer

**Goal:** Test whether transferring specific components is sufficient to transfer moral behavior.

**Why:** This is the "expensive but rigorous" version of RQ3. If you can take the base model, swap in only the weights from layers 15-18, and get moral behavior, you've proven that fine-tuning can be targeted. This has practical implications for efficient alignment fine-tuning.

**Method:**

1. Identify candidate "moral circuit" components from Tasks 4 and 7
2. Create a hybrid model: base weights everywhere except the candidate components, which come from the moral model
3. Evaluate on your prompt set — does it behave morally?

**Variants:**

- Transfer only attention weights for identified heads
- Transfer only MLP weights for identified layers
- Transfer full layers

**Interpretation:**

- If attention-only transfer works, the moral reasoning is in the attention circuits
- If MLP-only transfer works, it's in the feedforward circuits
- If both are needed, they work together as a distributed circuit

---

## Task 9: Validation with Gemma Scope (Optional)

**Goal:** Connect your IPD-specific findings to general model features documented in Gemma Scope.

**Why:** Neuronpedia's Gemma Scope provides SAE features trained on the base model. If the heads you identified as "selfish" or "moral" align with known features (competition, fairness, negation), this strengthens your interpretation and connects your work to the broader interpretability literature.

**Caveats:**

- SAEs were trained on base model activations; your fine-tuned models may have drifted
- Treat this as "suggestive context" rather than "ground truth labels"
- Most useful for heads that appear important in _both_ base and fine-tuned models

**Method:**

1. Look up your identified heads/layers in Gemma Scope
2. Check what features have high activation in that component
3. Note any features that seem semantically related to your task (game theory, cooperation, betrayal, counting)

---

## Suggested Sequencing

The dependencies look like this:

```
Task 0 (Merge) ─┬──> Task 1 (Prompts)
                │
                ├──> Task 2 (Logit Lens) ──> Task 3 (Attribution) ──> Task 4 (Patching)
                │                                      │
                │                                      └──> Task 5 (Attention Patterns)
                │
                ├──> Task 6 (Probes) [can run in parallel with 2-5]
                │
                └──> Task 7 (LoRA Weights) ──> Task 8 (Frankenstein)

Task 9 (Gemma Scope) ──> [after you have specific heads to look up]
```

**Critical path:** 0 → 1 → 2 → 3 → 4 gives you the answer to RQ1 and candidate components for everything else.
