"""Model loading utilities for LoRA-adapted models with mechanistic interpretability support."""

import torch
from pathlib import Path
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from mech_interp.utils import get_model_path


class HookedGemmaModel:
    """
    Wrapper around HuggingFace Gemma model with activation caching.

    This provides similar functionality to TransformerLens HookedTransformer
    but works reliably with Gemma-2 models.
    """

    def __init__(self, hf_model, tokenizer, device="cuda"):
        self.model = hf_model
        self.tokenizer = tokenizer
        self.device = device

        # Unwrap PEFT wrappers for stable access to base architecture modules.
        if hasattr(hf_model, "get_base_model"):
            base_lm = hf_model.get_base_model()
        else:
            base_lm = hf_model

        # Gemma2ForCausalLM -> .model is the transformer stack with .layers/.norm
        self.base_lm = base_lm
        self.transformer_stack = base_lm.model

        # Model configuration
        self.n_layers = base_lm.config.num_hidden_layers
        self.n_heads = base_lm.config.num_attention_heads
        self.d_model = base_lm.config.hidden_size
        self.d_vocab = base_lm.config.vocab_size

        # Cache for storing activations
        self.cache = {}
        self.hooks = []

    def register_cache_hooks(self):
        """Register forward hooks to cache all intermediate activations."""
        self.cache.clear()
        self.hooks.clear()

        # Hook for each layer's outputs
        for layer_idx in range(self.n_layers):
            layer = self.transformer_stack.layers[layer_idx]

            # Hook for residual stream after attention + MLP
            def make_resid_hook(idx):
                def hook(module, input, output):
                    # output[0] is the hidden state (residual stream)
                    self.cache[f"blocks.{idx}.hook_resid_post"] = output[0].detach()
                    return output

                return hook

            handle = layer.register_forward_hook(make_resid_hook(layer_idx))
            self.hooks.append(handle)

            # Hook for attention outputs (before adding residual)
            def make_attn_hook(idx):
                def hook(module, input, output):
                    # output[0] is attention output before residual connection
                    # output[1] is attention weights (if return_attention_weights=True)
                    self.cache[f"blocks.{idx}.hook_attn_out"] = output[0].detach()

                    # Cache attention weights if available
                    if len(output) > 1 and output[1] is not None:
                        self.cache[f"model.layers.{idx}.self_attn.attn_weights"] = output[1].detach()

                    return output

                return hook

            handle = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
            self.hooks.append(handle)

            # Hook for MLP outputs (before adding residual)
            def make_mlp_hook(idx):
                def hook(module, input, output):
                    self.cache[f"blocks.{idx}.hook_mlp_out"] = output.detach()
                    return output

                return hook

            handle = layer.mlp.register_forward_hook(make_mlp_hook(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def run_with_cache(self, input_ids, output_attentions=False):
        """
        Run forward pass and return logits + cached activations.

        Args:
            input_ids: Input token IDs
            output_attentions: Whether to output attention weights
                Note: Requires model loaded with attn_implementation="eager"

        Returns:
            logits: Model output logits
            cache: Dict of cached activations
        """
        self.register_cache_hooks()

        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=output_attentions)
            logits = outputs.logits

            # Cache attention weights if requested
            if output_attentions and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                for layer_idx, attn_weights in enumerate(outputs.attentions):
                    # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                    if attn_weights is not None:
                        self.cache[f"model.layers.{layer_idx}.self_attn.attn_weights"] = attn_weights.detach()

        # Get embedding matrix for logit lens (W_U) and final layer norm
        self.W_U = self.base_lm.lm_head.weight.detach()  # (vocab_size, d_model)
        self.ln_final = self.transformer_stack.norm  # Final RMSNorm layer

        self.remove_hooks()

        return logits, self.cache.copy()

    def unembed(self, hidden_state):
        """
        Project hidden state to vocabulary space (logits).

        Applies layer normalization before unembedding, as Gemma does.

        Args:
            hidden_state: (..., d_model) tensor

        Returns:
            logits: (..., d_vocab) tensor
        """
        # Apply final layer norm (RMSNorm)
        normed_hidden = self.ln_final(hidden_state)

        # W_U is (vocab_size, d_model), we need (d_model, vocab_size)
        return normed_hidden @ self.W_U.T

    def __call__(self, input_ids):
        """Forward pass without caching."""
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits


class LoRAModelLoader:
    """
    Loads LoRA-adapted Gemma-2-2b-it models for mechanistic interpretability.

    Handles in-memory LoRA merging and provides activation caching.
    """

    BASE_MODEL_PATH = "/root/LLM_morality/models/gemma-2-2b-it"

    @staticmethod
    def load_base_hf_model(
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
        merge_lora: bool = True,
        use_4bit: bool = False,
    ):
        """
        Load HuggingFace model with optional LoRA adapter.

        Args:
            model_id: Model identifier ('base', 'PT2_COREDe', etc.)
            device: Device to load model on ('cuda' or 'cpu')
            dtype: Model dtype (default: bfloat16 for efficiency)
            attn_implementation: Attention implementation ('eager' or 'sdpa')
                Use 'eager' for attention analysis (supports output_attentions)

        Returns:
            Loaded (and potentially LoRA-merged) HuggingFace model
        """
        print(f"Loading base model from {LoRAModelLoader.BASE_MODEL_PATH}...")

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                LoRAModelLoader.BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map={"": 0} if device == "cuda" else device,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                LoRAModelLoader.BASE_MODEL_PATH,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )

        # If not base model, load LoRA adapter and merge
        if model_id != "base":
            adapter_path = get_model_path(model_id)
            print(f"Loading LoRA adapter from {adapter_path}...")

            peft_kwargs = {}
            if not use_4bit:
                peft_kwargs["torch_dtype"] = dtype

            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                **peft_kwargs,
            )

            if merge_lora:
                print("Merging LoRA weights into base model (in-memory)...")
                # In-memory merge - no disk write
                model = model.merge_and_unload()
                print("LoRA merge complete!")
            else:
                print("Using non-merged LoRA adapter (inference-style).")
        else:
            print("Using base model without LoRA adaptation.")
            model = base_model

        return model

    @staticmethod
    def load_hooked_model(
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
        merge_lora: bool = True,
        use_4bit: bool = False,
    ) -> HookedGemmaModel:
        """
        Load model with activation caching support.

        Args:
            model_id: Model identifier ('base', 'PT2_COREDe', etc.)
            device: Device to load model on
            dtype: Model dtype
            attn_implementation: Attention implementation ('eager' or 'sdpa')
                Default 'eager' for compatibility with attention analysis

        Returns:
            HookedGemmaModel with activation caching
        """
        # Load HuggingFace model (with LoRA merged if applicable)
        hf_model = LoRAModelLoader.load_base_hf_model(
            model_id,
            device,
            dtype,
            attn_implementation=attn_implementation,
            merge_lora=merge_lora,
            use_4bit=use_4bit,
        )
        tokenizer = LoRAModelLoader.load_tokenizer()

        print(f"Creating hooked model wrapper for {model_id}...")
        hooked_model = HookedGemmaModel(hf_model, tokenizer, device)

        print(f"Hooked model created successfully!")
        print(f"  Layers: {hooked_model.n_layers}")
        print(f"  Heads: {hooked_model.n_heads}")
        print(f"  d_model: {hooked_model.d_model}")
        print(f"  d_vocab: {hooked_model.d_vocab}")

        return hooked_model

    @staticmethod
    def load_tokenizer() -> AutoTokenizer:
        """
        Load shared tokenizer for all models.

        Returns:
            HuggingFace tokenizer
        """
        return AutoTokenizer.from_pretrained(LoRAModelLoader.BASE_MODEL_PATH)


def test_model_loading(model_id: str = "base"):
    """
    Test function to verify model loading works correctly.

    Args:
        model_id: Model to test
    """
    print(f"\n{'='*60}")
    print(f"Testing model loading for: {model_id}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = LoRAModelLoader.load_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load hooked model
    print("\nLoading hooked model...")
    hooked_model = LoRAModelLoader.load_hooked_model(model_id)

    # Test forward pass without cache
    print("\nTesting forward pass...")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt").input_ids.to(hooked_model.device)
    logits = hooked_model(inputs)
    print(f"Output logits shape: {logits.shape}")

    # Test forward pass with cache
    print("\nTesting forward pass with activation caching...")
    logits_cached, cache = hooked_model.run_with_cache(inputs)
    print(f"Cached logits shape: {logits_cached.shape}")
    print(f"Number of cached activations: {len(cache)}")
    print(f"Cache keys: {list(cache.keys())[:5]}...")

    # Test logit lens (unembed)
    print("\nTesting logit lens (unembed)...")
    # Get final layer hidden state
    final_layer_key = f"blocks.{hooked_model.n_layers - 1}.hook_resid_post"
    final_hidden = cache[final_layer_key][0, -1, :]  # Last token
    reconstructed_logits = hooked_model.unembed(final_hidden)
    print(f"Reconstructed logits shape: {reconstructed_logits.shape}")

    # Compare with actual logits
    diff = torch.abs(logits_cached[0, -1, :] - reconstructed_logits).max().item()
    print(f"Max difference between actual and reconstructed logits: {diff:.6f}")
    if diff < 1e-3:
        print("✓ Logit lens working correctly!")
    else:
        print("⚠ Warning: Large difference - may need debugging")

    # Test action token encoding
    from mech_interp.utils import get_action_token_ids

    action_tokens = get_action_token_ids(tokenizer)
    print(f"\nAction tokens: {action_tokens}")

    print(f"\n{'='*60}")
    print(f"Test complete for {model_id}!")
    print(f"{'='*60}\n")

    # Clean up
    del hooked_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run test
    test_model_loading("base")
