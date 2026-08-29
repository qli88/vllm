# Kimi-K3: Speculative Decoding — MTP and DSpark

[← Index](kimi_k3_index.md)

Covers K3's two speculative decoding mechanisms: MTP (Multi-Token Prediction heads attached to the base model) and DSpark (a separate 5-layer dense draft model). Includes architecture, weight mapping, forward pass mechanics, and DCP integration.

---

## Table of Contents

1. [Overview: MTP vs DSpark](#1-overview-mtp-vs-dspark)
2. [MTP: Multi-Token Prediction](#2-mtp-multi-token-prediction)
   - 2.1 [Architecture](#21-architecture)
   - 2.2 [fused_mtp_input: Triton kernel](#22-fused_mtp_input-triton-kernel)
   - 2.3 [Forward pass](#23-forward-pass)
   - 2.4 [Weight loading and layer index mapping](#24-weight-loading-and-layer-index-mapping)
   - 2.5 [Integration with vLLM spec-decode](#25-integration-with-vllm-spec-decode)
3. [DSpark: Dense MLA Draft Model](#3-dspark-dense-mla-draft-model)
   - 3.1 [Architecture](#31-architecture)
   - 3.2 [Non-causal multi-token decode](#32-non-causal-multi-token-decode)
   - 3.3 [Context KV projection (cross-layer fused)](#33-context-kv-projection-cross-layer-fused)
   - 3.4 [Markov head](#34-markov-head)
   - 3.5 [Weight sharing with target model](#35-weight-sharing-with-target-model)
   - 3.6 [Weight loading: _duplicate_context_kv_weights](#36-weight-loading-_duplicate_context_kv_weights)
4. [DCP + DSpark](#4-dcp--dspark)
   - 4.1 [Query gather under DCP](#41-query-gather-under-dcp)
   - 4.2 [Chunked-context prefill under DCP](#42-chunked-context-prefill-under-dcp)
5. [Model Registry and Config](#5-model-registry-and-config)
6. [Concrete Example: DSpark Draft Step](#6-concrete-example-dspark-draft-step)
7. [MTP Worked Example](#7-mtp-worked-example)

---

## 1. Overview: MTP vs DSpark

| Property | MTP | DSpark |
|---|---|---|
| Architecture | Extra heads on the base model | Separate 5-layer dense MLA model |
| Attention | Uses same KDA/MLA layers as base model | Full MLA (RoPE-enabled) only |
| Context access | Through base model's hidden states | Through target model's hidden states (fused projection) |
| Draft type | Next-N prediction heads (self-distillation) | Non-causal multi-token decode |
| Weight sharing | Shares `embed_tokens` + `lm_head` with base | Shares `embed_tokens` + `lm_head` with base |
| Config key | `num_nextn_predict_layers` | Separate `K3DSparkForCausalLM` model |
| Current K3 config | `num_nextn_predict_layers=0` (disabled) | Enabled separately |
| ROCm support | Yes | **No** (NVIDIA only) |

K3-Pro (the model at `/shareddata/data/models/moonshotai-Kimi-K3`) has `num_nextn_predict_layers=0`, so MTP is disabled. DSpark is a separately registered model class.

---

## 2. MTP: Multi-Token Prediction

### 2.1 Architecture

When `num_nextn_predict_layers > 0`, MTP prediction heads are appended to the base model:

```
KimiK3MTPModel:
  model: KimiLinearModel   (base, all 93 layers)
  mtp_layers:
    KimiK3MultiTokenPredictorLayer × num_nextn_predict_layers
      ├─ enorm:    RMSNorm [7168]    (normalize embed input)
      ├─ hnorm:    RMSNorm [7168]    (normalize hidden state input)
      ├─ eh_proj:  Linear [2×7168=14336 → 7168]  (fuse embed + hidden → new hidden)
      ├─ mtp_block: KimiDecoderLayer  (full transformer layer, no AttnRes)
      └─ shared_head:
          ├─ RMSNorm [7168]
          └─ ParallelLMHead [7168 → 163840]  (shared with base model's lm_head)
```

Each MTP layer takes:
- The **embedding** of the `i`-th future token (zeroed at position 0 — the "no previous token" sentinel)
- The **hidden state** from the base model at the same layer depth

And produces a logit distribution for the `(i+1)`-th future token.

**Why share `lm_head`?** MTP is trained with self-distillation: the draft head should mimic the base model's output distribution. Using the same `lm_head` ensures the draft operates in the same probability space as the base model.

### 2.2 `fused_mtp_input`: Triton kernel

**File:** `common/mtp.py`

A single Triton kernel that fuses four operations:

```python
def fused_mtp_input(
    embeds,      # [T, D]  token embeddings (from embed_tokens)
    hiddens,     # [T, D]  hidden states from base model
    eh_proj,     # [2D, D]  linear projection weight
    enorm_w,     # [D]  RMSNorm scale for embeds
    hnorm_w,     # [D]  RMSNorm scale for hiddens
    start_pos,   # [T]  absolute position of each token
    mask_first,  # bool: zero out embeds at position 0
) → output [T, D]:
```

**Fused operations in one kernel:**

```
For each token t:
  1. If mask_first and start_pos[t] == 0:
       embed_masked = 0   (sentinel: no previous token at position 0)
     Else:
       embed_masked = embeds[t]

  2. RMSNorm(embed_masked, enorm_w)   → normed_embed [D]
  3. RMSNorm(hiddens[t],   hnorm_w)   → normed_hidden [D]

  4. Concatenate: [normed_embed | normed_hidden]  [2D]
  5. Apply eh_proj: [2D] × W[2D, D] → output[t]  [D]
```

All four operations (mask, 2 RMSNorms, concat-and-project) happen in one Triton kernel launch, avoiding 4 separate HBM round-trips.

**Why zero position 0:** The MTP layer for the `i`-th draft token needs the embedding of the `i-1`-th token. At position 0, there is no prior token, so the embedding is set to zero (a learned sentinel).

### 2.3 Forward pass

```python
class KimiK3MultiTokenPredictorLayer:
    def forward(self, embed_input, hidden_input, positions, attn_metadata, kv_caches):
        # Fused input preparation:
        fused_hidden = fused_mtp_input(
            embeds=embed_input,
            hiddens=hidden_input,
            eh_proj=self.eh_proj.weight,
            enorm_w=self.enorm.weight,
            hnorm_w=self.hnorm.weight,
            start_pos=positions,
        )   # [T, 7168]

        # Run full transformer layer (same KimiDecoderLayer used by base model,
        # but attn_res_block_size=None → no AttnRes):
        hidden_out = self.mtp_block(fused_hidden, positions, attn_metadata, kv_caches)

        # LM head (shared with base model):
        hidden_normed = self.shared_head[0](hidden_out)   # RMSNorm
        logits = self.shared_head[1](hidden_normed)       # ParallelLMHead
        return logits, hidden_out
```

**No AttnRes in MTP blocks:** `mtp_block` is a `KimiDecoderLayer` initialized with `attn_res_block_size=None`, disabling AttnRes. The MTP layers are shallow (1 layer per draft step) and don't benefit from cross-layer residuals.

### 2.4 Weight loading and layer index mapping

MTP layers are stored in the checkpoint at layer indices `num_hidden_layers + i` (i.e., layers 93, 94, ...).

**Layer index rewriting:**

```python
# In KimiK3MTP.load_weights:
# Weights with "model.layers.93.mtp_block.*" → stored as mtp_layers[0].mtp_block.*
# Stripping ".mtp_block." to find the decoder layer weights inside KimiDecoderLayer.

def _mtp_layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{num_hidden_layers + layer_idx}"
```

**Shared head:** The `lm_head` weights are NOT duplicated — the MTP model's `shared_head.ParallelLMHead` uses the same parameter object as the base model.

**Multimodal checkpoint prefix stripping:**

When loading from a multimodal checkpoint (`KimiK3ForConditionalGeneration`), the `language_model.` prefix is stripped before looking up MTP layer weights.

### 2.5 Integration with vLLM spec-decode

`vllm/config/speculative.py` recognizes `kimi_k3_mtp`:

```python
# Config injection for MTP spec decode:
speculative_config = {
    "method": "kimi_k3_mtp",
    "n_predict": 2,   # generate 2 draft tokens per step
    "architectures": ["KimiK3MTPModel"],
}
```

During speculative decode:
1. Base model produces hidden states.
2. MTP layer 0 takes embed(token_0) + hidden(layer_93) → draft logit 0.
3. MTP layer 1 takes embed(draft_0) + hidden(layer_93) → draft logit 1.
4. Verification: base model runs on draft tokens; accepts or rejects.
5. On rejection: `RecoverSSM` replays KDA states from the checkpoint before the rejected tokens.

---

## 3. DSpark: Dense MLA Draft Model

### 3.1 Architecture

**File:** `nvidia/dspark_mla.py`

DSpark is a separate 5-layer model (`K3DSparkForCausalLM`) designed for fast speculative decoding of K3.

```
K3DSparkModel:
  embed_tokens:          shared with target (has_own_embed_tokens=False)
  context_kv_proj:       Linear [7168 → (L+R)×num_draft_layers]
                         = [7168 → 576×5]  (one KV slice per draft layer, fused)
  layers:                MultiHeadLatentAttention × num_layers=5
                         (use_rope=True, non_causal_multi_token_decode=True)
  norm:                  RMSNorm [7168]
  lm_head:               shared with target (has_own_lm_head=False)
  markov_head:           DSparkMarkovHead  (low-rank Markov bias)
```

**Key differences from the base K3 model:**

| Property | K3 Base | DSpark Draft |
|---|---|---|
| Layers | 93 (hybrid MLA+KDA) | 5 (full MLA only) |
| RoPE on MLA | `mla_use_nope=True` | `use_rope=True` |
| Causal masking | Yes | `non_causal_multi_token_decode=True` |
| KDA layers | 69 | None |
| MoE | Yes | No (dense) |
| Context KV | From own attention | From target model's hidden states |
| Embedding | Own (or shared) | Shared with target |

### 3.2 Non-causal multi-token decode

`non_causal_multi_token_decode=True` allows all draft tokens to attend to each other and to the full context simultaneously:

```
Standard causal decode:
  draft_0 sees: context (0..N-1)
  draft_1 sees: context + draft_0
  draft_2 sees: context + draft_0 + draft_1

Non-causal multi-token decode:
  draft_0, draft_1, draft_2 all see: context (0..N-1) simultaneously
  draft tokens do NOT attend to each other (across draft positions)
```

This allows all `n` draft tokens to be processed in **one** forward pass with a modified attention mask, rather than `n` sequential autoregressive steps.

**Implementation:** The attention mask in the backend is set to non-causal across draft token positions. Each draft token attends to the full context but not to sibling draft tokens. This is equivalent to running `n` independent attention operations with the same KV cache, done in one batched call.

**Why non-causal is valid:** Draft tokens are meant to be independent proposals — the verifier (base model) checks each one against the true distribution. Making them attend to each other would create dependencies that complicate verification.

### 3.3 Context KV projection (cross-layer fused)

Rather than computing KV from the draft model's own hidden states, DSpark reuses the **target model's final hidden states** to produce KV for all 5 draft layers in one GEMM:

```python
# context_kv_proj: Linear [7168 → 576×5 = 2880]
# (576 = L+R = 512 c_kv + 64 k_pe, per layer)

# During DSpark setup, called once per target forward pass:
target_hidden [T, 7168]   (last layer hidden state of base K3)

kv_all = context_kv_proj(target_hidden)   # [T, 2880]
# Split into per-layer KV:
for layer_idx in range(5):
    c_kv_layer  = kv_all[:, layer_idx*576 : layer_idx*576+512]   # [T, 512]
    k_pe_layer  = kv_all[:, layer_idx*576+512 : (layer_idx+1)*576]  # [T, 64]
    # Apply RMSNorm + RoPE:
    c_kv_layer  = rms_norm(c_kv_layer)
    k_pe_rotated = rotary_embedding(k_pe_layer, positions)
    # Store in draft model's KV cache for this layer:
    kv_cache_layer[layer_idx][slot] = cat(c_kv_layer, k_pe_rotated)
```

**Why fuse all 5 layers into one GEMM:**

Without fusion:
- 5 separate GEMMs: `[T, 7168] × [576, 7168]` each = 5 × 8.3 GFLOPs/token

With fusion:
- 1 GEMM: `[T, 7168] × [2880, 7168]` = 1 × 41.5 GFLOPs/token
- Same total FLOPS, but 5× fewer kernel launches and 5× better memory access locality for the weight matrix

The fused GEMM also allows PDL to overlap it with the draft model's first layer computation.

### 3.4 Markov head

`DSparkMarkovHead` provides a low-rank Markov transition bias for draft logit shaping:

```python
class DSparkMarkovHead:
    down_proj: Linear [hidden_size=7168 → markov_rank=512]
    up_proj:   Linear [markov_rank=512 → vocab_size=163840]

def forward(self, hidden_states) → markov_bias [T, 163840]:
    latent = F.linear(hidden_states, down_proj.weight)  # [T, 512]
    bias = F.linear(latent, up_proj.weight)              # [T, 163840]
    return bias
```

**What it does:** Adds a learned position-and-content-dependent bias to the draft logits before sampling. This acts as a learned prior that shifts draft probabilities toward likely next tokens given the current context, improving draft acceptance rates.

**How it's used:**

```python
draft_logits = lm_head(norm(hidden))           # [T, 163840]
markov_bias = markov_head(hidden)              # [T, 163840]
final_logits = draft_logits + markov_bias      # [T, 163840]
draft_tokens = sample(final_logits)
```

### 3.5 Weight sharing with target model

```python
class K3DSparkForCausalLM:
    has_own_embed_tokens = False   # shares embed_tokens with target
    has_own_lm_head = False        # shares lm_head with target
```

**Implications:**
- `embed_tokens` parameter is NOT stored in DSpark — it references the target model's embedding.
- `lm_head` is NOT stored in DSpark — same reference to target's LM head.
- Context KV weights are stored only in DSpark's `context_kv_proj`.

During weight loading, vLLM's speculative decode infrastructure routes the shared weights to the target model's parameter store and provides them to the draft model by reference.

### 3.6 Weight loading: `_duplicate_context_kv_weights`

At weight load time, the 5 draft layers' `kv_a_proj_with_mqa` weights are copied into the fused `context_kv_proj`:

```python
def _duplicate_context_kv_weights(self):
    """Clone each layer's kv_a_proj weights into context_kv_proj for cross-layer reuse."""
    for layer_idx, layer in enumerate(self.model.layers):
        offset = layer_idx * (self.kv_lora_rank + self.qk_rope_head_dim)
        # Copy this layer's kv_a_proj weights into the corresponding slice of context_kv_proj:
        self.model.context_kv_proj.weight.data[
            offset : offset + self.kv_lora_rank + self.qk_rope_head_dim, :
        ] = layer.self_attn.kv_a_proj_with_mqa.weight.data
```

This ensures that calling `context_kv_proj(target_hidden)` is equivalent to calling each layer's `kv_a_proj_with_mqa` separately, but in one fused operation.

---

## 4. DCP + DSpark

The recent commit "Support Kimi-K3 DCP with DSpark" (commit `d1e3eee6fb`) integrates DCP into DSpark draft decoding.

### 4.1 Query gather under DCP

When `dcp_world_size > 1`, the draft model's MLA layers must gather queries across DCP ranks before computing attention:

```python
# In MultiHeadLatentAttention._attention (for DSpark):
if self.dcp_manager is not None and self.dcp_manager.world_size > 1:
    # Broadcast query from this rank to all DCP ranks:
    q_gathered = self.dcp_manager.gather_q(q)   # [T × dcp_ws, H, L+R]

    # Each rank computes attention over its local KV shard:
    attn_latent_local, lse = self.impl.forward_mqa(
        q=q_gathered,
        kv_cache=local_kv_cache,
        return_lse=True,
    )

    # LSE-aware cross-rank combine:
    attn_latent = self.dcp_manager.combine(
        attn_latent_local, lse, seq_lens, query_start_loc
    )
```

The combine step uses `dcp_direct_a2a_lse_reduce.cu` for the all-to-all communication and numerically stable LSE merge.

### 4.2 Chunked-context prefill under DCP

For chunked context prefill under DCP:

```python
# In MultiHeadLatentAttention._compute_prefill_context (DSpark path):
if self.dcp_manager is not None:
    output = self.impl._context_parallel_compute_prefill_context(
        q, kv_cache, block_table, seq_lens, dcp_manager=self.dcp_manager
    )
else:
    output = self._compute_prefill_context(q, kv_cache, block_table, seq_lens)
```

The DCP-aware version distributes the chunked context iterations across DCP ranks — each rank processes the context chunks for its own KV shard, then combines results with the LSE-aware merge.

---

## 5. Model Registry and Config

**DSpark model class:**

```python
# vllm/models/kimi_k3/nvidia/dspark_mla.py:
class K3DSparkForCausalLM:
    # Registered as:
    architectures = ["K3DSparkModel"]
```

**Registry entry:**

```python
# vllm/model_executor/models/registry.py:
"K3DSparkModel": "vllm.models.kimi_k3.nvidia.dspark_mla"
```

**Config for speculative decoding:**

```python
vllm_engine = LLM(
    model="moonshotai/Kimi-K3",
    speculative_model="path/to/dspark",  # K3DSparkForCausalLM checkpoint
    num_speculative_tokens=5,
    use_v2_block_manager=True,
)
```

**DCP deployment config** (from `context_parallel_deployment.md`):

```python
LLM(
    model="moonshotai/Kimi-K3",
    tensor_parallel_size=8,
    context_parallel_deployment="decode",  # DCP for decode only
    context_parallel_size=4,               # 4 DCP ranks
    speculative_model="K3DSparkModel",
    num_speculative_tokens=5,
)
```

---

## 6. Concrete Example: DSpark Draft Step

**Setup:**
- Target model: K3 (93 layers, just ran a forward pass, hidden states available)
- DSpark draft model: 5 MLA layers (RoPE, non-causal)
- Context: 1000 tokens already in KV cache
- Generating: 5 draft tokens

**Step 1: Compute context KV from target hidden states**

```python
# Target model's final hidden state (after layer 92):
target_hidden [1000, 7168]   (for all context tokens)

# Fused context KV projection:
kv_all = context_kv_proj(target_hidden)   # [1000, 576×5=2880]

# Per-layer KV setup for 5 draft layers:
for layer in range(5):
    c_kv = kv_all[:, layer*576:layer*576+512]           # [1000, 512]
    k_pe = kv_all[:, layer*576+512:(layer+1)*576]       # [1000, 64]
    c_kv = rms_norm(c_kv)
    k_pe_rotated = rotary_embedding(k_pe, positions=range(1000))
    # Store in draft KV cache:
    draft_kv_cache[layer][0:1000] = cat(c_kv, k_pe_rotated, dim=-1)   # [1000, 576]
```

**Step 2: Draft forward pass (5 tokens simultaneously, non-causal)**

```python
# Start from last generated token's embedding:
draft_embed = embed_tokens([last_token_id])   # [1, 7168]

# Draft model's MLA (5 layers, all attend to context 0..999):
# Non-causal: token at draft position 0 and draft position 4 both see context 0..999
# but they DON'T see each other (across draft positions)

for layer_idx in range(5):
    # Each draft layer uses context_kv already in draft_kv_cache[layer_idx]
    hidden = draft_mla[layer_idx](draft_embed, draft_kv_cache[layer_idx])

# 5 hidden states produced → 5 logit distributions:
draft_logits = [lm_head(norm(hidden_i)) + markov_head(hidden_i)  for i in range(5)]
# draft_logits: 5 × [163840]

draft_tokens = [argmax(logits_i) for logits_i in draft_logits]
# draft_tokens: [t1, t2, t3, t4, t5]  5 candidate next tokens
```

**Step 3: Verification (base model)**

```python
# Run base K3 on [context + draft_tokens]:
verify_input = cat(last_accepted_token, draft_tokens)   # [5]
base_logits = base_model.forward(verify_input)           # 5 logit distributions

# Compare draft vs base at each position:
for i in range(5):
    if base accepts draft_tokens[i]:
        accept and continue
    else:
        reject (resample from base_logits[i])
        # RecoverSSM: roll back KDA states to before rejected tokens
        break
```

**Efficiency:** Without DSpark, generating 5 tokens requires 5 × 93-layer forward passes. With DSpark, it requires 1 × 93-layer pass (target) + 1 × 5-layer pass (draft, generating 5 candidates) + 1 × 5-token verification pass (target). If the acceptance rate is >50%, DSpark reduces latency.

**Weight-sharing initialisation detail — `_duplicate_context_kv_weights` in concrete numbers:**

At model init, `context_kv_proj` is built and filled from each draft layer's existing `kv_a_proj_with_mqa`:

```python
# At K3DSparkModel init:
# kv_lora_rank=512, qk_rope_head_dim=64, num_draft_layers=5
context_kv_proj = Linear(in_features=7168,
                         out_features=(512 + 64) * 5)  # = 576 × 5 = 2880

# _duplicate_context_kv_weights():
for layer_idx, layer in enumerate(self.model.layers):  # 5 draft layers
    offset = layer_idx * 576
    # Slice out this draft layer's kv_a_proj weights (shape [576, 7168])
    # and copy them into the corresponding slice of context_kv_proj:
    context_kv_proj.weight.data[offset : offset + 576, :] = \
        layer.self_attn.kv_a_proj_with_mqa.weight.data

# After: context_kv_proj.weight [2880, 7168] = 5 stacked KV projections
```

Memory / compute saving:

- Without fusion: 5 separate GEMMs `[T, 7168] × [576, 7168]` each → 5 kernel launches
- With fusion: 1 GEMM `[T, 7168] × [2880, 7168]` → 1 kernel launch, 5× better kernel launch efficiency; weight rows are also contiguous so the L2 working set is loaded once

---

## 7. MTP Worked Example

**Setup:** `num_nextn_predict_layers=2` (two MTP heads), generating 2 draft tokens. The base model has just processed the context up through token at position 5 ("Paris", token_id=14366).

```
Base model forward on token "Paris" (token_id=14366, position=5):
  hidden_states [1, 7168] after all 93 layers
  base_logits = lm_head(norm(hidden_states)) → [1, 163840]
  next_token_0 = argmax(base_logits) = 14366  (hypothetically "▁France")

MTP layer 0 (predicts token at position 6):
  embed_input  = embed_tokens([14366])     [1, 7168]  (embedding of next_token_0)
  hidden_input = hidden_states             [1, 7168]  (from layer 93 of base model)

  fused_mtp_input (one Triton kernel):
    enorm(embed_input)    [1, 7168]
    hnorm(hidden_input)   [1, 7168]
    concat → [1, 14336]
    eh_proj: [14336 → 7168] → fused_hidden [1, 7168]

  mtp_block (KimiDecoderLayer, attn_res_block_size=None → no AttnRes):
    KDA/MLA attention → [1, 7168]
    MoE              → [1, 7168]
    output: mtp_hidden_0 [1, 7168]

  mtp_logits_0 = lm_head(norm(mtp_hidden_0))  [1, 163840]
  draft_token_1 = argmax(mtp_logits_0) = 9784  (hypothetically "▁Germany")

MTP layer 1 (predicts token at position 7):
  embed_input  = embed_tokens([9784])      [1, 7168]  (embedding of draft_token_1)
  hidden_input = mtp_hidden_0              [1, 7168]  (output of MTP layer 0)

  fused_mtp_input → fused_hidden_1 [1, 7168]
  mtp_block → mtp_hidden_1 [1, 7168]

  mtp_logits_1 = lm_head(norm(mtp_hidden_1))  [1, 163840]
  draft_token_2 = argmax(mtp_logits_1)  (hypothetically "▁known")

Draft sequence: ["▁France", "▁Germany"]  — 2 tokens generated in one forward pass

Verification: run base model on [token_5, draft_token_1, draft_token_2]
  = ["Paris", "▁France", "▁Germany"]
  Accept or reject each draft token against base_logits at each position.
  If draft_token_1 rejected: resample from base_logits[1], discard draft_token_2,
                              RecoverSSM rolls KDA states back to before position 6.
  If draft_token_1 accepted, draft_token_2 rejected: resample from base_logits[2].
  If both accepted: 2 new tokens committed in one round trip.
```

**`spec_step_idx % num_mtp_layers` cycling:**

Each speculative step increments `spec_step_idx`. The MTP layer selected for the next draft step is `spec_step_idx % num_mtp_layers`. If both draft tokens above were accepted and the decode continues:

```
spec_step_idx=0 → use MTP layer 0 (generate draft at position 6)
spec_step_idx=1 → use MTP layer 1 (generate draft at position 7)
spec_step_idx=2 → use MTP layer 0 again (position 8) — wraps back to layer 0
spec_step_idx=3 → use MTP layer 1 (position 9)
...
```

With `num_nextn_predict_layers=2`, the two heads alternate indefinitely. Each head is tuned (via its separate `eh_proj` and `mtp_block` weights) to predict tokens at even or odd offsets relative to the current generation position.
