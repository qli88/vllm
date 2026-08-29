# Kimi-K3: End-to-End Walkthrough

[← Index](kimi_k3_index.md) | [MLA](kimi_k3_mla.md) | [KDA](kimi_k3_kda.md) | [MoE](kimi_k3_moe.md)

Traces the full inference pipeline from raw text to the generated next token. Uses concrete shapes and numbers. Covers both prefill and decode, with detailed walkthroughs of one MLA layer and one KDA layer.

---

## Table of Contents

1. [Setup: Batch and Prompt](#1-setup-batch-and-prompt)
2. [Tokenization](#2-tokenization)
3. [Embedding Lookup](#3-embedding-lookup)
4. [Layer 0: MLA + Dense MLP (first layer)](#4-layer-0-mla--dense-mlp-first-layer)
   - 4.1 [MLA forward: prefill tokens](#41-mla-forward-prefill-tokens)
   - 4.2 [AttnRes write](#42-attnres-write)
   - 4.3 [Dense MLP (KimiMLP)](#43-dense-mlp-kimimlp)
   - 4.4 [AttnRes read](#44-attnres-read)
5. [Layer 1: KDA + MoE (first KDA layer)](#5-layer-1-kda--moe-first-kda-layer)
   - 5.1 [KDA prefill: chunked Triton attention](#51-kda-prefill-chunked-triton-attention)
   - 5.2 [MoE routing and expert dispatch](#52-moe-routing-and-expert-dispatch)
6. [Layer 4: MLA + MoE (first standard MLA+MoE)](#6-layer-4-mla--moe-first-standard-mlamoe)
7. [After All 93 Layers: Logit Computation and Sampling](#7-after-all-93-layers-logit-computation-and-sampling)
8. [Decode Step: Single Token](#8-decode-step-single-token)
   - 8.1 [Decode through a KDA layer](#81-decode-through-a-kda-layer)
   - 8.2 [Decode through an MLA layer](#82-decode-through-an-mla-layer)
9. [Long-Context Decode: 32K Tokens](#9-long-context-decode-32k-tokens)
   - 9.1 [MLA layer: paged attention cost](#91-mla-layer-paged-attention-cost)
   - 9.2 [KDA layer: O(1) decode cost](#92-kda-layer-o1-decode-cost)
10. [Decode Step: Long Context (32K Tokens)](#10-decode-step-long-context-32k-tokens)
    - 10.1 [Per-layer cost breakdown](#101-per-layer-cost-breakdown)
    - 10.2 [Total latency estimate and comparison](#102-total-latency-estimate-and-comparison)
11. [Sequence Parallelism Example](#11-sequence-parallelism-example)
    - 11.1 [MLA+MoE layer with SP](#111-mlamoe-layer-with-sp)
    - 11.2 [GEMM-RS fusion benefit](#112-gemm-rs-fusion-benefit)

---

## 1. Setup: Batch and Prompt

```
Input text:  "What is the capital of France?"
Task:        Next-token prediction (prefill + one decode step)

Batch:
  Request A:  8-token prompt (prefill)
  Request B:  continuing sequence, 500 tokens of history (decode)

Hardware: TP=1, single GPU (NVIDIA, SM90+)
Shown in detail: layers 0 (MLA+Dense), 1 (KDA+MoE), 4 (MLA+MoE)
```

---

## 2. Tokenization

```
"What is the capital of France?"

→ tokens:    ["What", "▁is", "▁the", "▁capital", "▁of", "▁France", "?", ""]
→ token_ids: [2061, 374, 279, 6864, 315, 9784, 30, 0]   (illustrative)

T = 8 prefill tokens + 1 decode token = 9 total in batch
Token layout (decode first):
  index 0: decode token (request B, position=499)
  index 1: prefill "What"      (request A, position=0)
  index 2: prefill "▁is"       (position=1)
  index 3: prefill "▁the"      (position=2)
  index 4: prefill "▁capital"  (position=3)
  index 5: prefill "▁of"       (position=4)
  index 6: prefill "▁France"   (position=5)
  index 7: prefill "?"         (position=6)
  index 8: prefill "" (EOS)    (position=7)

positions = [499, 0, 1, 2, 3, 4, 5, 6, 7]  int64
```

---

## 3. Embedding Lookup

```python
hidden_states = embed_tokens(token_ids)   # [9, 7168]  bf16
```

Embedding matrix: `VocabParallelEmbedding [163840, 7168]`.

Each token ID maps to one row. The decode token (index 0) maps to the embedding of the last generated token for request B.

---

## 4. Layer 0: MLA + Dense MLP (first layer)

Layer 0 is the only layer using a dense MLP instead of MoE. The AttnRes bank starts empty (all zeros).

### 4.1 MLA forward: prefill tokens

**Input:** `hidden_states [9, 7168]` after `input_layernorm`.

**Step 1: fused_qkv_a_proj**

```
hidden [9, 7168] × W [2048, 7168]
→ q_a  [9, 1536]   ← Q low-rank activations
→ kv_a [9, 512]    ← KV latent (not used — kv_a is an intermediate, see Step 2)

After q_a_layernorm:
  q_a_norm [9, 1536]

# Example for prefill token 0 ("What", position=0):
q_a_norm[1, :5] ≈ [0.41, -0.23, 0.87, 0.12, -0.56, ...]
```

**Step 2: kv_a_proj_with_mqa**

```
hidden [9, 7168] × W [576, 7168]
→ c_kv   [9, 512]   (KV latent, normalized by kv_a_layernorm)
→ k_pe   [9, 64]    (RoPE key, position-dependent)

After kv_a_layernorm:
  c_kv_norm [9, 512]

# Apply rotary embedding to k_pe (NeoX-style):
# position=0: cos(0)=1, sin(0)=0 → no rotation
k_pe_rotated [9, 64]
```

**Step 3: q_b_proj**

```
q_a_norm [9, 1536] × W [96×192, 1536]
→ q [9, 96, 192]   bf16

split:
  q_nope [9, 96, 128]  ← NoPE dims
  q_pe   [9, 96, 64]   ← RoPE dims

q_pe_rotated = rotary_embedding(q_pe, positions)  [9, 96, 64]
```

**Step 4: Prefill — run_prefill_new_tokens**

Since request A has no prior context, this is a fresh prompt with no chunked-context step.

```
# Expand c_kv to full K/V for self-attention among 8 prefill tokens:
kv_b_proj(c_kv_norm[1:9]) → kv_expanded [8, 96*(128+128)] = [8, 24576]
k_nope = kv_expanded[:, :12288].view(8, 96, 128)
v      = kv_expanded[:, 12288:].view(8, 96, 128)

# fused_mla_key_concat_kv_cache_insert (PDL kernel):
#   Input: q_pe [8,96,64], q_nope [8,96,128], k_nope [8,96,128], v [8,96,128]
#          c_kv [8,512], k_pe_rotated [8,64]
#   → k_full [8, 96, 192] = cat(k_nope, k_pe_rotated per head)
#   → KV cache insert at slots 0..7 (request A's first 8 tokens):
#     kv_cache[block=0, offset=0..7, 0, :] = [c_kv, k_pe_rotated]
#     (576 values per slot: 512 c_kv + 64 k_pe)

# FlashAttention (causal, 8 query tokens × 8 KV tokens):
q_concat = cat(q_nope, q_pe_rotated, dim=-1)  [8, 96, 192]
attn_out [8, 96, 128] = flash_attn(q=q_concat, k=k_full, v=v, causal=True,
                                    scale=1/√192)

# Decode token (request B, position=499):
# BMM1: q_nope absorption
ql_nope = bmm(q_nope[0:1], W_UK_T)  [1, 96, 512]
# fused_mla_decode_q_concat_kv_cache_insert:
#   → q_concat [1, 96, 576], KV cache slot 499 ← [c_kv[0], k_pe_rotated[0]]
# forward_mqa over 500 slots (0..499):
attn_out_decode [1, 96, 512] = mqa_attention(ql_nope || q_pe_rotated, kv_cache, scale=1/√192)
# BMM2: W_UV projection
attn_out_decode_v = bmm(attn_out_decode, W_UV_T)  [1, 96, 128]
```

**Step 5: Output gate**

```
# g_proj on aux stream (9 tokens ≤ 512):
gate = sigmoid(g_proj(hidden_original))   [9, 1]

# Apply gate:
attn_out_combined = cat(attn_out_decode_v, attn_out[0:8, ...], dim=0)  [9, 96, 128]
attn_out_gated = attn_out_combined.reshape(9, -1) * gate   [9, 12288]

# o_proj:
attn_output = o_proj(attn_out_gated)  [9, 7168]
```

### 4.2 AttnRes write

```
# After attention at layer 0:
block_period = 93 // 12 ≈ 7
block_idx = 0 // 7 = 0   ← write to bank slot 0

attn_res_bank[0, :] = attn_output[t]  # write for the specific token position
# Bank starts empty; this is the first write.
```

### 4.3 Dense MLP (KimiMLP)

```
# After post_attention_layernorm:
normed [9, 7168]

gate, up = gate_up_proj(normed).split(33792, dim=-1)
# gate [9, 33792],  up [9, 33792]

# SiTU activation:
activated = sigmoid(4.0 * gate) * (gate + 25.0) × up   [9, 33792]

# For gate=0.5: situ(0.5) = sigmoid(2.0) × (0.5 + 25.0) = 0.880 × 25.5 = 22.4
# For gate=-1:  situ(-1)  = sigmoid(-4.0) × (-1 + 25.0) = 0.018 × 24   = 0.43

output = down_proj(activated)   [9, 7168]
```

### 4.4 AttnRes read

```
# After dense MLP:
scores = attn_res_norm(output) @ attn_res_proj   [9, 12]
weights = softmax(scores)   [9, 12]

# Bank has only slot 0 written (all others are zero):
residual = weights[:, 0:1] × attn_res_bank[0:1, :]   [9, 7168]
# (Other 11 slots are zero, so they contribute nothing.)

output = output + residual   [9, 7168]
```

---

## 5. Layer 1: KDA + MoE (first KDA layer)

### 5.1 KDA prefill: chunked Triton attention

```
# After input_layernorm at layer 1:
hidden [9, 7168]

# in_proj_qkvgfab:
proj_out = in_proj_qkvgfab(hidden)   [9, 37152]
# Reshape to [9, 96, 387]:
q       = proj_out[:, :, 0:128]      [9, 96, 128]
k       = proj_out[:, :, 128:256]    [9, 96, 128]
v       = proj_out[:, :, 256:384]    [9, 96, 128]
g       = proj_out[:, :, 384]        [9, 96]      ← sigmoid output gate
f_a     = proj_out[:, :, 385]        [9, 96]      ← conv1d forget gate
b_vec   = proj_out[:, :, 386:514]    [9, 96, 128] ← per-dim update gate
beta    = proj_out[:, :, 514]        [9, 96]      ← scalar update strength
```

**Prefill (8 tokens, request A):**

```python
# Triton chunked linear attention (chunk_size=64, but only 8 tokens here):
from ops.third_party.kda.chunk import chunk_gated_delta_rule

output_A, final_state_A = chunk_gated_delta_rule(
    q=q[1:9],    # [8, 96, 128]
    k=k[1:9],
    v=v[1:9],
    f=f_a[1:9],
    b=b_vec[1:9],
    beta=beta[1:9],
    g=g[1:9],
    initial_states=None,   # fresh prompt, no prior KDA state
    chunk_size=64,
)
# output_A: [8, 96, 128]   attention output for prefill tokens
# final_state_A: [1, 96, 128, 128]  delta memory after processing 8 tokens

# Store final_state_A as KDA state for request A (to be continued at decode)
kda_state_A = final_state_A   # will be used at next decode step for request A
```

**Decode (1 token, request B, has prior 500-token KDA state):**

```python
# Fused KDA decode kernel:
output_B = ops.fused_kda_decode(
    hidden=hidden[0:1],          # [1, 7168]  decode token
    in_proj_weight=...,          # [37152, 7168]
    out_proj_weight=...,         # [7168, 12288]
    conv1d_weight=...,           # [96, 4, 128]
    conv_state=conv_state_B,     # [1, 96, 4, 128]  ← in-place updated
    delta_state=delta_state_B,   # [1, 96, 128, 128] ← in-place updated
)
# output_B: [1, 7168]  (out_proj already applied inside kernel)
# conv_state_B and delta_state_B are updated in-place to reflect token 500
```

### 5.2 MoE routing and expert dispatch

```
# After post_attention_layernorm:
hidden_ffn = layer_norm(cat(output_B, output_A))  [9, 7168]

# Gate:
logits = gate_linear(hidden_ffn)     [9, 896]
scores = sigmoid(logits)             [9, 896]

# Biased top-16 for token 1 ("What", position=0):
scores_biased[1, :] = scores[1, :] + e_score_correction_bias
topk_ids[1, :] = argtopk(scores_biased[1], k=16)   # 16 expert IDs
raw_weights[1] = scores[1, topk_ids[1, :]]           # unbiased raw scores
renorm_weights[1] = raw_weights[1] / sum(raw_weights[1])

# Dispatch to 16 experts (MXFP4):
for i, eid in enumerate(topk_ids[1, :]):
    latent = routed_expert_down_proj[eid](hidden_ffn[1])    # [7168→3584]
    latent = latent_moe_norm(latent)                         # RMSNorm
    out_i  = routed_expert_up_proj(latent)                   # [3584→7168]
    contribution_i = renorm_weights[1, i] × out_i

routed_out[1] = sum(contribution_i for i in range(16))  [7168]

# Shared experts (2):
shared_out[1] = shared_expert_0(hidden_ffn[1]) + shared_expert_1(hidden_ffn[1])

ffn_output[1] = routed_out[1] + shared_out[1]   [7168]

# AttnRes write+read (same as layer 0):
attn_res_bank[0, :] = output_A[0]   (block_idx = 1//7 = 0)
# ... AttnRes read adds residual from bank
```

---

## 6. Layer 4: MLA + MoE (first standard MLA+MoE)

Layer 4 is the first `full_attn_layers` entry after the dense-prefix. By now, the hidden states have 4 layers of transformation (layers 0–3). The KV cache for request A has slots 0–7 from layer 0, and the decode token (request B) has slot 499 also from layer 0; subsequent MLA layers add their own KV cache pages.

**Key difference from layer 0 MLA:** Layer 4 uses MoE instead of Dense MLP.

```
# MLA at layer 4:
# New tokens for request A still use kv_b_proj (same as layer 0)
# Decode token uses BMM1 + paged MQA (same as layer 0)
# But now request A has 4 layers of context in its KV cache

# For the decode token at position 499 (request B):
# MLA forward_mqa reads 500 KV cache entries (positions 0..499)
# Each entry [576]: 512-dim c_kv + 64-dim k_pe
# Attention cost: 500 entries × 576-dim dot products × 96 heads

# After MLA: standard MoE (layer 4, not dense):
#   Gate logits [9, 896] → sigmoid → biased top-16 → latent MoE → output
```

---

## 7. After All 93 Layers: Logit Computation and Sampling

```python
# Final RMSNorm:
hidden_states = final_norm(hidden_states)   # [9, 7168]

# LM head:
logits = lm_head(hidden_states)   # [9, 163840]  bf16 → float32
```

For next-token prediction:
- Request A (prefill, 8 tokens): use `logits[8, :]` (last token "")
- Request B (decode, 1 token): use `logits[0, :]`

```python
# Greedy decoding for request A:
next_token_A = argmax(logits[8, :])   # e.g., token_id=9731 = "▁Paris"

# Decode to text:
tokenizer.decode([9731])   → "▁Paris"
```

---

## 8. Decode Step: Single Token

After prefill returns `"▁Paris"`, the model enters the decode loop with T=1:

```
new_token = "▁Paris"  (token_id=9731)
hidden_states = embed_tokens([9731])   # [1, 7168]
positions = [8]   # position 8 (after the 8-token prompt at positions 0-7)
```

### 8.1 Decode through a KDA layer

```
# Input: hidden [1, 7168]  (after input_layernorm)

# in_proj_qkvgfab:
proj_out = in_proj_qkvgfab(hidden)   [1, 37152]
# Extract q, k, v, g, f_a, b_vec, beta per head

# Fused KDA decode kernel:
output = ops.fused_kda_decode(
    hidden=hidden,
    in_proj_weight=...,
    out_proj_weight=...,
    conv1d_weight=...,
    conv_state=conv_state_A_layer1,   # [1, 96, 4, 128]  state from prior 8 tokens
    delta_state=delta_state_A_layer1, # [1, 96, 128, 128]
)

# Inside the kernel for head h=0:
# 1. Conv state update (kernel width 4):
#    Drop oldest k, insert new k_8 from projection
#    conv_state[:, 0, :3, :] ← conv_state[:, 0, 1:, :]  (shift)
#    conv_state[:, 0, 3, :]  ← k_new_head0              (insert)
#    k_filtered_h0 = conv1d(conv_state[:, 0, :, :], conv_weight[0])

# 2. Normalize:
#    k_norm = k_filtered / rms(k_filtered)

# 3. Retrieve from delta memory:
#    retrieved = delta_state[:, 0] @ k_norm   [128]

# 4. Update delta memory with position-9 input:
#    delta = v_head0 - retrieved
#    delta_state[:, 0] += (beta_gate × delta)[:, None] × k_norm[None, :]

# 5. Query from updated memory:
#    output_h0 = delta_state[:, 0].T @ q_norm_head0   [128]
#    output_h0 = sigmoid(g_head0) × output_h0

# Result: output [1, 7168]  (out_proj applied inside kernel)
# State updated in-place: conv_state_A_layer1, delta_state_A_layer1
```

**Total decode cost per KDA layer:** One fused kernel call, O(1) in sequence length. Memory read: `conv_state [96×4×128] + delta_state [96×128×128]` = ~3.1 MB regardless of sequence length.

### 8.2 Decode through an MLA layer

```
# Input: hidden [1, 7168]  (after input_layernorm)

# Step 1: projections
q_a = q_a_layernorm(fused_qkv_a_proj(hidden)[:, :1536])   [1, 1536]
c_kv, k_pe = kv_a_proj_with_mqa(hidden).split([512, 64], dim=-1)
k_pe_rotated = rotary_embedding(k_pe, positions=[8])       [1, 64]

# Step 2: q_b_proj
q = q_b_proj(q_a)   [1, 96, 192]
q_nope, q_pe = q.split([128, 64], dim=-1)
q_pe_rotated = rotary_embedding(q_pe, positions=[8])

# Step 3: BMM1 (W_UK_T absorption)
ql_nope = bmm(q_nope, W_UK_T)   [1, 96, 512]

# Step 4: fused decode Q concat + KV cache insert
#   Concatenate: q_concat = [ql_nope || q_pe_rotated]  [1, 96, 576]
#   Cache insert at slot=8: kv_cache[slot 8] = [c_kv, k_pe_rotated]

# Step 5: paged MQA attention over 9 tokens (0..8)
#   Reads 9 KV cache entries: slots 0..8
#   Each 576 dims: 512-dim c_kv dot + 64-dim k_pe_rope dot
attn_latent = forward_mqa(q_concat, kv_cache, attn_metadata)   [1, 96, 512]

# Step 6: BMM2
attn_v = bmm(attn_latent, W_UV_T)   [1, 96, 128]
attn_out = o_proj(attn_v.reshape(1, -1))   [1, 7168]

# (+ output gate on aux stream)
```

**Total decode cost per MLA layer:** Proportional to sequence length (9 KV entries here). Memory read: 9 × 576 × 2 bytes = 10 KB. At 128K tokens: 128K × 576 × 2 bytes = 148 MB per MLA layer per step.

---

## 9. Long-Context Decode: 32K Tokens

To illustrate the K3 efficiency profile at long context:

### 9.1 MLA layer: paged attention cost

```
seq_len = 32768 (32K tokens, request has 32K history)
Position = 32767 (new decode token)

KV reads per MLA layer:
  32768 slots × 576 bytes/slot = 18.9 MB
  → 24 MLA layers × 18.9 MB = 453 MB per step

Computation per MLA layer:
  QK dot: T × 576-dim × 96 heads = 32768 × 576 × 96 ≈ 1.81 TFLOPs/layer  (very small)
  Attention and V: similar

Total MLA memory bandwidth (decode):
  24 layers × 18.9 MB = 453 MB of KV data reads
  → At 3.35 TB/s (H100 HBM): ~0.14 ms just for KV reads at 32K context
```

### 9.2 KDA layer: O(1) decode cost

```
seq_len = 32768 (same sequence)

KDA state per layer (FIXED):
  conv_state:  [1, 96, 4, 128] × 2 bytes = 98 KB
  delta_state: [1, 96, 128, 128] × 2 bytes = 3.1 MB
  Total: ~3.2 MB

KDA state reads per layer: 3.2 MB  (SAME at 32K or 1M tokens)
→ 69 KDA layers × 3.2 MB = 221 MB per step  (constant regardless of seq_len)

Computation per KDA layer:
  Matrix read + outer product + matrix-vector: O(kDimK² × H) per token
  = 128² × 96 ≈ 1.57M FLOPs per decode step (tiny)

Comparison:
  MLA at 32K:  453 MB KV reads (linear in seq_len)
  KDA at 32K:  221 MB state reads (CONSTANT)
  KDA at 1M:   221 MB state reads (SAME — no growth!)

Total memory reads per decode step at 32K:
  MLA (24 layers): 453 MB
  KDA (69 layers): 221 MB
  Total: ~674 MB
  → At 3.35 TB/s HBM: ~0.20 ms memory-bound time

At 1M tokens:
  MLA (24 layers): 24 × 576 × 1M × 2 = 27.6 GB  (massive)
  KDA (69 layers): 221 MB  (unchanged)
  → The 3:1 ratio means KDA dominates decode cost at very long contexts
```

**This is the fundamental design advantage of K3:** For very long contexts, the 69 KDA layers have constant decode cost while MLA cost grows linearly. The 3:1 ratio ensures that at 128K tokens, KDA state reads (~221 MB) are small compared to MLA KV reads (~14 GB for 24 layers), but the KDA layers still provide meaningful recurrent context.

---

## 10. Decode Step: Long Context (32K Tokens)

**Setup:** 1 decode token, `position=32767`, `TP=8`, SP disabled, SM100 hardware.

### 10.1 Per-layer cost breakdown

**MLA layer (e.g., layer 4) at 32K context:**

```
KV cache reads:
  32768 slots × 576 bytes/slot = 18,874,368 bytes ≈ 18.9 MB per MLA layer
  (576 = 512 c_kv dims + 64 k_pe dims, each BF16 = 2 bytes per value)

KV read time at 3.35 TB/s HBM:
  18.9 MB / 3.35 TB/s ≈ 5.6 µs per MLA layer

Additional GEMM ops (q_a_proj, kv_a_proj, q_b_proj, o_proj):
  Each is [1, 7168] × W — negligible at T=1 (~0.5 µs total)

Total per MLA layer: ~5–6 µs  (KV reads dominate)
```

**KDA layer (e.g., layer 1) at 32K context:**

```
State reads (FIXED, independent of sequence length):
  conv_state:  [1, 96, 4, 128] × 2 bytes = 98,304 bytes ≈ 96 KB
  delta_state: [1, 96, 128, 128] × 2 bytes = 3,145,728 bytes ≈ 3.0 MB
  Total: ~3.1 MB per KDA layer

State read time at 3.35 TB/s HBM:
  3.1 MB / 3.35 TB/s ≈ 0.9 µs per KDA layer

in_proj_qkvgfab GEMM: [1, 7168] × [37152, 7168] ≈ 0.4 µs
out_proj GEMM:        [1, 12288] × [7168, 12288] ≈ 0.3 µs

Total per KDA layer: ~1.5 µs
```

**MoE layer (SM100 TAIL_FUSION, T=1):**

```
CollectiveKernel (NVLS allreduce + RMSNorm + reduce-scatter shared):
  Input: [1, 3584] routed latent + [1, 7168] shared output
  NVLS symmetric memory round-trip: ~4 µs

AdaptiveUpProjectionKernel (M=1 → static skinny GEMM):
  [1, 3584] × [3584, 7168/8] = [1, 896] per TP rank
  Very small GEMM, multicast result: ~5 µs

LamportCopyKernel (32 CTAs × 224 threads reading mailbox):
  Spin on sentinel, copy [1, 7168] to output: ~3 µs

Total per MoE tail: ~12 µs
(Note: the per-expert down_proj dispatch runs separately via FusedMoE/MegaMoE
 and adds ~10–15 µs before the tail; shown below in aggregate.)
```

### 10.2 Total latency estimate and comparison

K3 has 93 total layers: 24 MLA, 69 KDA, all 93 with MoE (except layer 0 which uses dense MLP).

```
Component                    Count   Per-layer   Subtotal
──────────────────────────────────────────────────────────
MLA layers (KV reads)         24     ~6 µs       ~144 µs
KDA layers (state reads)      69     ~1.5 µs     ~104 µs
MoE dispatch (FusedMoE)       93     ~15 µs      ~1395 µs
MoE tail (TAIL_FUSION T=1)    93     ~12 µs      ~1116 µs
GEMMs (in_proj, o_proj etc.)  93     ~2 µs       ~186 µs
──────────────────────────────────────────────────────────
Total                                            ~2945 µs ≈ 3 ms per token
```

These are rough order-of-magnitude estimates assuming memory-bound operation. In practice, kernel overlap and pipelining reduce wall-clock time.

**Comparison: 32K vs 1K context:**

```
Component         1K context     32K context    Ratio
──────────────────────────────────────────────────────
MLA (24 layers)   ~4 µs/layer    ~6 µs/layer    ~1.5×
KDA (69 layers)   ~1.5 µs/layer  ~1.5 µs/layer  1.0×  ← O(1)
MoE (93 layers)   ~27 µs/layer   ~27 µs/layer   1.0×
──────────────────────────────────────────────────────
Total MLA cost    ~96 µs         ~144 µs
Total KDA cost    ~104 µs        ~104 µs
MoE cost          ~2511 µs       ~2511 µs
```

At 32K context, MLA layers are only 1.5× more expensive than at 1K (because 32× more KV reads, but KV per layer is still small relative to MoE). KDA cost is unchanged. MoE dominates total decode time at both context lengths for T=1 on SM100 TAIL_FUSION.

---

## 11. Sequence Parallelism Example

**Setup:** SP enabled, `TP=8`, 512 tokens in batch.

Under sequence parallel (SP), the token dimension is split across TP ranks:

```
Each rank holds [512/8 = 64, 7168]  between collectives
```

### 11.1 MLA+MoE layer with SP

**MLA attention with SP:**

```
Start: each rank holds [64, 7168]

sp_all_gather (token dim):
  [64, 7168] → [512, 7168]   (8 ranks contribute, each sends 64 tokens)
  Cost: 512 × 7168 × 2 bytes × (8-1)/8 = ~7.7 MB transferred per rank

Attention forward (all 512 tokens visible):
  q, k, v computed from [512, 7168]
  FlashAttention causal: [512, 96, 192] queries × [512, 96, 192] keys
  attn_out: [512, 96, 128]

o_proj: [512, 12288] × [12288, 7168] → [512, 7168]

sp_reduce_scatter (token dim):
  [512, 7168] → [64, 7168]   (scatter result back, each rank owns 64 tokens)
  Cost: same ~7.7 MB per rank
```

**MoE with SP:**

```
Start: each rank holds [64, 7168]  (after attention reduce-scatter)

sp_all_gather:
  [64, 7168] → [512, 7168]   (same as attention gather)

Gate + routing:
  logits = gate_linear([512, 7168]) → [512, 896]   sigmoid → biased top-16
  topk_ids [512, 16], topk_weights [512, 16]

Expert dispatch (FusedMoE / MegaMoE):
  Dispatch all 512 tokens to experts → [512, 3584] latent per expert
  Latent MoE tail (COLUMN_PARALLEL at T=512):
    [512, 3584] @ up_proj_shard [3584, 7168/8] → [512, 896] per TP rank
    all_reduce → [512, 7168]

Shared experts:
  shared_expert_0([512, 7168]) + shared_expert_1([512, 7168])

sp_reduce_scatter:
  [512, 7168] → [64, 7168]

End: each rank holds [64, 7168]
```

Each rank processes its 64-token shard between all-gather and reduce-scatter. The attention and MoE computations see the full 512-token sequence, preserving model correctness.

**Collective summary per layer:**

```
Operation              Shape transferred    Direction
─────────────────────────────────────────────────────
Attention all-gather   [512, 7168] BF16     all ranks → each rank
Attention reduce-scat  [512, 7168] BF16     each rank → all ranks
MoE all-gather         [512, 7168] BF16     all ranks → each rank
MoE reduce-scatter     [512, 7168] BF16     each rank → all ranks
─────────────────────────────────────────────────────
Total per layer:       4 × ~7.7 MB = ~30 MB collective traffic per rank
```

### 11.2 GEMM-RS fusion benefit

Without GEMM-RS fusion (standard SP):

```
o_proj GEMM:       [512, 12288] × [12288, 7168] → [512, 7168]   (full output)
all_reduce:        [512, 7168] → [512, 7168]                     (TP reduce)
reduce_scatter:    [512, 7168] → [64, 7168]                      (SP scatter)

Collectives:  2 operations (all_reduce + reduce_scatter)
HBM writes:   [512, 7168] BF16 = 7.3 MB  (intermediate all-reduce output)
```

With GEMM-RS fusion (enabled when `enable_gemm_rs=True`):

```
o_proj + reduce_scatter (fused):
  [512, 12288] × [12288, 7168/8] → [64, 7168]  (each rank's output shard directly)

Collectives:  1 operation (fused reduce-scatter)
HBM writes:   [64, 7168] BF16 = 0.9 MB  (direct shard output, 8× smaller)
```

The fusion merges the GEMM with the reduce-scatter: instead of computing the full `[512, 7168]` output and then scattering, each TP rank computes only its `[512, 7168/8]` column shard and contributes it to the reduce-scatter in flight. This eliminates one collective and one 7.3 MB intermediate HBM write per `o_proj` call — a meaningful saving when attention runs 24 MLA layers and MoE has 93 layers per forward pass.

```
Saving per MLA layer (o_proj): 1 allreduce eliminated, ~7.3 MB write avoided
At 24 MLA layers:               24 × 7.3 MB = ~175 MB HBM writes saved per step
At 93 MoE layers (down_proj):  up to 93 × similar = ~679 MB additional savings
```
