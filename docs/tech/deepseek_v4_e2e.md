# DeepSeek V4: End-to-End Example

[← Index](deepseek_index.md) | [Attention](deepseek_v4_attention.md) | [MoE](deepseek_v4_moe.md)

Walks the full inference pipeline from raw text input to the generated next token. Uses concrete shapes, concrete numbers, and explicit tensor transformations at every step. Covers both prefill (first token) and decode (subsequent tokens), and shows how C4A and C128A layers differ in the same forward pass.

---

## Table of Contents

1. [Setup: Batch and Prompt](#1-setup-batch-and-prompt)
2. [Tokenization](#2-tokenization)
3. [Embedding Lookup](#3-embedding-lookup)
4. [Layer 0: C128A + Hash MoE (first layer)](#4-layer-0-c128a--hash-moe-first-layer)
   - 4.1 [Parallel Input GEMMs](#41-parallel-input-gemms)
   - 4.2 [Q Up-projection + SWA Insert](#42-q-up-projection--swa-insert)
   - 4.3 [Indexer Compressor: Save States](#43-indexer-compressor-save-states)
   - 4.4 [Indexer Q Quantization](#44-indexer-q-quantization)
   - 4.5 [MQA Logits and Top-K (first layer: no history yet)](#45-mqa-logits-and-top-k-first-layer-no-history-yet)
   - 4.6 [Main Compressor (aux stream)](#46-main-compressor-aux-stream)
   - 4.7 [Main MLA Attention: SWA + Sparse](#47-main-mla-attention-swa--sparse)
   - 4.8 [Output Projection](#48-output-projection)
   - 4.9 [Hash MoE](#49-hash-moe)
5. [Layer 2: C4A + Hash MoE (first C4A layer)](#5-layer-2-c4a--hash-moe-first-c4a-layer)
   - 5.1 [Compressor fires at position 3](#51-compressor-fires-at-position-3)
   - 5.2 [Indexer scoring with fresh compressed history](#52-indexer-scoring-with-fresh-compressed-history)
6. [Layer 5: C128A + Standard noaux_tc MoE](#6-layer-5-c128a--standard-noaux_tc-moe)
   - 6.1 [Standard MoE routing example](#61-standard-moe-routing-example)
7. [After All 61 Layers: Logit Computation and Sampling](#7-after-all-61-layers-logit-computation-and-sampling)
8. [Decode Step: One Token at a Time](#8-decode-step-one-token-at-a-time)
   - 8.1 [New token embedding](#81-new-token-embedding)
   - 8.2 [C4A decode attention](#82-c4a-decode-attention)
   - 8.3 [C128A decode attention](#83-c128a-decode-attention)
   - 8.4 [Decode MoE](#84-decode-moe)
9. [Long-Context Decode: 128K Tokens](#9-long-context-decode-128k-tokens)
   - 9.1 [Coverage breakdown: C4A vs C128A](#91-coverage-breakdown-c4a-vs-c128a)
   - 9.2 [Compressed index expansion](#92-compressed-index-expansion)
10. [Speculative Decode Step (DSpark)](#10-speculative-decode-step-dspark)
11. [DCP (Decode Context Parallelism) Example](#11-dcp-decode-context-parallelism-example)
12. [Key Implementation Files](#12-key-implementation-files)

---

## 1. Setup: Batch and Prompt

```
Input text:  "The Eiffel Tower is in Paris."
Task:        Next-token prediction (language model prefill + one decode step)

Batch:
  Request A:  7-token prompt (prefill)
  Request B:  continuing sequence, 4000 tokens of history (decode)

Tensor parallelism: TP=1 (single GPU) for simplicity
Layer shown in detail: layer 0 (C128A + Hash MoE)
```

---

## 2. Tokenization

The DeepSeek tokenizer (BPE, vocab_size=129280) segments the prompt:

```
"The Eiffel Tower is in Paris."

→ tokens:    ["The", "▁Eiffel", "▁Tower", "▁is", "▁in", "▁Paris", "."]
→ token_ids: [ 791,   96310,     17720,    374,   304,   14366,    13]

T = 7 prefill tokens + 1 decode token = 8 total tokens in the batch
Token layout (vLLM convention: decode tokens first):
  index 0: decode token (request B, position 3999)
  index 1: prefill token 0 ("The", position 0 in req A)
  index 2: prefill token 1 ("▁Eiffel", position 1)
  index 3: prefill token 2 ("▁Tower", position 2)
  index 4: prefill token 3 ("▁is", position 3)
  index 5: prefill token 4 ("▁in", position 4)
  index 6: prefill token 5 ("▁Paris", position 5)
  index 7: prefill token 6 (".", position 6)

positions = [3999, 0, 1, 2, 3, 4, 5, 6]   [T=8] int64
```

---

## 3. Embedding Lookup

```python
hidden_states = embed_tokens(token_ids)   # [8, 7168]  bf16
```

Embedding matrix: `[129280, 7168]`. Each token maps to one row:

```
hidden_states[0] = embed_tokens.weight[token_id_of_decode_token]  # [7168]
hidden_states[1] = embed_tokens.weight[791]    # "The"
hidden_states[2] = embed_tokens.weight[96310]  # "▁Eiffel"
...
hidden_states[7] = embed_tokens.weight[13]     # "."
```

No positional encoding added here — position enters via RoPE inside each attention layer.

The decoder loop now runs `hidden_states` through 61 attention+MoE layers. We walk layer 0 in full detail, then highlight differences at other layers.

---

## 4. Layer 0: C128A + Hash MoE (first layer)

**Layer type:** C128A (compress_ratio=128)
**MoE type:** Hash MoE
**Context for decode token:** position 3999 (request B has 4000 tokens of prior history)
**Context for prefill tokens:** positions 0–6 (fresh sequence, no prior history)

### 4.1 Parallel Input GEMMs

Four GEMMs start simultaneously on 4 CUDA streams:

```
Default stream:
  fused_wqa_wkv.weight: [2048, 7168]  fp8
  hidden_states [8, 7168] × weight.T [7168, 2048]
  → output [8, 2048]
  → qr [8, 1536]  (first 1536 cols)
  → kv [8, 512]   (last 512 cols)

Aux stream 0 (main compressor):
  main_compressor.fused_wkv_wgate.weight: [2048, 7168]
  → main_kv_score [8, 2048]

Aux stream 1 (indexer weights):
  indexer.weights_proj.weight: [64, 7168]
  → indexer_weights [8, 64]   float32

Aux stream 2 (indexer compressor):
  indexer.compressor.fused_wkv_wgate.weight: [512, 7168]
  → idx_kv_score [8, 512]

# All 4 streams join → fused_q_kv_rmsnorm:
qr [8, 1536] ← RMSNorm(qr, q_norm.weight [1536])
kv [8, 512]  ← RMSNorm(kv, kv_norm.weight [512])
```

**Concrete Q/KV values (token 1, "The"):**

```
Before rmsnorm:
  qr[1, :5] = [0.41, -0.23, 0.87, 0.12, -0.56, ...]
  kv[1, :5] = [-0.18, 0.63, 0.09, -0.44, 0.31, ...]

After q_norm:
  rms_q = sqrt(mean(qr[1,:]²) + 1e-6)  ≈ 0.449
  qr[1, :5] ← [0.41/0.449×w[0], -0.23/0.449×w[1], ...]  ≈ [0.914×w[0], -0.512×w[1], ...]

After kv_norm:
  rms_kv = sqrt(mean(kv[1,:]²) + 1e-6)  ≈ 0.372
  kv[1, :5] ← [-0.484×w[0], 1.694×w[1], ...]
```

### 4.2 Q Up-projection + SWA Insert

On the default stream, after qr and kv are normalized:

```python
# wq_b: [1536 → 128×512] = [65536, 1536]
q = wq_b(qr)          # [8, 128, 512]  bf16

fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q, kv, swa_kv_cache, swa_slot_mapping, positions, cos_sin_cache
)
```

**Per-head RMSNorm on Q (no learned weight, unit normalization):**

```
For token 1 ("The"), head 0:
  q[1, 0, :] = [1.23, -0.45, 0.87, ..., 0.12]   [512 values]
  rms = sqrt(mean(q[1,0,:]²) + 1e-6)  ≈ 0.731
  q[1, 0, :] ← q[1, 0, :] / 0.731 = [1.683, -0.616, 1.190, ..., 0.164]
  # Now ||q[1,0,:]||_2 ≈ √512 ≈ 22.6  (unit variance, not unit norm)
```

**GPT-J RoPE on last 64 dims of Q (for token 1 at position 0):**

```
# position = 0: cos(0)=1, sin(0)=0 for all frequencies
# → rotation has no effect at position 0
# (cos and sin tables vary with position; at pos=0 all freqs give cos=1, sin=0)

# For token 0 (decode, position 3999), head 0:
for k in range(32):
  a = q[0, 0, 448 + 2k]
  b = q[0, 0, 449 + 2k]
  cos_k = cos_sin_cache[3999, k]      # e.g. for k=0: cos_k ≈ 0.641
  sin_k = cos_sin_cache[3999, 32 + k] # sin_k ≈ 0.768
  q[0, 0, 448+2k]   = a*0.641 - b*0.768
  q[0, 0, 449+2k]   = b*0.641 + a*0.768
```

**SWA KV cache insert for decode token (position 3999):**

```
kv[0, :] = kv latent for decode token  [512]  (already normalized)

# RoPE on kv[0, 448:] at position 3999:
for k in range(32):
  a = kv[0, 448 + 2k]
  b = kv[0, 449 + 2k]
  kv[0, 448+2k]   = a*cos(3999,k) - b*sin(3999,k)
  kv[0, 449+2k]   = b*cos(3999,k) + a*sin(3999,k)

# FP8 block-quantize NoPE portion kv[0, :448] (7 blocks of 64):
Block 0 (dims 0-63):
  slice = kv[0, 0:64]
  amax = max(|slice|) = 0.91
  exponent = ceil(log2(0.91/448)) = ceil(-8.94) = -8
  ue8m0[0] = -8 + 127 = 119
  scale = 2^(-8) = 0.003906
  kv_fp8[0, 0:64] = clamp(slice / 0.003906, -448, 448).to(fp8)
  # Example: kv[0,3] = 0.47 → 0.47/0.003906 = 120.3 → fp8(120.3) = 120

Block 1 (dims 64-127): similar with its own amax and scale

# Write 584-byte slot to SWA cache:
swa_slot_mapping[0] = 5000   (physical slot for decode token in SWA paged cache)
swa_kv_cache[5000//block_size, 5000%block_size, :] = {
    kv_fp8_nope [448 bytes], kv_bf16_rope [128 bytes], ue8m0_bytes [7+1 bytes]
}
```

**SWA insert for prefill tokens (positions 0–6):**

```
For token 1 ("The", position 0):
  kv[1, 448:] = kv[1, 448:]  (position=0: cos=1, sin=0, rope is no-op)
  FP8 quantize kv[1, :448] → store at swa_slot_mapping[1]

For token 4 ("▁is", position 3):
  Apply GPT-J RoPE at position 3
  FP8 quantize → store at swa_slot_mapping[4]
```

### 4.3 Indexer Compressor: Save States

Runs on aux stream 0. compress_ratio=128, so compression fires only when position % 128 == 127.

```
idx_kv_score [8, 512]  (from indexer.compressor.fused_wkv_wgate)
split:
  idx_kv    [8, 256]
  idx_score [8, 256]

For each token t, position p = positions[t]:
  ape_pos = p % 128
  idx_kv[t]    += ape[ape_pos, :256]   # learned offset for this position in the 128-group
  idx_score[t] += ape[ape_pos, 256:]
  write to indexer.compressor.state_cache at state_slot[t]

Positions in this batch:
  t=0 (decode, p=3999): ape_pos = 3999 % 128 = 63   → ape[63, :]
  t=1 (prefill, p=0):   ape_pos = 0                  → ape[0, :]
  ...
  t=7 (prefill, p=6):   ape_pos = 6                  → ape[6, :]

C128A compress_norm_rope_store fires when (p+1) % 128 == 0:
  → p=127, 255, ..., 3968, 3999-should be 3999 but 3999%128=63 ≠ 127
  Wait: 3999 % 128 = 63, NOT 127. So for decode at position 3999, NO compression fires.
  For prefill at positions 0-6: none are ≡ 127 mod 128, so no compression fires.

  → This batch produces NO new compressed indexer K entries.
     The indexer k_cache already has 3999//128 = 31 compressed entries from prior decode steps.
```

### 4.4 Indexer Q Quantization

Overlapped with 4.3 on aux stream 0.

```python
wq_b(qr) → q_idx [8, 64, 128]   bf16   (indexer's own wq_b, separate from main MLA wq_b)
fused_indexer_q_rope_quant(q_idx, indexer_weights, positions, ...)
```

**For decode token (t=0, position=3999, head h=3):**

```
q_idx[0, 3, :] = [0.42, -0.31, 0.89, ...,  # dims 0-63: NoPE
                  0.17, -0.24, 0.63, ...]   # dims 64-127: RoPE (to be rotated)

# GPT-J interleaved RoPE on dims 64-127 (32 pairs):
k = 0 (pair of dims 64, 65):
  a = q_idx[0, 3, 64]  = 0.17
  b = q_idx[0, 3, 65]  = -0.24
  cos = indexer_cos_sin_cache[3999, 0]   # using indexer's own cos/sin (may differ from main)
  sin = indexer_cos_sin_cache[3999, 32]
  # cos_0 at pos 3999: theta_0 = 10000^(-0/32) = 1.0 → cos(3999) = cos(3999 rad) ≈ -0.904
  # sin_0 at pos 3999: sin(3999) ≈ 0.427
  r_even = 0.17×(-0.904) - (-0.24)×0.427 = -0.154 + 0.103 = -0.051
  r_odd  = (-0.24)×(-0.904) + 0.17×0.427 = 0.217 + 0.073 = 0.290

# Compute amax over all 128 dims (nope + rotated rope):
amax = max(|nope_dims|, |r_even_all|, |r_odd_all|) = 0.89

# UE8M0 FP8 scale:
scale_raw = 0.89 / 448.0 = 0.001987
exponent  = ceil(log2(0.001987)) = ceil(-8.977) = -8
q_scale   = 2^(-8) = 0.003906

# Quantize:
q_fp8[0, 3, 0]   = clamp(0.42/0.003906, -448, 448)    = clamp(107.5, -448, 448) → fp8(107.5) = 107
q_fp8[0, 3, 64]  = clamp(-0.051/0.003906, -448, 448)  = clamp(-13.1, -448, 448) → fp8(-13.1) = -13
q_fp8[0, 3, 65]  = clamp(0.290/0.003906, -448, 448)   = clamp(74.2, -448, 448)  → fp8(74.2)  = 74

# Scale absorption (FP8 path):
weights_out[0, 3] = indexer_weights[0, 3] × q_scale × (1/√128) × (1/√64)
                  = 0.041 × 0.003906 × 0.08839 × 0.125
                  ≈ 0.041 × 0.003906 × 0.011049
                  ≈ 1.768e-6
```

### 4.5 MQA Logits and Top-K (first layer: no history yet)

**Prefill tokens (positions 0–6):**

```
Compressed entries in indexer k_cache for this sequence: 0
(Request A is a fresh 7-token prompt; no compression has fired yet for positions 0-127)

cu_seqlen_ke for C128A:
  ke[i] = (pos_i + 1) // 128
  pos=0: ke=0,  pos=1: ke=0, ..., pos=6: ke=0
  → ALL prefill tokens see 0 compressed KV entries → no indexer scoring for prefill

topk_indices_buffer[1..7, :] = -1   (all padding)
```

**Decode token (position 3999, request B):**

```
Request B has 31 compressed indexer K entries (positions 0..30 in compressed space,
covering original positions 0..3967).
compressed_seq_len = 4000 // 128 = 31   (floor division; pos 3968-3999 not yet compressed)

fp8_fp4_paged_mqa_logits(
    q_fp8 = q_fp8[0:1].reshape(1, 1, 64, 128),   # [B=1, next_n=1, 64, 128]
    kv_cache = indexer_k_cache,
    weights = weights_out[0:1],                    # [1, 64]
    context_lens = [[31]],                         # 31 compressed entries
    block_tables = block_table_B,
    max_model_len = max_model_len // 128
) → logits [1, max_len//128]  float32   (valid range: [0..30])

# Example logits (31 values for the 31 compressed entries):
logits[0, :31] = [0.83, 0.71, 0.92, 0.45, 0.88, ..., 0.79]
logits[0, 31:] = -inf   (beyond compressed history)

# Top-1024 from 31 entries: select all 31 (31 << 1024)
top_k_per_row_decode(logits, seq_lens=[[31]], K=1024)
topk_indices_buffer[0, :31]  = [2, 4, 0, 7, ..., 30]   # sorted by score
topk_indices_buffer[0, 31:] = -1
```

### 4.6 Main Compressor (aux stream)

```
main_kv_score [8, 2048]  (from main_compressor.fused_wkv_wgate)
split: main_kv [8, 512], main_score [8, 512]

save_partial_states: saves all 8 tokens to main compressor state_cache

compress_norm_rope_store: fires if (position+1) % 128 == 0
  For this batch (positions 3999, 0, 1, 2, 3, 4, 5, 6):
  position 3999: 3999+1=4000, 4000%128=0? No, 4000=31×128+32, 4000%128=32≠0.
  Wait: 3999+1=4000, and 4000/128=31.25, so 4000%128=4000-31×128=4000-3968=32≠0.
  None of the 8 positions trigger compression.
  → No new main MLA KV entries are written this step.
  → Main MLA KV cache for request A has 0 compressed entries.
  → Main MLA KV cache for request B already has 3999//128=31 entries from prior steps.
```

### 4.7 Main MLA Attention: SWA + Sparse

**Decode token (position 3999, request B):**

```
SWA coverage: last window_size=4096 tokens
  → positions max(0, 3999-4095)=0 to 3999
  → all 4000 tokens are within SWA window (4000 < 4096)
  swa_indices [1, 1, 4000]: global physical slot IDs for all 4000 tokens in request B's history

Sparse (expanded from compressed indexer top-K):
  topk_indices_buffer[0, :31] = [2, 4, 0, 7, ..., 30]   (31 compressed positions)
  For C128A (compress_ratio=128), each compressed ck → 128 original slots:
    ck=2 → original positions 256..383 → 128 physical slots in main MLA KV cache
    ck=4 → original positions 512..639 → 128 physical slots
    ...
  global_topk_indices [1, 1, 31×128=3968]   (but only 31 valid, rest padded to -1)
  Actual valid entries: 31×128 = 3968 slots

flash_mla_with_kvcache(
    q = q[0:1].unsqueeze(1),           # [1, 1, 128, 512]
    k_cache = swa_kv_cache,            # SWA cache (all 4000 tokens at full resolution)
    indices = swa_indices,             # [1, 1, 4000]  global slot IDs
    topk_length = [4000],              # valid entries
    extra_k_cache = main_mla_kv_cache, # compressed 512-dim cache
    extra_indices_in_kvcache = global_topk_indices,  # [1, 1, 3968]
    extra_topk_length = [3968],
    softmax_scale = 1/√512 ≈ 0.04419,
    attn_sink = attn_sink,
)

# For this short sequence (4000 < 4096):
# SWA covers positions 0..3999 (all 4000 tokens)
# Sparse covers original positions 256..383, 512..639, 0..127, 896..1023, ...
#   (those corresponding to the 31 selected compressed entries)
# The FlashMLA kernel internally merges SWA and sparse indices, deduplicating overlaps.
```

**Prefill tokens (positions 0–6):**

```
N = 0 compressed main MLA entries (no compression fired for this fresh sequence)
SWA gather_lens = min(pos+1, window_size): [1, 2, 3, 4, 5, 6, 7] for positions 0-6

Gather SWA KV into workspace:
  workspace [1, 7, 512]  (7 SWA tokens, dequantized to BF16)

combine_topk_swa_indices:
  For token at position 0: topk_len=0, swa_len=1 → combined=[workspace col 0]
  For token at position 3: topk_len=0, swa_len=4 → combined=[cols 0,1,2,3]
  For token at position 6: topk_len=0, swa_len=7 → combined=[cols 0,1,2,3,4,5,6]

flash_mla_sparse_fwd(
    q = q[1:8],            # [7, 128, 512]
    kv = workspace[0],     # [7, 1, 512]
    indices = combined_indices,
    sm_scale = 1/√512,
    ...
)
# Attention is purely causal within the 7 prefill tokens.
# No compressed history exists yet for request A.
```

### 4.8 Output Projection

```
attn_out [8, 128, 512]   bf16

# fused_inv_rope_fp8_quant:
# For each token t, head h, undo the forward GPT-J RoPE:
For token 0 (position 3999), head 0:
  nope_dims [0..447]: pass through unchanged
  for k in range(32):  # rope dims
    y0 = attn_out[0, 0, 448+2k]
    y1 = attn_out[0, 0, 449+2k]
    cos = cos_sin_cache[3999, k]   ≈ -0.904  (same cos/sin used in forward)
    sin = cos_sin_cache[3999, 32+k] ≈ 0.427
    # Inverse rotation:
    out_even = y0*(-0.904) + y1*(0.427)    # note: +y1*sin (not -y1*sin)
    out_odd  = y1*(-0.904) - y0*(0.427)    # note: -y0*sin (not +y0*sin)
    attn_unrot[0, 0, 448+2k]   = out_even
    attn_unrot[0, 0, 449+2k]   = out_odd

# Verification: applying forward then inverse rotation gives identity:
# forward:  [a,b] → [a*c-b*s, b*c+a*s]
# inverse:  [a*c-b*s, b*c+a*s] → (a*c-b*s)*c + (b*c+a*s)*s = a*c² + a*s² = a ✓

# FP8 block quantize (4 blocks per 512-dim head):
For head 0, block 0 (dims 0-127):
  amax = max(|attn_unrot[0, 0, 0:128]|) = 0.76
  scale = 2^(ceil(log2(0.76/448))) = 2^(-9) = 0.001953
  o_fp8[0, group_g, head_offsets+0:128] = fp8(attn_unrot / 0.001953)

# wo_a grouped BMM (o_groups=16):
fp8_einsum("bgd,gdr→bgr", o_fp8 [8, 16, 8*512=4096], wo_a [16, 4096, 1024])
→ z [8, 16, 1024]  bf16

# wo_b:
z.flatten(1) [8, 16384]
→ wo_b [16384, 7168]
→ output [8, 7168]
```

### 4.9 Hash MoE

```
# After layer norm on output from attention:
hidden_states [8, 7168]

# For token 1 ("The", token_id=791):
topk_ids = tid2eid[791] = [17, 203, 91, 5, 318, 7]

logits = gate(hidden_states[1])   [384]  bf16
scores = sqrtsoftplus(logits)     [384]  float32

# At hash-selected positions only:
raw = scores[[17, 203, 91, 5, 318, 7]] = [1.134, 1.033, 0.832, 1.262, 0.660, 0.969]
                                          ↑ expert17  ↑ expert203  ...

renorm = raw / 5.890 = [0.193, 0.175, 0.141, 0.214, 0.112, 0.165]
final_weights = renorm * 2.5 = [0.483, 0.438, 0.353, 0.535, 0.280, 0.413]

# Expert dispatch (MXFP4 SwiGLU, TP=1):
For expert 17 (weight=0.483):
  gate = W1_gate[17] @ hidden[1]   [3072]
  up   = W1_up[17]   @ hidden[1]   [3072]
  act  = silu(clamp(gate, -10, 10)) * up   [3072]
  out  = W2[17] @ act * 0.483   [7168]

# Sum across 6 experts + shared:
routed_out = sum(E17_out + E203_out + E91_out + E5_out + E318_out + E7_out)
shared_out = W2_sh @ silu(clamp(W1_sh_gate @ h, -10, 10)) * W1_sh_up @ h   [7168]
hidden_after_moe = routed_out + shared_out   [7168]
```

---

## 5. Layer 2: C4A + Hash MoE (first C4A layer)

Layer 2 is the first C4A layer. After 2 layers of C128A processing, the hidden states now carry more semantic content. compress_ratio=4 means compression fires every 4 tokens.

### 5.1 Compressor fires at position 3

```
Tokens in batch: positions [3999, 0, 1, 2, 3, 4, 5, 6]
compress_ratio = 4

check: (p+1) % 4 == 0?
  p=3999: (3999+1) % 4 = 4000 % 4 = 0 → YES  (decode token triggers compression!)
  p=0: 1 % 4 ≠ 0 → no
  p=3: 4 % 4 = 0 → YES  (prefill token 3 triggers compression)
  p=4,5,6: 5,6,7 % 4 ≠ 0 → no

# For decode token at position 3999:
compressed_slot = 3999 // 4 = 999

# Gather the 8 most recent C4A states from indexer state_cache:
# These are positions [3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999]
# (coff=2 × compress_ratio=4 = 8 states)
kv_states    [8, 128]
score_states [8, 128]

group0_gate = softmax(score_states[0:4, :])   # weights for positions 3992-3995
group1_gate = softmax(score_states[4:8, :])   # weights for positions 3996-3999
compressed_kv = sum(kv_states[0:4] * group0_gate) + sum(kv_states[4:8] * group1_gate)
              # [128]: learned summary of positions 3992-3999

RMSNorm(compressed_kv, norm.weight)
GPT-J RoPE at rope_position = (3999 // 4) * 4 = 3996
  (Note: C4A uses compress_rope_theta=160000, not the main 10000)
FP8 quantize → write to indexer k_cache at slot 999:
  k_cache[block=999//256=3, offset=999%256=231, :132]

# For prefill token at position 3:
compressed_slot = 3 // 4 = 0

# Gather 8 states for positions [-4,-3,-2,-1, 0, 1, 2, 3]:
# positions < 0 don't exist (fresh sequence) → treated as zeros (padded states)
kv_states    [8, 128]  (states 0-3 are real, states -4..-1 are zero-padded)
score_states [8, 128]

group0_gate = softmax(score_states[0:4])   # positions -4,-3,-2,-1 (padded → near-uniform)
group1_gate = softmax(score_states[4:8])   # positions 0,1,2,3 (real)
compressed_kv = sum(padded_kv * group0_gate) + sum(real_kv * group1_gate)

→ write to indexer k_cache slot 0
```

### 5.2 Indexer scoring with fresh compressed history

```
After compression, request A (fresh prompt) has:
  indexer k_cache: 1 compressed entry (slot 0, covering positions 0-3)

Request B has:
  indexer k_cache: 1000 compressed entries (slots 0-999, covering positions 0-3999)

Prefill tokens, cu_seqlen_ke for C4A:
  ke[i] = (pos_i + 1) // 4
  pos=0: ke=0, pos=1: ke=0, pos=2: ke=0
  pos=3: ke=1   ← first compressed entry now visible
  pos=4: ke=1, pos=5: ke=1, pos=6: ke=1

Prefill logits for token at position 3:
  q_fp8[prefill_offset+2, :, :]   (token at position 3)
  k_quant [1, 128]                (1 compressed entry gathered)
  logit[pos3, ck0] = Σ_h dot(q_fp8, k_fp8[0]) × sk[0] × weights_out   # scalar

Top-K (1024 >> 1):
  topk_indices_buffer[prefill_offset+2, 0] = 0   (only 1 entry, select it)
  topk_indices_buffer[prefill_offset+2, 1:] = -1

Tokens at positions 4,5,6 also see 1 compressed entry (slot 0):
  Each selects slot 0 as their top-K result.

Expanded for main MLA (C4A, compress_ratio=4):
  For ck=0: original positions 0,1,2,3 → 4 physical slots in main MLA KV cache
  global_topk_indices[prefill+2, 0, :4] = [slot_0, slot_1, slot_2, slot_3]
```

---

## 6. Layer 5: C128A + Standard noaux_tc MoE

By layer 5, the hidden states carry richer semantic information. This is a C128A layer using noaux_tc routing.

### 6.1 Standard MoE routing example

```
Token 2 ("▁Tower", token_id=17720), hidden_states at layer 5:
  (now a rich 7168-dim vector after 5 layers of attention+MoE processing)

logits = gate(hidden_states[2])   [384]  bf16  (all 384 experts)
scores = sqrtsoftplus(logits)     [384]  float32

# Example scores at a few expert positions:
scores[0]   = 0.812
scores[42]  = 1.309
scores[100] = 0.523
scores[107] = 1.402   ← would be selected without bias
scores[175] = 0.891
scores[211] = 1.178
...

# e_score_correction_bias from checkpoint (layer 5 has its own bias vector):
e_score_correction_bias[42]  = +0.15   (under-loaded, boosted)
e_score_correction_bias[107] = -0.12   (over-loaded, suppressed)
...

# Biased scores:
scores_biased[42]  = 1.309 + 0.15  = 1.459   ← now beats expert 107
scores_biased[107] = 1.402 - 0.12  = 1.282

# Top-6 by biased score → select expert 42 instead of 107:
topk_ids = [42, 211, 175, 8, 301, 127]   (example)

# Routing weights (unbiased scores at selected positions):
raw_weights = [1.309, 1.178, 0.891, 0.834, 1.056, 0.923]   # no bias
sum = 6.191
renorm = [0.211, 0.190, 0.144, 0.135, 0.171, 0.149]
final_weights = [0.529, 0.476, 0.359, 0.337, 0.427, 0.372]

# Note: expert 42's weight = 0.529, based on its RAW score 1.309,
# not the inflated biased score 1.459.
# Expert 107 is NOT selected (bias = -0.12 demoted it below top-6).
```

---

## 7. After All 61 Layers: Logit Computation and Sampling

After all 61 attention+MoE layers, `hidden_states [8, 7168]` goes through the final normalization and language model head:

```python
# Final RMSNorm:
hidden_states = final_norm(hidden_states)   # [8, 7168]

# LM head (not tied to embedding in this config):
logits = lm_head(hidden_states)   # [8, 129280]  bf16 → float32
```

For next-token prediction, only the last position in each request matters:
- Request A (prefill, 7 tokens): use logits at index 7 (last token, ".")
- Request B (decode, 1 token): use logits at index 0

```python
logits_A = logits[7, :]   # [129280]  — predict what comes after "The Eiffel Tower is in Paris."
logits_B = logits[0, :]   # [129280]  — predict next token for request B
```

**Sampling (greedy for request A):**

```python
next_token_id_A = argmax(logits_A)   # e.g. 128001 (EOS) or a continuation token

# For temperature sampling:
temperature = 0.7
probs_A = softmax(logits_A / temperature)   # [129280]
next_token_id_A = multinomial(probs_A, 1)   # sample from distribution

# Decode to text:
next_token_A = tokenizer.decode([next_token_id_A])   # e.g. "\n" or "</s>"
```

---

## 8. Decode Step: One Token at a Time

After the prefill returns `next_token_id_A`, the model enters the decode loop:

### 8.1 New token embedding

```
new_token_ids = [next_token_id_A]   # [1]
hidden_states = embed_tokens([next_token_id_A])   # [1, 7168]

positions = [7]   # next position is 7 (after the 7-token prompt at positions 0-6)
```

Now `T=1` (single decode token) goes through all 61 layers.

### 8.2 C4A decode attention

At a C4A layer (e.g., layer 2) during decode:

```
Decode token at position 7:
compressed_seq_len_A = (7 + 1) // 4 = 2   (positions 0-7 → compressed slots 0 and 1)
  slot 0: covers positions 0-3
  slot 1: covers positions 4-7

Does compress_norm_rope_store fire? (7+1) % 4 = 0 → YES!
  The decode token at position 7 triggers compression:
  → gather 8 states (positions 0-7 with coff=2 overlap)
  → produce compressed slot 1 for both indexer k_cache and main MLA KV cache

Indexer scoring (decode path, paged):
  context_lens = [[2]]   (2 compressed entries visible after the new compression)
  fp8_fp4_paged_mqa_logits walks 2 compressed K slots
  → logits [1, max_len//4]  (valid: [0, 1])
  → top_k selects both (2 << 1024)
  topk_indices_buffer[0, :2] = [1, 0]   (sorted by score; slot 1 may score higher)

Expanded for main MLA:
  ck=1 → original slots for positions 4,5,6,7
  ck=0 → original slots for positions 0,1,2,3
  global_topk_indices [1, 1, 8]

SWA at position 7 (window_size=4096):
  swa covers positions max(0, 7-4095)=0 to 7 → all 8 positions
  swa_indices [1, 1, 8]

flash_mla_with_kvcache:
  SWA: 8 slots (all history)
  Sparse: 8 slots (same, via expanded compressed indices)
  → kernel merges and deduplicates; effectively attends all 8 positions
```

### 8.3 C128A decode attention

At a C128A layer (e.g., layer 0) during decode, position 7:

```
compressed_seq_len_A = (7 + 1) // 128 = 0
  → 7 < 127, no compression has fired for request A
  → 0 compressed entries in indexer k_cache and main MLA KV cache

Indexer scoring:
  context_lens = [[0]]   → no logits computed, topk_indices_buffer stays all -1

SWA at position 7:
  swa_indices [1, 1, 8]   (all 8 positions in window)

flash_mla_with_kvcache:
  SWA: 8 slots (all positions 0-7, full 512-dim quality)
  Sparse: 0 slots (no compressed entries yet)
  → attends only via SWA for short sequences
```

C128A provides no compressed long-range coverage until the sequence reaches at least 128 tokens. For the first 127 positions, SWA covers everything. Starting at position 127, the first compressed entry appears and C128A sparse attention kicks in.

### 8.4 Decode MoE

At each MoE layer during decode, the same routing runs on T=1 token. The overhead is dominated by the gate GEMM `[384] = W_gate [384, 7168] @ hidden [7168]` — small at T=1 since it's just one dot product per expert.

```
For decode token, any standard MoE layer:
  logits = gate(hidden_states[0])   [384]  bf16
  scores = sqrtsoftplus(logits)     [384]  float32
  scores_biased = scores + e_score_correction_bias
  topk_ids = argtopk(scores_biased, 6)   [6]
  final_weights = renorm(scores[topk_ids]) * 2.5   [6]

Expert dispatch: T=1, so each expert processes exactly 1 token.
  For each e in topk_ids:
    out_e = W2[e] @ silu(clamp(W1[e] @ hidden[0]))   [7168]
  routed_out = weighted sum of 6 expert outputs
  final = routed_out + shared_expert(hidden[0])
```

---

## 9. Long-Context Decode: 128K Tokens

To illustrate the full power of the architecture, here is a decode step at position 131071 (128K - 1).

### 9.1 Coverage breakdown: C4A vs C128A

```
Position: 131071
window_size: 4096

SWA window: positions 131071 - 4095 = 127,007 to 131,071 → 4096 tokens (recent history)
                                                               ↑ attended at full 512-dim quality

C4A compressed history:
  compressed_seq_len = 131071 // 4 = 32767 entries
  (each covering 4 original positions 0..131067)
  
  Indexer selects top-1024 from 32767 → 3.1% of compressed history
  Each selected compressed entry expands to 4 original slots:
    topk selects ck ∈ [0, 32767)  → 1024 entries
    global_topk_indices: 1024 × 4 = 4096 additional original tokens
  
  Total coverage:
    SWA:    4096 tokens (positions 127007-131071, full 512-dim)
    Sparse: 4096 tokens (scattered across positions 0-131067, 128-dim resolution)
    Total:  8192 / 131072 = 6.25% of all tokens attended

C128A compressed history:
  compressed_seq_len = 131071 // 128 = 1023 entries
  (each covering 128 original positions 0..130943)
  
  Indexer selects top-1024 from 1023 → 100% of compressed history (1023 < 1024)
  Each selected compressed entry expands to 128 original slots:
    1023 × 128 = 130944 original tokens
  
  Total coverage:
    SWA:    4096 tokens (positions 127007-131071)
    Sparse: 130944 tokens (positions 0-130943, compressed resolution)
    Total:  (4096 + 130944) / 131072 = 102.9% ← full history, with SWA overlap
  
  Effective: C128A attends 100% of history, just at 128× lower resolution than C4A.
```

### 9.2 Compressed index expansion

**C4A example at 128K:**

```
topk_indices_buffer[0, :1024] = [28103, 7441, 19822, ..., 31256]  (sorted by score)

For ck=28103 (highest-scored compressed entry):
  original_positions = [28103*4, 28103*4+1, 28103*4+2, 28103*4+3]
                     = [112412, 112413, 112414, 112415]
  block_table lookup:
    block = block_table[req, 112412 // block_size]
    slot_0 = block * block_size + 112412 % block_size
    slot_1 = block * block_size + 112413 % block_size
    ... etc
  global_topk_indices[0, 0, 0:4] = [slot_0, slot_1, slot_2, slot_3]

# flash_mla_with_kvcache reads main_mla_kv_cache[slot_k, :]
# for each slot k in global_topk_indices[0, 0, :]
# → 4096 FP8 KV slots read from main MLA cache
# → combined with 4096 SWA slots → softmax → output [1, 128, 512]
```

**C128A example at 128K:**

```
topk_indices_buffer[0, :1023] = [0, 1, 2, ..., 1022]  (all 1023 selected in some order)

For ck=500:
  original_positions = [500*128, 500*128+1, ..., 500*128+127]
                     = [64000, 64001, ..., 64127]
  → 128 physical slots in main MLA KV cache

global_topk_indices[0, 0, :130944]  (1023 × 128 entries)

# flash_mla_with_kvcache reads:
# - 4096 SWA slots (recent 4096 positions, full 512-dim quality)
# - 130944 compressed main MLA slots (full history, 128× lower resolution)
# → softmax over all these → output [1, 128, 512]
```

---

## 10. Speculative Decode Step (DSpark)

Setup: DSpark draft model with 5 layers, context = 1000 tokens, generating 3 draft tokens.

```
Target model produced hidden_states [1000, 7168] (last layer output for context)

Step 1: context_kv_proj (fused, 5 layers × 576 dims = 2880 total):
  context_kv_proj(hidden_states [1000, 7168]) → kv_all [1000, 2880]
  For draft layer 0: c_kv = kv_all[:, 0:512], k_pe = kv_all[:, 512:576]
  For draft layer 1: c_kv = kv_all[:, 576:1088], k_pe = kv_all[:, 1088:1152]
  ...
  After RMSNorm + RoPE at positions 0..999:
    draft_kv_cache[layer_l][0:1000] = [c_kv, k_pe_rotated]  for each layer l

Step 2: Draft model forward (3 tokens simultaneously, non-causal):
  Start from last_token_embedding = embed_tokens([token_at_pos_999])  [1, 7168]
  
  DSpark MLA layer 0:
    Q from draft hidden, KV from draft_kv_cache[0] (context 0..999)
    non_causal=True: all 3 draft tokens see full context but NOT each other
    attn_out [3, 5, 7168]  (3 draft positions, 5 DSpark heads, hidden)

Step 3: DSpark draft logits:
  draft_logits_0 = lm_head(norm(attn_out[0])) + markov_head(attn_out[0])
  draft_token_0 = argmax(draft_logits_0)  # e.g. token_id = 9784 "▁France"
  draft_token_1 = argmax(draft_logits_1)  # e.g. "▁is"
  draft_token_2 = argmax(draft_logits_2)  # e.g. "▁a"

Step 4: Verification by target model:
  Run base K3 on [context_1000_tokens, "▁France", "▁is", "▁a"]
  → base_logits [3, 129280] for positions 1000, 1001, 1002
  Accept if base agrees with draft (within temperature threshold)
  If draft_token_0 accepted: position 1000 "▁France" confirmed
  If draft_token_1 rejected: resample from base_logits[1]; discard draft_token_2
```

---

## 11. DCP (Decode Context Parallelism) Example

Setup: world_size=2 DCP, sequence with 10000 tokens history, one decode token at position 10000.

```
KV shard assignment (interleaved, interleave=1):
  GPU 0: tokens at positions [0, 2, 4, ..., 9998]  (5000 tokens)
  GPU 1: tokens at positions [1, 3, 5, ..., 9999]  (5000 tokens)

GPU 0 local indexer scoring:
  fp8_fp4_paged_mqa_logits over its 5000/4 = 1250 compressed KV entries (C4A)
  → local_logits [1, max_model_len/4] (valid: [0, 1249])
  → local_topk [1, 1024]: top-1024 local compressed positions

pack_dcp_topk_candidates_cutedsl:
  For local compressed position ck=847 (GPU 0):
    score = local_logits[0, 847]
    global_id = 847 * 2 + 0 = 1694   (interleaved: ck*world_size + rank)
  → packed_gpu0 [1, 1024, 2] = [(score_0, 1694), (score_1, global_1), ...]

all_gather → gathered [1, 2048, 2]  (both GPUs' top-1024 candidates)

stable_topk_from_gathered_candidates_cutedsl:
  Build 64-bit key for each: (flip(score_bits) << 32) | (~global_id)
  Radix select top-1024 by key
  → final_topk [1, 1024] global compressed token IDs

GPU 0 uses final_topk for its FlashMLA call over its local KV shard.
GPU 1 does the same for its shard.
dcp_direct_a2a_lse_reduce.cu: LSE-aware merge of both GPUs' attention outputs.
```

---

## 12. Key Implementation Files

| File | Role |
|---|---|
| `vllm/model_executor/models/deepseek_v2.py` | V3/V4 shared: embedding, final norm, `lm_head`, sampling logic |
| `vllm/models/deepseek_v4/attention.py` | `DeepseekV4Attention`; per-layer C4A/C128A dispatch |
| `vllm/models/deepseek_v4/compressor.py` | `DeepseekCompressor`; `save_partial_states`; `compress_norm_rope_store` |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | SWA + sparse FlashMLA decode/prefill dispatch |
| `vllm/models/deepseek_v4/sparse_mla.py` | `build_c128a_topk_metadata`; C128A index expansion |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | SWA index computation; `_compute_swa_indices_and_lens_kernel` |
