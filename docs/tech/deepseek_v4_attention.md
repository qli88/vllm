# DeepSeek V4: Attention Architecture

[← Index](deepseek_index.md) | [Layer Types](deepseek_v4_layers.md) | [End-to-End Example](deepseek_v4_e2e.md)

Covers the full V4 attention forward pass: Q/KV projections, the compressor, the sparse indexer, the combined SWA + compressed sparse FlashMLA, and the output projection. Kernel-level details with concrete shapes at each step.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Input Projections: fused_wqa_wkv](#2-input-projections-fused_wqa_wkv)
3. [Q Up-Projection and Normalization: wq_b + fused insert](#3-q-up-projection-and-normalization-wq_b--fused-insert)
4. [SWA KV Cache Insert: fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](#4-swa-kv-cache-insert-fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert)
5. [The Compressor](#5-the-compressor)
   - 5.1 [save_partial_states](#51-save_partial_states)
   - 5.2 [compress_norm_rope_store](#52-compress_norm_rope_store)
   - 5.3 [get_compressed_slot_mapping](#53-get_compressed_slot_mapping)
   - 5.4 [Compressor Variants by Kernel](#54-compressor-variants-by-kernel)
6. [The Sparse Indexer](#6-the-sparse-indexer)
   - 6.1 [Indexer Q: wq_b + fused_indexer_q_rope_quant](#61-indexer-q-wq_b--fused_indexer_q_rope_quant)
   - 6.2 [Indexer K: from the Compressor](#62-indexer-k-from-the-compressor)
   - 6.3 [Scale Absorption: FP8 vs MXFP4](#63-scale-absorption-fp8-vs-mxfp4)
   - 6.4 [get_paged_mqa_logits_metadata (Decode SM Scheduler)](#64-get_paged_mqa_logits_metadata-decode-sm-scheduler)
   - 6.5 [MQA Logit Kernels: Prefill and Decode](#65-mqa-logit-kernels-prefill-and-decode)
   - 6.6 [Top-K Selection Kernels](#66-top-k-selection-kernels)
   - 6.7 [DCP Merge](#67-dcp-merge)
7. [Main MLA Attention: SWA + Compressed Sparse](#7-main-mla-attention-swa--compressed-sparse)
   - 7.0 [SWA Index Computation](#70-swa-index-computation)
   - 7.1 [Decode Path: flash_mla_with_kvcache](#71-decode-path-flash_mla_with_kvcache)
   - 7.2 [Prefill Path: Gather Workspace + flash_mla_sparse_fwd](#72-prefill-path-gather-workspace--flash_mla_sparse_fwd)
   - 7.3 [C4A vs C128A: Expanding Compressed Indices](#73-c4a-vs-c128a-expanding-compressed-indices)
8. [Output Projection: Inverse RoPE + wo_a + wo_b](#8-output-projection-inverse-rope--wo_a--wo_b)
9. [Stream Parallelism](#9-stream-parallelism)
10. [Spec-Decode Padding Helpers](#10-spec-decode-padding-helpers)
    - [`pack_seq_triton` — ragged → padded](#pack_seq_triton--ragged--padded)
    - [`unpack_seq_triton` — padded → ragged (inverse)](#unpack_seq_triton--padded--ragged-inverse)
    - [Usage in the indexer decode path](#usage-in-the-indexer-decode-path)
11. [KV Cache Format Reference](#11-kv-cache-format-reference)
12. [Concrete Examples](#12-concrete-examples)
    - 12.1 [Decode step — one token, two cache reads](#121-decode-step--one-token-two-cache-reads)
    - 12.2 [Prefill — 8 tokens, C4A](#122-prefill--8-tokens-c4a)
    - 12.3 [Compressor state update — C4A example at position 7](#123-compressor-state-update--c4a-example-at-position-7)
    - 12.4 [Scale absorption — concrete calculation](#124-scale-absorption--concrete-calculation)
    - 12.5 [fp8_paged_mqa_logits decode walk — one block traversal](#125-fp8_paged_mqa_logits-decode-walk--one-block-traversal)
    - 12.6 [SWA index computation — two positions](#126-swa-index-computation--two-positions)
13. [Key Implementation Files](#13-key-implementation-files)

---

## 1. High-Level Architecture

Each V4 attention layer (C4A or C128A) has four concurrent compute paths:

```
                    hidden_states [T, 7168]
                          │
          ┌───────────────┼─────────────────────────┐
          │               │                          │
    Default stream    Aux stream 0             Aux stream 1       Aux stream 2
          │               │                          │               │
   fused_wqa_wkv   main_compressor           indexer            indexer
   [7168→1536+512]  .fused_wkv_wgate        .weights_proj      .compressor
          │         [7168→2048]             [7168→64]          .fused_wkv_wgate
          │               │                          │           [7168→512]
         qr,kv       main_kv_score          idx_weights          idx_kv_score
          │               │                          │               │
          └───────────────┴──────────────────────────┴───────────────┘
                    fan-in → fused_q_kv_rmsnorm(qr, kv)
                          │
              ┌───────────┼───────────────────────────┐
              │           │                            │
        Default         Aux 0                       Aux 1
              │           │                            │
         wq_b(qr)   indexer inner overlap:        main_compressor:
         [1536→      wq_b(qr) + q_quant            save_states
          128×512]   || compressor(idx_kv_score)    compress→main KV
              │           │                            │
         fused_insert  indexer_op(q, k, w)         [records C]
         (norm+RoPE+    topk_indices_buffer
          SWA insert)  [records B]
         [records A]
              │
        wait A+B+C
              │
         forward_mqa(q, kv, topk_indices_buffer)
              │
          _o_proj(attn_out, positions)
              │
          output [T, 7168]
```

The key property: the indexer scoring and the main compressor run while the default stream is doing Q up-projection and SWA insert. By the time the default stream reaches `forward_mqa`, both `topk_indices_buffer` and the main KV cache are already populated.

---

## 2. Input Projections: fused_wqa_wkv

**Module:** `self.fused_wqa_wkv` — `MergedColumnParallelLinear [7168 → 1536 + 512]`

This single GEMM replaces separate Q and KV projections. Both `qr` (Q low-rank activations) and `kv` (KV latent) read from the same `hidden_states`, so fusing them halves the memory bandwidth load and schedules one larger GEMM instead of two smaller ones.

```
hidden_states [T, 7168]  bf16
     × weight [2048, 7168]      (2048 = 1536 + 512)
     ───────────────────────────────────
output [T, 2048]
  ├─ qr  [T, 1536]   ← Q low-rank activations (cols 0:1536)
  └─ kv  [T, 512]    ← KV latent (cols 1536:2048)
```

**Immediate post-processing:**

```python
# fused_q_kv_rmsnorm — single Triton kernel, grid (T, 2):
#   pid_task=0: normalize qr[t, :] with q_norm.weight  [1536]
#   pid_task=1: normalize kv[t, :] with kv_norm.weight [512]
# Both run in parallel across thread blocks.
qr = q_norm(qr)    # RMSNorm, learned weight q_norm.weight [1536]
kv = kv_norm(kv)   # RMSNorm, learned weight kv_norm.weight [512]
```

After this, `kv [T, 512]` is the normalized KV latent. It will be:
- Written to the SWA cache (as FP8) — for attending recent tokens
- Fed to the main compressor — for building long-range compressed KV
- Fed to the indexer compressor — for building compressed indexer K

**Example with concrete values:**

```
T = 4 prefill tokens, hidden_size = 7168, q_lora_rank = 1536

hidden_states [4, 7168]:
  token 0: [0.12, -0.34, 0.08, ..., 0.21]   (7168 values)
  token 1: [0.05, 0.41, -0.17, ..., -0.09]
  token 2: [0.23, -0.11, 0.36, ..., 0.14]
  token 3: [0.18, 0.27, -0.08, ..., 0.31]

After fused_wqa_wkv:
  qr [4, 1536]:  qr[0, :] = hidden[0] @ W[:1536, :].T   → 1536 activations
  kv [4, 512]:   kv[0, :] = hidden[0] @ W[1536:, :].T   → 512 activations

After fused_q_kv_rmsnorm:
  For token 0, Q normalization:
    rms_q  = sqrt(mean(qr[0,:]²) + 1e-6)   e.g. rms_q = 0.387
    qr[0, :] = qr[0, :] / rms_q * q_norm.weight  (element-wise)

  For token 0, KV normalization:
    rms_kv = sqrt(mean(kv[0,:]²) + 1e-6)   e.g. rms_kv = 0.412
    kv[0, :] = kv[0, :] / rms_kv * kv_norm.weight
```

---

## 3. Q Up-Projection and Normalization: wq_b + fused insert

**Module:** `self.wq_b` — `ColumnParallelLinear [1536 → 128 × 512]`

After `q_norm(qr)`, the Q low-rank activations are projected to full Q heads:

```
qr [T, 1536]   (normalized)
     × W_qb [65536, 1536]     (65536 = 128 heads × 512 dims per head)
     ─────────────────────────
q [T, 128, 512]   bf16        (after reshape)
```

With tensor parallelism (TP=8): each rank holds `wq_b` weight `[65536/8, 1536] = [8192, 1536]`, producing `q [T, 16, 512]` for `n_local_heads = 128/8 = 16` heads.

**Immediately after `wq_b`:** the fused kernel below processes `q` in-place.

---

## 4. SWA KV Cache Insert: fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

This is the largest fused kernel in V4 — it replaces ~8 separate operations with one C++ custom op call. It normalizes Q, applies RoPE to Q and KV, quantizes KV to FP8, and writes to the SWA cache — all in one pass.

**Signature:**
```python
torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q,            # [T, n_local_heads, 512]   bf16   modified in-place
    kv,           # [T, 512]                  bf16
    swa_kv_cache, # [num_blocks × block_size, 584]  uint8  (2D view)
    slot_mapping, # [T]  int64   SWA physical slot per token
    positions,    # [T]  int64
    cos_sin_cache,# [max_pos, 64]  float32   (cos[:32] | sin[32:] interleaved)
    padded_heads, # int   output head count (padded for FlashMLA alignment)
    eps,          # float   RMSNorm epsilon
    block_size,   # int
)
```

**What happens per token `t`, per head `h` (for Q):**

```
Step 1 — Per-head unit RMSNorm (no learned weight):
  rms = sqrt(mean(q[t, h, :]²) + eps)
  q[t, h, :] = q[t, h, :] / rms

Step 2 — GPT-J RoPE on last 64 dims of q[t, h, :]:
  For k in range(32):    # 64 rope dims as 32 interleaved pairs
    a = q[t, h, 448 + 2k]       # even
    b = q[t, h, 449 + 2k]       # odd
    cos = cos_sin_cache[positions[t], k]
    sin = cos_sin_cache[positions[t], 32 + k]
    q[t, h, 448+2k]   = a*cos - b*sin   # rotated even
    q[t, h, 449+2k]   = b*cos + a*sin   # rotated odd
```

**For KV (one vector per token, MQA):**

```
Step 3 — GPT-J RoPE on last 64 dims of kv[t, :]:
  Same rotation as Q step 2 but on kv[t, 448:] using compress_rope_theta=160000
  (different base frequency → different cos/sin table for the compressor path)

Step 4 — UE8M0 FP8 block-quantize kv[t, :448] (NoPE dims, 7 blocks of 64):
  For block b in range(7):
    slice = kv[t, b*64 : (b+1)*64]   # 64 float32 values
    amax = max(|slice|)
    # UE8M0: power-of-2 scale, biased exponent byte
    exponent_unbiased = ceil(log2(amax / 448.0))   # 448 = OCP FP8 max
    ue8m0_byte = exponent_unbiased + 127            # biased, stored as uint8
    scale = 2^exponent_unbiased
    kv_fp8[t, b*64:(b+1)*64] = clamp(slice / scale, -448, 448).to(fp8)

Step 5 — Write to SWA cache at slot_mapping[t]:
  # SWA slot layout (584 bytes):
  # [0:448]    = 448 FP8 bytes  (NoPE, 7 blocks of 64)
  # [448:576]  = 128 BF16 bytes (RoPE, 64 values × 2 bytes, not quantized)
  # [576:583]  = 7 ue8m0 scale bytes
  # [583]      = 1 padding byte
  swa_kv_cache[slot_mapping[t], :448]  = kv_fp8_nope
  swa_kv_cache[slot_mapping[t], 448:576] = kv[t, 448:].view(uint8)  # BF16 as bytes
  swa_kv_cache[slot_mapping[t], 576:583] = ue8m0_bytes
```

**Concrete example (token at position 7, block b=0):**

```
kv[7, 0:64] = [0.23, -0.15, 0.41, ..., 0.08]   (64 float32 values after kv_norm)
amax = 0.89
fp8_max = 448.0
scale_raw = 0.89 / 448.0 = 0.001987...
ceil(log2(0.001987)) = ceil(-8.98) = -8
ue8m0_byte = -8 + 127 = 119
scale = 2^(-8) = 0.00390625

kv_fp8[7, 0:64] = clamp([0.23/0.00390625, ...], -448, 448)
                = clamp([58.88, -38.40, 104.96, ...], -448, 448)
                ≈ [58.88, -38.40, 104.96, ...]  → cast to fp8 (e4m3fn)
                = [59, -38, 105, ...]  (fp8 nearest-representable)
```

---

## 5. The Compressor

The compressor accumulates N raw token states and collapses them into 1 compressed KV entry. It runs twice per layer: once for the indexer (128-dim) and once for the main MLA (512-dim). Both use the `DeepseekCompressor` class with different `head_dim` values.

### 5.1 `save_partial_states`

**Triton kernel** `_save_partial_states_kernel`, grid `(T,)`. Fires for **every** input token.

```python
# For each token t at absolute position p = positions[t]:
ape_pos = p % compress_ratio             # 0, 1, 2, or 3 for C4A

kv_state    = fused_wkv_wgate(hidden)[t, :coff*head_dim]      # e.g. [256] for C4A
score_state = fused_wkv_wgate(hidden)[t, coff*head_dim:]       # e.g. [256]

# APE: add learned position embedding for this slot in the group
kv_stored    = kv_state    + ape[ape_pos, :coff*head_dim]
score_stored = score_state + ape[ape_pos, coff*head_dim:]

# Write to paged state_cache:
slot_id = slot_mapping[t]               # physical slot in state_cache
if slot_id < 0: continue               # padding token, skip
block = slot_id // block_size
offset = slot_id % block_size
state_cache[block, offset, :coff*head_dim] = kv_stored
state_cache[block, offset, coff*head_dim:] = score_stored
```

**APE example (C4A, head_dim=128, coff=2):**

```
ape shape: [4, 512]   (4 positions × 2×coff×head_dim = 2×2×128 = 512)

Token at position 9 (group 2, position 1 within group):
  ape_pos = 9 % 4 = 1
  kv_stored = kv_state + ape[1, :256]      # position-1 embedding for KV
  score_stored = score_state + ape[1, 256:] # position-1 embedding for score

Token at position 12 (group 3, position 0 within group):
  ape_pos = 12 % 4 = 0
  kv_stored = kv_state + ape[0, :256]      # position-0 embedding
```

### 5.2 `compress_norm_rope_store`

Fires only when `(positions[t] + 1) % compress_ratio == 0`, i.e., at the end of each compression group. For C4A: every 4 tokens. For C128A: every 128 tokens.

**Full algorithm (C4A, head_dim=128, FP8 cache):**

```
# 1. Gather the last coff × compress_ratio = 8 states from state_cache:
#    These correspond to positions [p-7 .. p] (overlapping two groups)
kv_states    [8, 128]  = gather(state_cache, positions=[p-7..p], slice=kv)
score_states [8, 128]  = gather(state_cache, positions=[p-7..p], slice=score)

# 2. Softmax-gated compression (two groups of 4 due to coff=2 overlap):
group0_gate = softmax(score_states[0:4, :], dim=0)   # [4, 128], sums-to-1 over 4 states
group1_gate = softmax(score_states[4:8, :], dim=0)   # [4, 128]
compressed_kv = sum(kv_states[0:4] * group0_gate, dim=0)   # [128]
              + sum(kv_states[4:8] * group1_gate, dim=0)    # [128]
# compressed_kv [128]: weighted average of 8 token representations

# 3. RMSNorm:
rms = sqrt(mean(compressed_kv²) + eps)
normalized = compressed_kv / rms * norm.weight    # [128]

# 4. GPT-J RoPE at the aligned position:
rope_pos = (p // compress_ratio) * compress_ratio  # e.g. for p=7: rope_pos=4
# rotate normalized[64:] using cos/sin at rope_pos
rotated = gptj_rope(normalized, rope_pos, compress_rope_theta=160000)

# 5. FP8 UE8M0 quantize and write to indexer k_cache:
amax = max(|rotated|)
exponent = ceil(log2(amax / 448.0))
scale = 2^exponent
k_fp8 = clamp(rotated / scale, -448, 448).to(fp8)

compressed_slot = p // compress_ratio    # e.g. p=7 → slot 1
block = compressed_slot // 256           # indexer k_cache block_size = 256
offset = compressed_slot % 256
k_cache[block, offset, :128] = k_fp8
k_cache[block, offset, 128:132] = float32_as_bytes(scale)
```

**Concrete example (C4A, tokens 4–7, second group):**

```
Positions 4, 5, 6, 7 produce compressed slot 1.

state_cache contents after save_partial_states for pos 0..7:
  slot 0 (pos 0): kv+ape[0], score+ape[0]
  slot 1 (pos 1): kv+ape[1], score+ape[1]
  slot 2 (pos 2): kv+ape[2], score+ape[2]
  slot 3 (pos 3): kv+ape[3], score+ape[3]
  slot 4 (pos 4): kv+ape[0], score+ape[0]  ← new group starts
  slot 5 (pos 5): kv+ape[1], score+ape[1]
  slot 6 (pos 6): kv+ape[2], score+ape[2]
  slot 7 (pos 7): kv+ape[3], score+ape[3]  ← fires compress_norm_rope_store

At pos 7:
  kv_states    [8, 128] = states 0..7 (with their APE)
  score_states [8, 128] = scores 0..7

  # Group 0 gate (pos 0-3):
  score_0..3: [[-0.2, 0.5, ...], [-0.1, 0.3, ...], [0.4, -0.2, ...], [0.1, 0.6, ...]]
  softmax over dim=0: each column sums to 1
  gate_0[0, :] ≈ [0.18, 0.22, ...]   (token 0's contribution per dim)
  gate_0[1, :] ≈ [0.20, 0.21, ...]
  gate_0[2, :] ≈ [0.37, 0.17, ...]
  gate_0[3, :] ≈ [0.25, 0.40, ...]
  → compressed_group0 = sum_over_4_tokens(kv × gate)

  # Group 1 gate (pos 4-7): same structure

  compressed_kv = compressed_group0 + compressed_group1   # [128]

  After RMSNorm + RoPE at position 4 + FP8 quant:
  k_cache[0, 1, :128] = k_fp8     (block=0, offset=1 for slot 1)
  k_cache[0, 1, 128:] = float32_bytes(scale)
```

### 5.3 `get_compressed_slot_mapping`

**File:** `vllm/v1/attention/backends/mla/compressor_utils.py:53`

Computes which physical cache slot each token writes to in the compressed KV cache. Called once per step to produce the `slot_mapping` tensor consumed by `save_partial_states` and `compress_norm_rope_store`.

**Triton kernel** `_compressed_slot_mapping_kernel`, grid `(num_reqs,)`:

```python
For each request req, for token i (absolute position pos = prefix_len + i):
    if (pos + 1) % compress_ratio == 0:
        compressed_pos = pos // compress_ratio
        block = block_table[req, compressed_pos // block_size]
        slot  = block * block_size + compressed_pos % block_size
        out[token_offset + i] = slot        # valid write slot
    else:
        out[token_offset + i] = -1          # PAD_ID: no write
```

**Result:** `slot_mapping [T] int64`. For an 8-token prefill with `compress_ratio=4`, positions 3 and 7 get valid slots; positions 0,1,2,4,5,6 get `-1`.

`save_partial_states` skips tokens with `slot_id < 0` entirely (their state is still stored in the sliding `state_cache` at their own raw slot, but they don't trigger compression output). `compress_norm_rope_store` uses the slot to locate where to write the compressed K in the paged cache.

**Concrete example (C4A, 12-token prefill, compress_ratio=4, block_size=256):**

```
Positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                        ↑           ↑            ↑
                    (3+1)%4=0   (7+1)%4=0   (11+1)%4=0

slot_mapping = [-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2]
  slot 0 = block_table[req, 0//256] * 256 + 0  (compressed_pos 0)
  slot 1 = block_table[req, 1//256] * 256 + 1  (compressed_pos 1)
  slot 2 = block_table[req, 2//256] * 256 + 2  (compressed_pos 2)

→ compress_norm_rope_store fires for t=3, t=7, t=11
→ indexer k_cache grows by 1 entry every 4 input tokens
```

---

### 5.4 Compressor Variants by Kernel

Three Triton kernel variants in `fused_compress_quant_cache.py`:

| Variant | head_dim | Cache format | Use |
|---|---|---|---|
| A `_fused_kv_compress_norm_rope_insert_indexer_attn` | 128 | FP8 (132 B/slot) | Indexer K cache, default GPU |
| B `_fused_kv_compress_norm_rope_insert_sparse_attn` | 512 | FP8 ds_mla (584 B/slot) | Main MLA KV cache |
| C `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` | 128 | MXFP4 (68 B/slot) | Indexer K cache, Blackwell B200 |

Variant B (main MLA) has additional structure: only the 448 NoPE dims are FP8-quantized in 7 blocks of 64; the 64 RoPE dims are stored as BF16. This matches the SWA cache layout.

---

## 6. The Sparse Indexer

The indexer is a separate 64-head, 128-dim FP8 MQA attention module per layer. Its only job: output `topk_indices_buffer [T, 1024]` — the compressed-space positions of the 1024 most relevant historical tokens for each query token.

### 6.1 Indexer Q: wq_b + fused_indexer_q_rope_quant

```
qr [T, 1536]   (after q_norm, shared with main MLA)
  → wq_b  [1536 → 64×128]  (indexer's own weight, not shared with main MLA)
  → q_idx [T, 64, 128]   bf16
```

**`fused_indexer_q_rope_quant` (Triton, grid=(T, 64)):**

One program per `(token, head)`. Performs RoPE, FP8/MXFP4 quantization, and scale absorption in one kernel.

```
For token t at position p, head h:

# GPT-J interleaved RoPE on rope dims [64..127]:
#   q_idx[t, h, :64]    = NoPE portion (unchanged)
#   q_idx[t, h, 64:128] = RoPE portion (rotated)
for k in range(32):
  a = q_idx[t, h, 64 + 2k]      # even rope dim
  b = q_idx[t, h, 65 + 2k]      # odd rope dim
  cos = indexer_cos_sin[p, k]
  sin = indexer_cos_sin[p, 32 + k]
  r_even = a*cos - b*sin   → bf16 → fp32 roundtrip
  r_odd  = b*cos + a*sin   → bf16 → fp32 roundtrip

# Compute per-token per-head absmax over all 128 dims:
amax = max(|q_nope[t,h,:]|, |r_even|, |r_odd|)    e.g. 0.73

# UE8M0 scale (power-of-2):
scale_raw = 0.73 / 448.0 = 0.00163
exponent  = ceil(log2(0.00163)) = ceil(-9.26) = -9
q_scale   = 2^(-9) = 0.001953125

# Quantize to FP8:
q_fp8[t, h, :64]   = clamp(q_nope / q_scale, -448, 448).to(fp8)
q_fp8[t, h, 64::2] = clamp(r_even / q_scale, -448, 448).to(fp8)
q_fp8[t, h, 65::2] = clamp(r_odd  / q_scale, -448, 448).to(fp8)

# Scale absorption into weights_out (FP8 path):
#   weights_out[t,h] = indexer_weights[t,h] × q_scale × (1/√128) × (1/√64)
#   This pre-folds q_scale, softmax_scale (1/√head_dim=1/√128),
#   and head_scale (1/√n_heads=1/√64) into a single scalar.
#   The MQA logit kernel only needs to multiply by this one value.
weights_out[t, h] = indexer_weights[t, h] × 0.001953125 × (1/11.314) × (1/8.0)
                  = indexer_weights[t, h] × 0.001953125 × 0.08839 × 0.125
                  ≈ indexer_weights[t, h] × 2.16e-5
```

**MXFP4 path (Blackwell B200, SM100):**

```
# 4 blocks of 32 elements per head (128 / 32 = 4 blocks):
for block_b in range(4):
  slice = q_idx_rotated[t, h, block_b*32 : (block_b+1)*32]
  amax  = max(|slice_even|, |slice_odd|)   # max over paired values
  ue8m0_byte = ceil(log2(amax / 6.0)) + 127   # 6.0 = max E2M1 value
  scale = 2^(ue8m0_byte - 127)
  packed_q[t, h, block_b*16 : (block_b+1)*16] = pack_e2m1_nibbles(slice / scale)
  q_scale[t, h, block_b] = ue8m0_byte

# q_scale NOT absorbed (4 scalars per head, not collapsible to one):
weights_out[t, h] = indexer_weights[t, h] × (1/√128) × (1/√64)
# q_scale [T, 64, 4] passed separately to the MQA logit kernel
```

### 6.2 Indexer K: from the Compressor

Unlike V3 (where indexer K comes from a direct projection of `hidden_states`), V4 indexer K is produced by the indexer's own `DeepseekCompressor` (128-dim). The key chain:

```
hidden_states [T, 7168]
  → indexer.compressor.fused_wkv_wgate [7168 → 512]   (on aux stream 2)
  → idx_kv_score [T, 512]
       split: idx_kv [T, 256], idx_score [T, 256]   (coff=2, head_dim=128)
  → save_partial_states → state_cache
  → compress_norm_rope_store → indexer k_cache [blocks, 256, 132]
                                                       ↑ block_size=256

# SparseAttnIndexer receives skip_k_cache_insert=True:
# K is already in k_cache from the compressor. No separate insert step.
```

### 6.3 Scale Absorption: FP8 vs MXFP4

The true scoring logit is:

```
logit[q, k] = Σ_h  dot(q_bf16[q,h,:], k_bf16[k,:])  ×  w[q,h]  ×  (1/√128)  ×  (1/√64)
```

After FP8 quantization (per-head scale `sq[q,h]` for Q, per-token scale `sk[k]` for K):

```
= Σ_h  dot(q_fp8[q,h,:] × sq, k_fp8[k,:] × sk)  ×  w × (1/√128) × (1/√64)
= Σ_h  dot(q_fp8, k_fp8) × sk × [sq × w × (1/√128) × (1/√64)]
                                  ╰── weights_out[q,h] ────────╯
```

| Path | q_scale granularity | q_scale absorbed? | weights_out formula |
|---|---|---|---|
| FP8 | 1 scalar per (token, head) | Yes | `w × sq × (1/√128) × (1/√64)` |
| MXFP4 | 4 block scalars per (token, head) | No | `w × (1/√128) × (1/√64)` |

The K-side scale `sk[k]` is handled inside the DeepGEMM/AITER kernel as a per-KV-token dequantization factor.

### 6.4 `get_paged_mqa_logits_metadata` (Decode SM Scheduler)

**File:** `vllm/utils/deep_gemm.py:544`

Called once per decode step, on CPU, before `fp8_fp4_paged_mqa_logits`. Pre-assigns `(request, block)` pairs to SMs so every SM processes roughly the same number of KV blocks.

```python
get_paged_mqa_logits_metadata(
    context_lens: [B, next_n]  int32,   # context length (in compressed space) per token
    block_size:   int,                   # indexer cache block_size (64 for V3, 256 for V4)
    num_sms:      int,                   # number of SMs on this GPU
) → schedule_metadata [num_sms+1, 2]  int32
```

`schedule_metadata[i, :]` = `[start_work_id, end_work_id]` for SM `i`. Work IDs enumerate `(request, block_within_sequence)` pairs in row-major order. The `+1` row is a fence (total work count).

**Why this matters:**

Without pre-scheduling, the naive kernel assigns SM 0 to process the entirety of long request A while SM N-1 sits idle after finishing tiny request B. This is the classic "load imbalance in ragged batch" problem.

**Example (B=3, block_size=64, num_sms=4):**

```
context_lens = [3000, 100, 5000]   (compressed lengths)
num_blocks   = [ceil(3000/64)=47, ceil(100/64)=2, ceil(5000/64)=79]
total_blocks = 47 + 2 + 79 = 128

Target per SM: 128 / 4 = 32 blocks each

schedule_metadata:
  SM 0: work_ids [0..31]   = req0 blocks 0..31  (first 32 of req0's 47 blocks)
  SM 1: work_ids [32..63]  = req0 blocks 32..46 + req1 blocks 0..1 + req2 blocks 0..14
  SM 2: work_ids [64..95]  = req2 blocks 15..46
  SM 3: work_ids [96..127] = req2 blocks 47..78
  SM 4 (fence): [128, 128]
```

The kernel reads `schedule_metadata` to find its work range, then walks those `(request, block)` pairs independently.

---

### 6.5 MQA Logit Kernels: Prefill and Decode

**Prefill: gather then GEMM**

```
# Step 1: Gather all historical compressed K tokens into flat buffer
cp_gather_indexer_k_quant_cache(
    kv_cache,      # [num_blocks, 256, 132]
    dst_k,         # [N, 128]  fp8   (N = total compressed tokens across all prefill reqs)
    dst_scale,     # [N, 4]    uint8 (float32 as bytes)
    block_table,   # [B, max_blocks]
    cu_seq_lens,   # [B+1]
)

# Step 2: Compute logits
fp8_fp4_mqa_logits(
    q = (q_fp8 [M, 64, 128], q_scale_or_None),
    kv = (dst_k [N, 128], dst_scale [N, 4]),
    weights = weights_out [M, 64],
    cu_seqlen_ks, cu_seqlen_ke,    # causal mask per query token
) → logits [M, N]  float32

# For C4A prefill, cu_seqlen_ke[i] = (pos_i + 1) // 4
# → query at position 5 (in a fresh seq) sees ke=1 (only slot 0 valid)
# → query at position 7 sees ke=2 (slots 0 and 1 valid)
```

**Logit formula:**

```
for each valid (q, k) pair where cu_seqlen_ks[q] ≤ k < cu_seqlen_ke[q]:
  raw[h] = dot(q_fp8[q, h, :], k_fp8[k, :])   # 128-dim FP8 dot product, [64] results
  logit[q, k] = sum_h( raw[h] × sk[k] × weights_out[q, h] )
```

**Decode: paged traversal (no gather)**

```
fp8_fp4_paged_mqa_logits(
    q = (q_fp8 [B, next_n, 64, 128], q_scale_or_None),
    kv_cache = [num_blocks, 256, 1, 132],   # paged
    weights = weights_out [B*next_n, 64],
    context_lens = [B, next_n],             # compressed context length per token
    block_tables = [B, max_blocks],
    schedule_metadata,                      # SM tile assignments (pre-computed)
    max_model_len,
) → logits [B*next_n, max_model_len // compress_ratio]  float32
```

The kernel walks each block in `block_tables` for each decode token, reading K and scale directly from the paged cache (no separate gather). `schedule_metadata` from `get_paged_mqa_logits_metadata` pre-assigns blocks to SMs to avoid load imbalance across sequences with different context lengths.

### 6.6 Top-K Selection Kernels

After `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`, we select the top-1024 positions per query token.

| Kernel | Platform | When | Algorithm |
|---|---|---|---|
| `cooperative_topk` | CUDA SM≥90 | rows≤32, topk∈{512,1024,2048} | Multi-CTA cluster, DSMEM all-reduce, TMA bulk loads |
| `persistent_topk` | CUDA | rows>32, topk∈{512,1024,2048} | Persistent CTAs, per-row dispatch to 3 sub-algorithms by seq_len |
| `top_k_per_row_prefill` | Both | Prefill fallback | Insertion sort (N<12288) or 2048-bin radix sort (N≥12288) |
| `top_k_per_row_decode` | Both | Decode fallback | Insertion/radix/two-pass by max_model_len |

**`persistent_topk` per-row dispatch:**

```
seq_len ≤ 1024:        trivial fill (seq_len ≤ topk, select all)
1024 < seq_len ≤ 8192: histogram_2048_topk  (2048-bin 11-bit histogram, CUB scan)
8192 < seq_len ≤ 32768: histogram_256_topk  (256-bin 8-bit, 4 radix passes)
seq_len > 32768:        radix_topk           (8-bit radix, multi-CTA, vectorized)
```

**Example (decode, seq_len=32000, topk=1024):**

```
logits [1, 32000] (1 decode token, 32000 compressed KV positions for C4A)
→ persistent_topk selects top-1024:
  Uses histogram_256_topk (32000 ≤ 32768):
    1. Build 256-bin histogram over float32 logit values (8-bit MSB of float)
    2. Scan from high bins: find first bin where cumulative count ≥ 1024
    3. Scatter: collect all logit positions above threshold into topk_indices
  → topk_indices [1, 1024] = [8173, 14209, 3001, ..., 31847]  (sorted by score)
  → topk_indices_buffer[0, :] ← these 1024 compressed positions
```

### 6.7 DCP Merge

When Decode Context Parallelism (DCP) is active, the KV sequence is sharded across GPUs. Each GPU holds every `world_size`-th slot. After local top-K, ranks exchange and merge:

```
GPU 0 holds: compressed slots 0, 2, 4, ...   (even)
GPU 1 holds: compressed slots 1, 3, 5, ...   (odd)

Local top-K on GPU 0:
  logits [T, N/2], select top-1024 → local_topk_indices [T, 1024]  (local shard positions)

pack_dcp_topk_candidates_cutedsl:
  Convert local_topk_indices to (score, global_id) pairs:
  global_id = local_idx * world_size + rank
  packed [T, 1024, 2]   where [:,  0] = score, [:, 1] = global_id

all_gather(packed, dim=1) → gathered [T, world_size*1024, 2]

stable_topk_from_gathered_candidates_cutedsl:
  # 64-bit key for radix sort:
  key = (flip_float_bits(score) << 32) | (~global_id & 0xFFFFFFFF)
  # Sorts descending by score; ties broken by smaller global_id
  select top-1024 by key
  → out [T, 1024] int32  global token IDs
```

---

## 7. Main MLA Attention: SWA + Compressed Sparse

After `topk_indices_buffer [T, 1024]` is ready and the main compressor has written to `mla_kv_cache`, the actual attention runs.

### 7.0 SWA Index Computation

Before calling FlashMLA, the SWA physical slot IDs for each token must be computed from the block table. This is done by a dedicated Triton kernel each step.

**File:** `vllm/v1/attention/backends/mla/sparse_swa.py:660`

**Triton kernel** `_compute_swa_indices_and_lens_kernel`, grid `(T,)`:

```python
For each token t:
    req_idx = token_to_req_indices[t]
    pos     = prefix_len + (t - query_start)     # absolute position in sequence
    start   = max(pos - window_size + 1, 0)
    end     = pos + 1                             # exclusive

    swa_len = end - start

    for i, p in enumerate(range(start, end)):
        block = block_table[req_idx, p // block_size]
        slot  = block * block_size + p % block_size
        swa_indices[t, 0, i] = slot               # global physical slot ID

    swa_indices[t, 0, swa_len:] = -1              # pad remainder with -1
    swa_lens[t] = swa_len
```

**Result:**
- `swa_indices [T, 1, window_size] int32` — global physical slot IDs into `swa_kv_cache`; `-1` pads unused positions
- `swa_lens [T] int32` — how many valid SWA slots each token has

These are passed directly to `flash_mla_with_kvcache` as `indices` / `topk_length`.

**Concrete example (decode token at position 4100, window_size=4096):**

```
start = max(4100 - 4096 + 1, 0) = max(5, 0) = 5
end   = 4101
swa_len = 4096   (positions 5..4100)

For position p=5:
  block = block_table[req, 5 // block_size]   e.g. block=0 if block_size=256
  slot  = 0 * 256 + 5 = 5
  swa_indices[t, 0, 0] = 5

For position p=4100:
  block = block_table[req, 4100 // 256] = block_table[req, 16]
  slot  = block_table[req, 16] * 256 + (4100 % 256) = block16 * 256 + 4100-4096 = block16*256+4
  swa_indices[t, 0, 4095] = block16*256 + 4

swa_lens[t] = 4096
```

**Decode token at position 100 (within window):**

```
start = max(100 - 4096 + 1, 0) = 0
swa_len = 101   (positions 0..100)
swa_indices[t, 0, 101:] = -1  (4096 - 101 = 3995 padding slots)
```

---

### 7.1 Decode Path: flash_mla_with_kvcache

```python
flash_mla_with_kvcache(
    q = q.unsqueeze(1),                         # [T, 1, n_heads, 512]
    k_cache = swa_kv_cache,                      # [num_blocks, block_size, 1, 512]  uint8

    # SWA path: attends recent window_size tokens unconditionally
    indices = swa_indices,                       # [T, 1, window_size]  int32  global slot IDs
    topk_length = swa_lens,                      # [T]  valid entries per token

    # Sparse path: attends top-K compressed tokens (expanded to original slots)
    extra_k_cache = main_mla_kv_cache,           # [num_blocks, block_size, 1, 584]  uint8
    extra_indices_in_kvcache = global_topk_idx,  # [T, 1, topk*expand]  int32
    extra_topk_length = topk_lens,               # [T]

    softmax_scale = 1 / math.sqrt(512),          # 1/√head_dim
    attn_sink = self.attn_sink,                  # [n_heads]  learned per-head denominator bias
    is_fp8_kvcache = True,
    out = output,                                # [T, 1, n_heads, 512]  bf16
)
```

**`attn_sink`:** Prevents attention collapse when the selected token set is uniformly uninformative. A learned per-head bias `s_h` is added to the softmax denominator:

```
# Standard softmax:  sum_k(exp(score[k] - m)) where m = max(scores)
# With attn_sink:    sum_k(exp(score[k] - m)) + exp(s_h - m)

# Effect: even if all token scores are very negative, the denominator
# stays bounded away from 0. The per-head sink value is learned during
# training to match the typical score range for that head.
```

**How the kernel reads FP8 KV from the SWA cache:**

```
For each swa_slot in swa_indices[t, 0, :swa_lens[t]]:
  block = swa_slot // block_size
  offset = swa_slot % block_size

  # Read 584-byte slot:
  nope_fp8  = swa_kv_cache[block, offset, :448]   # 448 uint8 values
  rope_bf16 = swa_kv_cache[block, offset, 448:576].view(bf16)  # 64 bf16
  ue8m0     = swa_kv_cache[block, offset, 576:583]  # 7 scale bytes

  # Dequantize NoPE:
  for b in range(7):
    scale = 2^(int(ue8m0[b]) - 127)
    k_nope_bf16[b*64:(b+1)*64] = fp8_to_float(nope_fp8[b*64:(b+1)*64]) * scale

  k = cat(k_nope_bf16, rope_bf16)   # [512] bf16

  # Compute attention score for head h:
  score[h] = dot(q[t, h, :], k) * softmax_scale   # 512-dim dot
```

### 7.2 Prefill Path: Gather Workspace + flash_mla_sparse_fwd

For prefill, gathering separately and calling `flash_mla_sparse_fwd` is more efficient because prefill KV tokens are computed fresh and can be gathered once into a contiguous BF16 workspace.

```
# Step 1: Dequantize compressed K cache → BF16 workspace (N compressed entries)
dequantize_and_gather_k_cache(
    workspace[:chunk_size, :N, :],   # [chunk_size, N, 512] bf16
    compressed_k_cache,              # paged FP8 compressed cache
    seq_lens // compress_ratio,      # compressed sequence lengths
    gather_lens=None,
    block_table, block_size, offset=0
)

# Step 2: Dequantize SWA K cache → same workspace (after N, starting at offset N)
dequantize_and_gather_k_cache(
    workspace[:chunk_size, N:, :],   # [chunk_size, window_size, 512] bf16
    swa_kv_cache,
    seq_lens, gather_lens,           # gather last gather_lens[b] tokens for SWA
    block_table, block_size, offset=N
)
# workspace now contains: cols [0..N-1] = compressed KV, cols [N..N+window_size-1] = SWA KV

# Step 3: Combine indices into workspace-local column indices
combine_topk_swa_indices(
    topk_indices [T, 1024],          # indexer output (compressed positions 0..N-1)
    swa_start_positions,             # where each token's SWA window starts in workspace
) → combined_indices [T, 1, max_combined]

# Step 4: Sparse prefill attention
flash_mla_sparse_fwd(
    q = q[prefill_tokens, :, :],              # [T_q, n_heads, 512]
    kv = workspace.view(-1, 1, 512),          # [N+window_size, 1, 512] bf16
    indices = combined_indices,               # workspace-local column indices
    sm_scale = 1/√512,
    topk_length = combined_lens,
    attn_sink = self.attn_sink,
    out = output[prefill_tokens, :, :],
)
```

**Example (C4A prefill, 8 tokens, seq fresh from position 0):**

```
T_q = 8 query tokens, compress_ratio = 4

After indexer scoring:
  topk_indices[0:3, :] = [-1, ...]  # positions 0,1,2: no complete group yet
  topk_indices[3, 0] = 0             # position 3: sees compressed slot 0
  topk_indices[4:7, 0] = 0           # positions 4-6: still only slot 0 visible
  topk_indices[7, 0] = 1, topk_indices[7, 1] = 0   # position 7: sees slots 0,1

N = 2 compressed entries (slots 0 and 1)

workspace layout (chunk_size=1 request, M=N+window=2+8=10 cols):
  workspace[0, 0, :] = BF16 dequant of compressed slot 0  (covers pos 0-3)
  workspace[0, 1, :] = BF16 dequant of compressed slot 1  (covers pos 4-7)
  workspace[0, 2, :] = BF16 dequant of SWA pos 0 (= raw token 0)
  workspace[0, 3, :] = BF16 dequant of SWA pos 1
  ...
  workspace[0, 9, :] = BF16 dequant of SWA pos 7

combine_topk_swa_indices for token at position 3:
  topk_len = min((3+1)//4, 1024) = 1  → workspace col 0 (compressed slot 0)
  swa_len  = min(3+1, window=4096) = 4  → workspace cols 2,3,4,5 (SWA pos 0..3)
  combined_indices[3, 0, :] = [0, 2, 3, 4, 5, -1, ...]
  combined_lens[3] = 5

flash_mla_sparse_fwd then attends 5 workspace columns for token at position 3:
  col 0 → compressed entry (tokens 0-3 summarized)
  cols 2,3,4,5 → individual SWA tokens 0,1,2,3
  → effectively: position 3 attends a summary of tokens 0-3 AND individual tokens 0-3
```

### 7.3 C4A vs C128A: Expanding Compressed Indices

The difference is in how `topk_indices_buffer [T, 1024]` (compressed positions) is converted to physical cache slots for `extra_indices_in_kvcache`.

**C4A (compress_ratio=4):**

```python
# compute_global_topk_indices_and_lens (V4 CUDA path)
for each token t and compressed_pos ck in topk_indices[t, :]:
    if ck < 0: continue
    for offset in range(4):
        original_pos = ck * 4 + offset
        block = block_table[req, original_pos // block_size]
        slot  = block * block_size + original_pos % block_size
        global_topk_indices[t, j] = slot   # j increments
        j += 1

# Result: global_topk_indices [T, 1, 4096]  (1024 compressed × 4)
```

**C128A (compress_ratio=128):**

```python
# build_c128a_topk_metadata (handled separately for C128A)
for each token t and compressed_pos ck in topk_indices[t, :]:
    for offset in range(128):
        original_pos = ck * 128 + offset
        slot = block_table[req, original_pos // block_size] * block_size + original_pos % block_size
        c128a_indices[t, j] = slot
        j += 1

# Result: c128a_global_decode_topk_indices [T, 1, 131072]  (1024 compressed × 128)
```

The FlashMLA kernel receives `extra_indices_in_kvcache` with either 4096 or 131072 entries per token. The kernel itself treats them identically — just more physical slot IDs to read.

---

## 8. Output Projection: Inverse RoPE + wo_a + wo_b

After FlashMLA returns `attn_out [T, n_heads, 512]`, the output projection pipeline restores the original representation space and projects to `hidden_size`.

**Why inverse RoPE is needed:**

```
Forward pass:
  q[t, h, 448:] was rotated by GPT-J RoPE at position p_t
  → attention was computed in the rotated frame
  → attn_out[t, h, :] mixes contributions from different rotated positions

wo_a's weight was trained expecting inputs in the UNROTATED frame.
→ The rotation applied to Q must be undone before wo_a.
```

**Step 1: `fused_inv_rope_fp8_quant` (CUDA) — Triton kernel, grid `(ceil4(T), n_groups × heads_per_group)`**

```
For token t, head h:

# NoPE dims [0..447]: pass through unchanged
attn_out_unrot[t, h, :448] = attn_out[t, h, :448]

# RoPE dims [448..511]: apply inverse GPT-J rotation
# Forward:  [a, b] → [a*cos - b*sin, b*cos + a*sin]
# Inverse:  [y0, y1] → [y0*cos + y1*sin, -y0*sin + y1*cos]
for k in range(32):
  y0 = attn_out[t, h, 448 + 2k]
  y1 = attn_out[t, h, 449 + 2k]
  cos = cos_sin_cache[p_t, k]
  sin = cos_sin_cache[p_t, 32 + k]
  out_even = y0*cos + y1*sin
  out_odd  = y1*cos - y0*sin        # note sign flip vs. forward
  attn_out_unrot[t, h, 448+2k]   = out_even
  attn_out_unrot[t, h, 449+2k]   = out_odd

# FP8 block quantize (QUANT_GROUP_SIZE=128, so 4 blocks per 512-dim head):
for block_b in range(4):
  slice = attn_out_unrot[t, h, block_b*128 : (block_b+1)*128]
  amax  = max(|slice|)
  scale = 2^(ceil(log2(amax / 448.0)))
  o_fp8[t, group_g, head_in_group*512 + block_b*128 : ...] = clamp(slice/scale, -448, 448).to(fp8)
  o_scale[t, group_g, block_b] = ue8m0_byte
```

**Step 2: wo_a — FP8 grouped BMM**

```
o_groups = 16, o_lora_rank = 1024
n_local_heads = 128/TP
heads_per_group = n_local_heads / n_groups   # for TP=1: 128/16 = 8

fp8_einsum(
    "bgd,gdr→bgr",           # b=token, g=group, d=head*512 dim, r=lora_rank
    o_fp8   [T, 16, 8*512],  # [T, 16, 4096] fp8
    wo_a    [16, 4096, 1024] # fp8 weight
) → z [T, 16, 1024]  bf16
```

**Step 3: wo_b — RowParallel**

```
z.flatten(1)  [T, 16*1024] = [T, 16384]
wo_b: RowParallelLinear [16384, 7168/TP]
→ output [T, 7168/TP]
→ AllReduce across TP ranks
→ output [T, 7168]
```

---

## 9. Stream Parallelism

See full stream diagram in §1. Key timing relationships:

- **fused_wqa_wkv** (default) and **3 aux GEMMs** run concurrently — all 4 GEMMs overlap.
- **wq_b + fused_insert** (default) overlaps with **indexer inner** (aux 0) and **main compressor** (aux 1).
- **indexer inner** itself runs **indexer Q** and **indexer compressor** concurrently via sub-stream events.
- `forward_mqa` can only start after: (A) Q is ready, (B) topk_indices_buffer is filled, (C) main MLA KV cache has new entries.
- The attn_sink lookup and SWA index computation are cheap CPU/Triton operations that pipeline with the above.

---

## 10. Spec-Decode Padding Helpers

Used in the decode path of the indexer when speculative decoding produces non-uniform decode lengths across requests (some have 1 draft token, others have `next_n`). The paged MQA logit kernel requires a regular `[B, next_n, H, D]` Q tensor; `pack_seq_triton` / `unpack_seq_triton` convert between ragged and padded shapes.

**File:** `vllm/v1/attention/ops/common.py`

### `pack_seq_triton` — ragged → padded

```python
pack_seq_triton(
    x:         [N, ...]   # N total tokens, ragged (different lengths per request)
    lengths:   [B]        # actual decode-token count per request
    pad_value,            # -inf for logit Q, 0 for uint8 FP8/MXFP4 Q
) → [B, Lmax, ...]       # Lmax = max(lengths), excess slots filled with pad_value
```

**Triton kernel** `_pack_seq_kernel`, grid `(B, ceil(Lmax/BLOCK_T), ceil(D/BLOCK_D))`:

```python
in_start = cumsum(lengths)[:pid_b]   # start index of request pid_b in x
for off_t in range(BLOCK_T):
    t = pid_t * BLOCK_T + off_t
    for off_d in range(BLOCK_D):
        if t < lengths[pid_b]:
            out[pid_b, t, off_d] = x[in_start + t, off_d]
        else:
            out[pid_b, t, off_d] = PAD_VALUE
```

### `unpack_seq_triton` — padded → ragged (inverse)

```python
unpack_seq_triton(
    packed: [B, Lmax, ...]
    lengths:[B]
) → [N, ...]   # N = sum(lengths)
```

Same kernel structure but reads from `packed[b, t, :]` and writes to `out[cumsum(lengths[:b]) + t, :]`.

### Usage in the indexer decode path

When requests have decode lengths `[3, 1, 2]` (spec decode, non-uniform):

```
# N = 3+1+2 = 6 total decode tokens
# q_fp8 [6, 64, 128]  (ragged)

1. pack_seq_triton(q_fp8, decode_lens=[3,1,2], pad_value=0)
   → q_padded [3, 3, 64, 128]   (B=3, Lmax=3, padded with 0)

2. fp8_fp4_paged_mqa_logits(q_padded, ...)
   → logits [3*3=9, max_model_len]

3. top_k_per_row_decode(logits, ...) → topk_indices [9, 1024]

4. unpack_seq_triton(topk_indices.reshape(3, 3, 1024), decode_lens=[3,1,2])
   → topk_indices [6, 1024]  (ragged, padding rows removed)

# Padded slots (e.g. row (req1, t=1) and (req1, t=2) in padded space) produce
# q_fp8=0 → logits=-inf → topk_indices=-1 → harmless to downstream attention
```

**Why pad with 0 for FP8 Q (not -inf):**

`-inf` is not representable in FP8. Padding Q with `0` produces all-zero dot products → logits of 0 (not -inf). However, the `weights_out` factor is also 0 for padding tokens (since `indexer_weights` for padding positions is 0 after the gate runs on dummy hidden states), so the logit stays 0. The top-K kernel sees logit=0 which ranks below all real logits, so padding entries are never selected.

---

## 11. KV Cache Format Reference

| Cache | Dims | Format | Bytes/slot | Written by |
|---|---|---|---|---|
| SWA KV cache | 448 NoPE FP8 + 64 RoPE BF16 + 8 ue8m0 | `uint8` | **584** | `fused_qnorm_rope_kv_fp8_insert` (C++) |
| Main MLA KV cache | 448 NoPE FP8 + 64 RoPE BF16 + 8 ue8m0 | `uint8` | **584** | Main compressor (Triton) |
| Indexer K cache (FP8) | 128 FP8 + 4 scale bytes | `uint8` | **132** | Indexer compressor (Triton) |
| Indexer K cache (MXFP4) | 64 packed uint8 + 4 ue8m0 | `uint8` | **68** | Indexer compressor (Triton, Blackwell) |
| Compressor state cache | `2 × coff × head_dim` float32 | `float32` | 2048–32768 | `save_partial_states` (Triton) |

**SWA/Main MLA slot layout (584 bytes):**

```
Offset  Size   Content
0       448    448 FP8 values (NoPE dims, OCP e4m3fn or FNUZ on gfx942)
448     128    64 BF16 values (RoPE dims, not quantized)
576       7    7 ue8m0 scale bytes (one per 64-element NoPE block)
583       1    Padding byte (alignment)
```

**Indexer K slot layout (132 bytes, FP8):**

```
Offset  Size   Content
0       128    128 FP8 values (one 128-dim compressed K vector)
128       4    1 float32 as bytes (single ue8m0 scale for the 128-dim block)
```

---

## 12. Concrete Examples

### 12.1 Decode step — one token, two cache reads

A single decode token at position 1000 with C4A (compress_ratio=4, window_size=4096):

```
Query token: position 1000, sequence length 1001

SWA path (§7.1):
  window covers positions max(1000-4096+1,0)=0 .. 1000 → 1001 raw KV slots
  flash_mla_with_kvcache reads swa_indices[0, 0, :1001]
  → 1001 × 584-byte SWA slots dequantized on-the-fly in the FlashMLA kernel

Compressed sparse path (§7.1):
  compressed context length = 1000 // 4 = 250 compressed slots (slots 0..249)
  indexer top-K: 1024 ≥ 250 → all 250 compressed slots selected
  C4A expansion: 250 × 4 = 1000 original-space slots
  flash_mla_with_kvcache reads extra_indices_in_kvcache[0, 0, :1000]
  → 1000 × 584-byte main MLA slots

Total KV reads: 1001 (SWA) + 1000 (sparse) = 2001 slots
Note: positions 0..999 appear in BOTH paths — the model sees them twice
(once compressed via the sparse path, once raw via the SWA path),
which is by design: the SWA path provides full-fidelity recent context
while the compressed path provides long-range compressed context.
Since 1000 < window_size=4096, the SWA window covers the entire sequence.
```

### 12.2 Prefill — 8 tokens, C4A

8-token fresh prefill (positions 0..7), C4A, compress_ratio=4:

```
After fused_wqa_wkv:
  qr [8, 1536], kv [8, 512]

After SWA insert (§4):
  swa_kv_cache slots 0..7 populated with FP8 KV for positions 0..7

Compressor (§5):
  save_partial_states: all 8 positions written to state_cache
  compress_norm_rope_store fires at pos 3 → compressed slot 0
  compress_norm_rope_store fires at pos 7 → compressed slot 1
  slot_mapping = [-1, -1, -1, 0, -1, -1, -1, 1]

Indexer (§6):
  2 compressed K entries available (slots 0, 1)
  fp8_fp4_mqa_logits: logits [8, 2]  (causal: pos i sees min((i+1)//4, 2) slots)
    pos 0..2: see 0 slots (no complete group yet)
    pos 3..6: see slot 0 only
    pos 7:    sees slots 0 and 1
  top-K selects up to 1024: all visible slots selected (≤2 in this case)

Main attention (§7.2):
  N = 2 compressed entries, SWA window covers all 8 raw tokens
  workspace [1, 10, 512]:
    col 0 = deq(compressed slot 0)
    col 1 = deq(compressed slot 1)
    cols 2..9 = deq(SWA slots 0..7)
  flash_mla_sparse_fwd attends workspace columns per token's combined_indices
```

### 12.3 Compressor state update — C4A example at position 7

C4A, compress_ratio=4, coff=2, head_dim=128. Positions 0–7 have been processed.

```
state_cache slots 0–7 contain (kv+ape, score+ape) for each position:
  slot 0 (pos 0): kv_0 + ape[0,:256], score_0 + ape[0,256:]
  slot 1 (pos 1): kv_1 + ape[1,:256], score_1 + ape[1,256:]
  slot 2 (pos 2): kv_2 + ape[2,:256], score_2 + ape[2,256:]
  slot 3 (pos 3): kv_3 + ape[3,:256], score_3 + ape[3,256:]  ← first group boundary
  slot 4 (pos 4): kv_4 + ape[0,:256], score_4 + ape[0,256:]  ← new group, pos%4=0
  slot 5 (pos 5): kv_5 + ape[1,:256], score_5 + ape[1,256:]
  slot 6 (pos 6): kv_6 + ape[2,:256], score_6 + ape[2,256:]
  slot 7 (pos 7): kv_7 + ape[3,:256], score_7 + ape[3,256:]  ← compress fires here

At pos=7, compress_norm_rope_store fires.
  Gather coff×compress_ratio = 2×4 = 8 states:
  kv_states    [8, 128] = [kv_0+ape0, kv_1+ape1, ..., kv_7+ape3]
  score_states [8, 128] = [score_0+ape0, ..., score_7+ape3]

  group0_gate = softmax(score_states[0:4], dim=0)   # [4,128]: gates for pos 0–3
  group1_gate = softmax(score_states[4:8], dim=0)   # [4,128]: gates for pos 4–7

  compressed_kv = sum(kv_states[0:4]*group0_gate, dim=0)   # [128]
                + sum(kv_states[4:8]*group1_gate, dim=0)    # [128]
  # Result: per-dimension weighted average of 8 token states

  Illustrative gate values (if pos 3 dominates group 0, pos 7 dominates group 1):
    group0_gate[3,:] ≈ 0.7 (other rows sum to 0.3)
    group1_gate[3,:] ≈ 0.6 (other rows sum to 0.4)
    compressed_kv ≈ 0.7*kv_3 + [smaller terms] + 0.6*kv_7 + [smaller terms]

  compressed_slot = 7 // 4 = 1
  rope_pos = (7 // 4) * 4 = 4

  After RMSNorm + GPT-J RoPE at rope_pos=4 + FP8 UE8M0 quantization:
    k_fp8, k_scale → written to indexer k_cache[block=0, offset=1, :132]
    (block = 1 // 256 = 0, offset = 1 % 256 = 1)
```

### 12.4 Scale absorption — concrete calculation

Decode token at position 1000, indexer head h=7, FP8 path.

```
indexer_weights[0, 7] = 0.052   (raw gate weight for this token–head pair)

FP8 quantization of q_fp8[0, 7, :]:
  amax over all 128 dims = 0.83
  scale_raw = 0.83 / 448.0 = 0.001853
  exponent  = ceil(log2(0.001853)) = ceil(-9.08) = -9
  q_scale   = 2^(-9) = 0.001953125

Scale absorption (§6.3):
  weights_out[0, 7] = indexer_weights[0,7] × q_scale × (1/√128) × (1/√64)
                    = 0.052 × 0.001953125 × 0.08839 × 0.125
                    = 0.052 × 0.001953125 × 0.011049
                    = 0.052 × 2.158e-5
                    ≈ 1.122e-6

This single scalar is all the decode kernel needs per (token, head) pair.

True logit formula for this token (summed over all 64 heads):
  logit[b, pos] = Σ_{h=0}^{63} dot(q_fp8[0,h,:], k_fp8[pos,:]) × k_scale[pos] × weights_out[0,h]

For h=7 specifically:
  contribution = dot(q_fp8[0,7,:], k_fp8[pos,:]) × k_scale[pos] × 1.122e-6
```

### 12.5 fp8_paged_mqa_logits decode walk — one block traversal

Decode token at position 5000, compress_ratio=4, block_size=256. Indexer K cache block 19
covers compressed positions 19×256=4864 to 4864+255=5119.

```
get_paged_mqa_logits_metadata assigns block 19 to SM 7.

SM 7, processing token b=0:
  physical_block = block_table[0, 19]   # e.g. physical block 73

  for slot in range(256):
    raw_slot = kv_cache[73, slot, 0, :]    # 132 bytes from indexer K cache
    k_fp8    = raw_slot[:128]              # 128 FP8 values
    k_scale  = reinterpret_float32(raw_slot[128:132])   # 1 float32

    compressed_pos = 19*256 + slot         # absolute compressed position
    if compressed_pos >= context_lens[0]:  # context_lens[0] = 5000//4 = 1250
      break                                # this and remaining slots are beyond context

    raw[h] = dot(q_fp8[0, h, :], k_fp8)  for h in 0..63   # [64] FP8 dot products
    logits[0, compressed_pos] += Σ_h raw[h] × k_scale × weights_out[0, h]

Slots 0..255 of block 19: compressed positions 4864..5119.
  context_lens[0] = 1250, so positions 4864..5119 are all ≥ 1250 → all skipped.
  (This illustrates that schedule_metadata may assign blocks beyond the context;
   the kernel checks context_lens per slot and skips out-of-range slots.)

For a token with context_lens[0]=5000 (compressed pos limit=1250):
  Block 4 covers compressed positions 4*256=1024..1279.
  Slots 0..225: compressed pos 1024..1249 → in context, logit computed.
  Slots 226..255: compressed pos 1250..1279 → out of context, skipped.
```

### 12.6 SWA index computation — two positions

window_size=4096, block_size=256.

```
Token at position 100:
  start = max(100 - 4096 + 1, 0) = 0
  end   = 101
  swa_len = 101

  For p=0:   block=block_table[req,0//256]=block_table[req,0], slot=block*256+0
  For p=100: block=block_table[req,100//256]=block_table[req,0], slot=block*256+100

  swa_indices[t, 0, 0:101]  = physical slots for positions 0..100
  swa_indices[t, 0, 101:]   = -1   (3995 padding slots)
  swa_lens[t] = 101

Token at position 4100:
  start = max(4100 - 4096 + 1, 0) = 5
  end   = 4101
  swa_len = 4096

  For p=5:    block=block_table[req,5//256]=block_table[req,0], slot=block*256+5
  For p=4100: block=block_table[req,4100//256]=block_table[req,16], slot=block16*256+(4100%256)
              = block16*256 + 4

  swa_indices[t, 0, 0]    = physical slot for position 5
  swa_indices[t, 0, 4095] = physical slot for position 4100
  # All 4096 slots filled; no padding (-1) entries
  swa_lens[t] = 4096

Observation: position 4100 falls in block 16 (4100//256=16), offset 4 (4100%256=4).
The first valid position (5) is in block 0 offset 5. Positions 0..4 are excluded by
the window and will not appear in swa_indices.
```

---

## 13. Key Implementation Files

| File | Role |
|---|---|
| `vllm/models/deepseek_v4/attention.py` | `DeepseekV4Attention`; SWA insert; stream orchestration; `DeepseekV4Indexer` |
| `vllm/models/deepseek_v4/compressor.py` | `DeepseekCompressor`; `save_partial_states`; `compress_norm_rope_store_*` |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | CUDA `DeepseekV4FlashMLAAttention`; SWA+sparse FlashMLA calls |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | CUDA output projection: inverse RoPE + FP8 quant + grouped BMM |
| `vllm/models/deepseek_v4/sparse_mla.py` | `DeepseekV4SparseMLABackend`; C128A metadata; `build_c128a_topk_metadata` |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | V4 `fused_indexer_q_rope_quant`; FP8 and MXFP4 Triton kernels |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | `compress_norm_rope_store_triton`; all 3 compressor kernel variants |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | `DeepseekSparseSWABackend`; SWA metadata builder; SWA index kernel |
| `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` | CUDA inverse GPT-J RoPE + FP8 block quantize for output projection |
| `vllm/v1/attention/ops/common.py` | `pack_seq_triton` / `unpack_seq_triton` for spec-decode ragged padding |
| `vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py` | DCP merge: `pack_dcp_topk_candidates_cutedsl`; `stable_topk_from_gathered_candidates_cutedsl` |
| `vllm/utils/deep_gemm.py` | `get_paged_mqa_logits_metadata` SM scheduler |
