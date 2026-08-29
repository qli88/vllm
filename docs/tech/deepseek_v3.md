# DeepSeek V3 / V3.2: Complete Reference

[← Index](deepseek_index.md)

All V3-specific content in one place: architecture, MLA with weight absorption, the sparse indexer, FlashMLA sparse attention, prefill/decode kernels, metadata building, and a full end-to-end example. V4 divergences are noted inline.

---

## Table of Contents

1. [V3 vs V4: Key Differences](#1-v3-vs-v4-key-differences)
2. [V3 Architecture Per Layer](#2-v3-architecture-per-layer)
3. [MLA Math: Q, K, V Projections](#3-mla-math-q-k-v-projections)
   - 3.1 [Q path: `fused_qkv_a_proj` + `q_b_proj`](#31-q-path-fused_qkv_a_proj--q_b_proj)
   - 3.2 [KV path: `kv_a_proj_with_mqa` + `kv_b_proj`](#32-kv-path-kv_a_proj_with_mqa--kv_b_proj)
   - 3.3 [Weight absorption: `W_UK_T` and `W_UV`](#33-weight-absorption-w_uk_t-and-w_uv)
4. [Indexer: Scoring K Cache](#4-indexer-scoring-k-cache)
   - 4.1 [`wk_weights_proj`: fused K + weights GEMM](#41-wk_weights_proj-fused-k--weights-gemm)
   - 4.2 [`wq_b` + `fused_indexer_q_rope_quant`](#42-wq_b--fused_indexer_q_rope_quant)
   - 4.3 [`indexer_k_quant_and_cache`](#43-indexer_k_quant_and_cache)
   - 4.4 [Scale absorption (FP8 path)](#44-scale-absorption-fp8-path)
5. [Indexer Prefill: Gather + MQA Logits + Top-K](#5-indexer-prefill-gather--mqa-logits--top-k)
   - 5.1 [`cp_gather_indexer_k_quant_cache` — CUDA kernel internals](#cp_gather_indexer_k_quant_cache--cuda-kernel-internals)
6. [Indexer Decode: Paged MQA Logits + Top-K](#6-indexer-decode-paged-mqa-logits--top-k)
7. [Main MLA Attention: FlashMLA Sparse](#7-main-mla-attention-flashmla-sparse)
   - 7.1 [`concat_mla_q`](#71-concat_mla_q)
   - 7.2 [`flash_mla_with_kvcache` (decode)](#72-flash_mla_with_kvcache-decode)
   - 7.3 [`flash_mla_sparse_fwd` (prefill)](#73-flash_mla_sparse_fwd-prefill)
   - 7.4 [`FlashMLASparseImpl.forward_mqa`: Full Dispatch Logic](#74-flashmlasparseimplforward_mqa-full-dispatch-logic)
8. [Output Projection](#8-output-projection)
9. [Metadata Builder](#9-metadata-builder)
10. [DCP: Decode Context Parallelism](#10-dcp-decode-context-parallelism)
11. [End-to-End Example: Prefill + Decode](#11-end-to-end-example-prefill--decode)
    - 11.1 [Setup](#111-setup)
    - 11.2 [Fused input GEMM](#112-fused-input-gemm)
    - 11.3 [KV cache write](#113-kv-cache-write)
    - 11.4 [Indexer Q and K](#114-indexer-q-and-k)
    - 11.5 [Prefill: gather + logits + top-K](#115-prefill-gather--logits--top-k)
    - 11.6 [Decode: paged logits + top-K](#116-decode-paged-logits--top-k)
    - 11.7 [Main MLA attention: prefill](#117-main-mla-attention-prefill)
    - 11.8 [Main MLA attention: decode](#118-main-mla-attention-decode)
    - 11.9 [Output projection](#119-output-projection)
    - 11.10 [Stage 9: V3 Output Projection (W_UV BMM detail)](#1110-stage-9-v3-output-projection-w_uv-bmm-detail)
12. [Additional V3 Worked Examples](#12-additional-v3-worked-examples)
    - 12.1 [Weight absorption BMM — concrete shapes with TP=1](#121-weight-absorption-bmm--concrete-shapes-with-tp1)
    - 12.2 [V3 prefill — causal mask in the indexer](#122-v3-prefill--causal-mask-in-the-indexer)
13. [Key Implementation Files](#13-key-implementation-files)

---

## 1. V3 vs V4: Key Differences

| Aspect | V3 / V3.2 | V4 |
|---|---|---|
| Layer types | All identical (no compress_ratio concept) | C4A / C128A (alternating), SWA-only possible |
| SWA cache | None | 512-dim FP8, every layer, window_size=4096 |
| Compressor | None | `DeepseekCompressor` (every layer) |
| Main KV cache | 512 BF16 latent + 64 BF16 rope = **576 BF16 / 1152 bytes** | 448 FP8 NoPE + 128 BF16 RoPE + 8 scale = **584 bytes** |
| Q head dim | 192 (128 NoPE + 64 RoPE) | 512 (448 NoPE + 64 RoPE) |
| Weight absorption | Yes: `ql_nope = q_nope × W_UK_T` | None — FlashMLA handles 512-dim directly |
| Indexer K source | Direct projection of `hidden_states` | Compressor output |
| Indexer K fused with weights | Yes (one GEMM) | No (separate `weights_proj`) |
| Output projection | Single `o_proj` RowParallel | Inverse RoPE + `wo_a` grouped BMM + `wo_b` |
| Stream parallelism | Sequential (no aux streams) | 4-way parallel + 3-way attention overlap |
| `topk_indices` meaning | Direct token positions `[0..seq_len)` | Compressed positions |
| RoPE style | NeoX (first half / second half split) | GPT-J (interleaved pairs) |

---

## 2. V3 Architecture Per Layer

All V3 layers have the same structure — there is no per-layer type variation.

**V3 / V3.2 model config:**

```
hidden_size         = 7168
num_attention_heads = 128   (full MLA attention heads)
q_lora_rank         = 1536  (Q low-rank bottleneck)
kv_lora_rank        = 512   (KV latent dimension = main MLA cache dim)
qk_nope_head_dim    = 128   (nope part of each Q/K head)
qk_rope_head_dim    = 64    (rope part of each Q/K head)
v_head_dim          = 128
qk_head_dim         = 192   (= nope + rope)

index_n_heads       = 64    (indexer: scoring heads)
index_head_dim      = 128   (indexer: 64 rope + 64 nope per head)
index_topk          = 1024  (K: tokens selected per query)
indexer block_size  = 64    (paged indexer K cache)
max_model_len       = 128000
```

**Per-layer module layout with all weight shapes:**

```
┌─────────────────────────────────────────────────────────────┐
│  DeepseekV32Attention  (main MLA attention)                  │
│    fused_qkv_a_proj  [7168 → 1536 + 512 + 64]  (1 GEMM)    │
│    q_a_layernorm     [1536]                                  │
│    q_b_proj          [1536 → 128×192]                       │
│    kv_a_layernorm    [512]                                   │
│    kv_b_proj         [512 → 128×(128+128)]  ← absorbed      │
│    o_proj            [128×128 → 7168]                        │
│    W_UK_T            [128, 128, 512]  ← pre-computed        │
│    W_UV              [128, 512, 128]  ← pre-computed        │
│    attn (FlashMLA sparse)                                    │
│    ┌───────────────────────────────────────────────────┐    │
│    │  DeepseekV32Indexer  (sparse scoring module)      │    │
│    │    wq_b            [1536 → 64×128]                │    │
│    │    wk_weights_proj [7168 → 128+64]  (1 GEMM)     │    │
│    │    k_norm          LayerNorm [128]                 │    │
│    │    k_cache         paged FP8, block_size=64       │    │
│    │    indexer_op      SparseAttnIndexer              │    │
│    └───────────────────────────────────────────────────┘    │
│    KV cache: paged [kv_lora_rank+rope=576 dims/token]        │
└─────────────────────────────────────────────────────────────┘
```

```
Input: hidden_states [T, 7168]

┌─────────────────────────────────────────────────────────────┐
│  DeepseekV32Attention  (main MLA)                           │
│                                                             │
│  fused_qkv_a_proj  [7168 → 1536 + 512 + 64]  (single GEMM)│
│      ├─ q_c   [T, 1536]  → q_a_layernorm                   │
│      ├─ kv_c  [T, 512]   → kv_a_layernorm  → KV cache      │
│      └─ k_pe  [T, 64]    → RoPE → KV cache                  │
│                                                             │
│  q_b_proj  [1536 → 128×192]                                 │
│      split: q_nope [T,128,128],  q_pe [T,128,64]            │
│      ql_nope = bmm(q_nope, W_UK_T)  [T, 128, 512]           │
│      RoPE(q_pe)                                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  DeepseekV32Indexer  (sparse scoring)                │   │
│  │  wk_weights_proj  [7168 → 128+64]                   │   │
│  │  k_norm, RoPE, FP8 quant → indexer k_cache          │   │
│  │  wq_b  [1536 → 64×128]                              │   │
│  │  fused_indexer_q_rope_quant → q_fp8 [T, 64, 128]    │   │
│  │  MQA logits → top-1024 → topk_indices_buffer        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  FlashMLA sparse:                                           │
│    q=(ql_nope,q_pe),  KV=kv_c[topk_indices]                 │
│    output [T, 128, 512] (latent space)                      │
│                                                             │
│  bmm(output, W_UV) → [T, 128, 128]  (v-space)              │
│  o_proj RowParallel [128×128 → 7168]                        │
└─────────────────────────────────────────────────────────────┘

↓ RMSNorm

MoE (same Hash/Standard split as V4, no DSpark)
```

---

## 3. MLA Math: Q, K, V Projections

### 3.1 Q path: `fused_qkv_a_proj` + `q_b_proj`

**`fused_qkv_a_proj`** — a single GEMM replacing separate Q and KV projections:

```
hidden_states [T, 7168]
    × weight [2112, 7168]     (2112 = 1536 + 512 + 64)
    ─────────────────────────────────────────────────────
    output [T, 2112]
      ├─ q_c  [T, 0:1536]    ← Q low-rank activations
      ├─ kv_c [T, 1536:2048] ← KV latent (512-dim)
      └─ k_pe [T, 2048:2112] ← shared rope K (64-dim)
```

**Why fused:** Both Q and KV activations read from the same `hidden_states`. One GEMM with weight `[2112, 7168]` halves the bandwidth load and schedules a single larger GEMM (better tensor core utilization).

**Low-batch optimization** (decode path, T ≤ 16):

```python
if 0 < num_tokens <= 16:
    ops.dsv3_fused_a_gemm(output, input_, weight.T)  # PDL (Persistent Data Loading) kernel
else:
    F.linear(input_, weight)                          # cuBLAS path
```

The `dsv3_fused_a_gemm` PDL kernel is tuned for memory-bandwidth-bound execution at tiny batch sizes, prefetching the weight matrix tiles asynchronously.

**Post-processing:**

```python
q_c  = output[:, :1536]
kv_c = output[:, 1536:2048]
k_pe = output[:, 2048:2112]

q_c  = q_a_layernorm(q_c)    # RMSNorm with learned weight [1536]
kv_c = kv_a_layernorm(kv_c)  # RMSNorm with learned weight [512]
```

**`q_b_proj`** — projects Q from 1536-dim bottleneck to 128 full heads:

```
q_c [T, 1536]  → q_b_proj [1536 → 128×192]  →  q [T, 128, 192]

split:
  q_nope [T, 128, 128]   ← content-dependent part (will be absorbed into W_UK_T)
  q_pe   [T, 128,  64]   ← position-dependent part (gets NeoX RoPE)
```

**NeoX RoPE on q_pe** (V3 style, differs from V4 GPT-J):

```
# NeoX: first half and second half of the 64-dim rope vector
# For q_pe[t, h, :] = [x0, x1, ..., x31, x32, ..., x63]:
x0 = q_pe[t, h, 0:32]     # first 32 dims
x1 = q_pe[t, h, 32:64]    # last 32 dims

r0 = x0*cos(pos) - x1*sin(pos)   → bf16 → fp32  # rotate pairs
r1 = x1*cos(pos) + x0*sin(pos)   → bf16 → fp32

# Compare to V4 GPT-J (interleaved):
# x_even = q[64::2], x_odd = q[65::2]  ← pairs by (2k, 2k+1), not halves
```

### 3.2 KV path: `kv_a_proj_with_mqa` + `kv_b_proj`

After `fused_qkv_a_proj`:

```
kv_c [T, 512]  (after kv_a_layernorm)  ← stored in main MLA KV cache
k_pe [T, 64]   (after NeoX RoPE)       ← stored alongside kv_c

Main MLA KV cache slot: [kv_c (512 dims) | k_pe (64 dims)] = 576 dims per token
  BF16 format: 576 × 2 = 1152 bytes per slot
  FP8 format: 576 bytes (if fp8_kvcache=True, with additional scale metadata)
```

**`kv_b_proj`** is used at attention time, NOT cached:

```
kv_b_proj.weight: [512, 128×(128+128)] = [512, 32768]
# Input: kv_c [512] per cached token
# Output: [128, 256] per cached token (128 heads × (128 K_nope + 128 V))
# K_nope [128, 128], V [128, 128]

# BUT this is never computed in the hot path!
# Weight absorption (§3.3) moves this computation to the Q side.
```

### 3.3 Weight absorption: `W_UK_T` and `W_UV`

**The problem:** Standard MLA would require running `kv_b_proj` for every cached token at every step:

```
# Without absorption (expensive):
for each cached token k:
    kv_expanded = kv_b_proj(kv_c[k])   # [128, 256]
    k_nope[k] = kv_expanded[:, :128]   # [128, 128]
    v[k]      = kv_expanded[:, 128:]   # [128, 128]
    score[h, k] = dot(q_nope[h], k_nope[k, h]) + dot(q_pe[h], k_pe[k])
    # Cost: N × (2×512×128×128) operations = huge at long contexts
```

**The solution:** Absorb `kv_b_proj`'s K-side weights into Q (done once per step):

```python
# At model init: extract and pre-transpose the nope-key and value slices
W = kv_b_proj.weight   # [512, 128*(128+128)] = [512, 32768]
W_reshaped = W.view(512, 128, 256)           # [kv_lora_rank, num_heads, nope+v]
W_UK = W_reshaped[:, :, :128]               # [512, 128, 128] = [kv_rank, heads, nope]
W_UV = W_reshaped[:, :, 128:]               # [512, 128, 128] = [kv_rank, heads, v]

# Pre-transpose for efficient matmul:
W_UK_T = W_UK.permute(1, 2, 0)              # [128, 128, 512] = [heads, nope, kv_rank]
W_UV   = W_UV.transpose(0, 1)               # [128, 512, 128] = [heads, kv_rank, v]
# These are stored on the module and reused every inference step.

# At inference — Q side (done ONCE per step):
ql_nope = torch.bmm(
    q_nope.transpose(0, 1),    # [128, T, 128]  (batch over heads)
    W_UK_T,                    # [128, 128, 512]
).transpose(0, 1)              # → [T, 128, 512]
# ql_nope now carries the pre-absorbed W_UK_T factor

# Score computation (no kv_b_proj needed!):
# dot(ql_nope[t, h, :], kv_c[k, :])  replaces  dot(q_nope[t,h,:], W_UK[h,:,:], kv_c[k,:])
# The 512-dim dot is computed directly against the cached 512-dim kv_c.
```

**Why this works:**

```
score[t, h, k]
  = dot(q_nope[t,h,:], k_nope[k,h,:])     + dot(q_pe[t,h,:], k_pe[k,:])
  = dot(q_nope[t,h,:], W_UK[h,:,:] @ kv_c[k,:]) + dot(q_pe[t,h,:], k_pe[k,:])
  = dot(q_nope[t,h,:] @ W_UK[h,:,:]^T, kv_c[k,:]) + dot(q_pe[t,h,:], k_pe[k,:])
  = dot(ql_nope[t,h,:], kv_c[k,:])        + dot(q_pe[t,h,:], k_pe[k,:])
       ↑ one 512-dim dot per (t,h,k)            ↑ one 64-dim dot per (t,h,k)
```

No `kv_b_proj` runs per cached token. The 128-dim K reconstruction is replaced by a 512-dim dot with the cached latent.

**Value side (W_UV):**

```
# Without absorption: v[k] = W_UV[h,:,:] @ kv_c[k,:]  per cached token (expensive)
# With absorption: attention runs in latent space, outputs latent-space weighted sum
out_latent[t, h, :] = sum_k( alpha[t,h,k] × kv_c[k, :] )   # [T, 128, 512]

# Then apply W_UV ONCE after attention:
attn_out[t, h, :] = bmm(out_latent, W_UV)[t, h, :]           # [T, 128, 128]
# [T, 128, 512] × [128, 512, 128] → [T, 128, 128]  (batch over heads)
```

**Concrete example:**

```
T=1, H=128, kv_lora_rank=512, nope=128, v=128

q_nope [1, 128, 128]:
  q_nope[0, 0, :] = [0.41, -0.23, 0.87, ...]   (head 0 of token 0)

W_UK_T [128, 128, 512]:
  W_UK_T[0, :, :] = W_UK's head-0 slice, transposed

bmm step (head 0):
  ql_nope[0, 0, :] = q_nope[0, 0, :] @ W_UK_T[0, :, :]   [128] × [128, 512] → [512]
  ql_nope[0, 0, :] = [w₀ × 0.41 + w₁ × (-0.23) + ... for 512 output dims]

At attention time (decode, T=1, N cached tokens):
  For each cached token k:
    dot(ql_nope[0, 0, :], kv_c[k, :])   # 512-dim dot → scalar score contribution
    + dot(q_pe_rotated[0, 0, :], k_pe[k, :])   # 64-dim dot
  Softmax → alpha [N]
  out_latent[0, 0, :] = sum_k(alpha[k] × kv_c[k, :])   # [512]

After attention:
  attn_out[0, 0, :] = out_latent[0, 0, :] @ W_UV[0, :, :]   # [512] × [512, 128] → [128]
```

---

## 4. Indexer: Scoring K Cache

### 4.1 `wk_weights_proj`: fused K + weights GEMM

**V3 specific:** Both the indexer K vector and the per-head weight scalars are produced by a single GEMM, unlike V4 which has separate GEMMs for each.

```
wk_weights_proj: MergedColumnParallelLinear [7168 → 128 + 64]
  weight: [192, 7168]

hidden_states [T, 7168]
    × weight [192, 7168]
    ──────────────────────────────────
    output [T, 192]
      ├─ k       [T, 128]   ← indexer K vector (128-dim, one per token)
      └─ weights [T, 64]    ← per-head weight scalars (one per indexer head)
```

**Why fused:** Both `k` and `weights` read from `hidden_states`. One GEMM halves bandwidth.

**What `weights [T, 64]` means:**

Each value `weights[t, h]` is a learned "pre-score" — how much head `h` thinks the K from token `t` should be attended to, derived purely from `hidden_states` and without seeing the query. This is analogous to a content-based key importance signal. It amplifies or suppresses each head's contribution to the final logit independent of the query.

**Post-processing:**

```python
k       = output[:, :128]    # [T, 128]  bf16
weights = output[:, 128:]    # [T, 64]   bf16

k = k_norm(k)                # LayerNorm [128] (not RMSNorm in V3)

# Split k into NoPE and RoPE halves:
k_nope = k[:, :64]           # [T, 64]  content part
k_pe   = k[:, 64:]           # [T, 64]  position part

# Apply NeoX RoPE to k_pe:
# k_pe uses indexer's own rope embeddings, separate from the main MLA rope
k_pe = neox_rope(k_pe, positions)

# Concatenate back:
k = torch.cat([k_pe_rotated, k_nope], dim=-1)  # [T, 128]
```

Note: V3 concatenates as `[k_pe_rotated | k_nope]` (RoPE first). V4 uses `[k_nope | k_pe]` (NoPE first, different indexer head layout).

### 4.2 `wq_b` + `fused_indexer_q_rope_quant`

```
q_c [T, 1536]   (shared with main MLA's Q bottleneck)
  → wq_b  [1536 → 64×128]   (indexer's own projection)
  → q_idx [T, 64, 128]   bf16
```

**`fused_indexer_q_rope_quant` (Triton, grid=(T, 64)) — V3/FP8 variant:**

```
For token t, head h (NeoX style):
  x0 = q_idx[t, h, 0:32]      # first 32 dims (rope, NeoX half)
  x1 = q_idx[t, h, 32:64]     # second 32 dims (rope, NeoX half)
  q_nope = q_idx[t, h, 64:]   # 64 nope dims

  r0 = x0*cos(pos) - x1*sin(pos)  → bf16 → fp32  (NeoX rotation)
  r1 = x1*cos(pos) + x0*sin(pos)  → bf16 → fp32

  # FP8 e8m0 quantization:
  amax = max(|r0|, |r1|, |q_nope|)
  q_scale = 2^(ceil(log2(amax / 448.0)))   # UE8M0 power-of-2 scale

  q_fp8[t, h, 0:32]  = clamp(r0 / q_scale, -448, 448).to(fp8)
  q_fp8[t, h, 32:64] = clamp(r1 / q_scale, -448, 448).to(fp8)
  q_fp8[t, h, 64:]   = clamp(q_nope / q_scale, -448, 448).to(fp8)

  # Scale absorption (single scalar per head for FP8):
  weights_out[t, h] = weights[t, h] × q_scale × (1/√128) × (1/√64)
```

**V3 uses NeoX RoPE, V4 uses GPT-J:**

```
NeoX (V3):
  [x0₀, x0₁, ..., x0₃₁, x1₀, x1₁, ..., x1₃₁]
   ╰─── first half ──────╯  ╰─── second half ──╯
  r0 = x0*cos - x1*sin
  r1 = x1*cos + x0*sin

GPT-J (V4):
  [pair₀_even, pair₀_odd, pair₁_even, pair₁_odd, ...]  (interleaved)
  r_even = x_even*cos - x_odd*sin
  r_odd  = x_odd*cos  + x_even*sin
```

### 4.3 `indexer_k_quant_and_cache`

**File:** `csrc/libtorch_stable/cache_kernels.cu`

CUDA kernel `indexer_k_quant_and_cache_kernel`, grid `(T, ceil(128/VEC_SIZE/4))`, block `(32, 4)`.

```
For each token t:
  1. Load k[t, :128]  (4 elements per thread, 32 threads per warp)
  2. Warp-level absmax via __shfl_xor_sync across all 32 lanes
  3. UE8M0 scale: exponent = ceil(log2(amax / 448.0))
  4. FP8 quantize: k_fp8 = clamp(k / 2^exponent, -448, 448).to(fp8)
  5. Write to paged indexer k_cache:
     block_idx = slot_mapping[t] // 64   (block_size=64 in V3)
     block_offset = slot_mapping[t] % 64
     kv_cache[block_idx, block_offset, 0:128] = k_fp8     (128 FP8 bytes)
     kv_cache[block_idx, block_offset, 128:132] = scale    (4 bytes as float32)
```

**V3 indexer block_size=64, V4=256.** This means V3 pages the indexer cache in 64-slot blocks; V4 in 256-slot blocks.

**Cache slot layout (132 bytes per slot):**

```
bytes   0..127  : 128 FP8 values (one 128-dim K vector)
bytes 128..131  : float32 scale (one ue8m0 block covering all 128 dims)
```

### 4.4 Scale absorption (FP8 path)

The true logit:

```
logit[q, k] = Σ_h  dot(q_bf16[q,h,:], k_bf16[k,:])  ×  w[q,h]  ×  (1/√128)  ×  (1/√64)
```

After FP8 quantization (q_scale per token-head, sk for K):

```
= Σ_h  dot(q_fp8 × q_scale, k_fp8 × sk)  ×  w  ×  (1/√128)  ×  (1/√64)
= Σ_h  dot(q_fp8, k_fp8)  ×  sk  ×  [q_scale × w × (1/√128) × (1/√64)]
                                       ╰── weights_out[q,h] ─────────────╯
```

`weights_out [T, 64]` folds `q_scale`, `w[t,h]`, `(1/√128)`, and `(1/√64)` into one scalar. The MQA logit GEMM multiplies this scalar into each tile accumulator in its epilogue.

---

## 5. Indexer Prefill: Gather + MQA Logits + Top-K

### `cp_gather_indexer_k_quant_cache` — CUDA kernel internals

**File:** `vllm/_custom_ops.py:2852` (Python), `csrc/libtorch_stable/cache_kernels.cu:612` (CUDA)

Gathers all historical compressed K vectors from the paged indexer k_cache into a contiguous flat buffer for the MQA GEMM.

**Signature:**
```python
cp_gather_indexer_k_quant_cache(
    kv_cache:    [num_blocks, 64, 132] uint8,  # paged FP8 cache (block_size=64)
    dst_k:       [N, 128]             fp8,     # output: flat gathered K values
    dst_scale:   [N, 4]               uint8,   # output: per-token float32 scale as raw bytes
    block_table: [B, max_blocks]      int32,   # (request, block_idx) → physical block
    cu_seq_lens: [B+1]                int32,   # cumulative sequence lengths [0, L0, L0+L1, ...]
)
```

`N = sum(seq_lens)` — total historical tokens across all prefill requests in the chunk.

**CUDA kernel** `cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>`,
grid `(ceil(N/BY), ceil(128/(8×16)))`, block `(8, BY)`:

`BLOCK_Y_SIZE` is selected by total token count N:
- N < 32: BY=1
- N < 64: BY=2
- N < 128: BY=4
- N < 256: BY=8
- N < 512: BY=16
- else: BY=32

This balances occupancy (more threads = better SM utilization) vs shared memory (larger BY needs more smem for the cu_seq_lens cooperative scan).

**For each token `t` in `[0, N)`:**

```
Step 1: Determine which request token t belongs to.
  Each row of BY threads cooperatively binary-searches cu_seq_lens in shared memory.
  → batch_idx such that cu_seq_lens[batch_idx] ≤ t < cu_seq_lens[batch_idx+1]

Step 2: Compute inbatch position.
  inbatch_seq_idx = t - cu_seq_lens[batch_idx]

Step 3: Look up physical block.
  block_number = block_table[batch_idx, inbatch_seq_idx // 64]

Step 4: Compute slot offset within block.
  slot_offset = inbatch_seq_idx % 64

Step 5: Load 16 bytes (128-bit) of FP8 K values:
  src = kv_cache[block_number * block_stride + slot_offset * 128 + head_idx]
  dst_k[t * 128 + head_idx] = src   (via float4 store for coalescing)

Step 6: Thread 0 of each row reads the float32 scale:
  scale = kv_cache[block_number * block_stride + slot_offset * 128 + 128]  # last 4 bytes
  dst_scale[t * 4 : t * 4 + 4] = scale_as_4_bytes
```

**Result:** Dense `dst_k [N, 128] fp8` and `dst_scale [N, 4] uint8` matrices, laid out for `fp8_fp4_mqa_logits`.

The ROCm equivalent is `cp_gather_indexer_k_quant_cache_triton` which uses a SHUFFLE layout and is covered in [deepseek_v4_rocm.md §3.2](deepseek_v4_rocm.md).

---

**Full prefill flow:**

```python
# Step 1: Gather all historical K tokens from the paged indexer k_cache
cp_gather_indexer_k_quant_cache(
    kv_cache,      # [num_blocks, 64, 132]
    dst_k,         # [N, 128]  fp8   (N = total historical tokens across prefill reqs)
    dst_scale,     # [N, 4]    uint8 (float32 scale as raw bytes)
    block_table,   # [B, max_blocks]
    cu_seq_lens,   # [B+1]
)

# Step 2: MQA logits (causally masked)
fp8_fp4_mqa_logits(
    q = (q_fp8 [M, 64, 128], None),    # M = num prefill query tokens
    kv = (dst_k [N, 128], dst_scale [N, 4]),
    weights = weights_out [M, 64],
    cu_seqlen_ks,   # [M]  start of K range for each query (usually all 0)
    cu_seqlen_ke,   # [M]  end of K range = position_i + 1 (causal)
) → logits [M, N]  float32

# Causal masking: token at position p sees K positions [0, p]
# cu_seqlen_ke[i] = positions[i] + 1   (direct token positions, not compressed)
# This is different from V4 where ke[i] = (positions[i] + 1) // compress_ratio

# Step 3: Top-K
top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, K=1024)
→ topk_indices_buffer [M, 1024]  int32
```

**Key V3 vs V4 difference in `cu_seqlen_ke`:**

```
V3: ke[i] = position_i + 1   → query at position 50 sees K tokens [0..50] (direct positions)
V4: ke[i] = (position_i + 1) // compress_ratio
    → query at position 50 with compress_ratio=4 sees compressed tokens [0..12]
    (compressed positions, each covering 4 original tokens)
```

**Metadata building (`_build_prefill_chunk_metadata_kernel`):**

```python
# For each prefill token i at position p within a chunk:
cu_seqlen_ks[i] = 0            # always start at 0 (no DCP sharding in this simple case)
cu_seqlen_ke[i] = p + 1        # see up to and including this position

# Chunking: limits chunk size to avoid logit matrix OOM
# max_logit_mb = VLLM_SPARSE_INDEXER_MAX_LOGITS_MB × 1024²
# chunk such that M × N × 4 bytes ≤ max_logit_mb
```

---

## 6. Indexer Decode: Paged MQA Logits + Top-K

```python
fp8_fp4_paged_mqa_logits(
    q = (q_fp8 [B, next_n, 64, 128], None),
    kv_cache = [num_blocks, 64, 1, 132],   # block_size=64
    weights = weights_out [B*next_n, 64],
    context_lens = [B, next_n],            # actual sequence length per decode token
    block_tables = [B, max_blocks],
    schedule_metadata,                     # pre-computed SM tile assignments
    max_model_len,
) → logits [B*next_n, max_model_len]  float32
```

**`schedule_metadata`** from `get_paged_mqa_logits_metadata`:

```python
get_paged_mqa_logits_metadata(
    context_lens [B, next_n],
    block_size = 64,
    num_sms,
) → schedule_metadata [num_sms+1, 2]
```

Assigns (request, block) pairs to SMs. Without this, SM 0 handles all of request 0's long history while other SMs are idle. With it, each SM processes roughly `total_blocks / num_sms` blocks across all requests.

**V3 decode top-K:**

Uses the same kernel set as V4:
- `cooperative_topk` (CUDA SM≥90, rows≤32, topk∈{512,1024,2048})
- `persistent_topk` (CUDA, larger batch)
- `top_k_per_row_decode` (fallback, any platform)

The output `topk_indices_buffer [B*next_n, 1024]` contains **direct token positions** (not compressed positions as in V4). Each entry `topk_indices_buffer[b, i]` is a physical slot index in the main MLA KV cache.

---

## 7. Main MLA Attention: FlashMLA Sparse

### 7.1 `concat_mla_q`

Before calling FlashMLA, V3 must concatenate the NoPE and RoPE halves of Q:

```python
# After weight absorption: ql_nope [T, H, 512] and q_pe_rotated [T, H, 64] are separate tensors
# FlashMLA expects a single concatenated [T, H, 576] Q tensor.
ops.concat_mla_q(
    ql_nope,    # [T, H, 512]   absorbed NoPE query
    q_pe,       # [T, H, 64]    RoPE query
    q_out,      # [T, H, 576]   pre-allocated output buffer
)
# q_out[t, h, :512] = ql_nope[t, h, :]
# q_out[t, h, 512:] = q_pe[t, h, :]
```

V4 does NOT need `concat_mla_q` because V4 uses the full 512-dim head without splitting.

### 7.2 `flash_mla_with_kvcache` (decode)

```python
flash_mla_with_kvcache(
    q = q_out.unsqueeze(1),              # [B, 1, H, 576]
    k_cache = kv_c_and_k_pe_cache,      # [num_blocks, 64, 1, 576]  (block_size=64)
    block_table = ...,                   # ignored when indices is not None
    head_dim_v = 512,                    # V is in kv_lora_rank=512 space (latent)
    tile_scheduler_metadata,
    cache_seqlens,                       # ignored when indices is not None
    is_fp8_kvcache = True,               # or False for BF16 cache
    indices = topk_indices_buffer,       # [B, 1, 1024]  physical slot IDs
    topk_length = topk_lens,             # [B, 1]
    softmax_scale = 1/√192,              # 1/√(nope_dim + rope_dim) = 1/√192
    attn_sink = None,                    # V3 does not use attn_sink
    extra_k_cache = None,                # V3 has no SWA cache
    extra_indices_in_kvcache = None,
    out = output,                        # [B, 1, H, 576]  latent-space output
)
```

**Important: `head_dim_v = 512`, not 192.**

In V3, the FlashMLA kernel operates in the 512-dim latent space for value accumulation (even though V is ultimately 128-dim after applying W_UV). The kernel accumulates `kv_c` (512-dim) weighted by softmax attention weights, producing a latent-space output `[T, H, 512]`. W_UV is applied afterwards to convert from 512-dim latent to 128-dim V space.

The reason `softmax_scale = 1/√192` (not `1/√512`): the score is computed as `dot(ql_nope [512], kv_c [512]) + dot(q_pe [64], k_pe [64])`. The effective "head dimension" for scaling purposes is 192 (the original Q head dim before absorption), not 512.

### 7.3 `flash_mla_sparse_fwd` (prefill)

```python
# Pre-gather KV from the paged main MLA cache into a BF16 workspace:
cp_gather_and_upconvert_fp8_kv_cache(
    workspace,     # [T_q, 1, 576]  bf16
    kv_cache,      # paged FP8 main MLA KV cache
    topk_indices,  # workspace-local indices
    ...
)

flash_mla_sparse_fwd(
    q = q_out,                     # [T_q, H, 576]  (concat_mla_q output)
    kv = workspace,                # [T_kv, 1, 576]  bf16 gathered workspace
    indices = local_indices,       # [T_q, 1, 1024]  workspace-local indices
    sm_scale = 1/√192,
    topk_length = topk_lens,
    out = output,                  # [T_q, H, 576]  latent-space output
    # No attn_sink in V3
)
```

### 7.4 `FlashMLASparseImpl.forward_mqa`: Full Dispatch Logic

**File:** `vllm/v1/attention/backends/mla/flashmla_sparse.py:839`

This is the main dispatch point for V3 sparse MLA attention. It handles three different execution paths based on cache format and batch composition.

```python
def forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer):
    # q arrives as a (ql_nope, q_pe) tuple when weight absorption is done outside:
    if isinstance(q, tuple):
        ql_nope, q_pe = q
        ops.concat_mla_q(ql_nope, q_pe, q_out)   # [T, H, 512] + [T, H, 64] → [T, H, 576]
        q = q_out

    topk_indices = topk_indices_buffer[:num_actual_tokens]   # [T, 1024] from indexer

    # --- Path selection ---

    if not use_fp8_cache:
        # Path 1: BF16 KV cache
        # Rare — used with older V3 configs that didn't enable FP8 KV caching.
        # flash_mla_with_kvcache called with is_fp8_kvcache=False.
        _forward_bf16_kv(q, kv_cache, topk_indices, attn_metadata)

    elif attn_metadata.fp8_use_mixed_batch:
        # Path 2: FP8 mixed batch
        # Triggered when TP is large enough that each rank has very few heads
        # (e.g. TP=16, 128/16=8 heads per rank). At 8 heads per rank, the
        # BF16 prefill kernel's overhead (dequantize into workspace + kernel launch)
        # exceeds the benefit. Instead, the FP8 kernel handles both prefill and decode.
        # Prefill tokens are handled with extra padding to make the Q shape regular.
        _forward_fp8_kv_mixed_batch(
            q, kv_cache, topk_indices, attn_metadata,
            tile_scheduler_metadata, num_splits,
        )

    else:
        # Path 3: FP8 separate prefill + decode (default for V3.2)
        _forward_fp8_kv_separate_prefill_decode(
            q, kv_cache, topk_indices, attn_metadata)
```

**Path 3 internals (`_forward_fp8_kv_separate_prefill_decode`):**

```python
# Decode tokens (num_decodes > 0):
if num_decode_tokens > 0:
    _fp8_flash_mla_kernel(
        q = q[:num_decode_tokens].reshape(B, next_n, H, 576),
        k_cache = kv_cache,                  # [blocks, 64, 1, 576] paged FP8
        indices = topk_indices[:num_decode_tokens].reshape(B, next_n, 1024),
        topk_length = topk_lens[:B],
        is_fp8_kvcache = True,
        # → flash_mla_with_kvcache(is_fp8_kvcache=True)
        # Optimized for small Q, large K (decode scenario)
    )

# Prefill tokens (num_prefills > 0):
if num_prefill_tokens > 0:
    # Dequantize FP8 → BF16 into contiguous workspace:
    cp_gather_and_upconvert_fp8_kv_cache(
        workspace,         # [chunk_N, 1, 576] bf16
        kv_cache,          # paged FP8 cache
        prefill_topk_indices,  # local workspace indices
        ...
    )
    _bf16_flash_mla_kernel(
        q = q[num_decode_tokens:],   # [T_q, H, 576]
        kv = workspace,              # [chunk_N, 1, 576] bf16
        indices = local_indices,     # workspace-local column indices
        # → flash_mla_sparse_fwd(BF16 workspace)
        # Chunked FlashAttention-style prefill in BF16
    )
```

**Why separate prefill and decode:**

The FP8 FlashMLA kernel (`is_fp8_kvcache=True`) is tuned for decode: it reads FP8 K directly from the paged cache, dequantizing block by block. For prefill (many query tokens, many KV tokens), the dequantization overhead per KV block is amortized over more query tokens, but the gather step (linearizing the paged cache into a workspace) is unavoidable for prefill anyway. Given that the workspace must be created for prefill, using the BF16 kernel on the already-dequantized workspace is more efficient than re-quantizing to FP8 for the FP8 kernel.

**`get_mla_metadata` — tile scheduler for decode:**

```python
tile_md, num_splits = get_mla_metadata(
    cache_seqlens,                         # [B] full sequence lengths
    num_q_tokens_per_head_k,               # B × next_n × H // H_K
    num_heads_k = 1,                       # MLA: single shared K head
    num_heads_q = H,
    topk = 1024,
    is_fp8_kvcache = True,
)
```

Assigns contiguous `(batch, sparse_block)` ranges to each SM, balancing load across sequences with different sparse token counts. Similar to `get_paged_mqa_logits_metadata` for the indexer, but for the main MLA attention kernel.

---

## 8. Output Projection

V3 output projection is a simple two-step pipeline (no inverse RoPE needed):

```
# attn_out [T, H, 512]  (latent space, from FlashMLA)

# Step 1: W_UV application — convert from latent to V space
attn_out_v = torch.bmm(
    attn_out.transpose(0, 1),    # [128, T, 512]   (batch over heads)
    W_UV,                         # [128, 512, 128]  (per head)
).transpose(0, 1)                 # → [T, 128, 128]   (V space)

# Step 2: o_proj — V to hidden
attn_out_v = attn_out_v.reshape(T, -1)   # [T, 128×128=16384]
output = o_proj(attn_out_v)               # RowParallel [16384 → 7168]
                                          # → [T, 7168]
```

**Why no inverse RoPE in V3:**

V3 Q was split into `q_nope` and `q_pe`; the RoPE was applied only to `q_pe`. The FlashMLA kernel sees `ql_nope` (absorbed, no position rotation) and `q_pe` (rotated). The attention output is accumulated in the **latent space** (via `kv_c`), which was never rotated. So there is no rotation to undo.

V4 is different: the full 512-dim Q head is rotated (RoPE on the last 64 dims), and the attention output mixes contributions from the rotated frame — hence the need for inverse RoPE before `wo_a`.

---

## 9. Metadata Builder

`DeepseekV32IndexerMetadataBuilder.build()` runs on CPU each decode step to prepare:

**Decode metadata:**

```python
metadata.decode = DeepseekV32IndexerDecodeMetadata(
    seq_lens = [B, next_n]  int32,      # full context length per token
    block_table = [B, max_blocks] int32, # paged KV physical block indices
    decode_lens = [B] int32,             # decode tokens per request (1 or spec_len)
    schedule_metadata = ...,             # pre-computed SM assignments for paged MQA
)
```

**Prefill metadata (per chunk):**

```python
for chunk in chunks:
    metadata.prefill.chunks.append(DeepseekV32IndexerPrefillChunkMetadata(
        cu_seq_lens = [B_chunk + 1],      # cumulative K lengths per request
        cu_seqlen_ks = [M_chunk],          # causal mask start per query
        cu_seqlen_ke = [M_chunk],          # causal mask end per query
        block_table = [B_chunk, max_blocks],
        token_start, token_end,            # slice into the batch token array
        total_seq_lens,                    # sum of K lengths in this chunk
        token_to_seq = [M_chunk],          # which batch index each token belongs to
    ))
```

**Chunking policy:**

```
Chunk until either limit is hit:
  1. total_seq_lens > 40 × max_model_len
     (prevents gather workspace from exceeding memory)
  2. M × N × 4 bytes > VLLM_SPARSE_INDEXER_MAX_LOGITS_MB × 1024²
     (prevents logit matrix from exceeding memory)
```

---

## 10. DCP: Decode Context Parallelism

DCP splits the KV sequence across `world_size` GPUs using interleaved sharding:

```
world_size=2, interleave=1:
  GPU 0 owns: token positions [0, 2, 4, 6, ...] (even)
  GPU 1 owns: token positions [1, 3, 5, 7, ...] (odd)

world_size=2, interleave=I:
  GPU 0 owns: positions [0..I-1, 2I..3I-1, 4I..5I-1, ...]
  GPU 1 owns: positions [I..2I-1, 3I..4I-1, 5I..6I-1, ...]
```

**DCP merge algorithm:**

```python
# Step 1: Local top-K on each GPU's shard
local_topk = top_k(local_logits, K=1024)   # [T, 1024]  local shard positions

# Step 2: pack_dcp_topk_candidates_cutedsl
# Convert local positions to (score, global_id) pairs:
for each (query t, local_pos local_idx):
    score = local_logits[t, local_idx]
    global_id = (local_idx // interleave) * (world_size * interleave) + rank * interleave + (local_idx % interleave)
    packed[t, col, :] = [score, float(global_id)]
# packed [T, 1024, 2]

# Step 3: all_gather across DCP group
gathered = all_gather(packed, dim=1)   # [T, world_size*1024, 2]

# Step 4: stable_topk_from_gathered_candidates_cutedsl
# 64-bit sortable key: (flipped_float_bits << 32) | (~global_id)
# Radix sort to find top-1024 globally; ties broken by smaller global_id
out = topk_from_gathered(gathered, K=1024)   # [T, 1024] global token IDs
```

The 64-bit key encoding:
- Upper 32 bits: float score reinterpreted as uint32, then XOR'd to make sortable (IEEE 754: all zeros = min float, all ones = max float, after XOR flip)
- Lower 32 bits: `~global_id` = bitwise NOT of global_id, so smaller global_id → larger lower bits → preferred in top-K (stable tie-breaking: earlier tokens win)

---

## 11. End-to-End Example: Prefill + Decode

### 11.1 Setup

```
Batch: 2 requests
  Request A: 6-token prompt "The Eiffel Tower is in"  (prefill)
    token_ids: [791, 96310, 17720, 374, 304, 14366]  (note: missing "." for prefill-only)
    positions: [0, 1, 2, 3, 4, 5]
  Request B: continue a 3000-token conversation (decode)
    position: 2999

Token layout (decode first):
  index 0: decode token (req B, position=2999)
  index 1..6: prefill tokens 0..5 (req A, positions 0..5)
  T = 7 total tokens
```

### 11.2 Fused input GEMM

```
hidden_states [7, 7168]
    × fused_qkv_a_proj.weight [2112, 7168]
    ─────────────────────────────────────────
    output [7, 2112]
      q_c  [7, 1536]   ← normalized by q_a_layernorm
      kv_c [7, 512]    ← normalized by kv_a_layernorm
      k_pe [7, 64]     ← NeoX RoPE applied

# Example: decode token (t=0, position=2999)
q_c[0, :3] after norm ≈ [0.91×w₀, -0.43×w₁, 0.72×w₂, ...]
kv_c[0, :3] after norm ≈ [0.63×v₀, -0.28×v₁, 0.54×v₂, ...]
k_pe[0, :3] after NeoX RoPE at position 2999:
  x0 = k_pe_raw[0:32], x1 = k_pe_raw[32:64]
  r0 = x0*cos(2999) - x1*sin(2999)
  r1 = x1*cos(2999) + x0*sin(2999)
```

### 11.3 KV cache write

```
# Write kv_c and k_pe_rotated to main MLA KV cache for all 7 tokens:
slot_mapping = [5000, 200, 201, 202, 203, 204, 205]
  (physical slots: 5000 for decode, 200-205 for prefill)

# Each slot: 576 values = [kv_c (512 bf16) | k_pe_rotated (64 bf16)]
# → written as 1152 bytes or FP8 compressed
```

### 11.4 Indexer Q and K

**Indexer K (from `wk_weights_proj`):**

```
wk_weights_proj([7, 7168]) → k [7, 128] + weights [7, 64]

k_norm(k) → split → k_nope [7,64], k_pe [7,64]
NeoX RoPE on k_pe
k [7, 128] = cat([k_pe_rotated, k_nope])

For token 0 (decode, position=2999):
  k[0, 0:32] = k_pe_rotated[0, 0:32]  (position-dependent)
  k[0, 64:96] = k_nope[0, 0:32]       (content-dependent)

indexer_k_quant_and_cache(k, indexer_k_cache, slot_mapping=[5000,200,...]):
  For token 0:
    amax = max(|k[0, :]|) = 0.87
    exponent = ceil(log2(0.87/448)) = ceil(-8.987) = -8
    scale = 2^(-8) = 0.003906
    k_fp8[0, :] = clamp(k[0,:] / 0.003906, -448, 448).to(fp8)
    kv_cache[5000//64=78, 5000%64=8, :128] = k_fp8[0, :]
    kv_cache[78, 8, 128:132] = float32_as_bytes(0.003906)
```

**Indexer Q (from `wq_b` + `fused_indexer_q_rope_quant`):**

```
wq_b(q_c) → q_idx [7, 64, 128]

fused_indexer_q_rope_quant (NeoX style, FP8):
  For token 0 (position=2999), head 3:
    x0 = q_idx[0, 3, 0:32]    # first 32 (rope, NeoX)
    x1 = q_idx[0, 3, 32:64]   # second 32 (rope, NeoX)
    q_nope = q_idx[0, 3, 64:] # 64 nope dims

    r0 = x0*cos(2999) - x1*sin(2999)  → fp32
    r1 = x1*cos(2999) + x0*sin(2999)  → fp32

    amax = max(|q_nope|, |r0|, |r1|) = 0.94
    q_scale = 2^(ceil(log2(0.94/448))) = 2^(-8) = 0.003906

    q_fp8[0, 3, :32]  = fp8(r0 / 0.003906)
    q_fp8[0, 3, 32:64] = fp8(r1 / 0.003906)
    q_fp8[0, 3, 64:]  = fp8(q_nope / 0.003906)

    weights_out[0, 3] = weights[0, 3] × 0.003906 × (1/√128) × (1/√64)
                      = 0.028 × 0.003906 × 0.08839 × 0.125
                      ≈ 1.21e-6
```

### 11.5 Prefill: gather + logits + top-K

```
# Request A has 6 prefill tokens at positions 0..5.
# cu_seqlen_ke = [1, 2, 3, 4, 5, 6]  (token at pos p sees p+1 K tokens)

cp_gather_indexer_k_quant_cache → k_quant [6, 128] fp8, k_scale [6, 4] uint8
  (6 indexed tokens from req A's k_cache at slots 200..205)

fp8_fp4_mqa_logits(q_fp8[1:7], k_quant, weights_out[1:7],
                   cu_seqlen_ks=[0,0,0,0,0,0], cu_seqlen_ke=[1,2,3,4,5,6])
→ logits [6, 6]  (lower-triangular valid)

Example logit values (illustrative):
  row 0 (pos=0): [0.73, —, —, —, —, —]         (sees only itself)
  row 1 (pos=1): [0.64, 0.81, —, —, —, —]       (sees pos 0 and 1)
  row 2 (pos=2): [0.42, 0.55, 0.78, —, —, —]
  row 3 (pos=3): [0.38, 0.62, 0.71, 0.90, —, —]
  row 4 (pos=4): [0.44, 0.57, 0.66, 0.83, 0.72, —]
  row 5 (pos=5): [0.41, 0.78, 0.55, 0.90, 0.63, 0.82]

top_k_per_row_prefill(logits, ..., K=1024):
  6 << 1024, so all valid entries selected:
  topk_indices_buffer[1, :1] = [0]            (pos 0: only itself)
  topk_indices_buffer[2, :2] = [1, 0]         (pos 1: best to worst)
  topk_indices_buffer[3, :3] = [2, 1, 0]
  topk_indices_buffer[4, :4] = [3, 4, 2, 1, 0]  ← wait, pos 3 ranks highest
  Actually for row 3 (pos=3): scores [0.38, 0.62, 0.71, 0.90]
    sorted descending: [3, 2, 1, 0]
  topk_indices_buffer[4, :4] = [3, 2, 1, 0]
  topk_indices_buffer[5, :5] = [3, 5, 1, 4, 2, 0]  (sorted by score)
  topk_indices_buffer[6, :6] = [3, 5, 1, 4, 2, 0]  (sorted by score for pos 5)
  (Note: all -1 after the valid count)
```

### 11.6 Decode: paged logits + top-K

```
# Request B: 3000 historical tokens, positions 0..2999
seq_lens = [[3000]]
block_table_B: ceil(3000/64) = 47 blocks covering req B's k_cache

fp8_fp4_paged_mqa_logits(q_fp8[0:1].reshape(1,1,64,128), indexer_k_cache,
                          weights_out[0:1], seq_lens=[[3000]], block_table_B, schedule_metadata)
→ logits [1, max_model_len]  (valid: [0..2999])

# Example: highest-scoring positions in req B's history
logits[0, 847]  = 0.94   (token 847 is highly relevant to the decode query)
logits[0, 1203] = 0.91
logits[0, 2714] = 0.89
...

cooperative_topk(logits, seq_lens=[[3000]], K=1024):
→ topk_indices_buffer[0, :1024] = [847, 1203, 2714, 388, ..., 2999]
   (sorted by score, padded with -1 if seq_len < 1024)
```

### 11.7 Main MLA attention: prefill

```
# W_UK_T absorption:
q_b_proj(q_c) → q [7, 128, 192]
split: q_nope [7,128,128], q_pe [7,128,64]
NeoX RoPE on q_pe
ql_nope = bmm(q_nope.T(dim0,1), W_UK_T).T  → [7, 128, 512]
concat_mla_q(ql_nope, q_pe, q_out) → q_out [7, 128, 576]

# FlashMLA sparse prefill for req A (tokens 1..6):
# First: gather selected K from main MLA cache into BF16 workspace
# topk_indices_buffer[1, :1] = [0] → gather kv_c at slot 200 (position 0 in req A)
# topk_indices_buffer[6, :6] = [3, 5, 1, 4, 2, 0] → gather 6 slots

cp_gather_and_upconvert_fp8_kv_cache(workspace [6, 1, 576], k_cache, local_indices, ...)
# workspace[0, 0, :] = BF16 dequant of kv_c at slot 200 (position 0)
# ...

flash_mla_sparse_fwd(
    q = q_out[1:7],           # [6, 128, 576]
    kv = workspace,            # [6, 1, 576]  bf16
    indices = local_indices,   # workspace-local: [[0],  [1,0], [2,1,0], ...]
    sm_scale = 1/√192,
    topk_length = [1,2,3,4,5,6],
    out = latent_out[1:7],    # [6, 128, 576]
)

# Latent-space output → V space via W_UV:
attn_out[1:7] = bmm(latent_out[1:7].T(0,1), W_UV).T  # [6, 128, 128]
```

### 11.8 Main MLA attention: decode

```
# Decode token for req B: topk_indices_buffer[0, :1024] = [847, 1203, 2714, ...]
# These are DIRECT physical slot indices (not compressed positions, unlike V4)

tile_scheduler_metadata = get_mla_metadata(cache_seqlens=[3000], ...)

flash_mla_with_kvcache(
    q = q_out[0:1].unsqueeze(1),         # [1, 1, 128, 576]
    k_cache = main_mla_kv_cache,         # [num_blocks, 64, 1, 576]  paged FP8 cache
    indices = topk_indices_buffer[0:1],  # [1, 1, 1024]  global physical slot IDs
    topk_length = [[1024]],              # all 1024 entries valid
    softmax_scale = 1/√192,
    is_fp8_kvcache = True,
    out = latent_out[0:1],               # [1, 1, 128, 576]
)

# W_UV application:
attn_out[0] = bmm(latent_out[0].T(0), W_UV).T  # [1, 128, 128]
```

**Attention computation for one head (head h=0), decode:**

```
For each of the 1024 selected tokens k (e.g., k=847):
  Read from paged cache:
    block = slot_847 // 64
    offset = slot_847 % 64
    kv_data = main_mla_kv_cache[block, offset, 0, :]   [576] uint8 (FP8)
    kv_c_k = dequantize(kv_data[:512])                 [512] bf16 (latent)
    k_pe_k = kv_data[512:].view(bf16)                  [64] bf16

  Score:
    s_k = dot(ql_nope[0, h, :], kv_c_k)   # 512-dim, absorbed W_UK
           + dot(q_pe_rotated[0, h, :], k_pe_k)  # 64-dim
    s_k *= (1/√192)

Softmax over 1024 scores → α [1024]

Latent output for head h:
  latent_out[0, h, :] = Σ_{k=0}^{1023}  α[k] × kv_c[selected_k, :]   [512]

# V-space projection:
attn_out[0, h, :] = W_UV[h, :, :].T @ latent_out[0, h, :]   [128]
```

### 11.9 Output projection

```
attn_out [7, 128, 128]
  reshape to [7, 16384]
  o_proj: RowParallel [16384 → 7168]
→ output [7, 7168]
```

No inverse RoPE, no grouped BMM — just one matmul.

### 11.10 Stage 9: V3 Output Projection (W_UV BMM detail)

```
Stage 9: Output Projection for V3 Decode

attn_out [1, 128, 512]  ← latent-space attention output (from FlashMLA)

# W_UV application:
# W_UV [128, 512, 128]  = [H, kv_lora_rank, v_head_dim]
attn_out_v = torch.bmm(
    attn_out.transpose(0,1),   # [128, 1, 512]  — batch over 128 heads
    W_UV,                       # [128, 512, 128]
).transpose(0,1)               # → [1, 128, 128]

# Why no inverse RoPE here (V3 differs from V4):
# V3 stores kv_c (512-dim) in the paged cache — this latent has no RoPE applied to it.
# The attention output (ql_nope × kv_c) is in the unrotated latent frame.
# W_UV operates directly in this frame. No rotation to undo.
# (V4 applies full RoPE to the 512-dim Q head, hence needs inverse RoPE before wo_a)

attn_out_v.reshape(1, 128*128=16384)
→ o_proj [16384 → 7168]  (RowParallelLinear)
→ output [1, 7168]
```

---

## 12. Additional V3 Worked Examples

### 12.1 Weight absorption BMM — concrete shapes with TP=1

```
V3 config: H=128, kv_lora_rank=512, qk_nope_head_dim=128, v_head_dim=128
TP=1 → num_local_heads=128

W_UK_T [128, 128, 512]:
  = kv_b_proj.weight reshaped and permuted
  W_UK_T[h, p, l] = weight for head h, nope dim p, latent dim l

BMM1 (q_nope absorption):
  Input:  q_nope [1, 128, 128]  (1 decode token, 128 heads, 128 nope dims)
  Weight: W_UK_T [128, 128, 512]
  
  torch.bmm(
      q_nope.transpose(0,1),  # [128, 1, 128]
      W_UK_T,                  # [128, 128, 512]
  ).transpose(0,1)            # → [1, 128, 512]
  
  This is 128 independent matmuls: each (1×128) × (128×512) → (1×512)
  = 128 × 1 × 128 × 512 × 2 = 16.8M FLOPs
  
  cuBLAS batched GEMM handles this efficiently as a single grouped call.

Result ql_nope [1, 128, 512]: 
  Each of 128 heads now carries a 512-dim latent query.
  No KV expansion needed — directly dot-producted against cached kv_c [N, 512].
```

### 12.2 V3 prefill — causal mask in the indexer

```
Request A: fresh 6-token prompt (positions 0-5)
cu_seqlen_ke = [1, 2, 3, 4, 5, 6]  (V3: ke[i] = positions[i] + 1 directly)

Meaning for V3 indexer:
  Token at pos=0: ke=1 → sees indexer K tokens [0, 0) = just pos 0 itself
  Token at pos=2: ke=3 → sees [0, 3) = positions 0, 1, 2
  Token at pos=5: ke=6 → sees [0, 6) = all 6 positions

Compare to V4 C4A (compress_ratio=4):
  Token at pos=2: ke = (2+1)//4 = 0 → sees NO compressed tokens (no complete group yet)
  Token at pos=3: ke = (3+1)//4 = 1 → sees 1 compressed token (covering pos 0-3)
  Token at pos=5: ke = (5+1)//4 = 1 → still only 1 compressed token

Key difference: V3 indexer scores EVERY individual token (direct positions);
V4 indexer scores COMPRESSED tokens (groups of 4 or 128).
V3 is more precise but requires more indexer cache entries (one per original token).
```

---

## 13. Key Implementation Files

| File | Role |
|---|---|
| [vllm/model_executor/models/deepseek_v2.py](vllm/model_executor/models/deepseek_v2.py) | `DeepseekV2Attention`, `DeepseekV32Attention`, `Indexer`, `fused_qkv_a_proj`, weight absorption (`W_UK_T`, `W_UV`), `concat_mla_q` |
| [vllm/models/deepseek_v32/nvidia/attention.py](vllm/models/deepseek_v32/nvidia/attention.py) | Optimized V3.2 CUDA attention; `DeepseekV32Indexer`; fused norm+rope kernels; FP8/BF16 dispatch |
| [vllm/v1/attention/backends/mla/flashmla_sparse.py](vllm/v1/attention/backends/mla/flashmla_sparse.py) | V3.2 `FlashMLASparseImpl.forward_mqa`; `_fp8_flash_mla_kernel`; `_bf16_flash_mla_kernel`; mixed batch dispatch |
| [vllm/model_executor/layers/sparse_attn_indexer.py](vllm/model_executor/layers/sparse_attn_indexer.py) | `SparseAttnIndexer`; V3 `fused_indexer_q_rope_quant` (NeoX); `sparse_attn_indexer` function; DCP merge |
| [vllm/v1/attention/backends/mla/indexer.py](vllm/v1/attention/backends/mla/indexer.py) | `DeepseekV32IndexerMetadataBuilder`; chunking policy; `_build_prefill_chunk_metadata_kernel` |
| [vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py](vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py) | DCP top-K merge; `pack_dcp_topk_candidates_cutedsl`; `stable_topk_from_gathered_candidates_cutedsl` |
| [vllm/models/deepseek_v32/amd/model.py](vllm/models/deepseek_v32/amd/model.py) | V3 AMD model; ROCm MoE dispatch |
| [vllm/models/deepseek_v32/amd/rocm.py](vllm/models/deepseek_v32/amd/rocm.py) | V3 ROCm attention backend |
| [vllm/v1/attention/ops/rocm_aiter_mla_sparse.py](vllm/v1/attention/ops/rocm_aiter_mla_sparse.py) | ROCm Triton kernels for V3 indexer operations (shared with V4) |
