# DeepSeek V3 and V4 Sparse MLA: In-Depth Analysis

## Table of Contents

1. [Background: MLA and the Sparse Indexer](#1-background-mla-and-the-sparse-indexer)
2. [DeepSeek V3 — Full Analysis](#2-deepseek-v3--full-analysis)
   - [Architecture: MLA + Indexer Modules](#21-architecture-mla--indexer-modules)
   - [MLA Math: Q/K/V Projections](#22-mla-math-qkv-projections)
   - [Indexer: Scoring K Cache](#23-indexer-scoring-k-cache)
   - [Scale Absorption](#24-scale-absorption)
   - [Metadata Build](#25-metadata-build)
   - [Full Forward Pass End-to-End](#26-full-forward-pass-end-to-end)
3. [DeepSeek V4 — Full Analysis](#3-deepseek-v4--full-analysis)
   - [Architecture Changes](#31-architecture-changes)
   - [MLA Math: Larger Head, New Output Projection](#32-mla-math-larger-head-new-output-projection)
   - [The Compressor](#33-the-compressor)
   - [Indexer Module](#34-indexer-module)
   - [Stream Parallelism](#35-stream-parallelism)
   - [Main MLA Attention: SWA + Sparse](#36-main-mla-attention-swa--sparse)
   - [MXFP4 Q Path](#37-mxfp4-q-path)
4. [Side-by-Side Comparison](#4-side-by-side-comparison)
5. [Worked Examples with Concrete Numbers](#5-worked-examples-with-concrete-numbers)
   - [V3 Example: Prefill + Decode](#51-v3-example-prefill--decode)
   - [V4 Example: Prefill + Decode](#52-v4-example-prefill--decode)
6. [Key Implementation Files](#6-key-implementation-files)

---

## 1. Background: MLA and the Sparse Indexer

### What is MLA?

Standard multi-head attention (MHA) caches `num_heads` separate K/V vectors per token. At 128 heads with 128-dim heads, that is 128 × 128 × 2 = 32,768 elements (64KB in BF16) per token. At 128K tokens, the KV cache per layer is 8GB — completely impractical.

**Multi-head Latent Attention (MLA)** compresses the KV cache by projecting the hidden state into a small shared latent vector, then projecting that latent back up to full K/V at attention time:

```
hidden_states [T, hidden_size]
     │
     ▼  kv_a_proj  [hidden_size → kv_lora_rank + rope_dim]
     │
kv_lora_rank-dim latent  +  rope_dim-dim k_pe
     │
     ▼  stored in KV cache (one vector per token, regardless of num_heads)
     │
at attention time:
     ▼  kv_b_proj  [kv_lora_rank → num_heads × (nope_dim + v_dim)]
     │
per-head K_nope, V  (computed on the fly from the cached latent)
```

In DSv3, `kv_lora_rank = 512` so the cache stores 512 + 64 = 576 elements per token instead of 32,768 — a 57× reduction. The rope part `k_pe` (64 dims) is stored alongside to avoid recomputation.

### Why a Sparse Indexer?

Even with MLA, attending over 128K tokens still requires `kv_b_proj` (a 512 → num_heads × nope_dim GEMM) for every cached token. At decode time with 128 heads × 128 nope + 128 rope = 128-dim head and 128K tokens, the effective attention GEMM is enormous.

The **sparse indexer** preselects the top-K most relevant cached tokens using a cheap dedicated scoring path:

1. **Scoring pass**: A small 64-head, 128-dim FP8 MQA attention scores every cached token using a compressed representation. This is ~50× cheaper than the full MLA.
2. **Sparse MLA pass**: Full MLA attention runs over only the top-K scored tokens.

At 128K context with K=1024, this reduces full-precision attention to under 1% of tokens.

---

## 2. DeepSeek V3 — Full Analysis

### 2.1 Architecture: MLA + Indexer Modules

```
Model config (DeepSeek V3 / V3.2):
  hidden_size         = 7168
  num_attention_heads = 128   (full MLA attention heads)
  q_lora_rank         = 1536  (Q low-rank bottleneck)
  kv_lora_rank        = 512   (KV latent dimension = main MLA cache dim)
  qk_nope_head_dim    = 128   (nope part of each Q/K head)
  qk_rope_head_dim    = 64    (rope part of each Q/K head)
  v_head_dim          = 128
  qk_head_dim         = 192   (= nope + rope)

  index_n_heads       = 64    (indexer: scoring heads)
  index_head_dim      = 128   (indexer: 64 rope + 64 nope)
  index_topk          = 1024  (K: tokens selected)
  kv_cache block_size = 64    (indexer cache)
  max_model_len       = 128000
```

There are **two entirely separate attention modules** per layer:

```
Per layer:
  ┌─────────────────────────────────────────────────────────┐
  │  DeepseekV32Attention  (main MLA attention)             │
  │    fused_qkv_a_proj  [7168 → 1536 + 512 + 64]          │
  │    q_a_layernorm     [1536]                             │
  │    q_b_proj          [1536 → 128×192]                  │
  │    kv_a_layernorm    [512]                              │
  │    kv_b_proj         [512 → 128×(128+128)]             │
  │    o_proj            [128×128 → 7168]                   │
  │    attn (FlashMLA sparse)                               │
  │    ┌─────────────────────────────────────────────────┐  │
  │    │  DeepseekV32Indexer  (sparse scoring module)   │  │
  │    │    wq_b           [1536 → 64×128]              │  │
  │    │    wk_weights_proj [7168 → 128+64]             │  │
  │    │    k_norm         [128]                        │  │
  │    │    k_cache        (paged FP8, block_size=64)   │  │
  │    │    indexer_op     SparseAttnIndexer            │  │
  │    └─────────────────────────────────────────────────┘  │
  │    KV cache: paged [kv_lora_rank+rope=576 dims/token]   │
  └─────────────────────────────────────────────────────────┘
```

### 2.2 MLA Math: Q/K/V Projections

#### Q path

```
hidden_states [T, 7168]
     │
     ▼  fused_qkv_a_proj  (single GEMM, fused with KV)
     │
q_c [T, 1536]  ← Q low-rank activations
     │
     ▼  q_a_layernorm (RMSNorm, stabilizes the LoRA bottleneck)
     │
     ▼  q_b_proj  [1536 → 128×192]
     │
q [T, 128, 192]  split into:
  q_nope [T, 128, 128]   ← nope part: will be absorbed into latent Q
  q_pe   [T, 128, 64]    ← rope part: gets RoPE applied
```

**Key MLA insight for Q**: `q_nope` is not used directly. Instead it is absorbed into the latent query:

```
ql_nope [T, 128, 512] = q_nope [T, 128, 128]  ×  W_UK_T [128, 128, 512]
```

where `W_UK_T` is the pre-transposed nope slice of `kv_b_proj.weight`. This absorbs the `kv_b_proj` matrix into the query side, so at attention time only the latent `kv_c` (512-dim) needs to be read from cache — no `kv_b_proj` per cached token.

#### KV path

```
fused_qkv_a_proj output  →  split:
  kv_c [T, 512]   ← KV latent (normalized by kv_a_layernorm)
  k_pe [T, 64]    ← shared rope K (one vector for all heads, MQA style)
```

`kv_c` is **written directly to the main MLA KV cache** (the 576-dim paged cache). At attention time:

```
# For each token t already in cache:
kv [num_heads, 256] = kv_b_proj(kv_c[t])   ← on the fly per cached token

k_nope [num_heads, 128]  =  kv[:, :128]
v      [num_heads, 128]  =  kv[:, 128:]

k [num_heads, 192] = cat(k_nope, k_pe_rotated)
```

But with weight absorption, the actual FlashMLA implementation avoids materializing `k_nope` per head. Instead the full attention score is:

```
score[q_head, t] = dot(ql_nope[q_head, :], kv_c[t, :])   # 512-dim dot product
                 + dot(q_pe_rotated[q_head, :], k_pe_rotated[t, :])  # 64-dim dot product
```

The first term combines what would have been `q_nope @ W_UK @ kv_c` into a single 512-dim dot, avoiding the intermediate 128-dim K reconstruction.

#### Output projection

```
attn_out [T, 128, 128]
     │
     ▼  absorb W_UV:  out = attn_out × W_UV  [128, 128, 512]  (bmm)
     │     (W_UV = transposed v-slice of kv_b_proj.weight, pre-computed)
     │
     ▼  o_proj  RowParallelLinear  [128×128 → 7168]
```

The V matrix is similarly never materialized per cached token — instead `W_UV` is pre-multiplied into the attention output after softmax. The `W_UV` bmm maps the latent-space attention output to the full V space.

### 2.3 Indexer: Scoring K Cache

The indexer runs a **separate, cheap scoring attention** whose sole purpose is to produce `topk_indices_buffer`. It has its own 128-dim K head, its own paged FP8 KV cache, and its own Q projection. It does **not** share weights with the main MLA.

#### Indexer Q

```
q_c [T, 1536]  (same q_c from the main MLA fused GEMM)
     │
     ▼  wq_b  [1536 → 64×128]
     │
q_idx [T, 64, 128]  ← 64-head, 128-dim (64 nope + 64 rope)
     │
     ▼  fused_indexer_q_rope_quant  (Triton kernel, grid (T, 64))
        - GPT-J or NeoX RoPE on rope dims
        - FP8 quantize per-token per-head (e8m0 power-of-2 scale)
        - absorb q_scale × softmax_scale × head_scale into weights_out
     │
q_fp8 [T, 64, 128]  fp8
weights_out [T, 64]  fp32
```

#### Indexer K

```
hidden_states [T, 7168]
     │
     ▼  wk_weights_proj  MergedColumnParallelLinear  [7168 → 128+64]  (single GEMM)
     │
k [T, 128] bf16   +   weights [T, 64] bf16
     │
     ▼  k_norm  (LayerNorm on 128-dim K)
     │
split:  k_pe [T, 64],  k_nope [T, 64]
     │
     ▼  RoPE on k_pe (using indexer's rope embeddings, separate from main MLA rope)
     │
k [T, 128] = cat(k_pe_rotated, k_nope)
     │
     ▼  indexer_k_quant_and_cache:  quantize to FP8 + write to paged indexer k_cache
```

The `weights [T, 64]` scalar per (token, head) is a **learned per-head pre-score**: it encodes how much the K side thinks head `h` should attend to this token, derived purely from `hidden_states` without seeing the query.

#### Scoring

```
# Prefill (gather → GEMM → top-K):
cp_gather_indexer_k_quant_cache → k_quant [N, 128] fp8 (flat workspace)
fp8_fp4_mqa_logits(q_fp8, k_quant, weights_out, cu_seqlen_ks, cu_seqlen_ke)
  → logits [M, N] fp32  (causally masked)
top_k_per_row_prefill → topk_indices_buffer [M, 1024] int32

# Decode (paged traversal → top-K):
fp8_fp4_paged_mqa_logits(q_fp8, kv_cache, weights_out, seq_lens, block_table)
  → logits [B, max_model_len] fp32
cooperative_topk / persistent_topk / top_k_per_row_decode
  → topk_indices_buffer [B, 1024] int32
```

The MQA logit for each (query token `q`, KV token `k`) is:

```
logit[q, k] = Σ_h  dot(q_fp8[q,h,:], k_fp8[k,:])  ×  sk[k]  ×  weights_out[q,h]
```

where `weights_out[q,h] = weights[q,h] × sq[q,h] × (1/√128) × (1/√64)` (scale absorption, see §2.4).

After top-K, `topk_indices_buffer[q, :]` holds up to 1024 token positions with the highest scores for query token `q`. Padding slots are `-1`.

### 2.4 Scale Absorption

The true scoring logit is:

```
logit[q, k] = Σ_h  dot(q_bf16[q,h,:], k_bf16[k,:])  ×  w[q,h]  ×  (1/√128)  ×  (1/√64)
```

After FP8 quantization with per-token per-head scale `sq[q,h]` for Q and per-block scale `sk[k]` for K:

```
≈ Σ_h  dot(q_fp8[q,h,:] × sq[q,h],  k_fp8[k,:] × sk[k])  ×  w[q,h]  ×  (1/√128)  ×  (1/√64)
= Σ_h  dot(q_fp8[q,h,:], k_fp8[k,:])  ×  sk[k]  ×  [sq[q,h] × w[q,h] × (1/√128) × (1/√64)]
                                                       ╰──────────────────────────────────────╯
                                                                   weights_out[q, h]
```

The Triton kernel pre-computes this scalar once per `(token, head)`:

```python
# e8m0 scale (power-of-2, exact fp division):
scale_raw = max(amax, 1e-10) / fp8_max
q_scale   = 2 ^ ceil(log2(scale_raw))

# Absorbed into weights:
weights_out[t, h] = weights[t, h] × q_scale × (1/√head_dim) × (1/√n_heads)
```

The MQA logit GEMM then folds `weights_out[q,h]` into its epilogue — one scalar multiply per tile accumulation, no separate rescaling pass. K scales `sk` are handled inside DeepGEMM's FP8 dequantization.

### 2.5 Metadata Build

`DeepseekV32IndexerMetadataBuilder.build()` (CPU, each step):

**Decode metadata:**

- `seq_lens [B, 1]`: context length per sequence (shape `[B, next_n]` for spec decode)
- `block_table [B, max_blocks]`: physical page indices covering each request's KV history
- `decode_lens [B]`: decode tokens per request
- `schedule_metadata`: DeepGEMM tile scheduler — partitions the paged KV scan across GPU SMs

**Prefill metadata** (per chunk):

- `cu_seqlen_ks [M]`, `cu_seqlen_ke [M]`: for each query token, the `[start, end)` range in the flat gathered K buffer it may attend to — implements causal masking
- `local_cu_seq_lens [num_reqs+1]`: cumulative K lengths across requests in this chunk
- `block_table [num_reqs, max_blocks]`: KV cache pages for the prefill requests
- `token_start`, `token_end`: slice into the batch token array

Chunking splits prefills to respect: (1) workspace budget `total_seq_lens ≤ 40 × max_model_len`, and (2) logit memory `M × N × 4 ≤ MAX_LOGITS_MB × 1024²`.

### 2.6 Full Forward Pass End-to-End

```
hidden_states [T, 7168]
      │
      ├──────────────────────────────────────────────────────────────────────────
      │  FUSED INPUT GEMM (shared between MLA + indexer Q path)
      │
      ▼  fused_qkv_a_proj  [7168 → 1536 + 512 + 64]  (single GEMM)
      │
      ├─ q_c   [T, 1536]   → q_a_layernorm → used by BOTH MLA q_b_proj and indexer wq_b
      ├─ kv_c  [T, 512]    → kv_a_layernorm → stored in main MLA KV cache
      └─ k_pe  [T, 64]     → RoPE → stored alongside kv_c in main MLA KV cache
      │
      ├──────────────────────────────────────────────────────────────────────────
      │  INDEXER PATH (runs concurrently on CUDA path)
      │
      ├─ wk_weights_proj(hidden_states) → k [T,128] + weights [T,64]
      │    k_norm(k) → split → RoPE(k_pe) → cat → k [T,128]
      │    indexer_k_quant_and_cache(k → paged FP8 indexer k_cache)
      │
      ├─ wq_b(q_c) → q_idx [T,64,128]
      │    fused_indexer_q_rope_quant → q_fp8 [T,64,128] + weights_out [T,64]
      │
      │  [prefill: gather k_cache → flat buffer; then MQA GEMM → top-K]
      │  [decode:  paged MQA GEMM → top-K]
      │
      ▼  topk_indices_buffer [T, 1024]  int32
      │
      ├──────────────────────────────────────────────────────────────────────────
      │  MAIN MLA PATH
      │
      ├─ q_b_proj(q_c) → q [T, 128, 192]
      │    split: q_nope [T,128,128], q_pe [T,128,64]
      │    RoPE(q_pe)
      │    ql_nope = bmm(q_nope, W_UK_T)  → [T, 128, 512]  (W_UK_T absorbed)
      │
      │  FlashMLA sparse:
      │    input:    (ql_nope, q_pe_rotated)  +  kv_c_cache (512-dim latent per token)
      │    sparse:   only read kv_c[topk_indices_buffer[t, :]] for each query t
      │    score:    dot(ql_nope[t,h,:], kv_c[k,:]) + dot(q_pe[t,h,:], k_pe[k,:])
      │    output:   attn_out [T, 128, 512]  (in latent space)
      │
      ▼  bmm(attn_out, W_UV) → [T, 128, 128]  (back to v-space)
      ▼  o_proj [128×128 → 7168]
      ▼  output [T, 7168]
```

---

## 3. DeepSeek V4 — Full Analysis

### 3.1 Architecture Changes

V4 introduces five structural changes over V3:

| Change | Detail |
|---|---|
| Larger main MLA head | 512 dims (448 NoPE + 64 RoPE) vs V3's 128-dim; 584B/token FP8 cache |
| Per-layer compress ratio | C4A (ratio=4), C128A (ratio=128), or SWA-only per layer |
| The Compressor | Accumulates `compress_ratio` input tokens → 1 compressed K for the indexer |
| SWA cache on every layer | 4096-token sliding window in full 512-dim FP8, attended unconditionally |
| New output projection | `wo_a` (grouped BMM bottleneck) + `wo_b` (RowParallel), with fused inverse RoPE |

### 3.2 MLA Math: Larger Head, New Output Projection

```
Model config (DeepSeek V4, C4A layer):
  hidden_size         = 7168
  num_attention_heads = 128   (full MLA heads, TP-split to n_local_heads)
  q_lora_rank         = 1536
  head_dim            = 512   (448 NoPE + 64 RoPE)
  nope_head_dim       = 448
  rope_head_dim       = 64
  o_lora_rank         = (config value — output bottleneck dim)
  o_groups            = (config value — number of output projection groups)
  compress_ratio      = 4
  sliding_window      = 4096
```

#### Q path

```
hidden_states [T, 7168]
     │
     ▼  fused_wqa_wkv  MergedColumnParallelLinear  [7168 → 1536 + 512]
     │
     ├─ qr  [T, 1536]  → q_norm (RMSNorm) → normalized Q low-rank activations
     └─ kv  [T, 512]   → kv_norm (RMSNorm) → normalized KV latent
     │
     ▼  wq_b  ColumnParallelLinear  [1536 → num_heads × 512]
     │
q [T, num_heads, 512]   ← full 512-dim per head
     │
     ▼  per-head RMSNorm (no weight, just unit-normalization)
     ▼  GPT-J RoPE on last 64 dims (rope_head_dim)
     │
q [T, num_heads, 512]   ready for attention
```

Unlike V3, V4 does **not** split `q_nope`/`q_pe` at the model level — the full 512-dim head goes into FlashMLA which handles the rope/nope decomposition internally.

Also unlike V3, there is **no weight absorption** for `q_nope`. The main MLA KV cache in V4 stores the full 512-dim latent (448 NoPE FP8 + 64 RoPE BF16 + 8 scale bytes = 584B/token), and FlashMLA reads the full format directly.

#### KV path

```
kv [T, 512]   (from fused_wqa_wkv, after kv_norm)
     │
     ▼  GPT-J RoPE on last 64 dims
     ▼  FP8 UE8M0 block quantize (448 NoPE dims → block-scaled FP8)
     ▼  write to swa_kv_cache at swa_slot_mapping  (sliding-window 512-dim cache)
     │
     (for compress_ratio > 1 layers, also fed to DeepseekCompressor
      which writes to the full main MLA KV cache — see §3.3)
```

The SWA cache stores the recent 4096 tokens in full quality. The main MLA KV cache (written by the compressor) stores a compressed representation of the full history.

#### Output projection (new in V4)

V4 replaces the simple `o_proj` with a grouped two-stage LoRA with fused inverse RoPE:

```
attn_out [T, n_local_heads, 512]
     │
     ▼  fused_inv_rope_fp8_quant  (Triton/CuteDSL kernel)
        - Inverse GPT-J RoPE on the rope_head_dim dims of each head:
            x_even' = x_even×cos + x_odd×sin
            x_odd'  = x_even×cos - x_odd×sin   (note: this undoes the forward rotation)
        - Group heads into n_groups blocks of (n_local_heads / n_groups) heads each
        - FP8 block-quantize the grouped output (UE8M0 block scales)
     │
o_fp8 [T, n_groups, heads_per_group × 512]  fp8
o_scale [T, n_groups, n_scale_blocks]       fp32 or ue8m0
     │
     ▼  wo_a  (FP8 grouped BMM, one per group)
        fp8_einsum("bhr,hdr->bhd", o_fp8, wo_a.weight)
        Input:  [T, n_groups, heads_per_group × 512]  fp8
        Weight: [n_groups, heads_per_group × 512, o_lora_rank]  fp8
        Output: [T, n_groups, o_lora_rank]  bf16
     │
z [T, n_groups, o_lora_rank]
     │
     ▼  wo_b  RowParallelLinear  [n_groups × o_lora_rank → 7168]
     │
output [T, 7168]
```

The inverse RoPE is necessary because the forward RoPE was applied to the query but not stored — the cache stores the unrotated latent. Before projecting the attention output back to hidden space, the RoPE rotation applied to the heads must be undone so that the output projection works in a consistent basis.

### 3.3 The Compressor

`DeepseekCompressor` is the key V4 component that produces entries for both the main MLA KV cache and the indexer's scoring KV cache.

**Structure (C4A, compress_ratio=4, overlap=1, coff=2):**

```
DeepseekCompressor
  fused_wkv_wgate : MergedColumnParallelLinear  [7168 → 2×coff×head_dim]
                    = [7168 → 4×128=512]  for indexer compressor
                    = [7168 → 4×512=2048]  for main MLA compressor
  ape             : nn.Parameter  [compress_ratio, coff×head_dim]
                    = [4, 256]  (learned per-position-within-group embeddings)
  state_cache     : CompressorStateCache  (paged fp32, sliding window of 8 states)
  norm            : RMSNorm  [head_dim]
```

**Forward: two sub-steps per token `t`**

**Sub-step 1: `save_partial_states`** (fires for every token)

```
kv_score [T, 2×coff×head_dim]  from fused_wkv_wgate(hidden_states)
     │
split: kv [T, coff×head_dim],  score [T, coff×head_dim]
     │
# Add learned absolute position embedding for this token's position within its group:
kv    += ape[t % compress_ratio, :coff×head_dim]
score += ape[t % compress_ratio, coff×head_dim:]
     │
write (kv, score) to state_cache at slot_mapping[t]
  (paged fp32 sliding-window, stores last coff×compress_ratio=8 raw states)
```

**Sub-step 2: `compress_norm_rope_store`** (fires only when `t % compress_ratio == compress_ratio - 1`)

```
# Read the last 8 state entries (coff×compress_ratio = 2×4) from state_cache:
states = gather(state_cache, positions=[t-7..t])
  → kv_states    [8, head_dim]
  → score_states [8, head_dim]

# Gated weighted sum: compress 8 states → 1 compressed token
# (For C4A with overlap: two groups of 4, each gated separately)
compressed_kv [head_dim] = gated_sum(kv_states[:4], gate=softmax(score_states[:4]))
                         + gated_sum(kv_states[4:], gate=softmax(score_states[4:]))

# Normalize + RoPE:
compressed_kv = RMSNorm(compressed_kv, norm.weight)
apply GPT-J RoPE at position = (t // compress_ratio) × compress_ratio
  (on the last rope_head_dim dims)

# Quantize and write to the KV cache:
k_fp8 = FP8_quantize(compressed_kv)   (or MXFP4 on Blackwell)
write k_fp8 to indexer k_cache at compressed slot t // compress_ratio
  (or to main MLA KV cache for the main compressor)
```

One compressed K token appears in the indexer cache every 4 input tokens. At `seq_len=4000`, the indexer cache holds 1000 compressed entries.

### 3.4 Indexer Module

`DeepseekV4Indexer` differs from V3's indexer in two key ways:

- **`weights_proj` is separate from K** (V3 fused them in one GEMM)
- **K comes from the compressor**, not a direct projection (`skip_k_cache_insert=True`)

```
DeepseekV4Indexer
  ├── wq_b        : ReplicatedLinear  [q_lora_rank=1536 → 64×128=8192]
  ├── weights_proj: ReplicatedLinear  [hidden_size=7168 → n_head=64]
  │                 (separate GEMM — runs on aux stream, overlapped with main GEMMs)
  ├── compressor  : DeepseekCompressor  (writes compressed K to indexer k_cache)
  ├── k_cache     : DeepseekV4IndexerCache  (paged FP8/MXFP4, block_size=256)
  └── indexer_op  : SparseAttnIndexer(skip_k_cache_insert=True)
```

The indexer forward:

```python
def forward(hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb):

    # Two-way inner overlap:
    # - default sub-stream: wq_b + FP8/MXFP4 quant
    # - aux sub-stream: compressor (writes K to indexer k_cache)
    def wq_b_and_q_quant():
        q, _ = wq_b(qr)
        q = q.view(-1, 64, 128)
        return fused_indexer_q_rope_quant(
            positions, q, rotary_emb.cos_sin_cache,
            indexer_weights, 1/√128, 1/√64, use_fp4=use_fp4_kv)

    (q_quant, weights_out), _ = maybe_execute_in_parallel(
        wq_b_and_q_quant,
        lambda: compressor(compressed_kv_score, positions, rotary_emb),
        ...)

    # K is already in indexer k_cache (written by compressor)
    return indexer_op(hidden_states, q_quant, k=None, weights=weights_out)
```

### 3.5 Stream Parallelism

V4 runs 4 GEMMs and 3 compute paths concurrently:

**Phase 1 — `attn_gemm_parallel_execute`** (before `attention_impl`):

```
Default stream:
  fused_wqa_wkv(hidden_states)             → qr [T,1536] + kv [T,512]

Aux stream 0:
  main_compressor.fused_wkv_wgate(hidden)  → main_kv_score [T, 2048]

Aux stream 1:
  indexer.weights_proj(hidden_states)      → indexer_weights [T, 64]

Aux stream 2:
  indexer.compressor.fused_wkv_wgate(hidden) → idx_kv_score [T, 512]
```

All 4 join → `fused_q_kv_rmsnorm(qr, kv)` → enter `attention_impl`.

**Phase 2 — `attention_impl`** (3-way overlap):

```
Default stream (waits on fan-out event):
  wq_b(qr) → q [T, n_heads, 512]
  _fused_qnorm_rope_kv_insert:
    per-head RMSNorm(q), RoPE(q), RoPE(kv), FP8_quant(kv), write→swa_kv_cache
  → q ready                                                [records join event A]

Aux stream 0 (waits on fan-out event):
  ─────── indexer inner overlap ────────────────────────────────
  wq_b(qr) + fused_indexer_q_rope_quant → q_fp8, weights_out   │
  ─────────────────────────────────────────────────────────     │ 2-way
  indexer.compressor(idx_kv_score):                             │ overlap
    save_partial_states → state_cache                           │ via sub-events
    compress_norm_rope_store → indexer k_cache                  │
  ───────────────────────────────────────────────────────────────
  indexer_op(q_fp8, weights_out) → topk_indices_buffer         [records join event B]

Aux stream 1 (waits on fan-out event):
  main_compressor(main_kv_score):
    save_partial_states → compressor state_cache
    compress_norm_rope_store → main MLA KV cache (584B/token FP8)
                                                               [records join event C]

Default stream waits on events B and C:
  forward_mqa(q, kv, topk_indices_buffer) → o [T, n_heads, 512]
  _o_proj(o, positions) → [T, 7168]
```

### 3.6 Main MLA Attention: SWA + Sparse

V4's `forward_mqa` in `DeepseekV4FlashMLAAttention` combines two attention sources:

#### Sliding-Window Attention (SWA)

All tokens within the last `window_size=4096` positions are attended unconditionally from `swa_kv_cache`. The SWA cache stores the full 512-dim FP8 latent for recent tokens.

```
swa_metadata: block_table for the SWA window
swa_kv_cache: [num_blocks, block_size, 512]  uint8 (FP8 UE8M0)

For each decode token t:
  swa positions = max(0, pos[t] - 4095) .. pos[t]
  → swa_indices [T, 1, swa_len]  (global physical slot ids)
```

#### Sparse Attention from Indexer (out-of-window)

```
topk_indices_buffer [T, 1024]  (compressed positions, -1 for padding)
     │
     ▼  compute_global_topk_indices_and_lens:
        For each compressed index ck:
          for block_offset in 0..(compress_ratio-1):
            global_slot = block_table[req, ck // block_size] × block_size
                         + (ck % block_size) × compress_ratio + block_offset
        → global_topk_indices [T, 1, 1024×compress_ratio]  (physical slot ids)
        → topk_lens [T]  (valid entry count per token)
```

#### Combined FlashMLA call (decode)

```python
flash_mla_with_kvcache(
    q = q.unsqueeze(1),                          # [T, 1, n_heads, 512]
    k_cache = swa_kv_cache.unsqueeze(-2),         # [num_blocks, block_size, 1, 512]

    # SWA tokens:
    indices = swa_indices,                        # [T, 1, swa_len]
    topk_length = swa_lens,

    # Sparse out-of-window tokens:
    extra_k_cache = main_mla_kv_cache,            # [num_blocks, block_size, 1, 584]
    extra_indices_in_kvcache = global_topk_indices, # [T, 1, 1024×compress_ratio]
    extra_topk_length = topk_lens,

    softmax_scale = 1/√512,
    attn_sink = self.attn_sink,                   # learned per-head sink bias
    is_fp8_kvcache = True,
    out = output.unsqueeze(1),
)
```

FlashMLA handles the combined SWA + sparse index set in a single kernel pass, deduplicating any overlap between the SWA window and the expanded compressed sparse indices.

#### Prefill attention

For prefill tokens the path gathers K from both the compressed paged cache and the SWA cache into a BF16 workspace, combines the `topk_indices` (which for prefill are local-position indices 0..pos//compress_ratio) with SWA indices, and calls:

```python
flash_mla_sparse_fwd(
    q = q[query_start:query_end],
    kv = kv.view(-1, 1, 512),                   # gathered combined KV workspace
    indices = combined_indices.unsqueeze(1),
    sm_scale = 1/√512,
    attn_sink = self.attn_sink,
    topk_length = combined_lens,
    out = output[query_start:query_end],
)
```

### 3.7 MXFP4 Q Path

On Blackwell B200 (`SM100`, `use_fp4_indexer_cache=True`), Q is quantized to MXFP4 (4-bit E2M1, 32 elements/block, ue8m0 block scale) instead of FP8.

```
# MXFP4 kernel output:
q_packed [T, 64, 64]   uint8  — 2 E2M1 nibbles packed per byte
q_scale  [T, 64,  4]   uint8  — 4 ue8m0 block scales per head (128/32=4 blocks)

# Weight absorption (MXFP4 path — q_scale NOT folded):
weights_out[t, h] = weights[t, h] × (1/√128) × (1/√64)
# The 4 per-block q_scales are passed separately to the logit kernel
# which applies them per 32-element tile during dequantization
```

Compare to FP8 where the single per-token scalar `sq[t,h]` is folded:

```
# FP8 weight absorption:
weights_out[t, h] = weights[t, h] × sq[t,h] × (1/√128) × (1/√64)
```

The `SparseAttnIndexer` receives `q_quant=(q_packed, q_scale_int32)` as a tuple and passes `q_scale` separately to `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`.

---

## 4. Side-by-Side Comparison

### MLA Architecture

| Aspect | DeepSeek V3 | DeepSeek V4 |
|---|---|---|
| Main KV cache dims | 512 (kv_lora_rank) + 64 (rope) = 576 BF16 | 448 FP8 (nope) + 64 BF16 (rope) + 8 scale = 584B |
| Q projection | `q_a_proj` + `q_a_layernorm` + `q_b_proj` | `fused_wqa_wkv` + `q_norm` + `wq_b` |
| KV projection | `kv_a_proj_with_mqa` + `kv_a_layernorm` + `kv_b_proj` | `fused_wqa_wkv` (kv slice) + `kv_norm` |
| Q head dim | 192 (128 nope + 64 rope) | **512** (448 nope + 64 rope) |
| Weight absorption (Q nope) | `ql_nope = q_nope × W_UK_T` (precomputed) | None — FlashMLA handles full 512-dim head |
| SWA cache | None | Full 512-dim FP8, 4096-token window, every layer |
| Output projection | `o_proj` (single `RowParallel`) | **Inverse RoPE** + `wo_a` (grouped FP8 BMM) + `wo_b` |
| Attention kernel | FlashMLA sparse (512-dim latent) | FlashMLA sparse (combined SWA + compressed sparse) |
| Per-layer type | All layers use same structure | C4A / C128A / SWA-only per layer |

### Indexer Architecture

| Aspect | DeepSeek V3 | DeepSeek V4 |
|---|---|---|
| K source | `wk_weights_proj(hidden)` → direct 128-dim K | `DeepseekCompressor` → compressed K (4 tokens → 1) |
| Weights source | Fused with K in `wk_weights_proj` | Separate `weights_proj(hidden)` GEMM |
| K cache insert | Inside `SparseAttnIndexer` | Inside compressor (`skip_k_cache_insert=True`) |
| Indexer head dim | 128 (64 nope + 64 rope) | 128 (same) |
| Indexer scoring heads | 64 | 64 (same) |
| Block size (indexer cache) | 64 | 256 |
| `topk_indices` meaning | Direct token positions `[0..seq_len)` | **Compressed positions** `[0..seq_len//compress_ratio)` |
| Q quantization | FP8 only | FP8 (default) or MXFP4 (Blackwell B200) |

### Scale Absorption

| Factor | V3 FP8 | V4 FP8 | V4 MXFP4 |
|---|---|---|---|
| Q scale granularity | Per-token per-head scalar | Per-token per-head scalar | Per-block (32 elements) per head |
| `q_scale` folded into `weights_out`? | Yes | Yes | **No** |
| `softmax_scale` (1/√128) folded? | Yes | Yes | Yes |
| `head_scale` (1/√64) folded? | Yes | Yes | Yes |
| `weights_out` formula | `w × sq × (1/√128) × (1/√64)` | Same | `w × (1/√128) × (1/√64)` |

### Stream Parallelism

| | V3 | V4 |
|---|---|---|
| Input GEMMs | Sequential | 4-way parallel (3 aux streams) |
| Indexer vs main MLA | Sequential | Overlapped (aux stream 0 vs default + aux 1) |
| Compressor vs attention | N/A | Overlapped (aux stream 1 vs default) |
| Indexer Q vs indexer K | Sequential | 2-way inner overlap |

---

## 5. Worked Examples with Concrete Numbers

### 5.1 V3 Example: Prefill + Decode

**Setup:**

```
Batch: 2 requests
  Request A: 6-token new prompt  (prefill, seq_lens=6, query_lens=6)
  Request B: continuing with 3000 KV tokens  (decode, seq_lens=3000, query_lens=1)
Token layout: decode first (index 0), prefill second (indices 1..6)
```

---

#### Stage 1: Fused Input GEMM

```
hidden_states [7, 7168]
     │
     ▼  fused_qkv_a_proj  [7168 → 1536+512+64]

q_c  [7, 1536]    ← Q low-rank activations
kv_c [7, 512]     ← KV latent (after kv_a_layernorm)
k_pe [7, 64]      ← shared rope K
```

---

#### Stage 2: Write Main MLA KV Cache

```
kv_c [7, 512] and k_pe [7, 64] (after RoPE) are packed as 576-dim entries
and written to the main MLA paged KV cache at slot_mapping positions:

  decode token 0 → slot 5000  (position 2999 in req B's history)
  prefill tokens 1..6 → slots 200..205  (positions 0..5 in req A)
```

---

#### Stage 3: Indexer — Q and K Projections

**K (from `wk_weights_proj`):**

```
wk_weights_proj(hidden_states) → kw [7, 192]
  k = kw[:, :128]        [7, 128]  bf16
  weights = kw[:, 128:]  [7,  64]  bf16

k_norm(k) → split → RoPE(k_pe):
  For decode token 0 at position=2999:
    k_pe[0, :] rotated by cos/sin(2999)
  k [7, 128] = cat(k_pe_rotated, k_nope)

indexer_k_quant_and_cache(k, kv_cache, slot_mapping):
  For each of 7 tokens:
    quantize k[t, :128] to FP8 with ue8m0 block scale (block_size=128)
    write to indexer k_cache at slot_mapping[t]
```

**Q (from `wq_b`):**

```
wq_b(q_c) → q_idx [7, 64, 128]   bf16

fused_indexer_q_rope_quant  (Triton grid (7, 64)):

For decode token 0, head 0 at position=2999:
  # NeoX RoPE on rope part (dims 0..63 in this case):
  x0 = q_idx[0,0,:32],  x1 = q_idx[0,0,32:64]
  r0 = x0×cos(2999) - x1×sin(2999)  → bf16 → fp32
  r1 = x1×cos(2999) + x0×sin(2999)  → bf16 → fp32

  q_nope = q_idx[0,0,64:]  (unchanged)

  amax = max(|r0|, |r1|, |q_nope|) = 0.94
  q_scale = 2^(ceil(log2(0.94/448))) = 2^(-8) = 0.00390625

  q_fp8[0, 0, :64] = clamp(r / 0.00390625, -448, 448)
  q_fp8[0, 0, 64:] = clamp(q_nope / 0.00390625, -448, 448)

  # Scale absorption:
  weights_out[0, 0] = weights[0, 0] × 0.00390625 × (1/√128) × (1/√64)
                    = 0.031 × 0.00390625 × 0.08839 × 0.125
                    ≈ 1.34e-6
```

---

#### Stage 4: Indexer Prefill — Gather + Logits + Top-K

`_build_prefill_chunk_metadata_kernel` for req A (seq_len=6, query_len=6, start_pos=0):

```
cu_seqlen_ks = [0, 0, 0, 0, 0, 0]   (all queries: K buffer row starts at 0)
cu_seqlen_ke = [1, 2, 3, 4, 5, 6]   (token i sees i+1 historical KVs)
```

Gather 6 K tokens from indexer k_cache:

```
cp_gather_indexer_k_quant_cache → k_quant [6, 128] fp8,  k_scale [6, 4] uint8
```

MQA logits (causally masked):

```
fp8_fp4_mqa_logits(q_fp8[1:7], k_quant, weights_out[1:7], cu_seqlen_ks, cu_seqlen_ke)
→ logits [6, 6] fp32

For query 0 at pos 0 (sees KV 0 only):
  logit[0, 0] = Σ_h  dot(q_fp8[0,h,:], k_fp8[0,:])  ×  sk[0]  ×  weights_out[0,h]

logits matrix (lower triangle valid):
     KV pos:  0      1      2      3      4      5
Query 0:   [0.72,  —,     —,     —,     —,     —  ]
Query 1:   [0.65, 0.81,   —,     —,     —,     —  ]
Query 2:   [0.41, 0.55,  0.78,   —,     —,     —  ]
Query 3:   [0.38, 0.62,  0.71,  0.90,   —,     —  ]
Query 4:   [0.44, 0.57,  0.66,  0.83,  0.72,   —  ]
Query 5:   [0.41, 0.78,  0.55,  0.90,  0.63,  0.82]
```

Top-K: 6 << 1024, all valid entries selected:

```
topk_indices_buffer[1, :] = [0, -1, ..., -1]          (1 valid)
topk_indices_buffer[2, :] = [1, 0, -1, ..., -1]
topk_indices_buffer[3, :] = [2, 1, 0, -1, ..., -1]
topk_indices_buffer[4, :] = [3, 2, 1, 0, -1, ..., -1]
topk_indices_buffer[5, :] = [3, 4, 2, 1, 0, -1, ..., -1]
topk_indices_buffer[6, :] = [3, 5, 1, 4, 2, 0, -1, ..., -1]  ← sorted by score
```

---

#### Stage 5: Indexer Decode — Paged Logits + Top-K

```
seq_lens = [[3000]],  block_table[0,:] = 47 blocks (3000 tokens, block_size=64)

fp8_fp4_paged_mqa_logits(q_fp8[:1].reshape(1,1,64,128), kv_cache, weights_out[:1],
                          seq_lens=[[3000]], block_table, schedule_metadata)
→ logits [1, 128000] fp32  (valid range [0..2999], rest is padding)

cooperative_topk(logits, seq_lens=[[3000]], K=1024):
  → topk_indices_buffer[0, :] = [847, 1203, 2714, 388, ..., 2999]  (1024 entries)
```

---

#### Stage 6: Main MLA Q Projection

```
q_b_proj(q_c) → q [7, 128, 192]   bf16
split:
  q_nope [7, 128, 128]
  q_pe   [7, 128,  64]

RoPE(q_pe) at each token's position

# Key MLA optimization: absorb W_UK_T into q_nope
ql_nope = bmm(q_nope.transpose(0,1), W_UK_T).transpose(0,1)
        → [7, 128, 512]
# Now each query head carries a 512-dim latent query vector
# that can be directly dot-producted against the cached 512-dim kv_c
```

---

#### Stage 7: FlashMLA Sparse Attention

**Prefill (req A, tokens 1..6):**

```
FlashMLA sparse forward for each query token t in req A:
  KV candidates = tokens at topk_indices_buffer[t, :] in the main MLA cache
                  (for prefill: all positions 0..t, since there are only 6 total)

  For each candidate KV position k:
    # Score: two dot products combined
    score[t, k] = dot(ql_nope[t, h, :], kv_c[k, :])   # 512-dim (absorbed W_UK)
                + dot(q_pe[t, h, :], k_pe_rotated[k, :])  # 64-dim
                for all h simultaneously

  Softmax over candidates → attention weights α[t, k]

  # Output: weighted sum in LATENT space
  out_latent[t, h, :] = Σ_k  α[t, k]  ×  kv_c[k, :]   [512-dim]

# Convert latent output back to v-space:
attn_out[t, h, :] = dot(out_latent[t, h, :], W_UV[h, :, :])   [128-dim per head]
```

**Decode (req B, token 0):**

```
FlashMLA sparse decode for token 0 (position 2999):
  KV candidates = main MLA cache at positions topk_indices_buffer[0, :] = [847, 1203, ..., 2999]
  (1024 candidates from 3000-token history)

  score[0, k] = dot(ql_nope[0, h, :], kv_c[k, :]) + dot(q_pe[0, h, :], k_pe[k, :])
  softmax → α[0, k]
  out_latent[0, h] = Σ_k α[0,k] × kv_c[k, :]
  attn_out[0, h] = dot(out_latent, W_UV[h])   [128-dim]

Compute reduction: 1024 / 3000 = 34%  (at 128K context: 1024 / 128000 < 1%)
```

---

#### Stage 8: Output Projection

```
attn_out [7, 128, 128]  → reshape [7, 128×128=16384]
o_proj RowParallelLinear [16384 → 7168]
→ output [7, 7168]
```

---

### 5.2 V4 Example: Prefill + Decode

**Setup:**

```
Batch: 2 requests, C4A layer (compress_ratio=4)
  Request A: 8-token new prompt   (prefill, seq_lens=8,    query_lens=8)
  Request B: continuing 4000 KVs  (decode,  seq_lens=4000, query_lens=1)
    → compressed indexer seq_len = 4000 // 4 = 1000

Token layout: decode first (index 0), prefill second (indices 1..8)
```

---

#### Stage 1: Parallel Input GEMMs (`attn_gemm_parallel_execute`)

```
Default stream:
  fused_wqa_wkv(hidden_states [9,7168])  →  qr [9,1536] + kv [9,512]

Aux stream 0:
  main_compressor.fused_wkv_wgate(hidden)  →  main_kv_score [9, 2048]

Aux stream 1:
  indexer.weights_proj(hidden)  →  indexer_weights [9, 64]

Aux stream 2:
  indexer.compressor.fused_wkv_wgate(hidden)  →  idx_kv_score [9, 512]

→ join → fused_q_kv_rmsnorm(qr, kv):
    qr [9,1536] ← RMSNorm(qr, q_norm.weight)
    kv [9, 512] ← RMSNorm(kv, kv_norm.weight)
```

---

#### Stage 2: Q up-projection + SWA Insert (default stream)

```
wq_b(qr) → q [9, 128, 512]  bf16  (128 heads × 512-dim)

fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q, kv, swa_kv_cache, ...):

  For each token t and head h:
    # Per-head RMSNorm (no weight):
    q[t, h, :] = q[t,h,:] / rms(q[t,h,:])

    # GPT-J RoPE on last 64 dims of q[t,h,:]:
    q[t, h, 448:] = rotate(q[t, h, 448:], position[t])

  For each token t (MQA: single K/V):
    # GPT-J RoPE on last 64 dims of kv[t,:]:
    kv[t, 448:] = rotate(kv[t, 448:], position[t])

    # FP8 UE8M0 quantize kv[t,:]:
    #   NoPE part kv[t,:448]: block-scale FP8 (block_size=64)
    #   RoPE part kv[t,448:]: stored as BF16 (not quantized)
    kv_fp8 = [fp8_block_quant(kv[t,:448]), bf16(kv[t,448:])]  # 448+128+8 = 584B

    # Write to SWA cache:
    swa_kv_cache[block, offset, :] = kv_fp8   at swa_slot_mapping[t]

→ q [9, 128, 512]  ready for attention
```

---

#### Stage 3: Indexer Compressor — Save States (aux stream 0)

```
idx_kv_score [9, 512]  (from indexer.compressor.fused_wkv_wgate)
     │
split: idx_kv [9, 256],  idx_score [9, 256]

For each token t, apply APE and write state:
  idx_kv[t]    += ape[t % 4, :256]
  idx_score[t] += ape[t % 4, 256:]
  write (idx_kv[t], idx_score[t]) to indexer state_cache at state_slot[t]

Tokens written:
  pos=0: state_cache[state_slot_0] = (kv+ape[0,:256], score+ape[0,256:])
  pos=1: state_cache[state_slot_1] = (kv+ape[1,:256], score+ape[1,256:])
  ...
  pos=7: state_cache[state_slot_7] = (kv+ape[3,:256], score+ape[3,256:])
  pos=3999: state_cache[state_slot_3999] = (kv+ape[3,:256], score+ape[3,256:])
```

---

#### Stage 4: Indexer Compressor — Compress → Write to Indexer K Cache

Fires for tokens where `position % 4 == 3`: positions 3, 7, 3999.

**Position 3999 (decode token):**

```
Gather 8 state entries from state_cache for positions [3992..3999]:
  kv_states    [8, 256]   # each from a different input token
  score_states [8, 256]

Gated compression (two groups of 4 due to overlap/coff=2):
  group0_weights = softmax(score_states[:4, :256])   # [4, 256] → softmax → gates
  group1_weights = softmax(score_states[4:, :256])
  compressed_kv [128] = mean(kv_states[:4,:128] × group0_weights[:,:128], dim=0)
                      + mean(kv_states[4:,:128] × group1_weights[:,:128], dim=0)
  (simplified — actual uses all 256 dims with proper normalization)

RMSNorm(compressed_kv, norm.weight)

GPT-J RoPE at position = (3999 // 4) × 4 = 3996:
  rotate last 64 dims of compressed_kv

FP8 quantize compressed_kv → k_fp8 [128] + scale

Write to indexer k_cache:
  compressed_slot = 3999 // 4 = 999
  kv_cache[block=999//256=3, offset=999%256=231, :] = k_fp8 + scale
```

**Position 3 (prefill):** compressed slot 0 → written to `kv_cache[block=0, offset=0]`
**Position 7 (prefill):** compressed slot 1 → written to `kv_cache[block=0, offset=1]`

Tokens at positions 0,1,2,4,5,6: no compressed output this step.

---

#### Stage 5: Indexer Q Quantization (overlapped with Stage 3/4)

```
wq_b(qr) → q_idx [9, 64, 128]  bf16

fused_indexer_q_rope_quant (grid (9, 64)):

For decode token 0 at position=3999, head 0:
  # GPT-J interleaved RoPE on rope dims (64..127):
  x_even = q_idx[0, 0, 64::2]   # 32 values at even positions
  x_odd  = q_idx[0, 0, 65::2]
  r_even = x_even×cos(3999) - x_odd×sin(3999)  → bf16 → fp32
  r_odd  = x_odd×cos(3999)  + x_even×sin(3999) → bf16 → fp32

  q_nope = q_idx[0, 0, :64]  (unchanged)

  amax = max(|q_nope|, |r_even|, |r_odd|) = 0.91
  q_scale = 2^(ceil(log2(0.91/448))) = 2^(-9) = 0.001953125

  q_fp8[0, 0, :64]  = clamp(q_nope / 0.001953125, -448, 448)
  q_fp8[0, 0, 64::2] = clamp(r_even / 0.001953125, -448, 448)
  q_fp8[0, 0, 65::2] = clamp(r_odd  / 0.001953125, -448, 448)

  # FP8 path: q_scale absorbed into weights
  weights_out[0, 0] = indexer_weights[0, 0] × 0.001953125 × (1/√128) × (1/√64)
                    = 0.038 × 0.001953125 × 0.08839 × 0.125
                    ≈ 8.21e-7
```

---

#### Stage 6: Compressed Metadata

```
compressed_seq_lens  = [4000//4, 8//4] = [1000, 2]
Decode indexer seq_lens: [[1000]]   (shape [B=1, next_n=1])

Prefill cu_seqlen_ke (compress_ratio=4, start_pos=0):
  ke[i] = (start_pos + 1 + i) // 4 = (i+1) // 4
  → ke = [0, 0, 0, 1, 1, 1, 1, 2]

  Interpretation:
    pos 0,1,2 → ke=0: no complete 4-token group yet, 0 compressed KVs visible
    pos 3     → ke=1: first group [0..3] is complete, 1 compressed KV visible
    pos 4,5,6 → ke=1: second group [4..7] not complete
    pos 7     → ke=2: second group complete, 2 compressed KVs visible
```

---

#### Stage 7: Indexer Prefill — Gather + Logits + Top-K

```
cp_gather_indexer_k_quant_cache → k_quant [2, 128] fp8,  k_scale [2, 4] uint8
  (2 compressed tokens: from positions 0-3 and 4-7)

fp8_fp4_mqa_logits(q_fp8[1:9], k_quant, weights_out[1:9],
                   cu_seqlen_ks=[0,0,0,0,0,0,0,0],
                   cu_seqlen_ke=[0,0,0,1,1,1,1,2])

logits [8, 2] fp32  (masked by cu_seqlen_ke):

  Compressed KV:   slot 0              slot 1
                   (summarizes pos 0-3) (summarizes pos 4-7)
  Query pos=0:     —                   —       (ke=0: 0 visible)
  Query pos=1:     —                   —
  Query pos=2:     —                   —
  Query pos=3:     0.51                —       (ke=1: slot 0 visible)
  Query pos=4:     0.44                —       (ke=1: slot 0 still)
  Query pos=5:     0.63                —
  Query pos=6:     0.49                —
  Query pos=7:     0.38               0.71     (ke=2: both visible)
```

Top-K: 1024 >> 2, all valid entries selected:

```
topk_indices_buffer[1,:] = [-1, ...]         (pos 0-2: none)
topk_indices_buffer[4,:] = [0, -1, ...]      (pos 3: compressed slot 0)
topk_indices_buffer[8,:] = [1, 0, -1, ...]   (pos 7: slots [1,0] sorted by score)
```

---

#### Stage 8: Indexer Decode — Paged Logits + Top-K

```
compressed seq_lens = [[1000]],  block_table: 4 blocks of 256

fp8_fp4_paged_mqa_logits walks 1000 compressed KV entries:
  For each ck in [0..999]:
    logits[0, ck] = Σ_h dot(q_fp8[0,h,:], k_fp8_compressed[ck,:]) × sk[ck] × weights_out[0,h]

  Each k_fp8_compressed[ck] was produced by compressing 4 original tokens

→ logits[0, :1000] filled  (logits[0, 1000:] = padding)

cooperative_topk (or persistent_topk): selects top 1024 from 1000 available
  → topk_indices_buffer[0, :1000] = [847, 203, 712, ..., 999]  (all 1000 selected)
  → topk_indices_buffer[0, 1000:] = [-1, ...]
```

---

#### Stage 9: Main MLA Attention — Combined SWA + Sparse

The main compressor (aux stream 1) has written 4000//4=1000 compressed entries to the main MLA KV cache. These represent the full history in the 512-dim latent space.

**SWA coverage** (window=4096, req B at position 3999):

```
SWA covers positions: max(0, 3999-4095) = 0 .. 3999
→ all 4000 tokens are within the SWA window for this sequence
→ swa_kv_cache already has all 4000 tokens in 512-dim FP8 format
```

**Expanded sparse indices** (from indexer compressed indices):

```
topk_indices_buffer[0, :] = [847, 203, ..., 999]   (1000 compressed positions)

For each compressed index ck, expand to 4 original slots:
  original positions = [4×ck, 4×ck+1, 4×ck+2, 4×ck+3]
  global_topk_indices = [block_table[req, ck//block_size] × block_size
                         + (ck % block_size) × 4 + j  for j in [0..3]]

→ global_topk_indices [1, 1, 4000]  (1000 compressed × 4 = 4000 slots)
```

**FlashMLA call:**

```python
flash_mla_with_kvcache(
    q = q[0:1].unsqueeze(1),            # [1, 1, 128, 512]  decode token
    k_cache = swa_kv_cache,             # [num_blocks, block_size, 512]
    indices = swa_indices,              # [1, 1, 4000]  (all tokens in SWA window)
    topk_length = [4000],               # SWA covers full history

    extra_k_cache = main_mla_kv_cache,  # compressed 512-dim main cache
    extra_indices_in_kvcache = global_topk_indices,  # [1, 1, 4000]
    extra_topk_length = [4000],

    softmax_scale = 1/√512,
    attn_sink = self.attn_sink,
    out = output[0:1].unsqueeze(1),
)
```

Since SWA already covers all 4000 positions, the sparse indexer effectively doubles the coverage for this short sequence. At longer sequences (e.g., 163840 tokens), the SWA covers only the last 4096, and the indexer selects 1024 compressed tokens × 4 = 4096 additional original tokens from the distant past, for ~8% total coverage.

---

#### Stage 10: V4 Output Projection

```
attn_out [9, 128, 512]   bf16  (attention output, 128 heads × 512-dim)
     │
     ▼  fused_inv_rope_fp8_quant  (Triton kernel)
        For each token t, head h:
          # Inverse GPT-J RoPE on rope_head_dim=64 dims:
          #   Forward RoPE:  [x_even, x_odd] → [x_even×cos - x_odd×sin, x_odd×cos + x_even×sin]
          #   Inverse:       [y_even, y_odd] → [y_even×cos + y_odd×sin, -y_even×sin + y_odd×cos]
          out_h = [nope_unchanged, inv_rope(rope_part)]

          # Group heads: n_groups blocks of (128/n_groups) heads each
          # FP8 block quantize within each group:
          out_fp8[t, g, :] = fp8_block_quant(cat of heads in group g)
     │
o_fp8  [9, n_groups, heads_per_group×512]  fp8
o_scale [9, n_groups, n_scale_blocks]      ue8m0
     │
     ▼  wo_a  grouped FP8 BMM  (n_groups parallel matmuls)
        fp8_einsum("bhr,hdr->bhd", o_fp8, wo_a.weight)
        [9, n_groups, heads_per_group×512] × [n_groups, heads_per_group×512, o_lora_rank]
        → z [9, n_groups, o_lora_rank]   bf16
     │
     ▼  wo_b  RowParallelLinear  [n_groups×o_lora_rank → 7168]
     │
output [9, 7168]
```

The inverse RoPE is necessary because `q_pe` was rotated before attention, so the attention output mixes rotated and unrotated components. The inverse rotation restores the geometric basis that `wo_a`'s weight matrix was trained to expect.

---

## 6. Key Implementation Files

| File | Role |
|---|---|
| [vllm/model_executor/layers/sparse_attn_indexer.py](vllm/model_executor/layers/sparse_attn_indexer.py) | Core `SparseAttnIndexer` op; `fused_indexer_q_rope_quant` (V3 Triton kernel); `sparse_attn_indexer` function; DCP merge |
| [vllm/v1/attention/backends/mla/indexer.py](vllm/v1/attention/backends/mla/indexer.py) | `DeepseekV32IndexerMetadataBuilder`; prefill chunking; `_build_prefill_chunk_metadata_kernel`; decode tensor prep |
| [vllm/model_executor/models/deepseek_v2.py](vllm/model_executor/models/deepseek_v2.py) | V3 `DeepseekV2Attention` (MLA), `Indexer` class, `DeepseekV32IndexerCache`, weight absorption (`W_UK_T`, `W_UV`) |
| [vllm/models/deepseek_v32/nvidia/attention.py](vllm/models/deepseek_v32/nvidia/attention.py) | V3.2 optimized `DeepseekV32Attention`; fused norm+rope kernels; `DeepseekV32Indexer` |
| [vllm/v1/attention/backends/mla/flashmla_sparse.py](vllm/v1/attention/backends/mla/flashmla_sparse.py) | V3.2 FlashMLA sparse backend; `FlashMLASparseImpl.forward_mqa`; `_fp8_flash_mla_kernel`; KV cache format notes |
| [vllm/models/deepseek_v4/attention.py](vllm/models/deepseek_v4/attention.py) | `DeepseekV4Attention` base (MLA forward, SWA insert, stream orchestration); `DeepseekV4Indexer`; `DeepseekV4IndexerCache` |
| [vllm/models/deepseek_v4/compressor.py](vllm/models/deepseek_v4/compressor.py) | `DeepseekCompressor`; `CompressorStateCache`; `save_partial_states`; `compress_norm_rope_store_*` dispatch |
| [vllm/models/deepseek_v4/nvidia/flashmla.py](vllm/models/deepseek_v4/nvidia/flashmla.py) | V4 `DeepseekV4FlashMLAAttention`; `_forward_decode` / `_forward_prefill`; SWA+sparse FlashMLA calls |
| [vllm/models/deepseek_v4/nvidia/ops/o_proj.py](vllm/models/deepseek_v4/nvidia/ops/o_proj.py) | `deep_gemm_fp8_o_proj`; inverse RoPE + FP8 quant + grouped BMM (`wo_a`) + `wo_b` |
| [vllm/models/deepseek_v4/common/ops/fused_indexer_q.py](vllm/models/deepseek_v4/common/ops/fused_indexer_q.py) | V4 `fused_indexer_q_rope_quant`; FP8 and MXFP4 Triton kernels; weight absorption semantics |
| [vllm/models/deepseek_v4/sparse_mla.py](vllm/models/deepseek_v4/sparse_mla.py) | `DeepseekV4FlashMLABackend`; `DeepseekV4FlashMLAMetadata`; `build_c128a_topk_metadata`; C128A kernel |
| [vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py](vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py) | `compress_norm_rope_store_triton`; compressor state gather + RMSNorm + RoPE + FP8/MXFP4 quant + write |
| [vllm/v1/attention/backends/mla/sparse_swa.py](vllm/v1/attention/backends/mla/sparse_swa.py) | `DeepseekSparseSWABackend`; SWA metadata builder; `DeepseekV4SWACache` |
| [vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py](vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py) | `fused_inv_rope_fp8_quant` Triton kernel; inverse GPT-J RoPE + FP8 group quantize |
| [vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py](vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py) | DCP top-K merge: `pack_dcp_topk_candidates_cutedsl`; `stable_topk_from_gathered_candidates_cutedsl` |
| [vllm/v1/attention/ops/rocm_aiter_mla_sparse.py](vllm/v1/attention/ops/rocm_aiter_mla_sparse.py) | ROCm/AITER sparse indexer and MLA attention kernels |
