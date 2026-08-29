# Kimi-K3: Layer Types and Per-Layer Structure

[← Index](kimi_k3_index.md)

Explains the two attention types (MLA vs KDA), the two FFN types (MoE vs dense), how they are assigned across all 93 layers, and why the pattern is designed this way.

---

## Table of Contents

1. [Why Hybrid Attention?](#1-why-hybrid-attention)
2. [The Four Layer Component Types](#2-the-four-layer-component-types)
   - 2.1 [Full MLA (MultiHeadLatentAttention)](#21-full-mla-multiheadlatentattention)
   - 2.2 [KDA (KimiK3DeltaAttention)](#22-kda-kimik3deltaattention)
   - 2.3 [MoE FFN (KimiMoE)](#23-moe-ffn-kimimoe)
   - 2.4 [Dense MLP (KimiMLP)](#24-dense-mlp-kimimlp)
3. [Layer Assignment Rules](#3-layer-assignment-rules)
4. [Complete Per-Layer Table (all 93 layers)](#4-complete-per-layer-table-all-93-layers)
5. [Layer Internal Structure](#5-layer-internal-structure)
   - 5.1 [MLA + MoE layer (standard)](#51-mla--moe-layer-standard)
   - 5.2 [KDA + MoE layer (most common, 69 layers)](#52-kda--moe-layer-most-common)
   - 5.3 [MLA + Dense layer (layer 0 only)](#53-mla--dense-layer-layer-0-only)
6. [AttnRes: Cross-Layer Residual](#6-attnres-cross-layer-residual)
7. [3:1 KDA:MLA Ratio — Rationale](#7-31-kdamla-ratio--rationale)
8. [KV Cache Sizing by Layer Type](#8-kv-cache-sizing-by-layer-type)
9. [Worked Examples: KDA State vs MLA KV Cache Growth](#9-worked-examples-kda-state-vs-mla-kv-cache-growth)
   - 9.1 [KDA state is fixed; MLA KV cache grows linearly](#91-kda-state-is-fixed-mla-kv-cache-grows-linearly)
   - 9.2 [AttnRes block write schedule](#92-attnres-block-write-schedule-93-layers-attn_res_block_size12)
   - 9.3 [KDA as K3's "cheap approximation" — analogy to DeepSeek C4A/C128A](#93-kda-as-k3s-cheap-approximation--analogy-to-deepseek-c4ac128a)

---

## 1. Why Hybrid Attention?

Full MLA attention is `O(N²)` in sequence length. At 1M context with 96 heads, running full attention every layer is prohibitively expensive. KDA (Gated Delta Net linear attention) processes the same input in `O(N)` time by maintaining a fixed-size recurrent state, at the cost of some precision for long-range dependencies.

Kimi-K3 uses a **3:1 KDA:MLA ratio**: for every full MLA layer, there are 3 KDA layers. This gives:
- Full quadratic attention every 4th layer — capturing long-range exact dependencies at that frequency
- Linear recurrent updates for the other 3 layers — building up local context cheaply
- Total attention cost: ~25% of all-MLA cost

The AttnRes mechanism partially compensates for the KDA approximation by maintaining a bank of recent MLA layer outputs that any layer can blend in.

---

## 2. The Four Layer Component Types

### 2.1 Full MLA (MultiHeadLatentAttention)

- 96 heads, 192-dim head (128 NoPE + 64 RoPE)
- KV cache stores `(c_kv [512], k_pe [64])` = 576 dims per token
- Weight absorption at load time: `W_UK_T [96, 128, 512]`, `W_UV [96, 512, 128]`
- `mla_use_nope=True`: the 512-dim latent carries no position — only `k_pe` does
- `mla_use_output_gate=True`: sigmoid gate on attention output
- Supports chunked context, DCP, fp8/fp8_ds_mla KV cache
- See [kimi_k3_mla.md](kimi_k3_mla.md)

### 2.2 KDA (KimiK3DeltaAttention)

- 96 heads, `kDimK=kDimV=128`, conv1d kernel width 4
- Recurrent state: conv1d state `[B, 96, 4, 128]` + delta-rule memory `[B, 96, 128, 128]`
- Per-token KV cache: none (state is maintained in the recurrent buffer)
- Decode: single fused CUDA kernel (NVIDIA) or HIP kernel (ROCm gfx950)
- Prefill: Triton chunked linear attention kernels
- No KV paging needed — state fits in SRAM/HBM as a fixed tensor
- See [kimi_k3_kda.md](kimi_k3_kda.md)

### 2.3 MoE FFN (KimiMoE)

- 896 total routed experts, top-16 selected per token
- 2 shared experts always active
- **Latent MoE**: expert path is `hidden [7168] → latent [3584] → hidden [7168]` (not the standard gate+up+down)
- Expert weights quantized to MXFP4 (group_size=32)
- `SiTU` activation: `sigmoid(4.0*x) * (x + 25.0)` instead of SiLU
- Routing: grouped top-k with `noaux_tc` load-balancing bias
- See [kimi_k3_moe.md](kimi_k3_moe.md)

### 2.4 Dense MLP (KimiMLP)

Used only at layer 0 (`first_k_dense_replace=1`).

```
x → gate_up_proj [7168 → 2×33792]
  → split: gate [33792], up [33792]
  → situ(gate) × up
  → down_proj [33792 → 7168]
```

`intermediate_size = 33792` (much larger than expert intermediate of 3072).

---

## 3. Layer Assignment Rules

From the config and model code:

```python
# (1) FFN type:
if layer_idx < first_k_dense_replace:   # first_k_dense_replace = 1
    ffn = KimiMLP(intermediate_size=33792)
else:
    ffn = KimiMoE(num_experts=896, ...)

# (2) Attention type:
if layer_idx in linear_attn_config.kda_layers:    # 69 layers
    attn = KimiK3DeltaAttention(...)
elif layer_idx in linear_attn_config.full_attn_layers:  # 24 layers
    attn = MultiHeadLatentAttention(...)
# Note: layer 0 is in NEITHER list → defaults to MLA (not explicitly listed)
```

Layer 0 is not in `kda_layers` (indices 1-3, 5-7, …) nor in `full_attn_layers` (indices 4, 8, …, 92). The model defaults it to MLA. So layer 0 has MLA + Dense MLP.

---

## 4. Complete Per-Layer Table (all 93 layers)

| Layer | Attention | FFN | Notes |
|---|---|---|---|
| 0 | **MLA** | **Dense MLP** | Only dense layer; first_k_dense_replace=1 |
| 1 | KDA | MoE | |
| 2 | KDA | MoE | |
| 3 | KDA | MoE | |
| 4 | **MLA** | MoE | |
| 5 | KDA | MoE | |
| 6 | KDA | MoE | |
| 7 | KDA | MoE | |
| 8 | **MLA** | MoE | |
| 9 | KDA | MoE | |
| 10 | KDA | MoE | |
| 11 | KDA | MoE | |
| 12 | **MLA** | MoE | |
| 13 | KDA | MoE | |
| 14 | KDA | MoE | |
| 15 | KDA | MoE | |
| 16 | **MLA** | MoE | |
| 17 | KDA | MoE | |
| 18 | KDA | MoE | |
| 19 | KDA | MoE | |
| 20 | **MLA** | MoE | |
| 21 | KDA | MoE | |
| 22 | KDA | MoE | |
| 23 | KDA | MoE | |
| 24 | **MLA** | MoE | |
| 25 | KDA | MoE | |
| 26 | KDA | MoE | |
| 27 | KDA | MoE | |
| 28 | **MLA** | MoE | |
| 29 | KDA | MoE | |
| 30 | KDA | MoE | |
| 31 | KDA | MoE | |
| 32 | **MLA** | MoE | |
| 33 | KDA | MoE | |
| 34 | KDA | MoE | |
| 35 | KDA | MoE | |
| 36 | **MLA** | MoE | |
| 37 | KDA | MoE | |
| 38 | KDA | MoE | |
| 39 | KDA | MoE | |
| 40 | **MLA** | MoE | |
| 41 | KDA | MoE | |
| 42 | KDA | MoE | |
| 43 | KDA | MoE | |
| 44 | **MLA** | MoE | |
| 45 | KDA | MoE | |
| 46 | KDA | MoE | |
| 47 | KDA | MoE | |
| 48 | **MLA** | MoE | |
| 49 | KDA | MoE | |
| 50 | KDA | MoE | |
| 51 | KDA | MoE | |
| 52 | **MLA** | MoE | |
| 53 | KDA | MoE | |
| 54 | KDA | MoE | |
| 55 | KDA | MoE | |
| 56 | **MLA** | MoE | |
| 57 | KDA | MoE | |
| 58 | KDA | MoE | |
| 59 | KDA | MoE | |
| 60 | **MLA** | MoE | |
| 61 | KDA | MoE | |
| 62 | KDA | MoE | |
| 63 | KDA | MoE | |
| 64 | **MLA** | MoE | |
| 65 | KDA | MoE | |
| 66 | KDA | MoE | |
| 67 | KDA | MoE | |
| 68 | **MLA** | MoE | |
| 69 | KDA | MoE | |
| 70 | KDA | MoE | |
| 71 | KDA | MoE | |
| 72 | **MLA** | MoE | |
| 73 | KDA | MoE | |
| 74 | KDA | MoE | |
| 75 | KDA | MoE | |
| 76 | **MLA** | MoE | |
| 77 | KDA | MoE | |
| 78 | KDA | MoE | |
| 79 | KDA | MoE | |
| 80 | **MLA** | MoE | |
| 81 | KDA | MoE | |
| 82 | KDA | MoE | |
| 83 | KDA | MoE | |
| 84 | **MLA** | MoE | |
| 85 | KDA | MoE | |
| 86 | KDA | MoE | |
| 87 | KDA | MoE | |
| 88 | **MLA** | MoE | |
| 89 | KDA | MoE | |
| 90 | KDA | MoE | |
| 91 | KDA | MoE | |
| 92 | **MLA** | MoE | Last layer |

**Counts:** 1 × (MLA+Dense), 23 × (MLA+MoE), 69 × (KDA+MoE). Total: 93 layers.

---

## 5. Layer Internal Structure

### 5.1 MLA + MoE layer (standard)

```
hidden_states [T, 7168]
      │
      ▼ input_layernorm (RMSNorm, eps=1e-5)
      │
      ▼ MultiHeadLatentAttention
      │     fused_qkv_a_proj [7168 → q_lora_rank=1536 + kv_lora_rank=512]
      │     q_a_layernorm [1536] → q_b_proj [1536 → 96×192]
      │     kv_a_proj_with_mqa [7168 → 512 + 64]   (c_kv + k_pe)
      │     rotary_embedding (on q_pe and k_pe only, mla_use_nope=True)
      │     [optional] g_proj [7168 → 1]  (sigmoid output gate)
      │     o_proj [96×128 → 7168]  (+ optional GEMM-RS fusion)
      │
      ▼ AttnRes write (add current hidden to block bank at bank[current_block_idx])
      │
      ▼ post_attention_layernorm (RMSNorm)
      │
      ▼ KimiMoE
      │     GateLinear router [7168 → 896]
      │     top-16 selection (grouped top-k + noaux_tc bias)
      │     16 latent expert paths:
      │       x → routed_expert_down_proj [7168 → 3584]  (latent_down)
      │       → RMSNorm [3584]  (latent_moe_use_norm=True)
      │       → routed_expert_up_proj [3584 → 7168]  (latent_up, replicated)
      │     2 shared expert paths:
      │       x → shared_gate_up [7168 → 2×3072] → situ → down [3072 → 7168]
      │     output = weighted_sum(routed) + sum(shared)
      │
      ▼ AttnRes read (weighted mix of block bank added to output)
      │
hidden_states_out [T, 7168]
```

### 5.2 KDA + MoE layer (most common)

```
hidden_states [T, 7168]
      │
      ▼ input_layernorm (RMSNorm)
      │
      ▼ KimiK3DeltaAttention (Gated Delta Net)
      │     in_proj_qkvgfab [7168 → 96*(128+128+128+1+1+128+1)]  (Q,K,V,g,f_a,b,beta)
      │     short_conv1d (kernel_width=4, hardcoded)
      │     fused_kda_decode (CUDA kernel) OR Triton chunked prefill
      │     out_proj [96×128 → 7168]
      │
      ▼ AttnRes write
      │
      ▼ post_attention_layernorm (RMSNorm)
      │
      ▼ KimiMoE  (same as MLA layer)
      │
      ▼ AttnRes read
      │
hidden_states_out [T, 7168]
```

### 5.3 MLA + Dense layer (layer 0 only)

```
hidden_states [T, 7168]
      │
      ▼ input_layernorm (RMSNorm)
      │
      ▼ MultiHeadLatentAttention  (same as standard MLA layer)
      │
      ▼ AttnRes write
      │
      ▼ post_attention_layernorm (RMSNorm)
      │
      ▼ KimiMLP (Dense)
      │     gate_up_proj [7168 → 2×33792]  (gate+up packed)
      │     situ(gate) × up → [33792]
      │     down_proj [33792 → 7168]
      │
      ▼ AttnRes read
      │
hidden_states_out [T, 7168]
```

---

## 6. AttnRes: Cross-Layer Residual

AttnRes is applied at **every layer** regardless of attention type. It maintains a circular bank of `attn_res_block_size=12` stored attention hidden states, allowing any layer to access summaries of the last 12 blocks of computation.

**Write phase** (after attention, before FFN):

```python
# Block index = layer_idx // layers_per_block (where layers_per_block = num_layers / attn_res_block_size)
block_idx = layer_idx // block_period
attn_res_bank[block_idx, :] = hidden_after_attention[t, :]  # circular write
```

**Read phase** (after FFN):

```python
# Score each of the 12 stored blocks:
scores = attn_res_norm(hidden_after_ffn) @ attn_res_proj  # [T, 12] logits
weights = softmax(scores)  # [T, 12]
# Weighted mixture of 12 stored states:
residual = weights @ attn_res_bank  # [T, 7168]
output = hidden_after_ffn + residual
```

The actual implementation uses a fused kernel (SM100 native CUDA or Triton fallback) that handles both the score computation and weighted sum in one kernel to avoid round-trips through HBM.

**Why 12 blocks?** `attn_res_block_size=12` covers `93/12 ≈ 7.75 layers per block`, so the bank spans the full depth of the network. At any point, a layer can blend in context from every 8th layer going back to layer 0.

---

## 7. 3:1 KDA:MLA Ratio — Rationale

The interleaving is: `[MLA, KDA, KDA, KDA, MLA, KDA, KDA, KDA, ...]`.

Starting from layer 1 (after the dense layer 0):
- Layer 0: MLA (bootstrap context before any KDA state exists)
- Layers 1–3: KDA (build up recurrent state using layer-0 attention context)
- Layer 4: MLA (exact quadratic attention, resets/corrects KDA approximation)
- Layers 5–7: KDA again
- …and so on through layer 92

**Why exactly 3 KDA per MLA:**
- 1:1 would recover ~50% of full-MLA quality at 50% the cost — not enough savings
- 7:1 would be too inaccurate; KDA drifts over long sequences
- 3:1 empirically balances quality and cost: full quadratic refresh every 4 layers prevents error accumulation while reducing attention FLOPs by ~75%

**Why KDA layers 1,2,3 (not 0,1,2):** Layer 0 establishes the initial representation. KDA at layer 0 would have no prior attention context to refine, making the delta-rule updates less meaningful.

---

## 8. KV Cache Sizing by Layer Type

| Layer type | Per-token KV cache | Notes |
|---|---|---|
| MLA layer | 576 dims = 512 (c_kv) + 64 (k_pe) | Paged, bf16 or fp8 |
| KDA layer | 0 (no per-token cache) | State stored as fixed tensor per sequence |

**MLA KV cache per layer:**
```
BF16:        576 × 2 bytes = 1152 bytes/token/layer
FP8:         576 bytes/token/layer (per-token scale metadata extra)
fp8_ds_mla:  656 bytes/token/layer (block-scaled, see kimi_k3_mla.md §5)
```

**KDA state per sequence:**
```
Conv1d state:  96 heads × 4 (kernel_width) × 128 (kDimK) × 2 bytes = 98,304 bytes
Delta memory:  96 heads × 128 (kDimK) × 128 (kDimV) × 2 bytes = 3,145,728 bytes
Total per KDA layer:  ~3.1 MB per sequence (independent of sequence length!)
```

**Comparison at 128K tokens:**
```
MLA KV cache (24 layers × 128K tokens):  1152 × 24 × 128000 ≈ 3.5 GB/sequence
KDA state (69 layers):                   3.1 MB × 69 ≈ 215 MB/sequence (fixed!)
Total:                                   ~3.7 GB/sequence at 128K context
```

The KDA state does not grow with sequence length — this is the key memory advantage of linear attention for very long contexts.

---

## 9. Worked Examples: KDA State vs MLA KV Cache Growth

### 9.1 KDA state is fixed; MLA KV cache grows linearly

KDA recurrent state size per layer:
```
Conv1d state: 96 × 4 × 128 × 2 bytes  =     98,304 bytes  (~96 KB)
Delta memory: 96 × 128 × 128 × 2 bytes = 3,145,728 bytes  (~3.0 MB)
Total per KDA layer:                    ≈ 3.1 MB  (constant — no seq_len dependence)
```

MLA KV cache size per layer (BF16):
```
576 dims × 2 bytes × seq_len  =  1,152 × seq_len bytes per MLA layer
```

Growth table across six representative context lengths:

| seq_len | MLA KV/layer | 24 MLA layers | KDA state/layer | 69 KDA layers | Total |
|---|---|---|---|---|---|
| 1K (1,024) | 1.1 MB | 26.5 MB | 3.1 MB | 215 MB | ~242 MB |
| 8K (8,192) | 9.0 MB | 215 MB | 3.1 MB | 215 MB | ~430 MB |
| 32K (32,768) | 36 MB | 864 MB | 3.1 MB | 215 MB | ~1.1 GB |
| 128K (131,072) | 144 MB | 3.46 GB | 3.1 MB | 215 MB | ~3.7 GB |
| 512K (524,288) | 576 MB | 13.8 GB | 3.1 MB | 215 MB | ~14.1 GB |
| 1M (1,048,576) | 1.15 GB | 27.6 GB | 3.1 MB | 215 MB | ~27.8 GB |

Key observation: at 1K tokens the MLA KV cache (26.5 MB) is already smaller than the KDA state (215 MB) because the KDA state is always allocated at full capacity. The crossover happens around 8K tokens. For sequences beyond ~8K, the MLA KV cache dominates and grows without bound, while KDA state remains fixed at 215 MB regardless of whether seq_len is 100 or 1,000,000.

---

### 9.2 AttnRes block write schedule (93 layers, attn_res_block_size=12)

`block_period = 93 // 12 = 7` (integer division; each block spans 7 consecutive layers). A layer writes its attention output into the block bank when `layer_idx % block_period == 0`, i.e. every 7th layer starting from layer 0.

```
block_period = num_layers // attn_res_block_size = 93 // 12 = 7

Write layers (layer_idx % 7 == 0):  0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91
  (14 write layers — 12 full blocks plus 2 extra due to 93 not being a multiple of 7×12)
Read-only layers: all others (79 layers)
```

Write schedule for selected layers:

| Layer | layer_idx % 7 | Is write layer? | block_write_idx | Attention type |
|---|---|---|---|---|
| 0 | 0 | Yes | 0 | MLA (dense) |
| 1 | 1 | No | — | KDA |
| 2 | 2 | No | — | KDA |
| 7 | 0 | Yes | 1 | KDA |
| 8 | 1 | No | — | MLA |
| 14 | 0 | Yes | 2 | KDA |
| 21 | 0 | Yes | 3 | KDA |
| 28 | 0 | Yes | 4 | MLA |
| 35 | 0 | Yes | 5 | KDA |
| 42 | 0 | Yes | 6 | KDA |
| 49 | 0 | Yes | 7 | KDA |
| 56 | 0 | Yes | 8 | MLA |
| 63 | 0 | Yes | 9 | KDA |
| 70 | 0 | Yes | 10 | KDA |
| 77 | 0 | Yes | 11 | KDA |
| 84 | 0 | Yes | 12 | MLA |
| 91 | 0 | Yes | 13 | KDA |
| 92 | 1 | No | — | MLA |

Note that write layers are not always MLA layers — they fall wherever `layer_idx % 7 == 0` lands in the 3:1 KDA:MLA interleaving. The bank captures both MLA and KDA attention outputs.

---

### 9.3 KDA as K3's "cheap approximation" — analogy to DeepSeek C4A/C128A

DeepSeek V3/V4 uses two attention tiers:
- **C128A (MLA)**: full quadratic attention with 128-dim compressed KV. Exact. Expensive.
- **C4A (SWA)** *(V4 only)*: sliding-window attention over only the last 4K tokens. Cheap. Loses long-range context.

Kimi-K3 uses an analogous two-tier structure:
- **MLA** (every 4th layer, 24 total): full quadratic attention over the entire context. Exact. Expensive.
- **KDA** (3 of every 4 layers, 69 total): Gated Delta Net linear recurrence. O(N) time, fixed-size state. Cheap. Approximates long-range context.

Structural similarity:

| | DeepSeek V4 | Kimi-K3 |
|---|---|---|
| Exact tier | C128A MLA (all layers in V3, mixed in V4) | MLA every 4th layer |
| Approximate tier | C4A SWA (window=4096, V4 only) | KDA linear attention (O(N), fixed state) |
| Ratio | ~1:1 in V4 (alternating) | 1:3 MLA:KDA |
| Approximation mechanism | Hard window cutoff (no long-range) | Recurrent state compression (soft long-range) |
| Compensation | None | AttnRes bank blends in saved MLA states |
| KV cache | Grows with context for both tiers | Grows for MLA; fixed ~3.1 MB/layer for KDA |

The key difference: C4A simply discards tokens outside the window (hard cutoff), while KDA maintains a compressed summary of all past tokens in a fixed-size delta-rule memory matrix. KDA is a softer approximation — it degrades gracefully over long contexts rather than losing all information beyond a fixed window. The AttnRes mechanism further compensates by letting any layer blend in saved MLA outputs, partially restoring the long-range information that KDA's compression loses.
