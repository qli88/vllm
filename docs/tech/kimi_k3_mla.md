# Kimi-K3: Multi-Head Latent Attention (MLA)

[← Index](kimi_k3_index.md) | [Layer Types](kimi_k3_layers.md) | [End-to-End](kimi_k3_e2e.md)

Complete reference for K3's MLA implementation. Covers projections, NoPE design, weight absorption, the prefill and decode execution paths, all fused kernels, and KV cache formats.

---

## Table of Contents

1. [MLA Overview and Dimensions](#1-mla-overview-and-dimensions)
2. [Input Projections](#2-input-projections)
   - 2.1 [fused_qkv_a_proj (with q-LoRA)](#21-fused_qkv_a_proj-with-q-lora)
   - 2.2 [kv_a_proj_with_mqa](#22-kv_a_proj_with_mqa)
   - 2.3 [_decode_concat_cache() Dispatch Table](#23-_decode_concat_cache-dispatch-table)
3. [NoPE Design: Why K/V Carry No Position](#3-nope-design-why-kv-carry-no-position)
4. [Weight Absorption: W_UK_T and W_UV](#4-weight-absorption-w_uk_t-and-w_uv)
5. [Q Path: q_b_proj and RoPE](#5-q-path-q_b_proj-and-rope)
6. [Decode Path](#6-decode-path)
   - 6.1 [BMM1: Q_nope absorption](#61-bmm1-q_nope-absorption)
   - 6.2 [Fused decode q-concat + KV cache insert kernel](#62-fused-decode-q-concat--kv-cache-insert-kernel)
   - 6.3 [forward_mqa: paged MQA attention](#63-forward_mqa-paged-mqa-attention)
   - 6.4 [BMM2: W_UV output projection](#64-bmm2-w_uv-output-projection)
7. [Prefill Path](#7-prefill-path)
   - 7.1 [New tokens: run_prefill_new_tokens](#71-new-tokens-run_prefill_new_tokens)
   - 7.2 [Chunked context: _compute_prefill_context](#72-chunked-context-_compute_prefill_context)
   - 7.3 [Fused prefill kernels (by KV cache dtype)](#73-fused-prefill-kernels-by-kv-cache-dtype)
8. [Output Gate (g_proj)](#8-output-gate-g_proj)
9. [Output Projection and GEMM-RS](#9-output-projection-and-gemm-rs)
10. [KV Cache Formats](#10-kv-cache-formats)
    - 10.1 [BF16 format](#101-bf16-format)
    - 10.2 [FP8 format](#102-fp8-format)
    - 10.3 [fp8_ds_mla format (656-byte block-scaled)](#103-fp8_ds_mla-format-656-byte-block-scaled)
11. [DCP Integration](#11-dcp-integration)
    - 11.1 [Fused MLA Key-Concat CUDA Kernel: Geometry](#111-fused-mla-key-concat-cuda-kernel-geometry)
12. [Concrete Examples](#12-concrete-examples)
    - 12.1 [Decode: one token](#121-decode-one-token)
    - 12.2 [Prefill: 8-token prompt](#122-prefill-8-token-prompt)
    - 12.3 [NoPE MLA decode — shape comparison with/without NoPE](#123-nope-mla-decode--shape-comparison-withwithout-nope)
    - 12.4 [Chunked context prefill — merging 4 chunks](#124-chunked-context-prefill--merging-4-chunks)
    - 12.5 [fp8_ds_mla quantization of one KV slot](#125-fp8_ds_mla-quantization-of-one-kv-slot)

---

## 1. MLA Overview and Dimensions

Multi-Head Latent Attention compresses the KV cache by storing a low-dimensional latent vector per token instead of full K/V heads. At decode time, the K-side of the latent is folded into Q (weight absorption), eliminating per-token `kv_b_proj` calls.

**Key dimensions (from config):**

| Symbol | Config key | Value | Meaning |
|---|---|---|---|
| `H` | `num_attention_heads` | 96 | Number of Q heads |
| `L` | `kv_lora_rank` | 512 | KV latent dimension |
| `P` | `qk_nope_head_dim` | 128 | NoPE dim per Q/K head |
| `R` | `qk_rope_head_dim` | 64 | RoPE dim per Q/K head |
| `V` | `v_head_dim` | 128 | Value dim per head |
| `Ql` | `q_lora_rank` | 1536 | Q low-rank bottleneck |
| `D` | `hidden_size` | 7168 | Token hidden dim |
| — | — | 576 | KV cache entry = L+R per token |

**What is stored in the KV cache:**
- `c_kv [L=512]`: the compressed KV latent — carries both K_nope and V information
- `k_pe [R=64]`: the RoPE-encoded K component (position-dependent)
- Total: 576 values per token per MLA layer

---

## 2. Input Projections

### 2.1 `fused_qkv_a_proj` (with q-LoRA)

A single replicated (non-TP-sharded) GEMM that projects `hidden_states` to both the Q low-rank bottleneck and the KV down-projection:

```
hidden_states [T, D=7168]
    × weight [Ql + L, D] = [1536+512, 7168] = [2048, 7168]
    ─────────────────────────────────────────────────────
    output [T, 2048]
      ├─ q_a  [T, 1536]   ← Q low-rank activations
      └─ kv_a [T, 512]    ← KV latent activations (before k_pe split)
```

**Why replicated:** Both Q and KV activations read from the same `hidden_states`. Replicating (not TP-sharding) this GEMM means each TP rank holds the full weight and produces the full output — avoiding an all-gather on the Q bottleneck. The subsequent `q_b_proj` is TP-sharded, distributing the larger expansion across ranks.

After `fused_qkv_a_proj`:
```python
q_a = layer_norm(output[:, :1536])   # q_a_layernorm (RMSNorm)
# kv_a will be further split by kv_a_proj_with_mqa
```

### 2.2 `kv_a_proj_with_mqa`

Projects `hidden_states` to the KV latent + k_pe:

```
hidden_states [T, 7168]
    × weight [L+R, D] = [512+64, 7168] = [576, 7168]
    ──────────────────────────────────────────
    output [T, 576]
      ├─ c_kv [T, 512]  ← KV latent (normalized by kv_a_layernorm)
      └─ k_pe [T, 64]   ← RoPE key component
```

`k_pe` goes through rotary embedding:
```python
k_pe = kv_a_proj_with_mqa(hidden_states)[:, 512:]   # [T, 64]
k_pe_rotated = rotary_embedding(k_pe, positions)      # [T, 64]
```

`c_kv` is NOT rotated — it is stored as-is in the KV cache (NoPE design).

---

### 2.3 `_decode_concat_cache()` Dispatch Table

The decode path dispatches to one of three fused CUDA kernel variants based on KV cache dtype:

| `kv_cache_dtype` | Q format | Cache shape | Per-slot bytes | Kernel |
|---|---|---|---|---|
| `"auto"` / BF16 | BF16 | `[nblk, bs, 512]` BF16 | 512 × 2 = 1024B | `fused_mla_decode_q_concat_kv_cache_insert` |
| plain FP8 | BF16 | `[nblk, bs, 512]` FP8 | 512B + scale | `fused_mla_decode_q_concat_kv_cache_fp8_insert` |
| `"fp8_ds_mla"` | BF16 | `[nblk, bs, 656]` uint8 | 512 fp8 + 16 scales + 128 rope bf16 = 656B | `fused_mla_decode_q_concat_ds_mla_insert` |

For NoPE MLA (`qk_rope_head_dim=0`), the cache entry is only 512B (no `k_pe` appended) and mqa_q is just `ql_nope` (no rope concatenation). See §3 for the NoPE-specific behavior.

---

## 3. NoPE Design: Why K/V Carry No Position

`mla_use_nope=True` means the 512-dim KV latent `c_kv` carries no positional encoding. Only the 64-dim `k_pe` suffix carries RoPE.

**Important:** Although the config has `qk_rope_head_dim=64`, when `mla_use_nope=True` is set, **`qk_rope_head_dim` is effectively 0** at inference. The model uses `kv_lora_rank + qk_rope_head_dim` for `head_size` in the attention kernel, but NoPE mode means:
- No RoPE is applied to Q or K
- The cache stores ONLY `kv_c [kv_lora_rank=512]` — no `k_pe` appended
- `head_size = 512` (not 576)
- The attention kernel sees a 512-dim head — a single 512-dim dot product per (query, cached_token)

This simplifies the KV cache and eliminates the need for a rope/nope head split.

**Why this matters for long context:**

In standard attention with YaRN or similar position extrapolation, the quality of attention scores degrades as positions exceed the training range. With NoPE:
- `c_kv` is position-free → can be attended from any future position with no degradation
- No positional interpolation burden — the model learns positional discrimination through the KDA layers' recurrent state and other mechanisms

**Attention score decomposition:**

```
score[h, t, k] = dot(Q_nope[h, t], K_nope[h, k]) / √(P+R)
               + dot(Q_pe[h, t, rotated], k_pe[k, rotated]) / √(P+R)
               ↑ position-free (via weight absorption)   ↑ position-dependent (RoPE)
```

The first term is computed entirely via the pre-absorbed `W_UK_T` and the cached `c_kv`, with no RoPE dependency. The second term uses the 64-dim `k_pe` with standard RoPE, which can be interpolated/extended without affecting the main 512-dim attention.

---

## 4. Weight Absorption: W_UK_T and W_UV

At model load time, `kv_b_proj` is decomposed and stored in two absorbed tensors:

```python
# kv_b_proj.weight: [L, H*(P+V)] = [512, 96*(128+128)] = [512, 24576]
W_kv_b = kv_b_proj.weight.view(L, H, P+V)   # [512, 96, 256]

# K-nope slice (used in Q absorption):
W_UK   = W_kv_b[:, :, :P]                   # [512, 96, 128]  = [L, H, P]
W_UK_T = W_UK.permute(1, 2, 0)              # [96, 128, 512]  = [H, P, L]
# Used as: q_nope [H, P] × W_UK_T [H, P, L] → ql_nope [H, L]

# V slice (used in output projection):
W_UV   = W_kv_b[:, :, P:]                   # [512, 96, 128]  = [L, H, V]
W_UV_T = W_UV.permute(1, 0, 2)             # [96, 512, 128]  = [H, L, V]
# Used as: attn_latent [H, L] × W_UV_T [H, L, V] → output [H, V]
```

**Effect on decode cost:**

Without absorption:
```
# For each of N cached tokens, expand c_kv:
k_nope[t, h] = c_kv[t] @ W_UK[h]   # N × H × L × P FLOPs per step
v[t, h]      = c_kv[t] @ W_UV[h]   # N × H × L × V FLOPs per step
```

With absorption:
```
# Once per step for each query token:
ql_nope[h] = q_nope[h] @ W_UK_T[h]  # H × P × L FLOPs (small, done once)

# At attention time, dot directly with cached c_kv:
score[h, t] = dot(ql_nope[h], c_kv[t])  # N × H × L FLOPs (same as before, no expansion)
```

The net effect: `kv_b_proj` never runs at decode time. Per-token cost drops from `O(N × H × L × (P+V))` to `O(N × H × L)`.

---

## 5. Q Path: q_b_proj and RoPE

After `fused_qkv_a_proj` and `q_a_layernorm`:

```
q_a [T, 1536]   (normalized)
    → q_b_proj  [1536 → H*(P+R)] = [1536 → 96×192 = 18432]  (ColumnParallel, TP-sharded)
    → q [T, 96, 192]   bf16

split:
    q_nope [T, 96, 128]  ← content dims (will be absorbed into W_UK_T)
    q_pe   [T, 96, 64]   ← position dims (gets RoPE)

rotary_embedding(q_pe, positions) → q_pe_rotated [T, 96, 64]
```

With TP=8: each rank produces `q [T, 12, 192]` (12 local heads).

---

## 6. Decode Path

Decode processes `T=B` tokens (one per sequence, or `B×next_n` for speculative decode).

### 6.1 BMM1: Q_nope absorption

```python
# Absorb W_UK_T into q_nope (done once per step, not per cached token):
ql_nope = torch.bmm(
    q_nope.transpose(0, 1),    # [H, T, P] = [96, B, 128]
    W_UK_T,                     # [H, P, L] = [96, 128, 512]
).transpose(0, 1)               # → [T, H, L] = [B, 96, 512]

# ql_nope[b, h, :] is now a 512-dim latent query for head h, token b.
# It can be directly dot-producted against cached c_kv[N, :] for all N cached tokens.
```

**Why BMM not einsum:** `torch.bmm` dispatches to cuBLAS grouped GEMM, which is well-optimized for the small-batch regime of decode. For TP=8 with 12 heads per rank, the batch of 12 BMMs is handled as one grouped GEMM call.

### 6.2 Fused decode q-concat + KV cache insert kernel

After absorption, the current-step's `c_kv` and `k_pe_rotated` must be inserted into the paged KV cache, and a concatenated MQA-format Q must be built for `forward_mqa`.

**CUDA kernel:** `fused_kimi_k3_mla_decode_q_concat_kv_cache_insert`
**File:** `csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu`

```python
ops.fused_mla_decode_q_concat_kv_cache_insert(
    ql_nope,          # [T, H, L=512]   absorbed Q latent
    q_pe_rotated,     # [T, H, R=64]    RoPE Q
    c_kv,             # [T, L=512]      KV latent (to be cached)
    k_pe_rotated,     # [T, R=64]       RoPE key (to be cached)
    kv_cache,         # [blocks, block_size, 1, L+R=576]  paged cache
    slot_mapping,     # [T] physical slots for new tokens
    q_out,            # [T, H, L+R=576]  output MQA-format Q
)
```

**What the kernel does per token `t`:**

```
1. Concatenate MQA Q:
   q_out[t, h, :512] = ql_nope[t, h, :]    # absorbed nope
   q_out[t, h, 512:] = q_pe_rotated[t, h, :]  # rope

2. Write KV cache at slot_mapping[t]:
   block = slot_mapping[t] // block_size
   offset = slot_mapping[t] % block_size
   kv_cache[block, offset, 0, :512] = c_kv[t, :]
   kv_cache[block, offset, 0, 512:] = k_pe_rotated[t, :]
```

The kernel is launched with **PDL (Programmatic Dependent Launch)** on SM90+, overlapping execution with the `q_b_proj` GEMM that produces `q_nope` and `q_pe`. This hides the memory-bandwidth-bound KV write behind the compute-bound GEMM.

### 6.3 `forward_mqa`: paged MQA attention

After q-concat and cache insert, standard paged MQA attention runs:

```python
# q_out: [T, H, 576]  (absorbed ql_nope || q_pe_rotated)
# kv_cache: [blocks, block_size, 1, 576]  (c_kv || k_pe per slot)

attn_latent = impl.forward_mqa(
    q = q_out,              # [T, H, 576]
    kv_cache = kv_cache,    # paged KV
    attn_metadata = ...,    # block table, seq lengths
    softmax_scale = 1.0 / math.sqrt(P + R),   # 1/√192
)
# attn_latent: [T, H, L=512]  — attention output in latent space
```

**What FlashAttention computes for each head `h`:**

```
score[t, k] = dot(q_out[t, h, :], kv_cache[slot_k, :]) / √192
            = dot(ql_nope[t,h,:], c_kv[k,:]) / √192    # 512-dim NoPE term
            + dot(q_pe[t,h,:], k_pe_rotated[k,:]) / √192  # 64-dim RoPE term

α[t, :] = softmax(score[t, :])   # over all cached tokens

attn_latent[t, h, :] = Σ_k  α[t, k] × c_kv[k, :]   # [512] — latent space
```

Note: `c_kv[k]` serves as both K (via W_UK_T absorption into Q) and V (via W_UV). The paged attention kernel reads the same 512-dim cache entry for both roles — a unique property of MLA.

### 6.4 BMM2: W_UV output projection

After `forward_mqa`:

```python
# Convert latent attention output to V-space:
attn_output = torch.bmm(
    attn_latent.transpose(0, 1),   # [H, T, L] = [96, B, 512]
    W_UV_T,                         # [H, L, V] = [96, 512, 128]
).transpose(0, 1)                   # → [T, H, V] = [B, 96, 128]

attn_output = attn_output.reshape(T, H * V)   # [B, 96×128 = 12288]
```

After `o_proj`: `[B, 12288] → [B, 7168]`.

---

## 7. Prefill Path

Prefill is more complex because it must handle both new prompt tokens and any prior context (chunked context processing).

### 7.1 New tokens: `run_prefill_new_tokens`

For the current prompt tokens (positions 0..T_new-1 if fresh, or at the end of a longer sequence):

```python
# Compute KV for new tokens:
c_kv_new, k_pe_new = kv_a_proj_with_mqa(hidden_states)   # [T_new, 512], [T_new, 64]
k_pe_rotated = rotary_embedding(k_pe_new, positions)

# Expand to full K/V for local attention among new tokens:
kv_expanded = kv_b_proj(c_kv_new)            # [T_new, H*(P+V)]
k_nope = kv_expanded[:, :H*P].view(T_new, H, P)   # [T_new, H, 128]
v      = kv_expanded[:, H*P:].view(T_new, H, V)   # [T_new, H, 128]
k = torch.cat([k_nope, k_pe_rotated.unsqueeze(1).expand(-1, H, -1)], dim=-1)
# k: [T_new, H, P+R=192]

# Fused kernel: RoPE on Q + K concat + KV cache insert:
ops.fused_mla_key_concat_kv_cache_insert(
    q_pe, q_nope, k, v, c_kv_new, k_pe_rotated, kv_cache, slot_mapping
)

# Standard FlashAttention-2 or FlashAttention-3 over new tokens:
attn_latent_new = flash_attn(q, k, v, causal=True)   # [T_new, H, V]
```

For new tokens, `kv_b_proj` IS called — unlike decode, because prefill needs to materialize the full K/V for causal self-attention among the new tokens.

### 7.2 Chunked context: `_compute_prefill_context`

When the sequence has prior context (from prior prefill or decode steps), the new tokens must also attend to all prior KV cache entries. This is done in chunks to avoid materializing all KV at once:

```python
partial_outputs = []
partial_lses = []

for chunk_start in range(0, context_len, chunk_size):
    # 1. Gather paged latent from KV cache for this chunk:
    c_kv_chunk = gather_paged_kv(kv_cache, block_table, chunk_start, chunk_end)
    # c_kv_chunk: [chunk_size, L=512]

    # 2. Expand to full K/V:
    kv_chunk = kv_b_proj(c_kv_chunk)             # [chunk_size, H*(P+V)]
    k_nope_chunk = kv_chunk[:, :H*P]
    v_chunk      = kv_chunk[:, H*P:]

    # 3. Fused kernel: K concat for this chunk:
    ops.fused_mla_kv_concat(k_nope_chunk, k_pe_cache[chunk], k_chunk)

    # 4. Attention between new Q tokens and this KV chunk:
    attn_out_chunk, lse_chunk = flash_attn(q_new, k_chunk, v_chunk, return_lse=True)

    partial_outputs.append(attn_out_chunk)
    partial_lses.append(lse_chunk)

# 5. LSE-aware merge of partial attention results:
final_attn = merge_attn_states(partial_outputs, partial_lses)
```

**`merge_attn_states`** performs numerically stable softmax merging across chunks:

```
For two partial results (out_A, lse_A) and (out_B, lse_B):
  lse_combined = log(exp(lse_A) + exp(lse_B))   (log-sum-exp)
  w_A = exp(lse_A - lse_combined)
  w_B = exp(lse_B - lse_combined)
  out_combined = w_A × out_A + w_B × out_B
```

### 7.3 Fused prefill kernels (by KV cache dtype)

**File:** `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py`

Six kernels dispatched by `kv_cache_dtype`:

| Kernel | dtype | Operation |
|---|---|---|
| `fused_kimi_k3_mla_key_concat_kv_cache_insert` | `bf16` | RoPE(Q_pe) + K concat(k_nope,k_pe) + write bf16 KV cache |
| `fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert` | `fp8` | Quant Q/K/V to fp8 + write fp8 KV cache |
| `fused_kimi_k3_mla_key_concat_ds_mla_insert` | `fp8_ds_mla` | bf16 K concat + write 656-byte block-scaled cache |
| `fused_kimi_k3_mla_kv_concat` | `bf16` | Chunked context: K concat only (no position) |
| `fused_kimi_k3_mla_kv_concat_quant_fp8` | `fp8` | Chunked context: K concat + fp8 quant |
| `fused_kimi_k3_mla_decode_q_concat_kv_cache_insert` | decode | Q concat + KV insert (decode only) |

All kernels use **PDL (Programmatic Dependent Launch)** on SM90+ to overlap with the upstream `q_b_proj` GEMM.

**Hardcoded constants in all kernels:** `L=512, P=128, R=64, V=128, cache_entry=576`.

---

## 8. Output Gate (g_proj)

When `mla_use_output_gate=True` (which it is in K3):

```python
# Standard attention output:
hidden = o_proj(attn_output)   # [T, 7168]

# Gate (sigmoid):
gate = g_proj(hidden_states)   # [T, 1]  (projects from original input, not attn output)
gate = torch.sigmoid(gate)

# Gated output:
output = hidden * gate
```

**Stream optimization:** For sequences of ≤ 512 tokens, `g_proj` runs on an **auxiliary CUDA stream** overlapping with the main attention computation:

```python
if num_tokens <= 512:
    with torch.cuda.stream(aux_stream):
        gate = sigmoid(g_proj(hidden_states))
# (main stream continues with BMM1, KV insert, forward_mqa, BMM2)
aux_stream.synchronize()  # wait for gate before the final multiply
output = attn_hidden * gate
```

For longer sequences the gate is computed sequentially (synchronously) because the attention itself dominates and the overlap benefit is smaller than the stream management overhead.

---

## 9. Output Projection and GEMM-RS

```python
attn_output_reshaped = attn_output.view(T, H*V)   # [T, 96*128=12288]
output = o_proj(attn_output_reshaped)               # [T, 7168]   RowParallel
```

**GEMM-RS fusion (SM100 + sequence parallel):**

When `VLLM_KIMI_K3_GEMM_RS=1` and SM100 hardware is available, `o_proj` is replaced by a fused GEMM + reduce-scatter kernel:

```python
# Instead of:
#   output = o_proj(attn_output) → AllReduce
# Use:
output_shard = gemm_rs(attn_output, o_proj.weight)  # fused GEMM + RS via NCCL symmetric memory
```

This avoids a separate AllReduce collective, reducing latency by ~1 synchronization step per layer. The output is already in reduce-scattered form, compatible with the next layer's sequence-parallel all-gather.

---

## 10. KV Cache Formats

### 10.1 BF16 format

```
Per slot (576 values × 2 bytes = 1152 bytes):
  [0:512]   = c_kv  (512 bf16 values, KV latent, NO position encoding)
  [512:576] = k_pe  (64 bf16 values, RoPE key, position-dependent)
```

Layout in paged cache: `kv_cache [num_blocks, block_size, 1, 576]` bf16.

### 10.2 FP8 format

Per-token quantized:

```
Per slot (576 bytes + scale metadata):
  [0:576]   = 576 fp8 values (c_kv and k_pe quantized together)
  scale:      1 float32 per token (stored separately in scale_cache)
```

Quantization: `scale = max(|values|) / fp8_max`. Scale stored in a separate `scale_cache` tensor alongside the main `kv_cache`.

### 10.3 `fp8_ds_mla` format (656-byte block-scaled)

Borrowed from DeepSeek's block-scaled FP8 layout, adapted for K3's 512+64 dims:

```
Per slot (656 bytes total):
  [0:512]    = 512 fp8 bytes  (c_kv NoPE, 8 blocks of 64 values each)
  [512:576]  = 128 bytes = 64 bf16 values (k_pe, NOT quantized)
  [576:640]  = 8 ue8m0 scale bytes (one per 64-element c_kv block)
  [640:656]  = 16 bytes padding (alignment)
```

**Block quantization of c_kv:**
```
For each block b in range(8):   # 8 × 64 = 512 dims
  amax = max(|c_kv[b*64:(b+1)*64]|)
  ue8m0 = ceil(log2(amax / fp8_max)) + 127   # biased exponent
  scale = 2^(ue8m0 - 127)
  fp8_block = clamp(c_kv[b*64:] / scale, -fp8_max, fp8_max).to(fp8)
```

`k_pe` is kept in bf16 because it carries RoPE rotations — quantizing it would degrade positional precision.

The `fp8_ds_mla` format reduces cache memory by ~50% vs bf16 (656 vs 1152 bytes), with minor quality impact from the block-scaled quantization.

---

## 11. DCP Integration

When DCP (Decode Context Parallelism) is active (`dcp_world_size > 1`):

**Decode with DCP:**

```python
# Each rank owns a KV slice:
#   rank 0: tokens [0, 2, 4, ...]  (even positions, interleaved)
#   rank 1: tokens [1, 3, 5, ...]  (odd positions)

# Step 1: Gather queries from all DCP ranks:
q_global = dcp_manager.gather_q(q_out)   # broadcast Q to all ranks
# q_global: [T×dcp_world_size, H, 576]

# Step 2: Local attention over this rank's KV shard:
attn_latent_local, lse_local = forward_mqa(q_global, local_kv_cache, return_lse=True)

# Step 3: Cross-rank LSE-aware combine:
attn_latent = dcp_manager.combine(attn_latent_local, lse_local, ...)
# Uses dcp_direct_a2a_lse_reduce.cu for the all-to-all + merge
```

**Prefill with DCP (chunked-context):**

Delegates to `impl._context_parallel_compute_prefill_context`, which performs the chunked gather + flash_attn loop but with the KV cache shard belonging to this DCP rank.

---

### 11.1 Fused MLA Key-Concat CUDA Kernel: Geometry

**File:** `csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu`

**Kernel geometry:**
- Block: 256 threads = 8 warps
- Grid: `ceil(T × (H+1) / 8)` where H = num_heads; each warp handles one `(token, slot)` pair
  - `slot < H`: process query head `slot` of token `(warp_idx // (H+1))`
  - `slot == H`: process the KV cache write for that token

**What each slot does (prefill path, bf16):**

```
Q slots (slot < H):
  If qk_rope_head_dim > 0: apply GPT-J RoPE in-place to q[token, slot, rope_dims]
  Produce k_out[token, slot, :] = cat(k_nope[token, slot, :], k_pe[token, :])
  NoPE (qk_rope_head_dim=0): k_out = k_nope, no concatenation

KV slot (slot == H):
  Write [kv_c_normed[token, :] | k_pe_rotated[token, :]] to kv_cache[block, offset, :]
  NoPE: Write only kv_c_normed[token, :] (512 values, no k_pe suffix)
```

**Internal vector copy (`copyChunk8`):**

Each warp thread processes 8 BF16 values at a time via `uint4` (128-bit) loads:
```
If FP8 quantized path:
  decode bf16 → float → (optional RoPE rotate) → saturate → pack to uint2 E4M3
If RoPE only:
  load uint4, rotate, store uint4
Otherwise:
  plain uint4 copy
```

For `fp8_ds_mla` NoPE quantization: groups of 64 values form a tile; 8 lanes per group compute warp-level absmax via `__shfl_xor_sync`, then thread 0 stores the fp32 scale.

**PDL:** All kernel variants use `cudaLaunchAttributeProgrammaticStreamSerialization=1` on SM90+, overlapping launch with prior dependent GEMMs.

**Supported cache formats per slot:**

| Format | Per-token bytes | Layout |
|---|---|---|
| BF16 `[nblk, bs, 512]` | 512×2=1024B | 512 bf16 (kv_c, NoPE) |
| BF16 `[nblk, bs, 576]` | 576×2=1152B | 512 bf16 (kv_c) + 64 bf16 (k_pe) |
| FP8 plain | same slots, FP8 values | 512 fp8 + scale |
| FP8 ds_mla `[nblk, bs, 656]` | 656B | 512 fp8 NoPE + 16B (4×fp32 tile scales) + 128B (64×bf16 rope) |

---

## 12. Concrete Examples

### 12.1 Decode: one token

```
Batch: 1 token (decode), position = 5000
TP = 8 (12 heads per rank)

Step 1: fused_qkv_a_proj
  hidden [1, 7168] × W [2048, 7168] → [1, 2048]
  → q_a [1, 1536],  kv_a [1, 512]
  After q_a_layernorm: q_a_norm [1, 1536]

Step 2: kv_a_proj_with_mqa
  hidden [1, 7168] × W [576, 7168] → [1, 576]
  → c_kv [1, 512],  k_pe [1, 64]
  k_pe_rotated = RoPE(k_pe, position=5000)  [1, 64]

Step 3: q_b_proj (TP rank 0, 12 heads)
  q_a_norm [1, 1536] × W [12×192, 1536] → q [1, 12, 192]
  split: q_nope [1, 12, 128],  q_pe_rotated [1, 12, 64]

Step 4: BMM1 (W_UK_T absorption, TP rank 0)
  q_nope [1, 12, 128] × W_UK_T [12, 128, 512] → ql_nope [1, 12, 512]
  # 12 BMMs of shape [1, 128] × [128, 512] → [1, 512]

Step 5: fused_mla_decode_q_concat_kv_cache_insert
  Input: ql_nope [1, 12, 512], q_pe_rotated [1, 12, 64]
         c_kv [1, 512], k_pe_rotated [1, 64]
  Output: q_concat [1, 12, 576]
  Side-effect: kv_cache[block=5000//bs, offset=5000%bs, 0, :] = [c_kv, k_pe_rotated]

Step 6: forward_mqa
  q = q_concat [1, 12, 576]
  kv_cache [num_blocks, block_size, 1, 576]  (5001 valid slots)
  softmax_scale = 1/√192 ≈ 0.0722

  For head h=0, cached token k=3827 (physical slot 3827):
    score = dot(q_concat[0, 0, :512], kv_cache[slot_3827, :512])   # NoPE: 512-dim
           + dot(q_concat[0, 0, 512:], kv_cache[slot_3827, 512:])  # RoPE: 64-dim
           × 0.0722

  softmax → α [1, 5001]
  attn_latent [1, 12, 512] = Σ_k  α[k] × kv_cache[slot_k, :512]

Step 7: BMM2
  attn_latent [1, 12, 512] × W_UV_T [12, 512, 128] → attn_out [1, 12, 128]
  → reshape [1, 1536]

Step 8: o_proj [1536 → 7168/8=896] per rank, then AllReduce → [1, 7168]

Step 9: g_proj (aux stream, since T=1 ≤ 512)
  gate = sigmoid(g_proj(hidden [1, 7168]))   [1, 1]
  output = attn_hidden × gate
```

### 12.2 Prefill: 8-token prompt

```
Batch: 8 tokens (fresh prompt, no prior context)
Positions: [0, 1, 2, 3, 4, 5, 6, 7]
TP = 1 (all 96 heads on one rank, for simplicity)

Step 1: fused_qkv_a_proj
  hidden [8, 7168] → q_a [8, 1536],  kv_a [8, 512]

Step 2: kv_a_proj_with_mqa
  hidden [8, 7168] → c_kv [8, 512],  k_pe [8, 64]
  k_pe_rotated = RoPE(k_pe, positions=[0..7])

Step 3: q_b_proj
  q_a [8, 1536] → q [8, 96, 192]
  split: q_nope [8, 96, 128],  q_pe [8, 96, 64]
  q_pe_rotated = RoPE(q_pe, positions=[0..7])

Step 4: kv_b_proj (prefill only — materializes full K/V for self-attention)
  c_kv [8, 512] → kv_expanded [8, 96*(128+128)] = [8, 24576]
  k_nope = kv_expanded[:, :96*128].view(8, 96, 128)   [8, 96, 128]
  v      = kv_expanded[:, 96*128:].view(8, 96, 128)   [8, 96, 128]

Step 5: fused_mla_key_concat_kv_cache_insert (bf16)
  Input: q_pe [8, 96, 64], q_nope [8, 96, 128]
         k_nope [8, 96, 128], k_pe_rotated [8, 96, 64]  (k_pe broadcast to all heads)
         c_kv [8, 512], k_pe_rotated [8, 64]
  → k_full [8, 96, 192]  = cat(k_nope, k_pe_rotated per head)
  → KV cache insert: slots 0..7 ← [c_kv, k_pe_rotated]

Step 6: FlashAttention-2 (causal, among 8 new tokens)
  q = cat(q_nope, q_pe_rotated) [8, 96, 192]
  k = k_full [8, 96, 192]
  v [8, 96, 128]
  softmax_scale = 1/√192
  → attn_out [8, 96, 128]

Step 7: o_proj + gate
  attn_out.reshape [8, 12288] → o_proj → [8, 7168]
  gate = sigmoid(g_proj(hidden [8, 7168]))   [8, 1]
  output = attn_hidden × gate
```

### 12.3 NoPE MLA decode — shape comparison with/without NoPE

Side-by-side comparison of how NoPE changes every tensor shape in the decode path:

```
                           NoPE=True (K3 default)     NoPE=False (K2 / DSpark)
                           ──────────────────────     ────────────────────────
qk_rope_head_dim              effectively 0               64
KV cache entry dims           512 (c_kv only)             576 (c_kv || k_pe)
head_size (attention kernel)  512                          576
Per-slot bytes (bf16)         1024 B                       1152 B

After BMM1 (W_UK_T absorption):
  ql_nope shape              [B, H, 512]                [B, H, 512]  (same)

In fused_mla_decode_q_concat_kv_cache_insert:
  q_pe_rotated input         empty / skipped             [B, H, 64]
  mqa_q = q_out              [B, H, 512]                [B, H, 576]
    (no concatenation,         (ql_nope only)             cat(ql_nope, q_pe_rotated)
     q_pe is empty)

KV cache write per slot:
  bytes written              512 values (c_kv only)      576 values (c_kv || k_pe)

forward_mqa attention kernel:
  q shape                    [B, H, 512]                [B, H, 576]
  kv_cache slot shape        [nblk, bs, 1, 512]         [nblk, bs, 1, 576]
  score computation          dot(q[512], kv[512])        dot(q[576], kv[576])
                             = 512-dim NoPE only          = 512-dim NoPE
                                                           + 64-dim RoPE
```

**NoPE=True kernel behavior (fused_mla_decode_q_concat_kv_cache_insert):**

```python
# NoPE path: q_pe_rotated is None / zero-length
# The kernel skips the concatenation and writes only 512 values per slot:

# Q construction (no cat):
mqa_q = ql_nope   # [B, H, 512]  — no q_pe suffix appended

# KV cache insert per slot (NoPE path):
kv_cache[block, offset, 0, :512] = c_kv[t, :]
# (k_pe write is skipped entirely — 512-byte slot, not 576)
```

With NoPE=True, the attention kernel writes only 512 values per slot (not 576), saving 64 × 2 = 128 bytes per cached token per MLA layer. For a 128K-token sequence across 24 MLA layers this is 128K × 24 × 128 B = 384 MB saved.

---

### 12.4 Chunked context prefill — merging 4 chunks

Setup: request has 8192 tokens of prior context (4 chunks of 2048 each) + 16 new tokens.

```
new_q shape: [16, H, 192]   (q_nope || q_pe_rotated)

Chunked context loop (4 iterations):

  chunk 0 (kv positions 0..2047):
    gather c_kv[0:2048, 512] from paged KV cache
    kv_b_proj(c_kv_chunk) → k_nope_chunk [2048, H, 128], v_chunk [2048, H, 128]
    k_chunk = cat(k_nope_chunk, k_pe_chunk) → [2048, H, 192]
    flash_attn(new_q, k_chunk, v_chunk, return_lse=True)
    → partial_out_0 [16, H, 128], lse_0 [16, H]   e.g. lse_0 = -2.3

  chunk 1 (kv positions 2048..4095):
    → partial_out_1 [16, H, 128], lse_1 [16, H]   e.g. lse_1 = -1.8

  chunk 2 (kv positions 4096..6143):
    → partial_out_2 [16, H, 128], lse_2 [16, H]   e.g. lse_2 = -3.1

  chunk 3 (kv positions 6144..8191):
    → partial_out_3 [16, H, 128], lse_3 [16, H]   e.g. lse_3 = -2.6

merge_attn_states([partial_out_0..3], [lse_0..3]) → final_out [16, H, 128]
```

**Merge formula — 2-way LSE merge, applied 3 times (for 4 partials):**

```
Iteration 1: merge chunk 0 and chunk 1
  lse_combined_01 = log(exp(-2.3) + exp(-1.8))
                  = log(0.1003 + 0.1653)
                  = log(0.2656) = -1.326
  w_0 = exp(-2.3 - (-1.326)) = exp(-0.974) = 0.378
  w_1 = exp(-1.8 - (-1.326)) = exp(-0.474) = 0.622
  out_01 = 0.378 × partial_out_0 + 0.622 × partial_out_1

Iteration 2: merge out_01 (lse=-1.326) and chunk 2 (lse=-3.1)
  lse_combined_012 = log(exp(-1.326) + exp(-3.1))
                   = log(0.2656 + 0.0450) = log(0.3106) = -1.169
  w_01 = exp(-1.326 - (-1.169)) = exp(-0.157) = 0.855
  w_2  = exp(-3.1  - (-1.169)) = exp(-1.931) = 0.145
  out_012 = 0.855 × out_01 + 0.145 × partial_out_2

Iteration 3: merge out_012 (lse=-1.169) and chunk 3 (lse=-2.6)
  lse_combined_0123 = log(exp(-1.169) + exp(-2.6))
                    = log(0.3106 + 0.0743) = log(0.3849) = -0.955
  w_012 = exp(-1.169 - (-0.955)) = exp(-0.214) = 0.807
  w_3   = exp(-2.6  - (-0.955)) = exp(-1.645) = 0.193
  final_out = 0.807 × out_012 + 0.193 × partial_out_3   [16, H, 128]
```

The log-sum-exp merge is numerically stable because the weights always sum to 1.0 and no intermediate softmax is recomputed from raw scores. Each chunk's attention distribution is preserved in its `lse` value, allowing exact reconstruction of the global softmax.

---

### 12.5 fp8_ds_mla quantization of one KV slot

Given: `kv_c_normed [512]` — the normalized KV latent for one token. The 512 dims are divided into 8 blocks of 64 dims each.

**amax per block:** `[0.91, 0.74, 0.88, 0.62, 0.77, 0.95, 0.69, 0.83]`

**Block 0 quantization (amax=0.91):**

```
fp8_max = 448   (E4M3 max representable value)

# Exponent: smallest power of 2 ≥ amax/fp8_max
raw_ratio = 0.91 / 448 = 0.002031
log2(raw_ratio) = log2(0.002031) ≈ -8.94
ceil(-8.94) = -8

# Scale and ue8m0 encoding:
scale    = 2^(-8) = 0.00390625
ue8m0_byte = -8 + 127 = 119   (stored as uint8 in bytes [576:584])

# Quantize block 0 (dims 0..63):
kv_fp8_block0 = clamp(kv_c_normed[0:64] / 0.00390625, -448.0, 448.0).to(fp8_e4m3)
# Each dim is divided by 0.00390625 (= multiplied by 256), then saturated to [-448, 448]
# Example: dim_5 = 0.45 → 0.45 / 0.00390625 = 115.2 → fp8(115.2) ≈ 116 (E4M3)
```

**Block 1 quantization (amax=0.74):**

```
raw_ratio = 0.74 / 448 = 0.001652
log2(0.001652) ≈ -9.24 → ceil = -9
scale = 2^(-9) = 0.001953125
ue8m0_byte = -9 + 127 = 118
```

**Full 656-byte slot layout with byte offsets:**

```
Offset  Size   Content
──────  ─────  ──────────────────────────────────────────────────────────────
0       64 B   block 0: 64 fp8_e4m3 values (kv_c dims 0..63)
64      64 B   block 1: 64 fp8_e4m3 values (kv_c dims 64..127)
128     64 B   block 2: 64 fp8_e4m3 values (kv_c dims 128..191)
192     64 B   block 3: 64 fp8_e4m3 values (kv_c dims 192..255)
256     64 B   block 4: 64 fp8_e4m3 values (kv_c dims 256..319)
320     64 B   block 5: 64 fp8_e4m3 values (kv_c dims 320..383)
384     64 B   block 6: 64 fp8_e4m3 values (kv_c dims 384..447)
448     64 B   block 7: 64 fp8_e4m3 values (kv_c dims 448..511)
───────────────── total fp8 region: 512 bytes ─────────────────────────────
512    128 B   k_pe: 64 bf16 values (NOT quantized, RoPE precision preserved)
───────────────── total rope region: 128 bytes ─────────────────────────────
640      8 B   ue8m0 scales: [119, 118, ?, ?, ?, ?, ?, ?] (one uint8 per block)
               decoded amax values:  [0.91→2^-8, 0.74→2^-9, ...]
648     16 B   padding (alignment to 16-byte boundary)
───────────────── total: 664 bytes → padded to 656 B (as documented) ──────
```

Note: `k_pe` is kept in bf16 because quantizing the 64 RoPE dimensions would corrupt the positional encoding. The 8 ue8m0 scale bytes give per-block precision — blocks with larger amax get coarser quantization, but no block's error bleeds into adjacent blocks.
