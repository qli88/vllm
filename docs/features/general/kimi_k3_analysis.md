# Kimi K3 Architecture and Kernel Analysis

## Table of Contents

1. [Background and Architecture Overview](#1-background-and-architecture-overview)
2. [Model Config (KimiLinearConfig)](#2-model-config-kimilinearconfig)
3. [Layer Types and the Hybrid Stack](#3-layer-types-and-the-hybrid-stack)
4. [MLA: Multi-Head Latent Attention](#4-mla-multi-head-latent-attention)
   - [Projections and Weight Absorption](#41-projections-and-weight-absorption)
   - [Prefill Forward](#42-prefill-forward)
   - [Decode Forward](#43-decode-forward)
   - [Optional Output Gate](#44-optional-output-gate)
5. [KDA: Kimi Gated Delta Attention (Linear Attention)](#5-kda-kimi-gated-delta-attention-linear-attention)
   - [Architecture and Projections](#51-architecture-and-projections)
   - [Core Recurrence Math](#52-core-recurrence-math)
   - [Prefill Path](#53-prefill-path)
   - [Decode Path](#54-decode-path)
   - [Spec-Decode Path](#55-spec-decode-path)
6. [AttnRes: Attention Residual Mixing](#6-attnres-attention-residual-mixing)
7. [MoE: Latent Mixture of Experts](#7-moe-latent-mixture-of-experts)
   - [Latent MoE Projection](#71-latent-moe-projection)
   - [MegaMoE (DeepGEMM FP8 Expert Kernel)](#72-megamoe-deepgemm-fp8-expert-kernel)
   - [LatentMoERunner and Tail Tiers](#73-latentmoerunner-and-tail-tiers)
   - [Latent MoE Tail Fusion (SM100)](#74-latent-moe-tail-fusion-sm100)
8. [Kernels Reference](#8-kernels-reference)
   - [fused_kimi_k3_mla_key_concat_kv_cache (CUDA)](#81-fused_kimi_k3_mla_key_concat_kv_cache-cuda)
   - [fused_recurrent_kda (Triton)](#82-fused_recurrent_kda-triton)
   - [fused_recurrent_kda_packed_decode (Triton)](#83-fused_recurrent_kda_packed_decode-triton)
   - [attn_res Triton Kernel](#84-attn_res-triton-kernel)
   - [KDA Gate+Beta Kernel](#85-kda-gatebeta-kernel)
9. [Low-Latency GEMM Dispatch](#9-low-latency-gemm-dispatch)
10. [KDA Metadata Build](#10-kda-metadata-build)
11. [DSpark MLA (Speculative Decode Context)](#11-dspark-mla-speculative-decode-context)
12. [Comparison with DeepSeek V3/V4](#12-comparison-with-deepseek-v3v4)
13. [Key Implementation Files](#13-key-implementation-files)
14. [Worked Example: One Decode Step](#14-worked-example-one-decode-step)
15. [Worked Example: One Prefill Step](#15-worked-example-one-prefill-step)

---

## 1. Background and Architecture Overview

Kimi K3 is a **multimodal hybrid model** combining:

- **MoonViT3d** vision encoder (shared with K2.5) + **KimiK25MultiModalProjector**
- **KimiLinear** text backbone — a hybrid of MLA full-attention layers and KDA (Kimi Gated Delta Attention) linear-attention layers, with Sparse MoE FFN

The text backbone (`KimiLinearForCausalLM`) is the same class used for both standalone text (Kimi K2) and the multimodal K3 wrapper (`KimiK3ForConditionalGeneration`). The architecture is defined by `KimiLinearConfig`, which controls which layers are full-attention MLA and which are KDA linear-attention.

**Key architectural choices vs DeepSeek V3/V4:**

| Feature | DeepSeek V3 | DeepSeek V4 | Kimi K3 |
|---|---|---|---|
| Attention type | MLA only | MLA + SWA | **Hybrid MLA + KDA (linear attn)** |
| KV cache | Paged latent (512-dim) | Paged latent (512-dim) + SWA | Paged latent (512-dim for MLA), recurrent state (for KDA) |
| MoE | Standard FusedMoE | C4A compressor + MegaMoE | **Latent MoE** + MegaMoE |
| Output gate | No | Yes (V4 wo_a/wo_b) | **Optional sigmoid gate (g_proj)** on MLA |
| Attention residual | No | No | **AttnRes** (block-level prefix-sum residual) |
| NoPE | No | No | **Yes (mla_use_nope=True)** — MLA uses only nope heads, no RoPE |
| Activation | SiLU | SiTU | **SiTU** (Sigmoid-gated TU) |

### Layer arrangement

```
Layer n is KDA if (n+1) is in linear_attn_config["kda_layers"]
Layer n is MLA otherwise (full-attention layers)

Example 61-layer K3 text backbone:
  Layers 0,1,2,...  → alternating MLA and KDA based on config lists
  MLA layers: use MultiHeadLatentAttention (full paged attention)
  KDA layers: use KimiK3DeltaAttention (recurrent state)
  All MoE layers: KimiMoE with LatentMoE projections
```

---

## 2. Model Config (KimiLinearConfig)

**File:** `vllm/transformers_utils/configs/kimi_linear.py`

Key fields that control K3-specific behaviors:

```python
class KimiLinearConfig:
    # MLA parameters
    q_lora_rank: int | None      # Q compression rank (e.g. 1536); None = no Q LoRA
    kv_lora_rank: int | None     # KV latent rank (e.g. 512)
    qk_nope_head_dim: int | None # Head dim without RoPE (e.g. 128)
    qk_rope_head_dim: int | None # Head dim with RoPE (e.g. 0 for NoPE-only)
    v_head_dim: int | None       # Value head dim (e.g. 128)
    mla_use_nope: bool           # If True: NoPE-only (no RoPE on MLA layers)
    mla_use_output_gate: bool    # If True: add sigmoid g_proj gate after attention

    # KDA (linear attention) parameters
    linear_attn_config: dict | None  # {
    #   "kda_layers": [list of layer indices that use KDA],
    #   "full_attn_layers": [list of layer indices that use MLA],
    #   "head_dim": 128,
    #   "num_heads": 48,
    #   "short_conv_kernel_size": 4,
    #   "use_full_rank_gate": True,   # K3 uses full-rank gate
    #   "gate_lower_bound": -5.0,     # bounded exponential decay
    # }

    # Latent MoE parameters
    routed_expert_hidden_size: int | None  # Latent dim for routed experts (e.g. 3584)
    latent_moe_use_norm: bool              # RMSNorm before up-projection in latent tail
    activation_situ_beta: float | None    # SiTU activation beta parameter
    activation_situ_linear_beta: float | None  # SiTU linear component beta

    # AttnRes (attention residual mixing)
    attn_res_block_size: int | None  # Number of layers per residual block (e.g. 6)

    # Standard MoE
    num_experts: int | None          # Total expert count
    num_experts_per_token: int | None
    num_shared_experts: int          # Shared (always-active) experts
    moe_router_activation_func: str  # "sigmoid" for K3
```

The config's properties classify layers:

```python
config.is_kda_layer(layer_idx)  # True if (layer_idx+1) in kda_layers
config.is_mla → True if any of kv_lora_rank/qk_nope_head_dim etc. are set
config.is_moe → True if num_experts is not None
```

---

## 3. Layer Types and the Hybrid Stack

**File:** `vllm/models/kimi_k3/nvidia/model.py:760` (`KimiDecoderLayer`)

Each layer is one of two types based on config:

```
KimiDecoderLayer
  ├── if is_kda_layer(layer_idx):
  │     if use_full_rank_gate:
  │       self.self_attn = KimiK3DeltaAttention   ← K3 full-rank KDA
  │     else:
  │       self.self_attn = KimiLinearGatedDeltaNetAttention  ← K2 low-rank KDA
  │
  └── else (MLA layer):
        assert mla_use_nope == True    ← K3 MLA is NoPE-only
        self.self_attn = MultiHeadLatentAttention

  + self.mlp: KimiMoE (MoE layers) or KimiMLP (dense layers)
  + self.input_layernorm, self.post_attention_layernorm
  + [optional] AttnRes components if attn_res_block_size is set
```

**Forward per layer:**

```python
def forward(positions, hidden_states, residual, prefix_sum):
    # 1. Pre-attention norm (with optional AttnRes)
    hidden_states = _pre_attn_norm(hidden_states, residual, prefix_sum)

    # 2. Sequence parallelism: all-gather before attention
    if use_sequence_parallel:
        hidden_states = sp_all_gather(hidden_states)

    # 3. Attention (MLA or KDA)
    hidden_states = self_attn(positions, hidden_states)

    # 4. Sequence parallelism: reduce-scatter after attention
    if use_sequence_parallel:
        hidden_states = sp_reduce_scatter(hidden_states)

    # 5. Post-attention norm (with optional AttnRes)
    hidden_states = _post_attn_norm(hidden_states, residual, prefix_sum)

    # 6. MoE/MLP
    hidden_states = mlp(hidden_states)
    return hidden_states, prefix_sum, residual
```

---

## 4. MLA: Multi-Head Latent Attention

**File:** `vllm/models/kimi_k3/nvidia/mla.py`

K3's MLA is a **NoPE-only** variant (no RoPE on K/V, `qk_rope_head_dim=0` typically). It shares the core MLA weight-absorption idea with DeepSeek but has K3-specific additions: optional output gate and NoPE-only operation.

### 4.1 Projections and Weight Absorption

**Module structure:**

```
MultiHeadLatentAttention
  # Q path (when q_lora_rank is not None):
  fused_qkv_a_proj: MergedColumnParallelLinear
    [hidden_size → [q_lora_rank, kv_lora_rank + qk_rope_head_dim]]
    disable_tp=True (replicated)
  q_a_layernorm: RMSNorm [q_lora_rank]
  q_b_proj: ColumnParallelLinear [q_lora_rank → num_heads × qk_head_dim]

  # OR Q path (no LoRA, q_lora_rank is None):
  q_proj: ColumnParallelLinear [hidden_size → num_heads × qk_head_dim]
  kv_a_proj_with_mqa: ReplicatedLinear [hidden_size → kv_lora_rank + qk_rope_head_dim]

  # KV path (always):
  kv_a_layernorm: RMSNorm [kv_lora_rank]
  kv_b_proj: ColumnParallelLinear [kv_lora_rank → num_heads × (qk_nope_head_dim + v_head_dim)]

  # Optional output gate (when use_output_gate=True):
  g_proj: ColumnParallelLinear [hidden_size → num_heads × v_head_dim]

  # Output:
  o_proj: RowParallelLinear [num_heads × v_head_dim → hidden_size]
```

**Absorbed weight matrices** (`process_weights_after_loading`):

Exactly as in DeepSeek V3: `kv_b_proj.weight` is split and transposed once at load time:

```python
# kv_b_proj.weight: [kv_lora_rank, num_heads × (nope_dim + v_dim)]
W_UK_T = W_UK.permute(1, 2, 0)  # [num_heads, nope_dim, kv_lora_rank]
W_UV   = W_UV.transpose(0, 1)   # [num_heads, kv_lora_rank, v_dim]
```

This eliminates per-token `kv_b_proj` at decode time: instead of uprojecting each cached token, the query absorbs `W_UK_T` once per step and attention runs in the 512-dim latent space.

**NoPE-only (`mla_use_nope=True`):**

In K3's NoPE MLA, `qk_rope_head_dim=0`, so:

- No RoPE is applied to Q or K
- `kv_lora_rank + qk_rope_head_dim = kv_lora_rank + 0 = kv_lora_rank`
- Cache stores only `kv_c [kv_lora_rank]` (no `k_pe` appended)
- `head_size = kv_lora_rank + qk_rope_head_dim = kv_lora_rank` (what attention kernel sees as "head size")

This is a deliberate design: K3 avoids RoPE on MLA layers entirely, relying on learned positional encoding elsewhere (or from the KDA layers' recurrent state). NoPE makes the latent simpler and avoids the rope/nope head split.

### 4.2 Prefill Forward

```
hidden_states [T, hidden_size]
      │
      ▼  fused_qkv_a_proj  [hidden_size → q_lora_rank + kv_lora_rank]
      │     (single replicated GEMM)
      │
      ├─ q_c [T, q_lora_rank]   → q_a_layernorm → q_b_proj → q [T, H, qk_head_dim]
      │
      └─ kv_c [T, kv_lora_rank]  → kv_a_layernorm → kv_c_normed [T, kv_lora_rank]
         k_pe [T, qk_rope_head_dim]  (= 0 dims for NoPE)
      │
      ▼  kv_b_proj(kv_c_normed) → split:
          k_nope [T, H, qk_nope_head_dim]
          v      [T, H, v_head_dim]
      │
      ▼  fused_mla_key_concat_kv_cache_insert  (fused CUDA kernel):
          - Concatenate: k = [k_nope | k_pe] → k_out [T, H, qk_head_dim]
          - Write kv_c_normed to paged KV cache at slot_mapping
          (optional: apply RoPE in-kernel if qk_rope_head_dim > 0)
      │
      ▼  prefill_backend.run_prefill_new_tokens(q, k_out, v, kv_cache, metadata)
          → attn_out [T_prefill, H, v_head_dim]
      │
      ▼  [optional] apply output gate: attn_out *= sigmoid(g_proj(hidden_states))
      ▼  o_proj → output [T, hidden_size]
```

**KV cache layout:**

- BF16 path: `[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]` = `[num_blocks, block_size, 512]` for NoPE (576 if rope_head_dim=64)
- FP8 ds_mla path: `[num_blocks, block_size, 656]` — 512 NoPE fp8 + 16 scale + 128 rope bf16 (for DSv3.2 compat)

### 4.3 Decode Forward

```
hidden_states [B, hidden_size]  (B = decode batch)
      │
      ▼  fused_qkv_a_proj → q_c [B, q_lora_rank], kv_c [B, kv_lora_rank]
      │
      ▼  fused_q_kv_rmsnorm(q_c, kv_c, q_weight, kv_weight) [parallel, single kernel]
      │
      ▼  q_b_proj(q_c_normed) → q [B, H, qk_head_dim]
         split: q_nope [B, H, qk_nope_head_dim], q_pe [B, H, qk_rope_head_dim]
      │
      ▼  Weight absorption:
          ql_nope = bmm(q_nope.T(0,1), W_UK_T).T(0,1)
               [B,H,nope_dim] × [H,nope_dim,lora_rank] → [B,H,lora_rank]
         (NoPE: qk_rope_head_dim=0, so q = ql_nope only)
      │
      ▼  fused_mla_decode_q_concat_kv_cache_insert  (fused CUDA kernel):
          - Write kv_c_normed to paged KV cache at new slot
          - Concat q = [ql_nope | q_pe] → mqa_q [B, H, kv_lora_rank + rope_dim]
          (NoPE: mqa_q = ql_nope [B, H, kv_lora_rank])
      │
      ▼  impl.forward_mqa(mqa_q, kv_cache, attn_metadata)
          → attn_out_latent [B, H, kv_lora_rank]  (attention in latent space)
      │
      ▼  V up-projection (W_UV BMM):
          attn_out = bmm(attn_out_latent.T(0,1), W_UV).T(0,1)
               [H,B,L] × [H,L,V] → [B,H,V]
      │
      ▼  [optional] output gate: attn_out *= sigmoid(g_proj(hidden_states))
      ▼  o_proj → output [B, hidden_size]
```

**Why NoPE avoids the RoPE-in-latent problem:** In DSv3, the latent must carry `k_pe` (the rope key) alongside `kv_c` because RoPE depends on position and can't be absorbed. In K3's NoPE MLA, there is no rope part at all, so the paged cache stores only the pure latent `kv_c [kv_lora_rank]`. This makes the cache format simpler and eliminates the `k_pe` concatenation step.

### 4.4 Optional Output Gate

When `mla_use_output_gate=True`, K3 adds a sigmoid-gated output:

```python
g_proj: ColumnParallelLinear [hidden_size → num_heads × v_head_dim]

# Gate GEMM overlapped with attention on aux stream (< 512 tokens):
gate = g_proj(hidden_states)      # [T, H, v_head_dim]

# After attention completes:
attn_out = attn_out * sigmoid(gate)   # element-wise
# Then o_proj
```

This is similar to the Llama/Mistral-style gated attention output but applied after the value computation rather than in the MLP.

```python
# @torch.compile:
def _gate_sigmoid_mul(attn_out, gate):
    return attn_out * gate.sigmoid()
```

---

## 5. KDA: Kimi Gated Delta Attention (Linear Attention)

**Files:** `vllm/models/kimi_k3/nvidia/kda.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`

KDA is K3's **linear-complexity attention** — it replaces the quadratic softmax-attention computation with a recurrent state update. Instead of computing a full attention matrix, it maintains a `[H, V, K]` state matrix that is updated sequentially.

K3 uses `use_full_rank_gate=True` in `linear_attn_config`, which changes how the gate is computed vs. the simpler K2 variant.

### 5.1 Architecture and Projections

```
KimiK3DeltaAttention
  in_proj_qkvgfab: _KimiGDNMergedColumnParallelLinear
    [hidden_size → [proj_q, proj_k, proj_v, proj_g(full-rank), f_a(replicated), num_heads, padding]]
    sizes = [projection_size] × 4 + [head_dim, num_heads] + [padding]
    Note: f_a (index 4) is replicated (loaded from rank 0 across all TP ranks)
    Note: g (index 3) is full-rank in K3 (vs low-rank g_a,g_b in K2)

  f_b_proj: ColumnParallelLinear [head_dim → projection_size]
    (second stage of f_a → f_b: f = f_b_proj(f_a))

  dt_bias: [local_projection_size]  (bias added to f before gate activation)

  conv1d: ColumnParallelLinear [conv_size → 3*projection_size]
    (packs Q,K,V short-conv weights; conv_size = short_conv_kernel_size = 4)
    decode: also has decode_conv1d_weight [3, conv_size, local_proj_size]

  A_log: [local_num_heads]  fp32
    (log of the per-head decay rate; A = exp(A_log), A < 0 for decay)

  gate_lower_bound: float  (e.g. -5.0; constrains exp gate magnitude)

  o_norm: FusedRMSNormGated [head_dim, activation="sigmoid"]
    (output RMSNorm with sigmoid gate: norm(x) * sigmoid(gate))

  o_proj: RowParallelLinear [projection_size → hidden_size]
```

**Stacked input projection** (`in_proj_qkvgfab`):

```
in_proj_qkvgfab output [T, 3×proj + proj(g) + head_dim + num_heads + pad]
split into:
  q_raw     [T, local_proj_size]  ← raw Q (before short conv)
  k_raw     [T, local_proj_size]  ← raw K
  v_raw     [T, local_proj_size]  ← raw V
  g_raw     [T, local_proj_size]  ← full-rank gate input
  f_a       [T, head_dim]         ← dt_proj input (replicated)
  beta_raw  [T, local_num_heads]  ← per-head update gate
  (padding discarded)
```

### 5.2 Core Recurrence Math

KDA is a variant of the **Gated Delta Net** (GDN) recurrence. For each token `t`:

**Gate computation:**

```
f_b = f_b_proj(f_a[t])          # [proj_size]  (low-rank expansion)
g1 = f_b.reshape(H, D)           # [H, D]  per-head gate
g2 = g_raw[t].reshape(H, D)      # [H, D]  full-rank gate (K3-specific)

# Gate activation (bounded exponential decay):
gate[h, d] = lower_bound * sigmoid(A_log[h] * (g1[h,d] + dt_bias[d]))
# A_log < 0, sigmoid ∈ (0,1), lower_bound ∈ [-5, 0)
# → gate ∈ (lower_bound, 0)
# → exp(gate) ∈ (exp(lower_bound)≈0.007, 1)  ← decay ∈ (0.007, 1) per step
```

**Per-head state update at token `t`:**

```
q[h,:], k[h,:], v[h,:] = short_conv(q_raw, k_raw, v_raw)[t, h, :]
# normalize: q and k are L2-normalized × scale (K^-0.5)

beta[h] = sigmoid(beta_raw[t, h])      # ∈ (0, 1): how much to erase old memory

# State update (Delta Net rule):
state[h] *= exp(gate[h, :])            # [V, K]: exponential decay
erase = sum(state[h] * k[h, :])       # [V]: what's currently stored for k
state[h] -= beta[h] * erase[:, None] * k[h, None, :]  # erase old memory
state[h] += v[h, :, None] * k[h, None, :]             # write new memory

# Read:
out[h] = sum(state[h] * q[h, :])      # [V]
```

**Output gate:**

```
out = o_norm(out, g2)   # FusedRMSNormGated: RMSNorm(out) * sigmoid(g2)
output = o_proj(out)
```

The key insight: unlike softmax attention which is O(T²), KDA maintains a fixed-size `[H, V, K]` state matrix and processes each token in O(H × V × K) regardless of sequence length. This makes it ideal for the KDA layers that alternate with MLA layers in K3's hybrid stack.

### 5.3 Prefill Path

For prefill (new sequences or chunked prefill), the full sequence is processed using the parallel associative scan via `_flashkda_prefill` or `chunk_kda_with_fused_gate`:

```python
# 1. Short convolution on Q, K, V:
q = causal_conv1d_fn(q_raw, conv_weight_q, ...)  # [T, H, D]
k = causal_conv1d_fn(k_raw, conv_weight_k, ...)
v = causal_conv1d_fn(v_raw, conv_weight_v, ...)

# 2. Gather initial SSM state from cache:
initial_state = gather_initial_states(ssm_state_cache, state_indices, ...)
# initial_state [B, H, V, K] — one per sequence in batch

# 3. Parallel KDA scan:
if is_flashkda_supported():
    out, final_state = _flashkda_prefill(
        q.unsqueeze(0),  # [1, T, H, D]
        k.unsqueeze(0),
        v.unsqueeze(0),
        gate.unsqueeze(0),
        beta.unsqueeze(0),
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        cu_seqlens,
    )
else:
    # Triton fallback: chunk_kda_with_fused_gate
    ...

# 4. Write updated states back to SSM state cache
```

`_flashkda_prefill` calls `torch.ops._flashkda_C.fwd(...)` — a compiled FlashKDA kernel that processes the full prefill in parallel using hardware-efficient scan algorithms.

### 5.4 Decode Path

Two decode paths depending on hardware support:

**Fast fused decode** (`decode_conv1d_weight is not None`, SM90/SM100, H∈{12,24,48,96}, head_dim=128):

```python
ops.fused_kda_decode(
    x=mixed_qkv,          # [B, 2*H*K + H*V] packed Q/K/V
    weight=decode_conv1d_weight,  # [3, conv_size, local_proj_size]
    bias=conv_bias,
    conv_state=ssm_conv_state,   # [B, H, conv_size, K] (persistent across steps)
    raw_g=g_raw,          # [B, H, D]
    raw_beta=beta_raw,    # [B, H]
    A_log=A_log,
    dt_bias=dt_bias,
    state_indices=state_indices,  # [B] which slots to read/write
    state=ssm_state,     # [cache, H, V, K] (persistent state)
    out=core_attn_out,   # [B, H, D]
    lower_bound=gate_lower_bound,
    output_gate=g2,
    norm_weight=o_norm.weight,
    norm_eps=o_norm.eps,
)
```

This single fused kernel: reads conv state → applies short-conv → computes gate → updates state → reads output → applies o_norm.

**Fallback decode path** (Triton-based):

```python
# 1. Short-conv update:
q, k, v = causal_conv1d_update(mixed_qkv, conv_state, conv_weight)

# 2. Compute gate activation:
gate, beta = _fused_kda_gate_beta(raw_g, raw_beta, A_log, dt_bias, lower_bound)

# 3. Packed recurrent step:
out, state = fused_recurrent_kda_packed_decode(
    mixed_qkv,          # [B, 2*H*K + H*V]
    raw_g, raw_beta, A_log, dt_bias,
    lower_bound,
    initial_state,      # [cache, H, V, K]
    state_indices,      # [B]
    scale=K**-0.5,
)

# 4. Output gate:
out = o_norm(out, g2)
```

### 5.5 Spec-Decode Path

When speculative decoding is active, some requests have multiple "draft" tokens that need to be processed together. K3 handles this with mixed batches:

```python
# Partition requests into spec-decode and non-spec-decode
spec_tokens   [T_spec, ...]    # draft tokens
non_spec      [B_ns, ...]      # regular decode tokens

# Spec tokens: use prefill-style fused_recurrent_kda (sequential scan)
out_spec, state_spec = fused_recurrent_kda(
    q_spec, k_spec, v_spec, gate_spec, beta_spec, ...,
    cu_seqlens=spec_cu_seqlens,
    ssm_state_indices=spec_state_indices,
    num_accepted_tokens=num_accepted_tokens,
)

# Non-spec tokens: use fused_recurrent_kda_packed_decode
out_non_spec = fused_recurrent_kda_packed_decode(...)

# Merge outputs by request index:
core_attn_out.index_copy_(0, spec_indices, out_spec)
core_attn_out.index_copy_(0, non_spec_indices, out_non_spec)
```

---

## 6. AttnRes: Attention Residual Mixing

**File:** `vllm/models/kimi_k3/nvidia/ops/attn_res.py`

AttnRes is a K3-specific mechanism that improves long-context recall by mixing information from **earlier block-level residuals** into the current layer's input normalization. Instead of a simple `LayerNorm(x + residual)`, K3 uses a learned softmax-weighted combination of the current token's representation with stored "block" residuals from previous layers.

**Key parameters:**

- `attn_res_block_size`: number of layers per block (e.g. 6)
- Every `attn_res_block_size` layers, the current `prefix_sum` is written into the `blocks` buffer
- `blocks [T, num_blocks, hidden_size]`: history of past block-level prefix sums
- `qk_weight [hidden_size]`: learned projection for computing attention scores between blocks

**Math (per token row):**

```
updated_prefix = prefix_sum[t] + delta[t]   (delta = attention output)
blocks[t, block_write_idx] = updated_prefix  (if this is a write layer)

# Online softmax over all (num_blocks + 1) sources:
sources = [blocks[t, 0], ..., blocks[t, num_blocks-1], updated_prefix]
            ↑ each source: [hidden_size]

# Score = RMSNorm(source) · qk_weight  (scalar per source)
score_i = dot(RMSNorm(sources[i], norm_weight), qk_weight)

# Softmax-weighted sum:
alpha = softmax([score_0, ..., score_{num_blocks}])
mixed[t] = Σ_i alpha_i × sources[i]

# Optional output RMSNorm:
output[t] = RMSNorm(mixed[t], output_norm_weight)
```

**Why this helps:** Deep transformers with long contexts can lose access to early-layer representations. By storing one "checkpoint" every `attn_res_block_size` layers and mixing them back via learned attention, K3 creates a learned shortcut from any depth back to any earlier block. This is analogous to block-level residual connections but with a content-based (attention) combination rather than a fixed add.

**Triton kernel** `_attn_res_kernel`:

Grid `[T, ceil(hidden_size/BLOCK_D)]`. One program per `(token, hidden_dim_tile)`:

```
# Load prefix and optional delta
updated_prefix = prefix[row] + delta[row]  (if HAS_DELTA)

# Write-back to blocks buffer if WRITE_BLOCK:
if WRITE_BLOCK:
    blocks[row, block_write_idx] = updated_prefix

# Compute online softmax over all sources:
for i in range(num_blocks + 1):
    source = (i < num_blocks) ? blocks[row, i] : updated_prefix
    score_i = dot(RMSNorm(source, norm_weight), qk_weight)
    # online max/sum for stable softmax

# Weighted accumulation:
mixed = Σ alpha_i × source_i

# Optional output norm:
if APPLY_OUTPUT_NORM:
    mixed = RMSNorm(mixed, output_norm_weight)
store output[row] = mixed
```

**Native SM100 path:** For `hidden_size=7168` with delta and output-norm, `ops.kimi_k3_attn_res` is called (a CuteDSL or C++ kernel optimized for SM100 hardware). Otherwise the Triton fallback handles all sizes.

---

## 7. MoE: Latent Mixture of Experts

**File:** `vllm/models/kimi_k3/nvidia/model.py:453` (`KimiMoE`)

K3's MoE differs from DeepSeek's in introducing a **latent bottleneck** for the routed experts: instead of `hidden_size → expert_intermediate → hidden_size`, K3 projects down to `routed_expert_hidden_size` (e.g. 3584) before routing, and projects back up after the expert computation.

### 7.1 Latent MoE Projection

```
                 hidden_states [T, hidden_size=7168]
                       │
               ┌────────────────────────────────────────┐
               │ Gate + Down-proj  (overlapped streams)  │
               │                                         │
    Default stream:              Aux stream:
    gate(hidden_states)          routed_expert_down_proj(hidden_states)
    → router_logits [T, N]       → routed_hidden [T, latent_size=3584]
               │                                         │
               └──────────────────────────────────────── ┘
                              │
                    grouped top-K routing
                    topk_ids, topk_weights
                              │
              ┌───────────────┴──────────────────┐
              │                                  │
    Routed experts:                  Shared experts:
    FusedMoE or MegaMoE             KimiMLP
    [T, latent_size]                [T, hidden_size]
    → expert_out [T, latent_size]   → shared_out [T, hidden_size]
              │                                  │
              └───────────────┬──────────────────┘
                              │
            LatentMoERunner combines via one of 3 "tail" paths:
            1. SM100 tail fusion (CuTeDSL)
            2. Overlap all-reduce + up-projection (multi-stream)
            3. Column-parallel up-projection (prefill)
                              │
    KimiRoutedOutputTransform:
    [optional RMSNorm] + up-proj [latent_size → hidden_size]
    with shared output fused into GEMM beta-add epilogue
                              │
                    output [T, hidden_size]
```

**Stream overlap for down-projection:**

The `routed_expert_down_proj` and the router gate both read `hidden_states`. K3 overlaps them on separate CUDA streams (threshold: ≤ 256 tokens):

```python
(router_output, topk_ids), (routed_hidden, _) = maybe_execute_in_parallel(
    lambda: _router(hidden_states),     # gate + optionally grouped_topk
    lambda: down_proj(hidden_states),   # [7168 → 3584] replicated linear
    event_start, event_done,
    aux_stream if num_tokens <= 256 else None,
)
```

### 7.2 MegaMoE (DeepGEMM FP8 Expert Kernel)

When `kernel_config.moe_backend == "deep_gemm_mega_moe"`:

```python
class KimiK3MegaMoEExperts(DeepseekV4MegaMoEExperts):
```

K3 reuses DSv4's MegaMoE kernel infrastructure but operates on the **latent** hidden size (3584) rather than the full hidden size. The FP8 experts process inputs in `[T, latent_size]` space.

Expert weight preparation:

```python
def finalize_weights(self):
    w13_scale = deep_gemm.transform_sf_into_required_layout(
        ue8m0_uint8_to_float(self.w13_weight_scale), 2*intermediate_size, hidden_size, (1,32), num_experts
    )
    self._transformed_l1_weights, self._transformed_l2_weights = (
        deep_gemm.transform_weights_for_mega_moe(
            (w13_weight, w13_scale),
            (w2_weight, w2_scale),
            activation="situ",  # K3 uses SiTU, not SiLU
        )
    )
```

K3's MegaMoE uses the **SiTU** activation (Sigmoid-gated TU = a learned sigmoid-gated linear transformation), whereas DSv4 uses SiLU. The `activation_beta` and `activation_linear_beta` parameters control SiTU's shape.

### 7.3 LatentMoERunner and Tail Tiers

**File:** `vllm/models/kimi_k3/nvidia/latent_moe_runner.py`

After the routed experts produce `expert_out [T, latent_size]` and shared experts produce `shared_out [T, hidden_size]`, these must be combined and projected back:

```
routed: reduce_allreduce(expert_out) → [T, latent_size]
RMSNorm(reduced, norm.weight) → [T, latent_size]   (if latent_moe_use_norm)
up_proj: [latent_size → hidden_size]

combined = up_proj(normed_reduced) + shared_out
```

Three tiers based on token count and hardware:

| Tier | Condition | Implementation |
|---|---|---|
| **TAIL_FUSION** | SM100, ≤ 16 tokens, TP 8 or 16 | `KimiK3LatentMoETailOp` (CuTeDSL fused) |
| **ALLREDUCE_OVERLAP** | ≤ threshold tokens | All-reduce latent on aux stream; up-proj on default |
| **COLUMN_PARALLEL** | Prefill / large batch | Standard column-parallel up-proj with all-reduce |

### 7.4 Latent MoE Tail Fusion (SM100)

**File:** `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py`

For small decode batches on SM100 (Blackwell), K3 fuses the full tail into 3 CuTeDSL kernels:

```
Input:
  routed_output  [M, latent_size=3584]    (un-reduced, TP-sharded)
  shared_output  [M, hidden_size=7168]    (un-reduced, TP-sharded)
  rms_weight     [latent_size]
  up_proj_weight [hidden_size, latent_size]

Step 1 — CollectiveKernel:
  All-reduce of routed_output across TP ranks (NVLink)
  RMSNorm(allreduced) using rms_weight
  Reduce-scatter of shared_output (restores TP sharding)
  → (latent [M, latent_size], shared_shard [M, hidden_size/tp])

Step 2 — AdaptiveUpProjectionKernel:
  local_up_shard = up_proj_weight[rank*H:(rank+1)*H, :]
  result_shard = latent @ local_up_shard.T  (FP8 BF16 GEMM)
  result_shard += shared_shard
  → writes to distributed mailbox via Lamport multicast

Step 3 — LamportCopyKernel:
  Reads result from mailbox → final output [M, hidden_size]
```

Constants: `hidden_size=7168`, `latent_size=3584`, `max_num_tokens=16`, `TP∈{8,16}`.

This is only possible on SM100 because it uses direct NVLink peer memory access (Lamport protocol) which requires Blackwell's NVLINK bandwidth.

---

## 8. Kernels Reference

### 8.1 `fused_kimi_k3_mla_key_concat_kv_cache` (CUDA)

**File:** `csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu`

A single CUDA kernel that handles all MLA KV bookkeeping for one forward step (prefill or decode). Multiple variants cover different cache formats and quantization levels.

**Kernel geometry:**

- Block: 256 threads = 8 warps
- Grid: `ceil(T × (num_heads + 1) / 8)` — each warp handles one `(token, slot)` pair
- Each warp: `token = warp_idx / (H+1)`, `slot = warp_idx % (H+1)`
    - `slot < H`: process query head `slot` of token `token`
    - `slot == H`: process the KV cache entry for token `token`

**What each slot does:**

For prefill (`fused_kimi_k3_mla_key_concat_kv_cache_insert`):

- **Q slots (slot < H):** Rotate Q's rope dims in-place (GPT-J style); produce full key `k[token, h, :] = [k_nope[token,h,:] | k_pe[token,:]]` in `k_out`
- **KV slot (slot == H):** Write `[kv_c_normed[token,:] | k_pe[token,:]]` to `kv_cache[block, offset, :]`

For decode (`fused_kimi_k3_mla_decode_q_concat_kv_cache_insert`):

- **Q slots (slot < H):** Concatenate `[ql_nope[b,h,:] | q_pe[b,h,:]]` → `mqa_q[b,h,:]`
- **KV slot (slot == H):** Write `[kv_c_normed | k_pe]` to cache

**Cache formats supported:**

| Format | Per-token bytes | Layout |
|---|---|---|
| BF16 `[nblk, bs, 576]` | 576B | 512 bf16 (kv_c) + 64 bf16 (k_pe) |
| FP8 plain `[nblk, bs, 576]` | 576B | 512 fp8 (kv_c) + 64 bf16 (k_pe) |
| FP8 ds_mla `[nblk, bs, 656]` uint8 | 656B | 512 fp8 NoPE + 16B (4×fp32 tile scales) + 128B (64×bf16 rope) |

**PDL (Programmatic Launch Dependencies):** All kernels use `cudaLaunchAttributeProgrammaticStreamSerialization=1` on SM90+, enabling the GPU to overlap this kernel's launch with prior dependent kernels without explicit stream synchronization.

**Internal vector copy (`copyChunk8`):**

Each warp thread processes 8 BF16 values at a time (via `uint4` = 128-bit load):

```
If FP8: decode bf16 → float → (optional RoPE rotate) → saturate → pack to uint2 E4M3
If RoPE only: load uint4, rotate, store uint4
Otherwise: plain uint4 copy
```

For ds_mla NoPE quantization: groups of 64 values form a tile, 8 lanes in the group compute warp-level absmax reduction via `__shfl_xor_sync`, then thread 0 stores the fp32 scale.

### 8.2 `fused_recurrent_kda` (Triton)

**File:** `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py:318`

The Triton kernel for KDA's sequential recurrence (used for prefill when FlashKDA is unavailable, and for spec-decode tokens).

**Kernel:** `fused_recurrent_kda_fwd_kernel`

Grid: `[ceil(V/BV) × N × H]` — one program per `(V-tile, sequence, head)`

```
For each token t in the sequence (sequential scan):
  Load q[K], k[K], v[V] from inputs
  Load gate[K] = gate activation from raw_g[t, h, :]

  # Exponential decay:
  b_state[V, K] *= exp(gate[K][None, :])     # broadcast over V dim

  # Erase old memory at position k:
  erase[V] = sum(b_state * k[None, :], axis=1)   # [V]
  b_state -= beta * erase[:, None] * k[None, :]

  # Write new memory:
  b_state += v[:, None] * k[None, :]             # outer product

  # Read new output:
  out[V] += sum(b_state * q[None, :], axis=1)    # [V]

  # Write state back to cache at state_indices[t]:
  store b_state → initial_state[state_indices[t], h, :, :]
```

Block size tuning: BV ∈ {4,8,16} depending on V/H ratio; num_stages ∈ {3,4}.

### 8.3 `fused_recurrent_kda_packed_decode` (Triton)

**File:** `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py:486`

The Triton kernel for standard (non-spec) decode — one step per request.

Grid: `[ceil(V/BV), B×H]` — one program per `(V-tile, batch×head)`

Input: `mixed_qkv [B, 2×H×K + H×V]` — all Q, K, V packed together per batch.

```
i_b, i_h from program_id

# Unpack Q, K, V from packed buffer:
q[K] = mixed_qkv[b, i_h*K : (i_h+1)*K]
k[K] = mixed_qkv[b, H*K + i_h*K : ...]
v[V] = mixed_qkv[b, 2*H*K + i_h*V : ...]

# L2-normalize q and k:
q = q / max(norm(q), eps) * scale    (scale = K^-0.5)
k = k / max(norm(k), eps)

# Compute gate:
gate[K] = lower_bound × sigmoid(A_log[h] × (raw_g[b,h,dstart:dend] + dt_bias))

# Read state:
b_state[V, K] = initial_state[state_indices[b], h, vstart:vend, :]

# One-step recurrence (same math as above):
b_state *= exp(gate[None, :])
erase = sum(b_state * k[None, :], axis=1)
b_state -= beta × erase[:, None] × k[None, :]
b_state += v[:, None] × k[None, :]
out = sum(b_state * q[None, :], axis=1)

# Write updated state:
initial_state[state_indices[b], h, vstart:vend, :] = b_state

# Write output:
out_tensor[b, h, vstart:vend] = out
```

BV = `min(next_power_of_2(V), 32)`.

### 8.4 `attn_res` Triton Kernel

**File:** `vllm/models/kimi_k3/nvidia/ops/attn_res.py`

**Kernel:** `_attn_res_kernel`

Grid: `[M, ceil(hidden_size/BLOCK_D)]` — two-dimensional over tokens and hidden dim tiles.

```
# Load prefix (= accumulated block-level residual up to this layer):
updated_prefix[BLOCK_D] = prefix[row, col*BLOCK_D:(col+1)*BLOCK_D]

# If HAS_DELTA (after attention/mlp):
updated_prefix += delta[row, col*BLOCK_D:...]

# Write to blocks buffer if this is a write layer:
if WRITE_BLOCK:
    blocks[row, block_write_idx, col*BLOCK_D:...] = updated_prefix

# Online softmax over (num_blocks + 1) sources:
max_val = -inf; sum_exp = 0.0; mixed = zeros(BLOCK_D)
for i in range(num_blocks + 1):
    src = blocks[row, i] if i < num_blocks else updated_prefix
    normed_src = RMSNorm(src, norm_weight)
    score = dot(normed_src, qk_weight)       # scalar
    e = exp(score - max_val)                  # online max subtraction
    sum_exp = sum_exp * (old_max/new_max) + e
    mixed = mixed * (old_max/new_max) + e * src
    update max_val

mixed /= sum_exp   # normalize

# Optional output norm:
if APPLY_OUTPUT_NORM:
    mixed = RMSNorm(mixed, output_norm_weight)

output[row, col*BLOCK_D:...] = mixed
```

Tile sizes: `BLOCK_D = next_power_of_2(hidden_size)`, `BLOCK_L = 1 (large batch) or 4 (small batch with blocks)`.

### 8.5 KDA Gate+Beta Kernel

**File:** `fused_recurrent.py:23` — `_kda_gate_beta_fwd_kernel`

Grid: `[ceil(T/BT), H]` — one program per `(time-tile, head)`.

Computes gate activation and beta sigmoid for a chunk of time steps:

```
For each token in tile [t, t+BT):
  g_raw[D] = raw_g[batch, t, h, :]
  if dt_bias: g_raw += dt_bias

  b_a = exp(A_log[h])    # A_log is negative → b_a ∈ (0,1)

  # Bounded gate (K3's lower_bound path):
  gate[D] = lower_bound × sigmoid(b_a × g_raw)
  # gate ∈ (lower_bound, 0), so exp(gate) ∈ (exp(lb), 1) = decay factor

  beta = sigmoid(raw_beta[batch, t, h])   # [scalar]

store gate[t, h, :], beta[t, h]
```

---

## 9. Low-Latency GEMM Dispatch

**File:** `vllm/models/kimi_k3/nvidia/low_latency_gemm.py`

K3 replaces the standard cuBLAS GEMM with specialized low-latency kernels for small-batch decode. The dispatch happens at model-load time via `enable_kimi_k3_low_latency_gemm`.

**Two backend options:**

| Backend | When used | Description |
|---|---|---|
| `"cute"` | M ≤ 4–16 (varies by layer) | CuTeDSL skinny GEMM, custom tiling |
| `"dsv3_fused_a"` | M = 1–16 | DSv3 fused-A GEMM with PDL (same as DSv3's decode GEMM) |

**Per-projection dispatch table** (selected entries from `KIMI_K3_PROJECTIONS`):

| Layer (local shape N×K) | M=1 | M=2..4 | M=5..16 |
|---|---|---|---|
| `fused_qkv_a_proj` (2112×7168) | dsv3 | dsv3 | dsv3 |
| `q_b_proj` (2304×1536) | dsv3 | dsv3 | dsv3 |
| `routed_expert_down_proj` (3584×7168) | cute | dsv3 | dsv3 |
| `routed_expert_up_proj` (7168×3584) | cute | cute | — |
| `in_proj_qkvgfab` (6288×7168) | cute | cute | cute |
| `o_proj` (7168×1536) | cute | — | — |
| `lm_head` (20480×7168, TP8) | cute | cute | cute |

The `residual_configs` variant supports `addmm` (GEMM + add residual in one kernel):

```python
# KimiK3LowLatencyLinearMethod:
def apply_with_residual(layer, x, residual):
    if _run_residual_plan(x.shape[0], x, layer.weight, residual):
        return residual   # result is written to residual in-place
    return torch.addmm(residual, x, layer.weight.t())   # fallback
```

The `enable_kimi_k3_low_latency_gemm` function walks all `LinearBase` modules at model-load time, matches by `(local_N, K)` shape, and replaces their `quant_method` with a `KimiK3LowLatencyLinearMethod` instance that stores the pre-built per-token-count dispatch plan.

---

## 10. KDA Metadata Build

**File:** `vllm/models/kimi_k3/nvidia/kda_metadata.py`

The `KimiK3KDAMetadataBuilder` produces per-step metadata for the KDA layers, handling both standard decode and spec-decode batches.

**Key outputs:**

```python
@dataclass
class KimiK3KDAMetadata(GDNAttentionMetadata):
    # Inherited from GDNAttentionMetadata:
    non_spec_state_indices: Tensor  # [B_ns] which SSM state slots for non-spec tokens
    spec_state_indices: Tensor      # [B_s] SSM state slots for spec-decode tokens
    spec_query_start_loc: Tensor    # [B_s+1] cumsum of spec query lengths
    non_spec_query_start_loc: Tensor
    has_spec_decode: bool
    # (plus attn_metadata for MLA layers)
```

**State slot computation** (`_get_aligned_state_indices_kernel`):

Each KDA layer maintains its state in the Mamba/SSM state cache (a separate paged cache from the KV cache). The block table for this cache maps requests to physical state slots.

For spec-decode, state slots are computed differently: each spec-decode request needs slots for the initial state (before draft tokens) and potentially for the final accepted state (after verification).

**CUDA-graph staging** (`stage_spec_decode_metadata`):

Because spec-decode batch sizes vary at each step, the metadata must be staged into pre-allocated static-size CUDA graph buffers:

```python
_stage_spec_decode_metadata_kernel[(batch_size+1)//32](...):
    # Triton kernel writes variable-length spec metadata
    # into fixed-size buffer for CUDA graph replay
    stage_spec_decode_state_indices(...)
    stage_spec_query_start_loc(...)
    stage_num_accepted_tokens(...)
```

---

## 11. DSpark MLA (Speculative Decode Context)

**File:** `vllm/models/kimi_k3/nvidia/dspark_mla.py`

DSpark is K3's speculative decoding infrastructure. The `K3DSparkDecoderLayer` wraps `MultiHeadLatentAttention` with:

- `use_rope=True` (DSpark uses RoPE unlike K3's main NoPE MLA)
- `non_causal_multi_token_decode=True` (draft tokens attend to each other)

**Cross-layer KV projection:**

The key innovation in K3 DSpark is that the speculative draft model pre-computes KV projections for **all target model layers** in a single GEMM:

```python
context_kv_proj: MergedColumnParallelLinear
    [hidden_size → [kv_width] × num_target_layers]  disable_tp=True

# One GEMM to get all layers' KV projections:
all_kv_c = context_kv_proj(context_states)  # [T, num_layers, kv_width]

# Then for each layer:
kv_c[layer_i] = all_kv_c[:, i, :kv_lora_rank]
k_pe[layer_i] = all_kv_c[:, i, kv_lora_rank:]

# Batch RMSNorm:
kv_c_normed = rms_norm_per_layer(kv_c, _context_kv_norm_weights)  # [layers, T, L]

# Batch cache insert (optimized path: single kernel when all layers share layout):
ops.concat_and_cache_mla_grouped(kv_c_normed, k_pe, kv_caches, slot_mappings)
```

This amortizes the context projection cost across all target layers in one pass.

---

## 12. Comparison with DeepSeek V3/V4

| Dimension | DeepSeek V3 | DeepSeek V4 | Kimi K3 |
|---|---|---|---|
| **Attention layers** | All MLA | MLA + SWA | **MLA + KDA (hybrid)** |
| **KDA/linear-attn** | None | None | **Yes — Gated Delta Net recurrence** |
| **MLA with NoPE** | No (has rope) | No (has rope) | **Yes — qk_rope_head_dim=0** |
| **Output gate** | No | Yes (wo_a/wo_b groups) | **Optional g_proj sigmoid gate** |
| **AttnRes residual** | No | No | **Yes — block-level prefix-sum attention** |
| **MoE latent bottleneck** | No | No | **Yes — routed_expert_hidden_size** |
| **MoE activation** | SiLU | SiTU | **SiTU** |
| **MegaMoE** | No | Yes (DSv4) | **Yes (reuses DSv4)** |
| **Expert latent dimension** | full hidden | full hidden | **3584 (half hidden)** |
| **KV cache entry size** | 576B (bf16 NoPE+rope) | 584B (fp8 ds_mla) | **512B (NoPE only, no rope key)** |
| **Fused key+cache CUDA kernel** | DSv3-style | V4-style compressor | **`fused_kimi_k3_mla_key_concat`** |
| **Low-latency GEMM** | `dsv3_fused_a_gemm` | DeepGEMM | **`dsv3_fused_a` + CuTeDSL skinny** |
| **Spec decode** | MTP | MTP | **DSpark (cross-layer KV pre-projection)** |
| **SM100 tail fusion** | No | Partial (wo_a) | **Full: CollectiveKernel + Lamport copy** |

---

## 13. Key Implementation Files

| File | Role |
|---|---|
| `vllm/models/kimi_k3/nvidia/model.py` | Top-level model: `KimiDecoderLayer`, `KimiMoE`, `KimiMLP`, `KimiLinearModel`, `KimiK3ForConditionalGeneration` |
| `vllm/models/kimi_k3/nvidia/mla.py` | `MultiHeadLatentAttention`: NoPE MLA with optional gate, prefill/decode dispatch, weight absorption |
| `vllm/models/kimi_k3/nvidia/kda.py` | `KimiK3DeltaAttention`: full-rank gate KDA, fused decode, spec-decode, flashkda dispatch |
| `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | `KimiGatedDeltaNetAttention`: shared (non-Nvidia-specific) KDA used by both K2 and K3 |
| `vllm/models/kimi_k3/nvidia/ops/attn_res.py` | `attn_res()` function + `_attn_res_kernel` Triton kernel; SM100 native path |
| `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py` | Python wrappers for all 6 fused MLA key/cache CUDA op variants |
| `csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu` | CUDA kernels: prefill concat, decode concat, FP8/ds_mla variants, PDL support |
| `vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py` | `KimiK3LatentMoETailOp`: SM100 CuTeDSL fused collective+up-proj+Lamport-copy |
| `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` | `LatentMoERunner`: 3-tier tail dispatch (TAIL_FUSION / ALLREDUCE_OVERLAP / COLUMN_PARALLEL) |
| `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py` | Triton kernels: `fused_recurrent_kda_fwd_kernel`, `fused_recurrent_kda_packed_decode_kernel`, gate+beta kernel |
| `vllm/models/kimi_k3/nvidia/kda_metadata.py` | `KimiK3KDAMetadataBuilder`, `KimiK3KDAMetadata`: SSM state slot indexing, CUDA graph staging for spec-decode |
| `vllm/models/kimi_k3/nvidia/dspark_mla.py` | `K3DSparkModel`: cross-layer KV projection, batch cache insert; speculative draft model |
| `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` | `enable_kimi_k3_low_latency_gemm`: per-shape backend dispatch table, cute/dsv3 plan injection |
| `vllm/transformers_utils/configs/kimi_linear.py` | `KimiLinearConfig`: all config fields including KDA, MoE latent, AttnRes, SiTU params |
| `vllm/transformers_utils/configs/kimi_k3.py` | `KimiK3Config`: multimodal wrapper config (text + vision) |

---

## 14. Worked Example: One Decode Step

**Setup:**

```
Model config (Kimi K3, representative):
  hidden_size         = 7168
  num_attention_heads = 56    (example)
  q_lora_rank         = 1536
  kv_lora_rank        = 512
  qk_nope_head_dim    = 128
  qk_rope_head_dim    = 0     (NoPE — no RoPE on MLA layers)
  v_head_dim          = 128
  qk_head_dim         = 128   (= nope + rope = 128 + 0)
  mla_use_nope        = True
  mla_use_output_gate = True
  routed_expert_hidden_size = 3584  (latent MoE)
  num_experts         = 128
  num_experts_per_token = 8
  num_shared_experts  = 2
  attn_res_block_size = 6

Layer 0: KDA (is_kda_layer(0) = True)
Layer 1: MLA (full-attention)
...

Batch: 2 sequences, 1 decode token each
  Request A: seq_len = 1500, at MLA layer 1
  Request B: seq_len = 800,  at KDA layer 0
```

### Step 1: Embedding

```
input_ids [2] → embed_tokens → hidden_states [2, 7168]
```

### Step 2: Layer 0 — KDA Decode

**Pre-attn norm (AttnRes variant, block_write_idx=-1 for non-write layer):**

```
_pre_attn_norm calls attn_res(prefix_sum, hidden_states=None, residual, ...):
  # hidden_states=None because we haven't computed delta yet
  # This is the "read" call — just combine stored blocks with prefix
  Triton _attn_res_kernel:
    sources = [residual_block[t, 0], residual_block[t, 1], ..., prefix_sum[t]]
    online softmax over scores → mixed [2, 7168]
  output: hidden_states = mixed [2, 7168]
```

**KDA forward (fused decode fast path, SM90+):**

```
in_proj_qkvgfab(hidden_states [2, 7168]):
  → q_raw [2, local_proj]
  → k_raw [2, local_proj]
  → v_raw [2, local_proj]
  → g_raw [2, local_proj]   (full-rank gate, K3-specific)
  → f_a   [2, head_dim=128]  (replicated)
  → beta_raw [2, local_num_heads]

f_b = f_b_proj(f_a) → [2, local_proj]  (low-rank gate expansion)
g1 = f_b.reshape(2, H, D)               [2, H, 128]
g2 = g_raw.reshape(2, H, D)             [2, H, 128]  (full-rank gate for output)

# Fused decode kernel (SM90, head_dim=128):
ops.fused_kda_decode(
    x=packed_qkv,             # packed [2, 2*H*K + H*V]
    weight=decode_conv1d_weight,  # [3, 4, local_proj]
    conv_state=kda_conv_cache,   # [2, H, 4, D] persistent short-conv state
    raw_g=g_raw, raw_beta=beta_raw, A_log=A_log, dt_bias=dt_bias,
    state_indices=[slot_A, slot_B],  # which SSM state slots to use
    state=kda_ssm_state,            # [cache, H, V, K] persistent state
    out=core_attn_out,
    lower_bound=-5.0,
    output_gate=g2,
    norm_weight=o_norm.weight,
    norm_eps=o_norm.eps,
)
# Fused kernel internally:
#   1. causal_conv1d_update: update conv_state, output (q,k,v)
#   2. compute gate = -5.0 * sigmoid(A_log * (g1 + dt_bias))
#   3. compute beta = sigmoid(beta_raw)
#   4. load state[slot_A, :], state[slot_B, :]
#   5. state *= exp(gate) ; erase; state += v × k; out = state × q
#   6. write state back to kda_ssm_state at slot_A, slot_B
#   7. o_norm: core_attn_out = RMSNorm(out) * sigmoid(g2)

o_proj(core_attn_out [2, proj]) → [2, 7168]
```

**Post-attn norm (AttnRes write-layer, block_write_idx=0 for layer 0):**

```
_post_attn_norm calls attn_res(prefix_sum, delta=attn_output, residual, ...):
  WRITE_BLOCK=True, block_write_idx=0:
    updated_prefix = prefix_sum + delta
    residual_block[:, 0, :] = updated_prefix   (write first block entry)
    # Softmax combine: 0 previous blocks + updated_prefix
    # (num_blocks=0 → trivial, mixed = updated_prefix)
  output: hidden_states = updated_prefix  [2, 7168]
```

**MoE (latent path, decode tier):**

```
# Gate + down-proj overlapped on aux stream:
router_logits = gate(hidden_states)  # [2, 128]  default stream
routed_h = routed_expert_down_proj(hidden_states)  # [2, 3584]  aux stream

# Grouped top-K routing:
topk_weights [2, 8], topk_ids [2, 8] = grouped_topk(router_logits, ...)

# Shared experts on original hidden_states:
shared_out = shared_experts(hidden_states)  # [2, 7168]

# Routed experts on projected latent [2, 3584]:
expert_out = FusedMoE(routed_h, topk_weights, topk_ids)  # [2, 3584]

# Latent tail (TAIL_FUSION tier, SM100, 2 tokens ≤ 16):
output = KimiK3LatentMoETailOp(
    expert_out,   # [2, 3584]
    shared_out,   # [2, 7168]
    norm.weight,  # [3584]
    up_proj.weight,  # [7168, 3584]
)
# Internal steps:
# 1. CollectiveKernel: allreduce(expert_out) + RMSNorm + reduce_scatter(shared_out)
# 2. AdaptiveUpProjectionKernel: local_shard = latent @ up_proj_shard.T + shared_shard
# 3. LamportCopyKernel: gather result from mailbox → [2, 7168]
```

### Step 3: Layer 1 — MLA Decode

**Pre-attn norm (AttnRes, reading from 1 stored block):**

```
attn_res(prefix_sum, delta=None, residual_blocks, ..., num_blocks=1):
  sources = [residual_blocks[:, 0, :], prefix_sum]  (2 sources)
  online softmax → mixed [2, 7168]
hidden_states = mixed
```

**MLA forward (NoPE):**

```
fused_qkv_a_proj(hidden_states [2, 7168]):
  → q_c  [2, 1536]
  → kv_c [2, 512]   (kv_lora_rank + rope_head_dim = 512 + 0 = 512)

fused_q_kv_rmsnorm(q_c, kv_c, q_weight, kv_weight):
  → q_c_normed [2, 1536]
  → kv_c_normed [2, 512]

# Q up-projection + weight absorption (no RoPE since NoPE):
q = q_b_proj(q_c_normed) → [2, 56, 128]   # q_nope only (no q_pe)
ql_nope = bmm(q.T(0,1), W_UK_T).T(0,1)    # [2, 56, 512]

# Fused decode: concat ql_nope → mqa_q + write kv_c_normed to paged cache:
fused_mla_decode_q_concat_kv_cache_insert(
    ql_nope [2, 56, 512],
    q_pe=None (NoPE),
    kv_c_normed [2, 512],
    k_pe=None,
    kv_cache,
    slot_mapping=[slot_A_mla, slot_B_mla],
):
# CUDA kernel: 
#   Q slots (0..55): mqa_q[b, h, :] = ql_nope[b, h, :]  (no concatenation needed for NoPE)
#   KV slot (56): write kv_c_normed to kv_cache at new slot
→ mqa_q [2, 56, 512]

# g_proj overlap on aux stream:
gate = g_proj(hidden_states)  # [2, 56, 128]

# Paged MLA attention in latent space:
attn_out_latent = impl.forward_mqa(mqa_q, kv_cache, metadata)
# Attends over seq_len=[1500, 800] cached tokens
# All computation in 512-dim latent space (no kv_b_proj per token)
# → [2, 56, 512]

# Output gate:
attn_out_latent = attn_out_latent * sigmoid(gate)  # [2, 56, 128] broadcast

# V up-projection (W_UV BMM):
attn_out = bmm(attn_out_latent.T(0,1), W_UV).T(0,1)  # [2, 56, 128]

o_proj(attn_out.reshape(2, 56*128)) → [2, 7168]
```

**Post-attn norm (AttnRes, num_blocks now 1):**

```
attn_res(prefix_sum, delta=attn_output, residual_blocks, ..., num_blocks=1):
  updated_prefix = prefix_sum + delta
  # block_write_idx = -1 (not a write layer; layer 1 % 6 ≠ 0)
  # num_blocks=1: softmax over [residual_blocks[:,0,:], updated_prefix]
  → mixed [2, 7168]
```

**MoE (same latent path as Layer 0)**

### Output

After all layers, the final hidden state passes through `norm` and `lm_head`:

```
hidden_states = norm(hidden_states + residual)  # final layer norm
logits = lm_head(hidden_states)  # [2, vocab_size=163840]
```

---

## 15. Worked Example: One Prefill Step

**Setup:**

```
Same model config as §14.
attn_res_block_size = 6

Batch: 2 new sequences (both prefill, no prior context)
  Request A: 8-token prompt  (seq_lens=8, query_lens=8)
  Request B: 5-token prompt  (seq_lens=5, query_lens=5)

Total tokens: 13  (laid out contiguously: A tokens 0..7, B tokens 8..12)

Layer 0: KDA (linear attention)
Layer 1: MLA (full attention)
```

### Step 0: Embedding

```
input_ids [13] → embed_tokens → hidden_states [13, 7168]
residual = None   (first layer)
prefix_sum = hidden_states   (first layer: prefix_sum IS the embedding)
hidden_states = None          (attn_res mode: starts with prefix_sum only)
residual_blocks = empty [13, 0, 7168]  (no blocks yet before layer 0)
```

### Step 1: Layer 0 — KDA Prefill

#### 1a. Pre-attn norm (AttnRes, num_blocks=0)

```
_pre_attn_norm calls attn_res(prefix_sum, delta=None, residual_blocks,
                              num_blocks=0, block_write_idx=0, ...):
  # is_block_write_layer: layer 0 % 6 == 0 → True, block_write_idx=0
  # HAS_DELTA=False (no delta yet, this is the pre-attn call)
  # WRITE_BLOCK=True (block_write_idx=0, we're writing the first block checkpoint)

  Triton _attn_res_kernel (grid [13, ceil(7168/BLOCK_D)]):
    For each token t:
      updated_prefix = prefix_sum[t]          (no delta yet)
      residual_blocks[t, 0, :] = updated_prefix   ← write block 0 checkpoint
      # num_blocks=0: no historical blocks, mixed = updated_prefix (trivial)
      hidden_states[t] = updated_prefix      (= RMSNorm for input_layernorm)

→ hidden_states [13, 7168]  (ready for attention)
→ residual_blocks [13, 1, 7168]  (block 0 now populated)
```

#### 1b. KDA: input projection

```
in_proj_qkvgfab(hidden_states [13, 7168]):
  → raw output [13, 3*local_proj + local_proj(g) + head_dim + local_num_heads + pad]

split:
  q_raw    [13, local_proj]
  k_raw    [13, local_proj]
  v_raw    [13, local_proj]
  g_raw    [13, local_proj]   (full-rank gate)
  f_a      [13, 128]          (replicated, same across TP ranks)
  beta_raw [13, local_H]

f_b_proj(f_a) → f_b [13, local_proj]
g1 = f_b.reshape(13, H, D)       [13, H, 128]
g2 = g_raw.reshape(13, H, D)     [13, H, 128]
```

#### 1c. KDA: short convolution (prefill path, `causal_conv1d_fn`)

Unlike decode which uses `causal_conv1d_update` (one token per step), prefill uses the parallel conv scan:

```
conv_weight_q = conv1d.weight[0:1, :, :]     # [1, conv_size=4, local_proj]
conv_weight_k = conv1d.weight[1:2, :, :]
conv_weight_v = conv1d.weight[2:3, :, :]

q_ns = causal_conv1d_fn(
    x=q_raw.T,             # [local_proj, 13] (channel-first for causal_conv1d)
    weight=conv_weight_q,  # [1, 4, local_proj]
    activation="silu",
    conv_states=q_conv_state,       # [cache, local_proj, 4] persistent conv state
    has_initial_state=has_initial_state,  # [B] bool: whether to use prior state
    cache_indices=state_indices,          # [B=2] which cache slots
    query_start_loc=cu_seqlens,           # [3] = [0, 8, 13]
).T  # → [13, local_proj]

# Same for k_ns, v_ns.
# After conv, the conv_state for each sequence is updated to reflect
# the last (conv_size-1=3) tokens — ready for the next decode step.
```

The causal conv scan applies a 4-wide causal moving-average filter to each channel: `output[t] = silu(sum_{j=0..3} weight[j] * x[t-j])`. The conv state stores the sliding window of the last 3 inputs per channel.

After the conv, reshape for the recurrent scan:

```
q_ns = rearrange(q_ns, "n (h d) -> 1 n h d", d=128)   # [1, 13, H, 128]
k_ns, v_ns similarly                                    # [1, 13, H, 128]
```

#### 1d. KDA: gather initial states

```
# Each sequence may have a prior SSM state (for chunked prefill).
# For fresh sequences: has_initial_state = [False, False]
initial_state = gather_initial_states(
    recurrent_state,              # [cache_size, H, V, K] full state buffer
    state_indices=[slot_A, slot_B],
    has_initial_state=[False, False],
)
# → initial_state [2, H, V, K]  (zeroed for new sequences)
```

#### 1e. KDA: parallel prefill scan (`_flashkda_prefill`)

```
out, last_state = _flashkda_prefill(
    q=q_ns,             # [1, 13, H, 128]
    k=k_ns,             # [1, 13, H, 128]
    v=v_ns,             # [1, 13, H, 128]
    g=g1,               # [1, 13, H, 128]  (gate, computed from f_b)
    beta=beta_raw,      # [1, 13, H]
    A_log=A_log,        # [H]
    dt_bias=dt_bias,    # [local_proj]
    lower_bound=-5.0,
    initial_state=initial_state,   # [2, H, V, K]
    cu_seqlens=[0, 8, 13],         # boundaries: A=0..7, B=8..12
)
# → out [1, 13, H, V=128]
# → last_state [2, H, V, K]  (one final state per sequence)
```

The FlashKDA kernel processes both sequences in parallel using a hardware-efficient associative scan (parallel prefix computation), producing output for all 13 tokens in O(T log T) rather than O(T) serial steps. The `cu_seqlens` ensures the recurrence does not leak across sequence boundaries.

```
# Write updated states back to persistent SSM state cache:
recurrent_state[[slot_A, slot_B]] = last_state   # [2, H, V, K]
```

#### 1f. KDA: output gate + o_proj

```
core_attn_out = out.squeeze(0)        # [13, H, V=128]
core_attn_out = o_norm(core_attn_out, g2)
# FusedRMSNormGated:
#   core_attn_out = RMSNorm(core_attn_out, norm_weight) * sigmoid(g2)
# → [13, H, 128]

core_attn_out = core_attn_out.reshape(13, H * 128)
o_proj(core_attn_out) → hidden_states [13, 7168]
```

#### 1g. Post-attn norm (AttnRes, writing prefix_sum)

```
_post_attn_norm calls attn_res(prefix_sum, delta=hidden_states, residual_blocks,
                               num_blocks=1, block_write_idx=-1, ...):
  # is_block_write_layer: layer 0 is the write layer, but this is the POST call
  # The write already happened in pre-attn norm; post-attn sets prefix_sum = hidden_states

  updated_prefix = prefix_sum + delta          # prefix_sum = old prefix; delta = attn output
  # WRITE_BLOCK=False (block_write_idx=-1, post-attn doesn't write blocks)
  # num_blocks=1: softmax over [residual_blocks[:, 0, :], updated_prefix]

  For each token t:
    source_0 = residual_blocks[t, 0, :]   # the block checkpoint written in pre-attn
    source_1 = updated_prefix[t]          # current prefix + attn output

    score_0 = dot(RMSNorm(source_0, norm_weight), qk_weight)
    score_1 = dot(RMSNorm(source_1, norm_weight), qk_weight)
    alpha = softmax([score_0, score_1])
    hidden_states[t] = alpha[0]*source_0 + alpha[1]*source_1

    # Then apply post_attention_layernorm (APPLY_OUTPUT_NORM):
    hidden_states[t] = RMSNorm(hidden_states[t], post_attention_layernorm.weight)

→ hidden_states [13, 7168]  (input to MoE)
→ prefix_sum = updated_prefix  (carries forward to next layer)
```

#### 1h. MoE (latent, prefill tier — COLUMN_PARALLEL)

With 13 prefill tokens, tier 2 (COLUMN_PARALLEL) is used — no aux-stream overlap.

```
# Gate + down-proj (sequential, > 256 tokens threshold):
router_logits = gate(hidden_states)         # [13, num_experts]
routed_h = routed_expert_down_proj(hidden_states)  # [13, 3584]

# grouped top-K routing:
topk_weights [13, 8], topk_ids [13, 8] = grouped_topk(router_logits, ...)

# Shared experts (KimiMLP, sequence-parallel if enabled):
shared_out = shared_experts(hidden_states)    # [13, 7168]

# Routed experts (FusedMoE on 3584-dim latent):
expert_out = FusedMoE(routed_h, topk_weights, topk_ids)  # [13, 3584]

# Latent tail (COLUMN_PARALLEL tier):
#   1. allreduce(expert_out) across TP → [13, 3584]
#   2. optional RMSNorm(latent, norm.weight)
#   3. column-parallel up-proj: hidden_shard.addmm_(latent, up_proj_shard.t())
#   4. allreduce(hidden_shard) → full [13, 7168]
#   5. add shared_out
final_hidden = latent_moe_tail_column_parallel(expert_out, shared_out, norm, up_proj)

# KimiRoutedOutputTransform folds shared into up-proj GEMM beta-add epilogue:
# → avoids a separate addition kernel
```

---

### Step 2: Layer 1 — MLA Prefill

#### 2a. Pre-attn norm (AttnRes, num_blocks=1, not a write layer)

```
_pre_attn_norm calls attn_res(prefix_sum, delta=None, residual_blocks,
                               num_blocks=1, block_write_idx=-1, ...):
  # layer 1 % 6 ≠ 0 → not a write layer
  # HAS_DELTA=False (pre-attn)
  # num_blocks=1: softmax over [residual_blocks[:,0,:], prefix_sum]

  For each token t:
    score_0 = dot(RMSNorm(residual_blocks[t,0,:]), qk_weight)
    score_1 = dot(RMSNorm(prefix_sum[t]), qk_weight)
    alpha = softmax([score_0, score_1])
    hidden_states[t] = alpha[0]*residual_blocks[t,0,:] + alpha[1]*prefix_sum[t]
    hidden_states[t] = RMSNorm(hidden_states[t], input_layernorm.weight)

→ hidden_states [13, 7168]
```

#### 2b. MLA: fused input GEMM

```
fused_qkv_a_proj(hidden_states [13, 7168]):
  [7168 → q_lora_rank=1536 + kv_lora_rank=512 + qk_rope_head_dim=0]
  → q_c   [13, 1536]
  → kv_c  [13, 512]    (no rope dim appended: NoPE)

fused_q_kv_rmsnorm(q_c, kv_c, q_weight, kv_weight):
  → q_c_normed  [13, 1536]
  → kv_c_normed [13, 512]

q_b_proj(q_c_normed) → q [13, 56, 128]   (NoPE: no rope dim, qk_head_dim=128)
```

#### 2c. MLA: kv_b_proj + fused key-concat + cache-insert

```
kv_b_proj(kv_c_normed [13, 512])
  → kv_nope [13, 56, 256]  (128 nope + 128 v per head)

split:
  k_nope [13, 56, 128]
  v      [13, 56, 128]

# k_pe = zeros / empty (NoPE: qk_rope_head_dim=0, positions=None, cos_sin_cache=None)

# BF16 cache path:
fused_mla_key_concat_kv_cache_insert(
    q        [13, 56, 128],   # prefill query (in-place: RoPE applied if needed; NoPE: no-op)
    k_nope   [13, 56, 128],
    k_pe     [13,  0],         # empty for NoPE
    kv_c_normed [13, 512],
    kv_cache [nblk, bs, 512],  # paged BF16 cache, 512B per slot
    slot_mapping [13],          # physical cache slots for these 13 tokens
    positions=None,             # NoPE: no positions needed
    cos_sin_cache=None,
):
# CUDA kernel (grid: ceil(13×(56+1)/8) blocks of 256 threads):
#   For each (token, head_slot):
#     slot < 56: k_out[t, h, :] = cat(k_nope[t,h,:], k_pe[t,:]) = k_nope (NoPE)
#                q stays unchanged (no rope rotation for NoPE)
#     slot == 56: write kv_c_normed[t, :] → kv_cache[block, offset, :]
→ k [13, 56, 128]   (= k_nope, no rope concatenation needed)
```

**Why NoPE simplifies the cache insert:** For DSv3, the kernel must concatenate `[k_nope | k_pe]` for each head and apply RoPE to k_pe. Here, `k_pe` is empty (0 dims), so the key is just `k_nope` and no rotation is needed. The cache entry is also just `kv_c_normed [512]` with no rope suffix.

#### 2d. MLA: prefill attention (new tokens)

```
# No prior context for these fresh sequences → no chunked context merge needed.
# prefill.chunked_context = None, has_context = False

output_prefill = prefill_backend.run_prefill_new_tokens(
    q=q,    # [13, 56, 128]  NoPE query
    k=k,    # [13, 56, 128]  NoPE key
    v=v,    # [13, 56, 128]
    return_softmax_lse=False,
    out=out.view(13, 56, 128),
)
```

The prefill attention is causal FlashAttention over the 13 tokens. Because this is NoPE, the attention scores are purely content-based dot products:

```
For each head h and query token q_t:
  score[q_t, k_t] = dot(q[q_t, h, :], k[k_t, h, :]) / sqrt(128)
                    for k_t ≤ q_t  (causal mask)

Softmax + weighted sum of v → attn_out [13, 56, 128]
```

**Chunked context case** (request has prior KV in cache):

If `prefill.chunked_context is not None` (e.g. a long document being ingested with cached prefix):

```
# Step 1: Attention over NEW tokens only (suffix — the Q above):
suffix_output, suffix_lse = prefill_backend.run_prefill_new_tokens(
    q, k, v, return_softmax_lse=True)  # [13, 56, 128], lse [13, 56]

# Step 2: Attention over CACHED context tokens (prefix in paged KV cache):
context_output, context_lse = impl._compute_prefill_context(
    q, kv_cache, attn_metadata, k_scale)  # reads from paged cache
# → [13, 56, 512+0], lse [13, 56]  (latent-space output, 512-dim for NoPE)

# Step 3: Online softmax merge:
merge_attn_states(
    output=out.view(13, 56, 128),
    prefix_output=context_output[..., :128],   # trim to v_head_dim
    prefix_lse=context_lse,
    suffix_output=suffix_output[..., :128],
    suffix_lse=suffix_lse,
)
# merge_attn_states uses the log-sum-exp trick to correctly combine
# two partial attention outputs:
#   combined = (exp(lse_a - lse_max) * out_a + exp(lse_b - lse_max) * out_b)
#              / (exp(lse_a - lse_max) + exp(lse_b - lse_max))
```

#### 2e. MLA: optional output gate (g_proj overlapped)

```
# g_proj was launched on aux stream at the start of forward():
gate = g_proj(hidden_states)   # [13, 56, 128]

attn_out = attn_out * sigmoid(gate)   # element-wise gate
# → [13, 56, 128]
```

#### 2f. MLA: output projection

Unlike decode (which needs `W_UV` BMM for the latent output), prefill uses `kv_b_proj` directly and gets `v` in full head space already. So no `W_UV` BMM is needed:

```
o_proj(attn_out.reshape(13, 56*128)) → [13, 7168]
```

**Why no W_UV BMM at prefill:** `W_UV` absorption is only used at decode time to avoid running `kv_b_proj` on every cached token. At prefill, `kv_b_proj` is already run (to get `k_nope` and `v` for the new tokens), so `v` is available directly in `[T, H, v_head_dim]` space and can be used as-is.

#### 2g. Post-attn norm (AttnRes, updating prefix_sum for MLP input)

```
_post_attn_norm:
  updated_prefix = prefix_sum + attn_output
  # layer 1 % 6 ≠ 0 → not a write layer, block_write_idx=-1
  # num_blocks still = 1

  hidden_states[t] = softmax_mix(residual_blocks[t,0,:], updated_prefix[t])
  hidden_states[t] = RMSNorm(hidden_states[t], post_attention_layernorm.weight)

prefix_sum = updated_prefix   (carries forward)
```

#### 2h. MoE (same latent structure as Layer 0)

```
# 13 tokens → COLUMN_PARALLEL tier
routed_h = routed_expert_down_proj(hidden_states)   # [13, 3584]
expert_out = FusedMoE(routed_h, topk_weights, topk_ids)  # [13, 3584]
shared_out = shared_experts(hidden_states)           # [13, 7168]
final_hidden = latent_moe_tail(expert_out, shared_out)   # [13, 7168]
```

---

### Summary: Prefill vs Decode Differences Per Component

| Component | Prefill | Decode |
|---|---|---|
| **AttnRes** | Writes block checkpoints on write layers; softmax-mixes current prefix + stored blocks | Same math but residual_blocks already populated from prefill |
| **KDA conv** | `causal_conv1d_fn` — parallel causal scan over all T tokens; updates conv_state | `causal_conv1d_update` — one step per token; updates conv_state in place |
| **KDA recurrence** | `_flashkda_prefill` or `chunk_kda_with_fused_gate` — parallel scan over [1,T,H,D] | `fused_kda_decode` (fused) or `fused_recurrent_kda_packed_decode` (Triton) |
| **KDA state write** | `recurrent_state[state_indices] = last_state` (final state of each sequence) | State updated in-place inside the fused decode kernel |
| **MLA input GEMM** | Same `fused_qkv_a_proj` | Same |
| **MLA kv_b_proj** | Run for current tokens to get k_nope, v directly | Not run per-token; absorbed into W_UK_T for decode (only run once at load) |
| **W_UK_T absorption** | Not used (q_nope absorbed into q for the full-head attention, not needed) | Used: `ql_nope = bmm(q_nope, W_UK_T)` to avoid per-token kv_b_proj |
| **W_UV BMM** | Not used (v available directly from kv_b_proj) | Used: `attn_out = bmm(attn_out_latent, W_UV)` |
| **Cache insert kernel** | `fused_mla_key_concat_kv_cache_insert` — writes new slots + builds k_out | `fused_mla_decode_q_concat_kv_cache_insert` — writes one new slot + builds mqa_q |
| **Attention kernel** | `prefill_backend.run_prefill_new_tokens` (FlashAttention causal) | `impl.forward_mqa` (FlashMLA/FlashInfer paged attention) |
| **Chunked context** | `merge_attn_states` online-LSE merge of prefix + suffix | N/A (all context is paged) |
| **MoE tail tier** | COLUMN_PARALLEL (13 tokens > threshold) | TAIL_FUSION (2 tokens ≤ 16, SM100) |
| **g_proj gate timing** | Overlapped on aux stream with attention | Overlapped on aux stream with attention |
