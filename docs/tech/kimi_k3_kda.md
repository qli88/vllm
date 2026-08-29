# Kimi-K3: KDA (Kimi Delta Attention)

[← Index](kimi_k3_index.md) | [Layer Types](kimi_k3_layers.md) | [End-to-End](kimi_k3_e2e.md)

KDA is K3's linear attention mechanism — a Gated Delta Net that replaces quadratic attention at 69 of the 93 layers. This doc covers the full algorithm, all three weight projections, the fused decode CUDA kernel, Triton prefill paths, KDA metadata/state management, and RecoverSSM for speculative decoding.

---

## Table of Contents

1. [What Is KDA?](#1-what-is-kda)
2. [Input Projection: in_proj_qkvgfab](#2-input-projection-in_proj_qkvgfab)
3. [Short Conv1d State](#3-short-conv1d-state)
   - 3.1 [Full in_proj_qkvgfab Decomposition (g1 vs g2 gates)](#31-full-in_proj_qkvgfab-decomposition)
4. [Gated Delta Net Recurrence](#4-gated-delta-net-recurrence)
5. [Output Projection](#5-output-projection)
6. [Decode Path: Fused CUDA Kernel](#6-decode-path-fused-cuda-kernel)
   - 6.1 [Kernel parameters and dispatch](#61-kernel-parameters-and-dispatch)
   - 6.2 [Algorithmic steps inside the kernel](#62-algorithmic-steps-inside-the-kernel)
   - 6.3 [Template parameters](#63-template-parameters)
   - 6.4 [Three Decode Dispatch Paths](#64-three-decode-dispatch-paths)
7. [Prefill Path: Triton Chunked Linear Attention](#7-prefill-path-triton-chunked-linear-attention)
   - 7.1 [chunk.py — chunked recurrent attention](#71-chunkpy--chunked-recurrent-attention)
   - 7.2 [chunk_intra.py — intra-chunk parallelism](#72-chunk_intrapy--intra-chunk-parallelism)
   - 7.3 [fused_recurrent.py — short sequence path](#73-fused_recurrentpy--short-sequence-path)
8. [KDA State: Format and Sizing](#8-kda-state-format-and-sizing)
9. [KDA Metadata Builder (NVIDIA)](#9-kda-metadata-builder-nvidia)
   - 9.1 [State alignment kernel](#91-state-alignment-kernel)
   - 9.2 [PDL scheduling](#92-pdl-scheduling)
   - 9.3 [KDA Metadata: Spec-Decode State Slots](#93-kda-metadata-spec-decode-state-slots)
10. [KDA Gate+Beta Kernel](#10-kda-gatebeta-kernel)
11. [RecoverSSM: State Replay for Speculative Decoding](#11-recoverssm-state-replay-for-speculative-decoding)
12. [Concrete Examples](#12-concrete-examples)
    - 12.1 [Decode: conv1d state update](#121-decode-conv1d-state-update)
    - 12.2 [Decode: delta net memory update](#122-decode-delta-net-memory-update)
    - 12.3 [Gate computation — concrete numbers for one head](#123-gate-computation--concrete-numbers-for-one-head)
    - 12.4 [Delta Net state update — tracking one memory location over 3 steps](#124-delta-net-state-update--tracking-one-memory-location-over-3-steps)
    - 12.5 [Prefill vs decode cost for one KDA layer at seq_len=4096](#125-prefill-vs-decode-cost-for-one-kda-layer-at-seq_len4096)

---

## 1. What Is KDA?

KDA (Kimi Delta Attention) is a **Gated Delta Network** — a form of linear (recurrent) attention that processes sequences in `O(N)` time by maintaining a fixed-size associative memory state.

**Comparison with MLA:**

| Property | MLA (full attention) | KDA (linear attention) |
|---|---|---|
| Complexity per step | O(N) — reads all N cached tokens | O(1) — updates fixed state |
| Memory per sequence | N × 576 bytes/layer (grows with N) | ~3.1 MB/layer fixed |
| Expressive power | Exact attention over all past tokens | Approximate via low-rank state |
| Used at layers | 0, 4, 8, …, 92 (every 4th) | 1-3, 5-7, …, 89-91 (every 3 of 4) |

**Gated Delta Net recurrence:**

At each input token position `t`, KDA maintains two state tensors:
1. **Conv1d state** `h_conv [H, kw, kDimK]` — a short sliding window over recent tokens
2. **Delta memory** `S [H, kDimK, kDimV]` — an outer-product associative memory

The update rule (delta rule):
```
S_t = S_{t-1} + β_t × (v_t - S_{t-1} @ k_t) × k_t^T
```
where `β_t` is a learned per-token scalar, `k_t` and `v_t` are normalized key/value vectors, and `S_{t-1} @ k_t` is the "predicted" value retrieved from memory. The delta term `(v_t - S_{t-1} @ k_t)` is the prediction error — the memory is updated to store the residual.

---

## 2. Input Projection: `in_proj_qkvgfab`

A single **full-rank gate** fused projection of `hidden_states`:

```python
in_proj_qkvgfab: Linear [D=7168 → H*(kDimK + kDimK + kDimV + 1 + 1 + kDimK + 1)]
  = [7168 → 96*(128 + 128 + 128 + 1 + 1 + 128 + 1)]
  = [7168 → 96*387]
  = [7168 → 37152]
```

**Why `use_full_rank_gate=True`:** A full-rank gate means `in_proj_qkvgfab` has full hidden-size input. The alternative (low-rank gate) would compress input first. Full-rank is more expressive for gating but uses more parameters.

**Output splits per head h:**

| Slice | Symbol | Dim | Meaning |
|---|---|---|---|
| `q` | query | 128 | Key for associative retrieval |
| `k` | key | 128 | Key for memory update |
| `v` | value | 128 | Value to store |
| `g` | gate | 1 | Sigmoid output gate |
| `f_a` | forget α | 1 | Alpha parameter for conv1d gate |
| `b` | beta | 128 | Per-head, per-dim update scale |
| `beta_scalar` | β | 1 | Scalar update strength |

For each head `h`:
```python
out = in_proj_qkvgfab(hidden_states)   # [T, 37152]
# Reshape to [T, H, 387]:
q       = out[:, h, 0:128]             # [T, 128]
k       = out[:, h, 128:256]           # [T, 128]
v       = out[:, h, 256:384]           # [T, 128]
g       = out[:, h, 384]               # [T]  sigmoid gate
f_a     = out[:, h, 385]               # [T]  conv1d gate
b       = out[:, h, 386:514]           # [T, 128]  (note: actual slice based on kDimK)
beta    = out[:, h, 514]               # [T]  scalar
```

Note: the exact split indices depend on `kDimK=kDimV=128`. The gate lower bound is `-5.0` — `f_a` is clamped to `max(f_a, -5.0)` before sigmoid to prevent saturation.

---

## 3. Short Conv1d State

Before the delta-rule recurrence, `k` and `v` are passed through a **causal short conv1d** with kernel width 4:

```
h_conv: [B, H, kw=4, kDimK=128]  — circular buffer of last 4 k/v values

At each new token t:
  h_conv = roll(h_conv, shift=-1, dim=2)   # evict oldest
  h_conv[:, :, -1, :] = k[t]              # insert new k
  k_filtered[t] = sum(h_conv × conv_weight)  # depthwise conv with learned weights
```

The conv1d weights `decode_conv1d_weight [H, kw, kDimK]` are hardcoded kernel weights — no bias. This short-window filter smooths the key/value representations before they enter the associative memory.

**Why conv1d before delta rule:** Raw token embeddings can be noisy. The conv1d acts as a learned local smoother (analogous to local attention) before the global state update, improving the quality of `k_t` and `v_t` that get written into memory.

---

### 3.1 Full `in_proj_qkvgfab` Decomposition

The full split of `in_proj_qkvgfab` output `[T, 3*local_proj + local_proj(g) + head_dim + local_num_heads + pad]`:

```
mixed_qkv [T, 3*local_proj]  ← raw Q/K/V packed (all before short conv)
g_raw     [T, local_proj]    ← full-rank g2 gate input (K3-specific)
f_a       [T, head_dim=128]  ← dt input for g1 computation (replicated across TP)
beta_raw  [T, local_num_heads] ← per-head beta (memory write strength)
(padding discarded)
```

**Two distinct gates — g1 vs g2:**

| Gate | Source | Role |
|---|---|---|
| `g1` | `f_b_proj(f_a)`, reshaped `[T, H, D]` | **Key gate** — controls exponential state decay. `gate = lower_bound × sigmoid(A_log[h] × (g1 + dt_bias))`. The state decays at rate `exp(gate)`. |
| `g2` | `g_raw` from in_proj, reshaped `[T, H, D]` | **Output gate** — applied after reading from state: `o_norm(output, g2) = RMSNorm(output) × sigmoid(g2)`. |

**Additional learned parameters:**

```python
f_b_proj: ColumnParallelLinear [head_dim=128 → local_proj]
    # Expands the replicated f_a scalar embedding into a per-head gate vector g1.
    # Weight: [local_proj, head_dim], TP-sharded on output.

dt_bias: Parameter [local_proj]  fp32
    # Additive bias on f_b output before gate activation.

A_log: Parameter [local_num_heads]  fp32
    # Log of per-head decay rate. Always negative (enforced during training).
    # gate[h, d] = lower_bound × sigmoid(A_log[h] × (g1[h,d] + dt_bias[d]))
    # A_log < 0 → A_log*sigmoid() < 0 → gate ∈ (lower_bound, 0)
    # → exp(gate) ∈ (exp(-5)≈0.007, 1.0) — per-step decay ∈ [0.7%, 100%]

o_norm: FusedRMSNormGated [head_dim, activation="sigmoid"]
    # Applied after recurrence read:
    # out = RMSNorm(core_out) × sigmoid(g2)
```

**Why `f_a` is replicated (not TP-sharded):** `f_a` feeds `f_b_proj` which expands the `head_dim=128` scalar to `local_proj` values. Since `f_b_proj` must see the full `head_dim` input on every TP rank (to produce the per-head gate), `f_a` cannot be sharded — it is loaded from rank 0 and broadcast.

**Why K3 uses full-rank g2 vs K2's low-rank:** K2 uses `g_a_proj + g_b_proj` (low-rank factorization) for the output gate. K3 uses `g_raw` directly from `in_proj_qkvgfab` (full-rank, `hidden_size → local_proj`). Full-rank g2 gives K3 more expressive output gating at the cost of more parameters.

---

## 4. Gated Delta Net Recurrence

After conv1d filtering, the delta-rule update:

**Step 1 — Normalize keys:**
```
k_normalized = RMSNorm(k_filtered)   # [kDimK=128]  (element-wise, no learned weight)
```

**Step 2 — Retrieve from memory:**
```
retrieved = S @ k_normalized          # [kDimV=128]  inner product over kDimK
```

**Step 3 — Compute delta (prediction error):**
```
delta = v - retrieved                 # [kDimV=128]
```

**Step 4 — Update memory:**
```
beta_signed = sigmoid(beta_scalar) × b    # [kDimV]  effective per-dim update gate
S_new = S + beta_signed[:, None] × outer(delta, k_normalized)
      = S + (beta_signed × delta)[:, None] × k_normalized[None, :]
```

**Step 5 — Output:**
```
output_t = S_new @ q_normalized          # [kDimV]  query retrieval from updated memory
output_t = sigmoid(g) × output_t         # gated output
```

The output then goes through `out_proj [H*kDimV → D]`.

**Memory dimensions:**
```
S: [H=96, kDimK=128, kDimV=128]   ← 96 × 128 × 128 = 1,572,864 float32 values = ~6 MB per sequence
```

The entire past context is compressed into this fixed-size matrix per head.

---

## 5. Output Projection

```python
# After delta net, output per head: [T, 128]
# Concatenate across heads: [T, 96*128=12288]
output = torch.cat([outputs_per_head], dim=-1)   # [T, 12288]
output = out_proj(output)                          # [T, 7168]  RowParallel
```

`out_proj: RowParallelLinear [H*kDimV, D] = [12288, 7168]`, TP-sharded.

---

## 6. Decode Path: Fused CUDA Kernel

For decode (single token per sequence), a single CUDA kernel fuses the entire KDA forward: conv1d update → delta memory update → output.

**File:** `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu`

### 6.1 Kernel parameters and dispatch

```python
ops.fused_kda_decode(
    hidden_states,         # [B, D=7168]   input tokens
    in_proj_weight,        # [37152, 7168]  fused QKVGFABeta projection
    out_proj_weight,       # [7168, 12288]  output projection
    conv1d_weight,         # [H, kw=4, kDimK=128]  conv state weights
    conv_state,            # [B, H, kw=4, kDimK=128]  conv1d buffer (in-place updated)
    delta_state,           # [B, H, kDimK=128, kDimV=128]  (in-place updated)
    gate_lower_bound=-5.0, # clamp for f_a
)
# Returns: [B, D=7168]
```

### 6.2 Algorithmic steps inside the kernel

The kernel processes one token per sequence (`B` sequences in parallel). Each CUDA block handles one sequence `b`:

**Block layout:** `blockDim.x = kThreads` (typically 256), one block per sequence `b`.

```
1. Load in_proj_weight rows for heads 0..(H-1) from global memory.
2. Compute in_proj_output = hidden_states[b] @ in_proj_weight.T
   → q, k, v, g, f_a, b_vec, beta per head.

3. For each head h (parallelized across threads):

   3a. Conv1d update:
       conv_state[b, h] = roll(conv_state[b, h], -1)   # shift buffer
       conv_state[b, h, -1, :] = k[h, :]              # insert new k
       k_filtered = dot(conv_state[b, h], conv1d_weight[h])   # [kDimK]

   3b. Normalize k:
       rms = sqrt(sum(k_filtered²) / kDimK + eps)
       k_norm = k_filtered / rms

   3c. Gate clamp:
       f_a_clamped = max(f_a[h], gate_lower_bound=-5.0)

   3d. Delta memory retrieve and update:
       retrieved = delta_state[b, h] @ k_norm      # [kDimV]
       delta = v[h] - retrieved
       gate = sigmoid(beta[h]) × b_vec[h]           # [kDimV] per-dim gate
       update = outer(gate × delta, k_norm)         # [kDimK, kDimV]... wait:
       # Note: outer product is (kDimV, kDimK) then transposed:
       # delta_state[b,h] += gate_delta[:, None] × k_norm[None, :]
       # where gate_delta = gate × delta  [kDimV]
       delta_state[b, h] += gate_delta[None, :] × k_norm[:, None]  # [kDimK, kDimV]

   3e. Output retrieval:
       q_norm = q[h] / rms(q[h])
       output_h = delta_state[b, h].T @ q_norm   # [kDimV]
       output_h = sigmoid(g[h]) × output_h        # gated

4. Concatenate output_h across heads → [H*kDimV=12288]
5. Apply out_proj: output → [D=7168]
```

### 6.3 Template parameters

The kernel is templated on:

| Parameter | NVIDIA | AMD (ROCm) |
|---|---|---|
| `kFixedHeads` | Compile-time head count | Same |
| `kApplyOnorm` | Whether to apply output RMSNorm | Same |
| `kUpdateConvState` | Whether to update conv1d state | Same |
| `kUseLowerBound` | Whether to apply f_a clamp | Same |
| `kApplyBetaSigmoid` | Whether beta is pre-sigmoided | Same |
| `kLanes` (CUDA) | 8 (quarter-warp) | 16 (half-wavefront) |
| `kGroups` | `kThreads / kLanes` = 32 (CUDA) | = 16 (ROCm) |

This template specialization allows the compiler to unroll loops and eliminate branches, giving near-optimal performance for the fixed `kDimK=kDimV=128` case.

---

### 6.4 Three Decode Dispatch Paths

The `_forward()` method selects among three decode sub-paths in priority order:

**Path 1 — Fused decode kernel** (fastest; requires SM90/SM100/SM120, `num_heads ∈ {12,24,48,96}`, `head_dim=128`, `conv_width=4`, bf16, no spec-decode):

```python
ops.fused_kda_decode(
    x=mixed_qkv[:num_actual_tokens],    # [B, 3*local_proj] packed Q/K/V
    weight=self.decode_conv1d_weight,    # [3, conv_size=4, local_proj] width-major
    bias=self.conv1d.bias,              # optional
    conv_state=conv_state,              # [cache, local_proj, conv_size-1] — in-place updated
    raw_g=g1,                           # [1, B, H, D] key gate input
    raw_beta=beta,                      # [1, B, H]
    A_log=self.A_log,
    dt_bias=self.dt_bias,
    state_indices=state_indices[:B],    # [B] which SSM state cache slots
    state=recurrent_state,              # [cache, H, V, K] — in-place updated
    out=core_attn_out[:, :B],          # [1, B, H, D] written in-place
    lower_bound=self.gate_lower_bound,
    output_gate=g2[:B],                 # [B, H, D] full-rank output gate
    norm_weight=self.decode_norm_weight,
    norm_eps=self.o_norm.eps,
)
# This single kernel fuses:
# conv_state update → short-conv(Q,K,V) → gate activation → state decay+erase+write
# → output read → RMSNorm(output) × sigmoid(g2)
# Note: o_norm is NOT called separately — the fused kernel handles it.
```

**Path 2 — Spec-decode path** (when `has_spec_decode=True`, mixed batch):

Spec-decode mixes regular single-step tokens with multi-step draft tokens. The batch is partitioned:

```python
# Non-spec tokens (regular decode, index m.non_spec_request_indices):
causal_conv1d_update(mixed_qkv_ns, conv_state, conv_weights,
    conv_state_indices=non_spec_state_indices, out=packed_conv_out_ns)
out_ns, _ = fused_recurrent_kda_packed_decode(mixed_qkv_ns, ...)

# Spec tokens (multi-step, index m.spec_request_indices):
causal_conv1d_update(mixed_qkv_spec, conv_state, conv_weights,
    conv_state_indices=spec_conv_indices,
    num_accepted_tokens=m.num_accepted_tokens,
    query_start_loc=spec_query_start_loc)
out_spec, last_state = fused_recurrent_kda(
    q_spec, k_spec, v_spec, g1_spec, beta_spec,
    initial_state=gather_initial_states(recurrent_state, spec_state_indices, ...),
    cu_seqlens=spec_cu_seqlens,
)
recurrent_state[spec_state_indices] = last_state

# Merge by index:
core_attn_out[non_spec_mask] = out_ns
core_attn_out[spec_mask]     = out_spec
```

**Path 3 — Plain fallback decode** (Triton, no fused kernel — gfx942, non-supported SM, etc.):

```python
# 1. Update conv state and get packed Q/K/V:
causal_conv1d_update(mixed_qkv, conv_state, conv_weights,
    activation="silu", conv_state_indices=state_indices, out=packed_conv_out)

# 2. Single-step recurrent update for all requests:
out, _ = fused_recurrent_kda_packed_decode(
    mixed_qkv=packed_conv_out,   # [B, 3*local_proj]  mixed Q+K+V
    raw_g=g1,
    raw_beta=beta,
    A_log=self.A_log,
    dt_bias=self.dt_bias,
    lower_bound=self.gate_lower_bound,
    initial_state=recurrent_state,  # [cache, H, V, K] updated in-place
    state_indices=state_indices,
    scale=head_dim**-0.5,
)

# 3. Output gate (separate from the recurrence kernel):
core_attn_out.copy_(self.o_norm(out, g2))   # FusedRMSNormGated
```

---

## 7. Prefill Path: Triton Chunked Linear Attention

For prefill (multiple tokens), the delta net is computed using chunked Triton kernels that process the sequence in blocks.

**Backend selection (NVIDIA):**

```python
if backend == "cuda":
    # Use compiled CUDA FlashKDA kernel (fastest on NVIDIA):
    output, delta_state = ops.fused_kda_fwd(
        q, k, v, g, f_a, b_vec, beta, initial_state
    )
elif backend == "triton" or backend == "auto":
    # Triton chunked linear attention:
    from ops.third_party.kda.chunk import chunk_gated_delta_rule
    output, delta_state = chunk_gated_delta_rule(
        q, k, v, f=f_a, b=b_vec, beta=beta, g=g,
        initial_states=initial_state,
        chunk_size=64,
    )
```

### 7.1 `chunk.py` — chunked recurrent attention

**File:** `nvidia/ops/third_party/kda/chunk.py`

Processes the sequence in chunks of `chunk_size=64` tokens. For each chunk:

1. **Intra-chunk:** Compute attention within the chunk (analogous to FlashAttention intra-chunk).
2. **Inter-chunk:** Update the delta memory state across chunk boundaries.

The chunk size is a tradeoff between memory bandwidth (larger chunks = fewer state reads) and occupancy (smaller chunks = more parallelism).

**Key autotuning parameters:**

| Parameter | NVIDIA | AMD |
|---|---|---|
| `NUM_WARPS_AUTOTUNE` | `[4, 8, 16, 32]` | `[2, 4, 8, 16]` (capped at 16) |
| `num_stages=4` | Included | **Excluded** (causes LDS race on gfx950) |
| `_RECOMPUTE_W_U_NUM_STAGES` | `[2, 3, 4]` | `[2, 3]` |

The gfx950 `num_stages=4` exclusion is due to a hardware-specific issue: with 4 pipeline stages and a `u` loop of trip count 2, an extra async copy is emitted without a consumer, causing non-deterministic memory corruption at sequences ≥ 4096 tokens.

### 7.2 `chunk_intra.py` — intra-chunk parallelism

**File:** `nvidia/ops/third_party/kda/chunk_intra.py`

Handles the triangular (causal) attention within each chunk. The inner product precision differs by platform:

```python
solve_tril_dot_precision = (
    "tf32"   if current_platform.is_cuda() and capability >= 80
    else "ieee"
)
```

- **NVIDIA SM80+:** Uses TF32 (reduced precision, ~8× faster on tensor cores) for the triangular solve dot products.
- **AMD (all):** Always uses IEEE FP32 precision. TF32 is a CUDA-specific tensor core mode not available on AMD.

### 7.3 `fused_recurrent.py` — short sequence path

For very short sequences (or as a fallback), a fused recurrent Triton kernel processes tokens one at a time:

```python
from ops.third_party.kda.fused_recurrent import fused_recurrent_kda
output, final_state = fused_recurrent_kda(
    q, k, v, f=f_a, b=b_vec, beta=beta, g=g,
    initial_states=initial_state,
)
```

**AMD-specific `fused_recurrent_kda_packed_decode`:** On AMD, an additional packed-decode variant is exported that processes the post-conv QKV directly without an intermediate tensor copy, reducing memory traffic for short batch sizes.

---

## 8. KDA State: Format and Sizing

Each KDA layer maintains two state tensors per sequence:

**Conv1d state:**
```
Shape:  [batch_size, H=96, kw=4, kDimK=128]
Dtype:  bf16
Size:   96 × 4 × 128 × 2 = 98,304 bytes per sequence ≈ 96 KB
```

**Delta memory:**
```
Shape:  [batch_size, H=96, kDimK=128, kDimV=128]
Dtype:  bf16
Size:   96 × 128 × 128 × 2 = 3,145,728 bytes per sequence ≈ 3 MB
```

**Total per KDA layer:** ~3.1 MB per sequence (fixed, regardless of sequence length).

With 69 KDA layers:
```
Total KDA state = 69 × 3.1 MB = 214 MB per sequence
```

This is fixed — a 128K-token sequence has the same KDA state size as a 1-token sequence.

**State persistence:** The state is stored in the vLLM KV cache manager's "recurrent state" slot — a separate pool from the paged KV cache used by MLA layers.

---

## 9. KDA Metadata Builder (NVIDIA)

**File:** `nvidia/kda_metadata.py`

`KimiK3KDAMetadataBuilder` builds `KimiK3KDAMetadata` each step:

```python
@dataclass
class KimiK3KDAMetadata:
    # State indices aligned to warp boundaries (via Triton kernel):
    aligned_state_indices: torch.Tensor    # [B]  int32
    
    # PDL (Programmatic Dependent Launch) scheduling:
    pdl_scheduling_info: PdlSchedulingInfo
    
    # RecoverSSM support:
    recover_ssm_metadata: RecoverSSMMeta | None
    
    # For prefill: chunking info
    prefill_chunk_metadata: PrefillChunkMeta | None
    
    # Attention backend type for this step:
    backend_type: KDABackendType   # DECODE, PREFILL, MIXED
```

### 9.1 State alignment kernel

**Triton kernel** `_get_aligned_state_indices_kernel`:

The KDA CUDA decode kernel requires that each sequence's state tensors start at warp-aligned offsets in the shared state buffer. This Triton kernel computes the aligned indices:

```python
_get_aligned_state_indices_kernel[grid=(B,)](
    seq_lens,           # [B]  lengths for alignment calculation
    aligned_indices,    # [B]  output: aligned start indices per sequence
    alignment=32,       # warp size
)
# For each sequence b:
#   aligned_indices[b] = next_multiple_of_32(prefix_sum(seq_lens[:b]))
```

Using aligned addresses allows the kernel to use vectorized loads and avoid bank conflicts in shared memory.

### 9.2 PDL scheduling

**PDL (Programmatic Dependent Launch):** SM90+ supports launching a dependent kernel from within a running kernel, eliminating the CPU-GPU roundtrip for kernel scheduling.

The KDA decode kernel uses PDL to launch the subsequent `out_proj` GEMM:

```
[host]  →  launch KDA decode kernel  →  [GPU: KDA kernel running]
                                                   │
                                          [PDL: launch out_proj GEMM from GPU]
                                                   │
                                         [GPU: KDA kernel finishes]
                                                   │
                                         [GPU: out_proj GEMM runs]
```

vs. without PDL:
```
[host]  →  launch KDA kernel  →  wait for KDA  →  launch out_proj  →  ...
```

PDL eliminates the ~5–20 µs host latency for each kernel dispatch.

### 9.3 KDA Metadata: Spec-Decode State Slots

Each KDA layer maintains state in the **SSM state cache** — a separate paged cache from the KV cache used by MLA layers.

```python
@dataclass
class KimiK3KDAMetadata(GDNAttentionMetadata):
    # From GDNAttentionMetadata:
    non_spec_state_indices: Tensor    # [B_ns]  SSM state slots for non-spec tokens
    spec_state_indices:     Tensor    # [B_s]   SSM state slots for spec-decode tokens
    spec_query_start_loc:   Tensor    # [B_s+1] cumsum of spec query lengths
    non_spec_query_start_loc: Tensor
    has_spec_decode:        bool

    # K3-specific:
    aligned_state_indices:  Tensor    # [B] aligned to warp boundary (via Triton kernel)
    pdl_scheduling_info:    PdlSchedulingInfo
    recover_ssm_metadata:   RecoverSSMMeta | None
    prefill_chunk_metadata: PrefillChunkMeta | None
    backend_type:           KDABackendType   # DECODE, PREFILL, MIXED
```

**State slot computation for spec-decode:**

When speculative decoding is active, state slots must be tracked separately for:
- **Non-spec tokens:** Regular decode tokens with confirmed accepted states
- **Spec tokens:** Draft tokens whose states may need rollback if rejected

The `_get_aligned_state_indices_kernel` Triton kernel computes aligned indices for both groups:

```python
@triton.jit
def _get_aligned_state_indices_kernel(
    state_start_ptr,      # [B]  starting SSM state index per sequence
    aligned_indices_ptr,  # [B]  output: warp-aligned state indices
    alignment=32,         # warp width
):
    b = tl.program_id(0)
    raw_start = tl.load(state_start_ptr + b)
    # Align up to next multiple of warp width:
    aligned = (raw_start + alignment - 1) // alignment * alignment
    tl.store(aligned_indices_ptr + b, aligned)
```

This ensures the KDA decode kernel can use vectorized warp-level loads for state data without bank conflicts.

**RecoverSSM metadata for state rollback:**

```python
@dataclass
class RecoverSSMMeta:
    # State slot where the pre-draft checkpoint was saved:
    checkpoint_state_indices: Tensor    # [B_spec]
    # How many spec tokens were actually accepted:
    num_accepted_tokens:      Tensor    # [B_spec]
    # Flag: whether to perform rollback this step:
    needs_recovery:           bool
```

When speculative decode rejects tokens, `RecoverSSM.forward` runs the KDA forward on the `num_accepted_tokens` accepted tokens starting from the checkpoint state, producing the correct final state.

---

## 10. KDA Gate+Beta Kernel

**File:** `nvidia/ops/third_party/kda/fused_recurrent.py:23` — `_kda_gate_beta_fwd_kernel`

A Triton kernel used in the prefill path to compute the gate activation and beta sigmoid for a chunk of time steps. Applied before the main delta-rule recurrence.

**Grid:** `[ceil(T/BT), H]` — one program per (time-tile, head).

```python
@triton.jit
def _kda_gate_beta_fwd_kernel(
    raw_g,     # [B, T, H, kDimK]  raw gate logits
    A_log,     # [H]  log-scale A matrix (negative, head-level)
    dt_bias,   # [H, kDimK] or None  optional dt bias
    gate_out,  # [B, T, H, kDimK]  output gate values
    beta_out,  # [B, T, H]  output beta scalars
    BT: tl.constexpr,   # time-tile size
):
    for t in range(tile_start, tile_end):
        g_raw = raw_g[batch, t, head, :]   # [kDimK=128]

        if dt_bias is not None:
            g_raw = g_raw + dt_bias[head, :]

        # Compute decay factor from A (A_log is negative, so b_a ∈ (0,1)):
        b_a = tl.exp(A_log[head])           # scalar in (0,1)

        # Bounded gate (K3's gate_lower_bound=-5.0 path):
        gate = gate_lower_bound * tl.sigmoid(b_a * g_raw)
        # gate[i] ∈ (gate_lower_bound, 0)  → exp(gate[i]) ∈ (exp(-5), 1)
        # This is a decay factor: 1.0 = full memory retention, 0.007 = rapid forgetting

        beta = tl.sigmoid(raw_beta[batch, t, head])   # scalar in (0, 1)

        gate_out[batch, t, head, :] = gate
        beta_out[batch, t, head]    = beta
```

**Why bounded gate:** The gate `∈ (gate_lower_bound, 0) = (-5, 0)` means `exp(gate) ∈ (exp(-5), 1) ≈ (0.007, 1)`. When `exp(gate) = 1`, the conv state retains full memory. When `exp(gate) ≈ 0.007`, it forgets rapidly (99.3% decay per step). The `A_log × sigmoid` structure allows the model to learn different decay rates per head and per token, giving it adaptive memory retention.

---

## 11. RecoverSSM: State Replay for Speculative Decoding

When speculative decoding rejects draft tokens, the KDA state must be rolled back to the state before the rejected tokens. RecoverSSM handles this.

**File:** `nvidia/ops/recoverssm.py`

```python
class RecoverSSM:
    def forward(
        self,
        conv_state,     # current (potentially wrong) conv1d state
        delta_state,    # current (potentially wrong) delta memory
        saved_conv,     # checkpoint conv state from before speculation
        saved_delta,    # checkpoint delta state from before speculation
        num_accepted,   # how many draft tokens were accepted
    ) → (conv_state, delta_state):
```

**Algorithm:** Re-run the KDA forward pass on accepted tokens only, starting from the saved checkpoint state, producing the correct state.

This is the `SupportsReplaySSM` interface that AMD does not support.

**EAGLE3 integration:** Under EAGLE3, auxiliary hidden states from KDA layers are needed. `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1` controls whether the full AttnRes stream or just the prefix is used for EAGLE3 auxiliary states.

---

## 12. Concrete Examples

### 12.1 Decode: conv1d state update

```
B=1, H=96, kw=4, kDimK=128
Sequence so far: 100 tokens (conv state buffer has last 4 k-vectors)

conv_state[0, h, :, :]:
  slot 0: k_97  (oldest, will be evicted)
  slot 1: k_98
  slot 2: k_99
  slot 3: k_100_new  ← will be written here after shift

New token t=101, head h=0:
  in_proj_qkvgfab(hidden[0]) → q, k, v, g, f_a, b_vec, beta for head 0

  k_new [128] = in_proj_output_k_portion   # raw k from projection

  # Shift conv buffer:
  conv_state[0, 0, 0, :] = conv_state[0, 0, 1, :]  # k_98 → slot 0
  conv_state[0, 0, 1, :] = conv_state[0, 0, 2, :]  # k_99 → slot 1
  conv_state[0, 0, 2, :] = conv_state[0, 0, 3, :]  # k_100 → slot 2
  conv_state[0, 0, 3, :] = k_new                    # k_101 → slot 3

  # Compute k_filtered (depthwise conv, learned weights):
  k_filtered[i] = sum(conv_state[0, 0, j, i] × conv_weight[0, j, i]  for j in range(4))
  # Each dim i is filtered independently across the 4-wide kernel
  # conv_weight[h, j, i] is a scalar weight for head h, position j, dim i
```

### 12.2 Decode: delta net memory update

```
Continuing from above, head h=0:

# State before: delta_state[0, 0, :, :]  [128, 128]
# Input: k_filtered [128], q_new [128], v_new [128], g=0.3, beta_scalar=0.7, b_vec [128]

# 1. Normalize k:
k_norm = k_filtered / rms(k_filtered)   # [128]

# 2. Retrieve from memory:
retrieved = delta_state[0, 0] @ k_norm   # [128] × [128, 128] = [128]
# retrieved[i] = dot(delta_state[0, 0, :, i], k_norm)

# 3. Prediction error:
delta_vec = v_new - retrieved            # [128]

# 4. Per-dim update gate:
beta_gate = sigmoid(beta_scalar=0.7) × b_vec   # [128]  = 0.668 × b_vec
# beta_gate[i] controls how much head-0's memory for dim i is updated

# 5. Outer product update:
# outer = (beta_gate × delta_vec)[None, :] × k_norm[:, None]   [128, 128]
for i in range(128):
    for j in range(128):
        delta_state[0, 0, i, j] += beta_gate[j] * delta_vec[j] * k_norm[i]

# delta_state[0, 0] is now updated: it better remembers v_new given k_norm as key

# 6. Query retrieval from updated memory:
q_norm = q_new / rms(q_new)
output_h = delta_state[0, 0].T @ q_norm   # [128]  softmax-free retrieval

# 7. Output gate:
output_h = sigmoid(g=0.3) × output_h      # scale by 0.574
# output_h [128] is the KDA output for head 0, token 101
```

The key insight: unlike MLA which reads all N cached k/v pairs, KDA reads only the fixed-size `delta_state [128, 128]` — regardless of whether N=100 or N=100000.

### 12.3 Gate computation — concrete numbers for one head

Head h=5, token t=42. We trace how the g1 key-gate is computed step by step.

**Inputs:**

```
A_log[5]   = -1.2              # head-5 log decay (always negative, learned)
f_a[42, :] = [0.3, -0.1, 0.4, ..., 0.2]   # 128 values after f_b_proj
dt_bias[:] = [0.05, -0.03, 0.08, ..., 0.01]  # 128 values (learned parameter)
gate_lower_bound = -5.0
```

**Step 1 — Add dt_bias to f_b output:**

```
g1_inner = f_a[42] + dt_bias
         = [0.3+0.05, -0.1+(-0.03), 0.4+0.08, ...]
         = [0.35,     -0.13,         0.48,     ..., 0.21]
```

**Step 2 — Scale by A_log[h=5] = -1.2:**

```
A_scaled = A_log[5] × g1_inner
         = -1.2 × [0.35, -0.13, 0.48, ...]
         = [-0.42, 0.156, -0.576, ...]
```

**Step 3 — Apply bounded sigmoid gate:**

```
gate[5, 42, :] = gate_lower_bound × sigmoid(A_scaled)
               = -5.0 × sigmoid([-0.42, 0.156, -0.576, ...])

sigmoid([-0.42, 0.156, -0.576, ...]) = [0.397, 0.539, 0.360, ...]

gate[5, 42, :] = -5.0 × [0.397, 0.539, 0.360, ...]
               = [-1.985, -2.695, -1.800, ...]
```

**Step 4 — Per-step state retention (exp of gate):**

```
exp(gate[5, 42, :]) = [exp(-1.985), exp(-2.695), exp(-1.800), ...]
                    = [0.137,        0.067,        0.165,       ...]
```

**Interpretation for dim 0:**

```
exp(-1.985) ≈ 0.137   →   13.7% state retention per step
```

At this token, dimension 0 of head 5's associative memory retains only 13.7% of its previous value — effectively a rapid forgetting step. Dimension 1 (exp(-2.695)=0.067) forgets even faster at 6.7% retention. In contrast, if `A_log[5]` were closer to 0 or `g1_inner` were strongly negative, `A_scaled` would approach 0 from the negative side and `sigmoid(A_scaled) → 0.5`, giving `gate = -5 × 0.5 = -2.5` → `exp(-2.5) = 0.08` (8% retention). Full retention (`exp(gate) → 1`) occurs when `A_scaled → -∞`, i.e., when `gate → 0`.

---

### 12.4 Delta Net state update — tracking one memory location over 3 steps

We track `S[h=0, k_dim=3, v_dim=7]` — a single scalar entry in head 0's 128×128 memory matrix — over 3 decode steps.

**Step 0:** `S = 0.0` (freshly initialized state)

**Step 1:** `key_3=0.8, value_7=1.2, beta_scalar=0.6, gate_dim3=exp(-1.1)=0.333`

```
# Apply decay (gate to previous state):
S_decayed = S × exp(gate_dim3) = 0.0 × 0.333 = 0.0

# Retrieve from memory (prediction for v_dim=7):
retrieved_7 = dot(S_column_k3, key_3)   # with S=0, retrieved=0

# Delta error:
delta_7 = value_7 - retrieved_7 = 1.2 - 0 = 1.2

# Write (beta × delta × key):
S[h=0, k_dim=3, v_dim=7] = 0.0 + 0.6 × 1.2 × 0.8 = 0.576
```

**Step 2:** `key_3=0.3, value_7=0.8, beta_scalar=0.4, gate_dim3=exp(-0.357)=0.700`

```
# Decay:
S_decayed = 0.576 × 0.700 = 0.403

# Retrieve:
retrieved_7 = S_decayed × key_3 = 0.403 × 0.3 = 0.121
# (simplified: treating this dim independently)

# Delta:
delta_7 = 0.8 - 0.121 = 0.679

# Write:
S[h=0, k_dim=3, v_dim=7] = 0.403 + 0.4 × 0.679 × 0.3
                          = 0.403 + 0.081 = 0.484
```

**Step 3:** `key_3=0.9 (similar to step 1), value_7=1.1, beta_scalar=0.7, gate_dim3=exp(-0.916)=0.400`

```
# Decay:
S_decayed = 0.484 × 0.400 = 0.194

# Retrieve:
retrieved_7 = 0.194 × 0.9 = 0.175

# Delta:
delta_7 = 1.1 - 0.175 = 0.925

# Strong beta (0.7) + similar key (0.9) → large write:
write = 0.7 × 0.925 × 0.9 = 0.583

S[h=0, k_dim=3, v_dim=7] = 0.194 + 0.583 = 0.777
```

**Takeaway:** Step 3 uses a key very similar to step 1 (key_3=0.9 vs 0.8) and a high beta (0.7). The combination of:
- aggressive decay (`exp(-0.916)=0.40` → 60% of step 2 memory erased before the write), and
- large effective write (`beta × delta × key = 0.583`),

means step 3 nearly overwrites the memory cell, reducing the influence of the step 2 entry (value=0.8) in favor of the new step 3 entry (value=1.1). This is the "delta" in Delta Net: memory is updated by the prediction error, not by the raw value — a key that is already well-predicted contributes a near-zero delta and changes memory very little.

---

### 12.5 Prefill vs decode cost for one KDA layer at seq_len=4096

All FLOPs are multiply-add pairs × 2 (one multiply + one add per MAC).

**Prefill (4096 tokens, one KDA layer):**

```
in_proj_qkvgfab GEMM:
  input:  [4096, 7168]
  weight: [37152, 7168]
  FLOPs:  4096 × 7168 × 37152 × 2 ≈ 2186 GFLOPs

f_b_proj GEMM (replicated, head_dim=128 → local_proj=12288/TP):
  input:  [4096, 128]
  weight: [12288, 128]  (TP=1 for this comparison)
  FLOPs:  4096 × 128 × 12288 × 2 ≈ 13 GFLOPs

FlashKDA parallel scan (chunked O(T log T) algorithm):
  H=96, kDimK=128, chunk_size=64:
  approx: T × log2(T/chunk_size) × H × kDimK² × 2
        = 4096 × log2(64) × 96 × 128² × 2
        = 4096 × 6 × 96 × 16384 × 2 ≈ 7.73 GFLOPs
  (vs naive sequential O(T × H × kDimK²):
        = 4096 × 96 × 128² × 2 = 12.88 GFLOPs — ~1.7× more)

out_proj GEMM:
  input:  [4096, 12288]
  weight: [7168, 12288]
  FLOPs:  4096 × 12288 × 7168 × 2 ≈ 724 GFLOPs

Total prefill ≈ 2186 + 13 + 7.73 + 724 ≈ 2931 GFLOPs
```

**Decode (1 new token, 4096-token context):**

```
in_proj_qkvgfab GEMM:
  input:  [1, 7168]
  weight: [37152, 7168]
  FLOPs:  1 × 7168 × 37152 × 2 ≈ 533 MFLOPs

f_b_proj GEMM:
  FLOPs:  1 × 128 × 12288 × 2 ≈ 3.1 MFLOPs

Fused KDA decode state update
  (conv1d update + delta rule: O(H × kDimK × kDimV)):
  H=96, kDimK=kDimV=128:
  conv1d:   96 × 4 × 128 × 2 ≈ 0.10 MFLOPs
  retrieve: 96 × 128 × 128 × 2 ≈ 3.15 MFLOPs
  update:   96 × 128 × 128 × 2 ≈ 3.15 MFLOPs
  query:    96 × 128 × 128 × 2 ≈ 3.15 MFLOPs
  subtotal: ≈ 9.55 MFLOPs

out_proj GEMM:
  input:  [1, 12288]
  weight: [7168, 12288]
  FLOPs:  1 × 12288 × 7168 × 2 ≈ 176 MFLOPs

Total decode ≈ 533 + 3.1 + 9.55 + 176 ≈ 722 MFLOPs
```

**KDA vs MLA for decode at 4096-token context:**

```
KDA decode state:   read 96 × 128 × 128 × 2 B = 3.1 MB from HBM (fixed)
MLA KV cache read:  4096 tokens × 576 B/token = 2.4 MB from HBM (grows with N)

At 4096 tokens they are comparable. At 128K tokens:
  MLA KV read: 128K × 576 = 72 MB
  KDA state:   still 3.1 MB (69 KDA layers × 3.1 MB = 214 MB total, but per-layer stays fixed)
```

The KDA advantage grows linearly with context length: at 128K tokens, the MLA layer reads ~23× more KV data per layer than a KDA layer reads state. This is the core reason KDA layers dominate (69 of 93) and MLA appears only every 4th layer.
