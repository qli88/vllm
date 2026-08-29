# Kimi-K3: MoE Routing and Expert Computation

[← Index](kimi_k3_index.md) | [End-to-End](kimi_k3_e2e.md)

Covers K3's Mixture-of-Experts system: grouped top-k routing with noaux_tc load balancing, the Latent MoE two-stage expert projection, shared experts, the LatentMoERunner three-tier dispatch, MegaMoE on Blackwell, and AttnRes.

---

## Table of Contents

1. [MoE Overview](#1-moe-overview)
2. [Routing: Grouped Top-K with noaux_tc](#2-routing-grouped-top-k-with-noaux_tc)
   - 2.1 [GateLinear: sigmoid scoring](#21-gatelinear-sigmoid-scoring)
   - 2.2 [Grouped top-k selection](#22-grouped-top-k-selection)
   - 2.3 [e_score_correction_bias (noaux_tc)](#23-e_score_correction_bias-noaux_tc)
   - 2.4 [Renormalization and scaling](#24-renormalization-and-scaling)
3. [Latent MoE: Two-Stage Expert Projection](#3-latent-moe-two-stage-expert-projection)
   - 3.1 [Architecture comparison: standard vs latent MoE](#31-architecture-comparison-standard-vs-latent-moe)
   - 3.2 [SiTU activation](#32-situ-activation)
   - 3.3 [Latent norm](#33-latent-norm)
4. [Shared Experts](#4-shared-experts)
5. [LatentMoERunner: Three-Tier Dispatch](#5-latentmoerunner-three-tier-dispatch)
   - 5.1 [Tier 0: TAIL_FUSION (SM100 CuTe DSL)](#51-tier-0-tail_fusion-sm100-cute-dsl)
   - 5.2 [Tier 1: ALLREDUCE_OVERLAP (aux stream)](#52-tier-1-allreduce_overlap-aux-stream)
   - 5.3 [Tier 2: COLUMN_PARALLEL (prefill scale)](#53-tier-2-column_parallel-prefill-scale)
   - 5.4 [Complete _fused_forward() and tier selection](#54-complete-_fused_forward-and-tier-selection)
   - 5.5 [SM100 Tail Fusion: KimiK3LatentMoETailOp Internals](#55-sm100-tail-fusion-kimik3latentmoetailop-internals)
6. [MegaMoE: SM100 FP8/FP4 Experts](#6-megamoe-sm100-fp8fp4-experts)
7. [AttnRes: Cross-Layer Residual Mixture](#7-attnres-cross-layer-residual-mixture)
   - 7.1 [AttnRes algorithm](#71-attnres-algorithm)
   - 7.2 [Kernel implementation](#72-kernel-implementation)
   - 7.3 [Interaction with sequence parallel](#73-interaction-with-sequence-parallel)
8. [Expert Weight Quantization (MXFP4)](#8-expert-weight-quantization-mxfp4)
9. [Concrete Examples](#9-concrete-examples)
    - 9.1 [Routing: one token, layer 4](#91-routing-one-token-layer-4)
    - 9.2 [Latent MoE forward: expert 42](#92-latent-moe-forward-expert-42)
    - 9.3 [AttnRes: score and mix](#93-attnres-score-and-mix)
    - 9.4 [SiTU vs SiLU — numerical comparison](#94-situ-vs-silu--numerical-comparison)
    - 9.5 [AttnRes scoring with 12 blocks](#95-attnres-scoring-with-12-blocks)
    - 9.6 [LatentMoERunner tier selection at different token counts](#96-latentmoerunner-tier-selection-at-different-token-counts)

---

## 1. MoE Overview

Every layer in K3 (except layer 0) uses MoE for the FFN component:

```
hidden_states [T, 7168]   (after attention)
      │
      ▼ post_attention_layernorm (RMSNorm)
      │
      ▼ KimiMoE
      │     GateLinear [7168 → 896]   → routing scores [T, 896]
      │     grouped top-16 selection  → topk_ids [T, 16], topk_weights [T, 16]
      │     16 routed expert paths (Latent MoE):
      │       x → down_proj [7168→3584] → RMSNorm → up_proj [3584→7168]
      │     2 shared experts:
      │       x → gate_up [7168→2×3072] → situ → down [3072→7168]
      │     output = weighted_sum(routed) + sum(shared)
      │
      ▼ AttnRes read
      │
hidden_states_out [T, 7168]
```

Key numbers:
- 896 routed experts total, top-16 per token
- 2 shared experts (always active)
- Latent dim: `routed_expert_hidden_size = 3584`
- Expert weights: MXFP4 (4-bit, group_size=32), **not** applied to shared experts or attention weights

---

## 2. Routing: Grouped Top-K with noaux_tc

### 2.1 GateLinear: sigmoid scoring

Unlike DeepSeek V4's `sqrtsoftplus`, K3 uses **sigmoid** for routing scores:

```python
logits = gate_linear(hidden_states)   # [T, 896]  bf16
scores = torch.sigmoid(logits)        # [T, 896]  float32, range (0, 1)
```

**Why sigmoid instead of sqrtsoftplus:**
- Sigmoid always outputs in (0,1) — bounded above and below
- No instability for very large or very negative logits
- Simpler gradient: `σ(x) × (1 - σ(x))`

`moe_router_activation_func = "sigmoid"` is set explicitly in the config.

### 2.2 Grouped top-k selection

`use_grouped_topk=True` with `num_expert_group=1` and `topk_group=1`:

```python
# With num_expert_group=1: all 896 experts are in one group.
# topk_group=1: select 1 expert per group.
# num_experts_per_token=16: select 16 total.
# → This reduces to standard top-16 selection (1 group, 16 experts from it).
```

**Note:** `num_expert_group=1` means the "grouped" structure is degenerate — all experts are in one group, so grouped top-k is equivalent to standard top-k. The `use_grouped_topk=True` flag enables the code path that supports grouping but with only 1 group it behaves identically to ungrouped.

Selection:
```python
# Biased scores for selection:
scores_biased = scores + e_score_correction_bias.unsqueeze(0)  # [T, 896]
topk_ids = torch.topk(scores_biased, k=16, dim=-1).indices      # [T, 16]
```

### 2.3 `e_score_correction_bias` (noaux_tc)

```python
gate.e_score_correction_bias: Parameter [896]  float32
```

A trained per-expert bias added to routing logits **only for selection** (not for weights):

```python
# Selection uses biased scores:
scores_biased = scores + e_score_correction_bias
topk_ids = argtopk(scores_biased, 16)

# Weights use UNBIASED scores:
raw_weights = scores.gather(-1, topk_ids)   # [T, 16]
```

**`topk_method = "noaux_tc"`:** Same mechanism as DeepSeek V4 — per-expert biases are trained to correct load imbalance without an auxiliary loss at inference. Positive bias = under-used expert (boosted). Negative bias = over-used (suppressed).

### 2.4 Renormalization and scaling

```python
# moe_renormalize=True: normalize selected weights to sum=1
renorm_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)   # [T, 16]

# routed_scaling_factor=1.0: no additional scaling
final_weights = renorm_weights × 1.0   # [T, 16]  (effectively unchanged)
```

With `routed_scaling_factor=1.0`, the normalized routing weights sum to 1 directly. This differs from DeepSeek V4 which multiplies by 2.5.

---

## 3. Latent MoE: Two-Stage Expert Projection

### 3.1 Architecture comparison: standard vs latent MoE

**Standard MoE (DeepSeek V3/V4):**
```
x → gate_proj [hidden → intermediate]
x → up_proj   [hidden → intermediate]
→ silu(gate) × up
→ down_proj [intermediate → hidden]
```
Three matrices per expert, ~2 × `hidden × intermediate` parameters.

**Latent MoE (K3):**
```
x → down_proj [hidden=7168 → latent=3584]   (first stage, named "down" because it goes to latent)
→ RMSNorm (latent_moe_use_norm=True)
→ up_proj   [latent=3584 → hidden=7168]   (second stage, named "up" because it returns to hidden)
```
Two matrices per expert, `hidden × latent` parameters each ≈ `2 × 7168 × 3584 ≈ 51M` parameters per expert, but shared across all 896 experts in a **replicated** fashion for the up-projection.

**Key insight — replicated up_proj:**

The routed `down_proj` is per-expert (each expert has its own 7168→3584 matrix). But the `up_proj` is **shared across all experts** (one 3584→7168 matrix replicated on all ranks). This is the "latent" structure: experts only differ in how they map input to the latent space; they all use the same latent-to-output projection.

```python
# Per-expert down projection (MXFP4, different weights per expert):
latent = routed_expert_down_proj[expert_id](hidden)   # [7168 → 3584]

# Shared up projection (BF16, same weights for all experts):
output = routed_expert_up_proj(latent)               # [3584 → 7168]
```

This dramatically reduces the total expert weight count and enables the `LatentMoERunner` tail-fusion optimization.

### 3.2 SiTU activation

K3 uses `hidden_act="situ"` — a custom activation function applied in the latent MoE norm step:

```python
def situ(x, beta=4.0, linear_beta=25.0):
    return torch.sigmoid(beta * x) * (x + linear_beta)
```

For the default parameters (`β=4.0`, `linear_β=25.0`):
```
situ(0)    = sigmoid(0) × (0 + 25)    = 0.5 × 25    = 12.5
situ(1)    = sigmoid(4) × (1 + 25)    ≈ 0.982 × 26  ≈ 25.5
situ(-5)   = sigmoid(-20) × (-5 + 25) ≈ 2e-9 × 20   ≈ 0
```

**Why SiTU instead of SiLU:** SiTU outputs are always positive (sigmoid > 0, and `x + linear_beta` is positive for `x > -25`). This supports a different activation regime than SiLU. The large `linear_beta=25` shifts the activation: even at `x=0`, the output is substantial (12.5), providing a strong baseline signal.

SiTU is required for MegaMoE (`hidden_act == "situ"` is checked at MegaMoE init).

**Where applied:** `situ` is the activation in the shared experts' `KimiMLP`:

```python
gate, up = gate_up_proj(x).split(intermediate_size, dim=-1)
activated = situ(gate) × up
```

For routed experts with latent MoE, the `RMSNorm` between stages (see §3.3) replaces an explicit activation function.

### 3.3 Latent norm

When `latent_moe_use_norm=True` (which it is in K3):

```python
latent = routed_expert_down_proj[eid](hidden)   # [T', 3584]
latent = latent_moe_norm(latent)                # RMSNorm [3584] — NO activation!
output = routed_expert_up_proj(latent)          # [T', 7168]
```

The RMSNorm at the latent boundary serves as a normalization gate: it prevents the latent vectors from growing unbounded across expert updates, and provides implicit non-linearity through the learned scale weights.

---

## 4. Shared Experts

Two shared experts run on ALL tokens unconditionally:

```python
class KimiMLP:  # used for shared experts
    gate_up_proj: MergedColumnParallelLinear [7168 → 2×3072]
    down_proj:    RowParallelLinear [3072 → 7168]

def forward(x):
    gate, up = gate_up_proj(x).split(3072, dim=-1)
    activated = situ(gate) * up
    return down_proj(activated)
```

`moe_intermediate_size=3072` for shared experts (vs `routed_expert_hidden_size=3584` for routed experts).

**Final output:**
```python
routed_out = Σ_i  final_weights[:, i] × routed_expert_output_i  # [T, 7168]
shared_out = shared_expert_0(x) + shared_expert_1(x)             # [T, 7168]
output = routed_out + shared_out
```

**Sequence parallel shared experts:**

Under sequence parallel with `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1`:
```python
# Shared expert is fully TP-sharded (not replicated):
output_shard = shared_expert(x_shard)   # [T/TP, 7168/TP]
output = all_reduce(output_shard)       # [T/TP, 7168]
```

Without this flag (default), shared experts are replicated — each TP rank computes the full output and then all-reduces.

---

## 5. LatentMoERunner: Three-Tier Dispatch

`LatentMoERunner` manages the computationally expensive part of the latent MoE: the **replicated up-projection** (3584→7168) that happens after expert dispatch and before final reduction.

**Why a runner?** The up-projection is large (`3584×7168` in bf16 = ~50MB) and is the same for all experts. Different hardware/parallelism configurations call for different execution strategies.

### 5.1 Tier 0: TAIL_FUSION (SM100 CuTe DSL)

**Requirements:** SM100 (H100/A100 Tensor Cores = actually Blackwell SM100, not Hopper), sequence count ≤ `max_tokens`.

**What it does:** Fuses latent-reduce + RMSNorm + reduce-scatter + multicast up-proj into **one collective kernel** using NVIDIA CuTe DSL:

```python
# File: nvidia/ops/cute_dsl/latent_moe_tail/
# Sequence of operations fused into one kernel:
# 1. Reduce latent contributions across expert dispatches
# 2. Apply RMSNorm on the latent
# 3. Reduce-scatter (SP: each rank holds a shard)
# 4. Multicast up_proj GEMM (each rank computes its output shard)
# All via NCCL symmetric-memory multicast without intermediate HBM writes
```

This tier is the fastest but only works on SM100 with symmetric-memory multicast support.

### 5.2 Tier 1: ALLREDUCE_OVERLAP (aux stream)

**Requirements:** Default NVIDIA path.

**What it does:** Runs the `routed_expert_down_proj` (per-expert GEMM) and the `shared_expert` (dense MLP) concurrently on two CUDA streams, then all-reduces:

```python
# Main stream:
latent = dispatch_to_experts_and_down_proj(hidden, topk_ids, weights)
# This is the per-expert MXFP4 GEMM: [T, 7168] → [T', 3584]

# Aux stream (overlapped when T ≤ 256):
with aux_stream:
    shared_out = shared_expert(hidden)

# Wait for both:
aux_stream.synchronize()

# Shared all-reduce for latent norm + up-proj:
latent_normed = latent_moe_norm(latent)
up_out = all_reduce(routed_expert_up_proj(latent_normed))
output = up_out + shared_out
```

For T ≤ 256 tokens, the routed down-proj is short enough that the aux stream can complete the shared expert in parallel. For larger T, the operations run sequentially.

The all-reduce uses `flashinfer.norm.fused_add_rms_norm` when available — a fused kernel that combines the all-reduce and RMSNorm.

### 5.3 Tier 2: COLUMN_PARALLEL (prefill scale)

**Requirements:** Large prefill batches or fallback.

**What it does:** Each TP rank owns a column shard of `up_proj`, computes its shard, and reduces:

```python
# Each rank owns up_proj_shard [3584, 7168/TP]:
output_shard = latent_normed @ up_proj_shard   # [T', 7168/TP]
output = all_reduce(output_shard)               # [T', 7168]  + shared_expert
```

This is column-parallel linear — the standard TP approach. It avoids replicating the full 7168-dim up-proj output across all ranks before reducing, trading memory for communication.

### 5.4 Complete `_fused_forward()` and tier selection

```python
def _fused_forward(hidden_states, router_logits, input_ids, shared_experts_input):
    # 1. Optionally apply routed input transform (down-proj already done by KimiMoE):
    if self.routed_input_transform is not None:
        hidden_states = self.routed_input_transform(hidden_states)

    # 2. Run MoE expert kernel (FusedMoE or MegaMoE on latent [T, 3584]):
    result = self._forward_entry(
        hidden_states=hidden_states,           # [T, 3584]
        router_logits=router_logits,
        shared_experts_input=shared_experts_input or hidden_states,
    )

    # 3. Unpack: (shared_out, fused_output) or just fused_output
    shared_output, fused_output = _unpack(result)
    # fused_output: [T, 3584] un-reduced, TP-sharded expert output
    # shared_output: [T, 7168] un-reduced shared expert output

    # 4. Select tail tier and call:
    tier = _select_tail_tier(fused_output, shared_output)
    if tier == TAIL_FUSION:
        return _small_batch_tail(fused_output, shared_output, trunc_size)
    elif tier == ALLREDUCE_OVERLAP:
        return _overlap_allreduce_tail(fused_output, shared_output, trunc_size)
    else:
        return _shard_up_proj_tail(fused_output, shared_output, trunc_size)
```

**Tier selection:**

```python
_MAX_NUM_TOKENS = 16   # SM100 tail fusion only supports M ≤ 16

def _select_tail_tier(fused_output, shared_output):
    num_tokens = fused_output.shape[0]
    if (enable_k3_latent_moe_tail_fusion      # env var + SM100 op available
        and num_tokens <= _MAX_NUM_TOKENS     # ≤ 16 tokens
        and _k3_latent_moe_tail_op is not None):
        return TAIL_FUSION        # SM100, ≤ 16 tokens

    if num_tokens <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD:  # default 256
        return ALLREDUCE_OVERLAP  # small decode batch

    return COLUMN_PARALLEL        # large batch / prefill
```

**Tier 1 — `_overlap_allreduce_tail()` internals:**

```python
# All-reduce latent with optional fused RMSNorm:
def allreduce_norm_latent_out(hidden_states, norm):
    if flashinfer_trtllm_fused_allreduce_norm is available:
        # Fused: NCCL all-reduce + RMSNorm in one kernel (saves one HBM round-trip)
        return flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=hidden_states,       # [M, 3584]
            residual=zeros_like(hidden_states),
            rms_gamma=norm.weight,
            pattern_code=AR_RESIDUAL_RMS_NORM,
            fp32_acc=True,
        )
    else:
        reduced = all_reduce(hidden_states)
        return norm(reduced)

fused_latent = allreduce_norm_latent_out(fused_output, transform.norm)
# Overlap: up-proj on default stream; shared all-reduce on aux stream:
up_proj_result, shared_output_reduced = maybe_execute_in_parallel(
    lambda: torch.mm(fused_latent, transform.up_proj.weight.t()),  # [M, 7168]
    lambda: all_reduce(shared_output),                              # [M, 7168]
    event_start, event_done, aux_stream,
)
result = up_proj_result + shared_output_reduced
```

**Tier 2 — `_shard_up_proj_tail()` internals:**

```python
fused_latent = allreduce_norm_latent_out(fused_output, transform.norm)
# Column-parallel up-proj (each rank owns 7168/TP output rows):
local_up_shard = transform.up_proj.weight[rank*shard_h:(rank+1)*shard_h, :]
hidden_shard = torch.mm(fused_latent, local_up_shard.t())   # [M, 7168/TP]
# Add this rank's shared output slice:
hidden_shard += shared_output[:, rank*shard_h:(rank+1)*shard_h]
result = all_gather(hidden_shard)   # [M, 7168]
```

---

### 5.5 SM100 Tail Fusion: KimiK3LatentMoETailOp Internals

**File:** `nvidia/ops/latent_moe_tail.py`

For small decode batches (M ≤ 16) on SM100, a 3-step sequence of CuTeDSL kernels fuses the complete latent MoE tail using Blackwell's NVLink Symmetric Memory (NVLS):

**Hardcoded constants:**
```python
_MAX_NUM_TOKENS          = 16   # max M
_SKINNY_MAX_NUM_TOKENS   = 5    # M ≤ 5: static skinny GEMM
_MMA_TILER_MN            = (64, 32)   # WGMMA tile shape for M=6..16
_GEMM_CLUSTER_MN         = (1, 8)    # CTA cluster
_B_PRIME_STAGES          = 2    # pipeline stages for weight prefetch
_COLLECTIVE_TOKEN_CTAS   = 8    # CTAs for CollectiveKernel
_LAMPORT_COPY_CTAS       = 32   # CTAs for LamportCopyKernel
_LAMPORT_COPY_THREADS    = 224  # threads per Lamport CTA
_SUPPORTED_TP_SIZES      = (8, 16)
```

**3-step execution:**

```python
def __call__(routed_output, shared_output, rms_weight, up_weight):
    # routed_output: [M, 3584],  shared_output: [M, 7168]

    # Step 1: CollectiveKernel (NVLS, no CPU barrier):
    latent, shared_shard = self._collective(routed_output, shared_output, rms_weight)
    # a. AllReduce routed_output across all TP ranks → [M, 3584]
    # b. RMSNorm(latent, rms_weight) → [M, 3584] normed
    # c. ReduceScatter shared_output → [M, 7168/tp_size] this rank's shard

    # Step 2: AdaptiveUpProjectionKernel:
    local_up_weight = up_weight.narrow(0, rank * (7168//tp_size), 7168//tp_size)
    mailbox = self._up_projection(latent, local_up_weight, shared_shard)
    # Computes: partial = latent @ local_up_weight.T + shared_shard  [M, 7168/tp]
    # Two sub-variants:
    #   M ≤ 5 (Skinny): FusedAddMulticastSkinnyGemmKernel (224 threads, static M)
    #   M 6..16 (Dynamic): WGMMA kernel, (64,32) MMA tiler, cluster (1,8), 2 B stages
    # Multicasts result to symmetric mailbox [1, M, 7168] via NVLS

    # Step 3: LamportCopyKernel:
    return self._lamport_copy(mailbox, m=M).squeeze(0)
    # Reads mailbox using Lamport sentinel protocol (spin until sentinel appears)
    # 32 CTAs × 224 threads = 7168 threads covering full hidden_size → [M, 7168]
```

**Symmetric memory layout:**
```
_routed_workspace:  [NUM_LAMPORT_BUFFERS, M_max, tp_size, latent_size*2] bytes
    Used for routed AllReduce + intermediate storage
_shared_workspace:  [NUM_LAMPORT_BUFFERS, M_max, tp_size, shard_dim] bf16
    Used for shared ReduceScatter
mailbox:            [1, M_max, hidden_size] bf16
    Written by AdaptiveUpProjectionKernel via NVLS multicast
    Read by LamportCopyKernel using Lamport sentinel semantics
```

**Warmup compile units:** `get_cutedsl_warmup_compile_units()` returns 6 units: M=1..5 (static skinny, one per token count) + M=dynamic/6..16 (1 WGMMA kernel). All are JIT-compiled ahead of inference.

---

## 6. MegaMoE: SM100 FP8/FP4 Experts

`KimiK3MegaMoEExperts` wraps `DeepseekV4MegaMoEExperts` for SM100 Blackwell hardware.

**Requirements:**
- Expert parallelism enabled
- `hidden_act == "situ"`
- Latent MoE enabled
- Grouped top-k routing
- `deep_gemm.fp8_fp4_mega_moe` available

**What it does:**

Replaces the standard MXFP4 expert dispatch with a single fused FP8×FP4 GEMM kernel that processes all expert GEMMs in one call:

```python
deep_gemm.fp8_fp4_mega_moe(
    x_fp8,          # [T, 7168]  dynamically quantized to FP8
    w1_fp4,         # [896, 7168, 3584]  MXFP4 expert weights (down_proj)
    topk_ids,       # [T, 16]
    topk_weights,   # [T, 16]
    output,         # [T, 3584]  latent space
)
```

The kernel handles expert dispatch, MXFP4 weight dequantization, FP8 activation quantization, and weighted reduction in one pass — dramatically reducing kernel launch overhead and HBM traffic compared to dispatching 16 separate GEMMs.

**Symmetric buffer pre-allocation:**

```python
# Pre-allocated per (EP group, experts, max_tokens, top_k, hidden, intermediate):
self._sym_buffer = allocate_symmetric_buffer(
    (ep_world_size, num_experts, max_tokens, topk, hidden, intermediate)
)
```

The symmetric buffer is shared across EP ranks via NVLink NVSHMEM — avoiding explicit AllReduce for the latent outputs.

---

## 7. AttnRes: Cross-Layer Residual Mixture

AttnRes provides every layer with implicit access to the attention outputs of all previous blocks, mitigating representational collapse in deep networks.

### 7.1 AttnRes algorithm

**Configuration:** `attn_res_block_size=12` — a bank of 12 stored attention hidden states.

At each layer `l`:

**Write phase (after attention, before FFN):**
```python
block_idx = l // block_period   # which bank slot to write
# block_period = num_layers / attn_res_block_size = 93 / 12 ≈ 7.75

# Circular write to bank:
attn_res_bank[block_idx, :] = hidden_after_attention
```

**Read phase (after FFN):**
```python
# Score each bank slot:
scores_input = attn_res_norm(hidden_after_ffn)    # RMSNorm [T, 7168]
scores = scores_input @ attn_res_proj             # [T, 12]  learned linear projection
weights = softmax(scores, dim=-1)                 # [T, 12]

# Weighted mixture of bank states:
residual = weights @ attn_res_bank                # [T, 7168]
output = hidden_after_ffn + residual
```

**Block and "delta" modes:**

The implementation supports two modes:

- **Committed block write:** Only the attention hidden state at the block boundary is written. Most layers see the same bank state as the previous boundary.
- **Delta write:** Each layer updates its own slot in the bank with the current attention output. This gives finer granularity but requires more bank slots.

K3 uses committed block writes by default (`attn_res_block_size=12` covers 93 layers with ~8 layers per block).

### 7.2 Kernel implementation

**File:** `nvidia/ops/attn_res.py`

```python
def attn_res(
    attn_res_bank,    # [num_blocks, H=7168]  stored attention states
    hidden,           # [T, H]  current hidden state
    scores,           # [T]  pre-computed logits (from attn_res_norm + proj)
    new_block,        # new block value to write (or None if no write)
    write_block_idx,  # which bank slot to write (or -1)
    output_onorm,     # whether to apply output RMSNorm
) → output [T, H]:
```

**Triton kernel** `_attn_res_kernel`, grid `[T, ceil(H/BLOCK_D)]`:

```python
# Pseudocode for one (token t, hidden-tile col) program:

# 1. Load prefix = accumulated block-level residual up to this layer:
updated_prefix[BLOCK_D] = prefix[t, col*BLOCK_D : (col+1)*BLOCK_D]

# 2. If HAS_DELTA (after attention or after MLP, delta write path):
updated_prefix += delta[t, col*BLOCK_D : ...]

# 3. Write to block bank if this is a write layer:
if WRITE_BLOCK:
    blocks[t, block_write_idx, col*BLOCK_D:] = updated_prefix

# 4. Online softmax over (num_blocks + 1) sources:
#    Sources: the num_blocks stored bank states + the current updated_prefix
max_val = -inf; sum_exp = 0.0; mixed = zeros(BLOCK_D)

for i in range(num_blocks + 1):
    src = blocks[t, i, col*BLOCK_D:] if i < num_blocks else updated_prefix
    normed_src = RMSNorm(src, norm_weight[col*BLOCK_D:])   # element-wise RMSNorm
    score = dot(normed_src, qk_weight[col*BLOCK_D:])        # scalar score for this tile

    # Numerically stable online softmax update:
    new_max = max(max_val, score)
    scale = exp(max_val - new_max)
    e = exp(score - new_max)
    sum_exp = sum_exp * scale + e
    mixed   = mixed   * scale + e * src
    max_val = new_max

mixed /= sum_exp   # normalize (output is weighted sum of bank states)

# 5. Optional output RMSNorm:
if APPLY_OUTPUT_NORM:
    mixed = RMSNorm(mixed, output_norm_weight[col*BLOCK_D:])

# 6. Write output tile:
output[t, col*BLOCK_D : (col+1)*BLOCK_D] = mixed
```

**Key detail:** The online softmax processes `num_blocks + 1` sources: the 12 bank states **plus** the current `updated_prefix`. This means each layer's output is a mixture of the bank history AND the current layer's attention output — the current layer's attention is always in the mix, not just stored for future retrieval.

**Tuning:** `BLOCK_L=1` for decode (≥256 tokens OR ≤1 active block); `BLOCK_L=4` for small prefill (processes 4 hidden dims per thread).

**SM100 native kernel (`attn_res_kernel.cu`):**

On SM100 Blackwell hardware, when the compiled CUDA op is available:

- 1 TMA producer warp: uses `cp.async.bulk` to bulk-load all `num_blocks × 7168` bank entries into SMEM via `mbarrier` synchronization.
- 8 consumer warps: compute online softmax + weighted mixture + optional RMSNorm using **TMEM** (Tensor Memory — SM100-specific on-chip scratchpad).

Hardcoded constants: `H=7168`, `1 ≤ num_blocks ≤ 8`.

### 7.3 Interaction with sequence parallel

AttnRes is **not compatible with pipeline parallelism** (`SupportsPP` requires all layers on the same device for bank access).

Under sequence parallel (SP):
- The bank stores full `H=7168` vectors (not sharded)
- Each SP rank holds the same bank state (replicated across SP dimension)
- The all-gather before attention makes `hidden_after_attention` available in full to all ranks before the bank write

---

## 8. Expert Weight Quantization (MXFP4)

From config `quantization_config`:

```json
{
  "format": "mxfp4-pack-quantized",
  "group_size": 32,
  "num_bits": 4,
  "strategy": "group",
  "symmetric": true
}
```

**What is quantized:** Only `routed_expert_down_proj` weights (the per-expert 7168→3584 matrices). The following are explicitly excluded:

```json
"ignore": [
  "re:.*self_attn.*",          // attention weights (BF16)
  "re:.*shared_experts.*",     // shared expert weights (BF16)
  "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",  // dense MLP weights (BF16)
  "re:.*lm_head.*",            // language model head (BF16)
  "re:.*vision_tower.*",       // vision encoder (BF16)
  "re:.*mm_projector.*"        // multimodal projector (BF16)
]
```

**MXFP4 format:**
- 4 bits per value (E2M1: 2 exponent, 1 mantissa, 1 sign)
- 32 values per quantization block (group_size=32)
- 1 ue8m0 scale byte per block
- 2 values packed per byte
- Effective: 0.5 bytes/value (weights) + 1/32 bytes/value (scales) ≈ 0.53 bytes/value

**Storage per routed expert:**
```
down_proj: [7168 → 3584] = 7168 × 3584 × 0.53 bytes ≈ 13.6 MB per expert (MXFP4)
vs BF16:  7168 × 3584 × 2 bytes = 51.4 MB per expert
Compression: ~3.8×
```

With 896 experts: `896 × 13.6 MB ≈ 12.2 GB` routed expert weights total.

---

## 9. Concrete Examples

### 9.1 Routing: one token, layer 4

```
Token: "▁Paris" at position 500, layer 4 (MLA+MoE)
hidden_states [1, 7168]  (after MLA attention)

Gate projection:
logits = gate_linear(hidden) → [1, 896]  bf16

Sigmoid scores:
scores = sigmoid(logits) → [1, 896]  float32, range (0,1)
# Example: scores[0, 42]=0.81, scores[0, 107]=0.74, scores[0, 500]=0.67, ...

Biased scores for selection (e_score_correction_bias from checkpoint):
scores_biased = scores + bias
# bias[42]=-0.05 (slightly over-used), bias[107]=+0.08 (under-used)
scores_biased[0, 42]  = 0.81 - 0.05 = 0.76
scores_biased[0, 107] = 0.74 + 0.08 = 0.82   ← now higher than 42

Top-16 by biased score → topk_ids include 107, 500, ... (illustrative)
topk_ids[0, :] = [107, 500, 33, 82, 201, 447, 12, 89, 302, 650, 177, 421, 3, 800, 55, 730]

Unbiased weights at selected positions:
raw = scores[0, [107, 500, 33, ...]] = [0.74, 0.67, 0.71, ...]

Renormalize (sum to 1):
total = sum(raw) ≈ 10.4
renorm = raw / 10.4 = [0.071, 0.064, ...]

final_weights = renorm × 1.0 = [0.071, 0.064, ...]  (routed_scaling_factor=1.0)
```

### 9.2 Latent MoE forward: expert 42

```
Expert 42 receives token t from the dispatch:
  x [7168]  = hidden_states[t]

Stage 1 — down projection (MXFP4):
  latent = routed_expert_down_proj[42](x)  # [7168 → 3584]
  latent [3584]  bf16

Stage 2 — latent norm:
  rms = sqrt(mean(latent²) + 1e-5)   e.g. rms = 0.421
  latent_normed = latent / rms × latent_moe_norm.weight   [3584]
  # No activation function — RMSNorm serves as implicit normalization gate

Stage 3 — up projection (replicated, shared across all experts):
  output_42 = routed_expert_up_proj(latent_normed)   # [3584 → 7168]
  # This same matrix is used for all 896 experts

# output_42 is scaled by its routing weight before aggregation:
contribution_42 = final_weights[0, expert_42_rank] × output_42
```

### 9.3 AttnRes: score and mix

```
Layer 32 (MLA+MoE), attn_res_block_size=12, 93 total layers
block_period = floor(93 / 12) ≈ 7

Block index at layer 32:
block_idx = 32 // 7 = 4   ← bank slot 4

After MLA attention at layer 32:
hidden_after_attn [1, 7168]  = attention output for one token

Write phase:
  attn_res_bank[4, :] = hidden_after_attn   # overwrite slot 4

FFN runs → hidden_after_ffn [1, 7168]

Read phase:
  # Score the 12 bank slots:
  normed = attn_res_norm(hidden_after_ffn)    [1, 7168]
  scores = normed @ attn_res_proj             [1, 12]
  # Example: scores[0, :] = [-0.3, 0.2, 0.8, 1.1, 0.6, -0.1, 0.4, 0.7, 0.9, 0.3, -0.2, 0.5]
  weights = softmax(scores)
  # ≈ [0.04, 0.07, 0.12, 0.16, 0.10, 0.05, 0.08, 0.11, 0.13, 0.07, 0.04, 0.09]

  # Weighted mix of all 12 bank states:
  residual = Σ_b weights[b] × attn_res_bank[b, :]   [1, 7168]
  # Bank slot 3 (score 1.1, weight 0.16) dominates
  # That slot holds the attention output from ≈ layer 21 (block_idx=3, layers 21-27)

  output = hidden_after_ffn + residual   [1, 7168]

  # This token's FFN output is enriched with attention context from ~8 layers ago.
```

### 9.4 SiTU vs SiLU — numerical comparison

Given gate values `x = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]`:

```
SiTU(x) = sigmoid(4*x) * (x + 25.0)
SiLU(x) = x * sigmoid(x)
```

| x    | sigmoid(4x) | SiTU(β=4, lb=25) | SiLU(x)  |
|------|-------------|-------------------|----------|
| -5.0 | 2.1e-9      | ≈ 0.000           | -0.034   |
| -2.0 | 3.4e-4      | ≈ 0.008           | -0.238   |
| -1.0 | 0.018       | ≈ 0.432           | -0.269   |
|  0.0 | 0.500       | 12.500            |  0.000   |
|  0.5 | 0.880       | 22.440            |  0.311   |
|  1.0 | 0.982       | 25.532            |  0.731   |
|  2.0 | 0.9997      | 26.993            |  1.762   |
|  3.0 | ≈ 1.000     | 27.999 ≈ 28.0     |  2.858   |

**Key observations:**

- SiTU is always ≥ 0: `sigmoid > 0` and `x + 25 > 0` for any x > −25 (covering the entire realistic gate range).
- SiLU can be negative: for x < 0, `x < 0` and `sigmoid(x) < 1` so the product is negative.
- At x = 0: SiTU returns 12.5 (a strong positive baseline) while SiLU returns exactly 0.
- At x = 3: SiTU ≈ 28 (saturates near x + 25); SiLU ≈ 2.86 (grows without bound but slowly).

**Why this matters for MXFP4:** The MXFP4 format (E2M1, group_size=32) has a limited dynamic range — the maximum representable value per quantization block is constrained by the 8-bit ue8m0 scale. SiLU's negative outputs create large magnitude swings in gate activations; the difference between a large negative and a large positive value within one quantization group forces the scale to accommodate both extremes, wasting representational precision. SiTU's bounded minimum (≈ 0) and large `linear_beta=25` ensure all gate activations are non-negative and concentrated in a narrower positive range, reducing quantization error. This is why `hidden_act == "situ"` is a hard requirement for the MegaMoE MXFP4 path.

### 9.5 AttnRes scoring with 12 blocks

Setup: layer 50, `block_period=7`, `attn_res_block_size=12`, `num_layers=93`.

```
block_idx = 50 // 7 = 7   (the slot being written at this layer)
Number of slots written so far: blocks[0] through blocks[7] = 8 blocks
```

For one decode token, the Triton kernel computes scores over 8 stored block states plus the current `updated_prefix` (9 sources total):

```
Source               Layers covered   RMSNorm+dot score   softmax weight
─────────────────────────────────────────────────────────────────────────
blocks[0]            layers  0- 6     -0.30               0.03
blocks[1]            layers  7-13      0.10               0.04
blocks[2]            layers 14-20      0.40               0.11
blocks[3]            layers 21-27      0.80  ← highest    0.42
blocks[4]            layers 28-34      0.20               0.05
blocks[5]            layers 35-41     -0.10               0.03
blocks[6]            layers 42-48      0.50               0.16
blocks[7]            layers 49-55      0.35               0.08
updated_prefix       layers  0-50      0.30               0.08
─────────────────────────────────────────────────────────────────────────
```

softmax denominator: exp(-0.30) + exp(0.10) + exp(0.40) + exp(0.80) + exp(0.20) + exp(-0.10) + exp(0.50) + exp(0.35) + exp(0.30) ≈ 8.33

```python
# weighted mixture (dominant terms shown):
residual = 0.42 × blocks[3]       # layers 21-27: highest relevance
         + 0.16 × blocks[6]       # layers 42-48: third highest
         + 0.11 × blocks[2]       # layers 14-20: second highest
         + 0.08 × blocks[7]       # layers 49-55: recent
         + 0.08 × updated_prefix  # full prefix residual
         + ...
```

**Interpretation:** `blocks[3]` (the attention output accumulated over layers 21–27) dominates with weight 0.42. This means the model determined that attention computed around the 4th "block epoch" was most predictive for the current token — a cross-layer skip connection that lets layer 50 directly incorporate attention context from much earlier in the network without gradient path interference.

The online softmax processes all sources in one pass over `BLOCK_D`-wide tiles of the 7168-dim hidden state, so memory traffic is `(num_blocks + 1) × 7168 × 2 bytes = 9 × 14.3 KB = 129 KB` per decode token — comparable to a single MLA KV read.

### 9.6 LatentMoERunner tier selection at different token counts

Hardware: SM100 (Blackwell), TP=8.

```
T     Tier selected              Mechanism
────  ─────────────────────────  ────────────────────────────────────────────────────
  1   TAIL_FUSION (T ≤ 16)       CollectiveKernel + skinny GEMM (M≤5 static path)
                                 + LamportCopy. Full NVLS multicast, no CPU barrier.
  8   TAIL_FUSION (T ≤ 16)       M=8 → dynamic WGMMA path (M=6..16), (64,32) tiler,
                                 cluster (1,8), 2 B-stage pipeline.
 17   ALLREDUCE_OVERLAP (T ≤ 256) Two-stream: per-expert down-proj on main stream,
                                 shared_expert on aux stream, then allreduce + up-proj.
100   ALLREDUCE_OVERLAP (T ≤ 256) Same two-stream path. Aux stream shared_expert
                                 completes well within the main-stream expert GEMM.
257   COLUMN_PARALLEL (T > 256)  Each TP rank holds up_proj shard [3584, 7168/8].
                                 Compute [T, 3584] @ [3584, 896]^T then all_reduce.
4096  COLUMN_PARALLEL             Same path; large M amortizes allreduce overhead.
```

Rough latency estimates at SM100 / H100 equivalent (HBM BW ≈ 3.35 TB/s):

```
Tier              T      Dominant cost                      ~Latency
─────────────────────────────────────────────────────────────────────
TAIL_FUSION        1     NVLS symmetric memory read/write    ~12 µs
TAIL_FUSION        8     WGMMA over [8, 3584] × [3584,896]  ~12 µs
ALLREDUCE_OVERLAP 17     NCCL ring allreduce [17, 7168]      ~28 µs
ALLREDUCE_OVERLAP 100    NCCL ring allreduce [100, 7168]     ~45 µs
COLUMN_PARALLEL  257     2× allreduce [257, 7168]            ~80 µs
COLUMN_PARALLEL 1000     2× allreduce [1000, 7168]           ~95 µs
```

The TAIL_FUSION tier avoids PCIe hops entirely (NVLink symmetric memory), which is why T=1 and T=8 have nearly identical latency. The jump at T=17 reflects the NCCL ring-allreduce overhead that dominates for small M. COLUMN_PARALLEL incurs two allreduces (one for latent norm reduction, one for up-proj output) but amortizes them across larger M, making it the correct choice for prefill-scale batches.
