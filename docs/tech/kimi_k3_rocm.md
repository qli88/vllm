# Kimi-K3: ROCm / AMD Execution Paths

[← Index](kimi_k3_index.md)

All divergences between the AMD (`vllm/models/kimi_k3/amd/`) and NVIDIA (`vllm/models/kimi_k3/nvidia/`) implementations. Every missing feature, every kernel difference, every algorithm variation — with explanations for why they differ.

---

## Table of Contents

1. [Hardware Dispatch](#1-hardware-dispatch)
2. [Feature Gap: What AMD Does Not Have](#2-feature-gap-what-amd-does-not-have)
3. [MLA: Wrapper vs Full Implementation](#3-mla-wrapper-vs-full-implementation)
4. [KDA: Full Details of Differences](#4-kda-full-details-of-differences)
   - 4.1 [Backend selection (triton only on AMD)](#41-backend-selection-triton-only-on-amd)
   - 4.2 [Fused decode kernel: gfx950 only](#42-fused-decode-kernel-gfx950-only)
   - 4.3 [ROCm fused KDA decode kernel internals](#43-rocm-fused-kda-decode-kernel-internals)
   - 4.4 [AMD-specific: fused_recurrent_kda_packed_decode](#44-amd-specific-fused_recurrent_kda_packed_decode)
5. [KDA Metadata Builder: Device-Side Chunk Metadata](#5-kda-metadata-builder-device-side-chunk-metadata)
6. [LatentMoERunner: Two Tiers vs Three](#6-latentmoerunner-two-tiers-vs-three)
7. [AttnRes: Triton Only (No SM100 Native Kernel)](#7-attnres-triton-only-no-sm100-native-kernel)
8. [Triton KDA Primitives: ROCm-Specific Tuning](#8-triton-kda-primitives-rocm-specific-tuning)
   - 8.1 [chunk.py autotune differences](#81-chunkpy-autotune-differences)
   - 8.2 [chunk_intra.py precision difference](#82-chunk_intrapy-precision-difference)
   - 8.3 [fused_recurrent.py gate handling](#83-fused_recurrentpy-gate-handling)
9. [MTP: Functionally Equivalent](#9-mtp-functionally-equivalent)
10. [Model-Level and File Organization](#10-model-level-and-file-organization)
    - 10.1 [Supported interfaces comparison](#101-supported-interfaces-comparison)
    - 10.2 [File organization difference](#102-file-organization-difference)
11. [AMD-Specific Tests](#11-amd-specific-tests)
12. [Worked Examples: AMD vs NVIDIA Execution](#12-worked-examples-amd-vs-nvidia-execution)
    - 12.1 [KDA decode on gfx942 vs gfx950 vs NVIDIA SM90](#121-kda-decode-on-gfx942-vs-gfx950-vs-nvidia-sm90)
    - 12.2 [AttnRes Triton vs SM100 native kernel](#122-attnres-triton-vs-sm100-native-kernel)
    - 12.3 [Triton KDA chunk.py autotuning result](#123-triton-kda-chunkpy-autotuning-result)

---

## 1. Hardware Dispatch

`vllm/models/kimi_k3/__init__.py` is the sole dispatch entry point:

```python
from vllm.platforms import current_platform

if current_platform.is_cuda():
    from .nvidia.model import KimiLinearForCausalLM, KimiK3ForConditionalGeneration
    from .nvidia.mtp   import KimiK3MTP as KimiK3MTPModel
elif current_platform.is_rocm():
    from .amd.linear   import KimiLinearForCausalLM
    from .amd.model    import KimiK3ForConditionalGeneration
    from .amd.mtp      import KimiK3MTP as KimiK3MTPModel
# TPU and other platforms handled by plugin registration
```

`TYPE_CHECKING` always sees the NVIDIA branch (for IDE type resolution). The ROCm branch is only activated at runtime.

---

## 2. Feature Gap: What AMD Does Not Have

Complete list of NVIDIA features absent on AMD:

| Feature | NVIDIA file | AMD status | Reason |
|---|---|---|---|
| **DSpark draft model** | `nvidia/dspark_mla.py` | Missing | No K3DSparkForCausalLM registered |
| **RecoverSSM state replay** | `nvidia/ops/recoverssm.py` | Missing | Requires NVIDIA KDA metadata |
| **SupportsReplaySSM interface** | `nvidia/model.py` | Missing | RecoverSSM dependency |
| **SupportsEncoderCudaGraph** | `nvidia/model.py` | Missing | Vision encoder CUDA graph |
| **MambaStateShapes interface** | `nvidia/model.py` | Missing | SSM-style state shape reporting |
| **MegaMoE** | `nvidia/model.py` | Missing | DeepGEMM FP8/FP4 CUDA-specific |
| **CuTe-DSL TAIL_FUSION tier** | `nvidia/ops/cute_dsl/latent_moe_tail/` | Missing | SM100 CuTe DSL (CUDA-only) |
| **CuTe-DSL GEMM-RS fusion** | `nvidia/ops/cute_dsl/gemm_rs.py` | Missing | NCCL symmetric memory (CUDA SM100) |
| **Low-latency GEMM (SM103)** | `nvidia/low_latency_gemm.py` | Missing | B300 CUDA-specific skinny GEMMs |
| **Sequence parallelism** | `nvidia/model.py` | Missing | Requires GEMM-RS/SP ops |
| **Multi-stream MoE overlap** | `nvidia/model.py` | Missing | Aux CUDA stream MoE overlap |
| **Ubatch / DBO** | `nvidia/model.py` | Missing | NVIDIA-specific batching |
| **Self-contained NVIDIA MLA** | `nvidia/mla.py` | Replaced by generic wrapper | See §3 |
| **Fused MLA key-concat kernel** | `nvidia/ops/fused_mla_key_concat_kv_cache.py` | Missing | PDL CUDA C++ kernels |
| **Vision FA4 warmup** | `nvidia/ops/vision_fa4_warmup.py` | Missing | Flash-Attention-4 CUDA |
| **CUDA KDA prefill backend** | `nvidia/kda.py` (`backend="cuda"`) | Missing | Triton-only on AMD |
| **g_proj aux stream overlap** | `nvidia/mla.py` (≤512 tokens) | Missing | Integrated into generic MLA |

AMD-specific additions not on NVIDIA:

| Feature | AMD file | NVIDIA status |
|---|---|---|
| **Device-side chunk metadata** | `amd/kda_metadata.py` | CPU-side only |
| **`fused_recurrent_kda_packed_decode`** | `amd/ops/third_party/kda/fused_recurrent.py` | Missing |
| **`ops/kda_decode.py` weight loaders** | `amd/ops/kda_decode.py` | Inline in `kda.py` |

---

## 3. MLA: Wrapper vs Full Implementation

**NVIDIA** ships `nvidia/mla.py` — a fully self-contained `MultiHeadLatentAttention` (1026 lines) that owns the complete attention path:

- `fused_qkv_a_proj` (replicated) + `q_a_layernorm`
- `kv_a_proj_with_mqa` + `kv_a_layernorm`
- `q_b_proj` (TP-sharded)
- W_UK / W_UV weight absorption at load time
- Fused KV cache insert via PDL CUDA kernels (bf16 / fp8 / fp8_ds_mla variants)
- `g_proj` sigmoid output gate on aux stream (≤512 tokens)
- Chunked-context merge via `merge_attn_states`
- DCP manager integration
- GEMM-RS fused output projection

**AMD** defines `KimiMLAAttention` as a thin wrapper around `MultiHeadLatentAttentionWrapper` (from `vllm.model_executor.layers.mla`):

```python
# amd/linear.py:
class KimiMLAAttention(MultiHeadLatentAttentionWrapper):
    def __init__(self, ...):
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        # delegates everything to vllm's generic MLA implementation
```

This means AMD uses the **generic vLLM MLA path**, without:
- K3-specific PDL fused KV cache insert kernels
- `fp8_ds_mla` cache format support
- `g_proj` auxiliary stream overlap
- DCP manager
- GEMM-RS

The AMD constructor omits `use_sequence_parallel` and `run_gemm_rs` parameters that the NVIDIA version accepts.

**Implication:** AMD MLA is functionally correct but slower — it lacks the K3-specific kernel fusions and parallelism optimizations.

---

## 4. KDA: Full Details of Differences

### 4.1 Backend selection (triton only on AMD)

```python
# NVIDIA (nvidia/kda.py):
if backend == "cuda":
    ops.fused_kda_fwd(...)   # compiled CUDA FlashKDA prefill kernel
elif backend == "triton" or backend == "auto":
    chunk_gated_delta_rule(...)  # Triton chunked

# AMD (amd/kda.py):
assert backend == "triton", f"AMD only supports triton backend, got {backend}"
# Falls back unconditionally to Triton chunked attention
```

The CUDA `ops.fused_kda_fwd` is a CUDA-only compiled extension. It is not compiled for ROCm, so AMD always uses Triton.

**Performance implication:** The CUDA FlashKDA prefill kernel is highly optimized for NVIDIA hardware. On AMD, Triton chunked linear attention (from the shared `third_party/kda/chunk.py`) is used instead, which may be slower for long sequences.

### 4.2 Fused decode kernel: gfx950 only

```python
# AMD (amd/kda.py):
def is_fused_kda_decode_supported() -> bool:
    return on_gfx950()   # MI350X only — NOT MI300X (gfx942)

if is_fused_kda_decode_supported():
    ops.fused_kda_decode(...)   # ROCm HIP kernel from fused_kda_decode_kernel_rocm.cu
else:
    # Fallback: three separate Triton kernels
    #   1. conv1d update
    #   2. delta rule recurrence
    #   3. output gate
    _triton_kda_decode_step(...)
```

On gfx942 (MI300X, the most common AMD deployment GPU), the fused decode kernel is **not supported**. Each decode step requires 3 Triton kernel launches instead of 1 fused CUDA/HIP kernel.

### 4.3 ROCm fused KDA decode kernel internals

`csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu`

The algorithm (conv1d → gated delta rule → output) is identical between CUDA and ROCm. The hardware-level differences:

| Aspect | NVIDIA (CUDA) | AMD (ROCm) |
|---|---|---|
| BF16 type | `__nv_bfloat16` | `__hip_bfloat16` |
| Stream type | `cudaStream_t` | `hipStream_t` |
| Warp/wavefront width | 32 lanes | 64 lanes |
| `kLanes` (lanes per group) | 8 (quarter-warp) | 16 (half-wavefront) |
| `kGroups` = `kThreads / kLanes` | `256 / 8 = 32` | `256 / 16 = 16` |
| `kRows` = `kDimK / kGroups` | `128 / 32 = 4` | `128 / 16 = 8` |
| Cross-lane reduction | `__shfl_xor_sync` (warp shuffle) | `__builtin_amdgcn_update_dpp` (DPP) with `dpp_add` |
| Vector loads | `float4` + `__ldg` / `__stcs` | `float __attribute__((ext_vector_type(4)))` + `__builtin_nontemporal_load/store` |
| Error checking | `cudaGetLastError()` | `hipGetLastError()` |
| GPU gating | Any CUDA SM | `on_gfx950()` only |

**DPP (Data Parallel Primitives):** AMD's cross-lane communication mechanism. Instead of CUDA's `__shfl_xor_sync` which broadcasts between arbitrary lanes, AMD uses DPP — a hardware instruction that applies a permutation pattern to lanes before an operation. For reduction across 16 lanes, DPP with `dpp_add` achieves the same effect as `__shfl_xor_sync` on NVIDIA but uses the hardware's native shuffle unit.

**Why gfx950 only:** The ROCm kernel uses DPP intrinsics and non-temporal vector loads that require gfx950 (MI350X). gfx942 (MI300X) uses a different DPP encoding and may have different vector width constraints.

**Template parameters** (identical between CUDA and ROCm):

| Parameter | Purpose |
|---|---|
| `kFixedHeads` | Compile-time head count (96 for K3) |
| `kApplyOnorm` | Apply output RMSNorm in epilogue |
| `kUpdateConvState` | Update conv1d buffer (True for new tokens) |
| `kUseLowerBound` | Apply `gate_lower_bound=-5.0` clamp to f_a |
| `kApplyBetaSigmoid` | Whether beta scalar is pre-sigmoided |

### 4.4 AMD-specific: `fused_recurrent_kda_packed_decode`

**File:** `amd/ops/third_party/kda/fused_recurrent.py` (AMD only)

An additional Triton kernel that processes the KDA recurrence from packed QKV representation without an intermediate tensor copy:

```python
def fused_recurrent_kda_packed_decode(
    packed_qkvgfab,    # [B, H, 387]  packed QKVGFABeta (no intermediate unpack)
    conv_state,        # [B, H, kw=4, kDimK=128]  in-place updated
    delta_state,       # [B, H, kDimK=128, kDimV=128]  in-place updated
    conv1d_weight,     # [H, 4, 128]
) → output [B, H, kDimV]:
```

This saves one HBM round-trip by directly consuming the packed projection output (from `in_proj_qkvgfab`) without first unpacking to separate Q, K, V, G, F_A, B, Beta tensors.

**Why AMD needs this but NVIDIA doesn't:** The NVIDIA fused decode kernel already handles the projection + recurrence in one shot. On AMD where the fused kernel is only on gfx950, gfx942 uses Triton — the packed variant reduces the kernel count from 3 to 1 Triton kernel for gfx942 devices.

---

## 5. KDA Metadata Builder: Device-Side Chunk Metadata

**NVIDIA** `KimiK3KDAMetadataBuilder` builds metadata mostly on CPU and uses a Triton kernel `_get_aligned_state_indices_kernel` for state index alignment.

**AMD** `KimiK3ROCmKDABackend` / `KimiK3ROCmKDAMetadataBuilder` overrides `_build_chunk_metadata()`:

```python
# amd/kda_metadata.py:
class KimiK3ROCmKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    def _build_chunk_metadata(self, ...):
        # Instead of CPU-side computation:
        # Call prepare_chunk_metadata_device() → Triton kernel
        return prepare_chunk_metadata_device(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            chunk_size=self.chunk_size,
            device=device,
        )

# prepare_chunk_metadata_device uses _chunk_metadata_kernel (Triton):
@triton.jit
def _chunk_metadata_kernel(
    seq_lens_ptr,
    query_start_loc_ptr,
    chunk_indices_ptr,    # output: [T] chunk index for each token
    chunk_offsets_ptr,    # output: [T] within-chunk offset
    chunk_size: tl.constexpr,
    T: tl.constexpr,
):
    t = tl.program_id(0)
    seq_start = tl.load(query_start_loc_ptr + ...)
    pos_in_seq = t - seq_start
    chunk_indices_ptr[t] = pos_in_seq // chunk_size
    chunk_offsets_ptr[t] = pos_in_seq % chunk_size
```

**Why device-side on AMD:** The AMD chunked attention kernel requires chunk index metadata in GPU memory. Building it on CPU and doing a host-to-device transfer adds latency for each prefill step. The Triton kernel builds the metadata directly on the GPU, avoiding the H2D transfer.

NVIDIA's metadata builder (`KimiK3KDAMetadataBuilder`) builds a richer `KimiK3KDAMetadata` with:
- PDL scheduling info
- RecoverSSM metadata
- Aligned state indices (via `_get_aligned_state_indices_kernel`)

AMD's `KimiK3ROCmKDAMetadata` only needs chunk indices for the Triton chunked attention kernel.

**Backend name:**
- NVIDIA: `"KIMI_K3_KDA"` 
- AMD: `"KIMI_K3_KDA_ROCM"`

---

## 6. LatentMoERunner: Two Tiers vs Three

**NVIDIA** `LatentMoERunner` has 3 tiers (see [kimi_k3_moe.md §5](kimi_k3_moe.md)):
- Tier 0: TAIL_FUSION (SM100 CuTe DSL)
- Tier 1: ALLREDUCE_OVERLAP (aux stream, ≤256 tokens)
- Tier 2: COLUMN_PARALLEL (prefill scale)

**AMD** `ROCmLatentMoERunner` has 2 paths:

| Path | Condition | Mechanism |
|---|---|---|
| **Sharded tail** | `_tail_shardable()`: up-proj divisible by TP, shared experts present, not SP, `routed_scaling_factor==1.0` | Shard `up_proj` by TP rank → `hidden_shard.addmm_(latent, up_proj_shard.t())` → `_maybe_reduce_final_output` |
| **Fallback** | otherwise | `super().forward()` (base MoERunner, replicated up-proj) + warning log |

```python
# amd/latent_moe_runner.py:
class ROCmLatentMoERunner(LatentMoERunnerBase):
    def forward(self, x, latent, weights, topk_ids, shared_experts, ...):
        if self._tail_shardable(shared_experts):
            return self._shard_up_proj_tail(latent, shared_experts, ...)
        else:
            logger.warning("AMD: using fallback replicated up-proj path")
            return super().forward(x, latent, weights, topk_ids, shared_experts, ...)

    def _tail_shardable(self, shared_experts) -> bool:
        return (
            self.up_proj.weight.shape[0] % self.tp_size == 0  # divisible by TP
            and shared_experts is not None
            and not self.use_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )

    def _shard_up_proj_tail(self, latent, shared_experts, ...):
        # Each TP rank owns up_proj_shard [3584, 7168/TP]:
        up_proj_shard = self._get_local_up_proj_shard()
        hidden_shard = torch.zeros(latent.shape[0], 7168 // self.tp_size, device=latent.device)
        hidden_shard.addmm_(latent, up_proj_shard.t())  # in-place GEMM
        return self._maybe_reduce_final_output(hidden_shard, shared_experts)
```

**What AMD is missing:**
- Tier 0 (CuTe DSL): requires SM100 hardware + NCCL symmetric memory (CUDA-only)
- Tier 1 (aux stream overlap): requires CUDA auxiliary stream + flashinfer AllReduce fusion
- The AMD sharded-tail is closest to NVIDIA's Tier 2 (COLUMN_PARALLEL)

---

## 7. AttnRes: Triton Only (No SM100 Native Kernel)

**NVIDIA:** Two implementations:
1. Native SM100 CUDA kernel (`attn_res_kernel.cu`) — warp-specialized with TMA + TMEM
2. Triton fallback (`_attn_res_kernel`) — selected when SM100 op unavailable

**AMD:** Always uses the Triton kernel:

```python
# nvidia/ops/attn_res.py:
def attn_res(...):
    if _native_op_available():   # compiled SM100 CUDA op
        return ops.kimi_k3_attn_res(...)
    else:
        return _attn_res_triton(...)

# amd/ops/attn_res.py:
def attn_res(...):
    return _attn_res_triton(...)   # always Triton, no native op check
```

**Triton kernel tuning:**

Both platforms use the same Triton kernel with the same tuning:
- `BLOCK_L=1` for decode (T ≥ 256 OR num_blocks ≤ 1)
- `BLOCK_L=4` for small prefill (T < 256 AND num_blocks > 1)

The Triton kernel is platform-portable and produces correct results on both NVIDIA and AMD. The NVIDIA SM100 native kernel is a performance optimization only.

**No warmup profiles on AMD:**

```python
# nvidia/ops/attn_res.py:
def get_attn_res_triton_warmup_profiles(max_blocks) -> list[WarmupProfile]:
    # Returns profiles for various (token_count, block_count) combinations
    # to pre-compile Triton kernels at startup
    ...

# amd/ops/attn_res.py:
# No equivalent function — Triton JIT-compiles on first use
```

---

## 8. Triton KDA Primitives: ROCm-Specific Tuning

The Triton KDA kernels in `third_party/kda/` are shared between platforms but contain conditional logic.

### 8.1 `chunk.py` autotune differences

```python
# nvidia/ops/third_party/kda/chunk.py and amd/ops/third_party/kda/chunk.py:
# (Same file, platform-conditional inside)

if current_platform.is_rocm():
    NUM_WARPS_AUTOTUNE = [2, 4, 8, 16]    # max 16: AMD wavefronts are 64 lanes
else:
    NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]   # NVIDIA: 32-lane warps support higher counts
```

**Why max 16 warps on AMD:** AMD wavefronts contain 64 threads (vs 32 on NVIDIA). A 16-wavefront kernel has `16 × 64 = 1024` total threads — the same as NVIDIA's 32-warp kernel with `32 × 32 = 1024` threads. So the AMD warp count is halved to keep thread count comparable.

**`num_stages=4` exclusion (critical AMD bug fix):**

```python
if current_platform.is_rocm():
    _CHUNK_NUM_STAGES = [2, 3]           # 4 stages excluded
    _RECOMPUTE_W_U_NUM_STAGES = [2, 3]  # 4 stages excluded
else:
    _CHUNK_NUM_STAGES = [2, 3, 4]
    _RECOMPUTE_W_U_NUM_STAGES = [2, 3, 4]
```

**Why `num_stages=4` crashes on gfx950:** With 4 pipeline stages and a `u` loop (trip count 2), the Triton compiler emits a third `cp.async` prefetch (an LDS read ahead 3 iterations). But with only 2 total loop iterations, the third prefetch reads from a future iteration that never executes — the buffer location is then reused by the `w` loop, causing non-deterministic memory corruption. Observed at sequences ≥ 4096 tokens.

This is a hardware + Triton compiler interaction specific to gfx950's pipeline filling behavior.

### 8.2 `chunk_intra.py` precision difference

```python
# chunk_intra.py:
solve_tril_dot_precision = (
    "tf32"
    if current_platform.is_cuda() and current_platform.has_device_capability(80)
    else "ieee"
)
```

- **NVIDIA SM80+ (A100, H100):** TF32 precision for triangular solve dot products — approximately 8× faster, with ~3 decimal digits of precision loss (acceptable for routing logits)
- **AMD (all GPUs):** IEEE FP32 — full precision, no TF32 mode available on AMD hardware

**Performance implication:** The intra-chunk attention on AMD is slower than on NVIDIA for long sequences due to full-precision computation.

### 8.3 `fused_recurrent.py` gate handling

```python
# fused_recurrent.py:
if current_platform.is_rocm():
    fuse_gate = True   # gfx950 requires fused gate for numerical stability
else:
    fuse_gate = A_log is not None and dt_bias is not None  # conditional on inputs
```

On gfx950, the gate must be fused into the recurrent kernel (not applied separately) to avoid an intermediate tensor that causes numerical instability. On NVIDIA, gate fusion is optional and only enabled when specific inputs (log-space A matrix and dt bias) are present.

---

## 9. MTP: Functionally Equivalent

The AMD MTP implementation (`amd/mtp.py`) is nearly identical to NVIDIA's (`nvidia/mtp.py`).

**Difference:** Import source of `KimiDecoderLayer`:

```python
# nvidia/mtp.py:
from vllm.models.kimi_k3.nvidia.model import KimiDecoderLayer

# amd/mtp.py:
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer
```

The `KimiDecoderLayer` itself is equivalent — both use the same `input_layernorm`, attention (MLA or KDA via the dispatch), `post_attention_layernorm`, and FFN (MoE or dense). The only difference is that AMD's `KimiDecoderLayer` uses the AMD MLA wrapper instead of NVIDIA's self-contained MLA.

`fused_mtp_input` (from `common/mtp.py`) is shared between both platforms — it's a platform-portable Triton kernel.

---

## 10. Model-Level and File Organization

### 10.1 Supported interfaces comparison

| Interface | NVIDIA | AMD |
|---|---|---|
| `HasInnerState` | ✓ | ✓ |
| `SupportsPP` | ✓ | ✓ |
| `MixtureOfExperts` | ✓ | ✓ |
| `IsHybrid` | ✓ | ✓ |
| `SupportsEagle3` | ✓ | ✓ |
| `SupportsMultiModal` | ✓ | ✓ |
| `SupportsQuant` | ✓ | ✓ |
| `SupportsEncoderCudaGraph` | ✓ | ✗ (no FA4 warmup) |
| `SupportsReplaySSM` | ✓ | ✗ (no RecoverSSM) |
| `MambaStateShapes` | ✓ | ✗ |

### 10.2 File organization difference

On NVIDIA, `KimiMLP`, `KimiMoE`, `KimiDecoderLayer`, `KimiLinearModel`, `KimiLinearForCausalLM` all live in `nvidia/model.py`.

On AMD, these are split:
- `amd/linear.py`: `KimiMLP`, `KimiMoE`, `KimiDecoderLayer`, `KimiLinearModel`, `KimiLinearForCausalLM`
- `amd/model.py`: `KimiK3ForConditionalGeneration` (thin wrapper around `KimiLinearForCausalLM`)

This is a structural difference only. The class hierarchy and logic are equivalent.

---

## 11. AMD-Specific Tests

All AMD tests are gated by:
```python
pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
```

| Test file | What it validates |
|---|---|
| `tests/models/kimi_k3/test_amd_attn_res.py` | `amd/ops/attn_res.attn_res()` vs PyTorch reference. Parametrizes `(num_tokens, num_blocks, block_capacity, hidden_size, row_padding)`. Tests both "write block" and "delta" path. |
| `tests/models/kimi_k3/test_amd_kda_decode.py` | Fused HIP decode kernel vs 3-kernel Triton fallback. Gated by `on_gfx950() and hasattr(torch.ops._C, "fused_kda_decode")`. Covers batched decode, mixed batch sizes, output-gate epilogue. |
| `tests/models/kimi_k3/test_amd_latent_moe_runner.py` | Multi-GPU test (spawns subprocesses). Validates `ROCmLatentMoERunner._shard_up_proj_tail()` against replicated up-projection reference. Parametrized over TP sizes. |

**No AMD-specific tests for:** MLA fused kernels (missing), DSpark (missing), MegaMoE (missing), GEMM-RS (missing), RecoverSSM (missing), low-latency GEMM (missing), CuTe-DSL (missing).

---

## 12. Worked Examples: AMD vs NVIDIA Execution

### 12.1 KDA decode on gfx942 vs gfx950 vs NVIDIA SM90

Same decode step: B=4 sequences, H=96, kDimK=kDimV=128.

**On NVIDIA SM90+ (A100 / H100):**

```
Path: ops.fused_kda_decode(...)  → 1 CUDA kernel
  Reads: conv_state  [4, 96, 4, 128] + delta_state [4, 96, 128, 128]
         = 4 × (96×4×128×2B + 96×128×128×2B) = 4 × (98KB + 3.1MB) ≈ 12.8MB
  Writes: updated conv_state + delta_state + output [4, 96, 128]
  Kernel: 4 CUDA blocks (one per sequence), 256 threads each
  kLanes=8  (quarter-warp, 32÷4=8)
  kGroups=256÷8=32
  kRows=128÷32=4
  Cross-lane: __shfl_xor_sync()
```

**On AMD gfx950 (MI350X):**

```
Path: ops.fused_kda_decode(...)  → 1 HIP kernel  (same algorithm, different intrinsics)
  Reads/Writes: same 12.8MB
  Kernel: 4 blocks, 256 threads each
  kLanes=16  (half-wavefront, 64÷4=16)
  kGroups=256÷16=16
  kRows=128÷16=8
  Cross-lane: __builtin_amdgcn_update_dpp with dpp_add instead of __shfl_xor_sync
  Vector loads: float __attribute__((ext_vector_type(4))) + __builtin_nontemporal_load
```

**On AMD gfx942 (MI300X) — fused kernel not supported, fallback to 3 Triton kernels:**

```
Kernel 1: causal_conv1d_update
  Updates conv_state [4, 96, 4, 128] in-place
  Produces packed_conv_out [4, 96, 128]

Kernel 2: fused_recurrent_kda_packed_decode
  BV=32 tile over kDimV=128 → ceil(128/32)=4 Triton programs per (batch×head)
  Grid: [4 batches, 4×96] = [4, 384] programs
  Updates delta_state [4, 96, 128, 128] in-place

Kernel 3: o_norm (FusedRMSNormGated)
  Applies output gate g2 to produce final output [4, 96, 128]

Total: 3 kernel launches vs 1 on NVIDIA/gfx950
  Each launch has overhead ~2–5µs; 3× overhead adds up across 93 KDA layers per step.
```

### 12.2 AttnRes Triton vs SM100 native kernel

Same AttnRes call: T=1 decode token, num_blocks=8, hidden_size=7168.

**NVIDIA SM100 native** (selected when `_native_op_available()` is True and `hidden_size==7168`):

```
ops.kimi_k3_attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, ...)
  Hardware: warp-specialized TMA + TMEM kernel (attn_res_kernel.cu)
  1 producer warp:
    TMA bulk-load blocks [1, 8, 7168] = 8 × 7168 × 2B = 114KB → SMEM in one instruction
  8 consumer warps:
    Compute online softmax over 8+1=9 sources (8 blocks + prefix) in TMEM
    Weighted sum → output [1, 7168]
  Estimated latency: ~2µs for T=1
```

**Triton fallback** (always used on AMD; used on non-SM100 NVIDIA):

```python
_attn_res_kernel[(1, 1)](prefix, delta, blocks, ...)
# grid=(T=1, D//BLOCK_D=1)
# BLOCK_D = next_power_of_2(7168) = 8192  (rounds up to next power of 2 for tl.load)
# BLOCK_L = 1                              (T=1 decode path)
# One Triton program processes all 9 sources (8 blocks + prefix) sequentially:
#   for src_idx in range(9):
#     score[src_idx] = dot(query, key[src_idx])
#   weights = softmax(scores)
#   output = sum(weights[i] * value[i] for i in range(9))
# Estimated latency: ~5µs for T=1  (~2.5× slower than native SM100)
```

The Triton kernel produces numerically identical results to the native kernel; the difference is purely in throughput.

### 12.3 Triton KDA chunk.py autotuning result

Scenario: sequence length T=4096, H=96, kDimK=kDimV=128, chunk_size=64 → 64 chunks.

**NVIDIA best autotune config** (selected from `[4, 8, 16, 32]` warps × `[2, 3, 4]` stages × `[16, 32]` BV):

```
num_warps=16
num_stages=4
BV=16
Estimated throughput: ~800 GFLOPS for the chunk scan
```

**AMD gfx950 best autotune config** (selected from `[2, 4, 8, 16]` warps × `[2, 3]` stages × `[16, 32]` BV):

```
num_warps=8   (16 wavefronts × 64 threads = 1024 total, but 8 wins for occupancy)
num_stages=3  (4 excluded — gfx950 LDS race bug with trip count 2 in the u-loop)
BV=16
Estimated throughput: ~500 GFLOPS for the chunk scan
→ ~37% slower than NVIDIA at the same nominal FLOP count

Contributing factors:
  - num_stages=3 vs 4: one fewer prefetch stage, lower pipeline utilisation
  - IEEE FP32 in chunk_intra.py vs TF32 on NVIDIA: ~2–4× more FP32 ops for triangular solve
  - Wavefront width 64 vs 32: same thread count but higher register file pressure per CU
```

---

## Summary: AMD vs NVIDIA At a Glance

```
AMD K3 = NVIDIA K3 minus:
  - Fused MLA KV cache insert kernels (uses generic vLLM MLA path)
  - fp8_ds_mla KV cache format
  - g_proj aux-stream overlap
  - DCP manager in MLA
  - GEMM-RS output projection fusion
  - FlashKDA CUDA prefill (Triton only)
  - Fused KDA decode on gfx942 (Triton fallback, 3 kernels)
  - DSpark draft model
  - RecoverSSM state replay
  - MegaMoE (FP8/FP4 expert GEMM)
  - CuTe-DSL TAIL_FUSION tier in LatentMoERunner
  - Sequence parallelism
  - Multi-stream MoE overlap
  - SM100 native AttnRes kernel

AMD K3 has:
  - Fused KDA decode on gfx950 (MI350X) via HIP kernel
  - Device-side chunk metadata for KDA (avoids H2D transfer)
  - fused_recurrent_kda_packed_decode (reduces kernel launches on gfx942)
  - ROCm-tuned Triton KDA autotune (wavefront-aware warp counts)
  - Safety fix: num_stages=4 excluded from KDA autotuning (gfx950 LDS race)
```
