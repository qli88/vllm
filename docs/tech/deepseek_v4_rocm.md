# DeepSeek V4: ROCm Execution Paths

[← Index](deepseek_index.md) | [Attention](deepseek_v4_attention.md) | [MoE](deepseek_v4_moe.md)

All ROCm-specific divergences from the CUDA path. Organized by operation. Every fork is explained with the dispatch logic, the specific kernel/library used per GPU variant (gfx942=MI300X, gfx950=MI400-class), and the reason for the difference.

---

## Table of Contents

1. [ROCm vs CUDA: Complete Difference Map](#1-rocm-vs-cuda-complete-difference-map)
2. [FP8 Format: OCP vs FNUZ](#2-fp8-format-ocp-vs-fnuz)
3. [Indexer K Cache: Triton SHUFFLE Layout](#3-indexer-k-cache-triton-shuffle-layout)
   - 3.1 [Write: `indexer_k_quant_and_cache_triton`](#31-write-indexer_k_quant_and_cache_triton)
   - 3.2 [Gather: `cp_gather_indexer_k_quant_cache_triton`](#32-gather-cp_gather_indexer_k_quant_cache_triton)
4. [Prefill MQA Logits: AITER / FlyDSL / Torch Fallback](#4-prefill-mqa-logits-aiter--flydsl--torch-fallback)
5. [Decode MQA Logits: gfx942/gfx950 vs Other ROCm](#5-decode-mqa-logits-gfx942gfx950-vs-other-rocm)
6. [Sparse Attention: Triton BF16 Ragged Kernels](#6-sparse-attention-triton-bf16-ragged-kernels)
   - 6.1 [Why ragged instead of FlashMLA](#61-why-ragged-instead-of-flashmla)
   - 6.2 [Ragged index format and builders](#62-ragged-index-format-and-builders)
   - 6.3 [`rocm_sparse_attn_prefill`](#63-rocm_sparse_attn_prefill)
   - 6.4 [`rocm_sparse_attn_decode`](#64-rocm_sparse_attn_decode)
   - 6.5 [SWA cache FP8 format in the decode kernel](#65-swa-cache-fp8-format-in-the-decode-kernel)
7. [Inverse RoPE and wo_a: Fused Triton + Cached BF16](#7-inverse-rope-and-wo_a-fused-triton--cached-bf16)
   - 7.1 [`_fused_inverse_rope_gptj` Triton kernel](#71-_fused_inverse_rope_gptj-triton-kernel)
   - 7.2 [Cached BF16 wo_a dequantization](#72-cached-bf16-wo_a-dequantization)
8. [Preshuffled GEMM for `fused_wqa_wkv` and `wo_b`](#8-preshuffled-gemm-for-fused_wqa_wkv-and-wo_b)
9. [C128A Decode: Ragged Buffer and Adaptive Splits (gfx950)](#9-c128a-decode-ragged-buffer-and-adaptive-splits-gfx950)
10. [MoE: AITER `fused_moe` and gfx950 Padding Fix](#10-moe-aiter-fused_moe-and-gfx950-padding-fix)
11. [ROCm Backend Class Hierarchy](#11-rocm-backend-class-hierarchy)
12. [Worked Examples: ROCm vs CUDA Execution](#12-worked-examples-rocm-vs-cuda-execution)
    - [Example 1: Indexer K cache write — SHUFFLE vs NORMAL layout](#example-1-indexer-k-cache-write--shuffle-vs-normal-layout)
    - [Example 2: gfx942 vs gfx950 decode MQA logits dispatch](#example-2-gfx942-vs-gfx950-decode-mqa-logits-dispatch)
    - [Example 3: Ragged index format — SWA decode at B=3](#example-3-ragged-index-format--swa-decode-at-b3)
13. [Key Implementation Files](#13-key-implementation-files)

---

## 1. ROCm vs CUDA: Complete Difference Map

| Operation | CUDA path | ROCm path | File |
|---|---|---|---|
| Indexer K cache write | `indexer_k_quant_and_cache` (C++ CUDA) | `indexer_k_quant_and_cache_triton` (Triton) | `rocm_aiter_mla_sparse.py` |
| Indexer K cache gather (prefill) | `cp_gather_indexer_k_quant_cache` (C++ CUDA) | `cp_gather_indexer_k_quant_cache_triton` (Triton) | `rocm_aiter_mla_sparse.py` |
| Prefill MQA logits | `fp8_fp4_mqa_logits` (DeepGEMM JIT) | `rocm_fp8_mqa_logits` → FlyDSL/AITER/torch | `rocm_aiter_mla_sparse.py` |
| Decode MQA logits | `fp8_fp4_paged_mqa_logits` (DeepGEMM JIT) | `rocm_fp8_paged_mqa_logits` → AITER/torch | `rocm_aiter_mla_sparse.py` |
| Top-K selection | `cooperative_topk` / `persistent_topk` | `top_k_per_row_prefill` / `top_k_per_row_decode` | `sampler.cu` / `topk.cu` |
| MLA attention decode | `flash_mla_with_kvcache` (FlashMLA C++) | `rocm_sparse_attn_decode` (Triton ragged) | `rocm_aiter_mla_sparse.py` |
| MLA attention prefill | `flash_mla_sparse_fwd` (FlashMLA C++) | `rocm_sparse_attn_prefill` (Triton ragged) | `rocm_aiter_mla_sparse.py` |
| Inverse RoPE + wo_a | `fused_inv_rope_fp8_quant` + `fp8_einsum` | `rocm_inv_rope_einsum` (Triton + BF16 einsum) | `rocm_aiter_mla_sparse.py` |
| fused_wqa_wkv GEMM | Standard FP8 linear | Preshuffled `gemm_a8w8_blockscale_bpreshuffle` (if AITER) | `rocm.py` |
| wo_b GEMM | Standard RowParallel | Preshuffled GEMM (if AITER) | `rocm.py` |
| Decode index format | Dense `[T, 1, K]` (FlashMLA native) | Ragged `(flat_indices, indptr)` | `rocm.py` |
| FP8 dtype (SWA/prefill) | OCP `float8_e4m3fn` (max=448) | gfx942: FNUZ `float8_e4m3fnuz` (max=224); gfx950: OCP | platform detection |
| Expert GEMM | FP8 DeepGEMM / cuBLAS | AITER `fused_moe` BLOCK_1X32 MXFP4 | `rocm_aiter_moe.py` |
| C128A decode indices | Dense `c128a_global_decode_topk_indices` | Dense + ragged conversion via `build_ragged_indices_from_dense` | `rocm.py` |

---

## 2. FP8 Format: OCP vs FNUZ

The two FP8 formats differ in their bit pattern interpretation:

| Property | OCP `float8_e4m3fn` | FNUZ `float8_e4m3fnuz` |
|---|---|---|
| Platform | CUDA SM89+, gfx950 | gfx942 (MI300X) |
| Max representable value | **448.0** | **224.0** |
| Zero representation | Two zeros: `+0` (0x00) and `-0` (0x80) | One zero: `0x00`; `0x80` is NaN |
| NaN | `0x7f` and `0xff` | `0x80` (the old `-0` slot) |
| Name meaning | `fn` = no negative zero | `fnuz` = finite no-unsigned zero |
| PyTorch type | `torch.float8_e4m3fn` | `torch.float8_e4m3fnuz` |
| Triton type | `tl.float8e4nv` | `tl.float8e4b8` |

**Impact on quantization:**

```python
# OCP (gfx950, CUDA):
fp8_max = 448.0
scale_raw = max(amax, 1e-4) / 448.0
ue8m0_byte = ceil(log2(scale_raw)) + 127

# FNUZ (gfx942):
fp8_max = 224.0  (half the range!)
scale_raw = max(amax, 1e-4) / 224.0
ue8m0_byte = ceil(log2(scale_raw)) + 127
# → scale is 2× larger for same amax
# → quantized values are half as large → less precision in the mantissa
```

**Where the format matters:**

1. **SWA cache write** (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`): C++ encoder writes the format native to the GPU. gfx942 writes FNUZ; gfx950 writes OCP.

2. **Indexer K cache write** (`indexer_k_quant_and_cache_triton`): Triton kernel uses `IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz` to select the right quantization and encoding.

3. **Compressor output** (main MLA KV cache and indexer K cache from compressor): always writes OCP. The Triton compressor kernels use OCP everywhere regardless of platform, because they were written for portability.

4. **Sparse decode kernel**: separate `IS_FNUZ_MAIN` (SWA cache, platform-dependent) and `IS_FNUZ_EXTRA` (compressed cache, always OCP=False) flags allow the kernel to dequantize each cache type correctly.

**Example — dequantization in sparse decode kernel:**

```triton
@triton.jit
def _sparse_attn_decode_ragged_kernel(..., IS_FNUZ_MAIN: tl.constexpr, IS_FNUZ_EXTRA: tl.constexpr):
    # Load NoPE FP8 values from SWA cache:
    x_uint8 = tl.load(token_data_ptr + nope_offsets)

    if IS_FNUZ_MAIN:
        # gfx942: reinterpret as FNUZ, convert to bf16
        x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
    else:
        # gfx950 / OCP: reinterpret as OCP, convert to bf16
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)

    # Dequantize with UE8M0 scale:
    encoded_scale = tl.load(scale_ptr)
    scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    k_nope = x_fp8.to(tl.bfloat16) * scale.to(tl.bfloat16)
```

---

## 3. Indexer K Cache: Triton SHUFFLE Layout

### 3.1 Write: `indexer_k_quant_and_cache_triton`

**Why Triton instead of CUDA C++:** The CUDA C++ kernel (`indexer_k_quant_and_cache` in `cache_kernels.cu`) uses CUDA-specific intrinsics. ROCm uses a Triton port that also supports a "SHUFFLE" layout for better memory coalescing on AMD hardware.

**SHUFFLE layout:** When `block_size > 1`, values within a block are stored in tile-interleaved order rather than row-major. This matches the memory access pattern of the AITER MQA logit kernel.

```
NORMAL layout (block_size=1 or debug):
  Cache logically: [block, slot, dim]
  Physical: cache[block, slot, d] at byte = block*stride + slot*head_dim + d

SHUFFLE layout (block_size=16, HEAD_TILE=16, BLOCK_TILE=16):
  Physical offset for cache[block, slot, d]:
    tile_block_id    = slot // 16
    tile_block_offset = slot % 16
    offset = block*stride + tile_block_id * 16 * head_dim + tile_block_offset * 16 + (d // 16)*16*16 + (d % 16)
  
  Effect: 16 consecutive slots have their data interleaved in 16-byte tiles.
  Access pattern for AITER: when kernel reads 16 consecutive K values for 16 different slots,
  those 16 values are now contiguous in memory → coalesced load.
```

**Kernel dispatch:**

```python
def indexer_k_quant_and_cache_triton(k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
                                      block_tile_size=16, head_tile_size=16):
    block_size = kv_cache.shape[1]
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"

    _indexer_k_quant_and_cache_kernel[grid=(num_tokens,)](
        k, kv_cache_value, kv_cache_scale, slot_mapping,
        ...,
        LAYOUT=layout,          # compile-time constant: "NORMAL" or "SHUFFLE"
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )
```

**Quantization in the kernel:**

```triton
@triton.jit
def _indexer_k_quant_and_cache_kernel(..., IS_FNUZ: tl.constexpr, USE_UE8M0: tl.constexpr):
    val = tl.load(src_ptr + offset)   # [head_dim] float values
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)

    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0   # FNUZ max
    else:
        scale = tl.maximum(1e-4, amax) / 448.0   # OCP max

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))  # round to nearest power-of-2

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    # Write fp8_val to SHUFFLE or NORMAL offset
    # Write scale to kv_cache_scale
```

**Concrete example (head_dim=128, block_size=16, SHUFFLE):**

```
token t=5 → slot_id=37 in paged cache
block_id = 37 // 16 = 2
block_offset = 37 % 16 = 5
tile_block_id = 5 // 16 = 0
tile_block_offset = 5 % 16 = 5

For dim d=20:
  NORMAL: offset = block_id*stride + 5*128 + 20
  SHUFFLE: offset = block_id*stride + 0*(16*128) + 5*16 + (20//16)*16*16 + (20%16)
                  = block_id*stride + 0 + 80 + 1*256 + 4
                  = block_id*stride + 340
  (vs NORMAL: block_id*stride + 660)

SHUFFLE groups dim d=16..31 (the second 16-dim tile) together for all 16 slots,
enabling coalesced access when the AITER kernel reads all 16 slots' d=16..31 simultaneously.
```

### 3.2 Gather: `cp_gather_indexer_k_quant_cache_triton`

Two kernel variants dispatched at runtime:

```python
if _ON_GFX950:
    _cp_gather_indexer_quant_cache_gfx950_kernel[grid](*kernel_args, num_batches, ...)
else:
    _cp_gather_indexer_quant_cache_kernel[grid](*kernel_args, num_tokens, num_batches, ...)
```

**gfx950 kernel difference:** Removes the `valid_tid = tid < NUM_TOKENS` bounds check as a `constexpr` path. On gfx950's shader architecture this bounds check adds overhead even when always true; removing it and relying on the grid size to not launch excess threads is faster.

Both kernels:
1. Use `token_to_seq [T]` (precomputed by metadata builder) to find which batch the token belongs to — faster than the CUDA C++ kernel's `cu_seqlen` binary search.
2. Apply SHUFFLE → NORMAL address mapping in reverse (gather from SHUFFLE-laid-out cache into contiguous output).
3. Read float32 scale from `kv_cache_scale` (separate from `kv_cache_value`).

---

## 4. Prefill MQA Logits: AITER / FlyDSL / Torch Fallback

`rocm_fp8_mqa_logits` dispatches in priority order:

```python
def rocm_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    k_fp8, scale = kv

    # Priority 1: gfx942 + AITER enabled → FlyDSL (fastest on MI300X)
    if _ON_GFX942 and rocm_aiter_ops.is_enabled():
        from aiter.ops.flydsl import flydsl_fp8_mqa_logits
        return flydsl_fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    # Priority 2: AITER available (any ROCm platform, including RDNA)
    aiter_module = mqa_logits_module()   # looks for aiter.ops.triton.fp8_mqa_logits
    if aiter_module is not None:
        return aiter_module.fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    # Fallback: pure PyTorch reference
    return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
```

**FlyDSL (gfx942/MI300X):** AMD's JIT-compiled kernel DSL. The `flydsl_fp8_mqa_logits` kernel is precompiled for the exact matrix shape at the first call. Fastest option on gfx942 because it is tuned to the matrix layout and register file of the MI300X compute units.

**AITER Triton (other ROCm):** A Triton kernel in the AITER library that handles arbitrary shapes. Slower than FlyDSL but more portable.

**PyTorch fallback:**

```python
def fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    k_fp8, scale = kv
    # Manual: dequantize K, compute einsum, relu, weight, mask
    score = torch.einsum("mhd,nd->hmn", q.to(bf16), k_fp8.to(bf16)).float() * scale.reshape(-1)
    logits = (score.relu() * weights.T[:, :, None]).sum(dim=0)
    # Apply causal mask:
    mask = (arange[None, :] >= cu_seqlen_ks[:, None]) & (arange[None, :] < cu_seqlen_ke[:, None])
    return logits.masked_fill(~mask, float("-inf"))
```

**Note:** The `relu` in the torch fallback (and in the AITER kernel) corresponds to the non-negativity of `sqrtsoftplus`-derived weights. All scores are positive, so after multiplying by `weights_out` the logits are non-negative. The logit formula is:

```
logit = sum_h( dot(q[h,:], k[:]) × sk × weights_out[h] )
                                    ↑ all positive (sqrtsoftplus scores × positive scale)
→ logit ≥ 0 always, so relu is a no-op mathematically
→ but the kernel still includes it for numerical safety
```

---

## 5. Decode MQA Logits: gfx942/gfx950 vs Other ROCm

`rocm_fp8_paged_mqa_logits` dispatches:

```python
def rocm_fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens,
                               block_tables, schedule_metadata, max_model_len):
    aiter_module = paged_mqa_logits_module()

    if aiter_module is not None and (rocm_aiter_ops.is_enabled() or is_rdna_aiter_enabled()):

        if _ON_GFX942 or _ON_GFX950:
            # Single-pass: produces [B*next_n, max_model_len] float32 directly
            out_logits = workspace.get((batch*next_n, max_model_len), torch.float32)
            aiter_module.deepgemm_fp8_paged_mqa_logits(
                q_fp8, kv_cache_fp8, weights, out_logits,
                context_lens, block_tables, max_model_len,
                ChunkK=256,                  # K-dim tile size for block traversal
                Preshuffle=block_size > 1,   # match SHUFFLE layout of cache
                KVBlockSize=block_size,
                WavePerEU=2,                 # 2 wavefronts per EU for latency hiding
            )
            out_logits.nan_to_num_(float("-inf"))   # fill unvisited positions
            return out_logits

        else:
            # Two-pass: stage1 → [heads, B*next_n, max_model_len], then sum over heads
            out_qk = workspace.get((heads, batch*next_n, max_model_len), torch.float32)
            out_qk.fill_(float("-inf"))
            aiter_module.deepgemm_fp8_paged_mqa_logits_stage1(
                q_fp8, kv_cache_fp8, weights, out_qk,
                context_lens, block_tables, max_model_len,
                ChunkQ=heads,               # process all heads in one pass per Q tile
            )
            return out_qk.sum(dim=0)        # [B*next_n, max_model_len]

    else:
        return fp8_paged_mqa_logits_torch(q_fp8, kv_cache_fp8, weights,
                                          context_lens, block_tables, max_model_len)
```

**gfx942/gfx950 single-pass vs two-pass:**

The single-pass kernel (`deepgemm_fp8_paged_mqa_logits`) computes the weighted sum across all 64 indexer heads inside the kernel, producing the final scalar logit per (decode_token, compressed_kv_position) pair directly.

The two-pass variant first computes per-head logits (shape `[heads, B*next_n, max_model_len]`), then sums. This is used on other ROCm platforms where the single-pass implementation is not optimized. The extra sum step is cheap (linear scan of the output tensor).

**`ChunkK=256`:** The kernel processes 256 consecutive compressed KV positions per iteration. This controls the block size for walking the paged KV cache — smaller values reduce register pressure, larger values improve cache reuse.

**`Preshuffle=block_size > 1`:** Tells the kernel to apply the SHUFFLE address mapping when reading from the paged cache. Must match whether the cache was written with SHUFFLE layout.

---

## 6. Sparse Attention: Triton BF16 Ragged Kernels

On ROCm, `flash_mla_with_kvcache` and `flash_mla_sparse_fwd` (FlashMLA compiled C++ kernels) are replaced by two Triton kernels: `rocm_sparse_attn_decode` and `rocm_sparse_attn_prefill`.

### 6.1 Why ragged instead of FlashMLA

FlashMLA is a CUDA-specific library that uses CUDA-native intrinsics (TMA, wgmma, etc.). Porting it to ROCm would require rewriting it using HIP/AMDGPU intrinsics — a substantial engineering effort. The Triton alternative:
- Works on any ROCm GPU via Triton's HIP backend
- Uses BF16 math throughout (no FP8 kernel inside attention — dequantization happens in the kernel)
- Uses ragged (CSR) index format instead of dense — avoids padding overhead for variable-length sparse sets

### 6.2 Ragged index format and builders

Dense format (CUDA FlashMLA uses): `indices [T, 1, K]` — all rows padded to length K.

Ragged format (ROCm Triton uses): `(flat_indices [NNZ], indptr [T+1])` — each row has variable length.

```python
# Build ragged from dense:
def build_ragged_indices_from_dense(indices, lengths):
    # indices: [T, K]  int32
    # lengths: [T]     int32  (valid entries per row)
    indptr = cumsum([0] + lengths)              # [T+1]
    flat = pack valid entries from indices      # [NNZ]
    return flat, indptr

# Copying into graph-stable buffers:
def _copy_ragged_to_graph_buffers(ragged_indices, ragged_indptr,
                                   ragged_indices_buffer, ragged_indptr_buffer, ...):
    # Copy dynamic ragged tensors into pre-allocated persistent buffers
    # so CUDA graph captures always see the same base pointer.
    indptr_out = ragged_indptr_buffer[:num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)
    ragged_out = ragged_indices_buffer[:source_entries]
    ragged_out[:source_entries].copy_(ragged_indices, non_blocking=True)
    if _ON_GFX950:
        ragged_out = ragged_out[:max(source_entries, 1)]   # keep ≥1 entry for graph stability
    return ragged_out, indptr_out
```

**Concrete example:**

```
T = 3 decode tokens, each with different sparse token counts:
  token 0: 4 valid entries    [12, 7, 30, 45]
  token 1: 2 valid entries    [8, 19]
  token 2: 6 valid entries    [3, 41, 52, 17, 28, 60]

Dense (K=6, padded with -1):
  [[12, 7, 30, 45, -1, -1],
   [8, 19, -1, -1, -1, -1],
   [3, 41, 52, 17, 28, 60]]

Ragged:
  flat_indices = [12, 7, 30, 45, 8, 19, 3, 41, 52, 17, 28, 60]   NNZ=12
  indptr = [0, 4, 6, 12]   (row 0 in [0:4], row 1 in [4:6], row 2 in [6:12])

Advantage: Triton kernel reads indptr[t] to indptr[t+1] without processing -1 entries.
  Token 0: reads flat_indices[0:4]   = [12, 7, 30, 45]
  Token 1: reads flat_indices[4:6]   = [8, 19]
  Token 2: reads flat_indices[6:12]  = [3, 41, 52, 17, 28, 60]
```

### 6.3 `rocm_sparse_attn_prefill`

**Triton kernel** `_sparse_attn_prefill_ragged_kernel`, grid `(T_q, ceil(n_heads/BLOCK_H))`:

```triton
@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr, kv_ptr, kv_indices_ptr, kv_indptr_ptr, attn_sink_ptr, out_ptr,
    ..., scale, HAS_ATTN_SINK: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    query_idx = tl.program_id(0)   # which query token
    pid_h = tl.program_id(1)       # which head group

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)

    # Load BLOCK_H query heads (full 512-dim each):
    q = tl.load(q_ptr + query_idx * q_stride_t + head_offsets[:, None] * q_stride_h + dim_offsets[None, :])
    # q: [BLOCK_H, BLOCK_D]

    # Online softmax accumulators:
    m_i = -inf   # running max
    l_i = 0.0    # running denominator
    acc = 0.0    # running numerator

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end   = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len   = kv_end - kv_start

    # Process BLOCK_K KV entries at a time:
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
        in_range = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        # Load BF16 KV from flat workspace (already dequantized):
        kv = tl.load(kv_ptr + safe_slot[:, None] * kv_stride_n + dim_offsets[None, :],
                     mask=valid[:, None])
        # kv: [BLOCK_K, BLOCK_D]

        scores = tl.dot(q, tl.trans(kv)) * scale   # [BLOCK_H, BLOCK_K]
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, -inf)

        # Online softmax update:
        m_block = tl.max(scores, axis=1)         # [BLOCK_H]
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # attn_sink: add learned per-head denominator bias
    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_offsets)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        out = (acc * alpha[:, None]) / tl.maximum(l_final[:, None], 1e-30)
    else:
        out = acc / tl.maximum(l_i[:, None], 1e-30)

    tl.store(out_ptr + ...)
```

**Key properties:**
- Online softmax (no need to store all scores before softmax → O(1) memory in attention dim)
- Ragged index: skips invalid/padding slots efficiently
- Workspace KV is BF16 (pre-dequantized by `dequantize_and_gather_k_cache`) → no FP8 dequant inside attention
- `HAS_ATTN_SINK` is a compile-time constant → zero overhead when sink is disabled

### 6.4 `rocm_sparse_attn_decode`

**Triton kernel** `_sparse_attn_decode_ragged_kernel`, handles both SWA and compressed sparse in one kernel pass.

```triton
@triton.jit
def _sparse_attn_decode_ragged_kernel(
    q_ptr,
    main_cache_ptr, main_indices_ptr, main_indptr_ptr,   # SWA cache
    extra_cache_ptr, extra_indices_ptr, extra_indptr_ptr, # compressed cache
    attn_sink_ptr, out_ptr,
    ...,
    HAS_ATTN_SINK: tl.constexpr, HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
    IS_FNUZ_MAIN: tl.constexpr,   # SWA cache: OCP (gfx950) or FNUZ (gfx942)
    IS_FNUZ_EXTRA: tl.constexpr,  # Compressed cache: always OCP
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load query NoPE and RoPE parts separately:
    q_nope = tl.load(q_ptr + query_idx*q_stride0 + heads*q_stride1 + nope_offsets[None, :])
    q_rope = tl.load(q_ptr + query_idx*q_stride0 + heads*q_stride1 + NOPE_DIM + rope_offsets[None, :])
    # q_nope: [BLOCK_H, NOPE_DIM=448]
    # q_rope: [BLOCK_H, ROPE_DIM=64]

    # Separate accumulators for NoPE and RoPE output:
    acc_nope = 0.0   # [BLOCK_H, 448]
    acc_rope = 0.0   # [BLOCK_H, 64]

    # SWA tokens (main cache):
    for each SWA slot in main_indices[main_start:main_end]:
        # Read FP8 from paged SWA cache:
        # Layout: block*stride → pos_in_block * 576 bytes for data, separate scale
        token_data_ptr = main_cache + block_idx*stride + pos_in_block*576
        token_scale_ptr = main_cache + block_idx*stride + block_size*576 + pos_in_block*8

        # Load 448 NoPE FP8 bytes:
        x_uint8 = tl.load(token_data_ptr + nope_offsets)
        if IS_FNUZ_MAIN:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)    # FNUZ on gfx942
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)    # OCP on gfx950
        # Decode UE8M0 scales:
        encoded = tl.load(token_scale_ptr + nope_offsets//64)   # one scale per 64 elements
        scales = tl.exp2(encoded.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)

        # Load 64 RoPE BF16 bytes (stored as-is, not quantized):
        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(rope_ptr + rope_offsets)

        # Attention score:
        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        # Online softmax update + accumulate...

    # Compressed tokens (extra cache, HAS_EXTRA=True when swa_only=False):
    if HAS_EXTRA:
        for each compressed slot in extra_indices[extra_start:extra_end]:
            # Same structure as SWA but IS_FNUZ_EXTRA=False always (OCP)
            # extra cache was written by Triton compressor (OCP everywhere)
            ...
```

**`HAS_EXTRA`:** When a layer has no compressed long-range cache (SWA-only mode, or sequence too short for any compression), `HAS_EXTRA=False` is compiled in, eliminating the extra-cache loop at zero runtime cost.

**Score formula:**

```
score[h, k] = dot(q_nope[h, :448], k_nope[k, :448])
            + dot(q_rope[h, :64],  k_rope[k, :64])
# Both NoPE and RoPE contributions summed before softmax
# scale = 1/√512 (over the full 512-dim head)
```

This computes the same value as FlashMLA's FP8 path, but in BF16 throughout.

### 6.5 SWA cache FP8 format in the decode kernel

The SWA cache layout inside `_sparse_attn_decode_ragged_kernel`:

```
# Physical layout in swa_kv_cache (typed as uint8):
# Per CUDA block (paged): block has block_size slots
#
# For slot pos_in_block within block block_idx:
#   Data:   main_cache_ptr + block_idx*stride + pos_in_block * 576
#     [0:448]   = 448 uint8 = 448 FP8 values (NoPE)
#     [448:576] = 128 uint8 = 64 BF16 values as raw bytes (RoPE)
#   Scale:  main_cache_ptr + block_idx*stride + block_size*576 + pos_in_block * 8
#     [0:7]   = 7 uint8 = 7 UE8M0 scale bytes (one per 64-element NoPE block)
#     [7]     = 1 uint8 = padding

# On gfx950, a specialized function handles the 448-dim load in 128-element chunks
# for exact FP8 handling (two separate load functions: one for dims 0..383, one for 384..447):
_load_fp8_ds_mla_gfx950_nope_exact_chunk(token_data_ptr, token_scale_ptr, valid,
    CHUNK_START=0, CHUNK_SIZE=128, BLOCK_K=BLOCK_K)   # dims 0-127
_load_fp8_ds_mla_gfx950_nope_exact_chunk(..., CHUNK_START=128, CHUNK_SIZE=128)  # dims 128-255
_load_fp8_ds_mla_gfx950_nope_exact_chunk(..., CHUNK_START=256, CHUNK_SIZE=128)  # dims 256-383
_load_fp8_ds_mla_gfx950_tail128(token_data_ptr, token_scale_ptr, valid, NOPE_DIM=448)  # dims 384-447 + rope
```

---

## 7. Inverse RoPE and wo_a: Fused Triton + Cached BF16

On CUDA: `fused_inv_rope_fp8_quant` + DeepGEMM FP8 `fp8_einsum`.
On ROCm: `rocm_inv_rope_einsum` — Triton inverse RoPE writing BF16 + standard BF16 einsum.

### 7.1 `_fused_inverse_rope_gptj` Triton kernel

Grid `(T, n_heads)`, one program per `(token, head)`:

```triton
@triton.jit
def _inverse_rope_gptj_kernel(
    o_ptr, out_ptr, pos_ptr, cos_sin_ptr,
    s_t, s_h,   # input strides (token, head)
    os_t, os_h, # output strides
    cs_stride,  # cos_sin_cache row stride
    NOPE: tl.constexpr, HALF: tl.constexpr,
    BLOCK_NOPE: tl.constexpr, BLOCK_HALF: tl.constexpr,
):
    t = tl.program_id(0)   # token index
    h = tl.program_id(1)   # head index

    # NoPE dims: pass through, cast to bf16
    n = tl.arange(0, BLOCK_NOPE)
    vals = tl.load(o_ptr + t*s_t + h*s_h + n, mask=n < NOPE)
    tl.store(out_ptr + t*os_t + h*os_h + n, vals.to(tl.bfloat16), mask=n < NOPE)

    # RoPE dims: apply inverse GPT-J rotation
    pos = tl.load(pos_ptr + t).to(tl.int64)
    k = tl.arange(0, BLOCK_HALF)
    a = tl.load(o_ptr + t*s_t + h*s_h + NOPE + 2*k, mask=k < HALF).to(tl.float32)  # even
    b = tl.load(o_ptr + t*s_t + h*s_h + NOPE + 2*k + 1, mask=k < HALF).to(tl.float32)  # odd
    cos = tl.load(cos_sin_ptr + pos*cs_stride + k, mask=k < HALF)
    sin = tl.load(cos_sin_ptr + pos*cs_stride + HALF + k, mask=k < HALF)

    # Inverse GPT-J rotation:
    # Forward:  [a, b] → [a*cos - b*sin, b*cos + a*sin]
    # Inverse:  [y0, y1] → [y0*cos + y1*sin, y1*cos - y0*sin]
    out_even = a * cos + b * sin    # note: +b*sin (not -b*sin)
    out_odd  = b * cos - a * sin    # note: -a*sin (not +a*sin)

    tl.store(out_ptr + t*os_t + h*os_h + NOPE + 2*k,     out_even.to(tl.bfloat16), mask=k < HALF)
    tl.store(out_ptr + t*os_t + h*os_h + NOPE + 2*k + 1, out_odd.to(tl.bfloat16),  mask=k < HALF)
```

**Output:** `o_ref [T, n_heads, 512]` in BF16 with inverse RoPE applied.

**Compared to CUDA path:**

CUDA `fused_inv_rope_fp8_quant` does inverse RoPE + FP8 block quantization in one kernel. ROCm skips the FP8 quantization step because the ROCm wo_a path uses BF16, not FP8. This is simpler and avoids the dequantization overhead that CUDA must pay in `fp8_einsum`.

### 7.2 Cached BF16 wo_a dequantization

```python
def _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, hidden_dim):
    cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if cached is not None:
        return cached   # Fast path: already computed

    # First call: dequantize FP8 weight to BF16
    if hasattr(wo_a, "weight_scale_inv"):
        # FP8 weight with block scales:
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.float32)
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]),
            o_lora_rank, hidden_dim,
        )
        cached = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        # Already BF16:
        cached = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)

    wo_a._dsv4_wo_a_bf16 = cached   # Cache on the module — persists for the session
    return cached
```

**Why cache instead of dequantizing each step:**

The profiler shows `direct_copy float` (~55µs) and `MulFunctor float` (~31µs) per wo_a dequantization — total ~86µs per two layers. With 31 C128A + 30 C4A = 61 attention layers, that's ~2.6ms per decode step just for wo_a dequantization. Caching eliminates this entirely after the first call.

**The BF16 wo_a einsum:**

```python
def rocm_inv_rope_einsum(rotary_emb, o, positions, rope_head_dim, n_local_groups, o_lora_rank, wo_a):
    # Step 1: Fused inverse RoPE (BF16 output)
    o_ref = _fused_inverse_rope_gptj(o, positions, rotary_emb.cos_sin_cache, rope_head_dim)
    # o_ref: [T, n_heads, head_dim]  bf16

    # Step 2: Reshape for grouped matmul
    o_ref = o_ref.view(T, n_local_groups, -1)
    # o_ref: [T, n_groups, heads_per_group * head_dim]

    # Step 3: BF16 grouped einsum
    wo_a_weight = _get_cached_wo_a_bf16(wo_a, n_local_groups, o_lora_rank, o_ref.shape[-1])
    z = torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)
    # z: [T, n_groups, o_lora_rank]  bf16

    return z  # passed to wo_b.flatten(1) → output projection
```

---

## 8. Preshuffled GEMM for `fused_wqa_wkv` and `wo_b`

When AITER is enabled (`rocm_aiter_ops.is_enabled()`), `prepare_attn_preshuffle()` (called once at model init) transforms the weight matrices for faster GEMM:

```python
def prepare_attn_preshuffle(self) -> None:
    def _prep(linear):
        w = linear.weight           # [out, in]  fp8
        ws = linear.weight_scale_inv  # [out_blocks, in_blocks]  fp8 or ue8m0

        # Requirements:
        # - K (input dim) must be multiple of 128 (block quantization requirement)
        # - N (output dim) must be multiple of 16 (shuffle alignment)
        if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
            return None   # not eligible

        # Decode UE8M0 scales if needed
        if ws.dtype == torch.float8_e8m0fnu:
            ws = _upcast_e8m0_to_fp32(ws).contiguous()

        # Shuffle weight in-place — modifies the stored weight matrix
        replace_parameter(linear, "weight",
            rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)))
        return ws

    self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)   # None if not eligible
    self._wo_b_scale    = _prep(self.wo_b)             # None if not eligible
```

**What `shuffle_weight(w, layout=(16, 16))` does:**

Reorders the weight matrix rows/columns so that 16×16 tiles are contiguous in memory. The AITER kernel `gemm_a8w8_blockscale_bpreshuffle` expects this layout to efficiently load weight tiles into registers in coalesced memory accesses.

**Forward pass (preshuffled path):**

```python
def _bpre_attn_gemm(self, weight, scale, x, reduce_tp):
    # Dynamic quantize input to FP8:
    x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
    # x_fp8:  [T, in_features]  fp8
    # x_scale: [T_blocks, in_blocks]  (block scales, transposed for kernel)

    # FP8×FP8 GEMM with block scales:
    out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
        x_fp8, weight, x_scale, scale, output_dtype=x.dtype)
    # weight is already shuffled; scale is wo_a/wo_b's weight_scale_inv

    if reduce_tp and TP > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out
```

**When preshuffled GEMM is active:**

```python
def _fused_wqa_wkv_gemm(self, hidden_states):
    if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
        return self._bpre_attn_gemm(self.fused_wqa_wkv.weight, self._wqa_wkv_scale,
                                     hidden_states, reduce_tp=False)
    return super()._fused_wqa_wkv_gemm(hidden_states)   # fallback: standard F.linear

def _o_proj(self, o, positions):
    z = rocm_inv_rope_einsum(...)   # inverse RoPE + BF16 einsum → z [T, n_groups, o_lora_rank]
    zf = z.flatten(1)               # [T, n_groups*o_lora_rank]
    if self._wo_b_scale is not None and zf.dim() == 2:
        return self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, reduce_tp=True)
    return self.wo_b(zf)            # fallback: standard RowParallel
```

---

## 9. C128A Decode: Ragged Buffer and Adaptive Splits (gfx950)

`DeepseekV4ROCMAiterMLASparseMetadataBuilder` pre-converts C128A decode indices to ragged format and stores them in graph-stable buffers.

```python
class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    def __init__(self, ...):
        if self.compress_ratio == 128:
            max_tokens = scheduler_config.max_num_batched_tokens
            # Pre-allocate stable buffers (avoid reallocation during decode):
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed, dtype=torch.int32, device=device)
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1, dtype=torch.int32, device=device)

    def build(self, ...):
        base = super().build(...)   # computes c128a_global_decode_topk_indices [T, 1, 131072]

        dense_decode = base.c128a_global_decode_topk_indices   # [T, 1, 131072]
        decode_lens  = base.c128a_decode_topk_lens              # [T]  valid count per token

        if dense_decode is not None and decode_lens is not None:
            # Convert dense [T, 131072] to ragged (flat, indptr):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                dense_decode.reshape(T, -1), decode_lens)

            # Copy into stable buffers:
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices, ragged_indptr,
                self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                T, self.c128a_max_compressed,
            )

        return DeepseekV4ROCMAiterMLASparseMetadata(**vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr)
```

**gfx950 adaptive splits:**

For C128A decode on gfx950, when `for_cudagraph_capture=True`, the `rocm_sparse_attn_decode` call uses `adaptive_splits=True`. This allows the Triton kernel to dynamically assign more parallel CTAs to tokens with longer sparse index lists and fewer CTAs to tokens with fewer. This is more efficient than a fixed split because C128A decode index counts vary significantly across tokens in a batch (some tokens may have 0 compressed entries, others may have 1024).

---

## 10. MoE: AITER fused_moe and gfx950 Padding Fix

See [deepseek_v4_moe.md §10](deepseek_v4_moe.md#10-rocm-moe-aiter-fused_moe-and-gfx950-padding-fix) for the full ROCm MoE section. Key points:

- Uses AITER `fused_moe` with `quant_method=BLOCK_1X32` (MXFP4, 1×32 block quantization)
- `doweight_stage1=False`: weights applied at W2 output, not W1 input
- gfx950 padding crash: `topk_hash_softplus_sqrt` writes `-1` for all `topk_ids` during warmup → `fused_moe` OOB on expert weights
- Fix: during CUDA graph capture, clamp `-1→0` + zero the weights; outside graph capture, short-circuit to zeros

---

## 11. ROCm Backend Class Hierarchy

```
DeepseekV4Attention         (base class, platform-agnostic)
    ↓
DeepseekV4ROCMAiterMLAAttention   (ROCm override in amd/rocm.py)
    overrides:
      _fused_wqa_wkv_gemm()  → preshuffled AITER GEMM if available
      _o_proj()              → rocm_inv_rope_einsum + preshuffled wo_b
      forward_mqa()          → dispatches to _forward_prefill / _forward_decode
      _forward_decode()      → rocm_sparse_attn_decode (ragged Triton)
      _forward_prefill()     → dequantize_and_gather_k_cache + rocm_sparse_attn_prefill

DeepseekV4SparseMLABackend  (base metadata builder)
    ↓
DeepseekV4ROCMAiterMLASparseBackend  (ROCm metadata builder)
    get_name() → "ROCM_FLASHMLA_SPARSE_DSV4"
    get_builder_cls() → DeepseekV4ROCMAiterMLASparseMetadataBuilder

DeepseekV4SparseMLAMetadataBuilder
    ↓
DeepseekV4ROCMAiterMLASparseMetadataBuilder
    adds: c128a ragged index construction + graph-stable buffer management

DeepseekSparseSWAMetadataBuilder
    ↓
DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    adds: decode_swa_ragged_indices + decode_swa_ragged_indptr
    note: supports_draft_decode_metadata_update = False
          (fused multi-step decode disabled until SWA ragged refresh is supported)
```

**SWA metadata ragged construction:**

```python
class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    def build(self, ...):
        base = super().build(...)   # builds decode_swa_indices [T, 1, window_size]

        if base.num_decode_tokens > 0 and base.decode_swa_indices is not None:
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(T, swa_width),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices, ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                T, swa_width,
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(**vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr)
```

Both SWA and compressed decode indices are converted to ragged format so that `rocm_sparse_attn_decode` can use a single kernel interface for both.

---

## 12. Worked Examples: ROCm vs CUDA Execution

### Example 1: Indexer K cache write — SHUFFLE vs NORMAL layout

For B=4 decode tokens, block_size=16 (SHUFFLE layout active since block_size > 1), head_dim=128:

```
Token t=2 → slot_id=37 → block_id=37//16=2, block_offset=37%16=5

SHUFFLE layout address for dim d=20:
  tile_block_id    = 5 // 16 = 0  (which 16-slot tile within the block)
  tile_block_offset = 5 % 16  = 5
  tile_head_group   = 20 // 16 = 1
  tile_head_offset  = 20 % 16  = 4
  
  offset = block_id*stride + tile_block_id*16*128 + tile_block_offset*16 + tile_head_group*16*16 + tile_head_offset
         = 2*stride + 0 + 5*16 + 1*256 + 4
         = 2*stride + 340

NORMAL layout address for same (d=20):
  offset = block_id*stride + block_offset*128 + d = 2*stride + 5*128 + 20 = 2*stride + 660

SHUFFLE rearranges dim d=20 (in tile 1) to be adjacent to dims 16..31 for all 16 slots in this tile.
When AITER logit kernel reads 16 consecutive K values (dims 16..31) for all 16 slots simultaneously,
SHUFFLE layout makes those values contiguous in memory → coalesced 128B load.
NORMAL layout would scatter those reads across 16 separate 128B cache lines → 16× lower bandwidth.
```

### Example 2: gfx942 vs gfx950 decode MQA logits dispatch

Decode step, B=4 sequences, max_model_len=131072, compressed seq_len=32768 (C4A):

```
AITER enabled on gfx942:
  aiter_paged_mqa_logits_module = paged_mqa_logits_module()  # loads aiter.ops.triton.pa_mqa_logits
  _ON_GFX942 = True, _ON_GFX950 = False
  
  → deepgemm_fp8_paged_mqa_logits_stage1(
        q_fp8 [4, 1, 64, 128],
        kv_cache [num_blocks, 256, 1, 132],
        weights [4, 64],
        out_qk [64, 4, 32768],   # [heads, B, max_compressed_len]
        context_lens [[32768], ...],
        block_tables [4, max_blocks],
        ChunkQ=64,               # process all 64 heads in one pass per Q tile
    )
  return out_qk.sum(dim=0)   # [4, 32768] — sum across 64 heads

AITER enabled on gfx950:
  _ON_GFX950 = True
  
  → deepgemm_fp8_paged_mqa_logits(
        ...same args...,
        out_logits [4, 32768],   # [B, max_compressed_len] — single pass
        ChunkK=256,              # 256 consecutive compressed K entries per SM iteration
        Preshuffle=True,         # matches SHUFFLE layout written by the indexer kernel
        KVBlockSize=256,         # indexer cache block_size=256
        WavePerEU=2,             # 2 wavefronts per EU for latency hiding
    )
  out_logits.nan_to_num_(float("-inf"))
  return out_logits   # directly [4, 32768]

gfx950 is more efficient: 1 pass vs 2 passes, no intermediate sum.
gfx942 two-pass avoids register pressure at the cost of an extra sum kernel.
```

### Example 3: Ragged index format — SWA decode at B=3

```
B=3 decode tokens, swa_lens = [4096, 2000, 512]
Dense format (padded):  swa_indices [3, 1, 4096] int32
  Row 0: [slot_0, slot_1, ..., slot_4095]       (4096 valid)
  Row 1: [slot_0, slot_1, ..., slot_1999, -1, ..., -1]  (2000 valid, 2096 padding)
  Row 2: [slot_0, ..., slot_511, -1, ..., -1]   (512 valid, 3584 padding)

Ragged format (CSR):
  flat_indices = [slot0_0, slot0_1, ..., slot0_4095,   # row 0 entries (4096)
                  slot1_0, slot1_1, ..., slot1_1999,   # row 1 entries (2000)
                  slot2_0, slot2_1, ..., slot2_511]    # row 2 entries (512)
  NNZ = 4096 + 2000 + 512 = 6608
  indptr = [0, 4096, 6096, 6608]

Triton kernel reads for token 1:
  kv_start = indptr[1] = 4096, kv_end = indptr[2] = 6096
  Reads flat_indices[4096:6096] = 2000 valid entries
  No -1 checking needed; loop runs exactly 2000 iterations (no wasted iterations)

Memory saving vs dense: 3×4096 int32 = 49 KB dense vs 6608 int32 = 26 KB ragged + 4×4 = 16 B indptr
```

---

## 13. Key Implementation Files

| File | Role |
|---|---|
| `vllm/models/deepseek_v4/amd/rocm.py` | `DeepseekV4ROCMAiterMLAAttention`; preshuffled GEMM; ROCm-specific prefill/decode dispatch |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | All ROCm Triton kernels: indexer K cache, gather, `rocm_sparse_attn_decode/prefill`, ragged builders, inverse RoPE |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | `DeepseekSparseSWABackend`; `DeepseekV4ROCMAiterSparseSWAMetadataBuilder` |
| `vllm/models/deepseek_v4/sparse_mla.py` | `DeepseekV4ROCMAiterMLASparseMetadataBuilder`; C128A ragged index construction |
| `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` | `AiterExperts`; AITER `fused_moe`; gfx950 padding fix |
