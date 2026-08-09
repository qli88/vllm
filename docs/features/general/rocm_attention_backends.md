# vLLM ROCm Attention Backends: Complete Reference

---

## Table of Contents

1. [Backend Name → File → Runtime Name Mapping](#1-backend-name--file--runtime-name-mapping)
2. [Backend Selection on ROCm](#2-backend-selection-on-rocm)
3. [KV Cache Layouts](#3-kv-cache-layouts)
4. [Feature Matrix](#4-feature-matrix)
5. [Batch Dispatch Strategy](#5-batch-dispatch-strategy)
6. [ROCM_ATTN vs FLASH_ATTN: Deep Comparison](#6-rocm_attn-vs-flash_attn-deep-comparison)
7. [ROCM_AITER_FA vs ROCM_AITER_UNIFIED_ATTN vs TRITON_ATTN](#7-rocm_aiter_fa-vs-rocm_aiter_unified_attn-vs-triton_attn)
8. [Why ROCM_AITER_FA Uses an Explicit 3-Way Split](#8-why-rocm_aiter_fa-uses-an-explicit-3-way-split)
9. [KV Connectors](#9-kv-connectors)

---

## 1. Backend Name → File → Runtime Name Mapping

The `AttentionBackendEnum` in
`vllm/v1/attention/backends/registry.py` maps enum members to
implementation classes. The `get_name()` string is what appears at
runtime in logs and metadata — two enum entries can share the same
string.

| Enum (`AttentionBackendEnum`) | File | `get_name()` string | Class |
|---|---|---|---|
| `FLASH_ATTN` | `vllm/v1/attention/backends/flash_attn.py` | `"FLASH_ATTN"` | `FlashAttentionBackend` |
| `ROCM_ATTN` | `vllm/v1/attention/backends/rocm_attn.py` | `"ROCM_ATTN"` | `RocmAttentionBackend` |
| `ROCM_AITER_FA` | `vllm/v1/attention/backends/rocm_aiter_fa.py` | **`"FLASH_ATTN"`** | `AiterFlashAttentionBackend` |
| `ROCM_AITER_UNIFIED_ATTN` | `vllm/v1/attention/backends/rocm_aiter_unified_attn.py` | `"ROCM_AITER_UNIFIED_ATTN"` | `RocmAiterUnifiedAttentionBackend` |
| `TRITON_ATTN` | `vllm/v1/attention/backends/triton_attn.py` | `"TRITON_ATTN"` | `TritonAttentionBackend` |
| `TURBOQUANT` | `vllm/v1/attention/backends/turboquant_attn.py` | `"TURBOQUANT"` | `TurboQuantAttentionBackend` |

> **Important:** `ROCM_AITER_FA` (AMD's ROCm-optimized backend in
> `rocm_aiter_fa.py`) returns `"FLASH_ATTN"` from `get_name()` — the
> same string as NVIDIA's `FLASH_ATTN` enum (`flash_attn.py`). They
> are two distinct enum entries and two distinct implementation files.
> `ROCM_AITER_FA` is AMD's ROCm replacement for NVIDIA's `FLASH_ATTN`
> in the LLM decode/prefill path.

---

## 2. Backend Selection on ROCm

### 2.1 Normal MHA Priority Order (non-MLA, non-sparse)

Selection logic lives in `_get_backend_priorities()` in
`vllm/platforms/rocm.py`. The first valid candidate wins after
per-backend `validate_configuration()` checks (dtype, head size, block
size, etc.).

```
1. ROCM_ATTN              ← default; skipped when use_kv_connector=True
                            or encoder-decoder model
2. ROCM_AITER_FA          ← requires VLLM_ROCM_USE_AITER=1
                            + VLLM_ROCM_USE_AITER_MHA=1 + gfx9 + aiter installed
3. ROCM_AITER_UNIFIED_ATTN← requires gfx9 + aiter installed
                            (no VLLM_ROCM_USE_AITER env var gate)
4. TRITON_ATTN            ← always present, universal fallback
5. TURBOQUANT             ← always last
```

```python
# vllm/platforms/rocm.py  _get_backend_priorities()
backends = []
if not use_kv_connector:
    backends.append(AttentionBackendEnum.ROCM_ATTN)
if rocm_aiter_ops.is_mha_enabled():
    backends.append(AttentionBackendEnum.ROCM_AITER_FA)
if is_aiter_found_and_supported():
    backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
backends.append(AttentionBackendEnum.TRITON_ATTN)
backends.append(AttentionBackendEnum.TURBOQUANT)
```

### 2.2 MLA Priority Order

```
if rocm_aiter_ops.is_mla_enabled():
    [ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]
else:
    [TRITON_MLA]
```

### 2.3 Where `FLASH_ATTN` (the enum, `flash_attn.py`) Appears on ROCm

The `FLASH_ATTN` enum (NVIDIA's upstream backend) **never appears** in
the LLM decode/prefill priority list on ROCm. It is used only for
**ViT (vision encoder) attention** via `get_vit_attn_backend()`:

```
get_vit_attn_backend() on ROCm:
  1. ROCM_AITER_FA   (rocm_aiter_fa.py) ← if AITER enabled + gfx9
  2. FLASH_ATTN      (flash_attn.py)    ← if gfx9 + flash_attn package installed + fp16/bf16
  3. FLASH_ATTN      (flash_attn.py)    ← if RDNA3/4 + flash_attn_triton_amd
                                           + FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
  4. TORCH_SDPA                         ← fallback
```

---

## 3. KV Cache Layouts

This is the most critical structural difference between backends.

| Backend | KV cache shape | Layout name |
|---|---|---|
| `ROCM_ATTN` | `(2, num_blocks, block_size, num_kv_heads, head_size)` | "legacy" — K/V as outermost dim |
| `ROCM_AITER_FA` | `(num_blocks, num_kv_heads, block_size, 2 * head_size)` | "flash" — K/V packed in last dim |
| `ROCM_AITER_UNIFIED_ATTN` | `(num_blocks, num_kv_heads, block_size, 2 * head_size)` | "flash" (inherited) |
| `TRITON_ATTN` | `(num_blocks, num_kv_heads, block_size, 2 * head_size)` | "flash" (HND or NHD sub-layout) |
| `FLASH_ATTN` (NVIDIA) | `(num_blocks, num_kv_heads, block_size, 2 * head_size)` | "flash" |

`ROCM_ATTN` cannot share a KV cache with any other backend. All
flash-layout backends are interoperable.

Within the flash layout, a sub-layout preference applies for KV
connectors and RDMA:

- **HND** = `(blocks, heads, tokens, content)` — preferred by NIXL/RDMA
- **NHD** = `(blocks, tokens, heads, content)` — alternative layout

Controlled by `VLLM_ATTENTION_BACKEND_CACHE_LAYOUT` env var or by the
connector's `get_required_kvcache_layout()`.

---

## 4. Feature Matrix

| Feature | `ROCM_ATTN` | `ROCM_AITER_FA` | `ROCM_AITER_UNIFIED_ATTN` | `TRITON_ATTN` |
|---|---|---|---|---|
| `get_name()` string | `"ROCM_ATTN"` | `"FLASH_ATTN"` | `"ROCM_AITER_UNIFIED_ATTN"` | `"TRITON_ATTN"` |
| Inherits from | — | — | `RocmAttentionBackend` | — |
| Requires AITER env vars | No | **Yes** | No (just package) | No |
| Supported dtypes | fp16, bf16, **fp32** | fp16, bf16 | fp16, bf16 | fp16, bf16, **fp32** |
| FP8 KV cache | Yes | Yes | Yes | Yes |
| int8/int4/fp8 per-token-head KV | No | No | No | **Yes only** |
| Block sizes | `MultipleOf(16)` | **16 or 32 only** | `MultipleOf(16)` | `MultipleOf(16)` |
| Head sizes | 32–256 | **64, 128, 256** | ≥32 | ≥32 |
| CUDA graphs | `ALWAYS` | **`UNIFORM_BATCH`** | `ALWAYS` | `ALWAYS` |
| Sliding window | Yes (Triton decode fallback) | Yes | Yes | Yes |
| ALiBi | Yes | No | No | **Yes (+ alibi_sqrt)** |
| Attention sinks | **No** (Triton fallback) | Yes | Yes | Yes |
| Cascade/shared prefix | No | No | No | No |
| KV connector | **No** (layout incompatible) | Yes | Yes | Yes |
| Encoder-decoder | **No** | **No** | Yes (non-causal fallback) | Yes |
| Non-causal attention | Encoder only | Yes | Yes (cross-attn fallback) | Yes |
| Multimodal PrefixLM | Yes | No | Yes | Yes |
| Rolling SWA (R-SWA) | No | No | No | **Yes only** |
| Batch invariance | No | No | No | **Yes only** |
| `supports_kv_connector()` | **False** | True | True | True |

---

## 5. Batch Dispatch Strategy

Consider a mixed batch with three requests:

- **A** — Prefill: 512 new tokens, no KV cache context
- **B** — Extend (chunked prefill): 128 new tokens, 256 cached tokens
- **C** — Decode: 1 new token, 1024 cached tokens

### ROCM_ATTN — Two separate kernel calls

```
forward()
  ├── Phase 1 (if any query_len > 1):
  │     context_attention_fwd  [Triton, vllm/v1/attention/ops/prefix_prefill.py]
  │     Processes A (512q) + B (128q + reads 256 cached from block_table)
  │     Skips C via SKIP_DECODE=True
  │
  └── Phase 2 (always):
        if use_rocm_custom_paged_attention():   ← block_size∈{16,32}, head∈{64,128}
          paged_attention_rocm  [HIP C++, csrc/rocm/attention.cu]
        else:
          kernel_paged_attention_2d  [Triton fallback]
        Both process all seqs; output is the new token for C only
```

The fallback triggers (and logs a warning) for non-standard block
sizes (e.g., Qwen3-Next with block_size=544), sliding window, or
attention sinks.

### ROCM_AITER_FA — Explicit three-way split

```
forward()
  ├── Decode (C: query_len == 1):
  │     if head_size >= 64 + compatible layout:
  │       torch.ops.aiter.paged_attention_v1  [ASM ll4mi kernel]
  │     else:
  │       aiter.ops.triton.unified_attention  [aiter Triton]
  │
  ├── Prefill (A: query_len == seq_len, no prior cache):
  │     rocm_aiter_ops.flash_attn_varlen_func  [aiter flash attention]
  │
  └── Extend (B: query_len > 1 + has prior cache):
        Chunked loop over cached context in 32K-token chunks:
          cp_mha_gather_cache [Triton]
            → gather cached KV into dense workspace
          rocm_aiter_ops.flash_attn_varlen_func [causal=False]
            → query vs context chunk
          merge_attn_states
            → online softmax merge across chunks
```

### ROCM_AITER_UNIFIED_ATTN — Single kernel call (no splitting)

```
forward()
  └── aiter.ops.triton.unified_attention  [external aiter Triton kernel]
        Full token batch [A_512, B_128, C_1] in one pass (causal=True)
        Falls back to vllm's own triton_unified_attention for non-causal
        cross-attention (encoder-decoder) only
```

### TRITON_ATTN — Single kernel call with 2D/3D selector

```
forward()
  └── vllm.v1.attention.ops.triton_unified_attention  [vllm-native Triton]
        if max_seqlen_q > 1 (any prefill) or large decode batch:
          2D kernel: grid = (q_blocks, kv_heads)
        else (small pure-decode batch):
          3D kernel: grid = (q_blocks, kv_heads, 16 softmax segments)
          └── reduce_segments merge pass
        Single kernel code handles A, B, C in one pass
```

### Kernel source summary

| Phase | `ROCM_ATTN` | `ROCM_AITER_FA` | `ROCM_AITER_UNIFIED_ATTN` | `TRITON_ATTN` |
|---|---|---|---|---|
| Prefill / Extend | Triton `context_attention_fwd` (vllm, LightLLM-adapted) | `rocm_aiter_ops.flash_attn_varlen_func` (aiter) + chunked extend | aiter Triton `unified_attention` | vllm Triton `kernel_unified_attention` (2D mode) |
| Decode | HIP C++ `paged_attention_rocm` or Triton fallback | ASM `paged_attention_v1` (ll4mi) or aiter Triton | aiter Triton `unified_attention` | vllm Triton `kernel_unified_attention` (3D mode for small batches) |
| KV cache write | `PagedAttention.write_to_paged_cache` (C++) or `triton_reshape_and_cache_flash` | `reshape_and_cache_flash` (C++) | `reshape_and_cache_flash` (C++) | `triton_reshape_and_cache_flash` |

---

## 6. ROCM_ATTN vs FLASH_ATTN: Deep Comparison

This section compares `ROCM_ATTN` (AMD's default ROCm backend) with
`FLASH_ATTN` (NVIDIA's CUDA backend, `flash_attn.py`). These are the
two primary defaults on their respective platforms.

### Metadata

`FlashAttentionMetadata` carries significantly more fields than
`RocmAttentionMetadata`, reflecting richer feature support:

| Field | `RocmAttentionMetadata` | `FlashAttentionMetadata` |
|---|---|---|
| `num_actual_tokens` | Yes | Yes |
| `query_start_loc` | Yes | Yes |
| `seq_lens`, `block_table`, `slot_mapping` | Yes | Yes |
| `use_cascade`, `common_prefix_len` | Yes | Yes |
| `num_decode_reqs`, `num_prefill_tokens` | No | **Yes** |
| `scheduler_metadata` (FA3 AOT tile schedule) | Stub (None) | **Yes** (SM90+) |
| `max_dcp_context_kv_len` | No | **Yes** |
| `sliding_window` | No | **Yes** |
| `mm_prefix_range_tensor` | No | **Yes** (FA4) |
| `rswa_*` (Rolling SWA) | No | **Yes** (FA4) |
| `causal` as tensor (per-seq) | No | **Yes** (FA4) |

### Dispatch with a mixed batch

Given requests A (prefill, 512q), B (extend, 128q + 256 cached), C (decode, 1q + 1024 cached):

**ROCM_ATTN — Two separate kernel calls:**

```
Phase 1 (max_query_len > 1):
  context_attention_fwd [Triton]
  → A (512q) + B (128q, reads 256 cached), skips C

Phase 2 (always):
  if block_size∈{16,32} + head∈{64,128}:
    paged_attention_rocm [HIP C++]
  else:
    kernel_paged_attention_2d [Triton]
  → processes all seqs, outputs C's token
```

**FLASH_ATTN — Single varlen call:**

```
flash_attn_varlen_func(
  q=[A_512, B_128, C_1],          # all tokens concatenated
  cu_seqlens_q=[0, 512, 640, 641],
  seqused_k=[512, 384, 1025],
  block_table=..., causal=True
)
```

The library handles prefill/extend/decode natively in one kernel
invocation, using FlashDecoding internally for long decode sequences.

### CUDA Graph support

| Backend | CUDAGraph support | Special handling |
|---|---|---|
| `ROCM_ATTN` | `ALWAYS` | Sets `seq_lens.fill_(1)` during capture (avoids slowness); zeros `query_start_loc` to prevent invalid memory access in the prefill kernel |
| `FLASH_ATTN` FA3 | `ALWAYS` | Pre-built AOT tile `scheduler_metadata` |
| `FLASH_ATTN` FA2 | `UNIFORM_BATCH` | No special treatment |

### Flash version selection (CUDA only)

| GPU | FA version | Notable features |
|---|---|---|
| SM100+ (Blackwell) | FA4 (CuTE-DSL) | `mask_mod`, per-seq causal tensor, R-SWA |
| SM90 (Hopper) | FA3 | AOT tile scheduler, FP8, per-head scales |
| SM80 and below | FA2 | — |

Overrideable via `vllm_config.attention_config.flash_attn_version`.

### Cascade attention (shared prefix optimization)

`FLASH_ATTN` supports cascade attention when many requests share a long
common prefix (e.g., a system prompt). Threshold: `common_prefix_len >= 256`
AND `num_requests >= 8`.

```
cascade_attention():
  1. flash_attn_varlen_func over prefix KV cache  →  (lse_prefix, out_prefix)
  2. flash_attn_varlen_func over suffix KV cache  →  (lse_suffix, out_suffix)
  3. merge_attn_states(out_prefix, lse_prefix, out_suffix, lse_suffix)
```

`ROCM_ATTN` always returns `False` from `use_cascade_attention()`.
None of the ROCm backends currently support cascade attention.

---

## 7. ROCM_AITER_FA vs ROCM_AITER_UNIFIED_ATTN vs TRITON_ATTN

### What "unified" means

Both `ROCM_AITER_UNIFIED_ATTN` and `TRITON_ATTN` dispatch all token
types (prefill, extend, decode) in a single kernel call — no explicit
batch splitting. `ROCM_AITER_FA` by contrast sorts the batch into
three categories and runs a different kernel for each.

The difference between the two "unified" backends is the kernel source:

| | `ROCM_AITER_UNIFIED_ATTN` | `TRITON_ATTN` |
|---|---|---|
| Kernel source | `aiter.ops.triton.unified_attention` (external aiter package) | `vllm/v1/attention/ops/triton_unified_attention.py` (vllm-native, in-tree) |
| 2D/3D selector | No | **Yes** (3D for small decode batches — 16 parallel softmax segments + merge) |
| Non-causal | Falls back to vllm's kernel | Native |
| int4/int8/fp8 per-token-head KV | No | **Yes** |
| Sinks | Yes | Yes |
| R-SWA | No | **Yes** |
| ALiBi | No | **Yes** |
| fp32 | No | **Yes** |
| Platform | ROCm only | All (CUDA, ROCm, XPU) |

`ROCM_AITER_UNIFIED_ATTN` inherits from `RocmAttentionBackend`
(`rocm_attn.py`) and reuses `RocmAttentionMetadata` and
`RocmAttentionMetadataBuilder` from the parent. Only the attention
kernel call is overridden.

### ROCM_AITER_FA: explicit three-way split

`ROCM_AITER_FA` sorts every batch into decode / prefill / extend
buckets and dispatches a specialized kernel per bucket:

- **Decode** → AMD's ASM `ll4mi` kernel (`paged_attention_v1`) for
  maximum throughput on gfx9; falls back to aiter Triton for small
  heads or unsupported features
- **Prefill** → `rocm_aiter_ops.flash_attn_varlen_func`
- **Extend** → a Python chunked loop (32K tokens/chunk):
  gather cached KV → flash attention over chunk → merge

This is the only ROCm backend with an ASM-level decode kernel and
therefore the highest decode throughput on MI300X, at the cost of the
tightest constraints (block sizes 16/32 only, heads 64/128/256,
`UNIFORM_BATCH` CUDA graphs, no encoder-decoder, requires env vars).

### Triton kernel comparison

| Aspect | `TRITON_ATTN` (vllm-native) | `ROCM_AITER_UNIFIED_ATTN` (aiter) | `ROCM_ATTN` (Triton parts) |
|---|---|---|---|
| File | `vllm/v1/attention/ops/triton_unified_attention.py` | `aiter.ops.triton.unified_attention` | `vllm/v1/attention/ops/prefix_prefill.py` + `chunked_prefill_paged_decode.py` |
| Handles all phases | Yes — single dispatch | Yes — single dispatch | No — separate prefill and decode kernels |
| KV cache layout | Flash | Flash | Legacy `(2, blocks, ...)` |
| 2D/3D dispatch | Yes | No | No |
| int4/int8 per-token KV | Yes | No | No |
| Origin | vllm-native | External AMD aiter package | LightLLM-adapted |

### When to use which backend

| Scenario | Recommended |
|---|---|
| Default ROCm, no special requirements | **ROCM_ATTN** |
| Best decode throughput on gfx9 (MI300X), AITER enabled | **ROCM_AITER_FA** |
| KV connectors or encoder-decoder on ROCm | **ROCM_AITER_UNIFIED_ATTN** or **TRITON_ATTN** |
| Per-token-head quantization (int4/int8/fp8) | **TRITON_ATTN** only |
| Rolling sliding window (R-SWA) | **TRITON_ATTN** only |
| Maximum compatibility / universal fallback | **TRITON_ATTN** |
| ViT encoder on gfx9 without AITER | **FLASH_ATTN** enum (`flash_attn.py`) |

---

## 8. Why ROCM_AITER_FA Uses an Explicit 3-Way Split

### The Root Cause: `rocm_aiter_ops.flash_attn_varlen_func` Has No `block_table`

The short answer: **the aiter `flash_attn_varlen_func` does not accept a
`block_table` parameter**. It only operates on dense, pre-materialized
`k` and `v` tensors. It has no concept of a paged KV cache. NVIDIA's
`flash_attn_varlen_func` has `block_table` built in, which is what
allows it to handle prefill, extend, and decode in one call.

### API Signatures Side by Side

**NVIDIA `vllm_flash_attn.flash_attn_varlen_func`**
(`vllm/vllm_flash_attn/flash_attn_interface.py`):

```python
def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_k=None,   # for non-paged prefill
    seqused_k=None,      # for paged decode: total KV length per seq
    block_table=None,    # ← PAGED KV: maps seq → physical blocks
    ...
)
# When block_table is set, k/v are ignored — the kernel reads
# KV tokens directly from the paged cache via block_table[seq][block]
```

**`rocm_aiter_ops.flash_attn_varlen_func`**
(`vllm/_aiter_ops.py`):

```python
def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,        # required — no seqused_k alternative
    max_seqlen_q,
    max_seqlen_k,
    ...
    # NO block_table parameter
)
# k and v must be dense tensors containing all KV tokens
```

### What This Means for Each Request Type

Given requests A (prefill), B (extend), C (decode):

| Request | New tokens | Cached tokens | Where is cached KV? |
|---|---|---|---|
| A (prefill) | 512 | 0 | nowhere — first pass |
| B (extend) | 128 | 256 | paged KV cache (block_table) |
| C (decode) | 1 | 1024 | paged KV cache (block_table) |

- **Prefill (A):** no cached context. The new `k`/`v` tensors from the
  current forward pass _are_ the full key/value. `cu_seqlens_k ==
  cu_seqlens_q`. `rocm_aiter_ops.flash_attn_varlen_func` works directly
  because `k`/`v` are already dense.

- **Extend (B):** the 256 cached tokens live in scattered physical
  blocks in the paged KV cache. To call `flash_attn_varlen_func`, those
  blocks must first be **gathered into a dense buffer**. That is
  exactly what `cp_mha_gather_cache` does — it walks the block table
  and materializes the cached tokens into a contiguous workspace. Then
  `flash_attn_varlen_func` is called on the dense result (`causal=False`,
  query-vs-context), merged with the self-attention output via
  `merge_attn_states`.

- **Decode (C):** almost the entire KV is in the paged cache (1024
  tokens). Gathering it all for one token of output would be expensive.
  Instead, `paged_attention_v1` (the ASM ll4mi kernel) is used — it
  natively understands the block table and reads KV directly from
  paged memory, no gather needed.

### Why NVIDIA's Single Call Works

NVIDIA's `flash_attn_varlen_func` has `block_table` as a first-class
parameter handled inside the CUDA kernel. When provided, the inner
loop reads KV tokens by walking the block table directly:

```
NVIDIA kernel (conceptual):
  for each query token q_i in seq s:
    for each kv position j in [0, seqused_k[s]):
      block_idx = block_table[s][j // block_size]
      offset    = j % block_size
      k_j = kv_cache[block_idx][offset]   # direct paged read
      score += dot(q_i, k_j)
```

FA2, FA3, and FA4 all support this. A single varlen call handles
prefill (no block_table needed — `cu_seqlens_k` used instead),
extend (paged context, `seqused_k` + `block_table`), and decode
(paged context, `seqused_k` + `block_table`) uniformly.

The aiter `flash_attn_varlen_func` has no equivalent path — `k` and
`v` must be dense on input.

### The 3-Way Split Mapped to This Constraint

```
Prefill (A): no cached KV — k/v are the new tokens, already dense
  → rocm_aiter_ops.flash_attn_varlen_func(k=new_k, v=new_v,
      cu_seqlens_k=cu_seqlens_q)
  ✓ Works directly

Extend (B): new tokens + context scattered in paged KV cache
  → Cannot pass paged k/v to flash_attn_varlen_func
  → cp_mha_gather_cache materializes paged KV into dense workspace
  → flash_attn_varlen_func on dense context (causal=False)
  → merge_attn_states to combine self-attn + context outputs
  ✓ Works via gather + flash + merge

Decode (C): 1 new token + all context in paged KV cache
  → Cannot pass paged k/v to flash_attn_varlen_func (and gather
    would be wasteful for a single output token)
  → paged_attention_v1 (ASM ll4mi) natively reads block_table
  ✓ Works via block_table-aware ASM kernel, no gather
```

Each category has a fundamentally different relationship between the
current-step `k`/`v` tensors and the paged KV cache:

- **Prefill**: KV is 100% in current-step dense tensors
- **Extend**: KV is split — new tokens in dense tensors, context in paged cache
- **Decode**: KV is almost entirely in paged cache; only 1 new token in current step

The 3-way split exists because each case maps to a different kernel
that can actually read the data from wherever it lives. If the aiter
library added a `block_table` parameter to `flash_attn_varlen_func`,
the split could be collapsed — but that is a library-level gap, not a
vLLM architectural choice.

---

## 9. KV Connectors

### 8.1 What KV Connectors Are

A KV connector is a pluggable transfer subsystem that lets KV caches be
moved between separate vLLM instances or between GPU and other storage,
without going through the normal inference path.

**Primary use case — Disaggregated prefill:**

```
                 User Request
                      │
           ┌──────────▼──────────┐
           │   Prefill Instance   │
           │  kv_role="kv_producer"
           │  Runs forward pass,  │
           │  computes KV cache   │
           └──────────┬──────────┘
                      │  RDMA WRITE (NIXL/UCX)
                      │  KV blocks pushed into
                      │  Decode Instance's GPU memory
                      ▼
           ┌──────────────────────┐
           │   Decode Instance    │
           │  kv_role="kv_consumer"
           │  Skips prefill,      │
           │  uses received KV,   │
           │  generates tokens    │
           └──────────────────────┘
```

**Other use cases:**

- **KV offloading** — spill completed GPU blocks to CPU/disk/S3,
  promote on cache hit (`OffloadingConnector`)
- **Cross-node prefix caching** — share a common KV prefix cache across
  multiple instances (`FlexKVConnectorV1`, `LMCacheConnectorV1`)
- **RDMA-accelerated P/D** — zero-copy GPU-to-GPU via NIXL/UCX/GDS
  (`NixlPushConnector`)

### 8.2 Three-Layer Architecture

```
KVConnector            ← top-level integration (scheduler + worker hooks)
    └── KVLookupBuffer ← content-addressed cache (insert / drop_select by token hash)
            └── KVPipe ← FIFO tensor pipe (send_tensor / recv_tensor)
```

### 8.3 Scheduler-Side Interface (`KVConnectorBase_V1`)

| Method | Purpose |
|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | Returns how many tokens can be loaded from external KV cache |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | Decides what to load after block allocation |
| `build_connector_meta(scheduler_output)` | Packages instructions for the worker; resets internal state |
| `request_finished(request, block_ids)` | Called on completion; optionally takes async responsibility for freeing blocks |
| `on_new_request(request)` | Bookkeeping hook for new requests |

### 8.4 Worker-Side Interface

| Method | Purpose |
|---|---|
| `start_load_kv(forward_context)` | Begin async loading before forward pass |
| `wait_for_layer_load(layer_name)` | Block until one layer's KV is ready (enables layer-by-layer pipeline overlap with compute) |
| `save_kv_layer(layer_name, kv_layer, attn_metadata)` | Begin async saving of one layer's KV after forward |
| `wait_for_save()` | Block until all saves complete |
| `register_kv_caches(kv_caches)` | Pre-register GPU memory with RDMA NIC (NIXL) |

### 8.5 Concrete Example: NIXL Disaggregated Prefill

**Step-by-step flow:**

1. Prefill instance runs forward. After each attention layer,
   `save_kv_layer()` issues an RDMA WRITE directly into the decode
   instance's pre-allocated paged KV buffer.
2. Decode instance scheduler calls `get_num_new_matched_tokens()`,
   which waits for the transfer. The request's computed token count is
   updated to include the received prefix.
3. Decode instance runs forward from where prefill left off — KV
   blocks are already in GPU paged memory, no recompute needed.

**Config:**

```bash
--kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_connector_extra_config": {"backends": ["UCX", "GDS"]}
}'
```

### 8.6 Concrete Example: KV Offloading

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {"cpu_bytes_to_use": 10000000000}
}'
```

When a request finishes, `request_finished()` returns `True` and
asynchronously DMA's KV blocks to pinned CPU memory. On a later
request hitting the same prefix, `start_load_kv()` DMA's them back
to GPU — no recompute needed.

### 8.7 How KV Connectors Affect Backend Selection

#### The gate: `use_kv_connector`

In `vllm/v1/attention/selector.py`:

```python
kv_transfer_config = vllm_config.kv_transfer_config
use_kv_connector = (
    kv_transfer_config is not None
    and kv_transfer_config.is_kv_transfer_instance  # True when kv_connector + kv_role set
)
```

This flag is passed to the platform's `get_valid_backends()` and to
each backend's `validate_configuration()`.

#### Backend response

```
Backend                    supports_kv_connector()   KV cache leading dim
──────────────────────────────────────────────────────────────────────────
ROCM_ATTN                  False (explicit)          2 (K/V)    ← EXCLUDED
ROCM_AITER_FA              True  (default)           num_blocks ✓
ROCM_AITER_UNIFIED_ATTN    True  (default)           num_blocks ✓
TRITON_ATTN                True  (default)           num_blocks ✓
FLASH_ATTN (NVIDIA)        True  (default)           num_blocks ✓
```

On ROCm with `use_kv_connector=True`, the effective backend chain is:

```
ROCM_AITER_FA  →  ROCM_AITER_UNIFIED_ATTN  →  TRITON_ATTN
```

### 8.8 Why ROCM_ATTN Is Incompatible: The Layout Problem

KV connectors must **address, register, and transfer individual KV
blocks by block index**. The transfer code in
`vllm/distributed/kv_transfer/kv_connector/utils.py` enforces this
with hard assertions:

```python
kv_cache_shape = get_kv_cache_shape(num_blocks=1, ...)
assert kv_cache_shape[0] == 1   # num_blocks must be leading dim
assert len(kv_cache_shape) == 4  # [blocks, H, sz, content]
```

For `ROCM_ATTN`, `get_kv_cache_shape(num_blocks=1)` returns
`(2, 1, block_size, num_kv_heads, head_size)`:

- `result[0]` is `2` (K/V), not `1` → first assertion fails
- Result is 5-dimensional, not 4 → second assertion also fails

For RDMA backends (NIXL, MoRIIO), GPU memory is pre-registered as
contiguous regions with the NIC. With blocks-first layout, block `i`
is a contiguous slice `kv_cache[i]`. With K/V-first layout, block `i`'s
key is at `kv_cache[0][i]` and its value at `kv_cache[1][i]` —
two non-contiguous regions, making registration and transfer
significantly more complex.

**Three independent enforcement layers:**

| Layer | Mechanism |
|---|---|
| Platform pre-filter | `_get_backend_priorities()` omits `ROCM_ATTN` when `use_kv_connector=True` |
| Backend opt-out | `RocmAttentionBackend.supports_kv_connector()` returns `False` |
| Connector assertion | `TransferTopology.__post_init__` asserts blocks-first shape |

### 8.9 KV Cache Sub-Layout: HND vs NHD

Within the blocks-first flash layout, connectors can request a further
sub-layout preference:

| Layout | Shape | Preferred by |
|---|---|---|
| **HND** | `(blocks, heads, tokens, content)` | `NixlConnector`, `OffloadingConnector` — heads outer for efficient RDMA per (block, head) range |
| **NHD** | `(blocks, tokens, heads, content)` | Default for most CUDA backends |

Set via `VLLM_ATTENTION_BACKEND_CACHE_LAYOUT` env var or by the
connector's `get_required_kvcache_layout()` method (which calls
`set_kv_cache_layout()` at backend selection time).

### 8.10 Available Connectors

| Connector | Transport | Use Case |
|---|---|---|
| `NixlConnector` / `NixlPushConnector` | RDMA (UCX/GDS) | High-perf disaggregated P/D on NVIDIA |
| `MoRIIOConnector` | AMD MoRI-IO | High-perf disaggregated P/D on ROCm |
| `OffloadingConnector` | CPU pinned / S3 / P2P (NIXL) | GPU-to-CPU KV offload |
| `LMCacheConnectorV1` | LMCache (NIXL-backed) | Cross-instance prefix cache |
| `FlexKVConnectorV1` | FlexKV distributed store | Distributed LRU prefix caching |
| `MooncakeConnector` | Mooncake RDMA | P/D with prefix caching |
| `ExampleConnector` | Local disk (safetensors) | Reference / debugging |
| `SimpleCPUOffloadConnector` | CPU | Simple CPU offload |
| `HF3FSKVConnector` | HF3FS filesystem | High-throughput filesystem-backed KV |
| `MultiConnector` | Wrapper | Combine multiple connectors |

### 8.11 KVTransferConfig Fields

Configured via `--kv-transfer-config` (JSON string) or Python API:

```python
KVTransferConfig(
    kv_connector="NixlConnector",   # connector name
    kv_role="kv_both",              # "kv_producer", "kv_consumer", "kv_both"
    kv_buffer_device="cuda",        # "cuda", "cpu", "xpu"
    kv_buffer_size=1e9,             # bytes (for TorchDistributed connector)
    kv_rank=None,                   # 0=prefill, 1=decode in 1P1D setups
    kv_parallel_size=1,
    kv_ip="127.0.0.1",
    kv_port=14579,
    kv_connector_extra_config={},   # connector-specific opaque dict
)
```

`is_kv_transfer_instance` is `True` when both `kv_connector` and a
valid `kv_role` are set — this is the gate that sets
`use_kv_connector=True` in the attention backend selector.
