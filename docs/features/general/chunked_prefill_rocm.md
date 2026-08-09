# Chunked Prefill in vLLM ROCm Attention Backends

> Covers `ROCM_ATTN` and `TRITON_ATTN` backends in `vllm/v1/`.
> All file references are relative to the vLLM repo root.

---

## Table of Contents

1. [Terminology: seq\_len vs query\_len](#1-terminology-seq_len-vs-query_len)
2. [Request Ordering in a Batch](#2-request-ordering-in-a-batch)
3. [How Chunked Prefill Works in ROCM\_ATTN](#3-how-chunked-prefill-works-in-rocm_attn)
4. [Flash Attention Version in ROCM\_AITER\_FA](#4-flash-attention-version-in-rocm_aiter_fa)
5. [ROCM\_ATTN vs TRITON\_ATTN: Full Comparison](#5-rocm_attn-vs-triton_attn-full-comparison)

---

## 1. Terminology: seq\_len vs query\_len

The comment in
`vllm/v1/attention/backends/rocm_attn.py:45`
defines these precisely:

```
# |---------- N-1 iteration --------|
# |---------------- N iteration ---------------------|
# |- tokenA -|......................|-- newTokens ---|
# |---------- context_len ----------|
# |-------------------- seq_len ---------------------|
#                                   |-- query_len ---|
```

| Term | What it counts | Source |
|---|---|---|
| `seq_len` | Total tokens in KV cache **after** this forward pass = cached tokens + new tokens being processed now | `CommonAttentionMetadata.seq_lens`, shape `(batch_size,)` |
| `query_len` | New tokens being processed **in this forward pass** — what Q is built from | Derived: `query_start_loc[i+1] − query_start_loc[i]` |
| `context_len` | Already-cached tokens = `seq_len − query_len` | Computed inside kernels as `cur_batch_ctx_len` |

**In practice:**

| Request state | `query_len` | `context_len` | `seq_len` |
|---|---|---|---|
| Decode (generating token N) | 1 | N−1 | N |
| First prefill chunk (no cache yet) | chunk\_size | 0 | chunk\_size |
| Later prefill chunk (K chunks done) | chunk\_size | K × chunk\_size | (K+1) × chunk\_size |

### How `query_start_loc` encodes the batch

`query_start_loc` is a CSR (Compressed Sparse Row) offset array of length `batch_size + 1`.
It indexes into the flat `query` / `key` / `value` / `output` tensors.

```
query_start_loc[0]   = 0
query_start_loc[i+1] = query_start_loc[i] + query_len[i]
query_start_loc[N]   = total tokens in batch
```

Request `i`'s query tokens live at `query[query_start_loc[i] : query_start_loc[i+1]]`.

---

## 2. Request Ordering in a Batch

### Default ordering

By default `ROCM_ATTN` does **not** reorder. Requests stay in whatever order the scheduler placed them.

### Optional reordering

`gpu_model_runner.py:1119` calls `reorder_batch_to_split_decodes_and_prefills`
(`vllm/v1/attention/backends/utils.py:663`)
when `reorder_batch_threshold` is configured. It sorts requests into **4 contiguous regions**:

```
Region 0 │ decode        │ query_len ≤ threshold  AND  done prefilling
Region 1 │ short_extend  │ query_len ≤ threshold  AND  still prefilling (small chunk)
Region 2 │ long_extend   │ query_len > threshold  AND  still prefilling (large chunk)
Region 3 │ pure_prefill  │ context_len == 0            (first chunk of a new prompt)
```

Classification logic:

```python
has_context        = num_computed_tokens > 0
is_below_threshold = num_scheduled_tokens <= decode_threshold
done_prefilling    = num_computed_tokens >= num_prompt_tokens

is_pure_prefill = ~has_context                                          # region 3
is_long_extend  = has_context & ~is_below_threshold                     # region 2
is_short_extend = has_context & is_below_threshold & ~done_prefilling   # region 1
is_decode       = has_context & is_below_threshold & done_prefilling    # region 0
```

### Concrete example: 5 requests

| Req | Type | `query_len` | `context_len` | `seq_len` |
|---|---|---|---|---|
| A | Decode | 1 | 199 | 200 |
| B | Decode | 1 | 511 | 512 |
| C | Short extend | 4 | 128 | 132 |
| D | Long extend | 256 | 512 | 768 |
| E | Pure prefill | 256 | 0 | 256 |

After reordering: **A, B, C, D, E**

Resulting tensors:

```
query_start_loc = [0,  1,  2,  6,  262, 518]
seq_lens        = [200, 512, 132, 768, 256]

query[0:1]    = Req A  (1 decode token)
query[1:2]    = Req B  (1 decode token)
query[2:6]    = Req C  (4 prefill tokens, chunk 2)
query[6:262]  = Req D  (256 prefill tokens, chunk 3)
query[262:518]= Req E  (256 prefill tokens, first chunk)
```

Inside `_fwd_kernel`, for Req D:

```
cur_batch_seq_len   = 768
cur_batch_query_len = 256   (query[6:262])
cur_batch_ctx_len   = 512   ← Phase 1: attend these from KV cache
                             ← Phase 2: attend 256 new tokens causally
```

---

## 3. How Chunked Prefill Works in ROCM\_ATTN

### Why it exists

vLLM's scheduler batches **partial-prefill** requests (large prompts split into chunks
across forward passes) together with **decode** requests (single-token generation) in
the same forward pass. The `ROCM_ATTN` backend handles this mixture with a single call
to `chunked_prefill_paged_decode`.

### Entry point: `RocmAttentionImpl.forward`

`vllm/v1/attention/backends/rocm_attn.py:446`

```python
chunked_prefill_paged_decode(
    query=query[:num_actual_tokens],
    key=key[:num_actual_tokens],
    value=value[:num_actual_tokens],
    output=output[:num_actual_tokens],
    ...
    max_query_len=max_seqlen_q,   # > 1 means at least one prefill seq in the batch
    skip_decode=True,
)
```

### The two dispatch paths

`vllm/v1/attention/ops/chunked_prefill_paged_decode.py:271`

```
if max_query_len > 1:
    ┌─────────────────────────────────────────┐
    │  PREFILL PATH: context_attention_fwd()  │  ← Triton kernel, SKIP_DECODE=True
    │  Handles sequences where query_len > 1  │
    └─────────────────────────────────────────┘

DECODE PATH (always runs):
    if use_custom:
        ops.paged_attention_rocm()            ← HIP C++ kernel (block_size 16/32)
    else:
        kernel_paged_attention_2d()           ← Triton fallback
    Both skip sequences where query_len > 1
```

Both paths write into different slots of the **shared `output` tensor** in one forward pass —
no separate passes for prefill vs decode.

### Prefill kernel: `_fwd_kernel` — two-phase algorithm

`vllm/v1/attention/ops/prefix_prefill.py:38`

For each prefill sequence the kernel runs two loops sharing one online-softmax accumulator
`(m_i, l_i, acc)`:

#### Phase 1 — attend over the cached context (no causal mask)

```python
# context_len = seq_len - query_len  (tokens already in KV cache)
for start_n in range(0, context_len, BLOCK_SIZE):
    bn = B_Loc[seq, token_indices // PHYSICAL_BLOCK_SIZE]  # paged block ID
    K = load(K_cache[bn, kv_head, ...])   # read from KV cache
    V = load(V_cache[bn, kv_head, ...])
    qk = dot(Q, K) * scale
    # NO causal mask — all cached tokens are strictly in the past
    acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, V)
```

#### Phase 2 — attend over new query tokens (causal)

```python
for start_n in range(0, query_len, BLOCK_N):
    K = load(K_raw[batch_start + start_n])   # from RAW input tensor, not KV cache
    V = load(V_raw[batch_start + start_n])
    qk = dot(Q, K) * scale
    # Causal mask: Q[i] attends K[j] only if j <= i (within the new tokens)
    qk = where(offs_m >= (start_n + offs_n), qk, -inf)
    acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, V)

output = acc / l_i
```

Phase 2 reads from the **raw `K`/`V` input tensors** because the new tokens have not yet
been written to the paged KV cache when `forward()` runs. The KV cache update is a
separate step that happens after `forward()` returns.

#### Skip decode sequences

`prefix_prefill.py:110`

```python
if SKIP_DECODE and cur_batch_query_len == 1:
    return   # early exit — decode seqs are handled by the paged_attention kernel
```

### Decode kernel: selector logic

`chunked_prefill_paged_decode.py:357`

```python
use_custom     = use_rocm_custom_paged_attention(...)
is_pow2        = block_size & (block_size - 1) == 0
has_native_layout = key_cache.stride(0) == key_cache.shape[1:].numel()

if not is_pow2 or not has_native_layout:
    use_custom = False   # non-standard blocks (e.g. Qwen3's 544) → Triton fallback
```

| Condition | Decode kernel used |
|---|---|
| `block_size` ∈ {16, 32}, contiguous layout, supported dtype | `ops.paged_attention_rocm` (HIP C++) |
| Non-power-of-2 block or non-contiguous (e.g. hybrid Mamba cache) | `kernel_paged_attention_2d` (Triton) |

Both decode kernels skip prefill sequences:

```python
# kernel_paged_attention_2d, filter_by_query_len=True:
cur_batch_query_len = query_start_loc[seq+1] - query_start_loc[seq]
if cur_batch_query_len > 1:
    return   # skip prefill seq
```

### KV cache layout

`ROCM_ATTN` uses separate K/V arrays:

```python
# RocmAttentionBackend.get_kv_cache_shape:
return (2, num_blocks, block_size, num_kv_heads, head_size)

key_cache   = kv_cache[0]   # (num_blocks, block_size, num_kv_heads, head_size)
value_cache = kv_cache[1]
```

### Worked example: 2 sequences

**Scenario:**

- Seq A: 512-token prompt, chunked at 256. Chunk 2: 256 tokens cached, 256 new tokens now.
- Seq B: Decode, generating token 50. `query_len=1`, `seq_len=512`.

**Metadata:**

```
query_start_loc = [0, 256, 257]
seq_lens        = [512, 512]
max_query_len   = 256
max_seq_len     = 512

query[0:256]   = 256 new tokens for Seq A
query[256:257] = 1 decode token for Seq B
```

**Execution:**

```
chunked_prefill_paged_decode(max_query_len=256, ...)

  max_query_len > 1 → run context_attention_fwd (SKIP_DECODE=True):
    Seq A (query_len=256, ctx_len=256):
      Phase 1: Q[0:256] × K_cache[blocks 0..7]  → partial acc (no causal mask)
      Phase 2: Q[0:256] × K_raw[0:256] (causal) → merge into acc
      → write output[0:256]
    Seq B (query_len=1):
      SKIP_DECODE fires → return immediately

  always → run paged_attention_rocm / kernel_paged_attention_2d:
    Seq A (query_len=256):
      filter_by_query_len → SKIP
    Seq B (query_len=1, seq_len=512):
      iterate all 512/block_size KV cache blocks
      → write output[256:257]
```

`output[0:257]` is fully populated in one forward pass.

---

## 4. Flash Attention Version in ROCM\_AITER\_FA

`ROCM_AITER_FA` is **not** based on Dao-AI Lab's FA2/FA3/FA4. It uses AMD's `aiter`
library which dispatches to AMD-specific kernels. The dispatcher in
`aiter/ops/mha.py:flash_attn_varlen_func` selects a backend based on hardware:

```
flash_attn_varlen_func()
  │
  ├── gfx1250 (RDNA4) + bf16 + causal, no SWA
  │     → fmha_fwd_with_sink_varlen_asm()   ← hand-written ASM for gfx1250
  │
  ├── gfx942 / gfx950 (MI300X / MI350) + bf16 or fp8 + hdim 128/192, no SWA
  │     → fmha_v3_varlen_fwd()              ← AMD's hand-written gfx9 ASM ("fmha v3")
  │
  └── everything else (other GPUs, SWA, alibi, fp16, etc.)
        → mha_varlen_fwd()                  ← AMD Composable Kernel (CK-tile)
```

| Path | What it is | Underlying tech |
|---|---|---|
| `fmha_v3_varlen_fwd` | AMD's internal "FMHA v3" — hand-written gfx9 ASM | Custom AMD assembly; bf16/fp8; gfx942/gfx950 only; hdim 128 or 192 |
| `mha_varlen_fwd` | CK-tile based FMHA | AMD Composable Kernel; conceptually equivalent to FA2 algorithm |
| `fmha_fwd_with_sink_varlen_asm` | gfx1250 ASM prefill | Custom ASM for RDNA4 (gfx1250) |

**The `fmha_v3` naming is AMD-internal** — it is not "Flash Attention 3" from Tri Dao.
CK-tile implements the same online-softmax tiled algorithm as FA2, but entirely in AMD's
own framework.

Eligibility check for the `fmha_v3` ASM path on MI300X (`gfx942`):

```python
def can_impl_fmha_v3_fwd():
    ret = get_gfx() in ("gfx942", "gfx950")
    ret = ret and alibi_slopes is None
    ret = ret and bias is None
    ret = ret and dropout_p == 0.0
    ret = ret and hdim_v == 128
    ret = ret and hdim_q in (128, 192)
    ret = ret and nhead_q % nhead_k == 0
    ret = ret and not swa                      # no sliding window
    ret = ret and q.dtype in (bf16, fp8)
    ret = ret and logits_soft_cap == 0.0
    return ret
```

On MI300X the `ROCM_AITER_FA` backend uses `fmha_v3_varlen_fwd` (AMD ASM) for the
standard prefill path and `mha_varlen_fwd` (CK-tile) as fallback. Neither is the
Dao-AI FA3 implementation (which is CUDA/Hopper-specific).

---

## 5. ROCM\_ATTN vs TRITON\_ATTN: Full Comparison

### 5.1 High-level summary

| | `ROCM_ATTN` | `TRITON_ATTN` |
|---|---|---|
| **Prefill kernel** | `_fwd_kernel` (`prefix_prefill.py`) | `kernel_unified_attention` (`triton_unified_attention.py`) |
| **Decode kernel** | Separate: `paged_attention_rocm` (HIP C++) or `kernel_paged_attention_2d` | Same `kernel_unified_attention` — no separate kernel |
| **Number of GPU kernel launches** | 2 (one prefill, one decode) | 1 (handles the entire batch) |
| **Prefill/decode split** | Explicit: two separate kernel dispatches | Implicit: one kernel, per-sequence routing via binary search |
| **Skip decode in prefill kernel** | `if SKIP_DECODE and query_len == 1: return` | Not needed — 2D path handles any query length |
| **Skip prefill in decode kernel** | `if query_len > 1: return` | Not needed — IS\_3D=True only for pure-decode batches |
| **Chunked prefill algorithm** | Two-phase: Phase 1 = context loop (KV cache, no causal mask); Phase 2 = self loop (raw input tensors, causal) | Single unified loop over all KV positions; uniform causal mask; always reads from KV cache |
| **KV cache layout** | Separate K/V: `(2, num_blocks, block_size, num_kv_heads, head_size)` | Interleaved: `(num_blocks, num_kv_heads, block_size, 2×head_size)` |
| **When new tokens enter KV cache** | After `forward()` | Before `forward()` (registered as a dependency) |
| **Decode parallelism** | Partition-based (`_PARTITION_SIZE_ROCM=256`), separate HIP kernel | 3D softmax segments (`num_par_softmax_segments=16`), same kernel |

---

### 5.2 Dispatch Strategy

**`ROCM_ATTN` — two GPU kernel launches per forward pass:**

```python
# chunked_prefill_paged_decode.py:303
if max_query_len > 1:
    context_attention_fwd(...)        # Launch 1: prefill kernel
    # SKIP_DECODE=True → early return for decode seqs

# Always:
ops.paged_attention_rocm(...)         # Launch 2: decode HIP C++ kernel
# or kernel_paged_attention_2d(...)   # Launch 2: decode Triton fallback
# filter_by_query_len=True → early return for prefill seqs
```

**`TRITON_ATTN` — one GPU kernel launch:**

```python
# triton_unified_attention.py:1041
use_3d = not (
    ...
    or max_seqlen_q > 1         # any prefill in batch forces 2D (prefill-capable) path
    or num_seqs > seq_threshold_3D
    ...
)

if not use_3d:
    grid = (total_num_q_blocks, num_kv_heads)                       # 2D — prefill path
    tile_size = TILE_SIZE_PREFILL
else:
    grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)  # 3D — decode path
    tile_size = TILE_SIZE_DECODE

kernel_unified_attention[grid](...)   # single launch handles entire batch
```

Inside the kernel, a **binary search** (`find_seq_idx` over `query_start_loc`) maps each
workgroup's `q_block_global_idx` to the right sequence and local q-block position.
Decode sequences (`query_len=1`) get one workgroup that attends over their full `seq_len`.

---

### 5.3 Prefill Kernel Algorithm

**`ROCM_ATTN` `_fwd_kernel` — explicit two-phase:**

```python
context_len = seq_len - query_len

# Phase 1: cached context (old tokens), NO causal mask
for start_n in range(0, context_len, BLOCK_SIZE):
    bn = B_Loc[seq, token_indices // PHYSICAL_BLOCK_SIZE]
    K  = load(K_cache[bn, kv_head, ...])   # ← paged KV cache
    V  = load(V_cache[bn, kv_head, ...])
    qk = dot(Q, K) * scale                 # no mask
    acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, V)

# Phase 2: new query tokens, CAUSAL mask
for start_n in range(0, query_len, BLOCK_N):
    K  = load(K_raw[batch_start + start_n])   # ← raw input tensor, NOT cache
    V  = load(V_raw[batch_start + start_n])
    qk = dot(Q, K) * scale
    qk = where(query_row >= key_col, qk, -inf)   # causal
    acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, V)

output = acc / l_i
```

Phase 2 reads from **raw `K`/`V` input tensors** because the new tokens haven't been
written to cache yet when `forward()` runs.

---

**`TRITON_ATTN` `kernel_unified_attention` — single unified loop:**

```python
context_len = seq_len - query_len

# compute_tile_loop_bounds → [loop_lo, loop_hi) covers all of [0, seq_len)
for j in range(loop_lo, loop_hi):
    seq_offset   = j * TILE_SIZE + offs_t
    block_id     = block_table[seq_idx, seq_offset // BLOCK_SIZE]
    K = load(key_cache[block_id, kv_head, slot])    # ← always from KV cache
    V = load(value_cache[block_id, kv_head, slot])

    qk = dot(Q, K) * scale
    # Uniform causal mask over all positions:
    query_abs_pos = context_len + query_pos_in_chunk
    qk = where(query_abs_pos >= seq_offset, qk, -inf)

    acc, M, L = online_softmax_update(acc, M, L, qk, V)

output = acc / L
```

This works because `do_kv_cache_update` runs **before** `forward()`:

```python
# attention.py:549
kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)
unified_attention_with_output(..., kv_cache_dummy_dep=kv_cache_dummy_dep)
# ↑ dependency ensures the cache write completes first
```

By attention time, every position — both old context and new query tokens — is in the
paged cache and can be read uniformly through the block table.

The single causal mask handles both "phases" with one condition:

- `seq_offset < context_len`: `query_abs_pos >= seq_offset` is always true → attend freely (equivalent to ROCM Phase 1)
- `seq_offset >= context_len`: standard causal among new tokens (equivalent to ROCM Phase 2)

---

### 5.4 Decode Parallelism

**`ROCM_ATTN`** launches a separate HIP C++ kernel. Grid: `(num_seqs, num_kv_heads)`.
Long contexts are parallelized over 256-token partitions:

```python
_PARTITION_SIZE_ROCM = 256
max_num_partitions = (max_seq_len + 255) // 256

tmp_output = torch.empty(total_num_seq, num_query_heads, max_num_partitions, head_size)
exp_sums   = torch.empty(total_num_seq, num_query_heads, max_num_partitions)
max_logits = torch.empty_like(exp_sums)
# → reduce partitions to final output inside the kernel
```

**`TRITON_ATTN`** uses the IS\_3D=True path of `kernel_unified_attention`.
Grid: `(q_blocks, num_kv_heads, 16)`. Each of 16 segments independently processes
a slice of KV tiles; a separate `reduce_segments` kernel merges them:

```python
# IS_3D=True — each segment writes partials:
tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS * TILE_SIZE)
segm_output[token, head, segm_idx, :] = partial_acc
segm_max[token, head, segm_idx]       = partial_m
segm_expsum[token, head, segm_idx]    = partial_l

# reduce_segments:
for s in range(act_num_segments):
    alpha = exp(segm_max[s] - global_max)
    global_acc += segm_output[s] * alpha
    global_lse += segm_expsum[s] * alpha
output = global_acc / global_lse
```

This is FlashDecoding-style split-KV parallelism, giving better GPU occupancy for
long-context decode in small batches.

---

### 5.5 KV Cache Layout

This is the root reason the prefill algorithms differ.

**`ROCM_ATTN`** — K and V in **separate** arrays:

```python
# RocmAttentionBackend.get_kv_cache_shape:
return (2, num_blocks, block_size, num_kv_heads, head_size)

key_cache   = kv_cache[0]   # (num_blocks, block_size, num_kv_heads, head_size)
value_cache = kv_cache[1]
```

New query tokens' raw `K`/`V` arrive as separate tensors and can be read directly in
Phase 2, before the cache update step.

**`TRITON_ATTN`** — K and V **interleaved** in the last dimension:

```python
# TritonAttentionBackend.get_kv_cache_shape:
return (num_blocks, num_kv_heads, block_size, 2 * head_size)

# In forward():
kv_cache = kv_cache.transpose(1, 2)             # → (num_blocks, block_size, num_kv_heads, 2*head_size)
key_cache, value_cache = kv_cache.split(hs, -1) # → each (num_blocks, block_size, num_kv_heads, head_size)
```

Because the cache update runs before `forward()`, the kernel reads everything (old and
new) through the block table.

---

### 5.6 Capability Comparison

| Feature | `ROCM_ATTN` | `TRITON_ATTN` |
|---|---|---|
| Per-token-head quant (INT8 / FP8 / INT4 KV) | No | Yes — inline dequant in kernel |
| `ENCODER_DECODER` cross-attention | **No** — explicitly blocked; two-phase approach would incorrectly mix encoder K/V | Yes |
| KV connectors (disaggregated prefill) | **No** — layout incompatible with blocks-first format | Yes |
| Sink tokens | Partial — falls back to Triton for decode | Full native support |
| Logit soft cap | Via Triton fallback only | Native |
| Alibi slopes | Yes | Yes |
| Sliding window attention | Yes | Yes |
| 3D parallel decode | No (partition-based, separate kernel) | Yes (`num_par_softmax_segments=16`) |
| FP8 query with non-1.0 q\_scale | Not supported | Supported |
| `ENCODER_ONLY` / `ENCODER` attention | Yes (direct Q/K/V path) | Yes |
| Batch invariance for CUDA graph | No | Yes (`supports_batch_invariance=True`) |
| Supported platforms | ROCm only | Any (ROCm, CUDA, XPU) |

---

### 5.7 Worked Example: Mixed Batch Through Both Backends

**Batch:** Seq A (`query_len=256, ctx_len=256, seq_len=512`), Seq B (`query_len=1, seq_len=512`)

```
query[0:256]   = Seq A tokens
query[256:257] = Seq B token
```

#### ROCM\_ATTN execution

```
Launch 1 — context_attention_fwd (max_query_len=256 > 1):
  Seq A workgroups:
    Phase 1: Q[0:256] × K_cache[blocks 0-7], no mask    → partial acc
    Phase 2: Q[0:256] × K_raw[0:256], causal mask       → merge into acc
    → write output[0:256]
  Seq B workgroups:
    query_len==1 → SKIP_DECODE → return immediately

Launch 2 — paged_attention_rocm:
  Seq A:  query_len=256 > 1 → filter_by_query_len → return
  Seq B:  iterate seq_len=512 across 2 partitions, reduce
    → write output[256:257]
```

#### TRITON\_ATTN execution

```
KV cache update runs first (dependency):
  key_cache[A's new blocks] ← A's 256 new tokens written
  key_cache[B's block]      ← already current

max_seqlen_q=256 > 1 → use_3d=False → 2D path
grid = (q_blocks, num_kv_heads)

kernel_unified_attention (single launch):
  Seq A workgroups:
    context_len=256
    unified loop [0, 512): old tokens [0,256) + new tokens [256,512) — all via block table
    causal mask: query_abs = 256 + query_row; gate on >= seq_offset
    → write output[0:256]

  Seq B workgroup:
    context_len=511
    unified loop [0, 512): query_abs=511 >= all seq_offsets → attend all
    → write output[256:257]
```

Both backends produce identical attention outputs. `ROCM_ATTN` uses two kernel launches;
`TRITON_ATTN` uses one.
