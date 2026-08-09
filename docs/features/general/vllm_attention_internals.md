# vLLM Attention Internals: Block Table & Attention Metadata

## Table of Contents

1. [How Block Table Works](#1-how-block-table-works)
2. [Why a Separate Attention Metadata Builder](#2-why-a-separate-attention-metadata-builder)
3. [Attention Metadata Fields In Detail](#3-attention-metadata-fields-in-detail)

---

## 1. How Block Table Works

### What Is a Block Table?

The KV cache on GPU is too large to allocate contiguously per-request. Instead, vLLM uses **paged memory**: the GPU KV cache is divided into fixed-size **physical blocks** (e.g., 16 tokens each), and each request is assigned a list of physical blocks. The **block table** is the mapping from *logical block index* (request-local) to *physical block ID* (global GPU index).

### Data Structures

**`KVCacheBlock`** — `vllm/v1/core/kv_cache_utils.py:118`

The atomic unit of physical KV cache memory. Each block is a fixed-size chunk of GPU memory holding KV tensors for `block_size` tokens.

```
KVCacheBlock:
  block_id: int       ← index into the flat GPU KV cache tensor
  ref_cnt: int        ← 0 = free, >0 = in use
  _block_hash: ...    ← set when full, for prefix caching
```

**`BlockTable`** — `vllm/v1/worker/block_table.py:24`

A 2D int32 tensor of shape `[max_num_reqs, max_num_blocks_per_req]`:

```
block_table[req_idx][logical_block_idx] = physical_block_id
```

It also holds a flat `slot_mapping` tensor mapping each scheduled token to a specific KV cache slot.

### Step-by-Step Example

Setup:

- `block_size = 4` (4 tokens per block)
- 3 requests: A (7-token prefill), B (1 decode token, 10 total context), C (3-token extend, 7 total context)
- Physical blocks assigned: A → [5, 9], B → [2, 7, 3], C → [12, 8]

#### Step 1 — Scheduler allocates physical blocks

`vllm/v1/core/kv_cache_manager.py:283` → `vllm/v1/core/single_type_kv_cache_manager.py:321`

For each request, `ceil(num_tokens / block_size)` blocks are popped from `BlockPool.free_queue`:

```
Request A (7 tokens) → needs 2 blocks → gets [block_id=5, block_id=9]
Request B (5 tokens) → needs 2 blocks → gets [block_id=2, block_id=7]
Request C (3 tokens) → needs 1 block  → gets [block_id=12]
```

On the scheduler side:

```
req_to_blocks["A"] = [KVCacheBlock(id=5), KVCacheBlock(id=9)]
req_to_blocks["B"] = [KVCacheBlock(id=2), KVCacheBlock(id=7)]
req_to_blocks["C"] = [KVCacheBlock(id=12)]
```

New block IDs are sent to the worker in `SchedulerOutput`.

#### Step 2 — Worker populates the block table tensor

`vllm/v1/worker/gpu_input_batch.py:381`

The block table is a CPU-pinned tensor, later DMA'd to GPU:

```
                 blk[0]  blk[1]  blk[2]  ...
req_idx=0 (A):    5       9       0       0
req_idx=1 (B):    2       7       0       0
req_idx=2 (C):   12       0       0       0
```

`0` is the null/padding block ID — a sentinel that attention backends skip.

#### Step 3 — Compute `slot_mapping`

`vllm/v1/worker/block_table.py:348` (`_compute_slot_mapping_kernel`)

For each scheduled token at position `pos` in its request:

```
logical_block_idx   = pos // block_size
offset_within_block = pos % block_size
physical_block_id   = block_table[req_idx][logical_block_idx]
slot_id             = physical_block_id * block_size + offset_within_block
```

For request A (7 tokens, block_size=4, blocks=[5, 9]):

```
pos 0 → block_table[0][0]=5 → slot = 5*4+0 = 20
pos 1 → block_table[0][0]=5 → slot = 5*4+1 = 21
pos 2 → block_table[0][0]=5 → slot = 5*4+2 = 22
pos 3 → block_table[0][0]=5 → slot = 5*4+3 = 23
pos 4 → block_table[0][1]=9 → slot = 9*4+0 = 36
pos 5 → block_table[0][1]=9 → slot = 9*4+1 = 37
pos 6 → block_table[0][1]=9 → slot = 9*4+2 = 38
```

`slot_mapping` is a flat tensor concatenating all requests' token slots.

#### Step 4 — KV cache write (prefill and decode)

`vllm/v1/attention/backends/flash_attn.py:1071`

After the attention layer computes key/value tensors, `reshape_and_cache_flash` scatters them into the GPU KV cache using `slot_mapping`:

```
kv_cache[slot_id // block_size, :, slot_id % block_size, :] = key/value
```

For token at pos=4 of request A: `slot_id=36` → written to `kv_cache[9, :, 0, :]` (physical block 9, offset 0).

#### Step 5 — Attention reads via block table

`vllm/v1/attention/backends/flash_attn.py:1027`

`flash_attn_varlen_func(..., block_table=block_table, ...)` reads KV from non-contiguous physical blocks. For request A at decode step (token 7), it attends to all past KV at pos 0–6, following `block_table[0] = [5, 9, ...]` to gather KV from physical blocks 5 and 9.

### The Full Picture

```
Scheduler side                         Worker/GPU side
──────────────                         ───────────────
req_to_blocks["A"]                     block_table tensor (CPU→GPU)
= [KVCacheBlock(5),                    [req=0: 5, 9, 0, 0, ...]
   KVCacheBlock(9)]                    [req=1: 2, 7, 0, 0, ...]
                                       [req=2: 12, 0, 0, 0, ...]
       │                                         │
       │ new_block_ids in SchedulerOutput        │ _compute_slot_mapping_kernel
       ▼                                         ▼
BlockPool (free list)              slot_mapping = [20,21,22,23,36,37,38, ...]
allocate / free / LRU evict                       │           │
                                                  │           ▼
                                    reshape_and_cache_flash   flash_attn_varlen_func
                                    (writes KV → cache)       (reads KV ← cache via block_table)
```

### Key Concepts Summary

| Concept | Details |
|---|---|
| **Block size** | Fixed token count per physical block (e.g., 16). Set at startup. |
| **Physical block ID** | Index into the flat `kv_cache` tensor on GPU. |
| **block_table tensor** | 2D `int32[max_reqs, max_blocks_per_req]` mapping logical→physical. |
| **slot_mapping** | Flat `int64` tensor: one entry per scheduled token → flat KV cache index. |
| **Null block (ID=0)** | Padding sentinel; slots mapping to it get `PAD_SLOT_ID=-1`, skipped. |
| **Prefix caching** | A completed, hashed block is kept in `BlockPool` LRU and reused on cache hit — no recomputation needed. |
| **Sliding window** | Out-of-window blocks are replaced with the null block in `req_to_blocks`. |

---

## 2. Why a Separate Attention Metadata Builder

### The Problem

Different attention backends need fundamentally different metadata. Computing that metadata is complex, stateful, and must be GPU-graph-safe. A single struct cannot serve all backends, so vLLM uses the builder pattern.

The abstract base is at `vllm/v1/attention/backend.py:607`:

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    @abstractmethod
    def build(self, common_prefix_len, common_attn_metadata, ...) -> M: ...
```

### Reasons for the Builder Pattern

**1. Backend heterogeneity**

Each backend implements its own builder returning its own metadata subclass. FlashAttention needs `block_table + seq_lens + scheduler_metadata`; FlashInfer needs wrapper objects; Mamba needs SSM state indices. There is no single struct that fits all backends.

**2. Per-layer/per-group specialization**

`vllm/v1/worker/utils.py:243` — the model runner groups layers into `AttentionGroup` objects, one builder per group:

```
Group 0: layers using FlashAttn (full attention),     kv_cache_spec=full
Group 1: layers using FlashAttn (sliding window),     kv_cache_spec=sliding_window_N
Group 2: layers using Mamba SSM,                      kv_cache_spec=mamba
```

Hybrid models (e.g., Jamba, Zamba) have layers that need different metadata simultaneously within the same forward pass.

**3. Pre-allocated GPU-graph-safe buffers**

The builder's `__init__` pre-allocates persistent tensors at startup. CUDA graph replay forbids dynamic allocation — keeping these in the builder keeps `build()` allocation-free:

```python
# FlashAttentionMetadataBuilder.__init__, flash_attn.py:356
self.scheduler_metadata         = torch.zeros(1 + round_up(max_batch_size, 4) * 4, int32)
self._dcp_context_kv_lens       = torch.zeros(max_num_reqs, int32)
self.persistent_rswa_prefix_lens = ...
```

**4. Fast reuse across similar groups**

`vllm/v1/worker/gpu_model_runner.py:2495` — when two KV cache groups share the same backend and differ only in their block table, `update_block_table()` shallow-copies cached metadata and patches just the block table tensor, skipping full recomputation.

**5. CUDA graph support negotiation**

Each builder class declares `_cudagraph_support` (ALWAYS / UNIFORM_BATCH / UNIFORM_SINGLE_TOKEN_DECODE / NEVER) and `supports_update_block_table`. The model runner queries these at graph-capture time to decide the graph mode.

**6. Complex per-batch logic isolation**

The builder encapsulates cascade attention detection, AOT tile scheduling (FA3), DCP rank-local KV length computation, and sliding-window symmetrization. These vary per batch and cannot be statically compiled into the model.

### The Build Flow

```
execute_model()
    │
    ▼
_build_attention_metadata()                    gpu_model_runner.py:2256
    │
    ├── construct CommonAttentionMetadata       (block_table, slot_mapping, seq_lens, ...)
    │
    └── for each AttentionGroup:
            │
            ├── builder.build(common_prefix_len, cm)      ← full build
            │       OR
            ├── builder.update_block_table(cached, ...)   ← fast reuse
            │       OR
            └── builder.build_for_cudagraph_capture(cm)   ← static shapes
            │
            └── result stored in attn_metadata[layer_name]
                    │
                    ▼
            model layers pull their metadata by name during forward()
```

The key design insight: `CommonAttentionMetadata` contains the **batch state** (agnostic to backend), while each backend's builder translates it into exactly what its kernel needs. This isolates backend-specific complexity behind a single `build()` call.

---

## 3. Attention Metadata Fields In Detail

### Setup: Running Example Batch

Used throughout this section:

- `block_size = 4`
- 3 requests:
    - **Req A**: prefill, 6 tokens (positions 0–5)
    - **Req B**: decode, 1 new token (position 10), 10 tokens total context
    - **Req C**: extend (chunked prefill), 3 new tokens (positions 4–6), 7 tokens total context
- Physical blocks: A→[5,9], B→[2,7,3], C→[12,8]

---

### `CommonAttentionMetadata` Fields

`vllm/v1/attention/backend.py:401`

Shared, backend-agnostic batch state computed once per step and passed to every builder's `build()`.

---

#### `query_start_loc` / `query_start_loc_cpu`

**Shape**: `(num_reqs + 1,)` int32

Cumulative sum of query lengths. Tells you where each request's tokens begin in the flat token dimension. The CPU copy avoids GPU→CPU sync in control-flow paths.

```
query_lens = [6, 1, 3]   # A: 6 prefill, B: 1 decode, C: 3 extend

query_start_loc = [0, 6, 7, 10]
                   ^  ^  ^   ^
                   A  B  C  end
```

Tokens `[0:6]` belong to A, `[6:7]` to B, `[7:10]` to C. FlashAttention's varlen API uses this directly as `cu_seqlens_q`.

---

#### `seq_lens`

**Shape**: `(num_reqs,)` int32

Total KV context length for each request after this step — how many past tokens the query must attend to.

```
seq_lens = [6, 10, 7]
            ^   ^   ^
            A   B   C
```

- A: 6 (just finished prefill; context = all 6 tokens)
- B: 10 (9 prior tokens + 1 new decode token)
- C: 7 (4 prior tokens + 3 new tokens)

Used by FlashAttention as `cu_seqlens_k` (via prefix-sum) to bound how far back each query attends.

---

#### `num_reqs`, `num_actual_tokens`, `max_query_len`, `max_seq_len`

Scalar sizing fields:

```
num_reqs          = 3     # requests in this batch
num_actual_tokens = 10    # 6+1+3; may be padded to a multiple of 8
max_query_len     = 6     # longest query = A's 6-token prefill
max_seq_len       = 10    # longest context = B's 10-token history
```

These drive kernel grid dimensions and workspace allocation. `max_query_len=1` signals all-decode batches, enabling packed-GQA optimizations in FA2.

---

#### `block_table_tensor`

**Shape**: `(num_reqs, max_blocks_per_req)` int32

```
           blk0  blk1  blk2
req A(0):   5     9     0     # 0 = null/padding
req B(1):   2     7     3
req C(2):  12     8     0
```

FlashAttention uses this to gather KV from non-contiguous GPU memory during attention computation.

---

#### `slot_mapping`

**Shape**: `(num_actual_tokens,)` int64

Flat array mapping each scheduled token to its physical KV cache slot (where to *write* its KV vectors).

```
Token idx  Req  Pos  block_id  offset   slot = block_id*4 + offset
0          A    0    5         0        20
1          A    1    5         1        21
2          A    2    5         2        22
3          A    3    5         3        23
4          A    4    9         0        36
5          A    5    9         1        37
6          B    10   3         2        14
7          C    4    8         0        32
8          C    5    8         1        33
9          C    6    8         2        34
```

`reshape_and_cache_flash` uses this to scatter-write newly computed key/value tensors into the KV cache.

---

#### `causal`

**Type**: `bool | torch.Tensor`

Controls the attention mask shape.

- `True` (default): standard autoregressive causal mask.
- `False`: bidirectional (encoder self-attention, cross-attention).
- `torch.Tensor` of shape `(num_reqs,)`: per-request flag for mixed batches.

```python
# Multimodal batch: image request (non-causal) + text request (causal)
causal = tensor([False, True])
```

---

#### `encoder_seq_lens` / `encoder_seq_lens_cpu`

**Shape**: `(num_reqs,)` int32 — only for encoder-decoder models (Whisper, T5).

Length of the encoder output each decoder request attends to in cross-attention layers. `None` for decoder-only models.

```python
# Two audio clips with different lengths
encoder_seq_lens = tensor([1500, 800])
```

---

#### `dcp_local_seq_lens` / `dcp_local_seq_lens_cpu`

**Shape**: `(num_reqs,)` int32 — only when **Decode Context Parallelism (DCP)** is active.

In DCP, the KV cache is sharded across tensor-parallel ranks. This field holds the number of KV tokens this rank is responsible for, per request.

```python
# dcp_world_size=2, seq_lens=[10, 6]
# rank 0:
dcp_local_seq_lens = tensor([5, 3])   # first half of each sequence
# rank 1:
dcp_local_seq_lens = tensor([5, 3])   # second half
```

---

#### `positions`

**Shape**: `(num_actual_tokens,)` int64. Optional.

Absolute token position for positional encoding. Set when position-dependent sparse metadata needs to be precomputed (e.g., DeepSeek V4's C128A layers).

```python
positions = tensor([0,1,2,3,4,5,  10,  4,5,6])
#                   ←── A ──────►  B   ←─ C ─►
```

---

#### `is_prefilling`

**Shape**: `(num_reqs,)` bool. Optional.

True if the request is still in the prefill phase. Used to distinguish genuine decodes (query_len=1) from chunked-prefill extends.

```python
is_prefilling = tensor([True, False, True])
#                         A      B      C
# B is False: genuine decode (one new token after completed prompt)
```

---

#### `seq_lens_cpu_upper_bound`

**Shape**: `(num_reqs,)` int32 CPU tensor. Optional.

An optimistic upper bound on `seq_lens` used only where avoiding GPU→CPU sync matters more than exactness. For speculative decoding it assumes all draft tokens were accepted.

```python
# Spec decode, 3 draft tokens proposed for req B:
seq_lens_cpu_upper_bound = tensor([6, 13, 7])
#                                       ^^ assumes all 3 drafts accepted (true: 10–13)
```

Only used for control-flow decisions, never passed to kernels needing exact lengths.

---

#### `mm_req_doc_ranges`

**Type**: `dict[int, list[tuple[int, int]]]` — only for multimodal PrefixLM models.

Maps request index → list of `(start_pos, end_pos)` ranges where **bidirectional attention** applies (e.g., image patch tokens).

```python
# req 0: image tokens at positions 5–20
mm_req_doc_ranges = {
    0: [(5, 20)],
    # req 1 is text-only → absent
}
```

The builder converts this into `mm_prefix_range_tensor` (a GPU tensor) for the kernel.

---

#### `rswa_prefix_lens`

**Shape**: `(num_reqs,)` int32 — only for **Reference Sliding Window Attention (R-SWA)** models.

R-SWA is a hybrid attention pattern: prompt/image tokens remain globally visible; generated tokens additionally see only a fixed sliding window of recent tokens. This field stores the prompt length (the boundary) for each request.

```python
rswa_prefix_lens = tensor([120, 80])
# Tokens 0–119 of req A are globally visible; tokens 120+ follow the sliding window rule.
```

---

### `FlashAttentionMetadata` Fields

`vllm/v1/attention/backends/flash_attn.py:231`

Backend-specific fields the builder computes on top of the common ones.

---

#### `num_actual_tokens`, `max_query_len`, `query_start_loc`, `max_seq_len`, `seq_lens`

Copied directly from `CommonAttentionMetadata` — same semantics as above, now held in the FA-specific struct for the kernel to consume.

---

#### `block_table`, `slot_mapping`

Copied from `CommonAttentionMetadata.block_table_tensor` and `.slot_mapping`. Renamed to match FlashAttention's expected field names.

---

#### `use_cascade`, `common_prefix_len`, `cu_prefix_query_lens`, `prefix_kv_lens`, `suffix_kv_lens`

**Cascade attention** — an optimization for batches where all requests share a long common prefix (e.g., a system prompt). Instead of each request redundantly attending to the same prefix KV, cascade splits attention into two passes:

1. **Prefix pass**: all queries attend to the shared prefix KV blocks at once.
2. **Suffix pass**: each query attends to its own unique suffix.

Results are merged via the log-sum-exp trick.

```python
# All 3 requests share a 100-token system prompt
common_prefix_len = 100
use_cascade = True

# Prefix pass: treat all 10 tokens as one sequence attending to the prefix
cu_prefix_query_lens = tensor([0, 10])  # one "batch" of all tokens
prefix_kv_lens       = tensor([100])    # attends to 100 prefix KV tokens

# Suffix pass: each request attends to its own suffix
suffix_kv_lens = seq_lens - 100  # per-request suffix context lengths
```

When `use_cascade=False`, `cu_prefix_query_lens`, `prefix_kv_lens`, and `suffix_kv_lens` are all `None` and a single standard attention pass runs.

---

#### `max_dcp_context_kv_len`, `dcp_context_kv_lens`

**Decode Context Parallelism (DCP)** fields. In DCP, KV tokens are sharded across TP ranks; each rank runs attention only over its local KV slice.

- `dcp_context_kv_lens`: per-request KV token count this rank holds, shape `(num_reqs,)`
- `max_dcp_context_kv_len`: max across requests, for workspace sizing without GPU→CPU sync

```python
# dcp_world_size=2, rank=0, seq_lens=[10, 6], query_lens=[1, 1]
# context_kv_lens = [9, 5]; rank 0 holds first half:
dcp_context_kv_lens    = tensor([5, 3])
max_dcp_context_kv_len = 5
```

---

#### `num_decode_reqs`, `num_prefill_reqs`, `num_decode_tokens`, `num_prefill_tokens`

In DCP with mixed prefill+decode batches, FA2 handles them separately (decode rows attend to DCP-sharded KV; pure-prefill rows do not). The builder reorders the batch (decodes first) and records the split counts.

```python
# Batch: [req_A=decode(1 tok), req_B=prefill(6 toks), req_C=decode(1 tok)]
# After reorder: [A, C, B]
num_decode_reqs    = 2   # A, C
num_prefill_reqs   = 1   # B
num_decode_tokens  = 2
num_prefill_tokens = 6
```

All four are `0` when DCP is off or the batch is all-decode.

---

#### `scheduler_metadata`, `prefix_scheduler_metadata`, `max_num_splits`

**FlashAttention3 AOT tile scheduling.** FA3 pre-computes which GPU thread blocks handle which attention tiles before the kernel launches, improving GPU utilization for variable-length sequences.

- `scheduler_metadata`: int32 tensor filled by FA3's `get_scheduler_metadata()` with tile assignments for the main attention pass.
- `prefix_scheduler_metadata`: same for the cascade prefix pass.
- `max_num_splits`: how many output splits FA3 uses. `0` = FA3's internal heuristic. `> 1` only under CUDA graphs (pre-allocated buffers required).

```python
# FA3, batch of 3 reqs
scheduler_metadata = get_scheduler_metadata(
    batch_size=3, cu_seqlens_q=[0,6,7,10],
    max_seqlen_q=6, max_seqlen_k=10, ...
)  # → int32 tensor pre-assigning tiles to SM thread blocks
max_num_splits = 4  # when CUDA graph captured
```

All three are `None` for FA2.

---

#### `causal`

Same as in `CommonAttentionMetadata` — copied through. The builder may convert a bool tensor from int8 to int32 (`flash_attn.py:651`) for kernel compatibility.

---

#### `sliding_window`

**Type**: `tuple[int, int] | None`

Restricts how far back each token attends. The tuple is `(left_window, right_window)`.

- `None` / `(-1, -1)`: full attention.
- `(W-1, 0)`: causal sliding window of size W.
- `(W, W)`: symmetric window — applied automatically when attention is non-causal (image tokens).

```python
# Mistral-7B, window_size=4096, causal attention:
sliding_window = (4095, 0)

# Same layer, non-causal (multimodal tokens):
sliding_window = (4095, 4095)   # symmetrized by _maybe_symmetrize_window()

# Llama (full attention):
sliding_window = (-1, -1)
```

---

#### `mm_prefix_range_tensor`

**Shape**: `(num_reqs, max_ranges, 2)` int32 — only for multimodal PrefixLM models.

GPU tensor form of `CommonAttentionMetadata.mm_req_doc_ranges`. Each `[i, j, :]` entry is `[start_pos, end_pos]` for the j-th bidirectional range of request i. Used by the attention kernel to apply a non-causal mask to image token ranges only.

```python
# req 0: image tokens at positions 5–20; req 1: text only
mm_prefix_range_tensor[0, 0] = tensor([5, 20])
mm_prefix_range_tensor[1, 0] = tensor([-1, -1])   # padding sentinel
```

---

#### `rswa_prefix_lens`, `rswa_window`, `rswa_window_tensor`

**Reference Sliding Window Attention (R-SWA)** fields, built from `CommonAttentionMetadata.rswa_prefix_lens`.

The builder copies `rswa_prefix_lens` into a **pre-allocated persistent GPU tensor** so no allocation happens during CUDA graph replay. `rswa_window` (a scalar from model config) is stored in `rswa_window_tensor` (a `[1]` int32 CUDA tensor) for kernel access without CPU→GPU copies during forward.

```python
# Model: rswa_window=512, req A has 120-token prompt, req B has 80-token prompt
rswa_prefix_lens   = persistent_tensor[:2]   # tensor([120, 80]) on GPU, pre-allocated
rswa_window        = 512                     # scalar int for logic checks
rswa_window_tensor = persistent_1elem        # tensor([512]) on GPU, no alloc at runtime
```

During forward: if `token_pos < rswa_prefix_lens[req]` → global attention; else → sliding window of size `rswa_window`.

---

### Summary Table

| Field | Shape | When set | Purpose |
|---|---|---|---|
| `query_start_loc` | `(B+1,)` | always | Varlen query boundaries for FA |
| `seq_lens` | `(B,)` | always | Total KV context per request |
| `num_actual_tokens` | scalar | always | Flat token count (incl. padding) |
| `max_query_len` | scalar | always | Kernel grid sizing; =1 triggers decode-only fast path |
| `max_seq_len` | scalar | always | KV dimension bound |
| `block_table` | `(B, max_blk)` | always | Logical→physical block mapping |
| `slot_mapping` | `(T,)` | always | Per-token KV write address |
| `causal` | bool or `(B,)` | always | Mask shape |
| `sliding_window` | `(2,)` tuple | sliding-window models | Attention window bounds |
| `use_cascade` / `common_prefix_len` | bool + scalar | shared prefix exists | Enable two-pass cascade attention |
| `cu_prefix_query_lens` / `prefix_kv_lens` / `suffix_kv_lens` | tensors | cascade on | Cascade pass boundaries |
| `scheduler_metadata` | int32 tensor | FA3 only | Pre-assigned tile schedule |
| `prefix_scheduler_metadata` | int32 tensor | FA3 + cascade | Tile schedule for prefix pass |
| `max_num_splits` | scalar | FA3 + CUDA graph | Split-K count for pre-alloc |
| `encoder_seq_lens` | `(B,)` | encoder-decoder models | Cross-attention KV lengths |
| `dcp_context_kv_lens` / `max_dcp_context_kv_len` | `(B,)` + scalar | DCP enabled | Per-rank KV lengths |
| `num_decode/prefill_reqs/tokens` | 4 scalars | DCP + mixed batch | Batch split point for FA2 DCP |
| `is_prefilling` | `(B,)` bool | mixed batches | Distinguish decode from chunked prefill |
| `positions` | `(T,)` | sparse attention models | Token positions for sparse pattern precompute |
| `seq_lens_cpu_upper_bound` | `(B,)` CPU | spec decode | Optimistic seq_lens for async control flow |
| `mm_req_doc_ranges` → `mm_prefix_range_tensor` | `(B, R, 2)` | multimodal PrefixLM | Bidirectional ranges for image tokens |
| `rswa_prefix_lens` / `rswa_window` / `rswa_window_tensor` | `(B,)` + scalar + `(1,)` | R-SWA models | Prompt/window boundary for hybrid attention |
