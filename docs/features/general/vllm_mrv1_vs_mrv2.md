# MRV1 vs MRV2: In-Depth Comparison with Examples

## Table of Contents

1. [What Are MRV1 and MRV2?](#1-what-are-mrv1-and-mrv2)
2. [Running Example Setup](#2-running-example-setup)
3. [Token ID Storage and `input_ids` Construction](#3-token-id-storage-and-input_ids-construction)
4. [Block Table: Full DMA vs Staged Scatter](#4-block-table-full-dma-vs-staged-scatter)
5. [Sampled Token Output: Synchronous vs Async D2H](#5-sampled-token-output-synchronous-vs-async-d2h)
6. [Request Removal / Finish: condense() vs free_indices](#6-request-removal-finish-condense-vs-free_indices)
7. [Other Differences](#7-other-differences)
8. [MRV2 on ROCm](#8-mrv2-on-rocm)
9. [Summary Table](#9-summary-table)

---

## 1. What Are MRV1 and MRV2?

Both live inside vLLM's v1 engine. They share the same scheduler, KV cache manager, EngineCore loop, and API server. The difference is entirely in the **GPU worker's model runner** — the component that takes a `SchedulerOutput` and produces sampled token IDs.

- **MRV1** — `vllm/v1/worker/gpu_model_runner.py`: the original v1 model runner, the production default.
- **MRV2** — `vllm/v1/worker/gpu/model_runner.py`: a cleaner rewrite with architectural improvements. Enabled via `VLLM_USE_V2_MODEL_RUNNER=1`.

MRV2 is the planned long-term replacement. The codebase has comments:

```python
# TODO(Wentao): Use full version instead of import when fully migrated to v2
```

The core theme of MRV2: move work from **CPU→GPU copies** into **GPU-native Triton kernels**, keep more state persistent on GPU, and use async D2H transfers to overlap output copies with the next step's computation.

---

## 2. Running Example Setup

Used throughout this document:

- `max_num_reqs = 4`, `max_model_len = 8`, `block_size = 4`
- 3 active requests at the start of a decode step:
    - **Req A** (slot 0): 6 computed tokens, decode 1 more, physical blocks [5, 9]
    - **Req B** (slot 1): 4 computed tokens, decode 1 more, physical blocks [2]
    - **Req C** (slot 2): new prefill this step, 3 tokens, physical blocks [12]

---

## 3. Token ID Storage and `input_ids` Construction

### MRV1 — CPU 2D table + `torch.index_select`

**Source**: `vllm/v1/worker/gpu_input_batch.py:133`

MRV1 maintains a `token_ids_cpu` array of shape `[max_num_reqs, max_model_len]` in non-pinned CPU memory. Each row is one request's full token history, zero-padded.

After adding all three requests, the table looks like:

```
token_ids_cpu  [max_num_reqs=4, max_model_len=8]  (CPU, int32, NOT pinned)

slot │ tok0  tok1  tok2  tok3  tok4  tok5  tok6  tok7
─────┼────────────────────────────────────────────────
  0  │  101   202   303   404   505   606     0     0   ← Req A (6 tokens)
  1  │  111   222   333   444     0     0     0     0   ← Req B (4 tokens)
  2  │  151   252   353     0     0     0     0     0   ← Req C (3 tokens, new)
  3  │    0     0     0     0     0     0     0     0   ← empty slot
```

At each step, `_prepare_inputs` must produce a flat `input_ids` for the scheduled tokens.
For this step: Req A at pos 5 (1 token), Req B at pos 3 (1 token), Req C at pos 0–2 (3 tokens).

The code builds flat indices into the **flattened** table (`gpu_model_runner.py:1983`):

```python
# positions_np for each scheduled token
positions_np = [5,   3,   0,  1,  2]
#               ^A   ^B   ^────C────^

# req_indices: which row in token_ids_cpu
req_indices  = [0,   1,   2,  2,  2]

# token_indices = positions_np + req_indices * max_model_len (=8)
token_indices = [5+0*8, 3+1*8, 0+2*8, 1+2*8, 2+2*8]
              = [5,     11,    16,    17,    18]
```

`torch.index_select` gathers from the flattened table:

```
flattened token_ids_cpu (32 elements):
idx: 0   1   2   3   4   5   6   7  | 8   9  10  11  12  13 ...| 16  17  18 ...
val:101 202 303 404 505 606   0   0  |111 222 333 444   0   0  ...|151 252 353 ...

select indices [5, 11, 16, 17, 18]:
→ [606, 444, 151, 252, 353]
```

This result is copied into `self.input_ids.cpu[:5]`, then **DMA'd to GPU**:

```
CPU → GPU (explicit cudaMemcpy):
input_ids_gpu = [606, 444, 151, 252, 353]   shape=(5,)  int32
```

**Cost per step**: CPU gather O(T) + PCIe DMA of `(total_scheduled_tokens,)` int32.

---

### MRV2 — `StagedWriteTensor` on GPU + Triton gather kernel

**Source**: `vllm/v1/worker/gpu/states.py:33`, `vllm/v1/worker/gpu/input_batch.py`

MRV2 stores `all_token_ids` as a `StagedWriteTensor` with `uva_instead_of_gpu=True`:

```python
# states.py:33
self.all_token_ids = StagedWriteTensor(
    (self.max_num_reqs, self.max_model_len),
    dtype=torch.int32,
    device=device,
    uva_instead_of_gpu=True,   # lives in CPU pinned memory, GPU-visible via UVA
)
```

The `uva_instead_of_gpu=True` flag means the backing store is a CPU-pinned `UvaBuffer` (accessible by the GPU over PCIe without an explicit copy) rather than a dedicated GPU allocation. This saves GPU memory for the potentially huge `[max_num_reqs, max_model_len]` table.

When Req C is added, only its token IDs are **staged** as a diff:

```python
# req_states.add_request(all_token_ids=[151, 252, 353], ...)
self.all_token_ids.stage_write(req_idx=2, start=0, x=[151, 252, 353])

# Internally:
_staged_write_indices  = [2]
_staged_write_starts   = [0]
_staged_write_contents = [151, 252, 353]
_staged_write_cu_lens  = [3]
```

At the start of `prepare_inputs`, `apply_staged_writes()` launches a Triton kernel that scatter-writes these 3 values into the UVA-backed `all_token_ids` array. After the flush:

```
all_token_ids GPU-visible table [4, 8] (UVA-pinned, int32):

slot │ tok0  tok1  tok2  tok3  tok4  tok5  tok6  tok7
─────┼────────────────────────────────────────────────
  0  │  101   202   303   404   505   606     0     0   ← Req A (unchanged)
  1  │  111   222   333   444     0     0     0     0   ← Req B (unchanged)
  2  │  151   252   353     0     0     0     0     0   ← Req C (just written by Triton)
  3  │    0     0     0     0     0     0     0     0
```

For prefill tokens (Req C), the Triton kernel `prepare_prefill_inputs` gathers directly from `all_token_ids.gpu` into `input_buffers.input_ids`:

```
Triton gather kernel: reads all_token_ids.gpu row 2, cols [0,1,2]
→ writes to input_ids[2], input_ids[3], input_ids[4]

input_ids (GPU) = [___, ___, 151, 252, 353]
                   ^A   ^B   ^────C────^
```

For decode tokens (Req A, B), they come from `last_sampled_tokens` — already on GPU from the previous step's sampling — written into positions [0] and [1] by `combine_sampled_and_draft_tokens` (another Triton kernel). No PCIe transfer needed for these tokens at all.

**Key difference**: no CPU gather, no PCIe DMA for token IDs. The only data crossing PCIe is the staged diff (3 values for Req C) written via the Triton scatter kernel.

---

## 4. Block Table: Full DMA vs Staged Scatter

### MRV1 — full DMA every step

**Source**: `vllm/v1/worker/gpu_input_batch.py:174`, `vllm/v1/worker/block_table.py:24`

MRV1 holds a CPU-pinned block table of shape `[max_num_reqs, max_num_blocks_per_req]`. After adding all requests:

```
block_table.np  [max_num_reqs=4, max_blocks=4]  (CPU pinned, int32)

slot │ blk0  blk1  blk2  blk3
─────┼────────────────────────
  0  │    5     9     0     0   ← Req A
  1  │    2     0     0     0   ← Req B
  2  │   12     0     0     0   ← Req C
  3  │    0     0     0     0   ← empty
```

Every step `commit_block_table(num_reqs=3)` issues a full DMA:

```
DMA (CPU pinned → GPU):   shape=[3, 4]  int32  = 48 bytes
→ block_table_gpu = [[5,9,0,0], [2,0,0,0], [12,0,0,0]]
```

This is unconditional — even when A and B had no new blocks, their unchanged rows (including trailing zeros) cross PCIe every step.

Now suppose Req B fills block 2 and needs a new page (block 7). The scheduler sends `new_block_ids=[7]` for B. MRV1 updates the CPU buffer and then on the next step the full table is DMA'd again:

```
block_table.np[1, 1] = 7   (CPU write)

Next step DMA:   shape=[3, 4]  = 48 bytes
→ block_table_gpu = [[5,9,0,0], [2,7,0,0], [12,0,0,0]]
```

### MRV2 — `StagedWriteTensor` + Triton scatter

**Source**: `vllm/v1/worker/gpu/block_table.py:17`, `vllm/v1/worker/gpu/buffer_utils.py:114`

MRV2 uses a `StagedWriteTensor` of shape `[max_num_reqs, max_num_blocks]` that lives on GPU. No bulk DMA ever occurs.

When Req C is added with `block_ids=([12],)`:

```python
# append_block_ids(req_index=2, new_block_ids=([12],), overwrite=True)
start = 0   # overwrite: start from column 0
self.block_tables[0].stage_write(index=2, start=0, x=[12])

# Staged internally:
_staged_write_indices  = [2]
_staged_write_starts   = [0]
_staged_write_contents = [12]
_staged_write_cu_lens  = [1]
```

When Req B gets a new page (block 7):

```python
# append_block_ids(req_index=1, new_block_ids=([7],), overwrite=False)
start = num_blocks.np[0, 1] = 1   # Req B already has 1 block
self.block_tables[0].stage_write(index=1, start=1, x=[7])

# Now staged:
_staged_write_indices  = [2, 1]
_staged_write_starts   = [0, 1]
_staged_write_contents = [12, 7]
_staged_write_cu_lens  = [1, 2]
```

`apply_staged_writes()` launches one Triton kernel:

```
Triton _apply_write_kernel inputs:
  gpu tensor (block table on GPU):         shape=[4, max_blocks]
  indices_uva (which rows, via UVA):       [2, 1]
  starts_uva  (start cols, via UVA):       [0, 1]
  write_contents (new block IDs, on GPU):  [12, 7]
  cu_lens_uva (cumulative lengths):        [1, 2]

After kernel — GPU block table:
slot │ blk0  blk1
─────┼────────────
  0  │    5     9   ← Req A: NO writes, GPU retains prior value
  1  │    2     7   ← Req B: col 1 written with 7
  2  │   12     0   ← Req C: col 0 written with 12
  3  │    0     0
```

**Key difference**: only the 2 new block IDs (12, 7) crossed PCIe as `write_contents`. Req A's row was never touched. For hybrid models with multiple KV cache groups, `FusedStagedWriter` batches all groups into a **single Triton kernel launch** instead of N sequential ones.

---

## 5. Sampled Token Output: Synchronous vs Async D2H

### MRV1 — synchronous GPU→CPU copy

After the sampler runs on GPU and produces `sampled_token_ids_gpu`:

```
sampled_token_ids_gpu (GPU, int32) = [707, 555, 161, 0]
                                      ^A    ^B    ^C   ^empty
```

MRV1's `sample_tokens()` calls a synchronous `.cpu()` copy:

```python
sampled_token_ids_cpu = sampled_token_ids_gpu.cpu()   # CPU stalls here
```

Timeline:

```
step N:
  GPU:  ──[forward pass]──────────[sampler]──[D2H copy]──
  CPU:  ────────────────────────────────────────────────── [update_from_output] ── [schedule N+1]
                                               ↑ CPU blocked here until copy done
```

### MRV2 — async D2H on a dedicated CUDA stream

**Source**: `vllm/v1/worker/gpu/async_utils.py:12`, `vllm/v1/worker/gpu/model_runner.py:1445`

MRV2's `sample_tokens()` returns an `AsyncOutput` immediately, having fired the D2H copy on a side stream:

```python
# AsyncOutput.__init__:
with stream(copy_stream, main_stream):
    copy_stream.wait_stream(main_stream)          # don't start until sampler done
    self.sampled_token_ids = async_copy_to_np(    # non-blocking D2H
        sampler_output.sampled_token_ids
    )
    self.copy_event.record(copy_stream)           # mark completion
# Returns immediately — copy still in flight
```

The caller (`EngineCore.step()`) starts the **next scheduling cycle** right away:

```
step N:
  Main GPU stream:  ──[forward N]──────[sampler N]──────────────────────────────────
  Copy stream:                                   ──[D2H: sampled_tokens N]──────────
  CPU:              ──────────────────────────────[schedule N+1]──[H2D inputs N+1]──

step N+1:
  Main GPU stream:  ───────────────────────────────────[forward N+1]──[sampler N+1]─
```

When the output for step N is finally needed (`update_from_output`), only a single sync check is needed:

```python
def get_output(self) -> ModelRunnerOutput:
    self.copy_event.synchronize()   # no-op if copy already done
    ...
```

The actual tensor content crossing PCIe is identical in both runners:

```
GPU → CPU (both MRV1 and MRV2):
sampled_token_ids: [707, 555, 161]   shape=(3,)  int32
                    ^A    ^B    ^C
```

The difference is purely **when** the CPU blocks: MRV2 hides the PCIe latency under useful compute.

---

## 6. Request Removal / Finish: `condense()` vs `free_indices`

This is one of the most architecturally distinct differences between the two runners.

### Setup for removal example

After the decode step above, Req B finishes (it hit its max_tokens limit). Its sampled token was 555.
The scheduler sends `finished_req_ids = {"B"}` in the next `SchedulerOutput`.

State before removal:

```
Slot occupancy:
slot 0 → Req A  (active)
slot 1 → Req B  (FINISHED — to be removed)
slot 2 → Req C  (active)
slot 3 → empty
```

---

### MRV1 — mark + `condense()` (slot compaction)

**Source**: `vllm/v1/worker/gpu_input_batch.py:513` (remove), `gpu_input_batch.py:686` (condense)

**Step 1 — mark as removed**

`_update_states` calls `input_batch.remove_request("B")`:

```python
# remove_request("B"):
req_index = self.req_id_to_index.pop("B")   # → 1
self.batch_update_builder.removed_append(1) # mark slot 1 as empty
self._req_ids[1] = None
self.block_table.clear_row(1)               # zero out block_table.np[1, :]
self.greedy_reqs.discard("B")
self.top_p_reqs.discard("B")
# ... clear all sampling params for slot 1
```

State after mark:

```
slot 0 → Req A  (active)
slot 1 → None   (hole — marked removed)
slot 2 → Req C  (active)
slot 3 → empty
```

```
token_ids_cpu after removal:
slot │ tok0  tok1  tok2  tok3  tok4  tok5  tok6  tok7
─────┼────────────────────────────────────────────────
  0  │  101   202   303   404   505   606     0     0   ← A (untouched)
  1  │  111   222   333   444     0     0     0     0   ← B (stale, will be overwritten)
  2  │  151   252   353     0     0     0     0     0   ← C (untouched)
  3  │    0 ...                                         ← empty
```

**Step 2 — `condense()`**

`condense()` compacts the active requests toward slot 0, filling the hole left by B. It finds the highest active slot (slot 2 = Req C) and moves it into the hole (slot 1):

```python
# condense():
empty_index = 1       # lowest hole
last_req_index = 2    # highest active slot (Req C)

# Move Req C from slot 2 to slot 1:
self._req_ids[1] = "C";   self._req_ids[2] = None
self.req_id_to_index["C"] = 1

# Copy token_ids_cpu row 2 → row 1 (only active prefix, 3 tokens):
self.token_ids_cpu[1, :3] = self.token_ids_cpu[2, :3]
# → token_ids_cpu[1] = [151, 252, 353, 0, 0, 0, 0, 0]

# Copy all metadata:
self.num_computed_tokens_cpu[1] = self.num_computed_tokens_cpu[2]
self.temperature_cpu[1] = self.temperature_cpu[2]
# ... all other per-slot CPU arrays copied ...
self.block_table.move_row(2, 1)   # copy block IDs row 2 → row 1
```

State after condense:

```
slot 0 → Req A  (active, slot unchanged)
slot 1 → Req C  (moved from slot 2)
slot 2 → empty  (was Req C)
slot 3 → empty

token_ids_cpu:
slot │ tok0  tok1  tok2  tok3  tok4  tok5  tok6  tok7
─────┼────────────────────────────────────────────────
  0  │  101   202   303   404   505   606     0     0   ← A (unchanged)
  1  │  151   252   353     0     0     0     0     0   ← C (moved here)
  2  │  151   252   353     0     0     0     0     0   ← stale copy (ignored, num_reqs=2)
  3  │    0 ...

block_table.np:
slot │ blk0  blk1
─────┼────────────
  0  │    5     9   ← A unchanged
  1  │   12     0   ← C moved from slot 2
  2  │   12     0   ← stale (ignored)
  3  │    0     0
```

`num_reqs` is decremented to 2. On the next DMA, only rows `[0:2]` are sent to GPU.

**Why condense?** MRV1's `token_ids_cpu` array is indexed by slot. The `torch.index_select` gather uses `req_indices * max_model_len + positions` as flat offsets. If there are holes in the slot layout (slot 1 empty, slot 2 used), the flat index math breaks — a request at slot 2 with `max_model_len=8` is at offset 16, but the gather kernel only processes `num_reqs=2` rows (slots 0 and 1). So the array must be kept compact. `condense()` is a mandatory O(num_removed) CPU memcpy operation every time requests finish.

---

### MRV2 — `free_indices` pool, no compaction

**Source**: `vllm/v1/worker/gpu/states.py:122` (remove), `gpu/model_runner.py:741` (_remove_request)

MRV2 uses a simple **free-list** (a Python `list` acting as a stack) to manage slot assignment. Slots are never compacted.

**Step 1 — remove and return slot to free list**

`_remove_request("B")` calls `req_states.remove_request("B")`:

```python
# states.py:remove_request:
req_idx = self.req_id_to_index.pop("B")   # → 1
self.index_to_req_id.pop(1, None)
self.free_indices.append(1)               # slot 1 returned to free pool
# Done — no data movement at all
```

```
free_indices before removal: [3]          (slot 3 was empty)
free_indices after removal:  [3, 1]       (slot 1 now also free)

req_id_to_index: {"A": 0, "C": 2}        (B removed, C stays at slot 2)
```

State after removal:

```
slot 0 → Req A  (active)
slot 1 → free   (returned to pool — no data cleared)
slot 2 → Req C  (active, STAYS at slot 2)
slot 3 → free

all_token_ids (UVA, unchanged):
slot │ tok0  tok1  tok2  tok3  tok4  tok5  tok6  tok7
─────┼────────────────────────────────────────────────
  0  │  101   202   303   404   505   606     0     0   ← A (unchanged)
  1  │  111   222   333   444     0     0     0     0   ← stale B data (ignored)
  2  │  151   252   353     0     0     0     0     0   ← C (unchanged, stays at slot 2)
  3  │    0 ...

block_table (GPU StagedWriteTensor, unchanged):
slot │ blk0  blk1
─────┼────────────
  0  │    5     9   ← A
  1  │    2     0   ← stale B (ignored — slot 1 not in req_id_to_index)
  2  │   12     0   ← C (stays at slot 2)
  3  │    0     0
```

**No data is moved.** The stale data in slot 1 is simply ignored because `req_id_to_index` no longer maps any request to slot 1.

**Step 2 — next request reuses slot**

Suppose a new request D arrives. `add_request("D", ...)` pops from `free_indices`:

```python
req_idx = self.free_indices.pop()   # → 1  (last-in, LIFO)
self.req_id_to_index["D"] = 1

# Stage writes to slot 1:
self.all_token_ids.stage_write(1, 0, D_token_ids)
self.block_tables.append_block_ids(1, D_block_ids, overwrite=True)
```

The Triton staged-write kernel then overwrites slot 1's stale data with D's tokens. No memcpy, no condense loop.

**How prepare_inputs handles non-compact slots**

MRV2's `prepare_inputs` builds an explicit `idx_mapping` array — a mapping from batch position to slot index — rather than assuming slots are compact:

```python
# prepare_inputs:
req_ids = sort_batch_req_ids(...)         # e.g. ["A", "C"]  (batch order)
idx_mapping_np = [req_states.req_id_to_index[r] for r in req_ids]
               = [0, 2]                   # A→slot 0, C→slot 2 (non-contiguous!)

idx_mapping = async_copy_to_gpu(idx_mapping_np, device=device)
# → sent to GPU as [0, 2]
```

All Triton kernels in MRV2 (`prepare_prefill_inputs`, `combine_sampled_and_draft_tokens`, `prepare_pos_seq_lens`) accept `idx_mapping` as an explicit indirection array. They use it to index into the slot-indexed GPU arrays (`all_token_ids`, `num_computed_tokens`, etc.) without requiring the slots to be contiguous.

```
Triton gather:
  for batch_pos=0: slot = idx_mapping[0] = 0  → read all_token_ids[0, ...]  (Req A)
  for batch_pos=1: slot = idx_mapping[1] = 2  → read all_token_ids[2, ...]  (Req C)
```

This indirection is the key that makes no-compaction possible. MRV1 can't do this because its `token_ids_cpu` gather uses flat arithmetic that requires compact slots.

---

## 7. Other Differences

### 7a — CUDA Graph Management

**MRV1**: Graph sizes are fixed multiples of `uniform_decode_query_len` computed in `CompilationConfig`. Logic is spread across the runner and compiler config.

```
# compilation.py:1425:
# MRV1 adjusts cudagraph sizes to be a multiple of uniform_decode_query_len
```

**MRV2**: Uses `ModelCudaGraphManager` (`gpu/cudagraph_utils.py:413`) with `BatchExecutionDescriptor`. Graphs are keyed by `(num_tokens, uniform_token_count)`, enabling:

- Better reuse across batch compositions.
- Separate graph capture for the speculator.
- Piecewise graph mode (`PIECEWISE`) where attention runs in eager mode and FFN is graphed.

### 7b — Model-Specific State: Monolithic vs Pluggable

**MRV1**: Mamba SSM state, encoder cache for multimodal, etc., are handled by scattered `if mamba:` / `if multimodal:` branches inside the giant `GPUModelRunner` class.

**MRV2**: Abstract `ModelState` class (`gpu/model_states/interface.py:39`) with lifecycle hooks:

```python
class ModelState(ABC):
    def add_request(self, req_index, new_req_data): ...
    def remove_request(self, req_id): ...
    def apply_staged_writes(self): ...
    def preprocess_state(self, ...): ...
    def postprocess_state(self, ...): ...
    def prepare_inputs(self, ...): ...
    def prepare_attn(self, ...): ...
```

Concrete subclasses in `gpu/model_states/`:

- `DefaultModelState` — standard transformer (text + multimodal)
- `MambaHybridModelState` — Mamba SSM state
- `EncoderDecoderModelState` — cross-attention
- `MMPruningModelState` — multimodal token pruning

New model types add a subclass without modifying the runner.

### 7c — Scheduler ↔ Runner Contract

**MRV1**: The scheduler sends the full `all_token_ids` (complete token history) in `CachedRequestData` for requests not scheduled in the previous step (e.g., after preemption), because MRV1's CPU table may have been overwritten during condense. Marked "MRV1-only" in `sched/output.py:123`.

**MRV2**: Only sends diffs (`new_block_ids`, `num_computed_tokens`). The GPU `all_token_ids` array retains its data persistently across steps — no full retransmission needed after preemption.

### 7d — Per-Step CPU Work Comparison

```
MRV1 per step (3 reqs, 5 tokens):
  CPU:
    - torch.index_select on token_ids_cpu_tensor.flatten()   O(T) gather
    - compute token_indices (positions + req_indices * M)     O(T)
    - update block_table.np cells for new pages              O(new_blocks)
    - condense() if any req finished                         O(num_removed × max_tokens)
  PCIe:
    - block_table DMA:    [3, 4] int32 = 48 bytes
    - input_ids DMA:      [5]  int32   = 20 bytes
    - positions DMA:      [5]  int32   = 20 bytes
    - query_start_loc:    [4]  int32   = 16 bytes
    - sampled_token_ids (D2H, blocking): [3] int32 = 12 bytes

MRV2 per step (same batch):
  CPU:
    - stage block ID diffs (list appends)                    O(new_blocks)
    - compute idx_mapping_np                                 O(num_reqs)
    - compute query_start_loc_np                             O(num_reqs)
    - NO per-token work; NO gather; NO condense
  PCIe:
    - staged block diffs (write_contents):  [2] int32 = 8 bytes
    - staged write metadata (UVA):          ~24 bytes via UVA pointer access
    - idx_mapping:         [2] int32        = 8 bytes
    - query_start_loc:     [3] int32        = 12 bytes
    - sampled_token_ids (D2H, async):       [3] int32 = 12 bytes (overlapped)
  GPU Triton kernels (zero PCIe):
    - staged write to all_token_ids + block_table
    - prefill gather from all_token_ids.gpu
    - decode tokens from last_sampled_tokens.gpu
    - pos/seq_len computation on GPU
```

The savings compound at scale: with 256 requests and 128K context length, MRV1's `token_ids_cpu` is 256 × 128K × 4 bytes = **128 MB** of CPU memory. The block table DMA at 256 reqs × (128K/16) blocks × 4 bytes = **8 MB per step**. MRV2 replaces both with Triton kernels operating on GPU-resident data, with PCIe traffic proportional only to what actually changed.

---

## 8. MRV2 on ROCm

### Does ROCm Support UVA?

**Yes.** AMD GPUs have supported Unified Virtual Addressing since the GCN architecture. The HIP runtime provides `hipHostGetDevicePointer` and `hipHostAlloc(hipHostMallocMapped)`, which are the exact HIP equivalents of the CUDA APIs used in vLLM's UVA implementation.

The key implementation is in `csrc/libtorch_stable/cuda_view.cu`:

```c
torch::stable::Tensor get_cuda_view_from_cpu_tensor(cpu_tensor) {
    // For pinned tensors:
    cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    // For non-pinned tensors:
    cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
}
```

This file is compiled for **both** CUDA and HIP builds (`CMakeLists.txt:384`):

```cmake
if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  set(VLLM_STABLE_EXT_SRC
    ...
    "csrc/libtorch_stable/cuda_view.cu"   ← compiled for both
```

On a ROCm/HIP build, `<cuda_runtime.h>` is provided by HIP's compatibility layer, which maps all `cuda*` symbols to their `hip*` equivalents at the preprocessor level:

- `cudaHostGetDevicePointer` → `hipHostGetDevicePointer`
- `cudaHostAlloc` → `hipHostAlloc`

The Python dispatch path correctly reaches this for ROCm:

```python
# torch_utils.py:766
def get_accelerator_view_from_cpu_tensor(cpu_tensor):
    if current_platform.is_cuda_alike():   # True for both CUDA AND ROCm
        return torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)
```

And `is_uva_available()` returns `True` for ROCm:

```python
# platform_utils.py:51
def is_uva_available() -> bool:
    return is_pin_memory_available() or current_platform.is_cpu()
    # ROCm doesn't override is_pin_memory_available(), inherits default True
```

### MRV2 Component Status on ROCm

| MRV2 Component | Mechanism | ROCm Status |
|---|---|---|
| `UvaBuffer` (pinned CPU ↔ GPU) | `cudaHostAlloc + cudaHostGetDevicePointer` → HIP compat | **Works** |
| `UvaBackedTensor` (`all_token_ids`, sampling params) | Uses `UvaBuffer` | **Works** |
| `StagedWriteTensor` (block table diffs) | `UvaBufferPool` + Triton scatter kernel | **Works** |
| `FusedStagedWriter` (multi-group) | Triton kernel, backend-agnostic | **Works** |
| `async_copy_to_gpu` (small metadata) | `pin_memory() + copy_(non_blocking=True)` → `hipMemcpyAsync` | **Works** |
| `AsyncOutput` (D2H side stream) | `torch.cuda.Stream / Event` → HIP stream/event | **Works** |
| Free-list slot management | Pure Python, no GPU calls | **Works** |
| Cross-layer KV cache offloading | KV connector feature | **TODO** (tracked in PR #45947) |

### Step-by-Step: Decode Step on ROCm

**Staged write flush**: `apply_staged_writes()` launches a Triton kernel. Triton generates valid AMDGPU ISA (GCN/CDNA) from the same Python kernel source via the HIP backend.

**Prefill gather**: `prepare_prefill_inputs` Triton kernel reads from `all_token_ids.gpu` — a UVA pointer derived from `hipHostGetDevicePointer`. AMDGPU compute units read the pinned CPU memory over the PCIe fabric on demand, the same as NVIDIA GPUs do with UVA.

**`idx_mapping` DMA**: `async_copy_to_gpu` pins the small numpy array and calls `copy_(non_blocking=True)`, which issues `hipMemcpyAsync` on ROCm.

**Forward pass**: Uses ROCm's Flash Attention backends (not NVIDIA's). MRV2 delegates to the same `FlashAttentionMetadataBuilder.build()` Python path — backend-agnostic.

**Async D2H output**: `torch.cuda.Stream` maps to `hipStream_t`, `torch.cuda.Event` maps to `hipEvent_t`. The side-stream D2H pattern works identically.

### The `DeviceType::CUDA` Note

`cuda_view.cu` hardcodes `DeviceType::CUDA`:

```c
const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);
```

This works correctly on ROCm because PyTorch's ROCm build **reuses** `DeviceType::CUDA` to represent HIP/ROCm devices — there is no separate `DeviceType::HIP` enum value in PyTorch. ROCm tensors created this way are valid HIP tensors.

### Known Limitation

There is one confirmed gap, noted directly in the test suite:

```python
# tests/v1/kv_connector/unit/test_offloading_connector.py:141
# TODO: Reintroduce latency test on ROCm once MRV2 supports cross
# layer KV Cache. See https://github.com/vllm-project/vllm/pull/45947
if current_platform.is_rocm():
    return
```

This is about **cross-layer KV cache offloading** (a `KVConnector` feature for offloading KV cache across pipeline stages or to CPU). It is a feature gap in the KV connector integration, not a UVA limitation.

---

## 9. Summary Table

| Dimension | MRV1 | MRV2 |
|---|---|---|
| **Token storage** | CPU 2D table `[max_reqs, max_model_len]`, non-pinned, not on GPU | `StagedWriteTensor` with UVA backing — GPU-visible pinned CPU memory, never bulk-uploaded |
| **`input_ids` construction** | CPU `torch.index_select` gather + PCIe DMA | Triton gather kernel from `all_token_ids.gpu` (no PCIe) |
| **Block table update** | Full DMA of `[max_reqs, max_blocks]` every step | Triton scatter of only changed cells via `StagedWriteTensor` |
| **Sampled token D2H** | Synchronous, blocks next step | Async on `output_copy_stream`, overlaps next step's compute |
| **Request removal** | Mark + `condense()` — O(N) CPU memcpy compaction required | Return slot to `free_indices` — O(1), no data movement |
| **Slot layout invariant** | Must be compact (slots 0..N-1), condense enforces this | Arbitrary holes allowed; `idx_mapping` tensor handles indirection |
| **CUDA graph sizing** | Fixed multiples of `uniform_decode_query_len` in config | `ModelCudaGraphManager` with `BatchExecutionDescriptor`, more flexible |
| **Model-specific state** | `if mamba:` / `if multimodal:` branches in monolithic runner | Pluggable `ModelState` subclasses (ABC) |
| **Scheduler contract** | Sends full `all_token_ids` for non-scheduled reqs | Sends only diffs; GPU array retains data persistently |
| **UVA support** | Not used | Core mechanism; works on CUDA and ROCm |
| **ROCm status** | Production | Works via HIP compat; cross-layer KV offload TODO |
| **Status** | Production default | Opt-in (`VLLM_USE_V2_MODEL_RUNNER=1`), planned replacement |
