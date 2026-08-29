# vLLM CUDA/HIP Stream Architecture

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [Main Compute Stream](#2-main-compute-stream)
3. [The Universal Sync Pattern: `execute_in_parallel`](#3-the-universal-sync-pattern-execute_in_parallel)
4. [Global Auxiliary Stream](#4-global-auxiliary-stream)
5. [Model-Level Aux Streams: NVIDIA vs ROCm](#5-model-level-aux-streams-nvidia-vs-rocm)
   - 5.1 [DeepSeek V4 — 3 streams (NVIDIA) / 0 (ROCm)](#51-deepseek-v4--3-streams-nvidia--0-rocm)
   - 5.2 [Kimi K3 — 1 stream (NVIDIA) / 0 (ROCm)](#52-kimi-k3--1-stream-nvidia--0-rocm)
6. [Worker-Level Aux Streams](#6-worker-level-aux-streams)
   - 6.1 [Async D2H Token Copy Streams](#61-async-d2h-token-copy-streams)
   - 6.2 [DBO Compute-Comm Overlap (`comm_stream`)](#62-dbo-compute-comm-overlap-comm_stream)
   - 6.3 [Pipeline Parallel Broadcast (`broadcast_stream`)](#63-pipeline-parallel-broadcast-broadcast_stream)
   - 6.4 [KV Offload Streams (low priority)](#64-kv-offload-streams-low-priority)
7. [Distributed & Infrastructure Streams](#7-distributed--infrastructure-streams)
8. [LoRA Auxiliary Stream](#8-lora-auxiliary-stream)
9. [CUDA Graph Modes: Comparison and Trade-offs](#9-cuda-graph-modes-comparison-and-trade-offs)
   - 9.1 [Mode Overview](#91-mode-overview)
   - 9.2 [Eager (NONE)](#92-eager-none)
   - 9.3 [Full CUDA Graph (FULL / FULL_DECODE_ONLY)](#93-full-cuda-graph-full--full_decode_only)
   - 9.4 [Piecewise torch.compile (PIECEWISE / FULL_AND_PIECEWISE)](#94-piecewise-torchcompile-piecewise--full_and_piecewise)
   - 9.5 [Breakable CUDA Graph](#95-breakable-cuda-graph)
   - 9.6 [Which Mode to Choose](#96-which-mode-to-choose)
10. [Aux Stream Behavior Across Execution Modes](#10-aux-stream-behavior-across-execution-modes)
11. [Platform Summary](#11-platform-summary)

---

## 1. Mental Model

All streams in vLLM are `torch.cuda.Stream` objects. There are two conceptual roles:

1. **Main compute stream** — returned by `current_stream()`; all forward-pass ops land here by default.
2. **Aux streams** — side streams for specific parallel or async work, always synchronized back to the main stream before their results are consumed.

"Worker-level" and "model-level" are descriptions of who *owns* a stream, not a difference in mechanism. Both are aux streams that follow the same fan-out/join pattern described in [§3](#3-the-universal-sync-pattern-execute_in_parallel).

---

## 2. Main Compute Stream

**File:** [vllm/utils/torch_utils.py:779](../vllm/utils/torch_utils.py#L779)

One stream per process, created lazily the first time `current_stream()` is called. On both CUDA and ROCm, a **new** dedicated stream is created instead of using PyTorch's default stream 0:

```python
torch.cuda.set_stream(torch.cuda.Stream())
```

Two reasons:
- **ROCm:** default stream 0 combined with RCCL hurts collective performance.
- **CUDA:** `capture_begin()` rejects the default stream — CUDA graph capture requires a non-default stream.

`current_stream()` also patches `torch.cuda.set_stream` to cache the stream in a thread-local, avoiding the cost of `torch.cuda.current_stream()`, which allocates a new Python object on every call.

---

## 3. The Universal Sync Pattern: `execute_in_parallel`

**File:** [vllm/utils/multi_stream_utils.py](../vllm/utils/multi_stream_utils.py)

Every multi-stream interaction in vLLM uses the same CUDA event fan-out/join pattern. `execute_in_parallel` (N streams) generalizes `maybe_execute_in_parallel` (2 streams):

```
main stream:  ── [start_event.record()] ── [default_fn()] ── [done[0].wait()] [done[1].wait()] [done[2].wait()] ──▶
                          │                                          ▲               ▲               ▲
aux stream 0:             └── [start_event.wait()] ── [fn0()] ── [done[0].record()]
aux stream 1:             └── [start_event.wait()] ── [fn1()] ── [done[1].record()]
aux stream 2:             └── [start_event.wait()] ── [fn2()] ── [done[2].record()]
```

`start_event` ensures aux streams don't start until the main stream reaches the fork point. `done[i].wait()` ensures the main stream doesn't continue until all aux work is complete.

**Two fallback conditions** collapse this to sequential execution on the main stream:
1. `aux_streams=None` — ROCm, or any path where the feature is disabled.
2. `BreakableCUDAGraphCapture.is_active()` — during breakable graph capture; explained in [§9.5](#95-breakable-cuda-graph) and [§10](#10-aux-stream-behavior-across-execution-modes).

---

## 4. Global Auxiliary Stream

**File:** [vllm/utils/torch_utils.py:828](../vllm/utils/torch_utils.py#L828)

One shared aux stream per process on any `cuda_alike` platform (CUDA + ROCm):

```python
_aux_stream: torch.cuda.Stream | None = None
```

The source comment explains the design choice: a **single** global stream deliberately avoids an explosion of per-layer streams and keeps profiling traces readable. Current uses:
- MoE shared-expert overlap with the router.
- Kimi K3 MLA `_down_proj` overlap (latent MoE path).

---

## 5. Model-Level Aux Streams: NVIDIA vs ROCm

These streams are created once at model `__init__` and threaded into every decoder layer. They are long-lived and reused across all forward passes.

### 5.1 DeepSeek V4 — 3 streams (NVIDIA) / 0 (ROCm)

**NVIDIA:** [vllm/models/deepseek_v4/nvidia/model.py:1023](../vllm/models/deepseek_v4/nvidia/model.py#L1023)
**ROCm:** [vllm/models/deepseek_v4/amd/model.py:548](../vllm/models/deepseek_v4/amd/model.py#L548)

```python
# NVIDIA — 3 aux streams, one per parallel input projection
aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

# ROCm — disabled due to hang issues
aux_stream_list = None if current_platform.is_rocm() else [torch.cuda.Stream() for _ in range(3)]
```

Each stream maps to one non-default input projection GEMM in `_run_parallel_input_projections`:

```
# One decoder layer, NVIDIA:

main stream:   ── [start_event.record()] ── [fused_wqa_wkv GEMM] ── [done[0].wait()] [done[1].wait()] [done[2].wait()] ──▶
                          │                                                ▲                ▲                ▲
aux_stream[0]:            └── [start_event.wait()] ── [compressor kv_score GEMM]       [done[0].record()]
aux_stream[1]:            └── [start_event.wait()] ── [indexer.weights_proj GEMM]      [done[1].record()]
aux_stream[2]:            └── [start_event.wait()] ── [indexer.compressor kv_score]    [done[2].record()]
```

The `fused_wqa_wkv` GEMM stays on the main stream. The same `aux_stream_list` is shared across all decoder layers — no new streams are created per layer.

On ROCm, `aux_stream_list=None` causes `execute_in_parallel` to fall back to serial:

```
# ROCm — same 4 GEMMs, sequential:

main stream:  ── [compressor kv_score] ── [weights_proj] ── [indexer.compressor] ── [fused_wqa_wkv] ──▶
```

Correctness is unchanged; only the overlap benefit is lost. Same pattern applies to MTP layers: [nvidia/mtp.py:207](../vllm/models/deepseek_v4/nvidia/mtp.py#L207) / [amd/mtp.py:187](../vllm/models/deepseek_v4/amd/mtp.py#L187).

### 5.2 Kimi K3 — 1 stream (NVIDIA) / 0 (ROCm)

**NVIDIA:** [vllm/models/kimi_k3/nvidia/model.py:1132](../vllm/models/kimi_k3/nvidia/model.py#L1132)

```python
# One aux stream created at model level, shared across all decoder layers
aux_stream = torch.cuda.Stream()
```

Used via `maybe_execute_in_parallel` (the 2-stream variant) to overlap the MLA `g_proj` output-gate GEMM with the attention front-end:

```
# One decoder layer, NVIDIA:

main stream:  ── [event0.record()] ── [attention front-end (fn0)] ── [event1.wait()] ──▶
                        │                                                   ▲
aux_stream:             └── [event0.wait()] ── [g_proj output-gate GEMM (fn1)] ── [event1.record()]
```

There is no ROCm Kimi K3 model that creates this stream. The AMD path runs both ops serially on the main stream.

---

## 6. Worker-Level Aux Streams

These streams are owned by the inference worker or runner. They handle async I/O between the GPU compute timeline and other destinations (host memory, network, other GPUs).

### 6.1 Async D2H Token Copy Streams

The most common pattern: the GPU writes sampled tokens to a GPU buffer, then a copy stream transfers them to CPU while the **next** forward pass is already running on the main stream.

```
forward pass N:     ── [decode] ── [sample → gpu_buf] ──────────────────────────────────▶ forward pass N+1 ...
output_copy_stream:                     └── [gpu_buf → cpu_buf (D2H)] ──▶ (host reads cpu_buf)
```

| Stream | File | Created when |
|---|---|---|
| `output_copy_stream` | [v1/worker/gpu/model_runner.py:189](../vllm/v1/worker/gpu/model_runner.py#L189) | Always |
| `async_output_copy_stream` | [v1/worker/gpu_model_runner.py:785](../vllm/v1/worker/gpu_model_runner.py#L785) | `use_async_scheduling` only |
| `draft_token_ids_copy_stream` | [v1/worker/gpu_model_runner.py:965](../vllm/v1/worker/gpu_model_runner.py#L965) | Spec decode only |
| `_copy_stream` (adaptive verify) | [v1/worker/gpu/spec_decode/adaptive_verification.py:125](../vllm/v1/worker/gpu/spec_decode/adaptive_verification.py#L125) | Spec decode only |
| `copy_stream` (structured outputs) | [v1/worker/gpu/structured_outputs.py:28](../vllm/v1/worker/gpu/structured_outputs.py#L28) | Structured outputs only |

### 6.2 DBO Compute-Comm Overlap (`comm_stream`)

**File:** [vllm/v1/worker/gpu_ubatch_wrapper.py:124](../vllm/v1/worker/gpu_ubatch_wrapper.py#L124) and [vllm/v1/worker/ubatching.py:28](../vllm/v1/worker/ubatching.py#L28)

Each micro-batch context (`UBatchContext`) carries a `compute_stream` + `comm_stream` pair. The same fan-out/join pattern applies, but at the batch level rather than the GEMM level:

```
compute_stream:  ── [micro-batch A compute] ── [event_a.record()] ──────────────────── [event_b.wait()] ── [micro-batch B compute] ──▶
comm_stream:                                        └── [event_a.wait()] ── [all-reduce] ── [event_b.record()]
```

### 6.3 Pipeline Parallel Broadcast (`broadcast_stream`)

**File:** [vllm/v1/worker/gpu/pp_utils.py:70](../vllm/v1/worker/gpu/pp_utils.py#L70)

One stream per `PPHandler`. Overlaps pipeline-parallel P2P send/recv with the main compute stream so stage N+1 begins receiving activations while stage N is still computing.

### 6.4 KV Offload Streams (low priority)

**File:** [vllm/v1/simple_kv_offload/worker.py:160](../vllm/v1/simple_kv_offload/worker.py#L160)

```python
low_pri, _ = torch.cuda.Stream.priority_range()
self.load_stream = torch.cuda.Stream(priority=low_pri)
self.store_stream = torch.cuda.Stream(priority=low_pri)
```

Low priority means the GPU scheduler yields these streams to the main compute stream automatically — no explicit synchronization is needed to prevent KV swap transfers from blocking forward passes.

---

## 7. Distributed & Infrastructure Streams

| Stream | File | Purpose |
|---|---|---|
| CUDA graph capture stream | [distributed/parallel_state.py:621](../vllm/distributed/parallel_state.py#L621) | Dedicated stream for graph capture in TP/PP groups |
| EPLB async thread stream | [distributed/eplb/async_worker.py:35](../vllm/distributed/eplb/async_worker.py#L35) | One per background expert-rebalancing thread |
| HF3FS `_save_stream` / `_load_stream` | [distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:134](../vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py#L134) | Async KV cache save/load to HF3FS |
| Weight transfer buffer streams | [distributed/weight_transfer/packed_tensor.py:157](../vllm/distributed/weight_transfer/packed_tensor.py#L157) | One per double/triple buffer slot for async weight broadcast (EPLB) |
| KV offload stream pool | [v1/kv_offload/cpu/gpu_worker.py:333](../vllm/v1/kv_offload/cpu/gpu_worker.py#L333) | Dynamic pool; one stream reused per active transfer |

---

## 8. LoRA Auxiliary Stream

**File:** [vllm/lora/layers/utils.py:15](../vllm/lora/layers/utils.py#L15)

One global stream for LoRA layer operations on any `cuda_alike` platform (CUDA + ROCm). Follows the same lazy-init singleton pattern as the [global `aux_stream`](#4-global-auxiliary-stream).

---

## 9. CUDA Graph Modes: Comparison and Trade-offs

### 9.1 Mode Overview

vLLM exposes five `CUDAGraphMode` values, defined in [vllm/config/compilation.py:53](../vllm/config/compilation.py#L53):

| Mode | Meaning |
|---|---|
| `NONE` | No CUDA graph capture; fully eager execution |
| `FULL` | One monolithic graph captures the entire forward pass |
| `FULL_DECODE_ONLY` | `FULL` for decode batches, eager for prefill/mixed |
| `PIECEWISE` | Graph captures per-layer segments; attention ops run eagerly outside each segment |
| `FULL_AND_PIECEWISE` | `FULL` for decode, `PIECEWISE` for prefill/mixed — **v1 default** |

There is also a separate path enabled by `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, called **Breakable CUDA Graph**, which replaces `PIECEWISE` for model classes that do not carry `@support_torch_compile`. It is auto-enabled for DSv4, Kimi K3, Inkling, MiniMaxM3, and others ([vllm/config/vllm.py:1313](../vllm/config/vllm.py#L1313)).

---

### 9.2 Eager (NONE)

**How it works:** Every forward pass runs as plain PyTorch ops. No capture, no replay.

**Pros:**
- Simplest to debug — Python execution, normal stack traces.
- No warm-up penalty.
- No static-shape constraint — variable-length inputs work natively.
- Multi-stream GEMM overlap is fully active (DSv4, Kimi K3).

**Cons:**
- CPU dispatch overhead on every op, every token — significant for decode where the batch is small.
- No kernel fusion from `torch.compile`.

**When to use:** Debugging, models incompatible with graph capture, or when flexibility matters more than peak decode throughput.

---

### 9.3 Full CUDA Graph (FULL / FULL_DECODE_ONLY)

**How it works:** The entire forward pass is captured into a single `CUDAGraph` with `torch.cuda.graph()`. On replay, the GPU reissues all kernels with near-zero CPU overhead.

```
Capture (once per batch shape):
  current_stream:  ── [embed] ── [layer 0 attn] ── [layer 0 ffn] ── ... ── [layer N ffn] ── [lm_head] ──▶
                                  (all 4 DSv4 streams recorded here if piecewise torch.compile is active)

Replay (every decode step):
  entry.cudagraph.replay()  ← single Python call, GPU does the rest
```

**Pros:**
- Lowest CPU overhead on decode — one Python call per forward pass.
- Best for small models or token-by-token decode with many small batches.

**Cons:**
- Requires **fixed batch shapes** — a separate graph must be captured per token count. `cudagraph_capture_sizes` lists the shapes to capture; requests are padded to the next captured size.
- Memory overhead: each captured graph holds its own activation buffers.
- Not all attention backends support it (custom kernels must be CUDA-graph-safe).
- **Capture time** on first encounter; `cudagraph_num_of_warmups` controls warmup runs before capture.

**`FULL_DECODE_ONLY`** runs `FULL` for decode batches (where CPU overhead matters most) and falls back to eager for prefill and mixed batches (where shape variability is high and the graph would rarely be reused).

---

### 9.4 Piecewise torch.compile (PIECEWISE / FULL_AND_PIECEWISE)

**How it works:** `torch.compile` with `CompilationMode.VLLM_COMPILE` traces the model. The graph is split at attention boundaries (marked `cudagraph_unsafe`), producing N+1 segments per layer. Each segment is wrapped in a `CUDAGraphWrapper` and captured separately.

```
Layer k (piecewise):
  [CUDAGraph segment k.0: pre-attn ops] → [attention (eager)] → [CUDAGraph segment k.1: post-attn ops]
```

Because `BreakableCUDAGraphCapture.is_active()` is never set in this path, **aux streams remain active during capture** and are recorded into each graph segment. The multi-stream GEMM overlap is preserved in replay (see [§10](#10-aux-stream-behavior-across-execution-modes) for the detailed example).

**Pros:**
- Kernel fusion from `torch.compile` (fused ops, constant folding, custom passes).
- Multi-stream GEMM overlap is **preserved in replay** — unlike breakable graphs.
- Attention ops can use dynamic shapes or non-graph-safe kernels without blocking the rest.
- `FULL_AND_PIECEWISE` (the v1 default) combines decode `FULL` graphs with `PIECEWISE` for prefill.

**Cons:**
- Requires `@support_torch_compile` on the model class — not all models support it.
- Longer startup: Dynamo tracing + Inductor compilation + per-shape graph capture.
- More complex debugging — FX graph transformations, piecewise wrappers.
- Incompatible with sequence parallelism (`fuse_attn_quant`, etc.).

---

### 9.5 Breakable CUDA Graph

**Enabled by:** `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (auto-set for DSv4, Kimi K3, Inkling, and others)
**File:** [vllm/compilation/breakable_cudagraph.py](../vllm/compilation/breakable_cudagraph.py)

**How it works:** A single capture context (`BreakableCUDAGraphCapture`) drives the entire forward pass. When the context encounters an attention op decorated with `@eager_break_during_capture`, it ends the current segment, runs the op eagerly, records the callable for replay, and starts a fresh segment. The result is an ordered list of segments — alternating `CUDAGraph.replay` callables and eager callables — replayed in order at inference time.

```
Capture (entire forward pass):

  [CUDAGraph seg 0: embed + layer 0 pre-attn] → [layer 0 attn: eager] → [CUDAGraph seg 1: layer 0 post-attn + layer 1 pre-attn] → ...

Replay:
  for segment in capture.segments:
      segment()   ← CUDAGraph.replay() or eager fn
```

Because `is_active()` is `True` throughout this capture, aux streams are forced to `None`. Each segment is a single-stream graph with no cross-stream dependencies.

**Pros:**
- Works for model classes without `@support_torch_compile` — no Dynamo tracing required.
- Faster startup than piecewise `torch.compile` (no Inductor compilation).
- Simpler implementation — no FX graph splitting.
- Still captures the bulk of work (non-attention ops) into CUDA graphs.

**Cons:**
- **Multi-stream GEMM overlap is lost** during capture and replay. DSv4's 3 parallel input projections run serially.
- No `torch.compile` kernel fusion or custom Inductor passes.
- Each segment is a separate `CUDAGraph`; cross-segment dependencies must go through static buffers.

---

### 9.6 Which Mode to Choose

```
Is the model class decorated with @support_torch_compile?
├── No  →  Breakable CUDA Graph is auto-enabled
│          (DSv4, Kimi K3, Inkling, MiniMaxM3, ...)
│          Override: VLLM_USE_BREAKABLE_CUDAGRAPH=0 to fall back to eager
│
└── Yes →  Use FULL_AND_PIECEWISE (v1 default)
           ├── Decode-heavy workload, small/fixed batch?  →  FULL or FULL_DECODE_ONLY
           ├── Prefill-heavy or variable shapes?           →  PIECEWISE
           ├── Debugging or dynamic shapes?                →  NONE (eager)
           └── Need sequence parallelism?                  →  NONE (PIECEWISE is incompatible)
```

Key trade-off summary:

| Mode | Startup cost | Decode CPU overhead | GEMM overlap | Flexibility |
|---|---|---|---|---|
| Eager (NONE) | None | High | Full | Full |
| FULL | Medium | **Lowest** | Preserved in graph | Fixed shapes only |
| PIECEWISE | High (compile + capture) | Low | **Preserved in graph** | Moderate |
| FULL_AND_PIECEWISE | High | **Lowest** for decode | Preserved | Moderate |
| Breakable | Low | Low | **Lost** (serial) | High (no compile needed) |

---

## 10. Aux Stream Behavior Across Execution Modes

Aux streams only produce overlap when GPU ops can run concurrently. Each execution mode handles them differently.

### Eager

`BreakableCUDAGraphCapture.is_active()` is `False`. Aux streams are fully active. All concurrent GEMM paths (DSv4, Kimi K3) execute in parallel on the GPU.

### Piecewise torch.compile

`CUDAGraphWrapper` uses `torch.cuda.graph(stream=current_stream())`. `is_active()` is **not** set by this path. Aux streams remain enabled during capture.

CUDA graph capture records ops on **all streams**, not just the origin stream, provided every aux stream is joined (via `done[i].wait()`) before `capture_end()`. The `for ev in pending: ev.wait()` at [multi_stream_utils.py:129](../vllm/utils/multi_stream_utils.py#L129) guarantees this.

**Example — DSv4 piecewise capture of one attention layer segment:**

```
Capture window (origin = current_stream):

current_stream:   ── [start_event.record()] ── [fused_wqa_wkv] ── [done[0].wait()] [done[1].wait()] [done[2].wait()] ──▶
                            │                                              ▲               ▲               ▲
aux_stream[0]:              └── [start_event.wait()] ── [compressor kv_score]  ── [done[0].record()]
aux_stream[1]:              └── [start_event.wait()] ── [weights_proj]         ── [done[1].record()]
aux_stream[2]:              └── [start_event.wait()] ── [indexer.compressor]   ── [done[2].record()]

◀────── ops on all 4 streams recorded into one CUDAGraph segment ──────▶
```

At replay, `entry.cudagraph.replay()` re-issues the exact same op sequence on all 4 streams. The parallelism is **preserved in replay** — stream-setup overhead is paid once at capture; subsequent replays run 4 GEMMs concurrently at near-zero CPU overhead.

### Breakable CUDA Graph

`is_active()` is `True` throughout capture. `multi_stream_utils.py:48-49` forces `aux_stream = None`. Every segment is a clean single-stream graph.

Why: `BreakableCUDAGraphCapture` calls `capture_begin()` / `capture_end()` many times per forward pass (once per segment). If aux streams were active, a `done[i].wait()` could reference ops from an aux stream that spans a segment boundary — CUDA would detect an unjoined stream at `capture_end()` and raise an error.

**Example — DSv4 breakable capture of the same attention layer:**

```
Segment M (one CUDAGraph, single stream):

current_stream:  ── [compressor kv_score] ── [weights_proj] ── [indexer.compressor] ── [fused_wqa_wkv] ──▶
                    (serial — aux_streams forced to None)

capture_end() ← no cross-stream dependencies, safe
```

At replay, `capture.replay()` iterates over all segments calling each `CUDAGraph.replay()` in order. The 4 GEMMs run serially. Correctness is identical to eager; the GEMM-level overlap benefit is lost.

### Summary Table

| Execution mode | `is_active()` | Aux streams | Parallel GEMMs |
|---|---|---|---|
| Eager | False | Enabled (NVIDIA) | Yes — concurrent on GPU |
| Piecewise `torch.compile` — capture | False | Enabled, recorded into graph | Yes — **concurrent, preserved in replay** |
| Piecewise `torch.compile` — replay | False | N/A — GPU replays captured ops | Yes — concurrent |
| Breakable CUDA graph — capture | **True** | Forced `None` | No — serial |
| Breakable CUDA graph — replay | False | N/A — no Python runs | No — captured serially |
| ROCm (any mode) | False | Forced `None` (hang issues) | No — serial |

---

## 11. Platform Summary

| Stream | NVIDIA | ROCm |
|---|---|---|
| Main compute stream | 1 (dedicated, not stream 0) | 1 (dedicated, not stream 0) |
| Global `aux_stream` | 1 | 1 |
| LoRA aux stream | 1 | 1 |
| Worker D2H / comm / offload streams | per feature | per feature (identical) |
| DeepSeek V4 attention aux streams | **3** | **0** (hang issues) |
| DeepSeek V4 MTP aux streams | **3** | **0** (hang issues) |
| Kimi K3 MLA aux stream | **1** | **0** (no ROCm model) |

The only platform divergence is in model-level GEMM-parallelism streams for DeepSeek V4 and Kimi K3. Everything else — including the reason the main compute stream avoids stream 0 — applies equally to both platforms.
