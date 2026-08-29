# vLLM CUDA Graphs and torch.compile: A Structural Analysis

This document covers how `torch.compile` and CUDA graphs interact in vLLM's v1 engine:
all graph modes, the compilation pipeline, runtime dispatch and buffer management,
backend support declarations, the auto-downgrade system, and the full configuration
reference. All code references point to the actual implementation.

---

## Table of Contents

1. [The Two Orthogonal Axes](#1-the-two-orthogonal-axes)
2. [The torch.compile Layer](#2-the-torchcompile-layer)
   - [2.1 `CompilationMode` enum](#21-compilationmode-enum)
   - [2.2 How `VLLM_COMPILE` is triggered](#22-how-vllm_compile-is-triggered)
   - [2.3 `VllmBackend` — FX Graph Splitting](#23-vllmbackend--fx-graph-splitting)
   - [2.4 `PiecewiseCompileInterpreter` — Compile and Wrap](#24-piecewisecompileinterpreter--compile-and-wrap)
3. [The CUDA Graph Layer](#3-the-cuda-graph-layer)
   - [3.1 `CUDAGraphMode` enum](#31-cudagraphmode-enum)
   - [3.2 `NONE` — No CUDA Graph](#32-none--no-cuda-graph)
   - [3.3 `PIECEWISE` — Piecewise CUDA Graphs](#33-piecewise--piecewise-cuda-graphs)
   - [3.4 `FULL` — Full Model CUDA Graph](#34-full--full-model-cuda-graph)
   - [3.5 `FULL_DECODE_ONLY` — Decode FULL, Mixed Eager](#35-full_decode_only--decode-full-mixed-eager)
   - [3.6 `FULL_AND_PIECEWISE` — Default](#36-full_and_piecewise--default)
4. [Backend Capability: `AttentionCGSupport`](#4-backend-capability-attentioncgsupport)
   - [4.1 The enum and base class](#41-the-enum-and-base-class)
   - [4.2 `ALWAYS` — ROCm Attention](#42-always--rocm-attention)
   - [4.3 `ALWAYS` — FlashAttention 3](#43-always--flashattention-3)
   - [4.4 `UNIFORM_BATCH` — FlashAttention 2](#44-uniform_batch--flashattention-2)
   - [4.5 `UNIFORM_SINGLE_TOKEN_DECODE` / `UNIFORM_BATCH` — FlashInfer](#45-uniform_single_token_decode--uniform_batch--flashinfer)
   - [4.6 `ALWAYS` — Triton / Flex Attention](#46-always--triton--flex-attention)
   - [4.7 Summary table](#47-summary-table)
5. [The Auto-Downgrade System](#5-the-auto-downgrade-system)
   - [5.1 Computing `min_cg_support`](#51-computing-min_cg_support)
   - [5.2 Downgrade rules](#52-downgrade-rules)
   - [5.3 Downgrade examples](#53-downgrade-examples)
6. [PIECEWISE — The Breakable CUDA Graph Path](#6-piecewise--the-breakable-cuda-graph-path)
   - [6.1 `@eager_break_during_capture`](#61-eager_break_during_capture)
   - [6.2 `BreakableCUDAGraphCapture`](#62-breakablecudagraphcapture)
   - [6.3 Auto-enable for specific models](#63-auto-enable-for-specific-models)
7. [Runtime: Dispatch, Buffers, and the Size Switch](#7-runtime-dispatch-buffers-and-the-size-switch)
   - [7.1 When dispatch happens](#71-when-dispatch-happens)
   - [7.2 `InputBuffers` — the shared static pool](#72-inputbuffers--the-shared-static-pool)
   - [7.3 Size selection via `_candidates`](#73-size-selection-via-_candidates)
   - [7.4 The size switch — step-by-step example](#74-the-size-switch--step-by-step-example)
8. [Execution Path Comparison: FULL vs PIECEWISE vs NONE](#8-execution-path-comparison-full-vs-piecewise-vs-none)
9. [`ForwardContext` and `BatchDescriptor`](#9-forwardcontext-and-batchdescriptor)
10. [Tensor Dumping Inside Graph Replay](#10-tensor-dumping-inside-graph-replay)
    - [10.1 The mechanism](#101-the-mechanism)
    - [10.2 Implementation pattern](#102-implementation-pattern)
    - [10.3 Constraints](#103-constraints)
    - [10.4 Multiple-size consideration](#104-multiple-size-consideration)
11. [Configuration Reference](#11-configuration-reference)
    - [11.1 `-O` level presets](#111--o-level-presets)
    - [11.2 Group 1 — Compilation mode](#112-group-1--compilation-mode)
    - [11.3 Group 2 — CUDA graph capture](#113-group-2--cuda-graph-capture)
    - [11.4 Group 3 — FX graph splitting](#114-group-3--fx-graph-splitting)
    - [11.5 Group 4 — Inductor compilation](#115-group-4--inductor-compilation)
    - [11.6 Group 5 — Multimodal encoder](#116-group-5--multimodal-encoder)
    - [11.7 Group 6 — Debug and IR](#117-group-6--debug-and-ir)
    - [11.8 Group 7 — Fusion passes (`pass_config`)](#118-group-7--fusion-passes-pass_config)
    - [11.9 Practical CLI examples](#119-practical-cli-examples)
12. [End-to-End Flow Diagram](#12-end-to-end-flow-diagram)
13. [How to Add a New Attention Backend](#13-how-to-add-a-new-attention-backend)
14. [Quick Reference: Mode Selection Matrix](#14-quick-reference-mode-selection-matrix)

---

## 1. The Two Orthogonal Axes

vLLM's performance system has two independent dimensions:

| Axis | Controls | Configured by |
|---|---|---|
| **torch.compile** (`CompilationMode`) | *What code runs* — fuses ops, generates Triton kernels, eliminates Python overhead | `CompilationConfig.mode` (0–3) |
| **CUDA graphs** (`CUDAGraphMode`) | *How the compiled code is dispatched* — replays pre-recorded GPU command sequences, eliminating per-step CPU→GPU launch overhead | `CompilationConfig.cudagraph_mode` |

They are orthogonal but compose together:

- **FULL graphs** work with or without `torch.compile` — the outer `torch.cuda.graph()` capture doesn't care how kernels were generated.
- **PIECEWISE graphs** require `torch.compile` (`VLLM_COMPILE`) on the standard path, because the FX graph must be split at Dynamo level. The `BreakableCUDAGraph` path avoids this.
- During FULL capture, the model runner passes `CUDAGraphMode.NONE` into the forward context, so inner `CUDAGraphWrapper`s pass through eagerly — the outer `torch.cuda.graph()` captures everything including already-compiled Triton kernels.

---

## 2. The torch.compile Layer

### 2.1 `CompilationMode` enum

**`vllm/config/compilation.py:37`**

```python
class CompilationMode(enum.IntEnum):
    NONE = 0             # pure eager PyTorch, no torch.compile
    STOCK_TORCH_COMPILE = 1  # standard torch.compile pipeline
    DYNAMO_TRACE_ONCE = 2    # single Dynamo trace, no recompilation
    VLLM_COMPILE = 3     # vLLM's custom Inductor backend with piecewise compilation
```

Only `VLLM_COMPILE` (3) activates vLLM's `VllmBackend`, which performs FX graph splitting and per-segment Inductor compilation.

### 2.2 How `VLLM_COMPILE` is triggered

Every vLLM model is decorated with `@support_torch_compile` (`vllm/compilation/decorators.py:118`). This decorator patches `__call__` to intercept the first real forward pass:

```python
# decorators.py:502–684 (simplified)
def __call__(self, *args, **kwargs):
    if self.do_not_compile:            # mode == NONE or STOCK_TORCH_COMPILE
        return self.forward(*args, **kwargs)

    if self.compiled:
        return TorchCompileWithNoGuardsWrapper.__call__(self, ...)  # replay compiled

    # First call only: mark batch dimension as dynamic for Dynamo
    _mark_dynamic_inputs(self, ds_type, *args, **kwargs)
    # torch.compile triggers; Dynamo traces forward(), hands FX graph to VllmBackend
    output = TorchCompileWithNoGuardsWrapper.__call__(self, ...)
    self.compiled = True
    return output
```

For example, `LlamaModel`'s `input_ids` has `dim=0` (num_tokens) marked dynamic. Dynamo traces through all 32 transformer layers, producing a single large FX graph.

### 2.3 `VllmBackend` — FX Graph Splitting

**`vllm/compilation/backends.py:805`**

`VllmBackend.__call__()` is invoked by Dynamo after tracing. Its work:
1. Split the FX graph at `splitting_ops` (attention ops).
2. Compile each non-attention segment with Inductor.
3. Wrap each compiled segment with `CUDAGraphWrapper`.

**`split_graph()` — `backends.py:553`**

```python
def split_graph(graph: fx.GraphModule, splitting_ops: list[str]):
    subgraph_id = 0
    node_to_subgraph_id: dict[fx.Node, int] = {}
    split_op_graphs: list[int] = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if should_split(node, splitting_ops):
            # Attention op → its own subgraph, NOT compiled with Inductor
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    split_gm = torch.fx.passes.split_module.split_module(
        graph, None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True,
    )
    return split_gm, outputs
```

The default `splitting_ops` list (`compilation.py:764`) covers all vLLM attention primitives:

```python
_attention_ops = [
    "vllm::unified_attention_with_output",      # standard attention (all backends)
    "vllm::unified_mla_attention_with_output",  # MLA (DeepSeek)
    "vllm::mamba_mixer2", "vllm::mamba_mixer",  # Mamba SSM
    "vllm::short_conv", "vllm::linear_attention",
    "vllm::sparse_attn_indexer",                # DeepSeekV4, MiniMax
    "vllm::deepseek_v4_attention",
    "vllm::hpc_rope_norm_forward",
    ...
]
```

**Example: Llama-3-8B with 2 layers**

Before split (one big FX graph):
```
placeholder(input_ids) → embed → norm → linear(Q,K,V) → unified_attention_with_output
→ output_proj → mlp → norm → linear(Q,K,V) → unified_attention_with_output
→ output_proj → mlp → lm_head → output
```

After `split_graph()`:
```
submod_0  [is_splitting=False]: embed → norm → linear(Q,K,V)
submod_1  [is_splitting=True]:  unified_attention_with_output  ← runs eagerly
submod_2  [is_splitting=False]: output_proj → mlp → norm → linear(Q,K,V)
submod_3  [is_splitting=True]:  unified_attention_with_output  ← runs eagerly
submod_4  [is_splitting=False]: output_proj → mlp → lm_head
```

A 32-layer model produces 65 submodules: 33 non-splitting (compiled) + 32 splitting (eager).

### 2.4 `PiecewiseCompileInterpreter` — Compile and Wrap

**`backends.py:687` and `633`**

For each non-splitting submodule the interpreter:
1. Creates `PiecewiseBackend(submod, ...)` — compiles with Inductor for each batch size range.
2. Wraps with `CUDAGraphWrapper(runtime_mode=PIECEWISE)`.

```python
# backends.py:633–684
def wrap_with_cudagraph_if_needed(piecewise_backend, vllm_config,
                                   compilation_config, is_first_graph, is_last_graph):
    if not compilation_config.cudagraph_mode.has_piecewise_cudagraphs():
        return piecewise_backend     # no-op if PIECEWISE not needed

    return CUDAGraphWrapper(
        runnable=piecewise_backend,
        vllm_config=vllm_config,
        runtime_mode=CUDAGraphMode.PIECEWISE,  # always PIECEWISE for subgraphs
        cudagraph_options=CUDAGraphOptions(
            debug_log_enable=is_first_graph,   # log only once
            gc_disable=not is_first_graph,     # GC only at first segment
            weak_ref_output=is_last_graph,     # drop output ref at last segment
        ),
    )
```

After `PiecewiseCompileInterpreter` runs, `split_gm` becomes:

```
split_gm:
  submod_0 → CUDAGraphWrapper(PiecewiseBackend(Inductor(embed+norm+linear)))
  submod_1 → raw FX attention subgraph  (eager)
  submod_2 → CUDAGraphWrapper(PiecewiseBackend(Inductor(proj+mlp+norm+linear)))
  submod_3 → raw FX attention subgraph  (eager)
  submod_4 → CUDAGraphWrapper(PiecewiseBackend(Inductor(proj+mlp+lm_head)))
```

---

## 3. The CUDA Graph Layer

### 3.1 `CUDAGraphMode` enum

**`vllm/config/compilation.py:53`**

```python
class CUDAGraphMode(enum.Enum):
    NONE             = 0
    PIECEWISE        = 1
    FULL             = 2
    FULL_DECODE_ONLY   = (FULL, NONE)          # split-routine: (decode_mode, mixed_mode)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)     # split-routine, v1 default
```

The last two are "split-routine" modes: `value[0]` = mode for pure decode batches, `value[1]` = mode for mixed prefill+decode batches.

Key helper methods:
```python
def decode_mode(self) -> CUDAGraphMode: ...  # mode for pure decode
def mixed_mode(self) -> CUDAGraphMode: ...   # mode for mixed/prefill
def has_piecewise_cudagraphs(self) -> bool:  # needs piecewise compilation?
def has_full_cudagraphs(self) -> bool:       # needs full graph capture?
```

**Mode comparison:**

| Mode | Decode batch | Mixed/Prefill batch | Backend req | Compile req |
|---|---|---|---|---|
| `NONE` | eager | eager | any | none |
| `PIECEWISE` | piecewise graph | piecewise graph | any | `VLLM_COMPILE` or breakable |
| `FULL` | full graph | full graph | `ALWAYS` | none |
| `FULL_DECODE_ONLY` | full graph | **eager** | `≥ UNIFORM_SINGLE_TOKEN_DECODE` | none |
| `FULL_AND_PIECEWISE` | full graph | piecewise graph | `≥ UNIFORM_SINGLE_TOKEN_DECODE` | `VLLM_COMPILE` or breakable |

### 3.2 `NONE` — No CUDA Graph

All forward passes run in plain eager mode. No warmup, no batch-shape constraints.

**When it applies:**
- `--enforce-eager`, `-O0`, or `enforce_eager=True`
- After auto-downgrade when the attention backend has no graph support

**Execution path** (`model_runner.py:1613`):
```python
with set_forward_context(..., cudagraph_runtime_mode=CUDAGraphMode.NONE, ...):
    model_output = self.model(**model_inputs)   # fully eager
```

**Use case:** Debugging, models with dynamic control flow, backends declaring `AttentionCGSupport.NEVER`.

---

### 3.3 `PIECEWISE` — Piecewise CUDA Graphs

Non-attention FX subgraphs are captured as separate CUDA graphs; attention ops run eagerly between replays. Works for all batch types: decode, prefill, mixed.

**Execution path** (`model_runner.py:1625`):
```python
with set_forward_context(..., cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, ...):
    model_output = self.cudagraph_manager.run_pw_graph(self.model, model_inputs)
```

**`CUDAGraphWrapper` dispatch** (`cuda_graph.py:233`):
```python
def __call__(self, *args, **kwargs):
    cudagraph_runtime_mode = get_forward_context().cudagraph_runtime_mode

    if (
        cudagraph_runtime_mode == CUDAGraphMode.NONE
        or cudagraph_runtime_mode != self.runtime_mode  # self.runtime_mode==PIECEWISE
    ):
        return self.runnable(*args, **kwargs)   # pass-through

    if entry.cudagraph is None:
        # First call: capture Inductor Triton kernels inside torch.cuda.graph()
        with torch.cuda.graph(cudagraph, pool=self.graph_pool):
            output = self.runnable(*args, **kwargs)
        entry.cudagraph = cudagraph
    else:
        entry.cudagraph.replay()   # subsequent calls: replay
        return entry.output
```

**GC optimization:** The first subgraph runs `gc.collect()` + `empty_cache()` once. All subsequent subgraphs patch these out via `gc_disable=not is_first_graph`.

**Example: capture sizes [8, 16, 32]**

For a 32-layer Llama model: 33 segments × 3 sizes = **99 CUDA graphs** total.

---

### 3.4 `FULL` — Full Model CUDA Graph

The entire model forward pass is captured as a single CUDA graph per `(num_tokens, num_reqs)` shape. Zero Python overhead during inference — one `cudaGraphLaunch` replays the entire forward.

**Capture path** (`cudagraph_utils.py:356`):
```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, self.pool):
    forward_fn(CUDAGraphMode.NONE)   # inner CUDAGraphWrappers pass through;
                                      # outer graph captures everything
self.graphs[desc] = graph
```

**Why `CUDAGraphMode.NONE` inside the outer capture?**

`CUDAGraphWrapper.__call__` checks `cudagraph_runtime_mode`. When it sees `NONE`, it calls `self.runnable(*args, **kwargs)` directly. The outer `torch.cuda.graph()` context then captures all resulting Triton kernel launches — including attention — into one monolithic replay script.

**Replay** (`model_runner.py:1598`):
```python
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    # Inputs already copied into static buffers before this call
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
    # → self.graphs[desc].replay()
```

**Example: Llama-3-8B with capture_sizes=[1,2,4,8,16,32,64,128,256]**

9 graphs, each capturing the entire 32-layer forward including all 32 attention kernels. Replay time per step: ~microseconds (one `cudaGraphLaunch` call).

---

### 3.5 `FULL_DECODE_ONLY` — Decode FULL, Mixed Eager

Decode batches (query_len==1) use FULL graphs. Any batch containing prefill or extend tokens runs **fully eager** (no graph at all).

**`mixed_mode()` is `NONE`**, so no mixed descriptors are ever created in `_init_candidates()`:

```python
mixed_mode = self.cudagraph_mode.mixed_mode()   # NONE

if mixed_mode:      # NONE is falsy → this block is never executed
    desc = BatchExecutionDescriptor(cg_mode=mixed_mode, ...)
```

Any non-decode batch falls through to the `NONE` fallback in `dispatch()` → **fully eager**.

**Chunked prefill interaction:** During chunked prefill, every step processes a chunk of tokens with `query_len > 1` — treated as "mixed". With `FULL_DECODE_ONLY`, every chunked-prefill step runs **fully eager** (no Triton kernels, no graph replay). Only steady-state decode steps (all `query_len == 1`) benefit from FULL graphs. This makes `FULL_DECODE_ONLY` a poor choice for workloads with heavy prefill traffic. `FULL_AND_PIECEWISE` is better: prefill/mixed steps use PIECEWISE instead of pure eager.

**Use case:** P/D disaggregation decode instances, where prefill is handled upstream and is rare.

---

### 3.6 `FULL_AND_PIECEWISE` — Default

Decode batches use FULL graphs. Mixed prefill+decode batches use PIECEWISE graphs.

**Capture order** (`cudagraph_utils.py:325`):
```python
# PIECEWISE first: larger activations → allocates bigger pool slots
# FULL second: smaller activations → fits inside already-allocated pool
for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
    for desc in self._capture_descs[mode]:
        forward_fn = create_forward_fn(desc, warmup=True)
        forward_fn(CUDAGraphMode.NONE)          # warmup

        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
            forward_fn(CUDAGraphMode.PIECEWISE) # triggers CUDAGraphWrapper capture
        else:
            with torch.cuda.graph(graph, self.pool):
                forward_fn(CUDAGraphMode.NONE)  # outer FULL capture
            self.graphs[desc] = graph
```

**Dispatch** (`cudagraph_utils.py:382`):
```python
def dispatch(self, num_reqs, num_tokens, uniform_token_count, ...):
    key = (num_tokens, effective_loras)
    for desc in self._candidates[key]:   # priority-ordered: FULL first
        if _is_compatible(desc, ...):
            return desc
    return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, ...)  # fallback
```

**Example: Llama-3-8B, capture_sizes=[1,2,4,8,16,32,64,128,256]**

- Decode step (8 reqs, q_len=1, tokens=8): → `FULL desc(num_tokens=8)` → `graph.replay()`
- Prefill step (4 reqs, mixed q_lens, tokens=512): → `PIECEWISE desc(num_tokens=512)` → piecewise replay

---

## 4. Backend Capability: `AttentionCGSupport`

### 4.1 The enum and base class

**`vllm/v1/attention/backend.py:633`**

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3
    """Supports full CUDA graphs for any batch type: decode, prefill, mixed."""
    UNIFORM_BATCH = 2
    """Supports graphs for batches where all requests have the same query length."""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Supports graphs only for pure decode batches with query_len == 1."""
    NEVER = 0
    """No CUDA graph support for attention."""
```

**Base class** (`vllm/v1/attention/backend.py:650`):

```python
class AttentionMetadataBuilder(ABC):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        return cls._cudagraph_support  # override for dynamic logic
```

Default is `NEVER`. Backends opt in by overriding the class variable (static) or the classmethod (dynamic/conditional).

### 4.2 `ALWAYS` — ROCm Attention

**`vllm/v1/attention/backends/rocm_attn.py:77`**

```python
class RocmAttentionMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
```

ROCm's attention kernel handles any batch shape, parameterized by `query_start_loc` and `seq_lens` tensors from the globally shared `CommonAttentionMetadata`. No backend-owned static buffers needed.

**`build_for_cudagraph_capture()`** (`rocm_attn.py:97`):
```python
def build_for_cudagraph_capture(self, common_attn_metadata):
    attn_metadata = self.build(0, common_attn_metadata)
    # Use seq_lens=1 to avoid slow capture on long-sequence codepaths
    attn_metadata.seq_lens.fill_(1)
    # Zero to avoid invalid memory access in prefix_prefill kernel (#25985)
    common_attn_metadata.query_start_loc.zero_()
    common_attn_metadata.query_start_loc_cpu.zero_()
    return attn_metadata
```

### 4.3 `ALWAYS` — FlashAttention 3

**`vllm/v1/attention/backends/flash_attn.py:356`** (FA3 branch)

FA3's AOT tile scheduler pre-computes per-batch dispatch tables that must live at **stable GPU addresses**. Pre-allocated in `__init__` (`flash_attn.py:474`):

```python
if self.use_full_cuda_graph and self.aot_schedule:
    max_batch_size = max(vllm_config.scheduler_config.max_num_seqs, self.max_cudagraph_size or 0)
    self.scheduler_metadata = torch.zeros(
        1 + round_up(max_batch_size, 4) * 4,
        dtype=torch.int32, device=self.device,
    )
    self.max_num_splits = self.attention_config.flash_attn_max_num_splits_for_cuda_graph
```

At each `build()`, per-batch data is copied into the static buffer:
```python
def _store_scheduler_metadata(self, scheduler_metadata):
    if self.use_full_cuda_graph and scheduler_metadata is not None:
        n = scheduler_metadata.shape[0]
        self.scheduler_metadata[:n] = scheduler_metadata  # copy into static addr
        self.scheduler_metadata[n:] = 0                   # zero tail to prevent stale dispatch
        return self.scheduler_metadata[:n]                 # return view of static buffer
    return scheduler_metadata
```

`max_num_splits` is also pinned for FULL graphs (`flash_attn.py:578`):
```python
max_num_splits = 0   # 0 = FA3 heuristic (varies per step, not CG-safe)
if self.use_full_cuda_graph and num_actual_tokens <= self.max_cudagraph_size:
    max_num_splits = self.max_num_splits  # fixed → static intermediate buffer size
```

Additional static buffers for special models: `persistent_rswa_prefix_lens` (R-SWA), `mm_prefix_query_ranges_gpu` (PrefixLM).

### 4.4 `UNIFORM_BATCH` — FlashAttention 2

```python
_cudagraph_support = (
    AttentionCGSupport.ALWAYS       # FA3
    if get_flash_attn_version() == 3
    else AttentionCGSupport.UNIFORM_BATCH   # FA2
)
```

FA2 has a `max_query_len=1` packed-GQA kernel path that cannot handle mixed prefill+decode batches. `resolve_cudagraph_mode_and_sizes()` sees `mixed_mode==FULL` + `UNIFORM_BATCH` → auto-downgrades to `FULL_AND_PIECEWISE`.

### 4.5 `UNIFORM_SINGLE_TOKEN_DECODE` / `UNIFORM_BATCH` — FlashInfer

**`vllm/v1/attention/backends/flashinfer.py:959`** — dynamic, hardware-dependent:

```python
@classmethod
def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
    if current_platform.is_device_capability(90):   # H100 XQA
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    is_sm12x = current_platform.is_device_capability_family(120)  # B200
    if is_sm12x and decode_context_parallel_size > 1:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    if has_trtllm_support and (is_sm12x or not use_non_causal):
        return AttentionCGSupport.UNIFORM_BATCH
    return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
```

**Why decode-only:** `BatchDecodeWithPagedKVCacheWrapper` pre-plans dispatch tables assuming all requests have `query_len==1`. Mixed batches need `BatchPrefillWithPagedKVCacheWrapper`, which cannot be graph-captured (plan shape varies per batch).

**One wrapper per batch size** (`flashinfer.py:727`):
```python
if self.enable_cuda_graph:
    self._decode_wrappers_cudagraph: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}
```

Each wrapper is created with pinned static buffer slices (`flashinfer.py:1141`):
```python
def _get_decode_wrapper(self, batch_size, use_cudagraph=False):
    if use_cudagraph:
        paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]   # static addr
        paged_kv_indices = self.paged_kv_indices.gpu                    # full static buffer
        paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
    else:
        paged_kv_indptr = paged_kv_indices = paged_kv_last_page_len = None

    return BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout,
        use_cuda_graph=use_cudagraph,          # binds to static buffer addresses
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    )
```

Graph path gated on pure decode (`flashinfer.py:1628`):
```python
use_cudagraph = self.enable_cuda_graph and (num_prefills == 0) and (num_decode_tokens <= max_bs)
```

### 4.6 `ALWAYS` — Triton / Flex Attention

**`triton_attn.py:102`**, **`flex_attention.py:857`**:

```python
class TritonAttentionMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

class FlexAttentionMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
```

Pure PyTorch/Triton kernels accept arbitrary batch shapes. Metadata is fully parameterized by static tensors from `CommonAttentionMetadata`. No backend-owned static buffers needed.

### 4.7 Summary table

| Backend file | Support level | Key implementation detail |
|---|---|---|
| `rocm_attn.py` | `ALWAYS` | Static buffers in `CommonAttentionMetadata`; `build_for_cudagraph_capture` zeros seq_lens |
| `flash_attn.py` (FA3) | `ALWAYS` | `self.scheduler_metadata` pre-allocated in `__init__`; `max_num_splits` pinned |
| `flash_attn.py` (FA2) | `UNIFORM_BATCH` | packed-GQA restriction on mixed batches |
| `triton_attn.py` | `ALWAYS` | Pure Triton, no static buffers needed |
| `flex_attention.py` | `ALWAYS` | PyTorch flexible attention |
| `flashinfer.py` (SM90/H100) | `UNIFORM_SINGLE_TOKEN_DECODE` | One `BatchDecodeWithPagedKVCacheWrapper` per batch size |
| `flashinfer.py` (SM120+TRT-LLM) | `UNIFORM_BATCH` | B200 with spec-decode via TRT-LLM XQA |
| `hpc_attn.py` | `UNIFORM_SINGLE_TOKEN_DECODE` / `UNIFORM_BATCH` | Upgrades to `UNIFORM_BATCH` when spec-decode active |
| `mla/flashattn_mla.py` | `UNIFORM_BATCH` | MLA spec-decode |
| `mla/triton_mla.py` | `UNIFORM_SINGLE_TOKEN_DECODE` | MLA decode only |
| `mla/indexer.py` | `ALWAYS` or `UNIFORM_BATCH` | Conditional on varlen MQA logits support |
| `chunked_local_attention.py` | `NEVER` | Chunked attn not graph-safe |

---

## 5. The Auto-Downgrade System

**`vllm/config/compilation.py:1368`**

`resolve_cudagraph_mode_and_sizes()` takes the user's requested `cudagraph_mode` and the `min_cg_support` (minimum across all active backends) and automatically resolves conflicts with warnings.

### 5.1 Computing `min_cg_support`

**`vllm/v1/worker/gpu/attn_utils.py:148`**

```python
min_cg_support = AttentionCGSupport.ALWAYS
for kv_cache_group_id, groups in enumerate(attn_groups):
    for group in groups:
        cg_support = builder.get_cudagraph_support(vllm_config, group.kv_cache_spec)
        if cg_support.value < min_cg_support.value:
            min_cg_support = cg_support           # take the minimum
            min_cg_attn_backend = group.backend.__name__
```

One backend with `NEVER` overrides all others. For hybrid models (e.g., Mamba + attention), the most restrictive backend determines the floor.

### 5.2 Downgrade rules

```python
# vllm/config/compilation.py:1387–1472

# Rule A: user wants mixed_mode=FULL but backend can't handle mixed batches
if cudagraph_mode.mixed_mode() == FULL and min_cg_support != ALWAYS:
    if min_cg_support == NEVER:
        raise ValueError(...)               # fatal: can't do FULL for mixed
    if splitting_ops_contain_attention():
        cudagraph_mode = FULL_AND_PIECEWISE # warn + downgrade
    else:
        cudagraph_mode = FULL_DECODE_ONLY   # warn + downgrade

# Rule B: decode_mode=FULL but backend is NEVER
if cudagraph_mode.decode_mode() == FULL and min_cg_support == NEVER:
    if VLLM_COMPILE and splitting_ops_contain_attention():
        cudagraph_mode = PIECEWISE          # warn + downgrade
    else:
        cudagraph_mode = NONE               # warn + downgrade

# Rule C: spec-decode + decode_mode=FULL but backend < UNIFORM_BATCH
if (decode_mode == FULL and uniform_decode_query_len > 1
        and min_cg_support < UNIFORM_BATCH):
    if splitting_ops_contain_attention():
        cudagraph_mode = PIECEWISE          # warn + downgrade
    else:
        cudagraph_mode = NONE               # warn + downgrade
```

### 5.3 Downgrade examples

| Hardware | Backend | User requests | Effective mode | Reason |
|---|---|---|---|---|
| H100 + FA3 | `ALWAYS` | `FULL_AND_PIECEWISE` | `FULL_AND_PIECEWISE` | No downgrade |
| H100 + FA3 | `ALWAYS` | `FULL` | `FULL` | No downgrade |
| H100 + FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | `FULL_AND_PIECEWISE` | `FULL_AND_PIECEWISE` | decode=FULL OK; mixed=PIECEWISE OK |
| H100 + FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | `FULL` | → `FULL_DECODE_ONLY` | mixed_mode=FULL not supported |
| H100 + FlashInfer + spec-decode | `UNIFORM_SINGLE_TOKEN_DECODE` | `FULL_AND_PIECEWISE` | → `PIECEWISE` | decode=FULL + spec-decode needs UNIFORM_BATCH |
| H100 + FlashInfer + spec-decode | `UNIFORM_BATCH` | `FULL_AND_PIECEWISE` | `FULL_AND_PIECEWISE` | UNIFORM_BATCH supports spec-decode |
| Any + `NEVER` backend | `NEVER` | `FULL_AND_PIECEWISE` | → `PIECEWISE` | if VLLM_COMPILE + splitting_ops |
| Any + `NEVER` backend | `NEVER` | `FULL` | `ValueError` | No FULL possible with NEVER |
| AMD ROCm | `ALWAYS` | `FULL_AND_PIECEWISE` | `FULL_AND_PIECEWISE` | No downgrade |

---

## 6. PIECEWISE — The Breakable CUDA Graph Path

**`vllm/compilation/breakable_cudagraph.py`** — enabled by `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.

Implements PIECEWISE graphs **without** `torch.compile`. Instead of pre-splitting at Dynamo FX level, it intercepts attention ops at runtime during a single forward pass.

### 6.1 `@eager_break_during_capture`

Applied to every attention function (`attention.py:758`):

```python
@eager_break_during_capture   # outermost — intercepts capture
@maybe_transfer_kv_layer      # P/D disaggregation (inner — runs inside eager break)
def unified_attention_with_output(query, key, value, output, layer_name, ...) -> None:
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    self.impl.forward(self, query, ...)
```

The decorator logic (`breakable_cudagraph.py:59`):
```python
def wrapper(*args, **kwargs):
    capture = BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        return fn(*args, **kwargs)

    mode = get_forward_context().cudagraph_runtime_mode
    if mode == CUDAGraphMode.FULL:
        return fn(*args, **kwargs)   # in FULL mode: captured into outer graph

    # In PIECEWISE capture: break here
    weak_args = tuple(weak_ref_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)
    return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))
```

The `FULL` mode guard means the same decorated function works for both modes: during FULL capture, the decorator is a no-op and the attention op is captured inside the outer `torch.cuda.graph()`.

### 6.2 `BreakableCUDAGraphCapture`

**`breakable_cudagraph.py:125`**

```python
class BreakableCUDAGraphCapture:
    def __enter__(self):
        BreakableCUDAGraphCapture._tls.active = self
        self._begin_segment()    # g.capture_begin(pool=...)
        return self

    def __exit__(self, ...):
        self._end_segment()      # g.capture_end() → segments.append(g.replay)

    def add_eager(self, fn):
        self._end_segment()      # end current graph segment
        result = fn()            # run attention eagerly
        self.segments.append(fn) # record fn for replay
        self._begin_segment()    # start new graph segment
        return result

    def replay(self):
        for r in self.segments:
            r()   # alternates: graph.replay(), eager_attn_fn(), graph.replay(), ...
```

**2-layer model capture timeline:**
```python
capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
with capture:
    output = model(input_ids, positions, ...)
    # 1. embed+norm+linear(Q,K,V)  → captured into graph0
    # 2. @eager_break_during_capture → add_eager(attn_layer0_fn)
    #    → end graph0, run attention eagerly, start graph1
    # 3. proj+mlp+norm+linear(Q,K,V) → captured into graph1
    # 4. @eager_break_during_capture → add_eager(attn_layer1_fn)
    #    → end graph1, run attention eagerly, start graph2
    # 5. proj+mlp+lm_head → captured into graph2

# capture.segments = [graph0.replay, attn_layer0_fn, graph1.replay,
#                     attn_layer1_fn, graph2.replay]
```

Replay each step: `for r in capture.segments: r()`

### 6.3 Auto-enable for specific models

`VLLM_USE_BREAKABLE_CUDAGRAPH` is automatically set to `1` for DeepSeekV4, Inkling, KimiK3, MiniMaxM3 — models that use custom attention ops not yet integrated into the `torch.compile` piecewise path.

---

## 7. Runtime: Dispatch, Buffers, and the Size Switch

### 7.1 When dispatch happens

Dispatch happens **before input preparation**, at the start of `execute_model()`:

```
execute_model()
│
├── 1. update_requests / finish_requests / add_requests    ← request state management
│
├── 2. gather_batch_req_state()                            ← count tokens, check uniformity
│       → num_toks, uniform_tok_count
│
├── 3. dispatch_cg_and_sync_dp()                           ← ★ DISPATCH HAPPENS HERE ★
│       → batch_desc (cg_mode, num_tokens, num_reqs, ...)
│
├── 4. prepare_inputs(batch_desc)                          ← copy request state into static buffers
│       → InputBatch (views into InputBuffers)
│
├── 5. prepare_attn()                                      ← build attention metadata
│
└── 6. [FULL]      run_fullgraph(batch_desc)  → graph.replay()
    [PIECEWISE]    run_pw_graph(model, inputs)
    [NONE]         model(**model_inputs)
```

`dispatch_cg_and_sync_dp()` (`dp_utils.py:98`) is a pure CPU operation — a dict lookup:

```python
def dispatch_cg_and_sync_dp(cudagraph_manager, num_reqs, num_tokens,
                              uniform_token_count, ...):
    if need_eager:   # profile run or encoder-decoder with encoder inputs
        return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, ...)

    batch_desc = cudagraph_manager.dispatch(
        num_reqs, num_tokens, uniform_token_count, ...
    )
    # + DP sync if dp_size > 1 (all ranks must agree on the same graph size)
    return batch_desc, num_tokens_across_dp
```

No GPU work at dispatch time. The result `batch_desc.num_tokens` is the **padded** captured size.

### 7.2 `InputBuffers` — the shared static pool

**`vllm/v1/worker/gpu/input_batch.py:17`**

```python
class InputBuffers:
    def __init__(self, max_num_reqs, max_num_tokens, device):
        # Allocated ONCE at startup, at maximum possible size.
        # GPU addresses NEVER change for the lifetime of the process.
        self.input_ids    = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions    = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding   = torch.zeros(max_num_tokens, dtype=torch.bool,  device=device)
        self.query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.seq_lens     = torch.zeros(max_num_reqs,   dtype=torch.int32, device=device)
```

Every captured CUDA graph — for size 4, 8, 32, or 256 — was captured reading from these same GPU base addresses. On replay, each graph reads from those fixed addresses (now containing the current step's data). There is no "copy between graph pools" when switching sizes.

`prepare_inputs()` (`model_runner.py:1083`) writes into these static buffers every step:

```python
def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
    num_tokens            = batch_req_state.num_tokens          # actual (e.g. 6)
    num_tokens_after_pad  = batch_desc.num_tokens               # padded (e.g. 8)

    # Mark padding slots
    is_padding = self.input_buffers.is_padding
    is_padding[:num_tokens].fill_(False)
    is_padding[num_tokens:num_tokens_after_pad].fill_(True)

    # Copy token ids, positions, seq_lens into static buffers (async GPU copy)
    async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
    prepare_prefill_inputs(self.input_buffers.input_ids, ...)
    prepare_pos_seq_lens(..., self.input_buffers.positions, self.input_buffers.seq_lens)
```

`InputBatch` is then created as **views into those static buffers** — no new allocations:

```python
input_batch = InputBatch(
    input_ids=self.input_buffers.input_ids[:num_tokens_after_pad],  # view [0:8]
    positions=self.input_buffers.positions[:num_tokens_after_pad],
    seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
    ...
)
```

### 7.3 Size selection via `_candidates`

`_init_candidates()` (`cudagraph_utils.py:287`) builds a lookup table mapping every possible actual token count to the **smallest captured size ≥ it**:

```python
for token_cg_size in all_token_counts:        # e.g. [4, 8, 16, 32]
    for i in range(current_range_start, token_cg_size + 1):
        self._candidates[(i, num_active_loras)] = descs_by_token_lora[(token_cg_size, ...)]
```

With capture sizes `[4, 8, 16, 32]`:

| Actual tokens | Padded to | Graph used |
|---|---|---|
| 1–4 | 4 | `graphs[desc(num_tokens=4)]` |
| 5–8 | 8 | `graphs[desc(num_tokens=8)]` |
| 9–16 | 16 | `graphs[desc(num_tokens=16)]` |
| 17–32 | 32 | `graphs[desc(num_tokens=32)]` |
| 33+ | — | eager fallback (NONE) |

### 7.4 The size switch — step-by-step example

**Setup:**
```
capture_sizes = [4, 8, 16, 32]
max_num_tokens = 32  (simplified)
InputBuffers at fixed GPU addresses:
  input_ids    @ 0xA000  [32 x int32]
  positions    @ 0xB000  [32 x int64]
  seq_lens     @ 0xC000  [32 x int32]
  query_start_loc @ 0xD000  [33 x int32]
  is_padding   @ 0xE000  [32 x bool]

Captured graphs:
  graphs[desc(num_tokens=4)]  → <CUDAGraph@0x1000>
  graphs[desc(num_tokens=8)]  → <CUDAGraph@0x2000>
  graphs[desc(num_tokens=16)] → <CUDAGraph@0x3000>
  graphs[desc(num_tokens=32)] → <CUDAGraph@0x4000>
```

**Step 1: 3 decode requests, seq_lens=[100, 200, 300]**

```
dispatch(num_tokens=3) → _candidates[(3,0)] → desc(num_tokens=4)

prepare_inputs writes:
  input_ids   @ 0xA000: [t0][t1][t2][ 0 ][--- stale ---]
  positions   @ 0xB000: [100][200][300][ 0 ][--- stale ---]
  seq_lens    @ 0xC000: [101][201][301][ 0 ][--- stale ---]
  is_padding  @ 0xE000: [ F ][ F ][ F ][ T ][--- stale ---]

run_fullgraph → <CUDAGraph@0x1000>.replay()
  reads input_ids[0xA000 .. +16 bytes]  (4 elements)
  reads positions[0xB000 .. +32 bytes]  (4 elements)
  runs transformer for 4 token slots (slot 3 = padding)
```

**Step 2: 10 decode requests, seq_lens=[50..59]  ← SIZE SWITCH**

```
dispatch(num_tokens=10) → _candidates[(10,0)] → desc(num_tokens=16)

prepare_inputs overwrites the SAME static buffers:
  input_ids   @ 0xA000: [t0][t1]...[t9][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ][stale]
                         ← 10 real →  ← 6 padding →                    ← not touched
  seq_lens    @ 0xC000: [50][51]..[59][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ][stale]
  is_padding  @ 0xE000: [F][F]..[F][T][T][T][T][T][T][stale]

run_fullgraph → <CUDAGraph@0x3000>.replay()   ← DIFFERENT object
  reads input_ids[0xA000 .. +64 bytes]  (16 elements)
  same base address, wider read
```

**Step 3: 7 decode requests  ← SWITCH BACK**

```
dispatch(num_tokens=7) → desc(num_tokens=8)

prepare_inputs writes:
  input_ids   @ 0xA000: [t0]..[t6][ 0 ][stl][stl]..[stl]
                         ← 7 real ← pad  ← stale from step 2 (not touched, not read)

run_fullgraph → <CUDAGraph@0x2000>.replay()
  reads input_ids[0xA000 .. +32 bytes]  (8 elements)
  slots [8..31] contain stale step-2 data but are NEVER READ by graph_8
```

**GPU memory view across steps:**
```
Addr   0xA000 (input_ids, 32 slots)
       [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ][ 8 ]..[ 15 ][16..31]

Step1: [ t0 ][ t1 ][ t2 ][ 0 ][        stale                          ]
       ←── graph_4 reads ──→

Step2: [ t0 ][ t1 ][ t2 ][ t3 ][ t4 ][ t5 ][ t6 ][ t7 ][ t8 ][ t9 ][ 0*6][stale]
       ←───────────────── graph_16 reads ──────────────────────→

Step3: [ t0 ][ t1 ][ t2 ][ t3 ][ t4 ][ t5 ][ t6 ][ 0 ][stl]...[stl ][stale]
       ←─── graph_8 reads ────→
       stl = stale from step 2, harmless (graph_8 never reads past slot 7)
```

**Key takeaways:**
1. "Switching" is selecting a different `torch.cuda.CUDAGraph` object — a CPU-side choice.
2. No data movement between pools. One shared pool, same static buffer addresses.
3. Slots beyond `desc.num_tokens` are never overwritten and never read — safe to leave stale.
4. The batch is padded to `desc.num_tokens` with `is_padding=True`; most kernels skip padding slots.

---

## 8. Execution Path Comparison: FULL vs PIECEWISE vs NONE

**`vllm/v1/worker/gpu/model_runner.py:1597`**

```python
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    # ── FULL ────────────────────────────────────────────────────────────────
    # Inputs already in static buffers (prepare_inputs ran before this).
    # No set_forward_context — graph replays with zero Python involvement.
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
    # → self.graphs[desc].replay()   one cudaGraphLaunch for entire forward

else:
    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=input_batch.num_tokens_after_padding,
        cudagraph_runtime_mode=batch_desc.cg_mode,   # PIECEWISE or NONE
        batch_descriptor=batch_descriptor,
        ...
    ):
        if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
            # ── PIECEWISE ───────────────────────────────────────────────────
            model_output = self.cudagraph_manager.run_pw_graph(self.model, model_inputs)
            # compile path:  model(**inputs) → each CUDAGraphWrapper replays its segment
            # breakable path: breakable_cg_runner(**inputs) → capture.replay()
        else:
            # ── NONE ────────────────────────────────────────────────────────
            model_output = self.model(**model_inputs)   # fully eager
```

**Critical difference:**

| | FULL | PIECEWISE | NONE |
|---|---|---|---|
| `set_forward_context` | Not called at replay | Called every step | Called every step |
| Python overhead per step | ~0 (one graph launch) | Minimal (N segment dispatches) | Full (one dispatch per op) |
| Attention kernels | Captured inside graph | Run eagerly between graphs | Run eagerly |
| `cudagraph_runtime_mode` seen by wrappers | NONE (pass-through inside capture) | PIECEWISE (triggers capture/replay) | NONE (pass-through) |

---

## 9. `ForwardContext` and `BatchDescriptor`

**`vllm/forward_context.py`**

```python
@dataclass
class ForwardContext:
    attn_metadata: dict[str, AttentionMetadata]
    slot_mapping: dict[str, torch.Tensor]
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE   # key field
    batch_descriptor: BatchDescriptor | None = None
    ...

@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None   # None for PIECEWISE (no req padding needed)
    uniform: bool = False         # True for uniform-length spec-decode batches
    has_lora: bool = False
    num_active_loras: int = 0
```

`BatchDescriptor` is the **cache key** for `CUDAGraphWrapper`. Each unique `(num_tokens, num_reqs, uniform, num_active_loras)` gets its own captured graph.

- **PIECEWISE**: `num_reqs=None` — non-attention segments only depend on `num_tokens` (padded). Variable request counts within the same token count reuse the same graph.
- **FULL**: `num_reqs` is set — attention kernels (captured inside the outer graph) require exactly the right number of metadata entries.

---

## 10. Tensor Dumping Inside Graph Replay

### 10.1 The mechanism

Any CUDA operation executed inside `torch.cuda.graph()` capture is recorded and **re-executed verbatim on every replay**, using the exact same GPU source and CPU destination addresses. A device-to-host copy (`tensor.copy_()` from GPU to pinned CPU) is a CUDA operation — it can be captured and replayed.

This means: **if you pre-allocate a pinned CPU buffer before capture and include the D→H copy inside the capture context, then after each `graph.replay()` that buffer contains the current step's values.**

### 10.2 Implementation pattern

```python
# ── BEFORE capture (outside torch.cuda.graph) ─────────────────────────────
# Pre-allocate pinned CPU buffer. Address must be stable — never reallocate.
dump_buffer = torch.empty(
    tensor_to_dump.shape,
    dtype=tensor_to_dump.dtype,
    pin_memory=True,   # required for async D→H copy inside graph
)

# ── INSIDE torch.cuda.graph(...) capture ──────────────────────────────────
# In vLLM's FULL path (cudagraph_utils.py:365):
with torch.cuda.graph(graph, self.pool):
    forward_fn(CUDAGraphMode.NONE)
    # Insert the dump copy inside the capture context:
    dump_buffer.copy_(tensor_to_dump)   # D→H copy is captured → replayed every step

# ── AFTER graph.replay() at inference time ───────────────────────────────
graph.replay()
print(dump_buffer)   # contains this step's tensor values
```

### 10.3 Constraints

| Constraint | Why |
|---|---|
| GPU source must be a **static buffer** | The graph records the GPU pointer at capture time. Dynamically-allocated tensors change address each step → replay reads stale/wrong memory. |
| CPU destination must be **pinned memory** | CUDA cannot issue an async D→H copy to pageable memory inside a graph. Non-pinned allocation also cannot guarantee address stability. |
| Copy must be **inside** the graph capture context | Operations outside the `with torch.cuda.graph():` block run once at capture time but are NOT recorded. |
| Cannot dump **intermediate activations** from the CUDA graph pool | They are temporary and freed after the graph segment — their GPU addresses are pool-managed and may be reused. Only tensors backed by persistent static buffers are safe. |

### 10.4 Multiple-size consideration

Each captured size is an **independent `torch.cuda.CUDAGraph` object** with its own command sequence. A dump buffer registered inside `graph_for_size_8` is **not** replayed when `graph_for_size_16` runs.

To dump from every step regardless of batch size, allocate **one dump buffer per captured size** before the corresponding capture:

```python
self.dump_bufs: dict[int, torch.Tensor] = {}

# In the capture loop:
for desc in self._capture_descs[CUDAGraphMode.FULL]:
    self.dump_bufs[desc.num_tokens] = torch.empty(
        (desc.num_tokens, hidden_dim), pin_memory=True
    )
    with torch.cuda.graph(graph, self.pool):
        forward_fn(CUDAGraphMode.NONE)
        # Each graph captures its own copy into its own buffer
        self.dump_bufs[desc.num_tokens].copy_(
            self.hidden_states_buf[:desc.num_tokens]
        )
    self.graphs[desc] = graph

# After run_fullgraph(desc):
active_dump = self.dump_bufs[desc.num_tokens][:actual_num_tokens]
```

---

## 11. Configuration Reference

### 11.1 `-O` level presets

**`vllm/config/vllm.py:93`**

| Flag | `cudagraph_mode` | `CompilationMode` | Effect |
|---|---|---|---|
| `-O0` / `--enforce-eager` | `NONE` | `NONE` | Fully eager: no Triton kernels, no graphs |
| `-O1` | `PIECEWISE` | `VLLM_COMPILE` | Inductor Triton + piecewise graphs for all batches |
| `-O2` (default) | `FULL_AND_PIECEWISE` | `VLLM_COMPILE` | Full graphs for decode, piecewise for mixed |
| `-O3` | `FULL_AND_PIECEWISE` | `VLLM_COMPILE` | Same as O2 (O3 reserved for future) |

Passed as `vllm serve ... -cc '{"field": value}'`.

---

### 11.2 Group 1 — Compilation mode

#### `mode` — `int` (0–3), default `3`

| Value | Name | Behavior |
|---|---|---|
| `0` | `NONE` | Pure eager. No torch.compile, no Triton, no FX splitting. Same as `--enforce-eager`. |
| `1` | `STOCK_TORCH_COMPILE` | Standard `torch.compile`. vLLM does not control backend or splitting. |
| `2` | `DYNAMO_TRACE_ONCE` | Single Dynamo trace, guards removed, no recompilation. Requires no dynamic control flow. |
| `3` | `VLLM_COMPILE` | vLLM custom Inductor backend. FX splitting, per-segment compile, custom passes. **Default.** |

```bash
-cc '{"mode": 0}'   # fully eager
-cc '{"mode": 3}'   # vLLM custom backend (default)
```

---

### 11.3 Group 2 — CUDA graph capture

#### `cudagraph_mode` — `str`, default `"FULL_AND_PIECEWISE"`

One of the 5 modes. Subject to auto-downgrade by `resolve_cudagraph_mode_and_sizes()`.

```bash
-cc '{"cudagraph_mode": "NONE"}'
-cc '{"cudagraph_mode": "PIECEWISE"}'
-cc '{"cudagraph_mode": "FULL"}'
-cc '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
-cc '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

#### `cudagraph_capture_sizes` — `list[int]`, default inferred

Token counts for which to capture graphs. Smaller lists = faster startup + less memory, coarser granularity. Batches exceeding the maximum fall through to eager.

```bash
-cc '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]}'
```

#### `max_cudagraph_capture_size` — `int`, default `None`

Upper bound on capture sizes. Any entry above this is dropped.

```bash
-cc '{"max_cudagraph_capture_size": 512}'
```

#### `cudagraph_num_of_warmups` — `int`, default `0`

Warmup passes (eager mode) before capture begins.

#### `cudagraph_copy_inputs` — `bool`, default `False`

PIECEWISE only. When `True`, compiler inserts explicit copies of dynamic inputs into static buffers before each piecewise graph replay.

#### `cudagraph_specialize_lora` — `bool`, default `True`

Capture separate graphs per distinct number of active LoRA adapters. Set `False` to always use the LoRA-enabled graph (wastes compute when no LoRA active, but reduces capture count).

---

### 11.4 Group 3 — FX graph splitting

#### `splitting_ops` — `list[str]`, default `_attention_ops`

Ops at which the Dynamo FX graph is split. Each listed op becomes a subgraph boundary — the op runs eagerly; segments between ops are compiled and graph-captured.

Add a custom op:
```bash
-cc '{"splitting_ops": ["vllm::unified_attention_with_output", "mymodel::custom_attn"]}'
```

Disable splitting (whole-graph compilation):
```bash
-cc '{"splitting_ops": []}'
```

#### `use_inductor_graph_partition` — `bool`, default `False`

When `True`, splitting happens at Inductor codegen time instead of at Dynamo FX level. The complete FX graph goes to Inductor (all passes and fusions run on the full graph), then Inductor partitions at codegen. Advantage: cross-attention-boundary fusions are possible. Disadvantage: heavier compilation.

---

### 11.5 Group 4 — Inductor compilation

#### `backend` — `str`, default `""` (→ `"inductor"`)

- `""` — platform default (Inductor on CUDA/ROCm)
- `"inductor"` — explicitly select Inductor
- `"eager"` — compile but execute eagerly (no Triton; useful for debugging)
- Fully qualified name: `"mypackage.mymodule.my_backend_fn"`

#### `compile_sizes` — `list[int | str]`, default `None`

Specific batch sizes for Inductor to compile with fully static shapes. Accepts integers or `"cudagraph_capture_sizes"` to reuse graph capture sizes.

```bash
-cc '{"compile_sizes": [1, 2, 4, 8, "cudagraph_capture_sizes"]}'
```

#### `compile_ranges_endpoints` — `list[int]`, default `None`

Endpoints defining batch size ranges for Inductor compilation. Range `[a, b]` means one kernel handles all batch sizes from `a` to `b`. `compile_sizes` entries take priority over ranges.

```bash
-cc '{"compile_ranges_endpoints": [8, 64, 256]}'
# → compiles ranges [1,8], [9,64], [65,256], [257, max]
```

#### `inductor_compile_config` — `dict`, default `{}`

Raw kwargs forwarded to Inductor.

```bash
-cc '{"inductor_compile_config": {"max_autotune": true}}'
```

#### `inductor_passes` — `dict[str, str]`, default `{}`

Additional custom Inductor passes, as `{name: qualified_function_name}`.

#### `compile_cache_save_format` — `"binary"` | `"unpacked"`, default `"binary"`

- `"binary"` — single file, multiprocess-safe
- `"unpacked"` — directory for human inspection/debugging (NOT multiprocess-safe)

#### `cache_dir` — `str`, default auto-generated

Directory for compiled Inductor artifacts. If not set, vLLM generates a hash-based path. Set explicitly to share cache across runs.

```bash
-cc '{"cache_dir": "/scratch/my_model_cache"}'
```

---

### 11.6 Group 5 — Multimodal encoder

#### `compile_mm_encoder` — `bool`, default `False`

Apply `torch.compile` to the multimodal encoder (ViT). Currently supported for Qwen2.5-VL, mLLaMA4, and Transformers-backend models with compilable encoders.

#### `cudagraph_mm_encoder` — `bool`, default `False`

CUDA graph capture for the multimodal encoder. Captures the encoder forward for each token budget level. Requires `compile_mm_encoder=True`.

#### `encoder_cudagraph_token_budgets` — `list[int]`, default auto

Fixed token capacity levels for encoder graph capture. If empty, inferred from the model's min/max token budgets as power-of-2 levels.

```bash
-cc '{"encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824]}'
```

#### `encoder_cudagraph_max_vision_items_per_batch` — `int`, default `0` (auto)

Max images or videos per batch during encoder capture.

#### `encoder_cudagraph_max_frames_per_batch` — `int | None`, default `None` (auto)

Max total video frames per batch. Controls `cu_seqlens` buffer size.

---

### 11.7 Group 6 — Debug and IR

#### `debug_dump_path` — `Path | None`, default `None`

If set, dumps intermediate FX graphs, Triton kernel source, and compilation artifacts.

```bash
-cc '{"debug_dump_path": "/tmp/vllm_debug"}'
```

#### `ir_enable_torch_wrap` — `bool`, default auto

Controls vLLM IR custom op wrapping during forward. Defaults to `True` when using Inductor with `VLLM_COMPILE`. Only change if you understand vLLM's IR layer.

---

### 11.8 Group 7 — Fusion passes (`pass_config`)

Set via `{"pass_config": {"field": value}}`.

#### Cross-platform (CUDA + ROCm)

| Field | Default | Effect |
|---|---|---|
| `fuse_norm_quant` | auto | Fuse RMSNorm + quantization |
| `fuse_act_quant` | auto | Fuse SiluMul + quantization |
| `fuse_attn_quant` | auto | Fuse Attention/MLA + quantization |
| `eliminate_noops` | `True` | Remove identity/no-op ops |
| `enable_sp` | auto | Sequence parallelism (requires TP>1; auto-disabled if hidden_size too small) |
| `fuse_gemm_comms` | auto | Async TP: overlap GEMM + AllReduce |
| `fuse_allreduce_rms` | auto | FlashInfer fused AllReduce |
| `enable_qk_norm_rope_fusion` | auto | Fused Q/K RMSNorm + RoPE kernel |
| `fuse_rope_kvcache_cat_mla` | auto | Fused MLA KV cache update with RoPE |

#### ROCm / AITER-specific

| Field | Default | Effect |
|---|---|---|
| `fuse_act_padding` | auto | Fuse RMSNorm + padding (ROCm) |
| `fuse_mla_dual_rms_norm` | auto | Fused paired Q/KV RMS norms in MLA (ROCm) |
| `fuse_rope_kvcache` | auto | Fused RoPE + KV cache update (ROCm/AITER) |
| `fuse_qk_norm_rope_kvcache` | auto | Fused QK-Norm + RoPE + KV cache via AITER HIP kernel. Supersedes `enable_qk_norm_rope_fusion` and `fuse_rope_kvcache` for layers that support it. Auto-enabled at O1+ on ROCm for models with QK-norm (e.g. Qwen3-MoE). |

#### Numeric thresholds

| Field | Type | Effect |
|---|---|---|
| `rope_kvcache_fusion_max_token_num` | `int`, default `256` | Max token count for ROCm RoPE+KVCache fusion. Larger batches use unfused kernels. Also applies to QK-Norm+RoPE+KVCache pass. |
| `fi_allreduce_fusion_max_size_mb` | `float \| None`, default auto | Threshold in MB below which FlashInfer fused AllReduce is used. Auto-selected based on device capability and world size. |
| `sp_min_token_num` | `int \| None`, default auto | Minimum token count above which sequence parallelism is activated. |

### 11.9 Practical CLI examples

```bash
# Default (O2): full graphs for decode, piecewise for mixed
vllm serve meta-llama/Llama-3.1-8B

# Piecewise only — useful for backends without FULL support
vllm serve meta-llama/Llama-3.1-8B -O1

# Full decode only (P/D disaggregation decode instance)
vllm serve meta-llama/Llama-3.1-8B \
  -cc '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

# Custom capture sizes (reduce memory)
vllm serve meta-llama/Llama-3.1-8B \
  -cc '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128]}'

# Disable a fusion pass for debugging
vllm serve meta-llama/Llama-3.1-8B \
  -cc '{"pass_config": {"fuse_norm_quant": false}}'

# Disable ROCm QK-norm+RoPE+KVCache fusion, tune SP threshold
vllm serve meta-llama/Qwen3-MoE \
  -cc '{"pass_config": {"fuse_qk_norm_rope_kvcache": false, "sp_min_token_num": 512}}'

# No compilation (fastest startup, slowest inference)
vllm serve meta-llama/Llama-3.1-8B --enforce-eager

# Inductor graph partition instead of Dynamo FX splitting
vllm serve meta-llama/Llama-3.1-8B \
  -cc '{"use_inductor_graph_partition": true}'

# Add a custom attention op as a split point
vllm serve my-custom-model \
  -cc '{"splitting_ops": ["vllm::unified_attention_with_output", "mymodel::custom_attn"]}'

# Static shape specialization for small batch sizes
vllm serve meta-llama/Llama-3.1-8B \
  -cc '{"compile_sizes": [1, 2, 4, 8]}'

# Dump FX graphs and Triton kernels for debugging
vllm serve meta-llama/Llama-3.1-8B \
  -cc '{"debug_dump_path": "/tmp/vllm_debug"}'
```

---

## 12. End-to-End Flow Diagram

```
Server startup
│
├── VllmConfig.__post_init__()
│     ├── set_splitting_ops_for_v1()     → splitting_ops = _attention_ops
│     └── cudagraph_mode set from -O flag or --compilation-config
│
├── model_runner.__init__()
│     ├── InputBuffers(max_num_reqs, max_num_tokens)   ← one-time static allocation
│     ├── init_attn_backend()            → builds AttentionMetadataBuilder instances
│     │     └── for each builder: get_cudagraph_support()
│     │           → min_cg_support = min across all backends
│     └── compilation_config.resolve_cudagraph_mode_and_sizes(min_cg_support)
│           → may downgrade cudagraph_mode; warns on downgrade
│
├── First forward pass (profile run)
│     └── @support_torch_compile.__call__
│           ├── _mark_dynamic_inputs(num_tokens dim = dynamic)
│           └── TorchCompileWithNoGuardsWrapper.__call__
│                 └── Dynamo traces forward() → FX graph
│                       └── VllmBackend.__call__(graph, example_inputs)
│                             ├── split_graph(graph, splitting_ops)
│                             │     → submod_0..N (alternating split/non-split)
│                             └── PiecewiseCompileInterpreter.run()
│                                   for each non-split submod:
│                                     PiecewiseBackend(submod)  → Inductor compile
│                                     CUDAGraphWrapper(backend) → wrap with graph
│
├── cudagraph_manager.capture()
│     ├── for each PIECEWISE desc (large activations → bigger pool slots first):
│     │     forward_fn(PIECEWISE)
│     │     → each CUDAGraphWrapper sees PIECEWISE → captures its segment
│     └── for each FULL desc:
│           with torch.cuda.graph(graph, pool):
│             forward_fn(NONE)   ← inner wrappers pass through; outer captures all
│           self.graphs[desc] = graph
│
└── Each inference step
      │
      ├── dispatch_cg_and_sync_dp()   ← CPU dict lookup, no GPU work
      │     → BatchExecutionDescriptor(cg_mode, num_tokens_padded, num_reqs_padded)
      │
      ├── prepare_inputs(batch_desc)  ← write into static InputBuffers
      │     → overwrite [0..num_tokens_actual] with real data
      │     → overwrite [num_tokens_actual..num_tokens_padded] with padding
      │     → leave [num_tokens_padded..max] stale (never read by this graph)
      │
      ├── [FULL]
      │     run_fullgraph(desc) → self.graphs[desc].replay()
      │     one cudaGraphLaunch; no Python per-op overhead
      │
      ├── [PIECEWISE]
      │     set_forward_context(cudagraph_runtime_mode=PIECEWISE)
      │     run_pw_graph(model, inputs)
      │     → model() → split_gm() → each CUDAGraphWrapper.replay() per segment
      │                             → attention subgraph runs eagerly between segments
      │
      └── [NONE]
            set_forward_context(cudagraph_runtime_mode=NONE)
            model(**model_inputs)   fully eager
```

---

## 13. How to Add a New Attention Backend

### Step 1: Declare support level

```python
class MyAttentionMetadataBuilder(AttentionMetadataBuilder):
    # Static: unconditional
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    # OR dynamic: depends on hardware/config
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        if some_condition(vllm_config):
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
```

### Step 2: Pre-allocate static GPU buffers (FULL support)

Any tensor whose GPU address must be stable must be allocated in `__init__`, not in `build()`:

```python
def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
    super().__init__(kv_cache_spec, layer_names, vllm_config, device)
    self.use_full_cuda_graph = vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    if self.use_full_cuda_graph:
        max_batch = vllm_config.scheduler_config.max_num_seqs
        self.my_static_buffer = torch.zeros(max_batch, dtype=torch.int32, device=device)
```

### Step 3: Implement `build_for_cudagraph_capture()`

Override to produce capture-safe metadata — avoid slow codepaths and invalid memory accesses:

```python
def build_for_cudagraph_capture(self, common_attn_metadata):
    meta = self.build(0, common_attn_metadata)
    meta.seq_lens.fill_(1)                        # avoid long-sequence codepaths
    common_attn_metadata.query_start_loc.zero_()  # avoid invalid kernel accesses
    return meta
```

### Step 4: Write into static buffers in `build()`

```python
def build(self, common_prefix_len, common_attn_metadata, ...) -> MyMetadata:
    n = common_attn_metadata.num_reqs
    if self.use_full_cuda_graph:
        self.my_static_buffer[:n].copy_(compute_per_step_data())
        self.my_static_buffer[n:] = 0          # zero unused slots (prevent stale data)
        static_buf_view = self.my_static_buffer[:n]
    else:
        static_buf_view = compute_per_step_data()   # dynamic (non-graph) path
    return MyMetadata(..., my_field=static_buf_view)
```

### Step 5: Register op in `splitting_ops` (PIECEWISE support)

Add to `_attention_ops` in `CompilationConfig` (`compilation.py:764`) and/or decorate the Python function with `@eager_break_during_capture` for the breakable path.

---

## 14. Quick Reference: Mode Selection Matrix

| Condition | Recommended mode | Notes |
|---|---|---|
| Backend = FA3 or ROCm, standard serving | `FULL_AND_PIECEWISE` (default) | Best throughput; no configuration needed |
| Backend = FA2 or FlashInfer SM90 | `FULL_AND_PIECEWISE` (auto-downgraded) | System adjusts automatically; warn logged |
| P/D disaggregation decode instance | `FULL_DECODE_ONLY` | Save memory; prefill runs eager (rare on decode instance) |
| Heavy chunked prefill workload | `FULL_AND_PIECEWISE` | Mixed batches use PIECEWISE, not eager |
| Debugging / custom backend with `NEVER` support | `PIECEWISE` | No full graphs, but Triton fusions active |
| Dynamic control flow that breaks tracing | `-O0` / `NONE` | No compilation overhead |
| Custom model with new attention op | `PIECEWISE` + add op to `splitting_ops` | Safe starting point |
| DeepSeekV4 / KimiK3 / MiniMaxM3 / Inkling | `PIECEWISE` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | Auto-enabled; avoids torch.compile issues |
| Spec-decode + FlashInfer SM90 | `PIECEWISE` (auto-downgraded from `FULL_AND_PIECEWISE`) | System warns and adjusts |
| Spec-decode + FlashInfer SM120 | `FULL_AND_PIECEWISE` | TRT-LLM XQA supports uniform-batch spec-decode |
| Need to dump tensors at inference | Use PIECEWISE eager sections or pinned buffers inside FULL capture | See §10 |
