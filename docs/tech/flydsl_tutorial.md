# FlyDSL Tutorial

FlyDSL is a Python DSL and MLIR-based compiler for writing high-performance GPU kernels on AMD GPUs (MI300X / MI350 / gfx1250). It gives you explicit control over data layout, tiling, memory access patterns, and thread/wave scheduling — the things that determine kernel performance at the hardware level.

---

## Table of Contents

1. [What Is FlyDSL and When to Use It](#1-what-is-flydsl-and-when-to-use-it)
2. [Installation and Environment](#2-installation-and-environment)
3. [The Layout System — Core Abstraction](#3-the-layout-system--core-abstraction)
   - 3.1 [Layout Construction](#31-layout-construction)
   - 3.2 [Coordinate / Index Mapping](#32-coordinate--index-mapping)
   - 3.3 [Layout Algebra: Divide](#33-layout-algebra-divide)
   - 3.4 [Layout Algebra: Product](#34-layout-algebra-product)
   - 3.5 [Structural Operations](#35-structural-operations)
4. [Writing Your First Kernel](#4-writing-your-first-kernel)
   - 4.1 [Hello Kernel: Element-wise Scale](#41-hello-kernel-element-wise-scale)
   - 4.2 [Vectorized Loads (128-bit)](#42-vectorized-loads-128-bit)
5. [Parameter Types](#5-parameter-types)
6. [Thread and Block Hierarchy](#6-thread-and-block-hierarchy)
7. [Control Flow](#7-control-flow)
   - 7.1 [Compile-Time vs Runtime Branches](#71-compile-time-vs-runtime-branches)
   - 7.2 [Runtime Branches with Side Effects](#72-runtime-branches-with-side-effects)
   - 7.3 [Frontend Restrictions](#73-frontend-restrictions)
8. [Typed Arithmetic](#8-typed-arithmetic)
   - 8.1 [Scalar Types](#81-scalar-types)
   - 8.2 [Vector Type](#82-vector-type)
   - 8.3 [Operator Reference](#83-operator-reference)
9. [Data Movement — Copy Atoms](#9-data-movement--copy-atoms)
   - [Copy Atom Types](#copy-atom-types)
   - 9.1 [Simple Element-wise Copy](#91-simple-element-wise-copy)
   - 9.2 [AMD Buffer Resource (Preferred for Global Loads)](#92-amd-buffer-resource-preferred-for-global-loads)
   - 9.3 [TiledCopy (2D Partitioned Copy)](#93-tiledcopy-2d-partitioned-copy)
   - 9.4 [gfx1250 TDM Async DMA](#94-gfx1250-tdm-async-dma)
10. [Shared Memory (LDS)](#10-shared-memory-lds)
    - 10.1 [SharedAllocator (Preferred)](#101-sharedallocator-preferred)
    - 10.2 [Ping-Pong LDS for Pipelining](#102-ping-pong-lds-for-pipelining)
11. [MFMA and WMMA — Matrix Math](#11-mfma-and-wmma--matrix-math)
    - 11.1 [gfx942/gfx950: MFMA (wave64)](#111-gfx942gfx950-mfma-wave64)
    - 11.2 [gfx1250: WMMA (wave32)](#112-gfx1250-wmma-wave32)
    - 11.3 [Building Fragments](#113-building-fragments)
12. [Reductions](#12-reductions)
    - 12.1 [Warp Reduction](#121-warp-reduction)
    - 12.2 [Block Reduction](#122-block-reduction)
13. [Software Pipelining — Loop-Carried State](#13-software-pipelining--loop-carried-state)
14. [End-to-End Example: Naive GEMM](#14-end-to-end-example-naive-gemm)
15. [End-to-End Example: Tiled GEMM with LDS and MFMA](#15-end-to-end-example-tiled-gemm-with-lds-and-mfma)
16. [Fast Launch with `_run_compiled`](#16-fast-launch-with-_run_compiled)
17. [Ahead-of-Time (AOT) Compilation](#17-ahead-of-time-aot-compilation)
    - [Running AOT](#running-aot)
    - [AOT Environment Variables](#aot-environment-variables)
    - [Architecture from CSV](#architecture-from-csv)
    - [Verifying AOT Cache Hits](#verifying-aot-cache-hits)
18. [Using the Public aiter API](#18-using-the-public-aiter-api)
    - [Check Availability](#check-availability)
    - 18.1 [HGEMM](#181-hgemm)
    - 18.2 [MoE Stage 1 + Stage 2](#182-moe-stage-1--stage-2)
    - 18.3 [Fused QK Norm + RoPE + FP8 Quant](#183-fused-qk-norm--rope--fp8-quant)
    - 18.4 [MLA Reduce](#184-mla-reduce)
    - 18.5 [Flash Attention](#185-flash-attention)
19. [Autotuning](#19-autotuning)
20. [Debugging and IR Inspection](#20-debugging-and-ir-inspection)
    - [IR Dump](#ir-dump)
    - [Printf Debugging](#printf-debugging)
    - [Source-to-Assembly Debug Info (ATT Profiling)](#source-to-assembly-debug-info-att-profiling)
    - [Optimization Level](#optimization-level)
    - [Architecture Utilities](#architecture-utilities)
21. [Common Pitfalls](#21-common-pitfalls)
22. [Example: Activation Functions (SiLU, SWiGLU, exp2/rcp)](#22-example-activation-functions-silu-swiglu-exp2rcp)
    - [Sigmoid via exp2/rcp](#sigmoid-via-exp2rcp)
    - [SiLU batch (vectorized, for MoE epilogues)](#silu-batch-vectorized-for-moe-epilogues)
    - [SWiGLU (clamp + Gelu-like gate)](#swiglu-clamp--gelu-like-gate)
    - [Compile-time dispatch between activation variants](#compile-time-dispatch-between-activation-variants)
23. [Example: Fused SWiGLU Kernel (bf16, TiledCopy)](#23-example-fused-swiglu-kernel-bf16-tiledcopy)
24. [Example: Fused Activation + MX Quantization](#24-example-fused-activation--mx-quantization)
    - [sorted_ids packing](#sorted_ids-packing)
    - [FP4 quantization: amax → E8M0 scale → nibble packing](#fp4-quantization-amax--e8m0-scale--nibble-packing)
    - [FP8 quantization: two values per call](#fp8-quantization-two-values-per-call)
    - [Warp-level amax for block scaling](#warp-level-amax-for-block-scaling)
25. [Example: MoE Top-K Reduction](#25-example-moe-top-k-reduction)
26. [Example: Buffer Tensor Scalar View / Tiled Store Patterns](#26-example-buffer-tensor-scalar-view--tiled-store-patterns)
    - [Scalar load / store via buffer tensor](#scalar-load--store-via-buffer-tensor)
    - [bf16 row builder from raw i64 pointer](#bf16-row-builder-from-raw-i64-pointer)
    - [Tiled store: f32 register list → bf16 global](#tiled-store-f32-register-list--bf16-global)
    - [Important: avoid `from __future__ import annotations`](#important-avoid-from-__future__-import-annotations)
27. [Example: Testing FlyDSL Kernels](#27-example-testing-flydsl-kernels)
    - [ptr_arg + _run_compiled (hot-path test pattern)](#ptr_arg--_run_compiled-hot-path-test-pattern)
    - [Reference comparison with tolerances](#reference-comparison-with-tolerances)
    - [Benchmark utility](#benchmark-utility)
    - [sorted_ids packing/unpacking (MoE test helper)](#sorted_ids-packingunpacking-moe-test-helper)
    - [Parametrized pytest example](#parametrized-pytest-example)
    - [AOT cache-miss detection in tests](#aot-cache-miss-detection-in-tests)
28. [Reference: FlyDSL vs Triton vs Gluon](#28-reference-flydsl-vs-triton-vs-gluon)
29. [Quick-Reference Index](#29-quick-reference-index)

---

## 1. What Is FlyDSL and When to Use It

FlyDSL compiles Python kernel functions to AMD GPU ISA via an MLIR pipeline:

```
Python (@flyc.kernel / @flyc.jit)
  -> AST rewriting    (for/if -> scf.for / scf.if)
  -> MLIR tracing     (Fly dialect + gpu/arith/scf/memref/vector ops)
  -> MlirCompiler     (fly-layout-lowering -> convert-fly-to-rocdl -> LLVM -> HSACO)
  -> JITCFunction     (ExecutionEngine wrapper, loaded onto device)
```

**Target architectures:**

| Arch | GPU | Wave size | LDS | Notes |
|---|---|---|---|---|
| gfx942 | MI300X (CDNA3) | 64 | 64 KB | MFMA |
| gfx950 | MI350 (CDNA4) | 64 | 160 KB | MFMA, scaled MFMA |
| gfx1250 | RDNA4-family | 32 | 320 KB | WMMA, TDM async DMA |

FlyDSL shines when you need maximum control: precise data layouts in registers and LDS, explicit tiling algebra, and direct access to hardware features like MFMA/WMMA and AMD buffer resources. Choose it over Triton when kernel performance depends on layout choices that Triton's auto-tiling cannot express.

---

## 2. Installation and Environment

```bash
# FlyDSL is installed as a Python package (usually editable):
pip show flydsl

# Check availability for the current GPU:
python -c "from aiter.ops.flydsl.utils import is_flydsl_available; print(is_flydsl_available())"
```

**Key environment variables:**

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_DUMP_IR` | false | Dump MLIR at each pass |
| `FLYDSL_DUMP_DIR` | `./dumps` | Dump output directory |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | true | Disk cache (in-memory always on) |
| `FLYDSL_RUNTIME_CACHE_DIR` | `~/.flydsl/cache` | Cache location |
| `FLYDSL_COMPILE_OPT_LEVEL` | 2 | Optimization level 0–3 |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | false | DWARF debug info for ATT profiling |

```bash
# Force-clear cache when you change C++ passes or non-closure helpers:
FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_kernel.py
# or:
rm -rf ~/.flydsl/cache
```

---

## 3. The Layout System — Core Abstraction

Every FlyDSL kernel is built on **layouts**. A layout is a `(Shape, Stride)` pair that maps logical coordinates to linear memory indices. This abstraction lets you describe any regular data arrangement — row-major, column-major, tiled, interleaved — and compose them algebraically.

### 3.1 Layout Construction

```python
import flydsl.expr as fx

# 8x16 column-major: element (row, col) at index = row*1 + col*8
col_major = fx.make_layout((8, 16), (1, 8))

# 8x16 row-major: element (row, col) at index = row*16 + col*1
row_major = fx.make_layout((8, 16), (16, 1))

# Flat 128-element vector
flat = fx.make_layout(128, 1)

# Identity layout (strides = prefix products of shape)
identity = fx.make_identity_layout((8, 16))

# Nested shape for hierarchical tiling: 4 tiles of 8 each
nested = fx.make_layout((4, 8), (8, 1))   # 32 elements, 4 groups of 8

# Query layout properties
total_elems = fx.size(row_major)     # 128
shape       = fx.get_shape(row_major)
stride      = fx.get_stride(row_major)
```

### 3.2 Coordinate / Index Mapping

```python
layout = fx.make_layout((8, 16), (1, 8))   # 8x16 column-major

# Coordinate -> linear index: crd2idx((3,5)) = 3*1 + 5*8 = 43
idx   = fx.crd2idx((3, 5), layout)   # -> 43

# Linear index -> coordinate: idx2crd(43) -> (3, 5)
coord = fx.idx2crd(43, layout)

# Inside a kernel, these work on runtime values:
@flyc.kernel
def test_mapping(A: fx.Tensor, N: fx.Constexpr[int]):
    tid    = gpu.thread_id("x")
    layout = fx.make_layout((4, 64), (64, 1))   # 4 waves x 64 lanes
    coord  = fx.idx2crd(tid, layout)
    wave   = fx.get(coord, 0)   # which wave this thread belongs to
    lane   = fx.get(coord, 1)   # lane within wave
```

### 3.3 Layout Algebra: Divide

`fx.logical_divide(layout, divisor)` partitions a layout into tile + rest, exposing two modes: the within-tile coordinate and the tile index.

```python
# 1D: divide 1024-element flat layout into 256-element blocks
tensor_layout = fx.make_layout(1024, 1)
tile_layout   = fx.make_layout(256, 1)
divided       = fx.logical_divide(tensor_layout, tile_layout)
# divided has shape ((256), (4)) — 256 elements per tile, 4 tiles

# Select block bid's tile (None = keep that mode free)
block_tile = fx.slice(divided, (None, bid))   # shape = (256,)

# 2D: divide (M, N) matrix into (BLOCK_M, BLOCK_N) tiles
matrix    = fx.make_layout((M, N), (N, 1))
tile_size = fx.make_layout((BLOCK_M, BLOCK_N), (BLOCK_N, 1))
tiled     = fx.logical_divide(matrix, tile_size)

# Variants:
fx.zipped_divide(layout, divisor)   # interleaved modes
fx.tiled_divide(layout, divisor)    # hierarchical result
fx.flat_divide(layout, divisor)     # flattened result
```

### 3.4 Layout Algebra: Product

Products combine two layouts, expanding the dimensionality.

```python
# raked_product: interleaved access — common for thread/value TiledCopy layouts
# 4 threads, each handling 8 values
thr_layout = fx.make_layout((4, 1), (1, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
tv_layout  = fx.raked_product(thr_layout, val_layout)
# thread t accesses elements at positions interleaved across the 4 threads

# blocked_product: threads packed together (non-interleaved)
blocked = fx.blocked_product(thr_layout, val_layout)

# logical_product: basic mode concatenation
joined  = fx.logical_product(layout_a, layout_b)

# Other variants:
fx.zipped_product(a, b)
fx.tiled_product(a, b)
fx.flat_product(a, b)
```

### 3.5 Structural Operations

```python
# Coalesce: merge adjacent modes with compatible strides
simplified = fx.coalesce(layout)

# Compose: chain mappings A(B(x))
result = fx.composition(layout_a, layout_b)

# Structural manipulation
mode_0    = fx.get(int_tuple, 0)                   # get i-th mode
selected  = fx.select(int_tuple, indices=[0, 2])   # pick modes
grouped   = fx.group(int_tuple, begin=1, end=3)    # nest modes into subtuple
appended  = fx.append(base, elem)
prepended = fx.prepend(base, elem)

# Type recast: adjust layout for type-width change (e.g., f16 -> f8)
f8_layout = fx.recast_layout(f16_layout, old_bits=16, new_bits=8)
```

---

## 4. Writing Your First Kernel

### 4.1 Hello Kernel: Element-wise Scale

The minimal FlyDSL kernel has a `@flyc.kernel` (GPU function) and a `@flyc.jit` (host launch wrapper). FlyDSL auto-converts `torch.Tensor` to `fx.Tensor` via DLPack at the `@flyc.jit` boundary.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

@flyc.kernel
def scale_kernel(
    A:     fx.Tensor,           # input — auto-converted from torch.Tensor
    B:     fx.Tensor,           # output
    N:     fx.Constexpr[int],   # compile-time problem size
    scale: fx.Float32,          # runtime scalar multiplier
):
    BLOCK = 256

    tid = gpu.thread_id("x")   # thread index in block
    bid = gpu.block_id("x")    # block index in grid

    # Step 1: partition the flat tensor into 256-element per-block tiles
    tA = fx.logical_divide(A, fx.make_layout(BLOCK, 1))
    tB = fx.logical_divide(B, fx.make_layout(BLOCK, 1))

    # Step 2: select this block's tile
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))

    # Step 3: each thread owns one element — further divide
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))

    # Step 4: allocate register memory and copy atom
    copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA   = fx.make_rmem_tensor(1, fx.Float32)
    rB   = fx.make_rmem_tensor(1, fx.Float32)

    # Step 5: global -> register
    fx.copy_atom_call(copy, fx.slice(tA, (None, tid)), rA)

    # Step 6: compute
    v = fx.Vector(fx.memref_load_vec(rA)) * scale
    fx.memref_store_vec(v, rB)

    # Step 7: register -> global
    fx.copy_atom_call(copy, rB, fx.slice(tB, (None, tid)))


@flyc.jit
def scale(
    A:     fx.Tensor,
    B:     fx.Tensor,
    N:     fx.Constexpr[int],
    scale: fx.Float32,
    stream: fx.Stream = fx.Stream(None),
):
    scale_kernel(A, B, N, scale).launch(
        grid=(N // 256,), block=(256,), stream=stream
    )


# Host usage:
import torch
A = torch.randn(4096, device="cuda", dtype=torch.float32)
B = torch.empty_like(A)
scale(A, B, 4096, 2.0)   # B = A * 2.0
assert torch.allclose(B, A * 2.0)
```

### 4.2 Vectorized Loads (128-bit)

Loading 4 floats per thread (128-bit = 4 × 32 b) dramatically improves memory bandwidth utilization:

```python
@flyc.kernel
def scale_vec_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    N: fx.Constexpr[int],
    scale: fx.Float32,
):
    VEC   = 4
    BLOCK = 256   # 256 threads × 4 floats = 1024 floats per block

    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    # Divide tensor into per-block 1024-element tiles, then into per-thread 4-element chunks
    tA = fx.logical_divide(A, fx.make_layout(BLOCK * VEC, 1))
    tA = fx.slice(tA, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(VEC, 1))

    tB = fx.logical_divide(B, fx.make_layout(BLOCK * VEC, 1))
    tB = fx.slice(tB, (None, bid))
    tB = fx.logical_divide(tB, fx.make_layout(VEC, 1))

    # 128-bit copy atom (4 × f32)
    copy = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float32)
    rA   = fx.make_rmem_tensor(VEC, fx.Float32)
    rB   = fx.make_rmem_tensor(VEC, fx.Float32)

    # Load 4 contiguous floats
    fx.copy_atom_call(copy, fx.slice(tA, (None, tid)), rA)

    # Scale the vector
    v = fx.Vector(fx.memref_load_vec(rA)) * scale
    fx.memref_store_vec(v, rB)

    # Store 4 floats
    fx.copy_atom_call(copy, rB, fx.slice(tB, (None, tid)))


@flyc.jit
def scale_vec(
    A: fx.Tensor,
    B: fx.Tensor,
    N: fx.Constexpr[int],
    scale: fx.Float32,
    stream: fx.Stream = fx.Stream(None),
):
    scale_vec_kernel(A, B, N, scale).launch(
        grid=(N // (256 * 4),), block=(256,), stream=stream
    )
```

---

## 5. Parameter Types

| FlyDSL type | Python host type | Behavior |
|---|---|---|
| `fx.Tensor` | `torch.Tensor` | Auto-converted via DLPack |
| `fx.Constexpr[int]` | Python `int` | Baked into IR — different values → different kernels |
| `fx.Int32` | Python `int` | Runtime i32 |
| `fx.Int64` | Python `int` | Runtime i64 |
| `fx.Float32` | Python `float` | Runtime f32 |
| `fx.Float16` | Python `float` | Runtime f16 |
| `fx.BFloat16` | Python `float` | Runtime bf16 |
| `fx.Stream` | `torch.cuda.Stream` or `None` | `fx.Stream(None)` = default stream |
| `fx.Pointer` | raw pointer | Use `ptr_arg(tensor)` for hot-path dispatch |

**`fx.Constexpr` example:** tile sizes and dtype flags make good constexprs — each distinct combination compiles a separate binary, which is cached on disk.

```python
@flyc.jit
def gemm_op(
    A:       fx.Tensor,
    B:       fx.Tensor,
    M:       fx.Int32,           # runtime: different every call, no recompile
    N:       fx.Int32,           # runtime
    TILE_M:  fx.Constexpr[int],  # compile-time: 64 vs 128 = two different binaries
    TILE_N:  fx.Constexpr[int],
    stream:  fx.Stream = fx.Stream(None),
):
    ...
```

---

## 6. Thread and Block Hierarchy

```python
from flydsl.expr import gpu

# Preferred spelling (current):
tid_x = gpu.thread_id("x")   # i64
tid_y = gpu.thread_id("y")
bid_x = gpu.block_id("x")    # i64
bid_y = gpu.block_id("y")

# Legacy spelling found in older kernels (still works):
tid_x = gpu.thread_idx.x
bid_x = gpu.block_idx.x

# Synchronize all threads in a block (ds_barrier / __syncthreads):
gpu.barrier()
```

**Wave decomposition for gfx942/gfx950 (wave64):**

```python
@flyc.kernel
def kernel(A: fx.Tensor):
    tid = gpu.thread_id("x")

    # Decompose a 256-thread block into 4 waves × 64 lanes
    wave_layout = fx.make_layout((4, 64), (64, 1))
    coord       = fx.idx2crd(tid, wave_layout)
    wave_id     = fx.get(coord, 0)   # 0..3
    lane_id     = fx.get(coord, 1)   # 0..63

    # Decompose a 512-thread block into 8 waves × 64 lanes
    wave_layout8 = fx.make_layout((8, 64), (64, 1))
    coord8       = fx.idx2crd(tid, wave_layout8)
    wave8        = fx.get(coord8, 0)   # 0..7
    lane8        = fx.get(coord8, 1)   # 0..63
```

**gfx1250 uses wave32.** Replace `64` with `32` everywhere and use `fx.make_layout((N_WAVES, 32), (32, 1))`.

---

## 7. Control Flow

### 7.1 Compile-Time vs Runtime Branches

```python
from flydsl.expr import range_constexpr, const_expr

@flyc.kernel
def kernel(
    A:       fx.Tensor,
    trans_b: fx.Constexpr[bool],   # compile-time flag
    N:       fx.Constexpr[int],
):
    # Compile-time unrolled loop — emitted inline in IR, no scf.for:
    for i in range_constexpr(8):
        ...

    # Runtime loop — lowered to scf.for:
    for k in range(runtime_count):
        ...

    # Compile-time branch — only one path emitted per compiled kernel:
    if const_expr(trans_b):
        # transposed load path
        ...
    else:
        # normal load path
        ...

    # Runtime comparison — typed DSL value → scf.if:
    tid  = gpu.thread_id("x")
    lane = tid % fx.Int64(64)
    if lane == fx.Int64(0):
        # only lane 0 of each wave executes this
        ...

    # Predicated select (no branch overhead):
    in_range = lane < fx.Int64(32)
    val      = in_range.select(good_val, zero_val)
```

**Do NOT wrap GPU runtime values in `const_expr`:**

```python
lane = gpu.thread_id("x") % fx.Int64(64)

# WRONG — lane is a runtime SSA value:
if const_expr(lane == fx.Int64(0)):  # will misbehave
    ...

# CORRECT:
if lane == fx.Int64(0):
    ...
```

### 7.2 Runtime Branches with Side Effects

For branches that define values used after the block, or have complex bodies with loop-carried state, wrap them in a local `@flyc.jit`:

```python
@flyc.kernel
def kernel(A: fx.Tensor, B: fx.Tensor, write_b: fx.Int32):
    tid    = gpu.thread_id("x")
    result = ...   # some computation

    def write_path():
        # write result to B
        fx.copy_atom_call(copy, result_reg, fx.slice(B, (None, tid)))

    def skip_path():
        pass   # no-op

    @flyc.jit
    def dispatch(cond):
        if cond:
            write_path()
        else:
            skip_path()

    dispatch(write_b != fx.Int32(0))
```

### 7.3 Frontend Restrictions

1. **Do not define a value inside `if/else` and use it after the block:**
   ```python
   # PROBLEMATIC:
   if cond:
       dst = a
   else:
       dst = b
   use(dst)   # dst is not in scope the way Python expects

   # CORRECT pattern:
   dst = cond.select(a, b)
   use(dst)
   ```

2. **Do not mutate captured outer variables inside nested helpers** — pass explicitly:
   ```python
   def bad():
       acc = fx.Float32(0.0)
       def helper():
           acc = acc + fx.Float32(1.0)   # WRONG: outer acc not mutated
       helper()

   def good():
       acc = fx.Float32(0.0)
       def helper(x):
           return x + fx.Float32(1.0)
       acc = helper(acc)   # CORRECT: returned and reassigned
   ```

3. **Single exit — no early `return` or `return` inside branches:**
   ```python
   # AVOID:
   if cond:
       return v0
   return v1

   # PREFER:
   out = cond.select(v0, v1)
   return out
   ```

---

## 8. Typed Arithmetic

Use FlyDSL's typed numeric types with Python operators. Avoid raw `arith.*` dialect calls except when explicit fastmath flags are required.

### 8.1 Scalar Types

```python
# Integer constants
c42    = fx.Int64(42)
mask   = fx.Int32(0xFF)

# Float constants
pi     = fx.Float32(3.14159)
half   = fx.Float16(0.5)
one_bf = fx.BFloat16(1.0)

# Type conversions
i64   = fx.Int64(some_i32_value)       # widen i32 -> i64
i32   = fx.Int32(some_index_value)     # narrow index -> i32
bf16  = f32_value.to(fx.BFloat16)     # truncate f32 -> bf16
f32   = bf16_value.to(fx.Float32)     # extend bf16 -> f32
```

### 8.2 Vector Type

```python
Vec = fx.Vector

# Wrap a raw vector from memref load
v = Vec(fx.memref_load_vec(rA))          # vector<Nxf32>

# Scalar indexing
elem = v[0]                              # Float32 scalar

# Bitcast between numeric types
v_i32 = Vec(raw_f32_vec).bitcast(fx.Int32)   # f32 vec -> i32 vec

# Splat a constant to all lanes
zeros = Vec.filled(4, 0.0, fx.Float32)
ones  = Vec.filled(8, 1.0, fx.BFloat16)

# Build from individual scalars
v4 = Vec.from_elements([a, b, c, d], fx.Float32)

# Element-wise arithmetic
result = v * scale            # multiply
result = v + bias             # add
result = -v                   # negate

# Type cast
bf16_v = v.to(fx.BFloat16)

# Max / ReLU
relu_v = v.maximumf(zeros)

# Comparison -> bool vector
mask = v < zeros              # vector<Nxi1>

# Select (masked / predicated)
abs_v = (v < zeros).select(-v, v)   # abs value

# Store back
fx.memref_store_vec(result, rB)
```

### 8.3 Operator Reference

| Operation | Preferred syntax | Notes |
|---|---|---|
| Add | `a + b` | |
| Multiply | `a * b` | |
| Subtract | `a - b` | |
| Negate | `-a` | |
| Max | `a.maximumf(b)` | Use for ReLU |
| Compare | `a < b`, `a == b`, `a > b` | Returns bool (scalar or vector) |
| Select | `cond.select(true_val, false_val)` | No branch overhead |
| Abs | `(v < zero).select(-v, v)` | `arith.absf` does NOT exist |
| Type cast | `v.to(fx.BFloat16)` | |
| Index cast | `fx.Int64(v)`, `fx.Int32(v)` | |
| Wave shuffle | `val.shuffle_xor(fx.Int32(n), fx.Int32(64))` | wave64 |

---

## 9. Data Movement — Copy Atoms

Copy atoms are the primitive for moving data between address spaces. FlyDSL provides several atom types and two dispatch modes: `fx.copy_atom_call` (single instance) and `fx.copy` (tiled/partitioned).

### Copy Atom Types

| Type | Width | Use case |
|---|---|---|
| `fx.UniversalCopy32b()` | 32 b | 1 × f32 |
| `fx.UniversalCopy(64)` | 64 b | 2 × f32 |
| `fx.UniversalCopy(128)` | 128 b | 4 × f32 — most common |
| `fx.rocdl.BufferCopy128b()` | 128 b | AMD V# buffer load, OOB-safe, gfx942/gfx950 |
| `fx.rocdl.make_tdm_atom(...)` | tile-wide | gfx1250 async TDM DMA |

### 9.1 Simple Element-wise Copy

```python
@flyc.kernel
def elementwise_copy(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    BLOCK = 256
    tid   = gpu.thread_id("x")
    bid   = gpu.block_id("x")

    # Divide -> block -> thread
    tA = fx.slice(fx.logical_divide(A, fx.make_layout(BLOCK, 1)), (None, bid))
    tB = fx.slice(fx.logical_divide(B, fx.make_layout(BLOCK, 1)), (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))

    copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA   = fx.make_rmem_tensor(1, fx.Float32)

    fx.copy_atom_call(copy, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copy, rA, fx.slice(tB, (None, tid)))
```

### 9.2 AMD Buffer Resource (Preferred for Global Loads)

The AMD buffer resource (`V#`) descriptor provides hardware OOB checking and is the preferred way to read global memory on gfx942/gfx950.

```python
@flyc.kernel
def gemm_row(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
             M: fx.Int32, N: fx.Int32, K: fx.Int32):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    # Build a buffer-resource view over the global tensor
    bufA = fx.rocdl.make_buffer_tensor(A)
    bufB = fx.rocdl.make_buffer_tensor(B)

    # Overlay a 2D layout (row-major M×K and K×N)
    tA = fx.make_view(fx.get_iter(bufA), fx.make_layout((M, K), (K, 1)))
    tB = fx.make_view(fx.get_iter(bufB), fx.make_layout((K, N), (N, 1)))

    # 128-bit buffer load atom (4 × f32)
    copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    rA   = fx.make_rmem_tensor(4, fx.Float32)

    # Load this thread's 4 floats from row bid of A
    row_offset = bid
    fx.copy_atom_call(copy, fx.slice(tA, (row_offset, tid * fx.Int64(4))), rA)
```

### 9.3 TiledCopy (2D Partitioned Copy)

`TiledCopy` handles a 2D tile where threads cooperate. Each thread gets a partition of the source and destination.

```python
@flyc.kernel
def tiled_copy_kernel(A: fx.Tensor, B: fx.Tensor):
    BLOCK_M, BLOCK_K = 32, 128

    tid = gpu.thread_id("x")   # 0..127 (128-thread block)

    bufA  = fx.rocdl.make_buffer_tensor(A)
    src   = fx.make_view(fx.get_iter(bufA), fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    bufB  = fx.rocdl.make_buffer_tensor(B)
    dst   = fx.make_view(fx.get_iter(bufB), fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))

    # Thread layout: 32 threads across M, 1 thread across N
    thr_layout = fx.make_layout((32, 1), (1, 1))
    # Value layout: 1 value across M, 4 values per thread across K (128-bit)
    val_layout = fx.make_layout((1, 4), (1, 1))

    copy_atom  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    layout_tv  = fx.raked_product(thr_layout, val_layout)
    tile_mn    = fx.make_tile(32, 4)
    tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)

    # Each thread gets its view
    thr_copy = tiled_copy.get_slice(tid)
    src_part = thr_copy.partition_S(src)
    dst_part = thr_copy.partition_D(dst)
    frag     = fx.make_fragment_like(src_part)

    # Execute: src -> fragment (loads), fragment -> dst (stores)
    fx.copy(copy_atom, src_part, frag)
    fx.copy(copy_atom, frag, dst_part)
```

### 9.4 gfx1250 TDM Async DMA

gfx1250's Tensor DMA (TDM) engine performs whole-tile asynchronous DMA between global memory and LDS. Use a raw virtual address, not a buffer resource.

```python
@flyc.kernel
def tdm_example(A: fx.Tensor, N: fx.Constexpr[int]):
    M_TILE, K_TILE = 64, 64

    # Allocate LDS for the tile
    lds   = fx.SharedAllocator().allocate(fx.Array[fx.Float16, M_TILE * K_TILE]).peek()
    lds2d = fx.make_view(lds.ptr, fx.make_layout((M_TILE, K_TILE), (K_TILE, 1)))

    # Raw VA view of global tensor
    g2d = fx.make_view(fx.get_iter(A), fx.make_layout((M_TILE, K_TILE), (K_TILE, 1)))

    # Create TDM atom: tile shape [M_TILE, K_TILE], issued by 4 warps cooperatively
    atom = fx.rocdl.make_tdm_atom(g2d, [M_TILE, K_TILE], num_warps=4)

    # Issue async copy: global -> LDS (returns immediately)
    fx.copy_atom_call(atom, g2d, lds2d)

    # Do other work here while DMA runs...

    # Wait for DMA to complete (barrier group 0)
    fx.rocdl.tdm_ops.tensor_wait(0)

    # LDS is now ready to use
    # ...

    # K-loop: advance the atom's base pointer to the next K tile
    K_STRIDE_BYTES = K_TILE * 2   # float16 = 2 bytes
    atom = fx.rocdl.advance_tdm_atom(atom, K_STRIDE_BYTES)
    fx.copy_atom_call(atom, g2d, lds2d)
    fx.rocdl.tdm_ops.tensor_wait(0)
```

---

## 10. Shared Memory (LDS)

### 10.1 SharedAllocator (Preferred)

Declare LDS as an `@fx.struct` of `fx.Array` fields. The compiler allocates the LDS global automatically — no `finalize()` call needed.

```python
import flydsl.expr as fx

BLOCK_M = 64
BLOCK_K = 64

@fx.struct
class Smem:
    a: fx.Array[fx.Float16, BLOCK_M * BLOCK_K]   # A-tile in LDS
    b: fx.Array[fx.Float16, BLOCK_K * BLOCK_K]   # B-tile in LDS

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                M: fx.Int32, N: fx.Int32, K: fx.Int32):
    # Allocate all LDS via struct
    lds   = fx.SharedAllocator().allocate(Smem).peek()

    # Create 2D row-major views over each field
    lds_a = lds.a.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    lds_b = lds.b.view(fx.make_layout((BLOCK_K, BLOCK_K), (BLOCK_K, 1)))

    # ... load A and B tiles from global to LDS via copy atoms ...

    gpu.barrier()   # all threads must finish writing before reading

    # ... MFMA on lds_a and lds_b ...
```

**LDS capacity per architecture:**

| GPU | Arch | LDS per CU |
|---|---|---|
| MI300X | gfx942 | 64 KB |
| MI350  | gfx950 | 160 KB |
| RDNA4  | gfx1250 | 320 KB |

A `float16` element costs 2 bytes. For a `64×64` A-tile and `64×64` B-tile in f16:
`(64×64 + 64×64) × 2 = 16 384 bytes` — well within all architectures.

### 10.2 Ping-Pong LDS for Pipelining

Double-buffer the LDS so loading the next tile overlaps with computing the current:

```python
@fx.struct
class PingPong:
    a0: fx.Array[fx.Float16, BLOCK_M * BLOCK_K]   # stage 0 buffer
    a1: fx.Array[fx.Float16, BLOCK_M * BLOCK_K]   # stage 1 buffer

@flyc.kernel
def pipelined_kernel(A: fx.Tensor, ...):
    lds   = fx.SharedAllocator().allocate(PingPong).peek()
    lds_a = [
        lds.a0.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1))),
        lds.a1.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1))),
    ]

    # Prologue: fill stage 0
    load_tile(0, lds_a[0])
    gpu.barrier()

    for k in range(num_tiles - 1):
        stage = k % 2
        next  = 1 - stage
        load_tile(k + 1, lds_a[next])   # async load into next stage
        compute(lds_a[stage])            # MFMA on current stage
        gpu.barrier()                    # wait for next load

    # Epilogue: compute last tile
    compute(lds_a[(num_tiles - 1) % 2])
```

---

## 11. MFMA and WMMA — Matrix Math

### 11.1 gfx942/gfx950: MFMA (wave64)

Build an MMA atom from shape and dtype, then issue it over fragments with `fx.gemm`. Argument order is always **d, a, b, c** (destination / accumulator first).

```python
from flydsl.expr import rocdl

# FP16 -> FP32, 16×16×16 MFMA
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)   # d=frag_C, a, b, c

# BF16 -> FP32, 16×16×16 MFMA
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)

# FP8 -> FP32, 16×16×32 MFMA (gfx942: E4M3FNUZ, gfx950: E4M3FN)
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)

# INT8 -> INT32
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Int8, fx.Int32))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)

# Scaled MFMA (gfx950 / CDNA4 — block-scaled fp8)
mma = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
mma = fx.atom_set_value(mma, "scale_a", fx.Int32(scale_a))
mma = fx.atom_set_value(mma, "scale_b", fx.Int32(scale_b))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)
# Or pass state as kwargs to fx.gemm:
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C, scale_a=sa, scale_b=sb)
```

### 11.2 gfx1250: WMMA (wave32)

| dtype A/B → Acc | K | Notes |
|---|---|---|
| f16 / bf16 → f32 | 32 | |
| fp8 (E4M3FN / E5M2) → f32 | 64, 128 | any mix |
| i8 → i32 | 64 | `sign_a`, `sign_b`, `clamp` kwargs |
| i4 → i32 | 32 | `sign_a`, `sign_b`, `clamp` kwargs |

```python
# FP8 WMMA
mma = fx.make_mma_atom(rocdl.WMMA(16, 16, 128, fx.Float8E4M3FN))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)

# MX-scaled FP8 WMMA (block_size=32 -> i32 scale; 16 -> i64)
mma = fx.make_mma_atom(rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, block_size=32))
mma = fx.atom_set_value(mma, "scale_a", fx.Int32(scale_a))
mma = fx.atom_set_value(mma, "scale_b", fx.Int32(scale_b))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)

# INT4 signed
mma = fx.make_mma_atom(rocdl.WMMA(16, 16, 32, fx.Int4, fx.Int32,
                                   sign_a=True, sign_b=True, clamp=True))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)
```

### 11.3 Building Fragments

```python
# From explicit tile shape
frag_A = fx.make_fragment_A(mma, (BLOCK_M, BLOCK_K))
frag_B = fx.make_fragment_B(mma, (BLOCK_K, BLOCK_N))
frag_C = fx.make_fragment_C(mma, (BLOCK_M, BLOCK_N))

# Zero-initialize accumulator
frag_C = fx.make_fragment_C(mma, (BLOCK_M, BLOCK_N))
# fragments start zeroed by default

# From a tiled-copy partition (matched shape)
frag = fx.make_fragment_like(thr_copy.partition_S(src))
```

---

## 12. Reductions

### 12.1 Warp Reduction

XOR-shuffle across all lanes — works on wave64 (gfx942/gfx950):

```python
@flyc.kernel
def warp_reduce_kernel(A: fx.Tensor, Out: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    # Load one element per thread
    copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA   = fx.make_rmem_tensor(1, fx.Float32)
    tA   = fx.slice(fx.logical_divide(A, fx.make_layout(64, 1)), (None, bid))
    tA   = fx.logical_divide(tA, fx.make_layout(1, 1))
    fx.copy_atom_call(copy, fx.slice(tA, (None, tid)), rA)

    val = fx.Float32(fx.Vector(fx.memref_load_vec(rA))[0])

    # XOR-shuffle tree (wave64: shifts 32, 16, 8, 4, 2, 1)
    for sh in [32, 16, 8, 4, 2, 1]:
        peer = val.shuffle_xor(fx.Int32(sh), fx.Int32(64))
        val  = val + peer

    # val now holds the warp sum on every lane; store from lane 0
    if tid == fx.Int64(0):
        rOut = fx.make_rmem_tensor(1, fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([val], fx.Float32), rOut)
        tOut = fx.slice(fx.logical_divide(Out, fx.make_layout(1, 1)), (None, bid))
        fx.copy_atom_call(copy, rOut, fx.slice(tOut, (None, fx.Int64(0))))
```

For **wave32** (gfx1250), use shifts `[16, 8, 4, 2, 1]` and `fx.Int32(32)` for the wave size.

### 12.2 Block Reduction

```python
NUM_WAVES = 4   # 256-thread block = 4 waves

@fx.struct
class ReduceSmem:
    partials: fx.Array[fx.Float32, NUM_WAVES]

@flyc.kernel
def block_reduce_kernel(A: fx.Tensor, Out: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_id("x")

    # Decompose tid into (wave_id, lane_id)
    wave_layout = fx.make_layout((NUM_WAVES, 64), (64, 1))
    coord       = fx.idx2crd(tid, wave_layout)
    wave_id     = fx.get(coord, 0)
    lane_id     = fx.get(coord, 1)

    # Load and compute per-thread value
    val = ...   # fx.Float32

    # Step 1: warp reduce
    for sh in [32, 16, 8, 4, 2, 1]:
        peer = val.shuffle_xor(fx.Int32(sh), fx.Int32(64))
        val  = val + peer

    # Step 2: lane 0 of each wave stores partial to LDS
    lds       = fx.SharedAllocator().allocate(ReduceSmem).peek()
    lds_parts = lds.partials.view(fx.make_layout(NUM_WAVES, 1))
    copy      = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    if lane_id == fx.Int64(0):
        rPart = fx.make_rmem_tensor(1, fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([val], fx.Float32), rPart)
        fx.copy_atom_call(copy, rPart, fx.slice(lds_parts, (wave_id,)))

    gpu.barrier()

    # Step 3: wave 0 reads and reduces all partials
    if wave_id == fx.Int64(0):
        total = fx.Float32(0.0)
        for w in range_constexpr(NUM_WAVES):
            rP = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy, fx.slice(lds_parts, (fx.Int64(w),)), rP)
            total = total + fx.Vector(fx.memref_load_vec(rP))[0]
        # store total ...
```

---

## 13. Software Pipelining — Loop-Carried State

Software pipelining overlaps data loading for tile N+1 with computation for tile N, hiding global memory latency. FlyDSL supports this via `init=` on `range()`, which creates SSA phi nodes in the loop's MLIR representation.

```python
@flyc.kernel
def pipelined_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   M: fx.Int32, N: fx.Int32, K: fx.Int32,
                   NUM_TILES: fx.Constexpr[int]):
    BLOCK_K = 64
    acc = fx.Float32(0.0)

    def load_a_tile(k_idx):
        """Load A tile k_idx from global into registers."""
        rA = fx.make_rmem_tensor(4, fx.Float32)
        # ... copy_atom_call to fill rA ...
        return rA

    def compute(acc_in, rA_vals):
        """MFMA-based dot product."""
        # ... fx.gemm ...
        return acc_in   # updated accumulator

    # Prologue: load tile 0 before the loop
    rA_0     = load_a_tile(0)
    # Flatten register tile values into init list
    init     = [acc, fx.Vector(fx.memref_load_vec(rA_0))[0],
                     fx.Vector(fx.memref_load_vec(rA_0))[1],
                     fx.Vector(fx.memref_load_vec(rA_0))[2],
                     fx.Vector(fx.memref_load_vec(rA_0))[3]]

    # CRITICAL: loop bounds must be fx.Int64, not plain Python ints
    _start = fx.Int64(0)
    _stop  = fx.Int64(NUM_TILES - 1)   # tiles 0..NUM_TILES-2 in loop body
    _step  = fx.Int64(1)

    for iv, state in range(_start, _stop, _step, init=init):
        # Unpack loop-carried state
        acc_in = state[0]
        a0, a1, a2, a3 = state[1], state[2], state[3], state[4]

        # Load NEXT tile while computing CURRENT
        rA_next = load_a_tile(iv + fx.Int64(1))

        # Compute CURRENT tile
        acc_in = compute(acc_in, (a0, a1, a2, a3))

        # Carry forward: accumulator + next tile's values
        results = yield [
            acc_in,
            fx.Vector(fx.memref_load_vec(rA_next))[0],
            fx.Vector(fx.memref_load_vec(rA_next))[1],
            fx.Vector(fx.memref_load_vec(rA_next))[2],
            fx.Vector(fx.memref_load_vec(rA_next))[3],
        ]

    # Epilogue: process the last tile (from loop's final yield)
    acc_final = results[0]
    a0f, a1f, a2f, a3f = results[1], results[2], results[3], results[4]
    acc_final = compute(acc_final, (a0f, a1f, a2f, a3f))

    # Store accumulator to C
    # ...
```

**The three critical pitfalls:**

1. **`fx.Int64` bounds, not plain Python ints.** A plain `int` causes the AST rewriter to unroll the loop and silently discard `init=`.

2. **Clear `SmemPtr._view_cache = None`** (if using the legacy `SmemPtr`) before the epilogue to avoid SSA dominance errors.

3. **Match init list length.** The `state` tuple has exactly as many elements as `init`, and `yield` must return the same count.

---

## 14. End-to-End Example: Naive GEMM

This naive GEMM is not optimized but shows all core FlyDSL concepts together: layouts, buffer resources, copy atoms, and compile-time loops.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

@flyc.kernel
def naive_gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
):
    """C = A @ B  (f32 × f32 -> f32, no tiling, one element per thread)."""
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    # Each block handles BN columns; each thread handles one output element
    BN = 64

    # 2D logical views via buffer resource
    bufA = fx.rocdl.make_buffer_tensor(A)
    bufB = fx.rocdl.make_buffer_tensor(B)
    bufC = fx.rocdl.make_buffer_tensor(C)
    tA = fx.make_view(fx.get_iter(bufA), fx.make_layout((M, K), (K, 1)))
    tB = fx.make_view(fx.get_iter(bufB), fx.make_layout((K, N), (N, 1)))
    tC = fx.make_view(fx.get_iter(bufC), fx.make_layout((M, N), (N, 1)))

    # Map block -> (output row, base column)
    row  = bid // (N // BN)
    bcol = (bid % (N // BN)) * fx.Int64(BN)
    col  = bcol + tid

    copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA   = fx.make_rmem_tensor(1, fx.Float32)
    rB   = fx.make_rmem_tensor(1, fx.Float32)
    rC   = fx.make_rmem_tensor(1, fx.Float32)

    # Accumulate dot product over K
    acc = fx.Float32(0.0)
    for k in range_constexpr(K):
        fx.copy_atom_call(copy, fx.slice(tA, (row, fx.Int64(k))), rA)
        fx.copy_atom_call(copy, fx.slice(tB, (fx.Int64(k), col)), rB)
        a_val = fx.Vector(fx.memref_load_vec(rA))[0]
        b_val = fx.Vector(fx.memref_load_vec(rB))[0]
        acc   = acc + a_val * b_val

    # Write output
    fx.memref_store_vec(fx.Vector.from_elements([acc], fx.Float32), rC)
    fx.copy_atom_call(copy, rC, fx.slice(tC, (row, col)))


@flyc.jit
def naive_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    BN = 64
    naive_gemm_kernel(A, B, C, M, N, K).launch(
        grid=(M * (N // BN),), block=(BN,), stream=stream
    )


# Host test:
import torch

M, N, K = 128, 128, 64
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
C = torch.empty(M, N, device="cuda", dtype=torch.float32)

naive_gemm(A, B, C, M, N, K)
ref = A @ B
assert torch.allclose(C, ref, atol=1e-4), f"max err: {(C - ref).abs().max()}"
print("Naive GEMM correct!")
```

---

## 15. End-to-End Example: Tiled GEMM with LDS and MFMA

This example shows a tiled GEMM with LDS buffering and MFMA — the same structure used in production `preshuffle_gemm.py`.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@fx.struct
class GemmSmem:
    a: fx.Array[fx.Float16, BLOCK_M * BLOCK_K]
    b: fx.Array[fx.Float16, BLOCK_K * BLOCK_N]


@flyc.kernel
def tiled_gemm_kernel(
    A: fx.Tensor,       # (M, K) f16
    B: fx.Tensor,       # (K, N) f16  (column-major for MFMA B layout)
    C: fx.Tensor,       # (M, N) f32
    M: fx.Int32,
    N: fx.Int32,
    K: fx.Int32,
    NUM_K_TILES: fx.Constexpr[int],
):
    tid    = gpu.thread_id("x")
    bid_m  = gpu.block_id("x")
    bid_n  = gpu.block_id("y")

    # Decompose 256 threads into 4 waves × 64 lanes
    wave_layout = fx.make_layout((4, 64), (64, 1))
    coord       = fx.idx2crd(tid, wave_layout)
    wave_id     = fx.get(coord, 0)
    lane_id     = fx.get(coord, 1)

    # Buffer resource views of global tensors
    tA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)),
                      fx.make_layout((M, K), (K, 1)))
    tB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)),
                      fx.make_layout((K, N), (N, 1)))
    tC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)),
                      fx.make_layout((M, N), (N, 1)))

    # Allocate LDS
    lds   = fx.SharedAllocator().allocate(GemmSmem).peek()
    lds_a = lds.a.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
    lds_b = lds.b.view(fx.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))

    # Copy atoms
    copy_g = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float16)
    copy_s = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float16)

    # MMA atom: f16 -> f32, 16×16×16
    mma    = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))
    frag_C = fx.make_fragment_C(mma, (BLOCK_M, BLOCK_N))

    # K-tile loop
    for k_tile in range_constexpr(NUM_K_TILES):
        # --- Load A tile from global -> LDS ---
        # Each thread loads 4 f16 values from A
        ra = fx.make_rmem_tensor(4, fx.Float16)
        a_row = bid_m * fx.Int64(BLOCK_M) + wave_id * fx.Int64(16)
        a_col = fx.Int64(k_tile) * fx.Int64(BLOCK_K) + lane_id * fx.Int64(4)
        fx.copy_atom_call(copy_g, fx.slice(tA, (a_row, a_col)), ra)
        fx.copy_atom_call(copy_s, ra, fx.slice(lds_a, (wave_id * fx.Int64(16), lane_id * fx.Int64(4))))

        # --- Load B tile from global -> LDS ---
        rb = fx.make_rmem_tensor(4, fx.Float16)
        b_row = fx.Int64(k_tile) * fx.Int64(BLOCK_K) + wave_id * fx.Int64(16)
        b_col = bid_n * fx.Int64(BLOCK_N) + lane_id * fx.Int64(4)
        fx.copy_atom_call(copy_g, fx.slice(tB, (b_row, b_col)), rb)
        fx.copy_atom_call(copy_s, rb, fx.slice(lds_b, (wave_id * fx.Int64(16), lane_id * fx.Int64(4))))

        gpu.barrier()

        # --- MFMA over the K tile ---
        frag_A = fx.make_fragment_A(mma, (BLOCK_M, BLOCK_K))
        frag_B = fx.make_fragment_B(mma, (BLOCK_K, BLOCK_N))
        fx.copy(copy_s, fx.slice(lds_a, (None, None)), frag_A)
        fx.copy(copy_s, fx.slice(lds_b, (None, None)), frag_B)
        fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)

        gpu.barrier()

    # --- Store accumulator to C ---
    copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    c_row  = bid_m * fx.Int64(BLOCK_M)
    c_col  = bid_n * fx.Int64(BLOCK_N)
    # (fragment layout -> global store handled by fx.copy or explicit loop)
    # ...


@flyc.jit
def tiled_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    assert K % BLOCK_K == 0
    tiled_gemm_kernel(A, B, C, M, N, K, K // BLOCK_K).launch(
        grid=(M // BLOCK_M, N // BLOCK_N),
        block=(256,),
        stream=stream,
    )
```

---

## 16. Fast Launch with `_run_compiled`

Calling `@flyc.jit` every iteration re-runs DLPack conversion and cache lookup. For hot-path kernels, use `_run_compiled` to compile once and dispatch cheaply thereafter.

```python
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, _run_compiled
import flydsl.expr as fx
import torch

# Assume kernel_exe is a compiled @flyc.jit function object

out    = torch.empty(M, N, device="cuda", dtype=torch.float16)
a      = torch.randn(M, K, device="cuda", dtype=torch.float16)
b      = torch.randn(K, N, device="cuda", dtype=torch.float16)
stream = fx.Stream(torch.cuda.current_stream())

# ptr_arg extracts data_ptr() as fx.Pointer — bypasses DLPack
out_ptr = ptr_arg(out)
a_ptr   = ptr_arg(a)
b_ptr   = ptr_arg(b)

# First call: compiles and caches CompiledFunction as kernel_exe._cf, then runs
_run_compiled(kernel_exe, out_ptr, a_ptr, b_ptr, M, N, K, stream)

# All subsequent calls: direct _cf(*args) dispatch — no compilation overhead
for _ in range(100):
    _run_compiled(kernel_exe, out_ptr, a_ptr, b_ptr, M, N, K, stream)
```

Internally `_run_compiled` is:

```python
def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    cf(*args)
```

**Rules:**
- Pass flat scalars and `fx.Stream`. No `torch.Tensor` — only `ptr_arg(tensor)`.
- Argument order and types must exactly match the compiled kernel signature.
- Worth it for decode steps, per-token loops, and small kernels called millions of times.

---

## 17. Ahead-of-Time (AOT) Compilation

AOT pre-compiles kernels for a set of shapes before serving, so the first request never pays JIT cost.

### Running AOT

```bash
# MoE kernels (reads CSV shape configs):
python -m aiter.aot.flydsl.moe

# GEMM kernels:
python -m aiter.aot.flydsl.gemm

# Grouped MoE (gfx1250):
python -m aiter.aot.flydsl.grouped_moe

# Custom CSV:
python -m aiter.aot.flydsl.moe --csv /path/to/shapes.csv

# Parallel compilation with 16 workers:
AITER_FLYDSL_AOT_WORKERS=16 python -m aiter.aot.flydsl.moe
```

### AOT Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AITER_FLYDSL_AOT_WORKERS` | auto | Max parallel compile workers |
| `AITER_FLYDSL_AOT_MEM_PER_WORKER_GB` | 2.0 | Memory cap per worker |
| `AITER_FLYDSL_AOT_TIMEOUT` | 1200 | Per-kernel timeout (seconds) |
| `AITER_FLYDSL_AOT_MAX_RETRIES` | 2 | Retries on OOM/crash |
| `AITER_AOT_IMPORT` | 0 | Set to 1 for lightweight import during AOT |

### Architecture from CSV

| `cu_num` | Arch |
|---|---|
| 80 or 304 | gfx942 (MI300X) |
| 256 | gfx950 (MI350) |

### Verifying AOT Cache Hits

```python
from aiter.aot.flydsl.common import fail_on_aot_cache_miss

with fail_on_aot_cache_miss():
    # Raises if any kernel compiles at runtime (cache miss)
    out = flydsl_moe_stage1(tokens, gate_w, up_w, ...)
    out = flydsl_moe_stage2(activated, down_w, ...)

print("All kernels hit AOT cache!")
```

---

## 18. Using the Public aiter API

aiter wraps FlyDSL kernels in a stable Python API. Import from `aiter.ops.flydsl`.

### Check Availability

```python
from aiter.ops.flydsl import is_flydsl_available

if not is_flydsl_available():
    print("FlyDSL not available on this GPU — falling back to Triton")
else:
    print("FlyDSL ready")
```

### 18.1 HGEMM

```python
from aiter.ops.flydsl import flydsl_hgemm
import torch

M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(N, K, device="cuda", dtype=torch.float16)   # N-major layout

out = flydsl_hgemm(
    a, b,
    out=None,           # optional pre-allocated (M, N) output
    bias=None,          # optional (N,) bias tensor
    tile_m=128,
    tile_n=128,
    tile_k=64,
    split_k=1,
    stages=2,
    block_m_warps=2,
    block_n_warps=2,
    block_k_warps=1,
    b_to_lds=False,
    stream=None,
)
# out: (M, N) float16

# Correctness check:
ref = (a.float() @ b.float().T).half()
assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
```

**Tiling constraints:**
- `tile_k >= 32`, `tile_k % 32 == 0`
- `tile_m % (block_m_warps * 16) == 0`
- `tile_n % (block_n_warps * 16) == 0`
- `K % split_k == 0`, `(K // split_k) % tile_k == 0`

### 18.2 MoE Stage 1 + Stage 2

```python
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.flydsl import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_f4_quant
import torch

# Model parameters
T, E, D, inter = 128, 8, 4096, 14336   # tokens, experts, model_dim, inter_dim
topk = 2

# Inputs
inp     = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
w1      = torch.randn(E, inter * 2, D, dtype=torch.bfloat16, device="cuda")
w2      = torch.randn(E, D, inter, dtype=torch.bfloat16, device="cuda")
scores  = torch.randn(T, E, dtype=torch.bfloat16, device="cuda")

# Expert routing
topk_w, topk_ids = fused_topk(inp, scores, topk, renormalize=True)
sorted_ids, sorted_weights, sorted_expert_ids, num_valid, _ = moe_sorting(
    topk_ids, topk_w, E, D, torch.bfloat16, block_m=32
)

# Quantize activations and weights for fp8 x fp4 path
a_q, a_scale = per_1x32_f8_scale_f8_quant(inp, quant_dtype=dtypes.fp8,
                                            scale_type=dtypes.fp8_e8m0)
w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)

# Stage 1: gate + up projection (fp8 activation × fp4 weight)
gate_out, up_out = flydsl_moe_stage1(
    a_q, a_scale,
    w1_q, w1_scale,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
    topk=topk,
    inter_dim=inter,
    out_dtype=torch.bfloat16,
)

# Activation between stages
import torch.nn.functional as F
activated = F.silu(gate_out) * up_out

# Stage 2: down projection
output = flydsl_moe_stage2(
    activated,
    w2_q, w2_scale,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
    topk=topk,
    model_dim=D,
    out_dtype=torch.bfloat16,
)
# output: (T, D) bfloat16 — MoE output tokens
```

### 18.3 Fused QK Norm + RoPE + FP8 Quant

Fuses per-token RMSNorm, GPT-J RoPE, and optional FP8 quantization into a single kernel launch (one wave64 per token).

```python
from aiter.ops.flydsl import flydsl_qk_norm_rope_quant
import torch

T  = 256   # tokens
H  = 8     # query heads
D  = 128   # head dim
RD = 64    # rope head dim (last RD dims get RoPE)

q       = torch.randn(T, H * D, dtype=torch.bfloat16, device="cuda")
kv      = torch.randn(T, D,     dtype=torch.bfloat16, device="cuda")
kv_w    = torch.ones(D,         dtype=torch.bfloat16, device="cuda")   # RMSNorm weight
cos     = torch.randn(2048, RD // 2, dtype=torch.bfloat16, device="cuda")
sin     = torch.randn(2048, RD // 2, dtype=torch.bfloat16, device="cuda")
pos     = torch.randint(0, 2048, (T,), dtype=torch.int64, device="cuda")

# bf16 pass-through (no quantization):
q_out, kv_out, q_scale, kv_scale = flydsl_qk_norm_rope_quant(
    q, kv, kv_w, cos, sin, pos,
    num_q_heads=H,
    head_dim=D,
    rope_head_dim=RD,
    quant=False,
)
# q_out: (T, H*D) bf16, kv_out: (T, D) bf16, scales: None

# FP8 quantized output:
q_fp8, kv_fp8, q_scale, kv_scale = flydsl_qk_norm_rope_quant(
    q, kv, kv_w, cos, sin, pos,
    num_q_heads=H,
    head_dim=D,
    rope_head_dim=RD,
    quant=True,
    quant_group_size=None,   # per-row scaling
    scale_dtype="fp32",
)
# q_fp8: (T, H*D) fp8, kv_fp8: (T, D) fp8
# q_scale: (T, H) f32, kv_scale: (T,) f32

# Block-quantized FP8 (group_size=128):
q_fp8, kv_fp8, q_scale, kv_scale = flydsl_qk_norm_rope_quant(
    q, kv, kv_w, cos, sin, pos,
    num_q_heads=H,
    head_dim=D,
    rope_head_dim=RD,
    quant=True,
    quant_group_size=128,
    scale_dtype="e8m0",   # MX E8M0 block scale
)
```

### 18.4 MLA Reduce

Online-softmax reduce for Multi-head Latent Attention decode across KV splits.

```python
from aiter.ops.flydsl import flydsl_mla_reduce_v1
import torch

H, Dv = 16, 512
num_partial = 8    # partial results from 8 KV splits

partial_output = torch.randn(num_partial, H, Dv, dtype=torch.float32, device="cuda")
partial_lse    = torch.randn(num_partial, H,     dtype=torch.float32, device="cuda")

# CSR indptr: each query reduces over all splits
reduce_indptr  = torch.tensor([0, num_partial], dtype=torch.int32, device="cuda")
reduce_final_map = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
partial_pool_map = torch.arange(num_partial, dtype=torch.int32, device="cuda")

fout  = torch.empty(1, H, Dv, dtype=torch.float16, device="cuda")
flse  = torch.empty(1, H,     dtype=torch.float32, device="cuda")

flydsl_mla_reduce_v1(
    partial_output, partial_lse,
    reduce_indptr, reduce_final_map, partial_pool_map,
    max_seqlen_q=1,
    fout=fout,
    flse=flse,
    num_kv_splits=num_partial,
)
# fout: (1, H, Dv) fp16 — final attention output
```

### 18.5 Flash Attention

Self-attention kernel for gfx1201 (RDNA4) and gfx1250. Sequences are auto-padded to multiples of 128.

```python
from aiter.ops.flydsl import flydsl_flash_attn_func
import torch

B, S, H, D = 1, 32768, 12, 128   # batch, seq_len, heads, head_dim

q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

out = flydsl_flash_attn_func(
    q, k, v,
    causal=False,
    waves_per_eu=2,
    daz=False,       # denormals-as-zero for ROCm
)
# out: (B, S, H, D) bfloat16

# Causal mask (autoregressive):
out_causal = flydsl_flash_attn_func(q, k, v, causal=True)

# Correctness check vs torch SDPA:
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
).transpose(1, 2)
assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
```

---

## 19. Autotuning

FlyDSL includes a Triton-style autotuner. Wrap `@flyc.jit` with `@autotune` to benchmark and cache the best configuration.

```python
from flydsl.autotune import autotune, Config, do_bench
import flydsl.compiler as flyc
import flydsl.expr as fx

@autotune(
    configs=[
        Config(tile_m=64,  tile_n=64,  tile_k=32, block_m_warps=2, block_n_warps=2),
        Config(tile_m=128, tile_n=64,  tile_k=32, block_m_warps=4, block_n_warps=2),
        Config(tile_m=128, tile_n=128, tile_k=64, block_m_warps=4, block_n_warps=4),
        Config(tile_m=64,  tile_n=128, tile_k=32, block_m_warps=2, block_n_warps=4),
    ],
    key=["M", "N", "K"],   # re-tune when these change
    warmup=5,
    rep=25,
)
@flyc.jit
def autotuned_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Int32,
    N: fx.Int32,
    K: fx.Int32,
    # Config kwargs become Constexpr parameters:
    tile_m:       fx.Constexpr[int],
    tile_n:       fx.Constexpr[int],
    tile_k:       fx.Constexpr[int],
    block_m_warps: fx.Constexpr[int],
    block_n_warps: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    ...   # kernel body uses tile_m, tile_n, etc.


# Usage — first call benchmarks all configs, caches best:
autotuned_gemm(A, B, C, M, N, K)
# Subsequent calls use the cached best config (no benchmarking overhead):
autotuned_gemm(A, B, C, M, N, K)
```

**Benchmarking a kernel manually:**

```python
from flydsl.autotune import do_bench

ms = do_bench(lambda: autotuned_gemm(A, B, C, M, N, K), warmup=5, rep=25)
print(f"Median: {ms:.3f} ms")
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
print(f"TFLOPS: {tflops:.1f}")
```

**Autotune cache location:** `~/.flydsl/autotune/{func_name}.json`

**`waves_per_eu` note:** pass it in `Config(waves_per_eu=4)` — it's a special compiler-level hint. The `opts=` path in `gpu-module-to-binary` does not reliably propagate it; it must be set as an LLVM function attribute.

---

## 20. Debugging and IR Inspection

### IR Dump

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_kernel.py
```

Produces numbered `.mlir` files per pass and a `final_isa.s` file:

| File | Stage |
|---|---|
| `00_*.mlir` | Input Fly dialect (post-tracing) |
| `03_*.mlir` | After `fly-layout-lowering` |
| `05_*.mlir` | After `fly-promote-regmem-to-vectorssa` |
| `07_*.mlir` | After `convert-fly-to-rocdl` |
| `final_isa.s` | AMD ISA assembly |

### Printf Debugging

```python
@flyc.kernel
def debug_kernel(A: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    # Print from every thread (use sparingly — serializes execution):
    fx.printf("tid={} bid={}", tid, bid)

    # Print a computed value:
    val = fx.Float32(3.14)
    fx.printf("tid={} val={}", tid, val)
```

### Source-to-Assembly Debug Info (ATT Profiling)

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 python my_kernel.py
# Then profile with rocprofv3 ATT to get Python source -> ISA mapping
```

With this enabled, the MLIR pipeline inserts `ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` before binary emission. The resulting `.debug_line` section in the HSACO lets rocprofv3 annotate `code.json` with `"source_file:line"` entries.

Verify: `grep -n '\.loc\|\.file' dumps/final_isa.s | head -20`

### Optimization Level

```bash
FLYDSL_COMPILE_OPT_LEVEL=0 python my_kernel.py   # -O0: readable ISA, no inlining
FLYDSL_COMPILE_OPT_LEVEL=3 python my_kernel.py   # maximum optimization
```

### Architecture Utilities

```python
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size, default_f8_type
from aiter.ops.flydsl.utils import addressable_lds_bytes_for_gfx

arch = "gfx942"
print(get_warp_size(arch))                    # 64
print(default_f8_type(arch))                  # Float8E4M3FNUZ
print(addressable_lds_bytes_for_gfx(arch))   # 65536

arch = "gfx950"
print(get_warp_size(arch))                    # 64
print(default_f8_type(arch))                  # Float8E4M3FN
print(addressable_lds_bytes_for_gfx(arch))   # 163840

arch = "gfx1250"
print(get_warp_size(arch))                    # 32
print(default_f8_type(arch))                  # Float8E4M3FN
print(addressable_lds_bytes_for_gfx(arch))   # 327680
```

---

## 21. Common Pitfalls

### 1. Plain int loop bounds drop `init=`

```python
# WRONG — AST rewriter unrolls, ignores init=:
for iv, state in range(0, N-1, 1, init=init_state):
    ...

# CORRECT — fx.Int64 bounds produce a runtime scf.for:
for iv, state in range(fx.Int64(0), fx.Int64(N-1), fx.Int64(1), init=init_state):
    ...
```

### 2. `const_expr` on runtime values

```python
lane = gpu.thread_id("x") % fx.Int64(64)

# WRONG — lane is a runtime SSA value, not a compile-time constant:
if const_expr(lane == fx.Int64(0)):
    ...

# CORRECT:
if lane == fx.Int64(0):
    ...
```

### 3. `arith.absf` does not exist

```python
# WRONG:
result = arith.absf(v)

# CORRECT:
zeros  = fx.Vector.filled(N, 0.0, fx.Float32)
result = (v < zeros).select(-v, v)
```

### 4. `buffer_load` offset is in elements, not bytes

```python
# WRONG (off by sizeof(dtype)):
data = buffer_ops.buffer_load(rsrc, byte_offset, dtype=fx.Float32)

# CORRECT:
data = buffer_ops.buffer_load(rsrc, element_offset, dtype=fx.Float32)
# element_offset = byte_offset / sizeof(dtype)
```

### 5. `SmemPtr._view_cache` dominance error in epilogue

If a `SmemPtr.get()` is called inside a `range(..., init=...)` loop body and then reused after the loop, the cached view's SSA definition is out of scope:

```python
# Workaround for legacy SmemPtr:
my_smem_ptr._view_cache = None   # clear cache before epilogue
my_smem_ptr.get()                # now re-generates a fresh view in the right scope
```

This issue does not arise with `SharedAllocator` (preferred for new kernels).

### 6. DLTensorAdaptor segfault with varying Constexpr

```python
# WRONG — adaptor caches MLIR type from first context; segfaults on second call
# with different N (new context):
launch(flyc.from_dlpack(tensor), N=128)
launch(flyc.from_dlpack(tensor), N=256)   # segfault

# CORRECT — pass raw torch.Tensor:
launch(tensor, N=128)
launch(tensor, N=256)
```

### 7. LDS overflow

The compiler errors if static LDS exceeds the architecture limit. Budget your `@fx.struct`:

```python
# Check before writing the struct:
BLOCK_M, BLOCK_K, BLOCK_N = 128, 64, 128
ldsA = BLOCK_M * BLOCK_K * 2   # float16 = 2 bytes -> 16 384 bytes
ldsB = BLOCK_K * BLOCK_N * 2   #                   -> 16 384 bytes
total = ldsA + ldsB             # 32 768 bytes — fits gfx942 (64 KB limit)
print(f"LDS usage: {total / 1024:.1f} KB")
```

### 8. `tile_k` alignment for GEMM MFMA

`tile_k * element_bytes` must be divisible by 64 (the MFMA K64-byte micro-step):
- float16 (2 bytes): `tile_k` must be multiple of 32
- fp8 (1 byte): `tile_k` must be multiple of 64

### 9. Double-wrapping typed values

```python
# WRONG (redundant wraps — ArithValue is deprecated):
layout = fx.make_layout(fx.Int32(BLOCK_M), fx.Int32(1))
off    = fx.Int64(fx.Int64(base) + fx.Int64(4))

# CORRECT — layout builders take plain Python ints:
layout = fx.make_layout(BLOCK_M, 1)
off    = base + fx.Int64(4)   # wrap the literal once, not the result
```

### 10. `define-then-use across if/else` branch

```python
# PROBLEMATIC:
if cond:
    dst = a
else:
    dst = b
use(dst)   # dst may not be properly in scope for MLIR

# CORRECT — use select:
dst = cond.select(a, b)
use(dst)
```

---

## 22. Example: Activation Functions (SiLU, SWiGLU, exp2/rcp)

FlyDSL kernels implement activation functions using `rocdl.exp2` and `rocdl.rcp` directly — no `math` library call, no branch. The constant `LOG2E = 1.4426950408889634` converts `exp(x)` into `exp2(x * LOG2E)`.

### Sigmoid via exp2/rcp

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl, range_constexpr

LOG2E = fx.Float32(1.4426950408889634)

def _sigmoid_f32(g: fx.Float32) -> fx.Float32:
    """σ(g) = 1 / (1 + exp(-g))  implemented as  exp2(-g*LOG2E) + 1  then rcp."""
    neg_g    = -g
    exp_part = rocdl.exp2(neg_g * LOG2E)
    return rocdl.rcp(exp_part + fx.Float32(1.0))

def _tanh_f32(x: fx.Float32) -> fx.Float32:
    """tanh(x) = 2*σ(2x) - 1, sign-restored via negate trick."""
    two_x    = x * fx.Float32(2.0)
    sig2     = _sigmoid_f32(two_x)
    magnitude = sig2 * fx.Float32(2.0) - fx.Float32(1.0)
    # sign restore: if x < 0 negate (already handled by σ(2x) ≤ 0.5 → result ≤ 0)
    return magnitude
```

### SiLU batch (vectorized, for MoE epilogues)

```python
def silu_mul_batch(gs: list, us: list) -> list:
    """
    Element-wise SiLU-and-mul for a list of (gate, up) f32 SSA values.
    Returns  gate * sigmoid(gate) * up  for each pair.
    Used in MoE stage-1 epilogues.
    """
    results = []
    for g, u in zip(gs, us):
        sig = _sigmoid_f32(g)
        results.append(g * sig * u)
    return results


@flyc.kernel
def silu_and_mul_kernel(
    Gate: fx.Tensor,    # (T, N) f32 gate activations
    Up:   fx.Tensor,    # (T, N) f32 up activations
    Out:  fx.Tensor,    # (T, N) f32 output
    N:    fx.Constexpr[int],
):
    VEC   = 4
    BLOCK = 256

    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    bufG = fx.rocdl.make_buffer_tensor(Gate)
    bufU = fx.rocdl.make_buffer_tensor(Up)
    bufO = fx.rocdl.make_buffer_tensor(Out)

    copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    rG   = fx.make_rmem_tensor(VEC, fx.Float32)
    rU   = fx.make_rmem_tensor(VEC, fx.Float32)
    rO   = fx.make_rmem_tensor(VEC, fx.Float32)

    base = bid * fx.Int64(BLOCK * VEC) + tid * fx.Int64(VEC)
    tG   = fx.make_view(fx.get_iter(bufG), fx.make_layout((1, N), (N, 1)))
    tU   = fx.make_view(fx.get_iter(bufU), fx.make_layout((1, N), (N, 1)))
    tO   = fx.make_view(fx.get_iter(bufO), fx.make_layout((1, N), (N, 1)))

    fx.copy_atom_call(copy, fx.slice(tG, (fx.Int64(0), base)), rG)
    fx.copy_atom_call(copy, fx.slice(tU, (fx.Int64(0), base)), rU)

    vg = fx.Vector(fx.memref_load_vec(rG))
    vu = fx.Vector(fx.memref_load_vec(rU))

    # Vectorized SiLU: out[i] = gate[i] * σ(gate[i]) * up[i]
    gs = [vg[i] for i in range_constexpr(VEC)]
    us = [vu[i] for i in range_constexpr(VEC)]
    activated = silu_mul_batch(gs, us)

    vo = fx.Vector.from_elements(activated, fx.Float32)
    fx.memref_store_vec(vo, rO)
    fx.copy_atom_call(copy, rO, fx.slice(tO, (fx.Int64(0), base)))
```

### SWiGLU (clamp + Gelu-like gate)

```python
ALPHA = fx.Float32(1.702)   # SWiGLU α: σ(αx) approximates GeLU
LIMIT = fx.Float32(7.0)     # clamp |gate| to avoid exp2 overflow

def swiglu_element(g: fx.Float32, u: fx.Float32) -> fx.Float32:
    """SWiGLU: out = sigmoid(alpha * clamp(gate, -limit, limit)) * gate * up"""
    neg_limit  = -LIMIT
    g_clamped  = -((-g).maximumf(neg_limit))      # clamp lower bound via maximumf
    g_clamped  = -(g_clamped.maximumf(-LIMIT))     # clamp upper bound (mirror)
    sig        = _sigmoid_f32(ALPHA * g_clamped)
    return sig * g * u
```

### Compile-time dispatch between activation variants

```python
from flydsl.expr import const_expr

def gate_up_act(
    act:  str,    # "silu" | "swiglu" — compile-time string Constexpr
    gs:   list,   # f32 gate SSA values
    us:   list,   # f32 up SSA values
) -> list:
    if const_expr(act == "swiglu"):
        return [swiglu_element(g, u) for g, u in zip(gs, us)]
    else:
        return silu_mul_batch(gs, us)   # default: SiLU
```

---

## 23. Example: Fused SWiGLU Kernel (bf16, TiledCopy)

This example mirrors `swiglu_and_mul.py` — a full bf16 element-wise fused swiglu kernel using `TiledCopy` and bf16 buffer resources.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl, range_constexpr, const_expr

# V = 8 bf16 per load = 128 bits
V      = 8
NLANE  = 16   # per-warp output lanes
LOG2E  = 1.4426950408889634
ALPHA  = 1.702
LIMIT  = 7.0

def _swiglu_bf16_vec(vg: fx.Vector, vu: fx.Vector) -> fx.Vector:
    """
    Vectorized SWiGLU on a bf16 vector.
    Extends bf16 → f32, computes SWiGLU, truncates back to bf16.
    """
    LOG2E_f = fx.Float32(LOG2E)
    ALPHA_f = fx.Float32(ALPHA)
    LIMIT_f = fx.Float32(LIMIT)

    results = []
    for i in range_constexpr(V):
        g_bf16 = vg[i]                         # BFloat16 scalar
        u_bf16 = vu[i]
        g      = g_bf16.to(fx.Float32)         # extend to f32
        u      = u_bf16.to(fx.Float32)

        # clamp gate to [-LIMIT, LIMIT]
        neg_limit = -LIMIT_f
        gc = -((-g).maximumf(neg_limit))       # clamp lower
        # For upper: negate, clamp, negate again
        gc = -((-gc).maximumf(neg_limit))      # gc is now in [-LIMIT, LIMIT]

        # sigmoid(alpha * gc) via exp2
        neg_ag   = -(ALPHA_f * gc)
        exp_part = rocdl.exp2(neg_ag * LOG2E_f)
        sig      = rocdl.rcp(exp_part + fx.Float32(1.0))

        out_f32  = sig * g * u
        results.append(out_f32.to(fx.BFloat16))

    return fx.Vector.from_elements(results, fx.BFloat16)


def _bf16_buf_view(tensor: fx.Tensor, num_rows: int, num_cols: int):
    """Build a 2D bf16 buffer-resource view over a tensor."""
    buf = fx.rocdl.make_buffer_tensor(tensor)
    return fx.make_view(fx.get_iter(buf),
                        fx.make_layout((num_rows, num_cols), (num_cols, 1)))


@flyc.kernel
def swiglu_kernel(
    X:        fx.Tensor,           # (T, 2*N) bf16 — interleaved [gate | up]
    Out:      fx.Tensor,           # (T, N) bf16 output
    num_rows: fx.Int32,            # T (runtime)
    inter_dim: fx.Constexpr[int],  # N (compile-time)
):
    """
    Reads X as (T, 2*inter_dim) bf16 where X[:, :inter_dim] = gate,
    X[:, inter_dim:] = up.  Writes Out as (T, inter_dim) bf16.
    Each block processes one row; threads tile over inter_dim.
    """
    tid = gpu.thread_id("x")
    row = gpu.block_id("y")

    tX   = _bf16_buf_view(X,   1, inter_dim * 2)
    tOut = _bf16_buf_view(Out, 1, inter_dim)

    # Gate and up are in the first and second halves of each row
    copy   = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    rGate  = fx.make_rmem_tensor(V, fx.BFloat16)
    rUp    = fx.make_rmem_tensor(V, fx.BFloat16)
    rOut   = fx.make_rmem_tensor(V, fx.BFloat16)

    col_base = tid * fx.Int64(V)
    row_off  = row * fx.Int64(inter_dim * 2)
    gate_col = col_base
    up_col   = col_base + fx.Int64(inter_dim)

    # The buffer tensor row is a strided view — use soffset to pick the row
    fx.copy_atom_call(copy, fx.slice(tX, (fx.Int64(0), gate_col)),
                      rGate, soffset=fx.Int32(row * inter_dim * 2 * 2))
    fx.copy_atom_call(copy, fx.slice(tX, (fx.Int64(0), up_col)),
                      rUp,   soffset=fx.Int32(row * inter_dim * 2 * 2))

    vg = fx.Vector(fx.memref_load_vec(rGate))
    vu = fx.Vector(fx.memref_load_vec(rUp))
    vo = _swiglu_bf16_vec(vg, vu)

    fx.memref_store_vec(vo, rOut)
    fx.copy_atom_call(copy, rOut, fx.slice(tOut, (fx.Int64(0), col_base)),
                      soffset=fx.Int32(row * inter_dim * 2))


@flyc.jit
def swiglu(
    X:         fx.Tensor,
    Out:       fx.Tensor,
    num_rows:  fx.Int32,
    inter_dim: fx.Constexpr[int],
    stream:    fx.Stream = fx.Stream(None),
):
    threads_per_row = inter_dim // V
    swiglu_kernel(X, Out, num_rows, inter_dim).launch(
        grid=(1, num_rows, 1), block=(threads_per_row, 1, 1), stream=stream
    )


# Host test:
import torch
T, N = 64, 4096
X   = torch.randn(T, N * 2, dtype=torch.bfloat16, device="cuda")
Out = torch.empty(T, N,     dtype=torch.bfloat16, device="cuda")
swiglu(X, Out, T, N)

# Reference:
gate, up = X[:, :N].float(), X[:, N:].float()
ALPHA, LIMIT = 1.702, 7.0
gc  = gate.clamp(-LIMIT, LIMIT)
sig = torch.sigmoid(ALPHA * gc)
ref = (sig * gate * up).bfloat16()
assert torch.allclose(Out, ref, atol=5e-3, rtol=1e-2)
print("SWiGLU correct!")
```

---

## 24. Example: Fused Activation + MX Quantization

This pattern (from `silu_and_mul_fq.py`) shows how to fuse SiLU activation with FP4/FP8 quantization in a single kernel — the key MoE inter-stage operation.

### sorted_ids packing

The kernel uses a packed `sorted_ids` tensor where the upper 8 bits hold the slot index (output row) and the lower 24 bits hold the token index (input row):

```python
import torch

def pack_sorted_ids(token_ids: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
    """Pack token and slot indices into the format the fq kernel expects."""
    return (slot_ids.int() << 24) | (token_ids.int() & 0xFFFFFF)


# Kernel decodes as:
# token_id = fused_val & 0xFFFFFF   -> input row index
# slot_id  = fused_val >> 24        -> output row index (where to scatter)
```

### FP4 quantization: amax → E8M0 scale → nibble packing

```python
from flydsl.expr import rocdl

def emit_mx_e8m0_scale(amax: fx.Float32) -> fx.Int32:
    """
    Convert a per-group absolute-max to an MX E8M0 scale byte.
    E8M0 = biased exponent of amax, rounded to nearest representable power-of-2.
    The kernel stores this as an i32 (packed into the scale tensor later).
    """
    # In FlyDSL kernel code this is a rocdl intrinsic:
    return rocdl.cvt_f32_to_mx_e8m0(amax)   # returns i32 with E8M0 byte in low 8 bits


def pack_fp4_nibbles(v0: fx.Float32, v1: fx.Float32,
                     scale: fx.Float32) -> fx.Int32:
    """
    Quantize two f32 values to FP4 and pack them into one byte.
    v0 occupies bits [3:0], v1 occupies bits [7:4].
    """
    # Quantize: clamp to [-6, 6] in FP4 range, then round
    fp4_v0 = rocdl.cvt_f32_to_fp4(v0 * scale)   # i32 with fp4 nibble
    fp4_v1 = rocdl.cvt_f32_to_fp4(v1 * scale)
    return fp4_v0 | (fp4_v1 << fx.Int32(4))
```

### FP8 quantization: two values per call

```python
def pack_fp8_pair(v0: fx.Float32, v1: fx.Float32) -> fx.Int32:
    """
    Pack two f32 values into two fp8 bytes using rocdl intrinsic.
    Returns i32 with fp8_v0 in low byte, fp8_v1 in next byte.
    """
    return rocdl.cvt_pk_fp8_f32(v0, v1)   # 2 fp8 bytes packed into i32
```

### Warp-level amax for block scaling

The quantization scale for a group of values is the max absolute value, reduced across the warp before packing:

```python
@flyc.kernel
def silu_fq_kernel(
    Gate:       fx.Tensor,
    Up:         fx.Tensor,
    Out:        fx.Tensor,     # FP4 or FP8 output
    OutScale:   fx.Tensor,     # E8M0 block scales
    sorted_ids: fx.Tensor,     # packed (slot<<24 | token) routing
    inter_dim:  fx.Constexpr[int],
    VEC:        fx.Constexpr[int],   # elements per thread (power of 2, ≤8)
    QUANT_BLK:  fx.Constexpr[int],  # quantization group size (e.g. 32)
):
    tid = gpu.thread_id("x")

    # Decode token and slot from sorted_ids
    fused_val = ...   # load from sorted_ids[block_id]
    token_id  = fused_val & fx.Int32(0xFFFFFF)
    slot_id   = fused_val >> fx.Int32(24)

    # Load gate and up for this token (VEC elements each)
    # ... copy atoms using token_id as row index ...

    # SiLU activation
    gs = [vg[i] for i in range_constexpr(VEC)]
    us = [vu[i] for i in range_constexpr(VEC)]
    activated = silu_mul_batch(gs, us)   # list of VEC f32 values

    # Warp-level amax over activated (QUANT_BLK = 32 elements/group)
    local_amax = fx.Float32(0.0)
    for i in range_constexpr(VEC):
        abs_val    = (activated[i] < fx.Float32(0.0)).select(-activated[i], activated[i])
        local_amax = local_amax.maximumf(abs_val)

    # XOR-reduce amax across threads in the quantization block
    THREADS_PER_BLK = QUANT_BLK // VEC
    for sh in [16, 8, 4, 2, 1]:   # reduce across THREADS_PER_BLK threads (≤32)
        if const_expr(sh < THREADS_PER_BLK):
            peer       = local_amax.shuffle_xor(fx.Int32(sh), fx.Int32(64))
            local_amax = local_amax.maximumf(peer)

    # Convert amax -> E8M0 scale and store to OutScale at the slot position
    e8m0_scale = emit_mx_e8m0_scale(local_amax)
    # ... scatter e8m0_scale to OutScale[slot_id, group_col] ...

    # Quantize and pack
    inv_scale = rocdl.rcp(local_amax + fx.Float32(1e-12))
    packed    = pack_fp4_nibbles(activated[0] * inv_scale,
                                 activated[1] * inv_scale,
                                 fx.Float32(1.0))
    # ... store packed nibbles to Out[slot_id, col] ...
```

---

## 25. Example: MoE Top-K Reduction

From `moe_reduce.py`: reduce `X[token, topk, dim]` → `Y[token, dim]` weighted by routing scores. Each block handles one output token; threads tile over `dim`.

```python
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, const_expr, rocdl

def _make_row_view(base_i64: fx.Int64, ncols: int, dtype: type) -> fx.Tensor:
    """
    Build a single-row buffer-tensor view from a raw i64 pointer.
    Used to index into per-expert weight rows without carrying
    a full 3D tensor into the kernel.
    """
    ptr_ty = fx.PointerType.get(dtype, address_space=1, alignment=2)
    ptr    = fx.IntToPtr(base_i64, ptr_ty)
    buf    = fx.rocdl.make_buffer_tensor(ptr, max_size=True)
    return fx.make_view(fx.get_iter(buf), fx.make_layout((1, ncols), (ncols, 1)))


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    num_tokens: int,
    model_dim:  int,
    topk:       int,
    dtype_str:  str = "bf16",   # "bf16" | "fp8"
    use_weight: bool = True,
):
    """
    Compile and return a launch closure for the MoE topk-reduction kernel.
    Result is cached; subsequent calls with same args hit the cache instantly.
    """
    V        = 8 if dtype_str == "bf16" else 4
    THREADS  = model_dim // V
    IS_FP8   = dtype_str == "fp8"

    @flyc.kernel
    def moe_reduce_kernel(
        X:          fx.Tensor,    # (T*topk, model_dim) bf16 or fp8
        XScale:     fx.Tensor,    # (T*topk,) f32 e8m0 per-row scales (fp8 only)
        Weights:    fx.Tensor,    # (T*topk,) f32 routing weights
        Y:          fx.Tensor,    # (T, model_dim) output
        T:          fx.Int32,
        K:          fx.Constexpr[int],     # topk
        D:          fx.Constexpr[int],     # model_dim
        VEC:        fx.Constexpr[int],
        IS_FP8_:    fx.Constexpr[bool],
        USE_WEIGHT: fx.Constexpr[bool],
    ):
        tid   = gpu.thread_id("x")
        token = gpu.block_id("x")

        # Accumulate across topk expert outputs
        acc = fx.Vector.filled(VEC, 0.0, fx.Float32)

        copy_x = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        rX     = fx.make_rmem_tensor(VEC, fx.Float32)

        for k in range_constexpr(K):
            row = token * fx.Int64(K) + fx.Int64(k)

            # Load one expert's output row into registers
            tX = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(X)),
                fx.make_layout((T * K, D), (D, 1))
            )
            fx.copy_atom_call(copy_x, fx.slice(tX, (row, tid * fx.Int64(VEC))), rX)
            vx = fx.Vector(fx.memref_load_vec(rX))

            if const_expr(IS_FP8_):
                # Decode FP8: unpack via rocdl, multiply by per-row E8M0 scale
                # (abbreviated — real kernel uses rocdl.cvt_pk_f32_fp8)
                scale_row = ...   # load XScale[row]
                vx        = vx * fx.Vector.filled(VEC, scale_row, fx.Float32)

            if const_expr(USE_WEIGHT):
                # Scale by routing weight (deferred from stage2 epilogue)
                w   = ...   # load Weights[row]
                vx  = vx * fx.Vector.filled(VEC, w, fx.Float32)

            acc = acc + vx

        # Store accumulated result
        tY   = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Y)),
            fx.make_layout((T, D), (D, 1))
        )
        rOut = fx.make_rmem_tensor(VEC, fx.Float32)
        fx.memref_store_vec(acc, rOut)
        fx.copy_atom_call(copy_x, rOut, fx.slice(tY, (token, tid * fx.Int64(VEC))))


    @flyc.jit
    def launch(X, XScale, Weights, Y, T, stream=fx.Stream(None)):
        moe_reduce_kernel(
            X, XScale, Weights, Y, T, topk, model_dim, V, IS_FP8, use_weight
        ).launch(grid=(num_tokens,), block=(THREADS,), stream=stream)

    return launch


# Host usage:
import torch

T, K, D = 32, 2, 4096
X       = torch.randn(T * K, D, dtype=torch.bfloat16, device="cuda")
Weights = torch.rand(T * K,    dtype=torch.float32,   device="cuda")
Y       = torch.empty(T, D,    dtype=torch.bfloat16,  device="cuda")

launch = compile_moe_reduction(T, D, K, dtype_str="bf16", use_weight=True)
launch(X, None, Weights, Y, T)

# Reference:
ref = (X.float().view(T, K, D) * Weights.view(T, K, 1)).sum(dim=1).bfloat16()
assert torch.allclose(Y, ref, atol=1e-2, rtol=1e-2)
print("MoE reduction correct!")
```

---

## 26. Example: Buffer Tensor Scalar View / Tiled Store Patterns

These patterns appear in `qk_norm_rope_quant.py` and are useful whenever you need scalar indexed access or bf16 register-to-global stores.

### Scalar load / store via buffer tensor

```python
import flydsl.expr as fx
from flydsl.expr import rocdl

def _scalar_view(tensor: fx.Tensor, num_elems: int, dtype):
    """
    Build a 1D buffer-resource view for scalar indexed access.
    Use this instead of raw buffer_ops when you only need one element at a time.
    """
    buf = fx.rocdl.make_buffer_tensor(tensor, max_size=True)
    return fx.make_view(fx.get_iter(buf), fx.make_layout((num_elems, 1), (1, 1)))


def scalar_load(view, idx: fx.Int64, dtype) -> fx.Float32:
    """Load one scalar element from a buffer-resource view."""
    copy = fx.make_copy_atom(fx.UniversalCopy32b(), dtype)
    reg  = fx.make_rmem_tensor(1, dtype)
    fx.copy_atom_call(copy, fx.slice(view, (idx, fx.Int64(0))), reg)
    return fx.Vector(fx.memref_load_vec(reg))[0]


def scalar_store(val, view, idx: fx.Int64, dtype):
    """Store one scalar element to a buffer-resource view."""
    copy = fx.make_copy_atom(fx.UniversalCopy32b(), dtype)
    reg  = fx.make_rmem_tensor(1, dtype)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype), reg)
    fx.copy_atom_call(copy, reg, fx.slice(view, (idx, fx.Int64(0))))
```

### bf16 row builder from raw i64 pointer

Used when the kernel receives base pointers as `Int64` (e.g. from a folded pointer argument) instead of a full `fx.Tensor`:

```python
import ctypes
import flydsl.expr as fx
from flydsl.expr import rocdl

def _bf16_row_view(base_i64: fx.Int64, num_elems: int, nbytes: int):
    """
    Build a (1, num_elems) bf16 buffer-resource view from a raw i64 base pointer.
    nbytes = num_elems * 2 (for bf16).
    alignment=2 tells the compiler this is bf16-aligned (16-bit).
    """
    ptr_ty = fx.PointerType.get(fx.BFloat16, address_space=1, alignment=2)
    ptr    = fx.IntToPtr(base_i64, ptr_ty)
    buf    = fx.rocdl.make_buffer_tensor(ptr, num_records_bytes=fx.Int64(nbytes))
    return fx.make_view(fx.get_iter(buf), fx.make_layout((1, num_elems), (num_elems, 1)))
```

### Tiled store: f32 register list → bf16 global

```python
def _store_bf16_tiled(
    vals:    list,       # list of f32 SSA values (len = VEC)
    dst:     fx.Tensor,  # destination bf16 tensor (already partitioned to this block/thread)
    copy:    object,     # TiledCopy object
    vec:     int,        # VEC width
):
    """
    Convert a Python list of f32 SSA values to a bf16 fragment and
    store it through a TiledCopy to global memory.
    This is the canonical epilogue pattern for norm/quant kernels.
    """
    # Build a f32 vector, truncate to bf16
    v_f32  = fx.Vector.from_elements(vals, fx.Float32)
    v_bf16 = v_f32.to(fx.BFloat16)

    # Write into a register-memory fragment
    frag = fx.make_rmem_tensor(vec, fx.BFloat16)
    fx.memref_store_vec(v_bf16, frag)

    # Tiled-copy fragment to global (dst must be pre-partitioned by get_slice)
    fx.copy(copy, frag, dst)


# Example usage inside a kernel:
@flyc.kernel
def norm_store_example(
    X:   fx.Tensor,   # (T, D) f32 input
    Out: fx.Tensor,   # (T, D) bf16 output
    T:   fx.Int32,
    D:   fx.Constexpr[int],
):
    VEC = 8
    tid = gpu.thread_id("x")
    row = gpu.block_id("x")

    # Load f32 row
    tX   = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(X)),
                        fx.make_layout((T, D), (D, 1)))
    copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    rX   = fx.make_rmem_tensor(VEC, fx.Float32)
    fx.copy_atom_call(copy, fx.slice(tX, (row, tid * fx.Int64(VEC))), rX)
    vx   = fx.Vector(fx.memref_load_vec(rX))

    # Compute (e.g. RMSNorm scale)
    norm_vals = [vx[i] * fx.Float32(0.5) for i in range_constexpr(VEC)]

    # Store as bf16 via the tiled pattern
    tOut = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Out)),
                        fx.make_layout((T, D), (D, 1)))
    copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    _store_bf16_tiled(norm_vals, fx.slice(tOut, (row, tid * fx.Int64(VEC))),
                      copy_bf16, VEC)
```

### Important: avoid `from __future__ import annotations`

The `qk_norm_rope_quant.py` source explicitly forbids this at the top of any FlyDSL kernel file:

```python
# DO NOT add this line to files containing FlyDSL kernels:
# from __future__ import annotations   # PEP 563 — breaks FlyDSL type detection!

# Why: PEP 563 stringifies all annotations at import time, so fx.Int32 / fx.Float32
# parameter annotations become strings instead of types. FlyDSL's runtime type
# detection fails to recognise them, triggering a fresh ~30-70 ms JIT compile
# per call with a unique batch size — instead of a cache hit.
```

---

## 27. Example: Testing FlyDSL Kernels

Patterns from the aiter test suite for validating correctness and performance.

### ptr_arg + _run_compiled (hot-path test pattern)

```python
import torch
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, _run_compiled

def run_my_kernel(kernel_exe, out, a, b, M, N, K, stream=None):
    """
    Thin wrapper that uses _run_compiled to avoid DLPack on hot paths.
    Call this from both tests and production code.
    """
    s = fx.Stream(stream or torch.cuda.current_stream())
    _run_compiled(kernel_exe, ptr_arg(out), ptr_arg(a), ptr_arg(b), M, N, K, s)
```

### Reference comparison with tolerances

```python
def check_allclose(ref: torch.Tensor, got: torch.Tensor,
                   label: str, atol: float = 1.0, rtol: float = 0.05,
                   max_err_ratio: float = 0.05):
    """
    Assert that `got` matches `ref` within tolerances.
    max_err_ratio: fraction of elements allowed to exceed atol/rtol.
    """
    assert not got.isnan().any(), f"{label}: output has NaN"
    assert not got.isinf().any(), f"{label}: output has Inf"

    close  = torch.isclose(ref.float(), got.float(), atol=atol, rtol=rtol)
    err    = (~close).float().mean().item()
    assert err <= max_err_ratio, (
        f"{label}: {err:.1%} of elements exceed tolerance "
        f"(atol={atol}, rtol={rtol}, max_ratio={max_err_ratio:.0%})"
    )
```

### Benchmark utility

```python
def bench_kernel(fn, warmup: int = 5, rep: int = 25,
                 label: str = "kernel") -> None:
    """Benchmark a FlyDSL kernel and print GB/s and TFLOPS."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms_median = start.elapsed_time(end) / rep
    print(f"{label}: {ms_median:.3f} ms")
```

### sorted_ids packing/unpacking (MoE test helper)

```python
def make_sorted_ids(token_ids: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
    """
    Pack routing indices into the format expected by silu_and_mul_fq and MoE kernels.
    Upper 8 bits = slot_id (output row), lower 24 bits = token_id (input row).
    """
    assert token_ids.max() < (1 << 24), "token_id exceeds 24-bit range"
    assert slot_ids.max()  < (1 << 8),  "slot_id exceeds 8-bit range"
    return ((slot_ids.int() << 24) | (token_ids.int() & 0xFFFFFF)).to(torch.int32)


def decode_sorted_ids(packed: torch.Tensor):
    """Inverse of make_sorted_ids — for debugging."""
    token_ids = packed & 0xFFFFFF
    slot_ids  = (packed >> 24) & 0xFF
    return token_ids, slot_ids


# Example: construct sorted_ids for a batch of T tokens, topk=2, E=8 experts
import torch

T, K, E = 32, 2, 8
topk_ids   = torch.randint(0, E, (T, K), dtype=torch.int32)   # (T, K) expert indices
slot_ids   = torch.arange(T * K, dtype=torch.int32)           # sequential output slots
token_ids  = torch.arange(T, dtype=torch.int32).repeat_interleave(K)
sorted_ids = make_sorted_ids(token_ids, slot_ids % (1 << 8))
```

### Parametrized pytest example

```python
import pytest
import torch
from aiter.ops.flydsl import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not is_flydsl_available(),
    reason="FlyDSL not available on this GPU",
)

@pytest.mark.parametrize("T,D", [(64, 4096), (128, 8192), (256, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_my_flydsl_kernel(T, D, dtype):
    torch.manual_seed(42)
    A   = torch.randn(T, D, dtype=dtype, device="cuda")
    Out = torch.empty_like(A)

    # Run kernel under test
    my_kernel(A, Out, T, D)

    # Reference
    ref = my_reference(A)

    check_allclose(ref, Out, label=f"T={T} D={D} dtype={dtype}", atol=1e-2, rtol=1e-2)
```

### AOT cache-miss detection in tests

```python
from aiter.aot.flydsl.common import fail_on_aot_cache_miss

def test_with_aot_cache(check_aot: bool = True):
    A = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
    # ...setup...

    if check_aot:
        ctx = fail_on_aot_cache_miss()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        result = my_flydsl_kernel(A, ...)

    assert result is not None
```

---

## 28. Reference: FlyDSL vs Triton vs Gluon

| Aspect | FlyDSL | Triton | Gluon |
|---|---|---|---|
| Layout control | Explicit `(Shape, Stride)` algebra | Implicit via block pointers | Implicit |
| Tiling | Manual divide/product | Auto with `tl.program_id` | Auto-tiling |
| Memory access | Copy atoms, buffer load, TiledCopy | `tl.load` / `tl.store` | `gluon.load` / `gluon.store` |
| Matrix math | `fx.make_mma_atom` + `fx.gemm` | `tl.dot` | `gluon.dot` |
| Shared memory | Explicit `@fx.struct` + `SharedAllocator` | Implicit scratchpad | Implicit |
| Compilation target | gfx942, gfx950, gfx1250 | General (AMD + NVIDIA) | General |
| Abstraction level | Low (near hardware) | Medium | Medium–High |
| Best for | Maximum control, layout-sensitive kernels | Productivity, portability | Portability |

FlyDSL requires more code than Triton for equivalent operations. The trade-off: you express exactly how data is arranged in registers and LDS, which banks threads map to, and what copy width the hardware uses. For layout-sensitive kernels (GEMM, MoE, flash attention) that extra verbosity directly translates into performance.

---

## 29. Quick-Reference Index

| Task | Section |
|---|---|
| Install, env vars, cache management | §2 |
| Layout construction, crd2idx, idx2crd | §3.1–3.2 |
| divide/product/coalesce/slice algebra | §3.3–3.5 |
| Minimal kernel and launch wrapper | §4.1 |
| 128-bit vectorized loads | §4.2 |
| Constexpr vs runtime parameter types | §5 |
| Thread/wave decomposition, barrier | §6 |
| Compile-time vs runtime branches | §7.1 |
| Branches with side effects via `@flyc.jit` | §7.2 |
| Scalar Int32/Int64/Float32 types | §8.1 |
| Vector splat, bitcast, operators, abs | §8.2–8.3 |
| 32/64/128-bit copy atoms | §9.1 |
| AMD buffer resource V# loads | §9.2 |
| TiledCopy for 2D cooperative copies | §9.3 |
| gfx1250 TDM async DMA | §9.4 |
| SharedAllocator + @fx.struct | §10.1 |
| Ping-pong LDS for pipelining | §10.2 |
| MFMA (wave64) FP16/FP8/INT8 | §11.1 |
| WMMA (wave32) FP8/INT4/MX-scaled | §11.2 |
| Building A/B/C fragments | §11.3 |
| Warp reduction (wave64 XOR shuffle) | §12.1 |
| Block reduction via LDS partials | §12.2 |
| Loop-carried state, `init=`, pitfalls | §13 |
| Complete naive GEMM example | §14 |
| Complete tiled GEMM + MFMA example | §15 |
| `_run_compiled` hot-path dispatch | §16 |
| AOT compilation, CSV configs, cache validation | §17 |
| `flydsl_hgemm` API | §18.1 |
| `flydsl_moe_stage1/2` API | §18.2 |
| `flydsl_qk_norm_rope_quant` API | §18.3 |
| `flydsl_mla_reduce_v1` API | §18.4 |
| `flydsl_flash_attn_func` API | §18.5 |
| Autotuning and `do_bench` | §19 |
| IR dump, printf, ATT debug info | §20 |
| 10 common pitfalls with fixes | §21 |
| SiLU, SWiGLU, sigmoid/tanh via exp2/rcp | §22 |
| Full bf16 SWiGLU kernel with TiledCopy | §23 |
| Fused SiLU + FP4/FP8 MX quantization | §24 |
| MoE top-k reduction kernel | §25 |
| Scalar view, raw-pointer view, bf16 tiled store | §26 |
| Testing patterns: ptr_arg, check_allclose, bench, sorted_ids | §27 |
| FlyDSL vs Triton vs Gluon | §28 |
