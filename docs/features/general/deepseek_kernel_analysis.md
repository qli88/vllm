# DeepSeek V3/V4 Kernel and Operation Analysis

Every kernel and operation referenced in the sparse MLA pipeline, explained in detail with shapes, algorithms, and how each fits in the larger system.

## Table of Contents

1. [Input Projections](#1-input-projections)
   - [fused_qkv_a_proj / DeepSeekV2FusedQkvAProjLinear](#11-fused_qkv_a_proj--deepseekv2fusedqkvaproj)
   - [wk_weights_proj (V3) / weights_proj + compressor (V4)](#12-wk_weights_proj-v3--weights_proj--compressor-v4)
   - [fused_q_kv_rmsnorm](#13-fused_q_kv_rmsnorm)
2. [Indexer Q Quantization](#2-indexer-q-quantization)
   - [fused_indexer_q_rope_quant (V3/FP8)](#21-fused_indexer_q_rope_quant-v3fp8)
   - [fused_indexer_q_rope_quant (V4/MXFP4)](#22-fused_indexer_q_rope_quant-v4mxfp4)
3. [Indexer K Cache Operations](#3-indexer-k-cache-operations)
   - [indexer_k_quant_and_cache](#31-indexer_k_quant_and_cache)
   - [cp_gather_indexer_k_quant_cache](#32-cp_gather_indexer_k_quant_cache)
4. [V4 Compressor Operations](#4-v4-compressor-operations)
   - [save_partial_states](#41-save_partial_states)
   - [compress_norm_rope_store_triton](#42-compress_norm_rope_store_triton)
   - [get_compressed_slot_mapping](#43-get_compressed_slot_mapping)
5. [MQA Logit Kernels (DeepGEMM)](#5-mqa-logit-kernels-deepgemm)
   - [fp8_fp4_mqa_logits (prefill)](#51-fp8_fp4_mqa_logits-prefill)
   - [fp8_fp4_paged_mqa_logits (decode)](#52-fp8_fp4_paged_mqa_logits-decode)
   - [get_paged_mqa_logits_metadata](#53-get_paged_mqa_logits_metadata)
6. [Top-K Selection Kernels](#6-top-k-selection-kernels)
   - [top_k_per_row_prefill](#61-top_k_per_row_prefill)
   - [top_k_per_row_decode](#62-top_k_per_row_decode)
   - [cooperative_topk](#63-cooperative_topk)
   - [persistent_topk](#64-persistent_topk)
7. [DCP Merge](#7-dcp-merge)
   - [pack_dcp_topk_candidates_cutedsl](#71-pack_dcp_topk_candidates_cutedsl)
   - [stable_topk_from_gathered_candidates_cutedsl](#72-stable_topk_from_gathered_candidates_cutedsl)
8. [Main MLA Attention (V3)](#8-main-mla-attention-v3)
   - [concat_mla_q](#81-concat_mla_q)
   - [W_UK_T and W_UV weight absorption](#82-w_uk_t-and-w_uv-weight-absorption)
   - [FlashMLASparseImpl.forward_mqa](#83-flashmlasparseimplforward_mqa)
   - [flash_mla_with_kvcache / flash_mla_sparse_fwd](#84-flash_mla_with_kvcache--flash_mla_sparse_fwd)
9. [Main MLA Attention (V4)](#9-main-mla-attention-v4)
   - [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (SWA insert)](#91-fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert)
   - [SWA index computation](#92-swa-index-computation)
   - [compute_global_topk_indices_and_lens](#93-compute_global_topk_indices_and_lens)
   - [dequantize_and_gather_k_cache](#94-dequantize_and_gather_k_cache)
   - [combine_topk_swa_indices](#95-combine_topk_swa_indices)
   - [fused_inv_rope_fp8_quant](#96-fused_inv_rope_fp8_quant)
   - [deep_gemm_fp8_o_proj (wo_a + wo_b)](#97-deep_gemm_fp8_o_proj-wo_a--wo_b)
10. [Spec-Decode Padding Helpers](#10-spec-decode-padding-helpers)
    - [pack_seq_triton / unpack_seq_triton](#101-pack_seq_triton--unpack_seq_triton)

---

## 1. Input Projections

### 1.1 `fused_qkv_a_proj` / `DeepSeekV2FusedQkvAProjLinear`

**File:** `vllm/model_executor/models/deepseek_v2.py:904`

**What it is:** A single weight matrix `[7168 → 2112]` that replaces two separate projections (`q_a_proj [7168→1536]` and `kv_a_proj_with_mqa [7168→576]`) with one fused GEMM. The output is then logically sliced:

```
hidden_states [T, 7168]
      × weight [2112, 7168]
      ─────────────────────────────────────────────────────
      output [T, 2112]  where 2112 = 1536 + 512 + 64
        ├─ [T, 0:1536]      → q_c   (Q low-rank activations)
        ├─ [T, 1536:2048]   → kv_c  (KV latent, 512-dim)
        └─ [T, 2048:2112]   → k_pe  (shared rope K, 64-dim)
```

**Why fused:** `q_c` and `kv_c + k_pe` both read from the same `hidden_states`. Fusing them into one matrix (`[2112, 7168]`) halves the memory bandwidth load on `hidden_states` and schedules a single larger GEMM (better tensor core utilization) instead of two smaller ones.

**Low-batch optimization:** For `num_tokens ≤ 16` (typical decode batches) and `weight.shape = [2112, 7168]` on SM90/SM100, the `dsv3_fused_a_gemm` kernel is used instead of cuBLAS `F.linear`. This is a low-latency GEMM with PDL (Persistent Data Loading), tuned for the memory-bandwidth-bound regime at tiny batch sizes.

```python
if 0 < num_tokens <= 16:
    ops.dsv3_fused_a_gemm(output, input_, weight.T)   # PDL path
else:
    F.linear(input_, weight)                           # cuBLAS path
```

**What each slice means:**

- `q_c [T, 1536]`: The Q low-rank bottleneck (LoRA-style). Instead of projecting `[7168 → 128×192=24576]` (MHA full), DSv3 goes through a 1536-dim bottleneck to reduce weight count. After `q_a_layernorm` → `q_b_proj`, produces full Q heads.
- `kv_c [T, 512]`: The KV latent — the core of MLA. One 512-dim vector per token is cached; at attention time `kv_b_proj` expands it to all K/V heads on the fly. The 512-dim is the entire KV memory footprint per token per layer.
- `k_pe [T, 64]`: The shared rope K. Since RoPE depends only on position, the rope part of K can be a single MQA vector (one per token, shared across all 128 heads). Stored alongside `kv_c` to avoid recomputing from `kv_c` at each decode step.

---

### 1.2 `wk_weights_proj` (V3) / `weights_proj` + compressor (V4)

**V3 — `wk_weights_proj`** (`deepseek_v2.py:675`):

A `MergedColumnParallelLinear [7168 → 128+64]` — two outputs in one GEMM:

```
hidden_states [T, 7168]
      ├─ k        [T, 128]   ← the indexer's 128-dim scoring K
      └─ weights  [T,  64]   ← one scalar per indexer head
```

The two halves have conceptually different roles: `k` is the content representation of a token for scoring; `weights` is a learned "pre-score" that says how much head `h` should attend to this token independently of the query (derived purely from the hidden state, without seeing Q). They are fused because they both read from `hidden_states` and the combined GEMM is more efficient.

**Why a single K for 64 heads:** The indexer uses MQA (multi-query attention) — all 64 scoring heads share the same K vector. Only Q varies per head. This is possible because the K here is just used for cheap approximate scoring, not for the actual value computation. Having a single K reduces cache size 64× and the GEMM cost similarly.

**V4 — separate `weights_proj` and compressor:**

```
hidden_states [T, 7168]
      ├─ weights_proj [7168 → 64]  → indexer_weights [T, 64]
      └─ compressor.fused_wkv_wgate [7168 → 4×128=512] → kv_score [T, 512]
           └─ compress: 4 tokens → 1 compressed K in indexer cache
```

V4 separates weights and K because K now comes from a compressor (not a direct projection), while weights are still a direct function of `hidden_states`. They run on different aux streams.

---

### 1.3 `fused_q_kv_rmsnorm`

**File:** `vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py`

**V4 only.** After `fused_wqa_wkv` produces `qr [T,1536]` and `kv [T,512]`, both need RMSNorm before further use. Doing them in two separate kernel launches would be wasteful.

**Triton kernel** `_fused_q_kv_rmsnorm_kernel`, grid `(T, 2)`:

- `pid_task=0`: normalize `qr[t, :]` with `q_norm.weight`
- `pid_task=1`: normalize `kv[t, :]` with `kv_norm.weight`

Algorithm per task (fp32 accumulation):

```
load row as fp32
variance = sum(x[i]^2) / SIZE
rrms = 1 / sqrt(variance + eps)
y[i] = x[i] * rrms * weight[i]
store y
```

Both the Q and KV norms run in parallel across different thread blocks in the same kernel launch.

---

## 2. Indexer Q Quantization

### 2.1 `fused_indexer_q_rope_quant` (V3/FP8)

**File (V3):** `vllm/model_executor/layers/sparse_attn_indexer.py:126`
**File (V4 FP8):** `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:70`

**Triton kernel**, grid `(T, 64)` — one program per `(token, head)`.

**What it does in one kernel launch:**

1. Load Q head (128 values: 64 rope + 64 nope)
2. Apply RoPE on the rope portion (NeoX or GPT-J interleaved depending on config)
3. Compute per-token per-head absmax → e8m0 (power-of-2) scale
4. Quantize all 128 values to FP8
5. Absorb scale into `weights_out`

**RoPE (V3 NeoX-style):**

```
x0 = q[0:32],  x1 = q[32:64]          # two halves
r0 = x0*cos(pos) - x1*sin(pos)  → bf16 → fp32   # roundtrip for numerics
r1 = x1*cos(pos) + x0*sin(pos)  → bf16 → fp32
```

**RoPE (V4 GPT-J interleaved):**

```
x_even = q[64::2],  x_odd = q[65::2]   # pairs (64 rope dims start at offset 64)
r_even = x_even*cos - x_odd*sin  → bf16 → fp32
r_odd  = x_odd*cos  + x_even*sin → bf16 → fp32
```

**FP8 quantization (e8m0 scale):**

```
amax = max(|rope_values|, |nope_values|)
scale_raw = max(amax, 1e-10) / fp8_max          # fp8_max=448 for e4m3fn
q_scale = 2^ceil(log2(scale_raw))               # round UP to nearest power-of-2
q_fp8[:] = clamp(values / q_scale, -fp8_max, fp8_max).to(fp8_dtype)
```

The power-of-2 rounding (e8m0 format) means dividing by `q_scale` is exact in floating point — just an exponent field decrement. This preserves maximum precision when clamping into the FP8 range.

**Scale absorption (FP8 path):**

```
weights_out[t, h] = weights[t, h] × q_scale × (1/√128) × (1/√64)
                  = weights[t, h] × q_scale × softmax_scale × head_scale
```

By pre-multiplying `q_scale` into `weights_out`, the downstream MQA GEMM needs no per-element scale lookup — it just multiplies the scalar `weights_out[t,h]` into its tile accumulator in the epilogue.

**Why head_scale `1/√64`?** The scoring logit is `Σ_h score_h × w_h`, a weighted sum over 64 heads. Without normalization, more heads → larger logits → sharper softmax → less stable gradients during training. The `1/√64` factor normalizes for the number of scoring heads.

---

### 2.2 `fused_indexer_q_rope_quant` (V4/MXFP4)

**File:** `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:179`

**Triton kernel**, grid `(T, 64)`. Same RoPE as V4 FP8, but quantization uses MXFP4 (4-bit E2M1) instead of FP8.

**MXFP4 format:** 4 bits per value (E2M1: 2 exponent bits, 1 mantissa bit, 1 sign bit), 32 values per block, 2 values packed per byte (low nibble = first value, high nibble = second), 1 ue8m0 scale byte per block.

**Quantization per 32-element block:**

```
amax = max(|x_lo|, |x_hi|)                       # x_lo = even positions, x_hi = odd
amax = max(amax, 6.0 × 2^-126)                   # floor to avoid log2(0)
scale = 2^ceil(log2(amax / 6.0))                  # ue8m0: 2^(biased exponent - 127)
ue8m0 = (ceil(log2(amax/6.0)) + 127).to(uint8)   # stored as biased exponent byte
packed = fp32_to_e2m1(x_lo / scale, x_hi / scale) # two nibbles per byte
```

Constant `6.0 = max(e2m1 value)` (the largest representable E2M1 value is 6.0, not the typical fp8 448).

**Scale NOT absorbed into weights (MXFP4 path):**

```
weights_out[t, h] = weights[t, h] × (1/√128) × (1/√64)
# q_scale is NOT folded — there are 4 per-block scales per head (128/32=4 blocks)
# which cannot be collapsed into a single per-head scalar
```

The downstream `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` receives `(q_packed, q_scale_int32)` as a tuple and applies per-block dequantization internally.

---

## 3. Indexer K Cache Operations

### 3.1 `indexer_k_quant_and_cache`

**File:** `vllm/_custom_ops.py:2796` (Python), `csrc/libtorch_stable/cache_kernels.cu:549` (CUDA)

**What it does:** Takes the freshly-computed K vectors for all tokens in the current batch, quantizes them to FP8, and writes them into the paged indexer KV cache.

**Signature:**

```python
indexer_k_quant_and_cache(
    k: [T, 128]          bf16,    # K vectors for current batch
    kv_cache: [B, 64, 132] uint8, # paged cache: B blocks, 64 slots/block, 132 bytes/slot
    slot_mapping: [T]    int64,   # physical slot for each token
    quant_block_size: int,        # 128 (one block per 128-dim K)
    kv_cache_dtype: str,          # "ue8m0"
)
```

**CUDA kernel** `indexer_k_quant_and_cache_kernel`, grid `(T, ceil(128/VEC_SIZE/4))`, block `(32, 4)`:

Each thread block handles one token (blockIdx.x) and a 4-element slice of the 128-dim K.

Step by step for token `t`:

1. Each warp of 32 threads cooperatively loads K values in 4-element VEC chunks
2. **Warp-level absmax:** each thread contributes its 4 absolute values, then `__shfl_xor_sync` reduces across 32 lanes to get the max across the whole 128-dim K
3. **Scale computation (ue8m0):**
   ```
   scale_raw = max(amax, 1e-4) / fp8_max
   exponent  = ceil(log2(scale_raw))
   scale     = exp2f(exponent)             # power-of-2
   ```
4. **FP8 quantize:** `fp8_val = clamp(k_val / scale, -fp8_max, fp8_max)` cast to `float8_e4m3fn`
5. **Write to cache:** using `slot_mapping[t]` to find `block_idx = slot // 64` and `block_offset = slot % 64`:
   - FP8 values → `kv_cache[block_idx, block_offset, 0:128]`
   - Scale (float32) → `kv_cache[block_idx, block_offset, 128:132]` (4 bytes)

**Cache layout per slot (132 bytes):**

```
bytes [0..127]   : 128 FP8 values (e4m3fn)
bytes [128..131] : float32 ue8m0 block scale (one block covers all 128 dims)
```

The block size is always 128 (the full K dim), so there is exactly one scale per cached K vector.

**Used only in V3.** In V4 the compressor writes K instead (`skip_k_cache_insert=True`).

---

### 3.2 `cp_gather_indexer_k_quant_cache`

**File:** `vllm/_custom_ops.py:2852` (Python), `csrc/libtorch_stable/cache_kernels.cu:612` (CUDA)

**What it does:** For prefill, all historical K vectors for the current batch of requests need to be gathered from the paged cache into a contiguous flat buffer that the MQA GEMM can process.

**Signature:**

```python
cp_gather_indexer_k_quant_cache(
    kv_cache:    [num_blocks, 64, 132] uint8,  # paged FP8 cache
    dst_k:       [N, 128]            fp8,      # output: flat gathered K values
    dst_scale:   [N, 4]              uint8,    # output: flat scales (as float32 bytes)
    block_table: [B, max_blocks]     int32,    # maps (request, block_idx) → physical block
    cu_seq_lens: [B+1]               int32,    # cumulative sequence lengths [0, L0, L0+L1, ...]
)
```

`N = sum(seq_lens)` — total tokens across all prefill requests in this chunk.

**CUDA kernel** `cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>`, grid `(ceil(N/BY), ceil(128/(8×16)))`, block `(8, BY)`:

- `BLOCK_Y_SIZE` is selected by token count: 1/2/4/8/16/32 for <32/<64/<128/<256/<512/else — balances occupancy vs shared memory use
- Each thread row handles one token, each column group handles 16 bytes (128-bit) of the 128-dim K

For token `t`:

1. Determine which request it belongs to by scanning `cu_seq_lens` in shared memory (cooperative binary search across BLOCK_Y_SIZE rows)
2. Compute `inbatch_seq_idx = t - cu_seq_lens[batch_idx]`
3. Look up `block_number = block_table[batch_idx, inbatch_seq_idx // 64]`
4. `slot_offset = inbatch_seq_idx % 64`
5. Load 16 bytes from `kv_cache[block_number * block_stride + slot_offset * 128 + head_idx]`
6. Store to `dst_k[t * 128 + head_idx]` via `float4` store
7. Thread 0 reads the float32 scale from `kv_cache[... + 128]` and stores to `dst_scale[t * 4]`

**Result:** A dense `[N, 128]` FP8 matrix and `[N, 4]` scale matrix, ready for `fp8_fp4_mqa_logits`.

---

## 4. V4 Compressor Operations

### 4.1 `save_partial_states`

**File:** `vllm/models/deepseek_v4/common/ops/save_partial_states.py`

**What it does:** Stores the raw `(kv, score)` pair for each input token into the compressor's sliding-window state cache, with APE (Absolute Positional Embedding) added to the score.

**Signature:**

```python
save_partial_states(
    kv:          [T, coff×head_dim]    fp32,  # KV portion from fused_wkv_wgate
    score:       [T, coff×head_dim]    fp32,  # score portion
    ape:         [compress_ratio, coff×head_dim]  fp32,  # learned per-position embedding
    positions:   [T]                   int64,
    state_cache: [num_blocks, block_size, 2×coff×head_dim]  fp32,  # paged state
    slot_mapping:[T]                   int64,  # physical slot for each token
    block_size:  int,
    compress_ratio: int,               # e.g., 4 for C4A
)
```

**Triton kernel** `_save_partial_states_kernel`, grid `(T,)`, one program per token:

```
slot_id = slot_mapping[t]
if slot_id < 0: return  (padding token)

block_idx    = slot_id // block_size
pos_in_block = slot_id % block_size
base = state_cache[block_idx, pos_in_block, :]   (shape [2×coff×head_dim])

# Write KV into first half:
base[0 : coff×head_dim] = kv[t, :]

# Write score+APE into second half:
ape_row = ape[position % compress_ratio, :]
base[coff×head_dim : 2×coff×head_dim] = score[t, :] + ape_row
```

**What APE does:** Each position within a compression group (0, 1, 2, 3 for compress_ratio=4) gets a distinct learned embedding added to its score. This lets the compressor learn different importance weights for early vs late positions within a group — e.g., the last token before a compression boundary might get higher weight since it contributes most recently.

**State cache layout (C4A, coff=2, head_dim=128):**

```
[block, slot, :] = [kv_half_a[256], kv_half_b[256], score_half_a[256], score_half_b[256]]
                    total: 1024 float32 values = 4096 bytes per state entry
```

This state is a sliding window — it stores the most recent `coff×compress_ratio = 8` states per compression slot (block_size=4 for C4A).

---

### 4.2 `compress_norm_rope_store_triton`

**File:** `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`

**What it does:** For every token `t` where `(position+1) % compress_ratio == 0`, reads the last `coff×compress_ratio` saved states from the state cache, compresses them into one vector, applies RMSNorm and RoPE, quantizes to FP8 (or MXFP4), and writes to the indexer KV cache.

**Three kernel variants:**

#### Variant A: head_dim=128, FP8 cache — `_fused_kv_compress_norm_rope_insert_indexer_attn`

Grid `(T,)`, one program per token. Fires only when `(position+1) % compress_ratio == 0`.

**Step 1 — State gather:**

```
compressed_slot = position // compress_ratio
For j in [0 .. coff×compress_ratio - 1]:
    slot = look up state_cache via block_table at compressed_slot-boundary position
    kv_state[j, :]    = state_cache[block, offset, :head_dim]     fp32
    score_state[j, :] = state_cache[block, offset, head_dim:]     fp32
```

**Step 2 — Softmax-gated compression:**

```
gate = softmax(score_state, dim=0)   # [coff×compress_ratio, head_dim]
compressed_kv = sum(kv_state × gate, axis=0)    # [head_dim]
```

For C4A with overlap (`coff=2`): the first 4 and last 4 states form two groups, each gated separately, then summed.

**Step 3 — RMSNorm:**

```
variance = sum(compressed_kv^2) / head_dim
rrms = 1 / sqrt(variance + eps)
normed = compressed_kv * rrms * rms_norm_weight
```

**Step 4 — GPT-J RoPE:**

```
rope_pos = (position // compress_ratio) * compress_ratio   # aligned position
cos, sin = cos_sin_cache[rope_pos, :]
# interleaved pairs on rope dims:
even = normed[rope_start::2]
odd  = normed[rope_start+1::2]
normed[rope_start::2]   = even*cos - odd*sin
normed[rope_start+1::2] = odd*cos  + even*sin
```

**Step 5 — FP8 quantize and write:**

```
amax = max(|normed|)
q_scale = exp2(ceil(log2(amax / fp8_max)))
k_fp8 = clamp(normed / q_scale, -fp8_max, fp8_max).to(fp8)

# Write to indexer k_cache at compressed_slot:
block = compressed_slot // block_size
offset = compressed_slot % block_size
k_cache[block, offset, :128]  = k_fp8      (128 FP8 bytes)
k_cache[block, offset, 128:]  = float32_as_bytes(q_scale)  (4 bytes)
```

#### Variant B: head_dim=512, FP8 UE8M0 — `_fused_kv_compress_norm_rope_insert_sparse_attn`

Same flow but for the 512-dim main MLA KV cache. Key differences:

- 4 warps (instead of 1) for parallelism over the 512-dim
- NoPE portion (448 dims) → block-scaled FP8, 7 ue8m0 scale bytes + 1 padding = 8 bytes
- RoPE portion (64 dims) → stored as BF16 (not quantized, 128 bytes)
- Total per slot: 448 + 128 + 8 = **584 bytes** (the `fp8_ds_mla` format)

#### Variant C: head_dim=128, MXFP4 — `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`

Same as variant A but quantizes to MXFP4 per 32-element block after RoPE:

```
for each block of 32 values (even and odd halves):
    amax = max(|even|, |odd|)
    ue8m0 = ceil(log2(amax / 6.0)) + 127
    packed = [fp32_to_e2m1(even/scale), fp32_to_e2m1(odd/scale)]   # nibble pairs
write packed uint8 + ue8m0 scale bytes
```

---

### 4.3 `get_compressed_slot_mapping`

**File:** `vllm/v1/attention/backends/mla/compressor_utils.py:53`

**What it does:** Computes which physical cache slot each query token writes to in the compressed KV cache. Only tokens at positions `(pos+1) % compress_ratio == 0` write anything (they complete a group); others get slot `-1` (no write).

**Triton kernel** `_compressed_slot_mapping_kernel`, grid `(num_reqs,)`:

For each request:

```
For query token i (absolute position pos = prefix_len + i):
    if (pos + 1) % compress_ratio == 0:
        compressed_pos = pos // compress_ratio
        block = block_table[req, compressed_pos // block_size]
        slot  = block * block_size + compressed_pos % block_size
        out[token_offset + i] = slot
    else:
        out[token_offset + i] = -1   (PAD_ID, no write)
```

**Result:** `slot_mapping [T] int64`. For a 8-token prefill with `compress_ratio=4`, only tokens at positions 3 and 7 get valid slots; tokens 0,1,2,4,5,6 get `-1`.

`save_partial_states` uses this: tokens with `slot_id < 0` are skipped (they still save their state into the state_cache at their own slot, but they don't trigger compression).

---

## 5. MQA Logit Kernels (DeepGEMM)

### 5.1 `fp8_fp4_mqa_logits` (prefill)

**File:** `vllm/utils/deep_gemm.py:499` (wrapper) → `deep_gemm.fp8_fp4_mqa_logits` (JIT-compiled kernel)

**What it does:** The core scoring GEMM for prefill. Computes attention logits for each query token against all its causally-valid historical K tokens, using FP8 (or MXFP4) precision and folding the per-head weight scalars into the output.

**Signature:**

```python
fp8_fp4_mqa_logits(
    q:              (q_values [M, H, D], q_scale_or_None),   # FP8: scale=None; MXFP4: scale=[M,H,4]
    kv:             (k_packed [N, D], k_scales [N, 4]),       # gathered flat K buffer
    weights:        [M, H]  fp32,                             # weights_out (with scale absorbed for FP8)
    cu_seqlen_ks:   [M]     int32,                            # per-token K range start
    cu_seqlen_ke:   [M]     int32,                            # per-token K range end (exclusive)
    clean_logits:   bool,
) → logits [M, N]  fp32
```

`M` = prefill query tokens, `N` = total gathered K tokens across the chunk, `H` = 64 indexer heads, `D` = 128.

**Algorithm:**

```
for each query token q (row) and KV token k (col):
    valid = (cu_seqlen_ks[q] <= k < cu_seqlen_ke[q])   # causal mask
    if valid:
        raw = dot(q_fp8[q, :, :], k_fp8[k, :])          # [H, D] × [D] = [H] FP8 accumulator
        logit[q, k] = sum_h(raw[h] × sk[k] × weights_out[q, h])
                    ≈ sum_h(dot(q_fp8, k_fp8) × sk × weights_out)
```

The `cu_seqlen_ks` / `cu_seqlen_ke` pair implements **ragged causal masking**: query token `q` at absolute position `p` may attend only to K positions `[row_start, row_start+p+1)` in the gathered buffer. Tokens earlier in the request's history are excluded. V4 uses the same kernel with `cu_seqlen_ke[i] = (p+1) // compress_ratio` to operate in compressed token space.

For FP8 Q: `weights_out` already contains `sq × softmax_scale × head_scale`, so no additional per-element scale is needed.
For MXFP4 Q: `q_scale [M, H, 4]` (4 blocks × 1 ue8m0 byte each, stored as int32) is passed separately; the kernel dequantizes per-block during accumulation.

The result `logits [M, N] fp32` has only the lower-causal-triangle filled; invalid positions are either 0 (if `clean_logits=True`) or untouched (if `False`; faster, and the top-K kernel uses `cu_seqlen_ke` to ignore them).

---

### 5.2 `fp8_fp4_paged_mqa_logits` (decode)

**File:** `vllm/utils/deep_gemm.py:565` (wrapper) → `deep_gemm.fp8_fp4_paged_mqa_logits` (JIT kernel)

**What it does:** Same scoring as the prefill logit kernel but operates **directly on the paged KV cache** without a prior gather step. This avoids the `O(N)` workspace that prefill needs.

**Signature:**

```python
fp8_fp4_paged_mqa_logits(
    q:               (q_values [B, next_n, H, D], q_scale_or_None),
    kv_cache:        [num_blocks, block_size, 1, 132]  uint8,  # paged FP8 cache
    weights:         [B×next_n, H]  fp32,
    context_lens:    [B, next_n]    int32,              # context length per token
    block_tables:    [B, max_blocks] int32,
    schedule_metadata,                                  # SM tile assignments
    max_model_len:   int,
    clean_logits:    bool,
) → logits [B×next_n, max_model_len]  fp32
```

`B` = decode batch size, `next_n` = 1 (plain decode) or >1 (spec decode).

**Algorithm:**

Instead of a flat buffer, the kernel walks the block table:

```
for each decode token (b, n):
    for block_idx in range(num_blocks_for_seq):
        block_number = block_tables[b, block_idx]
        for slot in range(block_size):
            kv_slot = kv_cache[block_number, slot, 0, :]   # 132 bytes
            k_fp8  = kv_slot[:128]
            k_scale = reinterpret_float(kv_slot[128:132])

            pos = block_idx * block_size + slot
            if pos < context_lens[b, n]:
                raw = dot(q[b, n, :, :], k_fp8)    # [H, D] × [D] → [H]
                logits[b*next_n+n, pos] = sum_h(raw[h] × k_scale × weights[b*next_n+n, h])
```

The `schedule_metadata` from `get_paged_mqa_logits_metadata` pre-assigns blocks to SMs to balance load — each SM gets a roughly equal share of the paged KV blocks to process. This avoids the naive approach where SM0 always handles the beginning of long sequences.

Output `logits [B×next_n, max_model_len] fp32`: valid range is `[0, context_lens[b,n])` per row; positions beyond the sequence length contain garbage (masked by the top-K kernel via `seq_lens`).

---

### 5.3 `get_paged_mqa_logits_metadata`

**File:** `vllm/utils/deep_gemm.py:544`

**What it does:** Pre-computes per-SM work assignments for the paged MQA logit kernel so that all SMs have roughly equal amounts of KV blocks to process.

```python
get_paged_mqa_logits_metadata(
    context_lens: [B, next_n]  int32,
    block_size:   int,
    num_sms:      int,
) → schedule_metadata [num_sms+1, 2]  int32
```

`schedule_metadata[i, :]` tells SM `i` which range of (request, block) pairs to process. The `+1` row is a fence (total work count).

This is needed because decode batches are irregular — request 0 might have 100 KV blocks while request 1 has 10,000. Without pre-scheduling, some SMs would sit idle while others are overloaded.

---

## 6. Top-K Selection Kernels

After scoring, we need to select the top-1024 positions from the logit matrix. Four kernels cover the different shape regimes.

### 6.1 `top_k_per_row_prefill`

**File:** `csrc/libtorch_stable/sampler.cu:725`

**Used when:** SM < 90, or rows too many for `cooperative_topk`, or `topk` not in {512,1024,2048}. The fallback prefill path.

**Signature:**

```cpp
top_k_per_row_prefill(
    logits:   [M, N]  float32,      # prefill logit matrix
    rowStarts:[M]     int32,         # cu_seqlen_ks per row
    rowEnds:  [M]     int32,         # cu_seqlen_ke per row
    indices:  [M, K]  int32,         # output: top-K indices per row
    numRows, stride0, stride1, topK
)
```

Two dispatch paths based on row count:

- **Insertion sort** (`rows < 12288`): `topKPerRowPrefill<512, false>` — one CUDA block per row, `K * sizeof(int32)` shared memory. Simple insertion sort into a sorted buffer of size K. O(N×K) but fast for small K and N.
- **Radix sort** (`rows ≥ 12288`): `topKPerRowPrefill<512, true>` — 2048-bin histogram approach:
  1. Build 2048-bin histogram over float values (using float-bit reinterpretation for order)
  2. Find the threshold bin containing the K-th largest value
  3. Scatter indices above threshold into output

The per-row variable range `[rowStarts[q], rowEnds[q])` implements causal masking — positions outside the range are never considered.

---

### 6.2 `top_k_per_row_decode`

**File:** `csrc/libtorch_stable/sampler.cu:661`

**Used when:** decode path, fallback when `topk` not in {512,1024,2048} or SM < 90.

**Signature:**

```cpp
top_k_per_row_decode(
    logits:  [B, N]  float32,
    next_n:  int,
    seqLens: [B] or [B, next_n]  int32,
    indices: [B, K]  int32,
    numRows, stride0, stride1, topK
)
```

Three dispatch paths based on `numColumns` (= `max_model_len`):

- **< 12288**: insertion sort, one block per row
- **< 200K**: radix sort, one block per row
- **≥ 200K**: two-pass split — 10 blocks per row produce partial top-K, then a merge kernel

The `seqLens` mask ensures only valid positions `[0, seqLens[b])` are considered; invalid positions have logit=-inf or are simply excluded.

---

### 6.3 `cooperative_topk`

**File:** `csrc/libtorch_stable/cooperative_topk.cu:105`, kernel in `cooperative_topk.cuh`

**Used when:** `num_rows ≤ 32`, `topk ∈ {512, 1024, 2048}`, SM ≥ 90, not SM120.

**Why a dedicated kernel?** For small decode batches (1–32 sequences), a multi-CTA cooperative approach with CUDA cluster features (distributed shared memory, TMA bulk loads) dramatically outperforms single-CTA radix sort. The kernel exploits Hopper SM-to-SM communication via DSMEM without going through L2.

**Algorithm** (`cooperative_topk_body`):

For each row, a cluster of `CS` CTAs cooperates (CS = 4/8/16 based on num_rows):

**Case 1 — trivial** (`seq_len ≤ K`): Row 0 fills output with `[0..seq_len-1, -1, ...]`.

**Case 2 — short** (`seq_len ≤ kHist4096MaxLen`): Single CTA, 4096-bin in-register histogram:

```
1. In registers: build 4096-bin histogram over float values (12-bit indexing)
2. CUB prefix sum to find threshold bin (bin containing the K-th element)
3. Scatter: store indices of values in threshold+ bins to output
```

**Case 3 — large**: Multi-CTA cooperative with TMA:

**Sub-case A — fused** (data fits in smem): All CTAs load their portion with TMA async bulk, build histogram in shared memory simultaneously, then reduce via DSMEM all-reduce, scatter.

**Sub-case B — two-pass**: TMA streaming double-buffered histogram pass, then streaming scatter pass.

**DSMEM all-reduce** `dsmem_hist_reduce<CS>`: each CTA has a partial histogram in smem; DSMEM allows direct writes to other CTAs' smem without atomic overhead. Result: global histogram in O(1) all-reduce.

**find_threshold**: Warp-level inclusive prefix sum over bins from the top, finds the bin index `t` where `prefix_sum(bins > t) < K ≤ prefix_sum(bins ≥ t)`. This identifies the threshold value.

**Scatter**: Above-threshold → directly to output. At-threshold (ties) → to `tie_buffer`. After main scatter, `tie_handle` resolves ties by writing the first remaining needed tie entries.

**Output format:** `indices [B, K] int32` — global physical token positions (not block-relative), `-1` for padding.

---

### 6.4 `persistent_topk`

**File:** `csrc/libtorch_stable/topk.cu`, kernel in `persistent_topk.cuh`

**Used when:** `topk ∈ {512, 1024, 2048}`, CUDA, batch size too large for `cooperative_topk`.

**Why "persistent":** The kernel launch has `total_ctas = num_groups × ctas_per_group` CTAs that stay live and process multiple rows via round-robin, avoiding kernel re-launch overhead.

**Per-row dispatch inside the kernel:**

- **`seq_len ≤ K`** — trivial fill.
- **`K < seq_len ≤ 8192`** — `histogram_2048_topk<K>`: 2048-bin 11-bit histogram in smem, CUB suffix sum, warp-aggregated atomics for scatter, then 4 rounds of FP32 radix refinement to break ties.
- **`8192 < seq_len ≤ 32768`** — `histogram_256_topk<K>`: 256-bin 8-bit (FP16 top byte) histogram, 4-pass suffix-sum + radix refinement, double-buffered smem index arrays.
- **`seq_len > 32768`** — `radix_topk<K, VEC_SIZE>`: multi-CTA-per-row, 4 rounds of 8-bit radix select. Uses `arrival_counter` / `output_counter` spin-wait barriers in global memory (stream-ordered via `cudaMemsetAsync`) to synchronize CTA-level partial results, vectorized float4/float2 loads.

The vectorized loads (float4 = 16 bytes) ensure memory bandwidth is saturated during the histogram pass at large sequence lengths.

---

## 7. DCP Merge

Decode Context Parallelism (DCP) splits the KV sequence across GPUs — rank 0 owns tokens 0, 2, 4,... and rank 1 owns tokens 1, 3, 5,... (or a larger interleaved block). Each rank scores only its local shard. After local top-K, the ranks exchange candidates and select the global top-K.

### 7.1 `pack_dcp_topk_candidates_cutedsl`

**File:** `vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py:34`

**What it does:** Converts each rank's local top-K indices (local positions within its 1/N KV shard) into `(score, global_token_id)` pairs for cross-rank exchange.

**Triton kernel** `_pack_dcp_topk_candidates_triton_kernel`, one program per token row:

```
For each local index local_idx in topk_indices[row, :]:
    score = logits[row, local_idx + row_start]      # fetch score

    # Convert local index to global token ID:
    # Under interleaved DCP with interleave=1 (block=1):
    # global_id = local_idx * world_size + rank
    # General formula for interleave I:
    global_id = (local_idx // I) * (world × I) + rank × I + (local_idx % I)

    packed[row, col, 0] = score      (as float32)
    packed[row, col, 1] = float(global_id)
```

After this, `get_dcp_group().all_gather(packed, dim=1)` exchanges the `(score, global_id)` pairs across all ranks. Each rank then has a `[T, world_size × topk, 2]` tensor.

---

### 7.2 `stable_topk_from_gathered_candidates_cutedsl`

**File:** `vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py:17`

**What it does:** Selects the global top-K from the all-gathered candidates, stably (ties broken by smaller global token ID — consistent across ranks).

**Input:** `gathered [T, world_size × topk, 2]` — `(score, global_id)` pairs from all ranks.

**CuteDSL kernel** `StableTopKFromGatheredCandidatesKernel`, one CUDA block (512 threads) per token row.

**Step 1 — Key construction:**

```
key = uint64(score_bits_flipped) << 32 | uint64(~token_id)
```

- Floats are reinterpreted as uint32; XOR with `0xFFFFFFFF` (negative) or `0x80000000` (non-negative) makes them sortable as unsigned integers (IEEE 754 property)
- `~token_id` in the lower 32 bits: smaller token IDs → larger lower bits → stable preference for earlier tokens among ties
- Invalid entries (token_id < 0): key = 0 (smallest, always eliminated)

**Step 2 — Radix top-K (`_radix_pass`):**

Multi-pass 11-bit radix select (6 passes for 64-bit keys):

```
For each radix pass (highest bits first):
  1. Each thread holds (total_candidates / 512) keys in registers
  2. Build 2048-bin histogram over current 11-bit radix digit in shared memory
  3. Warp-level reduction to compute global histogram
  4. Find threshold bin (prefix sum from MSB down to find K-th element)
  5. Atomically commit above-threshold keys to output buffer
     (using separate committed_count and running_count in smem)
  6. Track prefix_smem for filtering in subsequent passes
```

**Output:** `out [T, topk] int32` — global token IDs, `-1` for padding.

**Constraint:** `num_candidates = world_size × topk` must be a multiple of 512. Hence `topk ∈ {512, 1024, 2048}` (world_size is always power-of-2).

---

## 8. Main MLA Attention (V3)

### 8.1 `concat_mla_q`

**File:** `vllm/_custom_ops.py:2781`

A C++ custom op that concatenates the NoPE and RoPE halves of Q after weight absorption:

```python
concat_mla_q(
    ql_nope: [T, H, kv_lora_rank=512],   # absorbed Q: q_nope × W_UK_T
    q_pe:    [T, H, rope_dim=64],         # RoPE-rotated Q
    q_out:   [T, H, 576],                 # output buffer (pre-allocated)
)
```

Result: `q_out [T, H, 576]` where `q_out[t, h, :512] = ql_nope[t, h, :]` and `q_out[t, h, 512:] = q_pe[t, h, :]`.

This is needed because FlashMLA's kernel interface expects a single concatenated Q tensor, not the split (nope, pe) representation used during the weight absorption computation.

---

### 8.2 W_UK_T and W_UV Weight Absorption

**File:** `vllm/model_executor/layers/attention/mla_attention.py:907`

**Problem:** MLA stores `kv_c [512]` per token. At attention time, to compute the score for head `h` against cached token `k`:

```
score = dot(q_nope[h, :128], k_nope[k, h, :128]) + dot(q_pe[h, :64], k_pe[k, :64])
```

`k_nope[k, h, :] = kv_b_proj(kv_c[k])[h, :128]` — this requires running `kv_b_proj` for every cached token at every attention step. That's `N × hidden_size × num_heads` operations, completely defeating the purpose of the latent cache.

**Solution: absorb `kv_b_proj`'s nope-weight into Q.**

```python
# kv_b_proj.weight shape: [kv_lora_rank, num_heads × (nope_dim + v_dim)]
# Reshape and split:
W_UK [kv_lora_rank, num_heads, nope_dim]   # key nope uprojection
W_UV [kv_lora_rank, num_heads, v_dim]      # value uprojection

# Transpose for batched matmul:
W_UK_T = W_UK.permute(1, 2, 0)   # [num_heads, nope_dim, kv_lora_rank]
W_UV   = W_UV.transpose(0, 1)    # [num_heads, kv_lora_rank, v_dim]
```

**At inference:**

```python
# Instead of: score = q_nope @ W_UK @ kv_c
# Pre-absorb W_UK into Q:
ql_nope = bmm(q_nope, W_UK_T)   # [T, H, 128] × [H, 128, 512] → [T, H, 512]

# Now the score is just a dot product with the cached latent:
score[t, h, k] = dot(ql_nope[t, h, :], kv_c[k, :])   # 512-dim dot
               + dot(q_pe[t, h, :], k_pe[k, :])        # 64-dim dot
```

No `kv_b_proj` is needed per cached token. FlashMLA reads only `kv_c [N, 512]` and `k_pe [N, 64]` from cache — exactly what's stored.

**Output side (W_UV):**

```python
# After attention: attn_out is in latent space [T, H, kv_lora_rank=512]
# Instead of: v = W_UV(kv_c); output = sum(alpha × v)
# Pre-gather: out_v[t, h] = sum_k(alpha[t,h,k] × kv_c[k, :]) → [T, H, 512]
# Then multiply by W_UV offline:
output[t, h] = bmm(out_latent, W_UV)   # [T, H, 512] × [H, 512, 128] → [T, H, 128]
```

The attention kernel never touches V — it operates entirely in latent space and returns `attn_out [T, H, 512]`. Only after attention is `W_UV` applied.

---

### 8.3 `FlashMLASparseImpl.forward_mqa`

**File:** `vllm/v1/attention/backends/mla/flashmla_sparse.py:839`

**Dispatch logic:**

```python
def forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer):
    # q may arrive as (ql_nope, q_pe) tuple if weight absorption was done outside:
    if isinstance(q, tuple):
        ops.concat_mla_q(ql_nope, q_pe, q_out)   # concat in-place

    topk_indices = topk_indices_buffer[:num_actual_tokens]   # from indexer

    if not use_fp8_cache:
        # BF16 KV cache (rare, for older configs)
        _forward_bf16_kv(q, kv_cache, topk_indices, attn_metadata)
    elif attn_metadata.fp8_use_mixed_batch:
        # High-TP path: too few heads per rank for BF16 prefill kernel overhead
        _forward_fp8_kv_mixed_batch(...)
    else:
        # Normal FP8 path: separate BF16 prefill + FP8 decode
        _forward_fp8_kv_separate_prefill_decode(q, kv_cache, topk_indices, attn_metadata)
```

**FP8 separate prefill/decode path:**

- **Decode tokens** (`_fp8_flash_mla_kernel`): `q [B, 1, H, 576]` → `flash_mla_with_kvcache` with paged FP8 cache and `indices [B, 1, topk]` = global slot IDs
- **Prefill tokens**: First dequantize FP8 → BF16 into a workspace (`cp_gather_and_upconvert_fp8_kv_cache`), then call `_bf16_flash_mla_kernel` with the BF16 workspace and local-indexed `topk_indices`

The split exists because FlashMLA's FP8 kernel path (`is_fp8_kvcache=True`) is optimized for decode (small Q, large K), while prefill uses BF16 for correctness (the prefill K is gathered into a workspace anyway).

---

### 8.4 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd`

**File:** `vllm/v1/attention/ops/flashmla.py` → `vllm.third_party.flashmla.flash_mla_interface`

These are compiled C++/CUDA kernels from the FlashMLA library.

**`flash_mla_with_kvcache`** — decode path:

```python
flash_mla_with_kvcache(
    q:                 [B, next_n, H, head_dim_k],  # query
    k_cache:           [num_blocks, block_size, 1, head_dim_k],  # paged KV
    block_table:       [B, max_blocks],              # ignored when indices!=None
    head_dim_v:        int,                          # 512 for DSv3/V4
    tile_scheduler_metadata: FlashMLASchedMeta,      # work assignment
    cache_seqlens:     [B],                          # ignored when indices!=None
    is_fp8_kvcache:    bool,
    indices:           [B, next_n, topk],            # global physical slot IDs
    topk_length:       [B, next_n],                  # valid entries per token
    softmax_scale:     float,
    attn_sink:         Tensor,                       # per-head learned sink bias
    extra_k_cache:     Tensor or None,               # second KV cache (V4: compressed)
    extra_indices_in_kvcache: Tensor or None,        # [B, next_n, extra_topk] global slots
    extra_topk_length: Tensor or None,
    out:               [B, next_n, H, head_dim_v],   # pre-allocated output
)
```

**Key behavior:**

- When `indices` is non-None, **block_table is ignored**. The kernel reads KV exclusively from the physical slots listed in `indices`, treating each element as a direct physical address: `k_cache[indices[b,n,i] // block_size, indices[b,n,i] % block_size, 0, :]`.
- `-1` entries in `indices` are masked (treated as padding, score set to -inf).
- When `extra_k_cache` is provided (V4), a second paged cache is attended simultaneously using `extra_indices_in_kvcache` — this combines SWA and compressed-sparse attention in a single kernel pass.
- `attn_sink [H]`: per-head learned bias added to the softmax denominator to prevent attention collapse on very sparse token sets.

**`get_mla_metadata`** — called once per decode step to build the tile scheduler plan:

```python
tile_md, num_splits = get_mla_metadata(
    cache_seqlens,                   # [B]
    num_q_tokens_per_head_k,         # B × next_n × H // H_K (ratio of Q to K heads)
    num_heads_k,                     # 1 (MLA uses single shared K)
    num_heads_q=H, topk=K, is_fp8_kvcache=True
)
```

The plan assigns contiguous ranges of `(batch, sparse_block)` work to each SM, balancing load across sequences with different sparse token counts.

**`flash_mla_sparse_fwd`** — prefill path (BF16 workspace):

```python
flash_mla_sparse_fwd(
    q:            [T_q, H, head_dim_k],    # prefill query tokens (flat, not batched)
    kv:           [T_kv, 1, head_dim_k],   # gathered BF16 KV workspace (flat)
    indices:      [T_q, 1, topk],          # workspace-local indices (not global slots)
    sm_scale:     float,
    topk_length:  [T_q],
    out:          [T_q, H, head_dim_v],
)
```

The KV here is a pre-gathered BF16 buffer (workspace), so `indices` are local workspace offsets, not global slot IDs. This is the standard chunked FlashAttention-style prefill.

---

## 9. Main MLA Attention (V4)

### 9.1 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`

**File (dispatch):** `vllm/models/deepseek_v4/attention.py:554`

**What it does:** The largest fused operation in V4 — combines 5 steps for SWA cache population into a single C++ custom op call.

**Signature:**

```python
torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q:            [T, H, 512]    bf16,   # in-place: norm+RoPE written back
    kv:           [T, 512]       bf16,   # KV latent (after kv_norm)
    swa_kv_cache: [num_blocks×block_size, 512+8] uint8,  # 2D view of SWA cache
    slot_mapping: [T]            int64,  # SWA physical slot per token
    positions:    [T]            int64,
    cos_sin_cache:[max_pos, 64]  float32,
    padded_heads: int,                   # output head count (may be > H for alignment)
    eps:          float,
    block_size:   int,
) → q_out [T, padded_heads, 512]
```

**Fused steps:**

1. **Per-head RMSNorm on Q** (no weight): for each head `h`, normalize `q[t, h, :]` by its own RMS. Unlike `q_a_layernorm` (which normalized the Q low-rank bottleneck `q_c`), this normalizes the final full-head Q before attention.

2. **GPT-J RoPE on Q** (last 64 dims): for each head `h`, rotate `q[t, h, 448:]` using position `positions[t]`.

3. **GPT-J RoPE on KV** (last 64 dims): rotate `kv[t, 448:]` using position `positions[t]`.

4. **UE8M0 FP8 quantize KV** (NoPE portion): block-scaled FP8 for `kv[t, :448]` with 64-element quantization blocks. Produces 448 FP8 bytes + 7 ue8m0 scale bytes (packed as `uint8`). The RoPE portion `kv[t, 448:]` is stored as BF16 (64 values × 2 bytes = 128 bytes), not quantized.

5. **Write to SWA cache**: total 448+128+8=584 bytes per slot. The `slot_mapping` maps each token to its physical location in the SWA paged cache.

**Output Q** is padded to `padded_heads` (e.g., 128 for FlashMLA alignment when `n_local_heads < 128`) with zeros in the extra head slots.

---

### 9.2 SWA Index Computation

**File:** `vllm/v1/attention/backends/mla/sparse_swa.py:660`

**Triton kernel** `_compute_swa_indices_and_lens_kernel`, grid `(T,)`.

For each token `t`:

```
req_idx = token_to_req_indices[t]
pos = prefix_len + (t - query_start)          # absolute position in sequence
start = max(pos - window_size + 1, 0)
end   = pos + 1                                # exclusive
swa_len = end - start

for i, p in enumerate(range(start, end)):
    block = block_table[req_idx, p // block_size]
    slot  = block * block_size + p % block_size
    swa_indices[t, 0, i] = slot

# fill swa_indices[t, 0, swa_len:] = -1
swa_lens[t] = swa_len
```

**Result:** `swa_indices [T, 1, window_size] int32` — global physical slot IDs for each token's SWA window. `-1` marks unused slots. This is passed directly to `flash_mla_with_kvcache` as the `indices` argument.

---

### 9.3 `compute_global_topk_indices_and_lens`

**File:** `vllm/models/deepseek_v4/nvidia/flashmla.py`

**What it does:** Translates the indexer's compressed-space top-K indices (local block indices within each request's KV cache) into global physical slot IDs in the full main MLA KV cache.

The indexer's `topk_indices_buffer [T, 1024]` contains values in `[0, compressed_seq_len)` — positions into the compressed indexer K cache (1 compressed position = 4 original tokens). The main MLA KV cache is in **uncompressed** space.

**Triton kernel**, one program per token:

```
For each compressed index ck in topk_indices[t, :]:
    if ck < 0: global_indices[t, j] = -1; continue

    # ck is a compressed position; expand to 4 physical slots:
    for offset in range(compress_ratio):   # 0,1,2,3
        original_pos = ck * compress_ratio + offset

        # look up original_pos in block_table of the MAIN MLA cache:
        block = block_table[req_idx, original_pos // block_size]
        slot  = block * block_size + original_pos % block_size
        global_indices[t, j*compress_ratio+offset] = slot

topk_lens[t] = number of valid entries (non -1)
```

**Result:** `global_topk_indices [T, 1, 1024×compress_ratio=4096] int32` — physical slot IDs into the main MLA KV cache. These are passed as `extra_indices_in_kvcache` to `flash_mla_with_kvcache`.

---

### 9.4 `dequantize_and_gather_k_cache`

**File:** `vllm/models/deepseek_v4/common/ops/cache_utils.py:381`

**Used in V4 prefill** to convert FP8 KV from the compressed paged cache into a contiguous BF16 workspace that the prefill FlashMLA kernel can process.

**Signature:**

```python
dequantize_and_gather_k_cache(
    workspace:    [B, chunk_M, head_dim]  bf16,   # output dense buffer
    kv_cache:     [num_blocks, block_size, 584]  uint8,  # paged FP8 cache
    seq_lens:     [B]  int32,                     # sequence lengths
    gather_lens:  [B] or None,                    # None=gather all; [B]=gather last N
    block_table:  [B, max_blocks]  int32,
    block_size:   int,
    offset:       int,                            # write start column in workspace
)
```

For each request `b` and each token `p` to gather:

```
block = block_table[b, p // block_size]
slot  = block * block_size + p % block_size

# Read 584-byte slot: 448 FP8 + 128 bf16 + 8 ue8m0 scales
nope_fp8  = kv_cache[slot, :448]
rope_bf16 = kv_cache[slot, 448:576].view(bf16)
scales    = kv_cache[slot, 576:584]   # 7 ue8m0 bytes + 1 pad

# Dequantize FP8 nope (per 64-element block):
for b_idx, scale_byte in enumerate(scales[:7]):
    scale = 2^(scale_byte - 127)
    nope_bf16[b_idx*64:(b_idx+1)*64] = fp8_to_float(nope_fp8[b_idx*64:...]) * scale

# Concat: 448 bf16 (nope) + 64 bf16 (rope) = 512 bf16
workspace[b, offset + i, :448] = nope_bf16
workspace[b, offset + i, 448:] = rope_bf16
```

For SWA, `gather_lens [B]` specifies that only the last `gather_lens[b]` tokens are gathered (the SWA window). `offset=chunk_N` places them after the compressed tokens in the workspace.

---

### 9.5 `combine_topk_swa_indices`

**File:** `vllm/models/deepseek_v4/nvidia/flashmla.py` (prefill only)

**What it does:** For V4 prefill, the gathered BF16 workspace contains compressed tokens in `[0..chunk_N)` and SWA tokens in `[chunk_N..chunk_M)`. Each prefill query token needs indices into this workspace combining its causal compressed prefix and its SWA window.

**Triton kernel** `_combine_topk_swa_indices_kernel`, one program per token `t`:

```
pos = prefix_len + (t - query_start)         # absolute position

# Compressed (indexer) entries — in workspace columns [0, chunk_N):
topk_len = min((pos + 1) // compress_ratio, TOP_K)
for i in range(topk_len):
    # topk_indices[t, i] is a local compressed index within this chunk:
    combined[t, i] = topk_indices[t, i] + chunk_offset  # workspace column

# SWA entries — in workspace columns [chunk_N, chunk_M):
swa_start = max(pos - window_size + 1, gather_start)
swa_len   = pos + 1 - swa_start
for j in range(swa_len):
    p = swa_start + j
    workspace_col = chunk_N + p - gather_start
    combined[t, topk_len + j] = workspace_col

combined_lens[t] = topk_len + swa_len
```

The result is passed as `indices` to `flash_mla_sparse_fwd` — all pointing into the same flat BF16 workspace which holds both compressed tokens and SWA tokens.

---

### 9.6 `fused_inv_rope_fp8_quant`

**File:** `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`

**What it does:** Before the output projection `wo_a`, applies the **inverse** GPT-J RoPE to the attention output and block-quantizes to FP8 for the DeepGEMM FP8 BMM.

**Why inverse RoPE?** During forward attention, Q's rope dims were rotated by `pos`. The attention output mixes heads that were in different rotated frames. The output projection weight `wo_a` was trained expecting inputs in the **unrotated** frame. So the rotation must be undone before projecting.

**Inverse GPT-J rotation** (from kernel, lines 97–107):

```
# Forward GPT-J: [x_even, x_odd] → [x_even*cos - x_odd*sin, x_odd*cos + x_even*sin]
# Inverse:       [y_even, y_odd] → [y_even*cos + y_odd*sin, -y_even*sin + y_odd*cos]
#   (multiply both rotations by [cos, -sin; sin, cos]^-1 = [cos, sin; -sin, cos])

x_even = o[t, h, ROPE_START + 2*i]
x_odd  = o[t, h, ROPE_START + 2*i + 1]
partner = x_odd if even else x_even
rotated = x*cos + partner*sin  if even else  x*cos - partner*sin
```

**FP8 block quantization:**

```
for each block of QUANT_GROUP_SIZE=128 values in the head:
    block_absmax = max(|values|)
    scale = exp2(ceil(log2(block_absmax / fp8_max)))
    fp8_block = clamp(values / scale, -fp8_max, fp8_max).to(fp8)
```

**Scale storage (SM100 TMA-aligned):**

```
CHUNKS_PER_HEAD = head_dim / QUANT_GROUP_SIZE = 512 / 128 = 4
# Pack 4 ue8m0 bytes into one int32 via shift-OR:
scale_int32 = (ue8m0[0] << 0) | (ue8m0[1] << 8) | (ue8m0[2] << 16) | (ue8m0[3] << 24)
# Store one int32 per (group, token, head_in_group)
```

For SM90 (non-TMA): store 4 float32 scales separately.

**Grid:** `(ceil4(T), n_groups × heads_per_group)` — `ceil4` pads to multiple of 4 for TMA alignment. Padding rows write zero scales.

---

### 9.7 `deep_gemm_fp8_o_proj` (`wo_a` + `wo_b`)

**File:** `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`

**What it does:** The complete V4 output projection — after `fused_inv_rope_fp8_quant` produces `o_fp8 [T, G, D]`, this does the FP8 grouped BMM (`wo_a`) followed by `wo_b`.

**`wo_a` — FP8 grouped batched matmul:**

```python
fp8_einsum(
    "bhr,hdr->bhd",
    (o_fp8, o_scale),         # [T, G, heads×512] fp8  + scales
    (wo_a.weight, wo_a.weight_scale_inv),  # [G, heads×512, o_lora_rank] fp8 + scales
    out=z,                    # [T, G, o_lora_rank] bf16
    recipe=einsum_recipe,
)
```

`G = n_local_groups`, each group handles `n_local_heads / n_groups` heads.

The `einsum_recipe` precomputed in `__init__` specifies the FP8 block scaling mode (per-tensor, per-row, or block-scaled depending on `wo_a.weight` dtype). This is a DeepGEMM FP8 GEMM, JIT-compiled.

**Why groups?** V4 introduces output LoRA groups (`o_groups`) to reduce the weight matrix size while preserving expressiveness. Instead of one large `[H×512 → hidden_size]` projection, it has `G` separate `[H/G × 512 → o_lora_rank]` matmuls followed by a single `[G×o_lora_rank → hidden_size]`. The BMM interpretation: each group independently projects its heads to `o_lora_rank` dims, then `wo_b` mixes across groups.

**`wo_b` — standard output projection:**

```python
wo_b(z.flatten(1))   # [T, G×o_lora_rank] → [T, 7168]
```

A `RowParallelLinear` — standard BF16 matmul with tensor-parallel reduction across devices.

---

## 10. Spec-Decode Padding Helpers

### 10.1 `pack_seq_triton` / `unpack_seq_triton`

**File:** `vllm/v1/attention/ops/common.py`

Used in the decode path of the indexer when speculative decoding generates non-uniform decode lengths (some requests have 1 decode token, others have `next_n`). The indexer's paged MQA logit kernel expects a regular `[B, next_n, H, D]` shaped Q, not a ragged layout.

**`pack_seq_triton`** — ragged → padded:

```python
pack_seq_triton(
    x:       [N, ...]    # N total tokens, ragged across B sequences
    lengths: [B]         # actual token count per sequence
    pad_value,           # -inf (for logits) or 0 (for uint8 FP8/MXFP4)
) → [B, Lmax, ...]     # Lmax = max(lengths), padded with pad_value
```

**Triton kernel** `_pack_seq_kernel`, grid `(B, ceil(Lmax/BT), ceil(D/BD))`:

```
in_start = sum(lengths[:pid_b])
for off_t in arange(BLOCK_T):
    for off_d in arange(BLOCK_D):
        t = pid_t * BLOCK_T + off_t
        if t < lengths[pid_b]:
            out[pid_b, t, off_d] = x[in_start + t, off_d]
        else:
            out[pid_b, t, off_d] = PAD_VALUE
```

**`unpack_seq_triton`** — padded → ragged (inverse):

```python
unpack_seq_triton(
    packed: [B, Lmax, ...]
    lengths:[B]
) → [N, ...]
```

Same kernel structure but reads from `packed` and writes to `out` at cumulative offsets.

**Usage in the indexer decode path:**

When decode lengths are non-uniform (e.g., requests have [3, 1, 2] decode tokens), the indexer:

1. `pack_seq_triton(q_fp8, decode_lens)` → `[B, Lmax, 64, 128]`
2. `fp8_fp4_paged_mqa_logits(padded_q, ...)` → `logits [B*Lmax, max_seq_len]`
3. `topk → topk_indices [B*Lmax, 1024]`
4. `unpack_seq_triton(topk_indices.reshape(B, Lmax, -1), decode_lens)` → `[N, 1024]`

The padding tokens get `-inf` logits (for Q) or `0` (for uint8 Q), ensuring they produce only `-1` top-K indices that are harmless to downstream attention.
