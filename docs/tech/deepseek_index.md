# DeepSeek V3 / V4 Documentation Index

Reference hub. Each linked doc covers one coherent topic in depth.

---

## Document Map

| Doc | What it covers |
|---|---|
| **[deepseek_v4_layers.md](deepseek_v4_layers.md)** | Per-layer type table (C4A/C128A/SWA-only), layer structure diagram, compress_ratio semantics |
| **[deepseek_v4_attention.md](deepseek_v4_attention.md)** | V4 MLA math, Q/KV projections, compressor, indexer, SWA+sparse FlashMLA, output projection |
| **[deepseek_v4_moe.md](deepseek_v4_moe.md)** | MoE routing: hash (layers 0–2), noaux_tc (layers 3–60), DSpark, SwiGLU, expert GEMM shapes |
| **[deepseek_v4_e2e.md](deepseek_v4_e2e.md)** | End-to-end walkthrough: tokenization → embedding → attention → MoE → sampling, concrete numbers |
| **[deepseek_v4_rocm.md](deepseek_v4_rocm.md)** | All ROCm execution paths: SHUFFLE layout, AITER/FlyDSL dispatch, ragged attention kernels, gfx942 vs gfx950, MoE padding fix |
| **[deepseek_v3.md](deepseek_v3.md)** | V3 complete reference: MLA, weight absorption, indexer, FlashMLA sparse, end-to-end example |

The original combined draft is preserved at [deepseek_v3_v4_complete.md](deepseek_v3_v4_complete.md).

---

## Quick Config Reference (V4-Pro-0813)

Source: `/shareddata/data/models/DeepSeek-V4-Pro-0813/config.json`

### Model Dimensions

| Parameter | Value | Notes |
|---|---|---|
| `hidden_size` | 7168 | Token hidden dim at every layer |
| `num_hidden_layers` | 61 | Layers 0–60 |
| `num_attention_heads` | 128 | Full MLA heads |
| `head_dim` | 512 | 448 NoPE + 64 RoPE |
| `qk_rope_head_dim` | 64 | RoPE portion of each head |
| `q_lora_rank` | 1536 | Q low-rank bottleneck dim |
| `o_lora_rank` | 1024 | Output projection bottleneck dim |
| `o_groups` | 16 | Groups for wo_a grouped BMM |
| `vocab_size` | 129280 | Token vocabulary size |
| `max_position_embeddings` | 1048576 | 1M context support |

### Indexer (Sparse Scoring)

| Parameter | Value | Notes |
|---|---|---|
| `index_n_heads` | 64 | Scoring heads (cheaper than 128 full MLA heads) |
| `index_head_dim` | 128 | 64 NoPE + 64 RoPE per scoring head |
| `index_topk` | 1024 | Compressed tokens selected per query |

### Compression

| Parameter | Value | Notes |
|---|---|---|
| `compress_ratios` | see §Layer Table | Per-layer list, length 64 (61 real + 3 padding zeros) |
| `compress_rope_theta` | 160000 | Separate RoPE base for compressor (vs 10000 for main) |
| `sliding_window` | 128 | Used internally in indexer; main SWA window is 4096 in code |

### MoE

| Parameter | Value | Notes |
|---|---|---|
| `n_routed_experts` | 384 | Total routed experts |
| `num_experts_per_tok` | 6 | Top-6 selected per token |
| `n_shared_experts` | 1 | Shared expert, always active |
| `num_hash_layers` | 3 | Layers 0–2 use hash routing |
| `moe_intermediate_size` | 3072 | Expert hidden dim |
| `scoring_func` | `sqrtsoftplus` | `sqrt(ln(1 + e^x))` |
| `topk_method` | `noaux_tc` | No-aux-loss, token-constrained |
| `routed_scaling_factor` | 2.5 | Multiply renormalized weights by 2.5 |
| `norm_topk_prob` | `true` | Renormalize top-6 weights to sum=1 before scaling |
| `expert_dtype` | `fp4` | MXFP4 expert weights |
| `swiglu_limit` | 10.0 | Gate clamp range to prevent MXFP4 overflow |

### DSpark

| Parameter | Value | Notes |
|---|---|---|
| `dspark_target_layer_ids` | [58, 59, 60] | Layers that also save hidden state for draft model |
| `dspark_markov_rank` | 512 | Draft model hidden dim |
| `dspark_block_size` | 5 | Draft block size |
| `dspark_noise_token_id` | 128799 | Noise injection token |
| `num_nextn_predict_layers` | 1 | Speculative prediction depth |

### Quantization (Main Weights)

| Parameter | Value | Notes |
|---|---|---|
| `quantization_config.quant_method` | `fp8` | Main linear weights in FP8 |
| `quantization_config.fmt` | `e4m3` | OCP FP8 format |
| `quantization_config.scale_fmt` | `ue8m0` | Power-of-2 block scale |
| `quantization_config.weight_block_size` | [128, 128] | 128×128 element blocks |
| `quantization_config.activation_scheme` | `dynamic` | Per-token dynamic activation quant |

---

## Layer Type Table

Full assignment of all 61 layers. See [deepseek_v4_layers.md](deepseek_v4_layers.md) for explanation of each type.

| Layer | compress_ratio | Type | Also |
|---|---|---|---|
| 0 | 128 | **C128A** | Hash MoE (layers 0–2) |
| 1 | 128 | **C128A** | Hash MoE |
| 2 | 4 | **C4A** | Hash MoE |
| 3 | 128 | **C128A** | Standard MoE |
| 4 | 4 | **C4A** | Standard MoE |
| 5 | 128 | **C128A** | Standard MoE |
| 6 | 4 | **C4A** | Standard MoE |
| 7 | 128 | **C128A** | Standard MoE |
| 8 | 4 | **C4A** | Standard MoE |
| 9 | 128 | **C128A** | Standard MoE |
| 10 | 4 | **C4A** | Standard MoE |
| 11 | 128 | **C128A** | Standard MoE |
| 12 | 4 | **C4A** | Standard MoE |
| 13 | 128 | **C128A** | Standard MoE |
| 14 | 4 | **C4A** | Standard MoE |
| 15 | 128 | **C128A** | Standard MoE |
| 16 | 4 | **C4A** | Standard MoE |
| 17 | 128 | **C128A** | Standard MoE |
| 18 | 4 | **C4A** | Standard MoE |
| 19 | 128 | **C128A** | Standard MoE |
| 20 | 4 | **C4A** | Standard MoE |
| 21 | 128 | **C128A** | Standard MoE |
| 22 | 4 | **C4A** | Standard MoE |
| 23 | 128 | **C128A** | Standard MoE |
| 24 | 4 | **C4A** | Standard MoE |
| 25 | 128 | **C128A** | Standard MoE |
| 26 | 4 | **C4A** | Standard MoE |
| 27 | 128 | **C128A** | Standard MoE |
| 28 | 4 | **C4A** | Standard MoE |
| 29 | 128 | **C128A** | Standard MoE |
| 30 | 4 | **C4A** | Standard MoE |
| 31 | 128 | **C128A** | Standard MoE |
| 32 | 4 | **C4A** | Standard MoE |
| 33 | 128 | **C128A** | Standard MoE |
| 34 | 4 | **C4A** | Standard MoE |
| 35 | 128 | **C128A** | Standard MoE |
| 36 | 4 | **C4A** | Standard MoE |
| 37 | 128 | **C128A** | Standard MoE |
| 38 | 4 | **C4A** | Standard MoE |
| 39 | 128 | **C128A** | Standard MoE |
| 40 | 4 | **C4A** | Standard MoE |
| 41 | 128 | **C128A** | Standard MoE |
| 42 | 4 | **C4A** | Standard MoE |
| 43 | 128 | **C128A** | Standard MoE |
| 44 | 4 | **C4A** | Standard MoE |
| 45 | 128 | **C128A** | Standard MoE |
| 46 | 4 | **C4A** | Standard MoE |
| 47 | 128 | **C128A** | Standard MoE |
| 48 | 4 | **C4A** | Standard MoE |
| 49 | 128 | **C128A** | Standard MoE |
| 50 | 4 | **C4A** | Standard MoE |
| 51 | 128 | **C128A** | Standard MoE |
| 52 | 4 | **C4A** | Standard MoE |
| 53 | 128 | **C128A** | Standard MoE |
| 54 | 4 | **C4A** | Standard MoE |
| 55 | 128 | **C128A** | Standard MoE |
| 56 | 4 | **C4A** | Standard MoE |
| 57 | 128 | **C128A** | Standard MoE |
| 58 | 4 | **C4A** | Standard MoE + **DSpark** |
| 59 | 4 | **C4A** | Standard MoE + **DSpark** |
| 60 | 4 | **C4A** | Standard MoE + **DSpark** |

**Summary counts:** 31 × C128A, 30 × C4A, 0 × SWA-only.
The three `compress_ratio=0` entries in the raw array are padding beyond the 61-layer count.

**MoE split:** Layers 0–2 use Hash MoE; layers 3–60 use Standard (noaux_tc) MoE.

---

## Key Tensor Shapes at a Glance

```
hidden_states:         [T, 7168]      bf16
Q (after wq_b):        [T, 128, 512]  bf16
KV latent (kv_c):      [T, 512]       bf16
SWA / main MLA cache:  [blocks, block_size, 1, 512]  uint8  (584 bytes/slot)
Indexer K cache:       [blocks, 256, 128+4]  uint8  (132 bytes/slot)
topk_indices_buffer:   [T, 1024]      int32
MoE gate logits:       [T, 384]       float32
Expert weights (W1):   [384, 768, 3584]  MXFP4
Expert weights (W2):   [384, 896, 3072]  MXFP4
```

---

## Worked Examples: Quick Reference

### Example 1: Tensor shape trace for one decode step (TP=1)

Single decode token at position 5000, tracing key tensor shapes through the full forward pass:

```
hidden_states                          [1, 7168]       bf16
After fused_wqa_wkv:
  qr (Q low-rank)                      [1, 1536]       bf16
  kv (KV latent)                       [1, 512]        bf16
After q_norm / kv_norm:                same shapes (in-place RMSNorm)
After wq_b:                            [1, 128, 512]   bf16  (128 heads × 512 head_dim)
After per-head RMSNorm + GPT-J RoPE:   [1, 128, 512]   bf16  (in-place, last 64 dims rotated)
c_kv after GPT-J RoPE:                 [1, 512]        bf16
FP8 block-quantized + SWA cache write: 584 bytes at slot 5000
  (448 FP8 NoPE in 7 ue8m0 blocks + 128 BF16 RoPE + 8 bytes scale/pad)

Indexer path:
  q_fp8:                               [1, 64, 128]    fp8  (64 scoring heads × 128 head_dim)
  weights_out:                         [1, 64]         fp32 (per-head scalar weights)
  topk_indices_buffer:                 [1, 1024]       int32

FlashMLA output:
  attn_out:                            [1, 128, 512]   bf16

Output projection:
  After inverse RoPE:                  [1, 128, 512]   bf16
  After wo_a (grouped FP8 BMM):        [1, 1024]       bf16
  After wo_b:                          [1, 7168]       bf16

After MoE:                             [1, 7168]       bf16
```

---

### Example 2: Memory budget at 128K tokens (TP=8)

All figures are per-GPU (TP=8 shards the model weights 8 ways; KV caches are replicated per sequence).

```
Model weights (FP8, 61 layers):                            ~70 GB total / 8 GPUs = ~8.75 GB/GPU

KV cache (per sequence, not sharded by TP):
  SWA KV cache (all 61 layers × 4096 slots × 584 B):
    61 × 4096 × 584 ≈ 146 MB  (each layer has its own SWA window)

  Main MLA KV cache — C128A layers (31 layers):
    At 128K context: 131072 / 128 = 1024 compressed slots per layer
    31 × 1024 × 584 ≈ 18 MB

  Main MLA KV cache — C4A layers (30 layers):
    At 128K context: 131072 / 4 = 32768 compressed slots per layer
    30 × 32768 × 584 ≈ 574 MB

  Indexer K cache (61 layers, 256 slots/block, 132 B/slot):
    C4A (30 layers):   30 × 32768 × 132 ≈ 130 MB
    C128A (31 layers): 31 × 1024  × 132 ≈ 4 MB
    Total indexer K: ~134 MB

  Compressor state cache:
    C4A (30 layers): 8 states × 8192 B × 32768 compressed slots ≈ small (amortized)
    C128A (31 layers): 256 states × 8192 B × 1024 compressed slots ≈ small (amortized)
    Practical: ~50 MB total across all layers

Total KV cache at 128K context (per sequence): ~700 MB–900 MB
  (dominated by C4A main MLA at ~574 MB + SWA at ~146 MB)
```

Key takeaway: the C4A layers hold roughly 30× more compressed KV slots than C128A at 128K context, making them the dominant KV memory consumer.

---

### Example 3: Layer-by-layer dispatch decision for layers 0–4

How `attention.py` determines the layer type and routing for each of the first five layers:

| Layer | compress_ratio | Layer Type | MoE Type | Why |
|---|---|---|---|---|
| 0 | 128 | **C128A** | Hash MoE | First layer; compress_ratio=128 → C128A; layer_idx < num_hash_layers (3) → hash routing |
| 1 | 128 | **C128A** | Hash MoE | Second C128A layer — unusual; the normal C128A/C4A alternation has not started yet |
| 2 | 4 | **C4A** | Hash MoE | First C4A layer; still in hash-routing zone (layer_idx=2 < 3); the only C4A+hash layer |
| 3 | 128 | **C128A** | Standard (noaux_tc) | compress_ratio=128 → C128A; layer_idx=3 ≥ 3 → noaux_tc routing starts here |
| 4 | 4 | **C4A** | Standard (noaux_tc) | compress_ratio=4 → C4A; alternation now established |

The two decisions are fully orthogonal: `compress_ratio` sets the attention type; `layer_idx < num_hash_layers` sets the MoE type. They are read from `config.json` independently.

---

## Glossary

| Term | Meaning |
|---|---|
| **MLA** | Multi-head Latent Attention — compresses KV by caching a low-dim latent |
| **C4A** | Compress-4-Aggregate: every 4 tokens → 1 compressed indexer/MLA KV entry |
| **C128A** | Compress-128-Aggregate: every 128 tokens → 1 compressed entry |
| **SWA** | Sliding-Window Attention — attends only recent `window_size` tokens |
| **Indexer** | Cheap 64-head FP8 MQA that scores all compressed KV tokens to select top-1024 |
| **Compressor** | Aggregates N raw token states into 1 compressed KV via softmax gating |
| **APE** | Absolute Positional Embedding — learned per-position-within-group offsets |
| **NoPE** | Non-positional (content) dims of each head; not rotated by RoPE |
| **RoPE** | Rotary Positional Embedding — applied to the last 64 dims of Q and K |
| **UE8M0** | Power-of-2 block scale format: `scale = 2^(biased_exponent - 127)` |
| **MXFP4** | 4-bit E2M1 micro-scaled float; 32 elements per block, ue8m0 scale byte |
| **noaux_tc** | No-auxiliary-loss token-constrained MoE routing via trained bias |
| **DCP** | Decode Context Parallelism — KV sequence sharded across GPUs |
| **DSpark** | Speculative decoding extension; layers 58–60 produce draft hidden states |
| **AITER** | AMD Inference Engine Runtime — optimized ROCm kernel library |
| **FlyDSL** | AMD JIT-compiled kernel DSL, used on gfx942 for MQA logits |
| **FNUZ** | FP8 format on gfx942: `float8_e4m3fnuz`, max=224 (vs OCP max=448) |
| **OCP** | FP8 format on gfx950 + CUDA: `float8_e4m3fn`, max=448 |
| **SHUFFLE** | ROCm tile-interleaved KV cache layout for coalesced AITER access |

---

## V3 vs V4 Side-by-Side

### MLA Architecture

| Aspect | V3 / V3.2 | V4 |
|---|---|---|
| Main KV cache | 512 BF16 latent + 64 BF16 rope = **576-dim (1152 bytes)** | 448 FP8 NoPE + 128 BF16 RoPE + 8 ue8m0 = **584 bytes** |
| Q head dim | 192 (128 NoPE + 64 RoPE) | **512** (448 NoPE + 64 RoPE) |
| Q projection | `fused_qkv_a_proj` + `q_a_layernorm` + `q_b_proj` | `fused_wqa_wkv` + `q_norm` + `wq_b` |
| KV projection | `kv_a_proj_with_mqa` + `kv_a_layernorm` + stored | `fused_wqa_wkv` (kv slice) + `kv_norm` + stored |
| Weight absorption | Yes: `ql_nope = q_nope × W_UK_T` → [T, 128, 512] | None — FlashMLA handles full 512-dim directly |
| SWA cache | None | Full 512-dim FP8, every layer, window_size=4096 |
| Per-layer type | All identical (no compress_ratio) | C4A / C128A (alternating) / SWA-only possible |
| Output projection | `o_proj` RowParallel | Inverse RoPE + `wo_a` grouped FP8 BMM + `wo_b` |
| `softmax_scale` | `1/√192` | `1/√512` |
| `attn_sink` | Not used | Learned per-head denominator bias |

### Indexer Architecture

| Aspect | V3 | V4 |
|---|---|---|
| K source | `wk_weights_proj(hidden)` → direct 128-dim K per token | `DeepseekCompressor` → 1 compressed K per N tokens |
| Weights source | Fused with K in single `wk_weights_proj` GEMM | Separate `weights_proj(hidden)` GEMM |
| K cache write | `indexer_k_quant_and_cache` (inside `SparseAttnIndexer`) | Inside compressor (`skip_k_cache_insert=True`) |
| `topk_indices` meaning | Direct token positions `[0..seq_len)` | Compressed positions `[0..seq_len//compress_ratio)` |
| Block size (indexer) | 64 slots/block | 256 slots/block |
| Q RoPE style | NeoX (first-half / second-half split) | GPT-J (interleaved pairs) |
| Q precision | FP8 only | FP8 (default) or MXFP4 (Blackwell SM100) |
| `cu_seqlen_ke` formula | `position + 1` | `(position + 1) // compress_ratio` |

### Stream Parallelism

| | V3 | V4 |
|---|---|---|
| Input GEMMs | Sequential (1 GEMM) | 4-way parallel (3 aux streams) |
| Indexer vs main MLA | Sequential | Overlapped (aux 0 vs default) |
| Compressor vs attention | N/A (no compressor) | Overlapped (aux stream 1 vs default) |
| Indexer Q vs K | Sequential | 2-way inner overlap (sub-stream events) |

### Scale Absorption

| | V3 FP8 | V4 FP8 | V4 MXFP4 |
|---|---|---|---|
| q_scale granularity | 1 scalar per (token, head) | 1 scalar per (token, head) | 4 block scalars per (token, head) |
| q_scale folded into weights_out? | Yes | Yes | No — passed separately to logit kernel |
| weights_out formula | `w × sq × (1/√128) × (1/√64)` | Same | `w × (1/√128) × (1/√64)` |

---

## KV Cache Formats: V3 and V4

| Cache | Model | Layout | Bytes/slot |
|---|---|---|---|
| Main MLA (BF16) | V3 | 512 BF16 latent + 64 BF16 rope = 576 dims | **1152 bytes** |
| Main MLA (FP8) | V3 | 512 FP8 + 64 FP8 + scale metadata | ~588 bytes |
| Indexer K | V3 | 128 FP8 + 4 float32 scale bytes | **132 bytes** |
| SWA KV cache | V4 | 448 FP8 (NoPE, 7×64 ue8m0 blocks) + 128 BF16 (RoPE) + 8 bytes scale/pad | **584 bytes** |
| Main MLA (compressed) | V4 | Same as SWA: 448 FP8 + 128 BF16 + 8 bytes | **584 bytes** |
| Indexer K (FP8) | V4 | 128 FP8 + 4 float32 scale | **132 bytes** |
| Indexer K (MXFP4) | V4 | 64 packed uint8 (4-bit E2M1) + 4 ue8m0 blocks | **68 bytes** |
| Compressor state | V4 | `2 × coff × head_dim` float32 per raw token | 1024–32768 bytes |

**V3 block_size:** indexer k_cache = 64 slots/block; main MLA KV = paged by vLLM default.
**V4 block_size:** indexer k_cache = 256 slots/block; SWA and main MLA KV = standard vLLM block size.

**FP8 NoPE block quantization (UE8M0):**

```
For 448-dim NoPE (7 blocks of 64 elements each):
  block_i: 64 FP8 values + 1 ue8m0 scale byte
  ue8m0 = ceil(log2(block_max / fp8_max)) + 127   (biased exponent, range [0,255])
  scale = 2^(ue8m0 - 127)
  fp8_max = 448 on OCP (gfx950/CUDA), 224 on FNUZ (gfx942)
```
