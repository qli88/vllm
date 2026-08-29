# Kimi-K3 Documentation Index

Master reference hub. Each linked doc covers one coherent topic in depth. Source: `moonshotai/Kimi-K3` config + vLLM implementation in `vllm/models/kimi_k3/`.

---

## Document Map

| Doc | What it covers |
|---|---|
| **[kimi_k3_layers.md](kimi_k3_layers.md)** | Per-layer type table (MLA vs KDA, MoE vs dense), layer assignment rules, layer structure diagram |
| **[kimi_k3_mla.md](kimi_k3_mla.md)** | Full MLA: Q/KV projections, weight absorption (W_UK/W_UV), prefill/decode paths, KV cache formats, all fused kernels |
| **[kimi_k3_kda.md](kimi_k3_kda.md)** | KDA (Gated Delta Net): architecture, fused decode kernel, Triton prefill backend, metadata builder, RecoverSSM |
| **[kimi_k3_moe.md](kimi_k3_moe.md)** | MoE routing (grouped top-k, noaux_tc), Latent MoE, LatentMoERunner tiers, MegaMoE, AttnRes residual |
| **[kimi_k3_e2e.md](kimi_k3_e2e.md)** | End-to-end walkthrough: tokenization → MLA layer → KDA layer → MoE → output token, concrete shapes |
| **[kimi_k3_speculative.md](kimi_k3_speculative.md)** | MTP draft heads and DSpark speculative decoding: architecture, weight mapping, DCP integration |
| **[kimi_k3_rocm.md](kimi_k3_rocm.md)** | All ROCm/AMD divergences: feature gap table, KDA differences, LatentMoERunner, AttnRes, gfx950 specifics |

The original combined draft is preserved at [kimi-k3.md](kimi-k3.md).

---

## Model Variants

| Class | HF model ID | Purpose |
|---|---|---|
| `KimiLinearForCausalLM` | `moonshotai/Kimi-Linear-48B-A3B-*` | Text-only MoE |
| `KimiK3ForConditionalGeneration` | `moonshotai/Kimi-K3` | Multimodal (text + vision) |
| `KimiK3MTPModel` | (spec-decode target) | MTP draft heads |
| `K3DSparkForCausalLM` | (spec-decode draft) | DSpark 5-layer dense MLA draft |

---

## Quick Config Reference (moonshotai-Kimi-K3)

Source: `/shareddata/data/models/moonshotai-Kimi-K3/config.json`

### Text Backbone (`text_config`)

| Parameter | Value | Notes |
|---|---|---|
| `hidden_size` | 7168 | Token hidden dimension |
| `num_hidden_layers` | 93 | Layers 0–92 |
| `num_attention_heads` | 96 | Full MLA heads |
| `num_key_value_heads` | 96 | Same as Q heads (MQA via latent) |
| `intermediate_size` | 33792 | Dense MLP intermediate size (layer 0 only) |
| `vocab_size` | 163840 | Token vocabulary size |
| `max_position_embeddings` | 1048576 | 1M context support |
| `rms_norm_eps` | 1e-5 | RMSNorm epsilon |
| `dtype` | `bfloat16` | Base compute dtype |

### MLA

| Parameter | Value | Notes |
|---|---|---|
| `q_lora_rank` | 1536 | Q low-rank bottleneck |
| `kv_lora_rank` | 512 | KV latent dim (= `L`) |
| `qk_nope_head_dim` | 128 | NoPE component per head (= `P`) |
| `qk_rope_head_dim` | 64 | RoPE component per head (= `R`) |
| `v_head_dim` | 128 | Value dim per head (= `V`) |
| `mla_use_nope` | `true` | K/V carry no positional encoding |
| `mla_use_output_gate` | `true` | Sigmoid output gate on attention |
| KV cache entry | 576 dims | = `kv_lora_rank` (512) + `qk_rope_head_dim` (64) |

### KDA (Linear Attention)

| Parameter | Value | Notes |
|---|---|---|
| `linear_attn_config.kda_layers` | 69 layers (see §Layer Table) | KDA instead of full MLA |
| `linear_attn_config.full_attn_layers` | 24 layers | Full MLA layers |
| `linear_attn_config.num_heads` | 96 | KDA heads |
| `linear_attn_config.head_dim` | 128 | `kDimK = kDimV = 128` |
| `linear_attn_config.short_conv_kernel_size` | 4 | Conv1d kernel width |
| `linear_attn_config.gate_lower_bound` | -5.0 | Clamp for gate stability |
| `linear_attn_config.use_full_rank_gate` | `true` | Full-rank gate projection |

### MoE

| Parameter | Value | Notes |
|---|---|---|
| `num_experts` | 896 | Total routed experts |
| `num_experts_per_token` | 16 | Experts selected per token |
| `num_shared_experts` | 2 | Always-active dense experts |
| `moe_intermediate_size` | 3072 | Expert intermediate dim |
| `routed_expert_hidden_size` | 3584 | **Latent MoE** bottleneck dim |
| `routed_scaling_factor` | 1.0 | Expert output scale (= no scaling) |
| `use_grouped_topk` | `true` | Grouped top-k routing |
| `num_expert_group` | 1 | Number of expert groups |
| `topk_group` | 1 | Experts per group |
| `topk_method` | `noaux_tc` | No-aux-loss token-constrained |
| `moe_renormalize` | `true` | Renormalize routing weights |
| `moe_router_activation_func` | `sigmoid` | Router score activation |
| `moe_layer_freq` | 1 | Every layer is MoE (after dense prefix) |
| `first_k_dense_replace` | 1 | Only layer 0 uses dense MLP |
| `latent_moe_use_norm` | `true` | RMSNorm between latent MoE stages |

### AttnRes

| Parameter | Value | Notes |
|---|---|---|
| `attn_res_block_size` | 12 | Bank size (# recent blocks) |

### Activation

| Parameter | Value | Notes |
|---|---|---|
| `hidden_act` | `situ` | SiTU activation (required for MegaMoE) |
| `activation_situ_beta` | 4.0 | SiTU `β` |
| `activation_situ_linear_beta` | 25.0 | SiTU linear `β` |

### Quantization (routed expert weights)

| Parameter | Value | Notes |
|---|---|---|
| `quant_method` | `compressed-tensors` | NVidia compressed-tensors format |
| `format` | `mxfp4-pack-quantized` | 4-bit MXFP4 quantized |
| `group_size` | 32 | 32-element quantization blocks |
| Ignored modules | `self_attn`, `shared_experts`, MLP `gate/up/down`, `lm_head`, `vision_tower`, `mm_projector` | These remain in BF16 |

### Vision (Kimi-K3 VL)

| Parameter | Value | Notes |
|---|---|---|
| `vt_hidden_size` | 1024 | Vision transformer hidden dim |
| `vt_num_hidden_layers` | 27 | ViT depth |
| `vt_num_attention_heads` | 12 | ViT heads |
| `vt_intermediate_size` | 4096 | ViT FFN size |
| `patch_size` | 14 | Vision patch size |
| `mm_hidden_size` | 1024 | Multimodal projection dim |
| `text_hidden_size` | 7168 | Target hidden size |
| `mm_projector_type` | `patchmergerv2` | Patch merger v2 |

---

## Layer Type Table

All 93 layers assigned from `linear_attn_config`:

```
Layer  0: Dense MLP  [full_attn_layers: NO — layer 0 is not in list; KDA: NO] → MLA + Dense MLP (first_k_dense_replace=1)
Layers 1-3:  KDA    + MoE  [kda_layers]
Layer  4:    MLA    + MoE  [full_attn_layers]
Layers 5-7:  KDA    + MoE
Layer  8:    MLA    + MoE
Layers 9-11: KDA    + MoE
Layer 12:    MLA    + MoE
Layers 13-15: KDA   + MoE
Layer 16:    MLA    + MoE
Layers 17-19: KDA   + MoE
Layer 20:    MLA    + MoE
Layers 21-23: KDA   + MoE
Layer 24:    MLA    + MoE
Layers 25-27: KDA   + MoE
Layer 28:    MLA    + MoE
Layers 29-31: KDA   + MoE
Layer 32:    MLA    + MoE
Layers 33-35: KDA   + MoE
Layer 36:    MLA    + MoE
Layers 37-39: KDA   + MoE
Layer 40:    MLA    + MoE
Layers 41-43: KDA   + MoE
Layer 44:    MLA    + MoE
Layers 45-47: KDA   + MoE
Layer 48:    MLA    + MoE
Layers 49-51: KDA   + MoE
Layer 52:    MLA    + MoE
Layers 53-55: KDA   + MoE
Layer 56:    MLA    + MoE
Layers 57-59: KDA   + MoE
Layer 60:    MLA    + MoE
Layers 61-63: KDA   + MoE
Layer 64:    MLA    + MoE
Layers 65-67: KDA   + MoE
Layer 68:    MLA    + MoE
Layers 69-71: KDA   + MoE
Layer 72:    MLA    + MoE
Layers 73-75: KDA   + MoE
Layer 76:    MLA    + MoE
Layers 77-79: KDA   + MoE
Layer 80:    MLA    + MoE
Layers 81-83: KDA   + MoE
Layer 84:    MLA    + MoE
Layers 85-87: KDA   + MoE
Layer 88:    MLA    + MoE
Layers 89-91: KDA   + MoE
Layer 92:    MLA    + MoE  [last full_attn_layer = 93, i.e. index 92]
```

**Summary:** 1 × Dense-MLP+MLA (layer 0), 24 × MLA+MoE (layers 4,8,12,...,92), 69 × KDA+MoE (layers 1-3, 5-7, ..., 89-91). Every layer except layer 0 is MoE.

**Pattern:** 3 KDA layers between each MLA layer (ratio 3:1), except layer 0 which is MLA+Dense.

---

## Key Tensor Shapes at a Glance

```
hidden_states:           [T, 7168]         bf16
Q (after q_b_proj):      [T, 96, 192]      bf16  (192 = 128 nope + 64 rope)
KV latent (c_kv):        [T, 512]          bf16
k_pe:                    [T, 64]           bf16
KV cache entry:          [576]             bf16 or fp8  (512 latent + 64 k_pe)
KDA state (conv1d):      [batch, 96, 4, 128]   bf16  (96 heads, kernel_width=4, dim=128)
KDA state (delta mem):   [batch, 96, 128, 128]  bf16  (outer product, kDimK×kDimV)
MoE gate logits:         [T, 896]          float32
Routing weights:         [T, 16]           float32
Expert W1 (latent up):   [896, 7168, 3584] MXFP4  (hidden→latent, per expert)
Expert W2 (latent down): [896, 3584, 7168] MXFP4  (latent→hidden, per expert)
AttnRes bank:            [12, 7168]        bf16  (12 blocks × hidden)
```

---

## Hardware Dispatch

`vllm/models/kimi_k3/__init__.py`:

```python
if current_platform.is_cuda():
    from .nvidia.model import KimiLinearForCausalLM, KimiK3ForConditionalGeneration
    from .nvidia.mtp   import KimiK3MTP as KimiK3MTPModel
elif current_platform.is_rocm():
    from .amd.linear   import KimiLinearForCausalLM
    from .amd.model    import KimiK3ForConditionalGeneration
    from .amd.mtp      import KimiK3MTP as KimiK3MTPModel
```

---

## K3 vs DeepSeek V3/V4: Architecture Comparison

| Dimension | DeepSeek V3 | DeepSeek V4 | Kimi K3 |
|---|---|---|---|
| **Attention layers** | All MLA | MLA + SWA | **MLA + KDA (hybrid 1:3 ratio)** |
| **KDA/linear-attn** | None | None | **Gated Delta Net recurrence** |
| **MLA positional encoding** | NoPE=False (has rope) | NoPE=False | **NoPE=True — no RoPE on K/V** |
| **Output gate** | No | Yes (wo_a/wo_b grouped BMM) | **Optional sigmoid g_proj gate** |
| **Attention residual** | No | No | **AttnRes — block-level prefix-sum softmax** |
| **MoE latent bottleneck** | No | No | **Yes — `routed_expert_hidden_size=3584`** |
| **MoE activation** | SiLU | SiTU | **SiTU (Sigmoid-gated TU)** |
| **MegaMoE** | No | Yes | **Yes (reuses DSv4, adapted for SiTU)** |
| **Expert latent dim** | full hidden (7168) | full hidden (7168) | **3584 (half hidden)** |
| **KV cache entry** | 576B (512 NoPE bf16 + 64 rope bf16) | 584B (fp8_ds_mla) | **512B (NoPE only, no rope key)** |
| **Fused key+cache kernel** | DSv3 key-concat | V4 compressor insert | **`fused_kimi_k3_mla_key_concat`** |
| **Low-latency GEMM** | `dsv3_fused_a_gemm` | DeepGEMM | **`dsv3_fused_a` + CuTeDSL skinny** |
| **Spec decode** | MTP | DSpark | **DSpark (cross-layer fused KV proj)** |
| **SM100 tail fusion** | No | Partial (wo_a grouped BMM) | **Full: CollectiveKernel + LamportCopy** |

## Prefill vs Decode Differences Per Component

| Component | Prefill | Decode |
|---|---|---|
| **AttnRes** | Writes block checkpoints on write layers; softmax-mixes prefix + stored blocks | Same math; residual_blocks already populated from prefill |
| **KDA conv** | `causal_conv1d_fn` — parallel causal scan over all T tokens; updates conv_state | `causal_conv1d_update` — one step per token; in-place conv_state update |
| **KDA recurrence** | `_flashkda_prefill` or `chunk_kda_with_fused_gate` — parallel scan `[1,T,H,D]` | `fused_kda_decode` (fused, SM90+) or `fused_recurrent_kda_packed_decode` (Triton) |
| **KDA state write** | `recurrent_state[state_indices] = last_state` (final state per sequence) | State updated in-place inside fused decode kernel |
| **MLA input GEMM** | Same `fused_qkv_a_proj` | Same |
| **MLA kv_b_proj** | Run for current tokens → gets k_nope, v in `[T, H, dim]` space directly | **Not run per-token** — absorbed into W_UK_T at load time; only run once |
| **MLA W_UV** | Not needed — v comes directly from kv_b_proj output | Applied post-attention via BMM2 to convert latent output → v-space |
| **MLA KV insert kernel** | `fused_mla_key_concat_kv_cache_insert` — prefill variant with K concat | `fused_mla_decode_q_concat_kv_cache_insert` — decode variant |
| **MLA attention** | `run_prefill_new_tokens` + optional chunked context merge | `impl.forward_mqa` — paged MQA attention over cached slots |
| **MoE latent tail** | COLUMN_PARALLEL (≥13 tokens → no aux stream overlap) | TAIL_FUSION (≤16, SM100) or ALLREDUCE_OVERLAP (≤256, any) |
| **MoE gate + down overlap** | Sequential (> 256 token threshold) | Overlapped on aux stream (≤ 256 tokens) |

---

## Low-Latency GEMM Dispatch (SM103 / B300)

**File:** `nvidia/low_latency_gemm.py`

On SM103 (NVIDIA B300/Blackwell) with BF16 dtype, `enable_kimi_k3_low_latency_gemm` replaces standard cuBLAS GEMMs with shape-tuned kernels at model-load time.

**Two backend options:**

| Backend | When used | Mechanism |
|---|---|---|
| `"dsv3_fused_a"` | M=1..16 for most projections | `ops.dsv3_fused_a_gemm` with PDL — same kernel as DeepSeek V3 decode |
| `"cute"` | M=1..16 for selected projections | `shape_dynamic_skinny_gemm` via CuTe DSL, per-(N,K,M) `SkinnyGemmConfig` |

**Per-projection dispatch table** (selected entries from `KIMI_K3_PROJECTIONS`):

| Projection | Local shape (N×K) | M=1 | M=2..4 | M=5..16 |
|---|---|---|---|---|
| `fused_qkv_a_proj` | 2112×7168 | dsv3 | dsv3 | dsv3 |
| `q_b_proj` | 2304×1536 | dsv3 | dsv3 | dsv3 |
| `routed_expert_down_proj` | 3584×7168 | cute | dsv3 | dsv3 |
| `routed_expert_up_proj` | 7168×3584 | cute | cute | — (cuBLAS) |
| `in_proj_qkvgfab` | 6288×7168 | cute | cute | cute |
| `o_proj` | 7168×1536 | cute | — | — |
| `lm_head` (TP8) | 20480×7168 | cute | cute | cute |
| `kv_a_proj_with_mqa` | 576×7168 | dsv3 | dsv3 | dsv3 |
| `out_proj` (KDA) | 12288×7168 | cute | cute | cute |

**Residual fusion:** `KimiK3LowLatencyLinearMethod.apply_with_residual` supports fusing the GEMM with a residual add:

```python
# Instead of: out = linear(x); out = out + residual
# Use:         residual = gemm_with_residual_add(x, weight, residual)
# One fused kernel, one HBM round-trip for residual
```

**Installation at load time:** `enable_kimi_k3_low_latency_gemm` walks all `LinearBase` modules, matches by `(local_N, K)` shape in the table, and replaces `quant_method` with a `KimiK3LowLatencyLinearMethod` carrying the per-token-count dispatch plan.

---

## Worked Examples: Config and Layer Lookup

### 1. Layer assignment lookup

Given the config arrays, determine the type of layers 0, 2, 3, 47, and 92:

```python
kda_layers      = set(config.linear_attn_config["kda_layers"])       # 69 indices
full_attn_layers = set(config.linear_attn_config["full_attn_layers"]) # 24 indices
first_k_dense   = config.first_k_dense_replace                        # = 1

for idx in [0, 2, 3, 47, 92]:
    is_kda   = idx in kda_layers
    is_mla   = not is_kda          # layer 0 defaults to MLA (in neither list)
    is_dense = idx < first_k_dense # True only for idx == 0
    is_moe   = not is_dense
    print(f"Layer {idx:2d}: {'KDA' if is_kda else 'MLA'} + {'Dense MLP' if is_dense else 'MoE'}")
```

Result:

| Layer | is_kda | is_mla | is_dense | is_moe | Type |
|---|---|---|---|---|---|
| 0 | False | True | True | False | MLA + Dense MLP |
| 2 | True | False | False | True | KDA + MoE |
| 3 | True | False | False | True | KDA + MoE |
| 47 | True | False | False | True | KDA + MoE |
| 92 | False | True | False | True | MLA + MoE |

Layer 0 is not in `kda_layers` or `full_attn_layers`; the model code falls through to the MLA default. Layers 2, 3, 47 are in `kda_layers` (positions 2, 3 in the first triple; 47 in the triple before MLA layer 48). Layer 92 is the last entry in `full_attn_layers`.

---

### 2. Key tensor shapes at different TP settings

`hidden_size=7168`, `num_heads=96`, `head_dim=128`, `q_lora_rank=1536`, `kv_lora_rank=512`.

| Tensor | TP=1 | TP=8 | TP=16 |
|---|---|---|---|
| Q after `q_b_proj` | `[T, 96, 192]` | `[T, 12, 192]` | `[T, 6, 192]` |
| KV latent `c_kv` | `[T, 512]` | `[T, 512]` (replicated) | `[T, 512]` (replicated) |
| `k_pe` | `[T, 64]` | `[T, 64]` (replicated) | `[T, 64]` (replicated) |
| Expert `W1` (latent down) | `[896, 7168, 3584]` MXFP4 | `[112, 7168, 3584]` MXFP4 | `[56, 7168, 3584]` MXFP4 |
| Expert `W2` (latent up) | `[896, 3584, 7168]` MXFP4 | `[112, 3584, 7168]` MXFP4 | `[56, 3584, 7168]` MXFP4 |
| `in_proj_qkvgfab` (KDA) | `[6288, 7168]` | `[786, 7168]` | `[393, 7168]` |

Notes:
- `c_kv` and `k_pe` are not sharded — they are replicated across TP ranks because weight absorption fuses them on each rank independently.
- `in_proj_qkvgfab` packs Q+K+V+g+f\_a+b+beta per head: `96 × (128+128+128+1+1+128+1) = 96 × 387 / TP` rows. At TP=8: `96/8 × 387 = 12 × 387 = 4644`; at TP=16: `6 × 387 = 2322`. The full-rank gate (`use_full_rank_gate=True`) replaces the scalar `g` with a `128`-dim projection, adding `96 × 128 / TP` rows; the exact total varies by gate config.
- Expert sharding: 896 experts ÷ TP ranks (expert parallelism, not tensor-parallel columns in practice). Shown above as equal splits; actual assignment depends on EP vs TP grouping.

---

### 3. Memory estimate at seq_len=32768, 93 layers, TP=8

**MLA KV cache (24 MLA layers, BF16):**
```
24 layers × 32768 tokens × 576 dims × 2 bytes = 906,854,400 bytes ≈ 865 MB
```
Breakdown: each token caches `c_kv [512 dims]` + `k_pe [64 dims]` = 576 dims × 2 bytes = 1152 bytes.

**KDA recurrent state (69 KDA layers, BF16, fixed regardless of seq_len):**
```
Conv1d state:  96 × 4 × 128 × 2 =    98,304 bytes/layer
Delta memory:  96 × 128 × 128 × 2 = 3,145,728 bytes/layer
Per KDA layer: ~3.1 MB

69 layers × 3,145,728 bytes ≈ 217 MB  (+ 69 × 98 KB ≈ 6.6 MB conv state)
Total KDA: ~224 MB per sequence
```

**Total per sequence at 32K tokens:**
```
MLA KV cache:   ~865 MB
KDA state:      ~224 MB  (fixed; same at 1K or 1M tokens)
Total:          ~1.09 GB per sequence
```

At 128K tokens the MLA KV cache grows ~4× (to ~3.5 GB) while KDA state remains at ~224 MB. This asymmetry makes KDA the memory-efficient path for very long contexts.

---

## Glossary

| Term | Meaning |
|---|---|
| **MLA** | Multi-head Latent Attention — compresses KV into a 512-dim latent vector |
| **KDA** | Kimi Delta Attention — linear (recurrent) attention using Gated Delta Net |
| **NoPE** | No Positional Encoding on K/V latent — only k_pe carries position |
| **AttnRes** | Attention Residual — block-level residual bank mixing recent layer outputs |
| **Latent MoE** | Two-stage expert: hidden→latent→hidden (vs standard gate+up+down) |
| **LatentMoERunner** | Dispatcher for the replicated routed up-projection with 3 hardware tiers |
| **MTP** | Multi-Token Prediction — additional draft heads for speculative self-distillation |
| **DSpark** | 5-layer dense MLA draft model for speculative decoding |
| **DCP** | Decode Context Parallelism — KV cache sharded across ranks |
| **RecoverSSM** | Mechanism to replay KDA states from checkpoint (for EAGLE3 + spec-decode reset) |
| **W_UK_T** | `kv_b_proj` K-nope weight transposed for Q-side absorption |
| **W_UV** | `kv_b_proj` V weight pre-computed for output latent projection |
| **BMM1** | First batched matmul in absorbed MLA decode: Q_nope × W_UK_T × c_kv |
| **BMM2** | Second batched matmul: attention_weights × W_UV × c_kv |
| **SiTU** | Custom activation: `sigmoid(beta*x) * (x + linear_beta)` (replaces SiLU) |
| **MXFP4** | 4-bit micro-scaled float for routed expert weights (group_size=32) |
| **MegaMoE** | SM100 FP8/FP4 expert GEMM via `deep_gemm.fp8_fp4_mega_moe` |
| **GEMM-RS** | Fused GEMM + reduce-scatter via NCCL symmetric memory (SM100 only) |
| **PDL** | Programmatic Dependent Launch — overlaps GEMM with dependent kernels |
| **TAIL_FUSION** | LatentMoERunner Tier-0: CuTe DSL collective fusing latent+norm+RS+up-proj |
