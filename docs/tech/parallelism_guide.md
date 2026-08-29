# Parallelism Strategies in LLM Serving (vLLM)

## Overview

Three orthogonal parallelism axes are used in large-model serving:

| Strategy | Splits | Communication | Best For |
|---|---|---|---|
| **TP** — Tensor Parallelism | Layer weight matrices | AllReduce per layer | Model too large for 1 GPU; latency-sensitive |
| **DP** — Data Parallelism | Requests | None during inference | High throughput; model fits on fewer GPUs |
| **EP** — Expert Parallelism | MoE experts | AlltoAll per MoE layer | Large MoE models (Mixtral, DeepSeek) |

These strategies compose: TP+DP, TP+EP, and TP+EP+DP are all valid and common.

---

## 1. Tensor Parallelism (TP)

### Concept

Each weight matrix is sharded across N GPUs. All GPUs hold a slice of every layer
and cooperate on every forward pass. After each layer's partial results are computed
locally, an **AllReduce** synchronizes the full output.

Two splitting patterns:

- **Column-parallel** (e.g., Q/K/V projections, FFN gate+up): split output dimension.
  Each GPU computes its portion of the output independently, then AllReduce sums them.
- **Row-parallel** (e.g., attention output proj, FFN down proj): split input dimension.
  Each GPU holds part of the weight rows, computes a partial output sum, then AllReduce.

### Example: Llama 3.1 70B on 8×H100 (TP=8)

**Model dimensions:**
```
d_model  = 8192
n_heads  = 64 (Q), n_kv_heads = 8 (GQA), d_head = 128
d_ffn    = 28672 (SwiGLU: gate + up each = 28672, total = 57344)
n_layers = 80
```

**Attention weight sharding (TP=8):**
```
W_q = [8192, 8192]  →  each GPU: W_q[:, i*1024:(i+1)*1024]  (8 Q-heads each)
W_k = [8192, 1024]  →  each GPU: W_k[:, i*128:(i+1)*128]   (1 KV-head each)
W_v = [8192, 1024]  →  each GPU: W_v[:, i*128:(i+1)*128]   (1 KV-head each)

Each GPU runs full attention locally on its heads.
W_o = [8192, 8192]  →  row-parallel: each GPU holds W_o[i*1024:(i+1)*1024, :]
```

**FFN weight sharding (SwiGLU, TP=8):**
```
W_gate = [8192, 28672]  →  each GPU: [8192, 3584]  (column-parallel)
W_up   = [8192, 28672]  →  each GPU: [8192, 3584]  (column-parallel)
W_down = [28672, 8192]  →  each GPU: [3584, 8192]  (row-parallel)

Each GPU computes:
  local_ffn  = SiLU(x @ W_gate_local) * (x @ W_up_local)   # shape [B*S, 3584]
  partial    = local_ffn @ W_down_local                      # shape [B*S, 8192]
```

**AllReduce — what, when, and how large:**
```
Trigger:     After each row-parallel projection (output proj, FFN down proj)
Operation:   Sum reduction across all 8 GPUs
Payload:     [batch × seq_len, 8192] in bf16

Example (batch=1, seq=2048):
  Raw tensor:   2048 × 8192 × 2 bytes = 32 MB
  Ring AllReduce sends/receives: 32 MB × (8-1)/8 ≈ 28 MB per GPU

Per layer:     2 AllReduces (attention output + FFN output)
Per pass:      80 layers × 2 = 160 AllReduces
Total data:    160 × 28 MB ≈ 4.5 GB transferred per GPU per decode step

NVLink bandwidth: ~900 GB/s → each AllReduce ≈ 0.06 ms → 160 × 0.06 ≈ 10 ms overhead
```

**vLLM flag:** `--tensor-parallel-size 8`

**When to use:** Model does not fit on a single GPU. Each additional TP degree adds
AllReduce latency on the critical path, so prefer the minimum TP needed to fit the model.

---

## 2. Data Parallelism (DP)

### Concept

Run N independent full-model replicas. Each replica handles a disjoint subset of
requests. There is **zero inter-replica communication** during inference — the frontend
load-balances incoming requests across replicas.

This differs from training DP: training requires AllReduce to sync gradients after each
backward pass. In inference, replicas are fully independent.

### Example: Llama 3.1 8B on 8×A10G (DP=8)

**Fit check:**
```
8B parameters × 2 bytes (bf16) = 16 GB weights per replica
A10G VRAM: 24 GB  →  8 GB remaining for KV cache per GPU  ✓
```

**Request routing:**
```
GPU0: full Llama 8B, serving [r0 .. r47]    (48 concurrent)
GPU1: full Llama 8B, serving [r48 .. r95]
...
GPU7: full Llama 8B, serving [r336 .. r383]

Frontend (least-busy routing):
  new request r400 → GPU3 (shortest queue)
```

**KV cache per GPU:**
```
Per token per layer: 2 (K+V) × 8 kv_heads × 128 d_head × 2 bytes = 4 KB
For 48 concurrent requests, 2048 tokens, 32 layers:
  48 × 2048 × 32 × 4 KB = 12.6 GB KV cache per GPU
```

**Per-GPU forward pass (one decode step, 48 requests):**
```
Input shape:   [48, 1, 4096]   (48 requests, 1 new token, hidden dim)
Output:        [48, 1, 32000]  → sample → 48 next tokens
FLOPs:         ~2 × 2 × 8B ≈ 32 GFLOPs
Communication: none
```

**Throughput:** 8× vs a single GPU replica — perfect linear scaling.
Bottleneck becomes frontend dispatcher and network I/O.

**vLLM flag:** `--data-parallel-size 8`

**When to use:** Model fits on available per-GPU memory and throughput (requests/s)
is the bottleneck. DP does not reduce per-replica KV cache capacity.

---

## 3. Expert Parallelism (EP)

### Concept

For Mixture-of-Experts (MoE) models, each transformer layer has N experts (each a
separate FFN). EP places different experts on different GPUs. Token routing uses
**AlltoAll**: each GPU sends tokens to whatever GPU owns the target expert, runs the
expert, then sends results back.

### Example: Mixtral 8×7B on 4×A100 (EP=4)

**Expert weight memory:**
```
Each expert FFN: [4096 → 14336 → 4096]
Per expert: 2 × 4096 × 14336 × 2 bytes = 234 MB
8 experts × 32 MoE layers = 59.8 GB total expert weights

EP=4: each GPU holds 2 experts = ~15 GB expert weights per GPU  ✓

GPU0: Expert 0, Expert 1
GPU1: Expert 2, Expert 3
GPU2: Expert 4, Expert 5
GPU3: Expert 6, Expert 7
```

**Token routing — full step-by-step (batch=512 tokens, top-2 routing):**
```
Step 0 — Router scores (runs on all GPUs, result same on all):
  scores: [512, 8]  →  top-2 per token  →  1024 expert calls total

Example assignments:
  token_0  → Expert 3 (GPU1),  Expert 7 (GPU3)
  token_1  → Expert 1 (GPU0),  Expert 5 (GPU2)
  token_2  → Expert 0 (GPU0),  Expert 2 (GPU1)
  ...

Step 1 — Dispatch AlltoAll:
  GPU0 keeps:         tokens routed to Expert 0, 1  (~125 tokens)
  GPU0 → GPU1:        tokens routed to Expert 2, 3  (~128 tokens)
  GPU0 → GPU2:        tokens routed to Expert 4, 5  (~130 tokens)
  GPU0 → GPU3:        tokens routed to Expert 6, 7  (~127 tokens)

  Token payload: [4096] × bf16 = 8 KB
  Data sent from GPU0: (128 + 130 + 127) × 8 KB ≈ 3.1 MB

Step 2 — Expert compute (parallel on all GPUs):
  GPU0: Expert 0 and 1 process their ~125 tokens each
  GPU1: Expert 2 and 3 process their ~128 tokens each
  ...

Step 3 — Combine AlltoAll (reverse of dispatch):
  Each GPU sends computed results back to originating GPU
  Same ~3.1 MB per GPU

Step 4 — Weighted sum on originating GPU:
  For each token, combine the 2 expert outputs with router weights
```

**Per-layer communication:** 2 AlltoAlls × ~3.1 MB × 32 layers ≈ 200 MB per GPU per pass.
Goes over InfiniBand (inter-node) or NVLink (intra-node depending on topology).

**Load imbalance risk:** With top-2 routing and 8 experts, expected ~128 tokens per
expert, but routing is not uniform. Expert 0 might get 200 tokens while Expert 7 gets
50. This is a known practical problem (addressed by auxiliary loss during training or
dynamic routing at inference time).

**vLLM flag:** `--expert-parallel-size 4`

**When to use:** Large MoE models where expert weights dominate memory. EP lets expert
weights spread across GPUs without replicating them, unlike DP.

---

## 4. Combinations

### 4.1 TP + DP

**Concept:** Each DP replica is itself tensor-parallel. TP reduces per-GPU memory so
the model fits; DP multiplies throughput.

**Example: Llama 3.1 70B on 16×H100 (TP=8, DP=2)**
```
DP Replica 0: GPUs  0-7   (TP=8 within)  ←  ~50% of requests
DP Replica 1: GPUs 8-15   (TP=8 within)  ←  ~50% of requests

Communication map:
  GPU0-7:   AllReduce with each other (160× per decode step, NVLink)
  GPU8-15:  AllReduce with each other (160× per decode step, NVLink)
  GPU0↔GPU8, ...: NO communication (independent replicas)
```

**Memory per GPU:**
```
70B weights / 8 GPUs = 17.5 GB weights per GPU
H100 VRAM: 80 GB  →  62.5 GB available for KV cache per GPU

KV cache per token (Llama 70B, 80 layers, 8 KV heads, d_head=128):
  2 × 8 × 128 × 2 × 80 = 327 KB per token

Max tokens per replica: 62.5 GB × 8 GPUs / 327 KB = 1.53M tokens
```

**vLLM flags:** `--tensor-parallel-size 8 --data-parallel-size 2`

**Tradeoff vs pure DP=16 (if model fit):** TP+DP has AllReduce overhead (10 ms/step);
pure DP has none. Use TP only when needed to fit the model.

---

### 4.2 TP + EP

**Concept:** Attention layers use TP (AllReduce within a GPU group via NVLink); MoE FFN
layers use EP (AlltoAll across GPU groups via InfiniBand). The two parallelisms target
different parts of the model.

**Example: DeepSeek-V3 (671B, 256 experts) on 32×H100 (TP=4, EP=8)**
```
GPU layout:
  EP rank 0: GPUs  0, 1, 2, 3   →  owns experts   0-31
  EP rank 1: GPUs  4, 5, 6, 7   →  owns experts  32-63
  EP rank 2: GPUs  8, 9,10,11   →  owns experts  64-95
  ...
  EP rank 7: GPUs 28,29,30,31   →  owns experts 224-255

Each EP rank is a 4-GPU NVLink island (TP group).
```

**Attention layers — TP=4 within each EP rank:**
```
DeepSeek-V3: d_model=7168, n_heads=128

Within EP rank 0 (GPUs 0-3):
  Each GPU owns 32 Q-heads
  AllReduce after output projection:
    payload = [batch×seq, 7168] × bf16
    For batch=1, seq=4096: 4096 × 7168 × 2 = 58 MB
    Ring AllReduce over 4 GPUs: ~44 MB moved per GPU
  Travels over NVLink (~900 GB/s) — fast
```

**MoE FFN layers — EP=8 across EP ranks:**
```
DeepSeek-V3: 256 experts, top-8 routing, d_expert_ffn=2048

Dispatch AlltoAll (across 8 EP ranks via InfiniBand):
  4096 tokens × top-8 = 32768 expert calls, distributed across 8 ranks
  ~4096 calls per EP rank on average

  Token hidden state: [7168] × bf16 = 14 KB
  Data from EP rank 0 to each other rank: 4096 × 14 KB = 57 MB
  Total off-rank data: 7 × 57 MB = 400 MB per EP rank per AlltoAll

  Two AlltoAlls per MoE layer → 800 MB per layer
  (Travels over InfiniBand, ~400 GB/s — the bottleneck)
```

**Communication hierarchy:**
```
NVLink  (intra-EP-rank, ~900 GB/s): TP AllReduces       ~0.06 ms each
InfiniBand (inter-EP-rank, ~400 GB/s): EP AlltoAlls     ~1 ms each
```

**vLLM flags:** `--tensor-parallel-size 4 --expert-parallel-size 8`

**Why not pure TP=32?** AllReduce scales poorly — 32-GPU AllReduce requires crossing
InfiniBand anyway and doubles communication volume. TP+EP keeps AllReduces local
(NVLink) and only crosses IB for EP dispatches.

---

### 4.3 TP + EP + DP

**Concept:** Full combination for maximum scale. Each DP replica is internally TP+EP.

**Example: DeepSeek-V3 on 64×H100 (TP=4, EP=8, DP=2)**
```
DP Replica 0: GPUs  0-31  (TP=4, EP=8 internally)  ←  ~50% of requests
DP Replica 1: GPUs 32-63  (TP=4, EP=8 internally)  ←  ~50% of requests

Within each replica:
  8 EP ranks × 4 TP GPUs = 32 GPUs
  NVLink AllReduces within each 4-GPU TP group
  InfiniBand AlltoAlls across 8 EP ranks

Zero cross-replica communication.
```

**vLLM flags:** `--tensor-parallel-size 4 --expert-parallel-size 8 --data-parallel-size 2`

---

## 5. Communication Cost Summary

Baseline: batch=32 tokens, one decode step.

| Strategy | Per-GPU Operation | Data Volume | Latency |
|---|---|---|---|
| TP=8 (Llama 70B) | 160× AllReduce (NVLink) | 4.5 GB | ~10 ms |
| DP=8 (Llama 8B) | None | 0 | 0 ms |
| EP=8 (Mixtral) | 64× AlltoAll (2 per MoE layer × 32 layers, IB) | ~200 MB | ~5 ms |
| TP=4 + EP=8 (DeepSeek-V3) | AllReduce (NVLink) + AlltoAll (IB) | ~2.2 GB + 400 MB | ~6 ms |

**Key insight:** TP AllReduces are on the critical path of every single layer and hurt
decode latency directly. EP AlltoAlls are large but only occur in MoE layers and can
partially overlap with compute. DP adds zero latency but requires enough VRAM to fit a
full model replica per DP unit.

---

## 6. Decision Guide

```
Does the model fit on 1 GPU?
  Yes → use DP only (zero communication overhead)
  No  → need TP and/or EP

Is it a MoE model?
  Yes → prefer EP to spread expert weights; add TP if attention is too large
  No  → use TP; minimize TP degree to minimum needed to fit

Need higher throughput after fitting the model?
  Add DP replicas on top of whatever TP/EP you need

Latency-critical (e.g., chat, interactive)?
  Minimize TP (more AllReduces = more latency per token)
  DP doesn't hurt latency, EP has moderate impact

Throughput-critical (e.g., batch inference)?
  Maximize DP; accept higher TP if needed for fit
```

---

## 7. Quick Reference: vLLM Flags

```bash
# Pure TP: Llama 70B on 8 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8

# Pure DP: Llama 8B, 8 replicas
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --data-parallel-size 8

# TP+DP: Llama 70B, 2 replicas each using 8 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --data-parallel-size 2

# EP: Mixtral on 4 GPUs
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --expert-parallel-size 4

# TP+EP: DeepSeek-V3 on 32 GPUs
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 4 \
  --expert-parallel-size 8

# TP+EP+DP: DeepSeek-V3 on 64 GPUs (2 replicas)
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 4 \
  --expert-parallel-size 8 \
  --data-parallel-size 2
```
