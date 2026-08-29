# DeepSeek V4: MoE Routing and Expert Computation

[← Index](deepseek_index.md) | [End-to-End Example](deepseek_v4_e2e.md)

Covers the two MoE variants used across V4's 61 layers: hash routing (layers 0–2) and noaux_tc routing (layers 3–60). Includes `sqrtsoftplus`, the `e_score_correction_bias` mechanism, SwiGLU computation, expert GEMM shapes, DSpark layers, and the shared expert.

---

## Table of Contents

1. [MoE Overview](#1-moe-overview)
2. [Routing Type Assignment](#2-routing-type-assignment)
3. [Hash MoE (Layers 0–2)](#3-hash-moe-layers-02)
   - 3.1 [tid2eid: the hash table](#31-tid2eid-the-hash-table)
   - 3.2 [Weight computation](#32-weight-computation)
   - 3.3 [Why hash routing in early layers](#33-why-hash-routing-in-early-layers)
4. [Standard noaux_tc MoE (Layers 3–60)](#4-standard-noaux_tc-moe-layers-360)
   - 4.1 [sqrtsoftplus activation](#41-sqrtsoftplus-activation)
   - 4.2 [e_score_correction_bias](#42-e_score_correction_bias)
   - 4.3 [Selection vs weighting: the critical separation](#43-selection-vs-weighting-the-critical-separation)
   - 4.4 [Full routing algorithm](#44-full-routing-algorithm)
5. [Expert Computation: SwiGLU + MXFP4](#5-expert-computation-swiglu--mxfp4)
   - 5.1 [Expert forward pass](#51-expert-forward-pass)
   - 5.2 [Expert weight shapes](#52-expert-weight-shapes)
   - 5.3 [Weighted reduction](#53-weighted-reduction)
6. [Shared Expert](#6-shared-expert)
7. [DSpark Layers (58, 59, 60)](#7-dspark-layers-58-59-60)
8. [Router Kernel: `FusedTopKBiasRouter`](#8-router-kernel-fusedtopkbiasrouter)
9. [Worked Examples](#9-worked-examples)
   - 9.1 [Hash MoE: token "▁Paris" at layer 0](#91-hash-moe-token-▁paris-at-layer-0)
   - 9.2 [Standard MoE: same token at layer 5](#92-standard-moe-same-token-at-layer-5)
   - 9.3 [Bias effect: under-loaded vs over-loaded expert](#93-bias-effect-under-loaded-vs-over-loaded-expert)
   - 9.4 [Expert computation — SwiGLU with clamp, concrete numbers](#94-expert-computation--swiglu-with-clamp-concrete-numbers)
   - 9.5 [Complete routing for one token across all 61 layers](#95-complete-routing-for-one-token-across-all-61-layers)
10. [ROCm MoE: AITER `fused_moe` and Padding Fix](#10-rocm-moe-aiter-fused_moe-and-padding-fix)
11. [Key Implementation Files](#11-key-implementation-files)

---

## 1. MoE Overview

Every one of the 61 decoder layers has an MoE FFN block (not dense). The structure is:

```
hidden_states [T, 7168]   (after attention + RMSNorm)
      │
      ├── gate(hidden_states)  [7168 → 384]   bf16 → float32
      │       ↓
      │   sqrtsoftplus(logits)  → scores [T, 384]
      │       ↓
      │   top-6 selection (hash or biased score)  → topk_ids [T, 6]
      │       ↓
      │   routing weights  → topk_weights [T, 6]  (normalized + scaled)
      │
      ├── Expert dispatch: send token t to its 6 selected experts
      │       For each expert e in topk_ids[t]:
      │         expert_out_e = W2[e] @ silu(clamp(W1[e] @ hidden[t])) × topk_weights[t, e]
      │       routed_out[t] = sum(expert_out_e for e in topk_ids[t])
      │
      └── shared_expert(hidden_states)   (always active, all tokens)
              ↓
      output = routed_out + shared_expert_out   [T, 7168]
```

Key constants from config:
- `n_routed_experts = 384` — the pool size for top-6 selection
- `num_experts_per_tok = 6` — top-6 per token
- `n_shared_experts = 1` — 1 shared expert
- `routed_scaling_factor = 2.5` — amplifies expert contributions post-normalization
- `norm_topk_prob = true` — renormalize 6 selected weights to sum=1 before scaling

---

## 2. Routing Type Assignment

| Layers | Routing type | Key parameter |
|---|---|---|
| 0, 1, 2 | **Hash MoE** | `tid2eid [129280, 6]` — lookup table |
| 3 – 60 | **Standard noaux_tc** | `e_score_correction_bias [384]` — trained bias |

Both types use `sqrtsoftplus` for the weight computation. The difference is in how expert **indices** are chosen:
- Hash: from `tid2eid[token_id]` — no score comparison
- Standard: `argtopk(scores + bias, k=6)` — biased score ranking

The `FusedTopKBiasRouter` kernel handles both modes. When `hash_indices_table is not None`, it uses the hash path; when `e_score_correction_bias is not None`, it uses biased top-K.

---

## 3. Hash MoE (Layers 0–2)

### 3.1 `tid2eid`: the hash table

```python
self.gate.tid2eid = nn.Parameter(
    torch.randint(0, 384, (129280, 6)),   # shape: [vocab_size, topk]
    requires_grad=False,
)
```

Shape: `[129280, 6]`, dtype `int32`. Loaded from checkpoint. One row per vocabulary token; each row is exactly 6 expert IDs that this token always routes to, regardless of content.

**Example entries** (illustrative):

```
tid2eid[791]   = [17, 203, 91, 5, 318, 7]      # token "The"
tid2eid[96310] = [42, 11, 201, 83, 317, 156]   # token "▁Eiffel"
tid2eid[17720] = [5, 88, 240, 107, 329, 21]    # token "▁Tower"
```

These are **fixed** — every occurrence of token 791 ("The") at layer 0 routes to experts {17, 203, 91, 5, 318, 7}, no matter the context.

### 3.2 Weight computation

Expert selection is hash-based, but the **weights** are still computed from gate logits:

```python
# 1. Run gate (linear layer):
logits = gate(hidden_states)   # [T, 384]  bf16

# 2. Apply sqrtsoftplus to all 384:
scores = sqrt(log(1 + exp(logits)))   # [T, 384]  float32

# 3. Gather weights at the 6 hash-selected positions:
raw_weights = scores[:, topk_ids]   # [T, 6]

# 4. Renormalize so selected weights sum to 1:
renorm = raw_weights / sum(raw_weights, dim=-1, keepdim=True)  # [T, 6]

# 5. Scale:
final_weights = renorm * 2.5   # [T, 6]
```

The gate logits are computed but only used for weight magnitudes. The `e_score_correction_bias` is **not** added in hash layers.

### 3.3 Why hash routing in early layers

Layers 0–2 process raw token embeddings where semantic content is not yet fully computed. Content-based routing (which expert to pick based on the meaning of the hidden state) would be unreliable at this stage. Hash routing provides:
- **Stability**: same token → same experts, no instability from immature representations
- **Load balance by design**: the hash table is initialized randomly and then fine-tuned so that experts are visited roughly equally across the vocabulary
- **Reduced complexity**: no argtopk over 384 experts for early layers

The learned gate weights `scores` are still used for routing **magnitudes** — how much each expert contributes — which remains meaningful even when the content representation is rough.

---

## 4. Standard noaux_tc MoE (Layers 3–60)

### 4.1 `sqrtsoftplus` activation

Applied element-wise to all 384 gate logits:

```
score(x) = sqrt(softplus(x)) = sqrt(ln(1 + e^x))
```

Properties compared to alternatives:

| Property | sqrtsoftplus | sigmoid | relu | softmax |
|---|---|---|---|---|
| Always positive | Yes | Yes | No | Yes |
| Bounded above | No (grows as √x) | Yes (→1) | No | Yes (→1) |
| Smoothness at 0 | Smooth | Smooth | Corner | Smooth |
| Gradient at x=0 | 0.5/√ln(2) ≈ 0.60 | 0.25 | 0 (subgrad) | varies |
| Score magnitudes | Moderate | Small | Large | Small |

The `sqrt` dampens outliers: a very large logit x gives score ≈ √x rather than x (softplus) or exp(x) (softmax). This makes the routing weights more robust to extreme gate values.

**Numerical example:**

```
logit x = 2.0:  score = sqrt(ln(1 + e^2.0)) = sqrt(ln(1 + 7.389)) = sqrt(ln(8.389)) = sqrt(2.127) = 1.459
logit x = 0.0:  score = sqrt(ln(1 + 1)) = sqrt(ln(2)) = sqrt(0.693) = 0.833
logit x = -2.0: score = sqrt(ln(1 + e^-2.0)) = sqrt(ln(1.135)) = sqrt(0.127) = 0.357
logit x = -5.0: score = sqrt(ln(1 + e^-5.0)) ≈ sqrt(e^-5.0) ≈ sqrt(0.0067) = 0.082
```

Even strongly negative logits produce small positive scores, so all 6 selected experts always receive meaningful (though small) weights.

### 4.2 `e_score_correction_bias`

```python
self.gate.e_score_correction_bias = nn.Parameter(
    torch.empty(384, dtype=torch.float32),
    requires_grad=False,   # trained parameter, fixed at inference
)
```

Shape: `[384]`, one bias per expert. Loaded from checkpoint as `*.ffn.gate.bias`.

**Purpose (`noaux_tc` = no-auxiliary-loss token-constrained):**

Traditional MoE uses an auxiliary load-balancing loss during training to prevent expert collapse (all tokens routing to the same few experts). This auxiliary loss has a weighting hyperparameter and can interfere with the main task loss.

V4 instead trains `e_score_correction_bias` to shift apparent expert attractiveness without distorting the actual routing weights. At inference, no auxiliary loss is needed because the biases already encode the balanced routing pattern.

**How biases are set:**
- Positive bias → expert is under-utilized → boost its apparent score → more tokens routed to it
- Negative bias → expert is over-utilized → suppress its apparent score → fewer tokens routed

**Example bias values** (illustrative, from a trained checkpoint):

```
expert   0: bias = +0.08   (slightly under-used)
expert  42: bias = +0.15   (notably under-used, gets boosted)
expert 107: bias = -0.12   (over-used, gets suppressed)
expert 200: bias =  0.00   (well-balanced)
expert 350: bias = -0.19   (heavily over-used)
```

### 4.3 Selection vs weighting: the critical separation

The bias affects only **which experts are selected**, not **how much they contribute**:

```python
# Selection uses biased scores:
scores_biased = scores + e_score_correction_bias   # [T, 384]
topk_ids = argtopk(scores_biased, k=6)             # [T, 6]

# Weights use UNBIASED scores at the selected positions:
raw_weights = scores[topk_ids]   # [T, 6]  ← no bias here
renorm = raw_weights / sum(raw_weights)
final_weights = renorm * 2.5
```

**Why this separation matters:**

If biases were also applied to weights, an under-loaded expert with a high bias would receive amplified weight, distorting the model's learned routing magnitudes. By keeping weights bias-free, the model's learned magnitude structure is preserved — only the selection decision is corrected for load balance.

**Example:**

```
Suppose expert 42 has bias=+0.15 and gate logit=1.2 for a token.
  score = sqrtsoftplus(1.2) = 1.163
  biased_score = 1.163 + 0.15 = 1.313   ← used for selection (more likely to be top-6)
  weight (if selected) = based on 1.163   ← bias NOT included
  → Expert 42 is selected more often but contributes exactly as much as its raw score says.
```

### 4.4 Full routing algorithm

```python
def route_standard(hidden_states, gate, e_score_correction_bias):
    # Step 1: Gate logits
    logits = gate(hidden_states)   # [T, 384]  bf16
    
    # Step 2: sqrtsoftplus
    scores = torch.sqrt(torch.log1p(torch.exp(logits.float())))   # [T, 384]
    
    # Step 3: Biased scores for selection
    scores_biased = scores + e_score_correction_bias.unsqueeze(0)  # [T, 384]
    
    # Step 4: Top-6 by biased score
    topk_ids = torch.topk(scores_biased, k=6, dim=-1).indices       # [T, 6]
    
    # Step 5: Raw (unbiased) weights at selected positions
    raw_weights = scores.gather(-1, topk_ids)                        # [T, 6]
    
    # Step 6: Renormalize
    if norm_topk_prob:
        renorm = raw_weights / raw_weights.sum(-1, keepdim=True)    # [T, 6]
    else:
        renorm = raw_weights
    
    # Step 7: Scale
    final_weights = renorm * routed_scaling_factor   # [T, 6], multiply by 2.5
    
    return topk_ids, final_weights
```

---

## 5. Expert Computation: SwiGLU + MXFP4

### 5.1 Expert forward pass

Each of the 6 selected experts for a token runs:

```
x = hidden_states[t]   # [7168]

# W1: [7168 → 2 × moe_intermediate_size] packed as gate+up
gate_logit = W1_gate[e] @ x   # [3072/TP]   (first half of W1)
up         = W1_up[e]   @ x   # [3072/TP]   (second half of W1)

# SwiGLU with clamp (swiglu_limit=10.0):
activated = silu(clamp(gate_logit, -10.0, 10.0)) * up   # [3072/TP]
# clamp prevents MXFP4 overflow in intermediate values

# W2: [(3072/TP) → 7168/TP]
expert_out = W2[e] @ activated   # [7168/TP]
```

`silu(x) = x * sigmoid(x)`. The clamp at ±10 is specific to MXFP4: without it, values outside this range underflow or overflow the 4-bit representation during the GEMM's intermediate accumulation.

With tensor parallelism (TP=8):
- Each rank computes `activated [3072/8=384]`
- W2 produces `expert_out [7168/8=896]`
- AllReduce across TP ranks gives full `[7168]` per expert

### 5.2 Expert weight shapes

```
With TP=8, n_routed_experts=384, moe_intermediate_size=3072, hidden_size=7168:

W1 (gate+up packed):
  Full:   [384, 2*3072, 7168]  = [384, 6144, 7168]
  Per TP: [384, 6144/8, 7168/2]  → after MXFP4 packing (2 values/byte):
          w1: [384, 768, 3584]  dtype uint8
  768 = 2 * (3072/8) = 2 * 384 (gate+up each 384 values, MXFP4 halves storage)
  3584 = 7168/2 (input dim packed 2-values-per-byte in MXFP4)

W2 (output projection):
  Full:   [384, 7168, 3072]
  Per TP: [384, 7168/8, 3072]  → MXFP4:
          w2: [384, 896, 3072]  dtype uint8
  896 = 7168/8 (output dim, 7168/8=896 int since not packed in the slow dim)
```

**MXFP4 format:**
- 4 bits per value (E2M1: 2 exp bits, 1 mantissa bit, 1 sign bit)
- 32 values per quantization block
- 2 values packed per byte (low nibble = first, high nibble = second)
- 1 ue8m0 scale byte per 32-value block
- Effective storage: 0.5 bytes/value + 1/32 bytes/value for scale ≈ 0.53 bytes/value

### 5.3 Weighted reduction

After all 6 experts run for token `t`:

```python
# topk_ids [T, 6], final_weights [T, 6], expert_outs [T, 6, 7168/TP]

# Weighted sum across 6 experts:
routed_out[t] = sum(final_weights[t, i] * expert_outs[t, i, :]  for i in range(6))
              # [7168/TP]

# With doweight_stage1=False (V4 default):
# The AITER fused_moe kernel applies weights at W2 output before reduction:
#   intermediate = W2[e] @ activated    [7168/TP]
#   weighted_out = intermediate * final_weights[t, e]
#   routed_out[t] = sum(weighted_out for e in topk_ids[t])
```

`doweight_stage1=False` means weighting happens at the output (after W2), not at the input (after W1). This is standard for V4's 6-expert selection.

---

## 6. Shared Expert

```python
self.shared_experts = DeepseekV4MLP(
    hidden_size = 7168,
    intermediate_size = 3072,   # n_shared_experts=1 × moe_intermediate_size=3072
    hidden_act = "silu",
    swiglu_limit = 10.0,
)
```

The shared expert runs on **all tokens unconditionally**:

```
shared_gate = W1_shared_gate @ hidden   # [3072]
shared_up   = W1_shared_up   @ hidden   # [3072]
shared_act  = silu(clamp(shared_gate, -10, 10)) * shared_up   # [3072]
shared_out  = W2_shared @ shared_act   # [7168]
```

No weight folded — the shared expert contributes at weight=1.0 (no `routed_scaling_factor`):

```
output[t] = routed_out[t] + shared_out[t]   # [7168]
```

The shared expert is **not fused** with the routed GEMM on ROCm by default (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=False`). It runs as a separate SwiGLU MLP.

---

## 7. DSpark Layers (58, 59, 60)

These three layers use identical `noaux_tc` MoE routing and expert computation to layers 3–57. Their only addition: after the MoE computes `hidden_out`, the hidden state is saved into a buffer consumed by the DSpark speculative decoder.

```python
# Standard noaux_tc MoE forward (same as any other standard layer):
routed_out = fused_moe(hidden, W1, W2, topk_weights, topk_ids)
shared_out = shared_expert(hidden)
hidden_out = routed_out + shared_out   # [T, 7168]

# DSpark-specific save (only for layers 58, 59, 60):
# Note: saved BEFORE the next layer's RMSNorm, so it's the post-MoE hidden state.
_mtp_hidden_buffer[layer_rel_idx, token_positions] = hidden_out
```

`_mtp_hidden_buffer` is consumed by the DSpark draft model to speculatively generate `dspark_block_size=5` candidate future tokens in parallel, reducing inference latency via speculation.

The 3 target layers (58, 59, 60) are all C4A — this is intentional: C4A provides finer-grained context for the speculation task compared to C128A.

---

## 8. Router Kernel: `FusedTopKBiasRouter`

All 61 layers use `FusedTopKBiasRouter` because every layer has either `hash_indices_table` or `e_score_correction_bias` set. The single kernel `topk_hash_softplus_sqrt` handles both modes:

```python
# vllm/model_executor/layers/fused_moe/router/router_factory.py:
if e_score_correction_bias is not None or hash_indices_table is not None:
    return FusedTopKBiasRouter(
        topk=6,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=e_score_correction_bias,   # None for hash layers
        hash_indices_table=hash_indices_table,             # None for standard layers
    )
```

**`topk_hash_softplus_sqrt` fused kernel:**

For hash layers: `topk_ids` come from `hash_indices_table[token_ids]`, then gate logits are computed only at those 6 positions for the weight. For standard layers: all 384 gate logits go through `sqrtsoftplus`, then biased `argtopk`.

The fusion avoids separate kernels for (1) gate projection, (2) activation, (3) top-K selection, (4) weight gathering — all in one pass over the gate logit matrix.

---

## 9. Worked Examples

### 9.1 Hash MoE: token "▁Paris" at layer 0

```
token_id = 14366  ("▁Paris")
layer = 0  (Hash MoE, C128A)

# Expert selection: hash lookup
topk_ids = tid2eid[14366] = [33, 187, 71, 251, 108, 9]   # from checkpoint

# Gate logits at these 6 positions (representative values):
logits[[33, 187, 71, 251, 108, 9]] = [1.62, 0.83, 1.21, 0.44, 1.08, 0.91]

# sqrtsoftplus:
scores = [
  sqrt(ln(1+e^1.62)) = sqrt(1.680) = 1.296,   # expert 33
  sqrt(ln(1+e^0.83)) = sqrt(0.931) = 0.965,   # expert 187
  sqrt(ln(1+e^1.21)) = sqrt(1.306) = 1.143,   # expert 71
  sqrt(ln(1+e^0.44)) = sqrt(0.604) = 0.777,   # expert 251
  sqrt(ln(1+e^1.08)) = sqrt(1.156) = 1.075,   # expert 108
  sqrt(ln(1+e^0.91)) = sqrt(0.993) = 0.997,   # expert 9
]

# Renormalize (sum = 6.253):
renorm = [0.207, 0.154, 0.183, 0.124, 0.172, 0.159]

# Scale by 2.5:
final_weights = [0.518, 0.386, 0.456, 0.311, 0.429, 0.399]

# Expert computation:
output_0  = E33(hidden)  * 0.518
output_1  = E187(hidden) * 0.386
output_2  = E71(hidden)  * 0.456
output_3  = E251(hidden) * 0.311
output_4  = E108(hidden) * 0.429
output_5  = E9(hidden)   * 0.399
routed_out = sum(output_0..5)
final = routed_out + shared_expert(hidden)
```

### 9.2 Standard MoE: same token at layer 5

```
token_id = 14366  ("▁Paris", same token, different hidden state after 5 layers)
layer = 5  (Standard noaux_tc MoE, C128A)

# Gate logits across all 384 experts:
logits [384]  (representative selection shown)

# sqrtsoftplus on all 384:
scores [384]

# e_score_correction_bias from checkpoint:
bias[[0..383]] = [..., +0.15, ..., -0.12, ..., +0.08, ...]

# Biased scores for selection:
scores_biased = scores + bias

# Top-6 by biased score:
topk_ids = argtopk(scores_biased, k=6)
         = [127, 42, 301, 8, 211, 175]   # (illustrative — depends on hidden state at layer 5)

# Unbiased weights:
raw = scores[[127, 42, 301, 8, 211, 175]]
    = [1.412, 1.308, 1.024, 0.891, 1.156, 0.934]

# Renormalize (sum = 6.725):
renorm = [0.210, 0.194, 0.152, 0.132, 0.172, 0.139]

# Scale by 2.5:
final_weights = [0.525, 0.486, 0.381, 0.331, 0.430, 0.347]

# Different experts are selected compared to layer 0:
# - Layer 0 (hash): always experts {33, 187, 71, 251, 108, 9} for "▁Paris"
# - Layer 5 (noaux_tc): experts depend on the current hidden state,
#   which now encodes contextual meaning after 5 layers of attention + MoE
```

### 9.3 Bias effect: under-loaded vs over-loaded expert

This example shows how `e_score_correction_bias` changes routing without distorting weights.

```
Scenario: expert 42 is under-loaded (bias=+0.15), expert 107 is over-loaded (bias=-0.12).

For some token t at layer 5:
  score[42]  = 1.20   (gate logit → sqrtsoftplus)
  score[107] = 1.22   (slightly higher than expert 42)

  Without bias:
    scores_biased = scores
    argtopk would select expert 107 over 42 (107 has higher raw score)
    If 107 is in top-6: raw_weight = 1.22 (unbiased)

  With bias:
    scores_biased[42]  = 1.20 + 0.15 = 1.35
    scores_biased[107] = 1.22 - 0.12 = 1.10
    argtopk now selects expert 42 over 107 (42 has higher biased score)

  Selected: expert 42
  Weight:   raw_weights = scores[[..., 42, ...]] = 1.20   (NO BIAS — uses raw score)
  → Expert 42 is selected but contributes exactly what its raw score says (1.20),
    not the inflated biased score (1.35).

  Effect: load on expert 42 increases, load on expert 107 decreases,
          but the model's output quality is unchanged because weights are unbiased.
```

### 9.4 Expert computation — SwiGLU with clamp, concrete numbers

For expert 42, token t="▁Paris", at layer 5 (TP=8, so moe_intermediate_size/TP = 3072/8 = 384 per rank):

```
hidden [7168] (after layer 5 attention)

W1_gate[42] @ hidden → gate_logits [384]
W1_up[42]   @ hidden → up_values   [384]

Example slice (dims 0-4):
  gate_logits[:5] = [11.3, -2.1,  8.7,  15.2, -0.4]
  up_values[:5]   = [0.41,  0.88, 0.23, -0.17,  0.95]

clamp(gate_logits, -10, 10):
  = [10.0, -2.1, 8.7, 10.0, -0.4]
  # dim 0 clamped from 11.3 → 10.0 (beyond swiglu_limit)
  # dim 3 clamped from 15.2 → 10.0

silu(clamped_gate)[:5]:
  silu(10.0)  = 10.0 * sigmoid(10.0)  = 10.0 * 0.99995 ≈ 9.9995
  silu(-2.1)  = -2.1 * sigmoid(-2.1)  = -2.1 * 0.1092  ≈ -0.229
  silu(8.7)   = 8.7  * sigmoid(8.7)   = 8.7  * 0.99984 ≈ 8.699
  silu(10.0)  = 9.9995  (clamped)
  silu(-0.4)  = -0.4 * sigmoid(-0.4)  = -0.4 * 0.4013  ≈ -0.161

activated[:5] = silu(clamp) × up = [9.9995×0.41, -0.229×0.88, ...]
              = [4.10, -0.202, 2.00, -1.700, -0.153]

expert_out[t] = W2[42] @ activated [384] → [896]  (7168/TP=896)
AllReduce across TP → [7168]
Multiply by final_weight[t, expert_42_position] ≈ 0.480
```

Why the clamp matters: without clamp, `silu(15.2) ≈ 15.2 × 1.0 = 15.2`, which after MXFP4 quantization (max representable E2M1 ≈ 6.0) would require a very large block scale, quantizing the entire 32-element block coarsely. The clamp at 10.0 limits the dynamic range, keeping MXFP4 precision useful for the majority of values.

### 9.5 Complete routing for one token across all 61 layers

Token "▁Paris" (token_id=14366) routing summary across layers 0-60:

```
Layer 0 (Hash MoE, C128A):
  topk_ids = tid2eid[14366] = [33, 187, 71, 251, 108, 9]  (fixed from checkpoint)
  all 6 experts: same as layer 1 unless tid2eid differs per layer
  (each layer has its own tid2eid table)

Layer 2 (Hash MoE, C4A):
  topk_ids = tid2eid_layer2[14366] = [44, 201, 12, 377, 88, 155]  (different table)
  
Layer 5 (Standard MoE, C128A):
  gate logits [384] → sqrtsoftplus → biased scores → top-6
  topk_ids = [42, 127, 301, 8, 211, 175]  (depends on hidden state at layer 5)

Observation: hash layers always produce the same 6 experts for "▁Paris",
regardless of context. Standard layers produce context-dependent experts —
the same token at position 100 vs position 5000 may select different experts
because the hidden state carries positional context.
```

---

## 10. ROCm MoE: AITER `fused_moe` and Padding Fix

On ROCm, expert computation uses AITER's `fused_moe` kernel:

```python
rocm_aiter_ops.fused_moe(
    hidden_states,          # [T, 7168]
    w1,                     # [384, 768, 3584]  MXFP4 uint8
    w2,                     # [384, 896, 3072]  MXFP4 uint8
    topk_weights,           # [T, 6]  float32
    topk_ids,               # [T, 6]  int32
    quant_method=QuantMethod.BLOCK_1X32,   # = 3  (MXFP4, 1×32 block quant)
    doweight_stage1=False,  # weight applied at W2 output, not W1 input
)
```

`BLOCK_1X32`: 1 row × 32 columns per quantization block. Each block has one ue8m0 scale byte. The kernel dequantizes MXFP4 → float32 during the GEMM accumulation.

**gfx950 padding crash and fix:**

During profiling/warmup (MRV2 `make_dummy()`), `is_padding.fill_(True)` causes `topk_hash_softplus_sqrt` to write `-1` into all `topk_ids`. `fused_moe` with `BLOCK_1X32` crashes on `-1` expert IDs (out-of-bounds access into weight array).

Fix in `rocm_aiter_moe.py`:

```python
if topk_ids.numel() > 0:
    if torch.cuda.is_current_stream_capturing():
        # CUDA graph capture: int() triggers CPU-GPU sync → forbidden during capture.
        # GPU-only fix: clamp -1→0 (makes access in-bounds) and zero their weights
        # (makes the clamped expert contribute 0 to output).
        # In real runs, topk_ids ≥ 0, so clamp and zero-mask are no-ops.
        _valid = topk_ids >= 0
        topk_ids = topk_ids.clamp(min=0)
        topk_weights = topk_weights * _valid.to(topk_weights.dtype)
    elif int(topk_ids.max()) < 0:
        # Eager/profile: CPU-GPU sync is safe here.
        # All tokens are padding → output is all zeros.
        return torch.zeros_like(hidden_states)
```

Why two branches:
1. **Eager**: `int(topk_ids.max())` is a CPU-GPU sync — safe outside graph capture. If all `-1`, short-circuit.
2. **Capture**: `int()` is forbidden (causes sync that breaks graph). Instead, clamp and mask on GPU. The captured graph replays with real non-negative `topk_ids`, so the clamp is a no-op at runtime.

---

## 11. Key Implementation Files

| File | Role |
|---|---|
| `vllm/models/deepseek_v4/amd/model.py` | AMD `DeepseekV4ForCausalLM`; MoE routing; hash MoE; weight mapper |
| `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` | `AiterExperts`; AITER `fused_moe` dispatch; gfx950 padding fix |
| `vllm/model_executor/layers/fused_moe/router/router_factory.py` | `FusedTopKBiasRouter`; `topk_hash_softplus_sqrt` kernel dispatch |
| `vllm/model_executor/models/deepseek_v2.py` | V3/V4 shared: `DeepseekV2MoE`, `gate`, `sqrtsoftplus`, `e_score_correction_bias` |
