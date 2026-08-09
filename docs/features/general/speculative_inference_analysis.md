# Speculative Decoding in vLLM — Complete Guide

---

## Table of Contents

1. [Background: How a Transformer Forward Pass Works](#1-background-how-a-transformer-forward-pass-works)
2. [How Speculative Decoding Works](#2-how-speculative-decoding-works)
   - 2.1 [Core Idea](#21-core-idea)
   - 2.2 [Why the Causal Mask Enables Parallel Verification](#22-why-the-causal-mask-enables-parallel-verification)
   - 2.3 [Rejection Sampling — In Full Detail](#23-rejection-sampling--in-full-detail)
   - 2.4 [KV Cache and Slot Pre-allocation](#24-kv-cache-and-slot-pre-allocation)
   - 2.5 [Normal Decode vs. Speculative Verification — Summary](#25-normal-decode-vs-speculative-verification--summary)
3. [vLLM Implementation Overview](#3-vllm-implementation-overview)
4. [All Implemented Speculation Methods](#4-all-implemented-speculation-methods)
5. [Shared Infrastructure](#5-shared-infrastructure)
6. [Method Comparison](#6-method-comparison)

---

## 1. Background: How a Transformer Forward Pass Works

Understanding speculative decoding requires a solid mental model of what happens during a normal prefill.

**Setup for this section:** Input prompt `"The cat sat"`. Tiny dimensions for readability: `d_model=4`, `n_heads=2`, `head_dim=2` (so `d_model = n_heads × head_dim`). One transformer layer; real models stack many identical layers.

---

### Step 1 — Tokenization

The prompt is split into integer token IDs by the vocabulary lookup:

```
"The cat sat"  →  [464, 3797, 6497]    (T = 3 tokens)
```

---

### Step 2 — Embedding Lookup

Each token ID indexes a row of the embedding matrix `E` of shape `[vocab_size, d_model]`:

```
token 464  → [  0.2,  0.8, -0.1,  0.5 ]   ← "The"
token 3797 → [  0.9, -0.3,  0.6,  0.2 ]   ← "cat"
token 6497 → [ -0.4,  0.7,  0.3, -0.8 ]   ← "sat"
```

Stack into the initial hidden state matrix `X` of shape `[T, d_model]` = `[3, 4]`:

```
X = [  0.2,  0.8, -0.1,  0.5 ]   ← position 0  ("The")
    [  0.9, -0.3,  0.6,  0.2 ]   ← position 1  ("cat")
    [ -0.4,  0.7,  0.3, -0.8 ]   ← position 2  ("sat")
```

---

### Step 3 — Pre-layer RMSNorm

Modern models (LLaMA, DeepSeek, Qwen) normalize `X` **before** each sub-block using RMSNorm:

```
RMSNorm(x) = x / sqrt(mean(x²)) * γ

Example — row 0 ("The"):  x = [0.2, 0.8, -0.1, 0.5]
  RMS = sqrt((0.04 + 0.64 + 0.01 + 0.25) / 4) = sqrt(0.235) ≈ 0.485
  normalized ≈ [0.41, 1.65, -0.21, 1.03]   (then scaled by learned γ)
```

Call the result `X_norm`, same shape `[3, 4]`.

---

### Step 4 — QKV Projection

Three learned weight matrices `W_Q`, `W_K`, `W_V` of shape `[d_model, d_model]` = `[4, 4]` project `X_norm` into queries, keys, and values:

```
Q = X_norm @ W_Q    shape [3, 4]   ← one query vector per token
K = X_norm @ W_K    shape [3, 4]   ← one key vector per token
V = X_norm @ W_V    shape [3, 4]   ← one value vector per token
```

---

### Step 5 — Split into Attention Heads

Reshape Q, K, V from `[T, d_model]` to `[n_heads, T, head_dim]` = `[2, 3, 2]`. Each head gets an independent slice:

```
Head 0:  Q0 = Q[:, 0:2]   K0 = K[:, 0:2]   V0 = V[:, 0:2]
Head 1:  Q1 = Q[:, 2:4]   K1 = K[:, 2:4]   V1 = V[:, 2:4]
```

Each head independently learns to attend to different aspects of the sequence.

---

### Step 6 — RoPE (Rotary Positional Encoding)

Position information is injected into **Q and K only** (not V) by rotating each vector by an angle proportional to its position. For `head_dim=2`, there is one rotation plane with frequency `θ`:

```
rotate(x, p) = [ x[0]·cos(p·θ) - x[1]·sin(p·θ),
                 x[0]·sin(p·θ) + x[1]·cos(p·θ) ]

Position 0 ("The"):  angle = 0    → no rotation
Position 1 ("cat"):  angle = 1.0  → rotate by 1 rad
Position 2 ("sat"):  angle = 2.0  → rotate by 2 rad
```

After rotation, Q and K carry positional identity. V is **not rotated** — attention weights (derived from Q·Kᵀ) encode relative position; values carry only content.

---

### Step 7 — Scaled Dot-Product Attention with Causal Mask

For each head independently. Using head 0 as the example:

**Compute raw attention scores** (every query attends to every key):

```
scores = Q0 @ K0ᵀ / sqrt(head_dim)     shape [3, 3]

         key0("The")  key1("cat")  key2("sat")
q0("The") [  s00,       s01,        s02   ]
q1("cat") [  s10,       s11,        s12   ]
q2("sat") [  s20,       s21,        s22   ]
```

**Apply the causal (lower-triangular) mask** — future positions set to `-inf` so they vanish after softmax:

```
masked_scores:
  [  s00,   -inf,  -inf ]
  [  s10,   s11,   -inf ]
  [  s20,   s21,   s22  ]
```

**Softmax row-wise** — converts scores to probabilities summing to 1 per row:

```
attn_weights:
  Row 0: softmax([s00, -inf, -inf]) = [1.0,  0.0,  0.0]   ← "The" sees only itself
  Row 1: softmax([s10, s11,  -inf]) = [a10,  a11,  0.0]   ← "cat" sees "The","cat"
  Row 2: softmax([s20, s21,  s22])  = [a20,  a21,  a22]   ← "sat" sees all three
```

**Weighted sum of values:**

```
attn_out[0] = 1.0·V0[0]
attn_out[1] = a10·V0[0] + a11·V0[1]
attn_out[2] = a20·V0[0] + a21·V0[1] + a22·V0[2]
```

Repeat identically for head 1, producing `attn_output1` of shape `[3, 2]`.

---

### Step 8 — Concatenate Heads + Output Projection

Concatenate head outputs and project back to `d_model`:

```
concat   = [attn_output0 | attn_output1]    shape [3, 4]
attn_out = concat @ W_O                     shape [3, 4]
```

---

### Step 9 — First Residual Connection

```
X = X + attn_out    shape [3, 4]
```

Preserves gradient flow and lets the model act as identity when the attention block adds nothing useful.

---

### Step 10 — FFN Block (SwiGLU)

Pre-norm again, then the feed-forward network:

```
X_norm2 = RMSNorm(X)

gate = X_norm2 @ W_gate    shape [3, d_ff]    # d_ff typically 4× or 8/3× d_model
up   = X_norm2 @ W_up      shape [3, d_ff]
down = (SiLU(gate) * up) @ W_down   shape [3, d_model]

X = X + down    ← second residual connection
```

`SiLU(x) = x · sigmoid(x)` acts as a smooth gate selecting which "up" features to pass through.

---

### Step 11 — Repeat for All L Layers

Steps 3–10 repeat `L` times (e.g., LLaMA-70B has 80 layers). Each layer's output `X` feeds into the next. By the final layer, each row of `X` is a rich contextual representation of that token.

---

### Step 12 — Final RMSNorm + LM Head

```
X_final = RMSNorm(X)          shape [3, 4]
logits  = X_final @ W_lm      shape [3, vocab_size]
```

`W_lm` is the LM head weight of shape `[d_model, vocab_size]`. In many models (and all MTP models) this is **weight-tied** to the embedding matrix `E`.

---

### Step 13 — Sampling (Normal Decode)

In **normal (non-speculative) decode**, only the **last row** of logits is used — it is the model's prediction for what follows the entire input sequence:

```
logits[0]  → p(· | "The")               ← discarded
logits[1]  → p(· | "The cat")           ← discarded
logits[2]  → p(· | "The cat sat")       ← sample next token → e.g. "on"
```

**Greedy:** `argmax(logits[2])` — pick the single highest-scoring token.

**Temperature + top-p (typical at inference):**

```
scaled = logits[2] / temperature        # flatten or sharpen the distribution
probs  = softmax(scaled)
# keep top tokens until cumulative probability > p, renormalize, then sample
token  = multinomial_sample(probs)
```

The sampled token is appended to the sequence. Its KV entries (`K_new`, `V_new`) are written to the KV cache. The next decode step processes only this one new token — all prior KV is already cached.

---

## 2. How Speculative Decoding Works

### 2.1 Core Idea

Standard autoregressive decode produces exactly **one token per target-model forward pass** — the forward pass is expensive, so throughput is limited. Speculative decoding breaks this bottleneck:

1. A **drafter** (cheap and fast) autoregressively proposes `N` candidate tokens `[d1, d2, ..., dN]`.
2. The **target model** verifies all `N` in a **single forward pass** by appending them to the context.
3. A **rejection sampler** accepts or rejects each draft token left to right, and recovers the correct distribution at the first rejection.

**Result:** Up to `N+1` tokens per target-model forward pass, with output distribution **identical** to the target model — no quality trade-off.

The efficiency gain depends on the **acceptance rate** α (fraction of draft tokens accepted on average). If α is high, you get many tokens per pass. If α is low (poor drafter), you fall back toward the one-token-per-pass baseline.

---

### 2.2 Why the Causal Mask Enables Parallel Verification

This is the key mechanism. The causal attention mask — standard in every decoder-only transformer — makes each output position's computation independent of all future positions. This turns parallel verification into a free by-product of a normal forward pass.

#### Setup

Confirmed context: `[A, B, C]`. Draft model proposed `[d1, d2, d3]`. Concatenate and run **one** target-model forward pass:

```
Position:  0    1    2    3    4    5
Token:    [A,   B,   C,   d1,  d2,  d3]
```

#### The Causal Attention Mask

```
              attends to →
         A    B    C    d1   d2   d3
         pos0 pos1 pos2 pos3 pos4 pos5

A   pos0 [  1,   0,   0,   0,   0,   0 ]
B   pos1 [  1,   1,   0,   0,   0,   0 ]
C   pos2 [  1,   1,   1,   0,   0,   0 ]
d1  pos3 [  1,   1,   1,   1,   0,   0 ]
d2  pos4 [  1,   1,   1,   1,   1,   0 ]
d3  pos5 [  1,   1,   1,   1,   1,   1 ]

1 = can attend,  0 = masked to -inf before softmax → 0 after softmax
```

#### What Each Position Computes

**Position 2 (token C):**

```
scores = [Q_C·K_A,  Q_C·K_B,  Q_C·K_C,  -inf,      -inf,      -inf  ]
weights = softmax → [w0,       w1,        w2,        0,         0,         0    ]
output_2 = w0·V_A + w1·V_B + w2·V_C
```

`output_2` sees only `{A, B, C}`. It is **bitwise identical** to running the model on just `[A, B, C]` alone — draft tokens are invisible because `-inf → 0` after softmax.

**Position 3 (token d1):**

```
scores = [Q_d1·K_A, Q_d1·K_B, Q_d1·K_C, Q_d1·K_d1, -inf,      -inf  ]
output_3 = w0·V_A + w1·V_B + w2·V_C + w3·V_d1
```

`output_3` sees exactly `{A, B, C, d1}` — the correct conditioning to verify d2.

**Position 4 (token d2):** sees `{A, B, C, d1, d2}`.

**Position 5 (token d3):** sees all six — the bonus token position.

#### Mapping Logit Rows to Verification Targets

After the LM head, `logits` has shape `[6, vocab_size]`. The logit at position `i` is the model's prediction for what comes at position `i+1`:

```
logits[0]  → p_target(· | A)                      ← discarded (already in context)
logits[1]  → p_target(· | A, B)                   ← discarded
logits[2]  → p_target(· | A, B, C)                ← compare against d1: verify d1
logits[3]  → p_target(· | A, B, C, d1)            ← compare against d2: verify d2
logits[4]  → p_target(· | A, B, C, d1, d2)        ← compare against d3: verify d3
logits[5]  → p_target(· | A, B, C, d1, d2, d3)    ← bonus token (free sample)
```

**Normal decode reads only `logits[-1]`. Speculative verification reads `logits[last_context_pos:]`.**

The forward pass itself — every layer, every matrix multiply — is identical. The model does not "know" it is doing verification. The only implementation difference is:

1. Append `N` draft tokens to the input before the forward pass.
2. Read `N+1` logit rows instead of just one.
3. Run rejection sampling across those rows.

---

### 2.3 Rejection Sampling — In Full Detail

#### Where p_draft Comes From

The draft model is a transformer too — every forward step outputs a full `[vocab_size]` distribution. When generating `d1, d2, d3` autoregressively, the drafter **retains the full distribution** at each step, not just the sampled token:

```
Draft step 0:  input=[A,B,C]       → softmax(logits) = p_draft(·|A,B,C)
                                                        ↓ sample
                                                        d1

Draft step 1:  input=[A,B,C,d1]    → softmax(logits) = p_draft(·|A,B,C,d1)
                                                        ↓ sample
                                                        d2

Draft step 2:  input=[A,B,C,d1,d2] → softmax(logits) = p_draft(·|A,B,C,d1,d2)
                                                        ↓ sample
                                                        d3
```

For verification, you only need the scalar probability the drafter assigned to each chosen token:

```
p_draft(d1 | A,B,C)          ← index d1 into the first distribution
p_draft(d2 | A,B,C,d1)       ← index d2 into the second
p_draft(d3 | A,B,C,d1,d2)    ← index d3 into the third
```

#### The Acceptance Condition

```
accept d_i   if   u < min(1,  p_target(d_i) / p_draft(d_i) ),   u ~ Uniform(0,1)
```

**Why the ratio `p_target / p_draft`?**

Think of it as a **confidence ratio**:

| Ratio | Meaning | Outcome |
|---|---|---|
| ≥ 1 | Target at least as confident as draft | Accept unconditionally (100%) |
| 0.8 | Target slightly less confident | Accept 80% of the time |
| 0.3 | Target strongly disagrees | Accept 30% of the time |
| 0.0 | Target assigns zero probability | Always reject |

**Why `Uniform(0,1)`?**

Drawing `u ~ U(0,1)` and checking `u < p` is the standard way to implement a probability-`p` coin flip in continuous arithmetic. It is the same mechanism as basic Monte Carlo rejection sampling — you accept with exactly the right probability over many draws, without needing to enumerate discrete outcomes.

#### Why This Preserves the Target Distribution (the Proof)

The non-obvious claim: these accept/reject decisions yield output tokens distributed **exactly as if the target model had generated them autoregressively**. Here is why.

Consider a single token position. The draft samples token `x` with probability `p_draft(x)`, and we accept with probability `min(1, p_target(x)/p_draft(x))`. The joint probability that `x` is emitted via acceptance is:

```
P(emit x via acceptance) = p_draft(x) · min(1, p_target(x) / p_draft(x))
                         = min(p_draft(x), p_target(x))
```

This covers accepted tokens. But what about rejections?

#### The Recovery Distribution

When a token is rejected, you cannot simply resample from `p_target` — that would over-represent tokens where the draft already assigned correct probability. Instead, sample from the **residual distribution**:

```
p_recover(x) = max(0, p_target(x) - p_draft(x))  /  Z

where Z = Σ_x max(0, p_target(x) - p_draft(x))
        = 1 - Σ_x min(p_target(x), p_draft(x))
```

Visually — `p_recover` is the part of `p_target` that "sticks out above" `p_draft`:

```
probability
    ↑
    │      ████                    ← p_target
    │   ██████████                 ← p_draft (overestimates these tokens)
    │                 ████         ← p_target > p_draft here → recovery mass
    └──────────────────────────→ tokens
         ↑                ↑
     draft too high     target sticks out → sampled on rejection
```

**Full proof that marginal = p_target:**

```
P(emit x) = P(accepted as d_i=x) + P(rejected somewhere, then recovered as x)

          = min(p_target(x), p_draft(x))
          + [1 - Σ_y min(p_target(y), p_draft(y))]  ·  max(0, p_target(x)-p_draft(x)) / Z

          = min(p_target(x), p_draft(x)) + max(0, p_target(x) - p_draft(x))

          = p_target(x)   ✓
```

Both cases (`p_target ≥ p_draft` and `p_target < p_draft`) collapse to `p_target(x)`. The proof is exact — not approximate.

#### Full Example Walk-through

```
After one target-model forward pass on [A, B, C, d1, d2, d3]:

  p_target(·|A,B,C)              from logits[2]
  p_target(·|A,B,C,d1)           from logits[3]
  p_target(·|A,B,C,d1,d2)        from logits[4]
  p_target(·|A,B,C,d1,d2,d3)     from logits[5]  ← bonus

Rejection loop (left to right, stop at first rejection):

  Verify d1:
    r = p_target(d1|A,B,C) / p_draft(d1|A,B,C)
    u ~ U(0,1)
    if u < min(1,r): accept d1 → continue to d2
    else: sample from max(0, p_target(·|A,B,C) - p_draft(·|A,B,C)), emit, STOP

  Verify d2:
    r = p_target(d2|A,B,C,d1) / p_draft(d2|A,B,C,d1)
    u ~ U(0,1)
    if u < min(1,r): accept d2 → continue to d3
    else: sample from residual of p_target(·|A,B,C,d1), emit, STOP

  Verify d3:
    r = p_target(d3|A,B,C,d1,d2) / p_draft(d3|A,B,C,d1,d2)
    u ~ U(0,1)
    if u < min(1,r): accept d3
          → sample bonus token from logits[5], emit d1,d2,d3,bonus. STOP

Possible outcomes per round:
  0 accepted → emit 1 token  (recovery sample at d1's position)
  1 accepted → emit 2 tokens (d1 + recovery at d2's position)
  2 accepted → emit 3 tokens (d1, d2 + recovery at d3's position)
  3 accepted → emit 4 tokens (d1, d2, d3 + bonus from logits[5])
```

In every case, each emitted token is distributed exactly as `p_target`. The draft quality affects **how many** tokens you get per step, not the correctness of what you get.

---

### 2.4 KV Cache and Slot Pre-allocation

`[A, B, C]` already has its KV entries cached from the prefill phase. During the verification forward pass, only the draft tokens need their KV computed:

```
Cached (from prefill):   K_A, V_A,  K_B, V_B,  K_C, V_C
Computed now:            K_d1, V_d1,  K_d2, V_d2,  K_d3, V_d3
```

This is why the scheduler **pre-allocates** `N` extra KV slots before the draft tokens even arrive — the verification step needs to write `K_{d1..dN}` into the cache.

If `k` draft tokens are accepted, those `k` KV entries are kept and the sequence advances by `k+1` (including the recovery/bonus token). The rejected entries are simply overwritten in the next round.

---

### 2.5 Normal Decode vs. Speculative Verification — Summary

```
Normal decode:
  Input:   [A, B, C]
  Logits:  shape [3, vocab_size]
  Use:     logits[2] only
  Output:  1 new token per target-model forward pass

Speculative verification:
  Input:   [A, B, C, d1, d2, d3]
  Logits:  shape [6, vocab_size]
  Use:     logits[2] → verify d1
           logits[3] → verify d2       ← same forward pass, same matrix ops
           logits[4] → verify d3
           logits[5] → bonus token
  Output:  1–4 new tokens per target-model forward pass
```

The speedup has two components:

1. **KV reuse** — `[A, B, C]` keys/values are read from cache once, not recomputed N times.
2. **Batched matmul** — all positions' Q/K/V projections and attention are fused into large matrix ops, which is what GPUs are optimized for.

---

## 3. vLLM Implementation Overview

vLLM's speculative decoding is fully implemented in the V1 engine, split across two layers:

| Layer | Location | Role |
|---|---|---|
| CPU-side proposer | `vllm/v1/spec_decode/` | Coordinates drafting logic, builds inputs |
| GPU-side speculator | `vllm/v1/worker/gpu/spec_decode/` | Runs draft forward pass, manages CUDA graphs |

**Config entry point:** `SpeculativeConfig` in `vllm/config/speculative.py`

Key config fields:

- `method` — which speculation algorithm to use (see Section 4)
- `num_speculative_tokens` — how many draft tokens `N` to propose per step
- `model` — path to draft model / eagle head / additional weights (if needed)
- `parallel_drafting` — generate all draft tokens in one pass vs. autoregressively
- `draft_sample_method` — `"greedy"` or `"probabilistic"` draft sampling
- `rejection_sample_method` — `"standard"`, `"synthetic"`, or `"block"`
- `num_speculative_tokens_per_batch_size` — dynamic N schedule by batch size

---

## 4. All Implemented Speculation Methods

### 4.1 N-gram / Prompt Lookup Decoding (CPU)

**Config:** `method="ngram"`
**Proposer:** `vllm/v1/spec_decode/ngram_proposer.py` — `NgramProposer`
**No GPU speculator**

**Mechanism:** Scans the existing token history for the longest suffix of the current context that matches a substring of the prompt (n-gram of length `min_n` to `max_n`). The tokens that follow that match in the prompt become draft tokens. Uses Numba JIT-compiled parallel CPU kernels (`batch_propose_numba()`).

**Key config:** `prompt_lookup_min`, `prompt_lookup_max` (n-gram length bounds).

**Best for:** Code generation, document editing — anywhere the output repeats or closely paraphrases the input prompt. Requires no extra model or weights.

---

### 4.2 N-gram GPU

**Config:** `method="ngram_gpu"`
**Proposer:** `vllm/v1/spec_decode/ngram_proposer_gpu.py` — `NgramProposerGPU`
**No GPU speculator**

**Mechanism:** Same algorithm as 4.1 but implemented in Triton GPU kernels (`_find_first_and_extract_all_n_parallel()`). Token history is maintained in GPU tensors and updated incrementally (`update_token_ids_ngram()`). Faster than CPU n-gram for large batches.

---

### 4.3 Medusa

**Config:** `method="medusa"`
**Proposer:** `vllm/v1/spec_decode/medusa.py` — `MedusaProposer`
**Draft model:** `vllm/model_executor/models/medusa.py` — `Medusa`, `ResidualBlock`

**Mechanism:** Multiple small "heads" (each a 2-layer MLP with a residual block) attached to the target model. Each head independently predicts the token at a different future offset from the **last hidden state** of the target model. All `N` heads run in parallel — one forward pass, no autoregressive loop.

```
target_model → last hidden state → head[0] → token t+1
                                 → head[1] → token t+2
                                 → head[2] → token t+3
```

Each head: `Linear → SiLU → Linear → LayerNorm → LM head`.

**Key property:** Zero additional KV cache. O(1) draft latency. Accuracy declines at higher `N` because heads ignore autoregressive dependencies between their own outputs.

---

### 4.4 MLP Speculator

**Config:** `method="mlp_speculator"` (auto-detected from `hf_config.model_type == "mlp_speculator"`)
**Proposer:** `vllm/v1/spec_decode/draft_model.py` — `DraftModelProposer`
**Draft model:** `vllm/model_executor/models/mlp_speculator.py` — `MLPSpeculator`

**Mechanism:** A tiny MLP (no attention, no KV cache) that autoregressively predicts draft tokens. At each step: last hidden state + previous draft token embedding → `LayerNorm` → stacked `Linear+GELU` layers → next draft token logits.

---

### 4.5 Draft Model (Generic Smaller LLM)

**Config:** `method="draft_model"` (or auto-detected from the `model=` path)
**Proposer:** `vllm/v1/spec_decode/draft_model.py` — `DraftModelProposer`

**Mechanism:** A full separate LLM (e.g., Llama-68M drafting for Llama-70B) autoregressively generates `N` draft tokens with its own KV cache. Vocabulary mismatch between draft and target is handled by `VocabMapping` (TLI token-level intersection, in `vllm/v1/spec_decode/vocab_mapping.py`).

**vs. EAGLE:** The draft model is independent — it does not receive hidden states from the target. More general, but typically lower acceptance rate than EAGLE.

---

### 4.6 EAGLE-1 / EAGLE-2

**Config:** `method="eagle"`
**Proposer:** `vllm/v1/spec_decode/eagle.py` — `EagleProposer`
**GPU speculator:** `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` — `EagleSpeculator → AutoRegressiveSpeculator`
**Draft models:** `vllm/model_executor/models/llama_eagle.py`, `deepseek_eagle.py`, `cohere_eagle.py`, `mistral_eagle.py`, etc.

**Mechanism:** The EAGLE draft model is a lightweight single-layer transformer that receives the **target model's last hidden states** as additional input. This gives it much richer context than a generic small draft model, while remaining fast.

Autoregressive drafting loop (in `AutoRegressiveSpeculator`):

```
Step 0 (prefill):
  target_hidden[t] → eagle_model → draft_token[t+1], eagle_hidden[t+1]

Step 1 (decode):
  eagle_hidden[t+1] → eagle_model → draft_token[t+2], eagle_hidden[t+2]
  ...
```

**EAGLE-2 extension:** Adds a dynamic draft tree. Instead of a fixed linear sequence, the drafter evaluates candidate confidence scores and adaptively expands/prunes a tree of continuations. More candidates are explored at positions where the model is uncertain.

**Target model requirement:** Must implement `SupportsEagle` protocol (`vllm/model_executor/models/interfaces.py`) — must expose last hidden states via `get_last_hidden_states()`.

---

### 4.7 EAGLE-3

**Config:** `method="eagle3"`
**Proposer:** `vllm/v1/spec_decode/eagle.py` — `EagleProposer`
**GPU speculator:** `EagleSpeculator` (same class as EAGLE-1/2)
**Draft models:** `vllm/model_executor/models/llama_eagle3.py`, `deepseek_eagle3.py`, `qwen3_eagle3.py`

**Mechanism:** Extends EAGLE by feeding the draft model hidden states from **multiple intermediate layers** of the target model (not just the last layer). The auxiliary layer indices are specified via `set_aux_hidden_state_layers()` / `get_eagle3_default_aux_hidden_state_layers()` from the `SupportsEagle3` protocol. Multi-layer features give the drafter substantially richer signal.

Config helper: `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`.

---

### 4.8 MTP — Multi-Token Prediction

**Config:** `method="mtp"` (auto-detected from HF model type)
**Proposer:** `vllm/v1/spec_decode/eagle.py` — `EagleProposer` (most MTP models)
**GPU speculator:** `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` — `MTPSpeculator`

#### What MTP Is

MTP is a **training-time technique** (introduced by DeepSeek-V3/R1) where the model is trained to predict multiple future tokens simultaneously using auxiliary prediction heads. At inference time, these heads become a built-in draft model — the model already knows how to predict tokens `t+2`, `t+3`, ... ahead without any separate checkpoint.

#### Architecture (DeepSeek MTP)

Each MTP layer (`vllm/model_executor/models/deepseek_mtp.py`) combines the target model's hidden state with the draft token embedding:

```
target_hidden[t] → RMSNorm (hnorm)  ─┐
embed(draft_token[t]) → RMSNorm (enorm) ─┤
                                        concat → Linear(eh_proj) → fused_hidden
                                        fused_hidden → DecoderLayer (attn + MoE)
                                                     → SharedHead (RMSNorm + LM head)
                                                     → logits → draft_token[t+1]
```

The MTP module **shares its LM head weight with the target model** (weight tying via `SharedHead`).

#### MTP vs. EAGLE — Key Distinction

| | EAGLE | MTP |
|---|---|---|
| Draft weights | Separate checkpoint to download | Part of the base model checkpoint |
| Training | Drafter trained specifically for spec-dec | Base model trained with MTP loss |
| Hidden state input | Last layer only (EAGLE-1/2) or multi-layer (EAGLE-3) | Last layer |
| Architecture | Lightweight single-layer transformer | Full decoder layer per MTP step |

#### Supported MTP Models

| Model | File | Org |
|---|---|---|
| DeepSeek V3/R1 | `deepseek_mtp.py` | DeepSeek |
| MiMo | `mimo_mtp.py` | Xiaomi |
| GLM-4 MoE | `glm4_moe_mtp.py` | Zhipu |
| Nemotron-H | `nemotron_h_mtp.py` | NVIDIA |
| Qwen3-Next / Qwen3.5 | `qwen3_next_mtp.py`, `qwen3_5_mtp.py` | Alibaba |
| Gemma4 | `gemma4_mtp.py` | Google |
| Step-3.5 | `step3p5_mtp.py` | Step |
| ERNIE | `ernie_mtp.py` | Baidu |
| ExaOne | `exaone_moe_mtp.py`, `exaone4_5_mtp.py` | LG |
| Bailing MoE | `bailing_moe_mtp.py` | Alibaba |
| OpenPangu | `openpangu_mtp.py` | Huawei |
| HYV3 | `hy_v3_mtp.py` | Tencent |
| LongCat Flash | `longcat_flash_mtp.py` | — |

`SpeculativeConfig.hf_config_override()` in `vllm/config/speculative.py` automatically rewrites HF model types (e.g., `deepseek_v3` → `deepseek_mtp`) so MTP loads transparently.

#### Special MTP Cases

**Step-3.5 MTP** (`vllm/v1/spec_decode/step3p5.py` — `Step3p5MTPProposer`):
Uses independent attention groups per MTP step, requiring custom slot mapping and per-group attention metadata.

**Gemma4 MTP** (`vllm/v1/spec_decode/gemma4.py` — `Gemma4Proposer`):
Uses **Q-only attention** — the MTP layers compute only queries and share the **target model's KV cache** directly (no new KV written). `Gemma4Speculator.advance_draft_positions` returns `False` since positions do not advance. Uses centroid-based CUDA graph capture.

---

### 4.9 DFlash (Parallel Draft with Non-Causal Attention)

**Config:** `method="dflash"`, forces `parallel_drafting=True`
**Proposer:** `vllm/v1/spec_decode/dflash.py` — `DFlashProposer`
**GPU speculator:** `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` — `DFlashSpeculator`
**Draft models:** `vllm/model_executor/models/qwen3_dflash.py`, `laguna_dflash.py`

**Mechanism:** Unlike EAGLE's autoregressive drafting loop, DFlash generates **all `N` draft tokens in a single parallel forward pass** by using **non-causal attention**. The draft model's mask tokens can attend to a full context KV buffer pre-computed from the target model:

```
Context → precompute_and_store_context_kv()   # pre-compute context KV buffer
[anchor_token, mask_1, mask_2, ..., mask_N]   # N+1 query positions
    ↓ non-causal attention over context KV
→ N draft tokens emitted simultaneously
```

`num_query_per_req = 1 + num_speculative_steps`. A special `parallel_drafting_token_id` is placed at each mask position. `_prepare_dflash_inputs_kernel()` (Triton) sets up the non-causal routing.

**vs. EAGLE:** Eliminates the O(N) sequential drafting loop — much lower draft latency. Tradeoff: mask tokens cannot condition on each other (unlike EAGLE's autoregressive chain).

---

### 4.10 DSpark (Block-Markov Parallel Drafting)

**Config:** `method="dspark"`, forces `parallel_drafting=True`
**GPU speculator:** `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` — `DSparkSpeculator(DFlashSpeculator)`
**Draft model:** `vllm/model_executor/models/qwen3_dspark.py` — `Qwen3DSparkModel`, `DSparkMarkovHead`

**Mechanism:** Extends DFlash with a **block-Markov head** that captures short-range autoregressive dependencies between draft tokens while still generating them in parallel blocks. The `DSparkMarkovHead` learns transition distributions: `compute_draft_logits()`, `map_draft_to_target()`, `markov_embed()`, `markov_bias()`. Within-block sampling uses `_sample_sequential()`.

The draft model ships inside the target checkpoint — no extra download. Sets `sample_from_anchor=True` (the anchor position is itself a sampled prediction, not just a bonus token, unlike DFlash).

---

### 4.11 Suffix Decoding

**Config:** `method="suffix"`
**Proposer:** `vllm/v1/spec_decode/suffix_decoding.py` — `SuffixDecodingProposer`
**External dependency:** `arctic_inference` library

**Mechanism:** Maintains **suffix trees** over past responses (both a global cross-request cache and per-request trees). At each step, finds the longest suffix of the current token sequence in the tree and proposes the stored continuation. More powerful than n-gram because suffix trees support exponentially longer matches in O(1) lookup time.

**Key config:** `suffix_decoding_max_tree_depth`, `suffix_decoding_max_cached_requests`, `suffix_decoding_max_spec_factor`, `suffix_decoding_min_token_prob`.

---

### 4.12 Extract Hidden States

**Config:** `method="extract_hidden_states"`
**Proposer:** `vllm/v1/spec_decode/extract_hidden_states.py` — `ExtractHiddenStatesProposer`

A **research primitive** — extracts intermediate hidden states from the target model without running a separate draft model. Used as a building block for custom spec decode methods that need access to target model internals.

---

### 4.13 Custom Class

**Config:** `method="custom_class"`, `model=` specifies `mymodule.MyProposerClass`
**Proposer:** `vllm/v1/spec_decode/custom_class_proposer.py` — `create_custom_proposer()`

Dynamically imports and instantiates a user-defined proposer class via dotted Python path. The class must implement `propose()`. Allows third-party spec decode methods without forking vLLM.

---

## 5. Shared Infrastructure

### 5.1 Rejection Sampling Strategies

Three strategies selected via `rejection_sample_method`:

| Strategy | When used | How it differs |
|---|---|---|
| `"standard"` | EAGLE, MTP, draft model, Medusa | Full `min(1, p_target/p_draft)` ratio; recovery from residual distribution |
| `"synthetic"` | N-gram, suffix | No `p_draft` available (non-neural proposer); uses modified recovery only from `p_target` |
| `"block"` | DSpark | Accepts/rejects entire token blocks using the block-Markov structure |

Implementation: `vllm/v1/sample/rejection_sampler.py` — all kernels (`rejection_sample()`, `rejection_greedy_sample_kernel()`, `rejection_random_sample_kernel()`, `sample_recovered_tokens_kernel()`) written in Triton.

### 5.2 Dynamic Spec Decode

`num_speculative_tokens_per_batch_size`: a list of `(batch_size_threshold, N)` pairs. At scheduling time, `N` scales down with batch size — fewer speculative tokens when the target model is already GPU-saturated by a large batch.

### 5.3 CUDA Graph Capture

All GPU speculators integrate with vLLM's CUDA graph infrastructure:

- `AutoRegressiveSpeculator` maintains separate CUDA graph managers for draft **prefill** (step 0) and draft **decode** (steps 1..N).
- `DFlashSpeculator` uses `DFlashCudaGraphManager`.
- Captures are keyed on `(num_tokens, num_requests)` shape tuples.

### 5.4 Proposer → GPU Speculator Routing

| Method | CPU-side Proposer | GPU-side Speculator |
|---|---|---|
| `ngram` | `NgramProposer` | None |
| `ngram_gpu` | `NgramProposerGPU` | None |
| `suffix` | `SuffixDecodingProposer` | None |
| `custom_class` | `create_custom_proposer()` | None |
| `draft_model` / `mlp_speculator` | `DraftModelProposer` | None |
| `extract_hidden_states` | `ExtractHiddenStatesProposer` | None |
| `medusa` | `MedusaProposer` | None |
| `eagle` / `eagle3` / `mtp` (most) | `EagleProposer` | `EagleSpeculator` / `MTPSpeculator` |
| `mtp` (step3p5) | `Step3p5MTPProposer` | `MTPSpeculator` |
| `mtp` (gemma4) | `Gemma4Proposer` | `Gemma4Speculator` |
| `dflash` | `DFlashProposer` | `DFlashSpeculator` |
| `dspark` | `EagleProposer` (via `use_eagle()`) | `DSparkSpeculator` |

### 5.5 KV Slot Pre-allocation

The scheduler reserves `num_lookahead_tokens` extra KV slots per request before any draft tokens arrive:

| Method family | Slots reserved |
|---|---|
| EAGLE / MTP / draft model | `num_spec_tokens` |
| DFlash | `num_spec_tokens + 1` (extra slot for the anchor) |

---

## 6. Method Comparison

| Method | Draft latency | Acceptance rate | Extra checkpoint | Uses target hidden states |
|---|---|---|---|---|
| N-gram CPU | CPU scan, O(context) | Low–medium | No | No |
| N-gram GPU | Triton kernel | Low–medium | No | No |
| Suffix | O(suffix tree lookup) | Medium | No (arctic lib) | No |
| Medusa | 1 GPU pass, parallel heads | Medium | Yes (head weights) | Yes (last layer) |
| MLP Speculator | Tiny MLP loop | Low–medium | Yes | Yes |
| Draft Model | Full LLM autoregressive loop | Medium | Yes (separate LLM) | No |
| EAGLE-1/2 | Small attn autoregressive loop | High | Yes (eagle head) | Yes (last layer) |
| EAGLE-3 | Small attn autoregressive loop | Higher | Yes (eagle head) | Yes (multi-layer) |
| **MTP** | **Small attn autoregressive loop** | **High** | **No (in base model)** | **Yes (last layer)** |
| DFlash | 1 GPU pass, parallel | High | Yes (in target ckpt) | Yes |
| DSpark | 1 block-parallel pass | High | Yes (in target ckpt) | Yes |
| Custom | User-defined | — | — | — |

**MTP's unique advantage:** Draft heads are baked into the base model checkpoint via multi-token prediction training. No separate weights to ship, download, or load — the model is inherently a better speculative drafter as a result of how it was trained.
