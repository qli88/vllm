# DeepSeek V4: Layer Types

[← Index](deepseek_index.md)

Explains the three layer types used in V4 (C4A, C128A, SWA-only), why they exist, what each one does differently in the attention path, and how the per-layer assignment was derived from the config.

---

## Table of Contents

1. [Why Multiple Layer Types?](#1-why-multiple-layer-types)
2. [Layer Type Definitions](#2-layer-type-definitions)
   - 2.1 [C4A — Compress-4-Aggregate](#21-c4a--compress-4-aggregate)
   - 2.2 [C128A — Compress-128-Aggregate](#22-c128a--compress-128-aggregate)
   - 2.3 [SWA-Only — Sliding Window Only](#23-swa-only--sliding-window-only)
3. [Per-Layer Assignment (all 61 layers)](#3-per-layer-assignment-all-61-layers)
4. [What Changes Between Layer Types](#4-what-changes-between-layer-types)
   - 4.1 [Attention Path Differences](#41-attention-path-differences)
   - 4.2 [KV Cache Differences](#42-kv-cache-differences)
   - 4.3 [Compressor Differences](#43-compressor-differences)
   - 4.4 [Indexer Differences](#44-indexer-differences)
5. [Worked Examples: Coverage at Different Sequence Lengths](#5-worked-examples-coverage-at-different-sequence-lengths)
   - 5.1 [C4A at 4K context](#51-c4a-at-4k-context)
   - 5.2 [C128A at 4K context](#52-c128a-at-4k-context)
   - 5.3 [C4A at 128K context](#53-c4a-at-128k-context)
   - 5.4 [C128A at 128K context](#54-c128a-at-128k-context)
   - 5.5 [Mixed batch: one C4A request + one C128A request at 64K tokens](#55-mixed-batch-one-c4a-request--one-c128a-request-at-64k-tokens)
   - 5.6 [Why C128A at 4K context sees only 32 compressed entries](#56-why-c128a-at-4k-context-sees-only-32-compressed-entries-not-1024)
6. [Layer Type and MoE Routing Interaction](#6-layer-type-and-moe-routing-interaction)
7. [Compressor State Cache Sizing](#7-compressor-state-cache-sizing)
8. [Timing Example: C4A vs C128A Decode Step](#8-timing-example-c4a-vs-c128a-decode-step)
9. [Key Implementation Files](#9-key-implementation-files)

---

## 1. Why Multiple Layer Types?

V4 is designed for 1M-context inference. Standard full attention is `O(N²)` in context length. MLA already compresses the KV cache to 512 dims per token, but attending 1M tokens is still expensive even at 512 dims.

V4 solves this at the architecture level: most layers do not attend all tokens. Instead, each layer uses a **compressor** to aggregate every N input tokens into 1 compressed KV entry, then the **sparse indexer** selects the top-1024 most relevant compressed entries for full attention. The remaining, very recent tokens are covered by the **sliding-window attention (SWA)** cache.

The tradeoff between N=4 (C4A) and N=128 (C128A):

| | C4A (N=4) | C128A (N=128) |
|---|---|---|
| Compression ratio | 4× | 128× |
| Compressed entries at 128K context | 32,768 | 1,024 |
| Indexer selects (top-1024) | 1,024 of 32,768 = **3%** seen | 1,024 of 1,024 = **100%** seen |
| Each selected compressed entry covers | 4 original tokens | 128 original tokens |
| Main MLA expansion per top-K hit | 4 original slots | 128 original slots |
| Recall quality | Higher (finer granularity) | Lower (coarser, lossy) |
| Memory / compute | More indexer cache entries | Fewer indexer cache entries |

Alternating C4A and C128A across layers balances fine-grained recall (C4A) with efficient long-range coverage (C128A). Neither type alone is optimal.

---

## 2. Layer Type Definitions

### 2.1 C4A — Compress-4-Aggregate

**compress_ratio = 4**

Every 4 consecutive input tokens are compressed into 1 KV entry. "A" stands for "Aggregate" (the gated aggregation step in the compressor). The suffix "C4" refers to `coff=2` (context overlap factor), meaning the compressor reads the last 8 states (2 groups × 4 tokens) to produce each compressed output with overlapping context.

**Token-level view:**

```
Input tokens:   t=0   t=1   t=2   t=3   t=4   t=5   t=6   t=7   ...
                ├──────────────────┤     ├──────────────────┤
                Group 0 (pos 0-3)         Group 1 (pos 4-7)
                    ↓                           ↓
             Compressed slot 0          Compressed slot 1
          (written when t=3)          (written when t=7)

Indexer k_cache grows by 1 entry every 4 input tokens.
Main MLA k_cache grows by 1 entry every 4 input tokens.
SWA k_cache grows by 1 entry every input token (always).
```

**What the overlap (coff=2) means:**

The compressor keeps a state_cache with the last `coff × compress_ratio = 8` token states. When compressing group 1 (positions 4–7), it uses states from positions 0–7 (not just 4–7). This creates an overlapping context window: each compressed entry is influenced by the 8 most recent raw tokens at the time of compression, not just the 4 in the current group.

```
When compressing at t=7 (group 1):
  state_cache holds states for positions: [0, 1, 2, 3, 4, 5, 6, 7]
                                           ╰── group0 (overlap) ──╯ ╰── group1 ──╯
  group0_gate = softmax(score[0:4])   → weighted average of positions 0-3
  group1_gate = softmax(score[4:8])   → weighted average of positions 4-7
  compressed_kv = sum(kv[0:4] × group0_gate) + sum(kv[4:8] × group1_gate)
```

### 2.2 C128A — Compress-128-Aggregate

**compress_ratio = 128**

Every 128 consecutive input tokens produce 1 KV entry. At 128K context, this yields exactly 1024 compressed entries — the full indexer budget — so the indexer effectively attends 100% of the history (just at 128× lower resolution than C4A).

**Token-level view:**

```
Input tokens:   t=0 ... t=127   t=128 ... t=255   ...
                ├────────────┤   ├────────────────┤
                Group 0             Group 1
                    ↓                   ↓
             Compressed slot 0   Compressed slot 1
          (written when t=127) (written when t=255)

At 128K context: 1024 compressed slots fill the full indexer budget.
At 4K context: 32 compressed slots — indexer selects all 32 (well under 1024).
```

C128A layers have sparser but broader coverage. A single compressed entry represents 128 original tokens, so top-K hits expand to 128 original slots for main MLA attention.

### 2.3 SWA-Only — Sliding Window Only

**compress_ratio = 0**

No compressor, no long-range attention. The SWA cache covers the most recent `window_size` tokens (4096 in the vLLM implementation). Beyond the SWA window, no attention is possible in this layer type.

In V4-Pro-0813, the raw `compress_ratios` array has three `0` entries — but these are at positions 61, 62, 63, which are **beyond the 61 real layers**. None of the actual 61 layers is SWA-only. The concept is defined for completeness and potential use in future models.

---

## 3. Per-Layer Assignment (all 61 layers)

Derived directly from `config.json`:
```json
"compress_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 4, 4, 0, 0, 0]
```

Layers 61–63 (the three trailing `0` entries) are beyond the model depth and are not real layers.

```
Layer  0: C128A  [Hash MoE]
Layer  1: C128A  [Hash MoE]
Layer  2: C4A    [Hash MoE]   ← only C4A in the hash-MoE zone
Layer  3: C128A  [Standard MoE]
Layer  4: C4A    [Standard MoE]
Layer  5: C128A  ...
Layer  6: C4A
Layer  7: C128A
Layer  8: C4A
Layer  9: C128A
Layer 10: C4A
Layer 11: C128A
Layer 12: C4A
Layer 13: C128A
Layer 14: C4A
Layer 15: C128A
Layer 16: C4A
Layer 17: C128A
Layer 18: C4A
Layer 19: C128A
Layer 20: C4A
Layer 21: C128A
Layer 22: C4A
Layer 23: C128A
Layer 24: C4A
Layer 25: C128A
Layer 26: C4A
Layer 27: C128A
Layer 28: C4A
Layer 29: C128A
Layer 30: C4A
Layer 31: C128A
Layer 32: C4A
Layer 33: C128A
Layer 34: C4A
Layer 35: C128A
Layer 36: C4A
Layer 37: C128A
Layer 38: C4A
Layer 39: C128A
Layer 40: C4A
Layer 41: C128A
Layer 42: C4A
Layer 43: C128A
Layer 44: C4A
Layer 45: C128A
Layer 46: C4A
Layer 47: C128A
Layer 48: C4A
Layer 49: C128A
Layer 50: C4A
Layer 51: C128A
Layer 52: C4A
Layer 53: C128A
Layer 54: C4A
Layer 55: C128A
Layer 56: C4A
Layer 57: C128A
Layer 58: C4A    [Standard MoE + DSpark]
Layer 59: C4A    [Standard MoE + DSpark]
Layer 60: C4A    [Standard MoE + DSpark]
```

**Pattern:** Layers 0 and 1 are both C128A (unusual — normally the pattern alternates). From layer 2 onward, it strictly alternates C4A / C128A. The last three layers (58–60) break the alternation: all three are C4A, which is why the `compress_ratios` ends with `[4, 4, 4, 0, 0, 0]` instead of `[4, 128, 4, 0, 0, 0]`.

**Counts:** 31 × C128A (layers 0,1,3,5,7,...,57), 30 × C4A (layers 2,4,6,...,56,58,59,60).

---

## 4. What Changes Between Layer Types

### 4.1 Attention Path Differences

Every layer runs the same high-level attention flow: indexer scoring → top-K selection → SWA + sparse FlashMLA. What differs is the **compressed-space resolution** and the **expansion factor** when going from compressed indices back to original slots.

```
C4A decode attention (compress_ratio=4):
  topk_indices_buffer [T, 1024]  contains compressed positions [0..seq_len/4)
  compute_global_topk_indices: each ck → 4 original slots
  global_topk_indices [T, 1, 4096]  →  extra_indices_in_kvcache
  flash_mla_with_kvcache(
      k_cache=swa_kv_cache, indices=swa_indices,          # SWA path
      extra_k_cache=main_mla_kv_cache,
      extra_indices_in_kvcache=global_topk_indices [T,1,4096],
  )

C128A decode attention (compress_ratio=128):
  topk_indices_buffer [T, 1024]  contains compressed positions [0..seq_len/128)
  build_c128a_topk_metadata: each ck → 128 original slots
  c128a_global_decode_topk_indices [T, 1, 131072]  (1024 × 128)
  flash_mla_with_kvcache(
      ...same SWA path...
      extra_indices_in_kvcache=c128a_global_decode_topk_indices [T,1,131072],
  )
```

The FlashMLA kernel itself is identical; only the `extra_indices_in_kvcache` tensor differs in the size of the last dimension.

### 4.2 KV Cache Differences

Both C4A and C128A use the same 584-byte per-slot format for the main MLA KV cache and the same 132-byte format for the indexer K cache. What differs is the **number of slots** populated per unit time.

```
At seq_len=4096:
  C4A main MLA KV cache:   4096 / 4   = 1024 compressed slots used
  C128A main MLA KV cache: 4096 / 128 = 32 compressed slots used

At seq_len=131072:
  C4A main MLA KV cache:   131072 / 4   = 32768 compressed slots used
  C128A main MLA KV cache: 131072 / 128 = 1024 compressed slots used (= index_topk)

SWA cache (both types): always stores last window_size=4096 original tokens
  → 4096 slots regardless of compress_ratio or total seq_len
```

Block sizes also differ:
```
C4A indexer k_cache block_size:   256 slots per block (block covers 256×4=1024 orig tokens)
C128A indexer k_cache block_size: same 256 slots per block (each covers 256×128=32768 orig tokens)
```

### 4.3 Compressor Differences

Both types use the same `DeepseekCompressor` class. What differs:

| | C4A | C128A |
|---|---|---|
| `compress_ratio` | 4 | 128 |
| `coff` (overlap factor) | 2 | model-dependent (typically 2 or more) |
| States read per compression | `coff × 4 = 8` | `coff × 128` |
| APE shape | `[4, 2×coff×head_dim]` | `[128, 2×coff×head_dim]` |
| `save_partial_states` fires | Every token | Every token |
| `compress_norm_rope_store` fires | Every 4 tokens | Every 128 tokens |
| Latency amortization | 1/4 tokens trigger compress | 1/128 tokens trigger compress |

**Example: APE (Absolute Positional Embedding)**

APE gives each position within a compression group a learned offset. For C4A:

```
compress_ratio = 4
ape: [4, 512]   ← 4 position embeddings, each 512-dim (2×coff×head_dim for head_dim=128)

Position 0 within a group (e.g., t=0, 4, 8, ...): uses ape[0, :]
Position 1 within a group (e.g., t=1, 5, 9, ...): uses ape[1, :]
Position 2 within a group (e.g., t=2, 6, 10, ...): uses ape[2, :]
Position 3 within a group (e.g., t=3, 7, 11, ...): uses ape[3, :]

# At t=5 (position 1 within group 1):
kv_state  = fused_wkv_wgate(hidden)[t]        # raw KV
kv_stored = kv_state + ape[5 % 4, :256]       # ape[1, :256]
score_stored = score_state + ape[5 % 4, 256:] # ape[1, 256:]
```

For C128A, `ape` has shape `[128, 2×coff×head_dim]` with 128 distinct position embeddings.

### 4.4 Indexer Differences

The indexer weights (`weights_proj`, `wq_b`) and the scoring heads (64 heads, 128-dim) are **identical** across layer types. The only difference is the compressed K cache entries: C4A entries each summarize 4 raw tokens; C128A entries each summarize 128.

**Example — what a compressed K represents:**

```
C4A compressed K at slot ck=7:
  Covers original positions [28, 29, 30, 31] (positions 7×4 to 7×4+3)
  Produced by: softmax gating over the kv_states and score_states for t=24..31
  Semantically: "the most informative aspect of tokens 28-31 according to learned gates"

C128A compressed K at slot ck=7:
  Covers original positions [896, 897, ..., 1023] (positions 7×128 to 7×128+127)
  Produced by: softmax gating over the kv_states for t=768..1023
  Semantically: "the most informative aspect of tokens 896-1023"
```

The scoring logit `Σ_h dot(q_fp8[q,h,:], k_fp8[ck,:]) × sk[ck] × weights_out[q,h]` has the same formula in both cases; only what `k_fp8[ck,:]` represents changes.

---

## 5. Worked Examples: Coverage at Different Sequence Lengths

### 5.1 C4A at 4K context

```
seq_len = 4096, compress_ratio = 4, window_size = 4096, index_topk = 1024

Compressed entries: 4096 / 4 = 1024
Indexer budget:    index_topk = 1024
Budget utilization: 1024 / 1024 = 100% → indexer selects ALL compressed entries

SWA covers: positions max(0, 4095-4095) = 0 .. 4095 → ALL 4096 tokens
→ for seq_len ≤ 4096, SWA already covers everything; sparse attention is redundant.

Main MLA call:
  swa_indices:              [T, 1, 4096]   → attends all 4096 SWA slots
  extra_indices_in_kvcache: [T, 1, 4096]   → 1024 compressed × 4 = 4096 original slots
  (These overlap completely at this sequence length)

Net coverage: 100% of all tokens.
```

### 5.2 C128A at 4K context

```
seq_len = 4096, compress_ratio = 128, window_size = 4096, index_topk = 1024

Compressed entries: 4096 / 128 = 32
Indexer budget:    1024
Budget utilization: 32 / 1024 = 3% → indexer selects ALL 32 compressed entries

SWA covers: all 4096 tokens (since seq_len ≤ window_size)

Main MLA call:
  swa_indices:              [T, 1, 4096]   → all 4096 SWA slots
  extra_indices_in_kvcache: [T, 1, 4096]   → 32 compressed × 128 = 4096 original slots
  (Same overlap as C4A case)

Net coverage: 100%.
```

At short contexts, both types achieve full coverage. The distinction only matters at long contexts.

### 5.3 C4A at 128K context

```
seq_len = 131072, compress_ratio = 4, window_size = 4096, index_topk = 1024

Compressed entries: 131072 / 4 = 32768
Indexer budget:    1024
Budget utilization: 1024 / 32768 = 3.1% → indexer selects top-1024 from 32768

SWA covers: last 4096 tokens (positions 127,007 .. 131,071)

Expanded slots from indexer:
  1024 compressed entries × 4 = 4096 original slots from anywhere in 0..131,071

Total tokens attended:
  SWA:     4096 tokens (most recent)
  Sparse:  4096 tokens (spread across history)
  Total:   ~8192 / 131072 = 6.25% of context

Granularity: each top-K hit covers exactly 4 consecutive tokens.
  → fine-grained: can target any 4-token window in the history.
```

### 5.4 C128A at 128K context

```
seq_len = 131072, compress_ratio = 128, window_size = 4096, index_topk = 1024

Compressed entries: 131072 / 128 = 1024 (= index_topk exactly)
Indexer budget:    1024
Budget utilization: 1024 / 1024 = 100% → indexer selects ALL compressed entries

SWA covers: last 4096 tokens

Expanded slots from indexer:
  1024 compressed entries × 128 = 131072 original slots → the ENTIRE history

Total tokens attended:
  SWA:     4096 (most recent, in full-resolution SWA cache)
  Sparse:  131072 (all of history, via compressed main MLA cache)
  → 100% of history is attended, just at lower resolution.

Granularity: each top-K hit covers 128 consecutive tokens.
  → coarser: cannot target individual 4-token windows, only 128-token blocks.
```

**Key insight:** At 128K context, C128A covers 100% of history but at 128× lower resolution than C4A. C4A covers only 3% of history but with 4× finer resolution. They are complementary: C128A ensures nothing is completely missed; C4A provides fine-grained recall for important regions.

### 5.5 Mixed batch: one C4A request + one C128A request at 64K tokens

For a single decode token at position 63999 (64K context), C4A and C128A layers within the same forward pass see very different compressed entry counts:

```
C4A layer (compress_ratio=4):
  Compressed entries: 63999 / 4 = 15999
  Indexer budget:    1024
  Budget utilization: 1024 / 15999 = 6.4% → top-1024 selected from 15999
  Expanded original slots: 1024 × 4 = 4096

C128A layer (compress_ratio=128):
  Compressed entries: 63999 / 128 = 499
  Indexer budget:    1024
  Budget utilization: 499 / 1024 = 100% → ALL 499 selected (well under budget)
  Expanded original slots: 499 × 128 = 63872 (≈ entire history)
```

In the same decode step, the C4A layer performs fine-grained recall over 6.4% of history while the C128A layer attends essentially the full 64K context at coarse resolution. Both operations share the same FlashMLA kernel; only `extra_indices_in_kvcache` differs in its last dimension (4096 vs 63872).

### 5.6 Why C128A at 4K context sees only 32 compressed entries (not 1024)

```
seq_len = 4096, compress_ratio = 128
Compressed entries: 4096 / 128 = 32
Indexer budget:    1024
Budget utilization: 32 / 1024 = 3.1% → indexer selects ALL 32

Each compressed entry expands to: 128 original slots
Total original slots attended: 32 × 128 = 4096 = 100% of history
```

At 4K context, C128A uses only 32 of its 1024 budget slots — but since each one expands to 128 tokens, it still covers 100% of history. The behavior is identical at 128K (where all 1024 slots are used, each expanding to 128 tokens, again covering 100% of history). C128A is always full-coverage; the budget only constrains it when context exceeds 128K tokens (i.e., more than 1024 compressed entries exist).

```
C4A at 4K for comparison:
  Compressed entries: 4096 / 4 = 1024
  Budget utilization: 1024 / 1024 = 100% → all selected
  Expanded: 1024 × 4 = 4096 = 100% of history
```

At short contexts, both types achieve 100% coverage. C4A just has finer granularity (4-token blocks vs 128-token blocks). The distinction between the types only becomes meaningful above 4K context (for C4A) or above 128K context (for C128A).

---

## 6. Layer Type and MoE Routing Interaction

Layer type (C4A/C128A) is purely an attention property. The MoE routing in each layer is independent:

- Layers 0–2: Hash MoE (any layer type) — `tid2eid` lookup, no `e_score_correction_bias`
- Layers 3–60: Standard noaux_tc MoE — `sqrtsoftplus` + `e_score_correction_bias` + top-6 argtopk

The combination for layer 2 (C4A + Hash MoE) is the only layer that is both:
- The only C4A in the hash-routing zone
- The first layer to use C4A compression

This has no special implementation treatment; the two systems (attention type, MoE type) are fully orthogonal.

---

## 7. Compressor State Cache Sizing

The compressor maintains a sliding-window state cache in paged GPU memory. Its size is determined by how many raw token states must be kept before the next compression fires.

**C4A state cache:**

```
coff = 2, compress_ratio = 4
States to keep: coff × compress_ratio = 2 × 4 = 8
State size per token: 2 × coff × head_dim × sizeof(float32)
                    = 2 × 2 × 128 × 4 = 2048 bytes
State cache per slot:  8 states × 2048 bytes = 16384 bytes (16 KB)

But paged: each "slot" in state_cache corresponds to 1 compressed output slot.
→ state_cache shape: [num_blocks, block_size, 2×coff×head_dim]
                   = [num_blocks, block_size, 512] float32
```

**C128A state cache:**

```
coff = 2 (example), compress_ratio = 128
States to keep: 2 × 128 = 256
State size: 2 × coff × head_dim × 4 = 2 × 2 × 128 × 4 = 2048 bytes
State cache: 256 × 2048 = 524288 bytes (512 KB) per compressed slot
```

C128A state caches are much larger because 256 raw token states must be buffered before compression fires. The paged design means this memory is allocated lazily per active sequence.

**Main MLA compressor state cache:**

Same structure but for `head_dim=512` (main MLA) vs `head_dim=128` (indexer):
```
State size: 2 × coff × 512 × 4 = 8192 bytes per raw token
For C4A:    8 × 8192 = 65536 bytes (64 KB) per compressed slot
For C128A:  256 × 8192 = 2097152 bytes (2 MB) per compressed slot
```

---

## 8. Timing Example: C4A vs C128A Decode Step

Approximate wall-clock times for one decode step at 128K context (single token, H100 SXM5, HBM3). Numbers are order-of-magnitude estimates to illustrate relative costs; actual times depend on batch size, kernel launch overhead, and tensor core utilization.

**Indexer path (`fp8_fp4_paged_mqa_logits`):**

```
C4A (compress_ratio=4):
  Compressed K entries to score: 131072 / 4 = 32768
  Data read: 32768 entries × 132 B = 4.3 MB
  HBM read time (H100 ~3.35 TB/s effective): 4.3 MB / 3.35 TB/s ≈ 1.3 µs

C128A (compress_ratio=128):
  Compressed K entries to score: 131072 / 128 = 1024
  Data read: 1024 entries × 132 B = 135 KB
  HBM read time: 135 KB / 3.35 TB/s ≈ 0.04 µs
```

The C128A indexer pass is ~32× faster because it scores 32× fewer compressed K entries.

**Main MLA path (`flash_mla_with_kvcache`):**

```
C4A (extra_indices_in_kvcache):
  Slots attended: 1024 compressed × 4 = 4096 original slots
  Data read: 4096 × 584 B = 2.4 MB
  HBM read time: 2.4 MB / 3.35 TB/s ≈ 0.7 µs

C128A (extra_indices_in_kvcache):
  Slots attended: 1024 compressed × 128 = 131072 original slots
  Data read: 131072 × 584 B = 76.5 MB
  HBM read time: 76.5 MB / 3.35 TB/s ≈ 23 µs
```

**Key insight:** C128A's main attention pass is ~32× slower than C4A's at 128K context, despite its indexer pass being ~32× faster. The indexer win is negligible; the attention cost dominates. C128A attends 131072 original slots vs C4A's 4096 — a 32× larger working set for the FlashMLA kernel.

```
Summary at 128K context (single token, H100):
                   Indexer       Main MLA      Total (approx)
  C4A layer:       ~1.3 µs       ~0.7 µs       ~2 µs
  C128A layer:     ~0.04 µs      ~23 µs        ~23 µs

C128A is ~11× more expensive per layer at 128K context.
The alternating C4A/C128A design balances recall quality (C4A) with
full-history coverage (C128A) while amortizing the C128A latency cost
across every other layer rather than every layer.
```

---

## 9. Key Implementation Files

| File | Role |
|---|---|
| `vllm/models/deepseek_v4/attention.py` | Layer type dispatch; C4A vs C128A routing; `compress_ratios` consumption |
| `vllm/models/deepseek_v4/compressor.py` | `DeepseekCompressor`; APE; `save_partial_states`; `compress_norm_rope_store` |
| `vllm/models/deepseek_v4/sparse_mla.py` | `DeepseekV4SparseMLABackend`; `build_c128a_topk_metadata`; layer-type metadata |
| `vllm/v1/attention/backends/mla/compressor_utils.py` | `get_compressed_slot_mapping`; `_compressed_slot_mapping_kernel` |
