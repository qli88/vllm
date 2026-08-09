# Rotary Position Embedding (RoPE) — Detailed Guide

## What Problem Does RoPE Solve?

Transformers have no built-in notion of order. Give them tokens A, B, C or C, A, B and they produce the same output unless you explicitly tell them where each token sits. Early work added a fixed sinusoidal vector to each token embedding before the first layer. RoPE takes a different approach: instead of adding position information once at the input, it injects it directly into the attention mechanism at every layer, and it does so in a way that makes attention scores depend only on **relative distance**, not absolute position.

---

## The Core Mathematical Insight

Attention scores are dot products:

```
score(q, k) = q^T * k
```

RoPE applies a rotation matrix R(m) to the query at position m and R(n) to the key at position n:

```
score = (R(m) * q)^T * (R(n) * k)
      = q^T * R(m)^T * R(n) * k
      = q^T * R(n - m) * k
```

The last step uses the fact that rotation matrices satisfy `R(m)^T * R(n) = R(n-m)`. The absolute positions m and n disappear and only the gap `n - m` survives. The model never has to see "token is at position 47" — it only sees "these two tokens are 3 apart."

---

## How the Rotation Works

For a vector of dimension d, RoPE splits it into d/2 consecutive pairs and rotates each pair independently with its own frequency.

For pair index i (zero-based), the rotation angle at position m is:

```
angle(m, i) = m * theta_i
theta_i     = base^(-2i / d)       (base is usually 10000)
```

The 2x2 rotation applied to pair (x, y) is:

```
[ cos(angle)  -sin(angle) ] [ x ]   =   [ x*cos - y*sin ]
[ sin(angle)   cos(angle) ] [ y ]       [ x*sin + y*cos ]
```

Pair 0 (i=0) has theta = 1.0, so it rotates fast and is sensitive to short distances.  
Pair d/2-1 (i=d/2-1) has theta close to zero, so it rotates slowly and encodes long-range structure.  
This is the same frequency hierarchy used in sinusoidal embeddings, just applied as rotation.

---

## Step-by-Step Example with d=4

We use d=4, so there are 2 pairs. Base = 10000.

### Computing the frequencies

```
theta_0 = 10000^(-0/4) = 10000^0.0  = 1.0
theta_1 = 10000^(-2/4) = 10000^-0.5 = 0.01
```

### Setting up for position m=3

```
angle_0 = 3 * 1.0  = 3.0   rad   (~171.9 degrees)
angle_1 = 3 * 0.01 = 0.03  rad   (~1.72 degrees)
```

### Rotating query q = [1.0, 0.5, 1.0, 0.5]

Pair 0: (q0, q1) = (1.0, 0.5), angle = 3.0

```
cos(3.0) = -0.9900
sin(3.0) =  0.1411

q0' = 1.0 * (-0.9900) - 0.5 * (0.1411) = -0.9900 - 0.0706 = -1.0606
q1' = 1.0 * ( 0.1411) + 0.5 * (-0.9900) =  0.1411 - 0.4950 = -0.3539
```

Pair 1: (q2, q3) = (1.0, 0.5), angle = 0.03

```
cos(0.03) =  0.9996
sin(0.03) =  0.0300

q2' = 1.0 * (0.9996) - 0.5 * (0.0300) = 0.9996 - 0.0150 =  0.9846
q3' = 1.0 * (0.0300) + 0.5 * (0.9996) = 0.0300 + 0.4998 =  0.5298
```

Rotated query at position 3:

```
q' = [-1.0606, -0.3539, 0.9846, 0.5298]
```

### Rotating key k = [0.8, 0.6, 0.8, 0.6] at position n=1

```
angle_0 = 1 * 1.0  = 1.0   rad
angle_1 = 1 * 0.01 = 0.01  rad
```

Pair 0: (0.8, 0.6), angle = 1.0

```
cos(1.0) =  0.5403
sin(1.0) =  0.8415

k0' = 0.8 * (0.5403) - 0.6 * (0.8415) = 0.4322 - 0.5049 = -0.0727
k1' = 0.8 * (0.8415) + 0.6 * (0.5403) = 0.6732 + 0.3242 =  0.9974
```

Pair 1: (0.8, 0.6), angle = 0.01

```
cos(0.01) =  0.99995
sin(0.01) =  0.01000

k2' = 0.8 * (0.99995) - 0.6 * (0.01000) = 0.7999 - 0.0060 =  0.7940
k3' = 0.8 * (0.01000) + 0.6 * (0.99995) = 0.0080 + 0.5999 =  0.6079
```

Rotated key at position 1:

```
k' = [-0.0727, 0.9974, 0.7940, 0.6079]
```

### Dot product

```
score = q' . k'
      = (-1.0606)(-0.0727) + (-0.3539)(0.9974) + (0.9846)(0.7940) + (0.5298)(0.6079)
      =  0.0771              + (-0.3530)         +  0.7818           +  0.3220
      =  0.8279
```

### Verifying relative position is what matters

Now rotate both q and k as if they were at positions m=2 and n=0 (gap is still 2, same as 3-1=2). The dot product should come out to the same value — the absolute positions changed but the gap did not. This is the guarantee RoPE provides.

---

## The rotate_half Trick

Doing 2x2 block rotations explicitly is inefficient. In practice, the rotation is rewritten as:

```python
def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

# cos and sin are precomputed for each position and broadcast to (seq, head_dim)
q_rotated = q * cos + rotate_half(q) * sin
k_rotated = k * cos + rotate_half(k) * sin
```

To see why this is equivalent, for a vector [a, b, c, d] with d=4:

```
rotate_half([a, b, c, d]) = [-c, -d, a, b]
```

cos and sin are tiled so position 0 gets (cos0, cos0, cos1, cos1) and sin gets (sin0, sin0, sin1, sin1). The negation and swap in rotate_half implement the cross-terms of the 2x2 block rotation in a single vectorized pass — no explicit matrix construction needed.

---

## What Happens at Different Distances

Using the frequencies from d=4 above (theta_0=1.0, theta_1=0.01):

```
Distance  |  angle_0 (fast pair)   |  angle_1 (slow pair)
----------|------------------------|----------------------
1         |  1.0  rad  (~57 deg)   |  0.01 rad (~0.6 deg)
10        |  10.0 rad  (~573 deg)  |  0.10 rad (~5.7 deg)
100       |  100  rad              |  1.00 rad (~57 deg)
1000      |  1000 rad              |  10.0 rad (~573 deg)
```

The fast pair wraps around many times for long distances — it distinguishes nearby tokens well but saturates at long range. The slow pair stays within a meaningful angular range across thousands of positions. Together they give the model multi-scale resolution.

---

## Context Length Extension

The base frequencies theta_i determine how fast each pair rotates. If a model is trained with sequences up to length L, the fast pairs will have wrapped many times and the slow pairs will have swept through their useful range. At inference time with longer sequences, the slow pairs venture into angles never seen during training, which degrades quality.

**RoPE scaling** addresses this by multiplying all angles by a factor less than 1 (stretching the position axis):

```
angle(m, i) = (m / scale) * theta_i
```

**YaRN** goes further by applying different scale factors to different frequency bands — preserving high-frequency pairs (which are already fine at long range) and only scaling down the low-frequency ones that are actually at risk.

---

## Where This Lives in vLLM

The implementation is in `vllm/model_executor/layers/rotary_embedding.py`. The main class `RotaryEmbedding` precomputes a `cos_sin_cache` at initialization covering all positions up to `max_position_embeddings`. At forward time it indexes into this cache by position IDs and applies the rotation in-place to q and k before they enter the attention kernel.

Variants like `LinearScalingRotaryEmbedding`, `DynamicNTKScalingRotaryEmbedding`, and `YaRNScaledRotaryEmbedding` inherit from the base class and override only how the cache is built, keeping the rotation logic identical.
