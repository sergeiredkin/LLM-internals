---
type: learning-note
status: doing
---
# RoPE — Rotary Position Embeddings

## 1. The problem

An attention score begins as a content comparison:

```text
score(query at position m, key at position n) = q_m · k_n
```

Without explicit positional information, that dot product knows whether two token representations
are similar, but not how far apart they are. A language model needs both content and order:

- `dog bites man` differs from `man bites dog`;
- a nearby adjective is usually more relevant than one hundreds of tokens away;
- generation needs to distinguish the first token from the hundredth token.

Our baseline adds a learned vector for each absolute position:

```text
input = token_embedding[token] + position_embedding[position]
```

That works, but it creates one trainable row per context position and does not directly encode
relative displacement.

## 2. The RoPE idea

RoPE represents position by rotating pairs of coordinates. For one pair `(x0, x1)` and angle
`θ`, the rotation is:

```text
x0' = x0 cos(θ) - x1 sin(θ)
x1' = x0 sin(θ) + x1 cos(θ)
```

In matrix form:

```text
R(θ) = [[cos θ, -sin θ],
        [sin θ,  cos θ]]
```

At token position `p`, each pair uses angle `p × frequency`. Different coordinate pairs use
different frequencies, so some rotate quickly and others slowly.

For head dimension `d`, a common frequency definition is:

```text
frequency_i = base^(-2i/d), where i = 0 ... d/2 - 1
angle(p, i) = p × frequency_i
```

The usual base is 10,000. These frequencies are fixed, not learned parameters.

## 3. Hand-worked rotation

Take vector:

```text
x = (2, 1)
θ = 90°
cos θ = 0
sin θ = 1
```

Apply the equations:

```text
x0' = 2×0 - 1×1 = -1
x1' = 2×1 + 1×0 =  2
```

Therefore:

```text
R(90°)(2, 1) = (-1, 2)
```

The norm does not change:

```text
before: sqrt(2² + 1²)   = sqrt(5)
after:  sqrt((-1)² + 2²) = sqrt(5)
```

This norm-preservation property is one of the tests we will implement.

## 4. Why relative position appears

RoPE rotates a query at position `m` and a key at position `n`:

```text
q'_m = R(m) q_m
k'_n = R(n) k_n
```

Their attention dot product becomes:

```text
q'_m · k'_n
= (R(m)q_m)ᵀ(R(n)k_n)
= q_mᵀ R(m)ᵀ R(n) k_n
= q_mᵀ R(n - m) k_n
```

The final rotation depends on `n - m`, the relative distance between positions.

The key identities are:

```text
R(m)ᵀ = R(-m)
R(-m)R(n) = R(n - m)
```

### Numeric relative-position example

Use one illustrative frequency of 30° per position and simple vectors:

```text
q = (1, 0), at position 1 -> angle 30°
k = (1, 0), at position 3 -> angle 90°
```

After rotation, the angle between them is `90° - 30° = 60°`, so their dot product is:

```text
cos(60°) = 0.5
```

Shift both positions by two:

```text
q at position 3 -> 90°
k at position 5 -> 150°
```

The angle difference remains 60°, and the dot product remains 0.5. Absolute positions changed,
but relative displacement did not.

Actual RoPE performs this calculation across every coordinate pair using multiple frequencies.

## 5. Why rotate Q and K, but not V?

Queries and keys determine the attention score: **where to read from**. Rotating Q and K makes
that routing decision position-aware.

Values contain the information being retrieved: **what to read**. Leaving V unchanged lets
attention move content without embedding a position rotation into the content itself.

In short:

```text
Q and K: position-aware routing
V: content transported by that routing
```

## 6. Difference from the current model

### Learned absolute positions

```python
x = token_embedding(input_ids) + position_embedding(positions)
```

- Adds position before all transformer blocks.
- Uses `context_length × d_model` trainable parameters.
- TinyStories position table: `512 × 512 = 262,144` parameters.

### RoPE

```python
x = token_embedding(input_ids)
q = rotate(q, positions)
k = rotate(k, positions)
```

- Applies inside every attention layer.
- Has no trainable position table.
- Directly gives Q/K dot products a relative-position structure.
- Requires an even head dimension so coordinates can form pairs.

Our TinyStories head dimension is:

```text
d_model / n_heads = 512 / 8 = 64
```

It is even, so it can form 32 rotation pairs per attention head.

## 7. What RoPE does not guarantee

RoPE is not automatically better on every dataset or training budget. It does not:

- add factual knowledge;
- fix story consistency by itself;
- guarantee lower validation loss;
- make context length unlimited;
- remove the need for causal masking.

We will test whether it gives similar or better learning with 262,144 fewer parameters and how it
affects speed and memory.

## 8. Check yourself before implementation

Answer these without looking above:

1. Rotate `(3, 4)` by 90°. What is the result, and what happens to its norm?
2. If a query is at position 7 and a key is at position 10, what relative displacement appears in
   their RoPE dot product? What happens if both positions increase by 100?
3. Why is V not rotated in standard RoPE attention?

## 9. Implementation map for the next step

When the answers are clear, implementation will proceed in this order:

1. Compute inverse frequencies.
2. Compute position angles.
3. Build cosine and sine tensors.
4. Rotate adjacent coordinate pairs in Q and K.
5. Test the rotation independently.
6. Only then connect it to causal attention.
