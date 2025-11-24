# Extending Effective Context with RoPE + ALiBi in a BDH-Style Architecture


#### Author

*   [Piotr Tomasiński](p.tomdsinski@gmail.com)  - context window extention concepts & implementation
## Abstract
This repository explores how to extend the **effective context window** and **usable input/output length** of a BDH-style language model by combining:

- **Rotary Position Embeddings (RoPE)**  
- **Attention with Linear Biases (ALiBi)**
- **BDH-style sparse latent attention** inspired by *The Dragon Hatchling* (BDH) architecture

The main implementation lives in `BDHStyleBiasAttentionALiBi`, which integrates these ideas into a single, brain-inspired, post-Transformer block.

---

## 1. Motivation: Train Short, Use Long

Standard Transformers are typically limited by:

1. **Quadratic attention cost** in context length \(T\)  
2. **Positional encodings** that *do not* extrapolate gracefully beyond the maximum training length

Even if we can physically fit longer sequences into memory, the model may **fail to generalize** to those lengths because the positional representation breaks down.

This project targets:

- Training on **moderate context length** (e.g. 1k–2k tokens)
- Generalizing to **much longer contexts** (e.g. 8k–16k+) at inference time
- Doing so in a **BDH-style architecture**, where attention operates in a **sparse latent “reasoning space”** instead of standard Q/K/V projections

We rely on two complementary mechanisms:

- **RoPE** for *length-aware, rotation-based positional encoding* in latent space  
- **ALiBi** for a *distance-dependent linear bias* that favors recency but generalizes linearly to longer sequences

---

## 2. Background

### 2.1 Rotary Position Embeddings (RoPE)

RoPE was introduced in:

- J. Su et al.,  
  **“RoFormer: Enhanced Transformer with Rotary Position Embedding”**, arXiv:2104.09864.  
  <https://arxiv.org/abs/2104.09864> :contentReference[oaicite:0]{index=0}  

RoPE encodes position by **rotating** the query and key vectors in a complex plane (or 2D subspaces) using position-dependent angles. For each position \(p\) and frequency \(\omega\), RoPE applies:

\[
\tilde{q}_p = R(\theta_p) q_p,\quad \tilde{k}_p = R(\theta_p) k_p
\]

where \(R(\theta_p)\) is a 2D rotation matrix in the relevant subspace, with:

\[
\theta_p = p \cdot \omega
\]

This has important consequences:

- The **attention score** between positions \(p\) and \(p'\) depends on their **relative offset** \(p - p'\), not just absolute positions.
- RoPE naturally supports **length extrapolation** better than many additive positional schemes.
- The method is **multiplicative** (rotation) rather than **additive**, keeping position and content entangled in a structured way.

In our implementation, we apply RoPE not directly to the token embeddings, but to the **high-dimensional latent space** used in BDH-style attention.

#### RoPE Scaling

Vanilla RoPE often struggles when extrapolating far beyond training context because high frequencies rotate too fast. We introduce a **position scaling factor**:

\[
p_{\text{scaled}} = p \cdot s
\]

for some scale \(s \in (0,1]\).

- If you train on length \(L_{\text{train}}\) and want to use context \(L_{\text{test}}\), a simple heuristic is:
  \[
  s = \frac{L_{\text{train}}}{L_{\text{test}}}
  \]
- This “stretches” the same angular range over a longer position axis, making the positional geometry more consistent between train and test lengths.

In code: positions are multiplied by `rope_scale` before being multiplied by the frequency bank.

---

### 2.2 ALiBi: Attention with Linear Biases

ALiBi was proposed in:

- O. Press, N. A. Smith, M. Lewis,  
  **“Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”**, arXiv:2108.12409.  
  <https://arxiv.org/abs/2108.12409> :contentReference[oaicite:1]{index=1}  

ALiBi modifies attention scores by adding a **linear penalty** proportional to the distance between query and key positions:

\[
\text{score}_{i,j}^{(h)} 
= \langle q_i^{(h)}, k_j^{(h)} \rangle 
- m_h \cdot (i - j) \cdot \mathbf{1}_{i \ge j}
\]

where:

- \(i\) is the query position, \(j\) is the key position,
- \(m_h\) is a **head-specific slope** (ALiBi slope),
- \(\mathbf{1}_{i \ge j}\) ensures we only consider past positions for causal models.

Properties:

- Encourages **recency bias**: closer tokens get less penalty.
- Does *not* require explicit positional embeddings.
- Crucially: the **functional form is the same** regardless of sequence length, enabling **train short, test long** behavior.

In this project:

- We add per-head ALiBi slopes to the BDH-style attention scores.
- We apply the biases directly to the **raw correlation scores** before the causal mask, then use those scores directly (no softmax), in line with the BDH-style implementation.

---

### 2.3 BDH: The Dragon Hatchling Architecture

BDH (“Baby Dragon Hatchling”) is introduced in:

- A. Kosowski et al.,  
  **“The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain”**, arXiv:2509.26507.  
  <https://arxiv.org/abs/2509.26507> :contentReference[oaicite:2]{index=2}  
- Official implementation: <https://github.com/pathwaycom/bdh> :contentReference[oaicite:3]{index=3}  

BDH is a **post-Transformer architecture** aiming to bridge:

- Transformer-like performance and scaling, and
- Brain-inspired, **sparse, locally interacting networks** with explicit “reasoning graphs.”

Key ideas (simplified):

1. **Sparse latent reasoning space**  
   - Inputs are projected into a high-dimensional latent space where most units are inactive (ReLU sparsity).
   - This latent space can be interpreted as a **graph of “neuron-particles”** with sparse activations and interactions.

2. **Non-softmax, correlation-based attention**  
   - Attention resembles a dynamic convolution / correlation in latent space, without a softmax normalization.
   - Supports more interpretable, local interactions.

3. **Local, rule-like update dynamics (in the theory)**  
   - At the theoretical level, BDH describes learning in terms of strengthening or weakening local connections in the reasoning graph (σ-graph), evoking Hebbian-like rules.
   - The public code still uses standard backprop for training, but the architecture is designed for **interpretability + brain-style dynamics**.

Our project uses a **BDH-style block** as the base, then extends it with RoPE + ALiBi to improve long-context behavior.

---

## 3. Architecture: BDH-Style Block with RoPE + ALiBi

The core model is `BDHStyleBiasAttentionALiBi`, which stacks multiple `BDHBlock`s. Each block:

1. **LayerNorm + dropout** over token embeddings \(x \in \mathbb{R}^{B \times T \times D}\).
2. Projects into a **per-head latent space** of dimension \(N\):
   \[
   x_{\text{latent}}^{(h)} = x W_{\text{enc}}^{(h)} \in \mathbb{R}^{B \times T \times N}
   \]
   followed by ReLU to obtain **sparse activations**:
   \[
   s^{(h)} = \text{ReLU}(x_{\text{latent}}^{(h)})
   \]
3. Builds per-head values \(V^{(h)}\) as views of \(x\) in the embedding space.
4. Applies **RoPE (with scaling)** to the latent \(s^{(h)}\) to obtain \(Q^{(h)}, K^{(h)}\).
5. Computes **correlation scores**:
   \[
   \text{score}_{i,j}^{(h)} = \langle Q_i^{(h)}, K_j^{(h)} \rangle
   \]
6. Adds **ALiBi linear bias**:
   \[
   \text{score}_{i,j}^{(h)} \leftarrow \text{score}_{i,j}^{(h)} - m_h \cdot \max(0, i-j)
   \]
7. Applies **causal masking**: only \(j < i\) (strictly past).
8. Uses *raw scores (no softmax)* to weight the values:
   \[
   y^{(h)}_i = \sum_{j < i} \text{score}_{i,j}^{(h)} V_j^{(h)}
   \]
9. Projects \(y^{(h)}\) back to latent space via another encoder, applies ReLU, and **gates** it with the original sparse stream \(s^{(h)}\) via elementwise multiplication:
   \[
   g^{(h)} = \text{ReLU}(y_{\text{latent}}^{(h)}) \odot s^{(h)}
   \]
10. Decodes \(g^{(h)}\) back to embedding dimension \(D\) and **averages across heads**, then adds a residual connection.

This architecture:

- Keeps the **sparse, BDH-style reasoning structure**;
- Incorporates **RoPE** to retain relative positional geometry;
- Uses **ALiBi** to bias attention toward recency and enable **length extrapolation**.

---

## 4. Extending Effective Context and I/O Length

### 4.1 Training vs Inference Context

Let:

- \(L_{\text{train}}\) = maximum sequence length used during training
- \(L_{\text{test}}\) = sequence length at inference (can be \(\gg L_{\text{train}}\))

A vanilla model with learned absolute position embeddings typically fails when \(L_{\text{test}} > L_{\text{train}}\), because:

- Positions beyond \(L_{\text{train}}\) are **untrained**, or
- Sinusoidal patterns do not match the training distribution.

Our approach:

1. **RoPE scaling**:  
   - Keep position representation continuous:
     \[
     p_{\text{scaled}} = p \cdot s,\quad s = \frac{L_{\text{train}}}{L_{\text{test}}}
     \]
   - This preserves the **relative angular structure** of RoPE used during training.
2. **ALiBi**:  
   - The **linear distance bias** does not rely on an embedding table and is well-defined for any \(i, j\).
   - As shown in the ALiBi paper, models can achieve comparable perplexity on longer sequences after training only on shorter ones.

The combination means:

- The **latent attention** sees position differences that follow a *scaled but consistent* rotation pattern.
- ALiBi ensures that attention scores still follow a **simple, length-invariant linear decay** with distance.
- BDH-style sparsity concentrates capacity on a **subset of active latent units**, which are re-used across positions and scales.

### 4.2 Longer Outputs

The output length is controlled purely by **autoregressive decoding**:

- At each step, we feed the entire (or windowed) prefix back into the model.
- With RoPE + ALiBi, the model is better aligned to **maintain consistent behavior** as the prefix grows, allowing **many more decoding steps** before context degradation.

Practical considerations:

- Memory remains \(O(T^2)\) in sequence length – so we still need to manage GPU memory (batch size, gradient checkpointing, sliding windows).
- However, **quality degradation** of long outputs is mitigated by the extended positional scheme.

---

## 5. Design Recommendations

1. **Choose \(L_{\text{train}}\) based on budget**, e.g. 1k–2k tokens.
2. **Target \(L_{\text{test}}\)** (desired context), e.g. 8k.
3. **Set RoPE scale** roughly to:
   \[
   \text{rope\_scale} \approx \frac{L_{\text{train}}}{L_{\text{test}}}
   \]
   and optionally tune.
4. **Keep ALiBi enabled** (default): it adds only a small overhead but delivers substantial improvement in length extrapolation.
5. **Monitor long-context validation**:
   - Validate both near \(L_{\text{train}}\) and at significantly larger contexts.
   - Compare with a baseline without RoPE scaling and/or ALiBi.

---

## 6. References

### Positional Encodings and Long-Context Modeling

- J. Su, Y. Lu, S. Pan et al.,  
  **“RoFormer: Enhanced Transformer with Rotary Position Embedding”**, 2021.  
  arXiv:2104.09864  
  <https://arxiv.org/abs/2104.09864> :contentReference[oaicite:4]{index=4}  

- O. Press, N. A. Smith, M. Lewis,  
  **“Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”**, 2021.  
  arXiv:2108.12409  
  <https://arxiv.org/abs/2108.12409> :contentReference[oaicite:5]{index=5}  

- F. Barbero et al.,  
  **“Round and Round We Go! What makes Rotary Positional Encodings so Effective?”**, 2024.  
  arXiv:2410.06205  
  <https://arxiv.org/abs/2410.06205> :contentReference[oaicite:6]{index=6}  

- Long-context RoPE variants (3D-RPE, CARoPE, ComRoPE) – not used directly here but relevant for further work:  
  - 3D-RPE: **“Enhancing Long-Context Modeling Through 3D Rotary Position Encoding”** (2024). :contentReference[oaicite:7]{index=7}  
  - CARoPE: **“Context-aware Rotary Position Embedding”** (2025). :contentReference[oaicite:8]{index=8}  
  - ComRoPE: **“ComRoPE: Scalable and Robust Rotary Position Embedding parameterized by Trainable Commuting Angle Matrices”** (2025). :contentReference[oaicite:9]{index=9}  

### BDH / Dragon Hatchling

- A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz,  
  **“The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain”**, 2025.  
  arXiv:2509.26507  
  <https://arxiv.org/abs/2509.26507> :contentReference[oaicite:10]{index=10}  

- Official BDH repository:  
  <https://github.com/pathwaycom/bdh> :contentReference[oaicite:11]{index=11}  

---

## 7. Summary

`BDHStyleBiasAttentionALiBi` demonstrates how to:

- Embed a **BDH-style sparse reasoning block** into a language model,
- Enhance its **long-context capabilities** via **RoPE with scaling**, and
- Further stabilize **train-short, test-long** behavior using **ALiBi**.

The combination aims to retain BDH’s interpretability and brain-inspired dynamics while addressing one of the central pain points of modern LLMs: **robust, efficient handling of long sequences** for both inputs and generated outputs.
