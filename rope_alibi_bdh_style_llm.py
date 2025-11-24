# BDH style LLM, with significatn context window extention using Linear bias attention (ALiBi) and Rotary embeddings scaling (ROPE)
# Copyright 2025 Piotr Tomasinski
import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@dataclasses.dataclass
class BiasAttentionALiBiRopeBDHStyleConfig:
    n_layer: int = 6  # number of BDH blocks
    n_embd: int = 256  # model (embedding) dimension
    dropout: float = 0.1
    n_head: int = 4  # number of heads
    mlp_internal_dim_multiplier: int = 8  # controls latent size N
    vocab_size: int = 256  # byte-level by default

    # Long-context helpers
    use_linear_bias: bool = True  # turn ALiBi on/off

    # Rotary embeddings scaling
    rope_theta: float = 2 ** 16  # base for RoPE frequencies
    # scale applied to positions *before* RoPE:
    # positions_scaled = positions * rope_scale
    # e.g. for train_ctx=2048, test_ctx=8192 → rope_scale = 2048/8192 = 0.25
    rope_scale: float = 1.0


# ---------------------------------------------------------
# RoPE frequencies
# ---------------------------------------------------------

def get_freqs(n: int, theta: float = 2 ** 16, dtype=torch.float32) -> torch.Tensor:
    """
    Construct a bank of rotary frequencies for RoPE over a half-dimension.

    Common pattern:
        freq[i] = theta^{-i / n}
    """
    idx = torch.arange(n, dtype=dtype)
    freqs = theta ** (-idx / n)
    return freqs  # (n,)


# ---------------------------------------------------------
# Attention with RoPE + ALiBi + RoPE scaling
# ---------------------------------------------------------

class Attention(nn.Module):
    """

      * Q/K live in a high-dimensional latent space (sparse after ReLU),
      * V is a per-head view of the token embedding space,
      * uses RoPE in latent space with configurable theta and scaling,
      * uses Linear Bias Attention (ALiBi) for better long-context extrapolation,
      * uses raw scores (no softmax), as in BDH.
      * BDH-style attention:
    """

    def __init__(self, config: BiasAttentionALiBiRopeBDHStyleConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        assert N % 2 == 0, "Internal latent N must be even for RoPE"

        self.N = N
        rope_dim = N // 2

        # rotary frequency bank for half-dimension, with configurable theta
        freqs = get_freqs(rope_dim, theta=config.rope_theta, dtype=torch.float32)
        self.register_buffer(
            "freqs",
            freqs.view(1, 1, 1, rope_dim),  # (1, 1, 1, rope_dim)
            persistent=False,
        )

        # RoPE position scaling
        self.rope_scale = float(getattr(config, "rope_scale", 1.0))

        # ALiBi slopes per head
        if getattr(config, "use_linear_bias", False):
            slopes = self._get_alibi_slopes(nh)  # (nh,)
            self.register_buffer(
                "alibi_slopes",
                slopes.view(1, nh, 1, 1),  # (1, nh, 1, 1)
                persistent=False,
            )
        else:
            self.alibi_slopes = None

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
        """
        Slopes from "Train Short, Test Long: Attention with Linear Biases"
        (Press & Smith, 2021).
        """

        def get_slopes_power_of_2(n: int) -> torch.Tensor:
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor(
                [start * (ratio ** i) for i in range(n)],
                dtype=torch.float32,
            )

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                           : (n_heads - closest_power_of_2)
                           ]
            return torch.cat([slopes, extra_slopes], dim=0)

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor):
        """
        Convert fractional phases to cosine & sine.
        Expect phases of shape (..., rope_dim).
        """
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    def rope(self, phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to v using given phases.

        v:      (B, nh, T, N)
        phases: (1, 1, T, rope_dim)
        """
        phases_cos, phases_sin = self.phases_cos_sin(phases)
        v1, v2 = v.chunk(2, dim=-1)  # each (B, nh, T, rope_dim)
        v_rot = torch.cat(
            [
                v1 * phases_cos - v2 * phases_sin,
                v1 * phases_sin + v2 * phases_cos,
            ],
            dim=-1,
        )
        return v_rot

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q, K: (B, nh, T, N) - latent sparse representations
        V:    (B, nh, T, D) - per-head value projections of the input

        Returns:
            (B, nh, T, D)
        """
        assert K is Q, "BDH-style attention expects Q and K to be the same tensor"
        B, nh, T, N = Q.size()
        device = Q.device
        assert N == self.N
        rope_dim = self.freqs.shape[-1]

        # build rotary phases with scaling:
        # positions_scaled = positions * rope_scale
        positions = torch.arange(
            0,
            T,
            device=self.freqs.device,
            dtype=self.freqs.dtype,
        ).view(1, 1, T, 1)  # (1, 1, T, 1)
        positions_scaled = positions * self.rope_scale

        r_phases = positions_scaled * self.freqs  # (1, 1, T, rope_dim)

        # apply RoPE in latent space
        QR = self.rope(r_phases, Q)
        KR = QR  # symmetric as in the public BDH variants

        # base dot-product scores: (B, nh, T, T)
        scores = torch.matmul(QR, KR.mT)

        # Linear Bias Attention (ALiBi)
        if self.alibi_slopes is not None:
            i = torch.arange(T, device=device).view(1, 1, T, 1)
            j = torch.arange(T, device=device).view(1, 1, 1, T)
            # distance into the past, negative clipped away
            rel = (i - j).clamp(min=0).to(self.freqs.dtype)  # (1, 1, T, T)
            bias = -self.alibi_slopes * rel  # (1, nh, T, T)
            scores = scores + bias

        # causal: attend only to strictly previous positions
        scores = scores.tril(diagonal=-1)

        # BDH-style: use raw scores (no softmax) as weights
        out = torch.matmul(scores, V)  # (B, nh, T, D)
        return out


# ---------------------------------------------------------
# BDH-style block
# ---------------------------------------------------------

class BDHBlock(nn.Module):
    """
    One BDH-style reasoning block operating on token embeddings.

    Steps:
      1. LayerNorm + dropout on x (B, T, D).
      2. Project into high-dimensional latent per head: (B, nh, T, N),
         ReLU → sparse.
      3. Per-head V from x: (B, nh, T, D).
      4. RoPE + ALiBi attention in latent space.
      5. Project attention output back to latent, ReLU → second sparse stream.
      6. Gate: elementwise multiply both sparse streams.
      7. Decode back to D and average across heads.
      8. Residual add to x.
    """

    def __init__(self, config: BiasAttentionALiBiRopeBDHStyleConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.attn = Attention(config)
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh, N, D)).normal_(std=0.02))

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.size()
        nh = self.config.n_head
        N = self.config.mlp_internal_dim_multiplier * D // nh

        # 1. normalize & drop
        h = self.ln(x)
        h = self.dropout(h)

        # 2. project into latent per head: (B, nh, T, N)
        h_heads = torch.einsum("btd,hdn->bhtn", h, self.encoder)
        x_sparse = F.relu(h_heads)

        # 3. values V per head from h: (B, nh, T, D)
        V = h.unsqueeze(1).expand(-1, nh, -1, -1)

        # 4. attention in latent space
        yKV = self.attn(x_sparse, x_sparse, V)  # (B, nh, T, D)
        yKV = self.ln(yKV)  # reuse LN shape on D

        # 5. project attention output back to latent
        y_latent = torch.einsum("bhtd,hdn->bhtn", yKV, self.encoder_v)
        y_sparse = F.relu(y_latent)

        # 6. gate both sparse streams
        xy_sparse = x_sparse * y_sparse  # (B, nh, T, N)

        # 7. decode: latent -> D, aggregate heads
        y_mlp = torch.einsum("bhtn,hnd->bhtd", xy_sparse, self.decoder)
        y_mlp = y_mlp.mean(dim=1)  # (B, T, D)

        # 8. residual add
        out = x + self.dropout(y_mlp)
        return out


# ---------------------------------------------------------
# Top-level model
# ---------------------------------------------------------

class BiasAttentionALiBiRopeBDHStyle(nn.Module):
    """
    BDH-style language model with:
      * byte/token embedding,
      * a stack of BDHBlocks,
      * RoPE in the latent space (with scaling),
      * Linear Bias Attention (ALiBi) in the attention scores,
      * standard LM head for next-token prediction.

    Interface:
      forward(idx, targets=None) -> (logits, loss)
      generate(idx, max_new_tokens, temperature=1.0, top_k=None)
    """

    def __init__(self, config: BiasAttentionALiBiRopeBDHStyleConfig):
        super().__init__()
        self.config = config
        D = config.n_embd

        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [BDHBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            idx: torch.Tensor,  # (B, T)
            targets: Optional[torch.Tensor] = None  # (B, T) or None
    ):
        B, T = idx.size()
        x = self.embed(idx)  # (B, T, D)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
            self,
            idx: torch.Tensor,  # (B, T_start)
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation only; no caching for simplicity.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
