import jax.numpy as jnp
from flax import nnx

from config import GrokkingConfig


class Attention(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, *, rngs: nnx.Rngs):
        self.mha = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=n_heads * d_head,
            out_features=d_model,
            use_bias=False,
            dropout_rate=0.0,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        return self.mha(x, mask=mask)


class MLP(nnx.Module):
    def __init__(self, d_model: int, d_mlp: int, *, rngs: nnx.Rngs):
        self.w_in = nnx.Linear(d_model, d_mlp, use_bias=False, rngs=rngs)
        self.w_out = nnx.Linear(d_mlp, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.w_out(nnx.relu(self.w_in(x)))


class GrokkingTransformer(nnx.Module):
    """1-layer, 4-head transformer with no LayerNorm and no biases, matching the
    minimal architecture used in grokking replications of modular arithmetic."""

    def __init__(self, cfg: GrokkingConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.pos_embed = nnx.Embed(cfg.seq_len, cfg.d_model, rngs=rngs)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.d_head, rngs=rngs)
        self.mlp = MLP(cfg.d_model, cfg.d_mlp, rngs=rngs)
        self.unembed = nnx.Linear(cfg.d_model, cfg.n_classes, use_bias=False, rngs=rngs)

    def __call__(self, tokens):
        """tokens: (batch, 3) int32 -> logits: (batch, n_classes), read from the
        final ("=") position."""
        positions = jnp.arange(tokens.shape[-1])
        x = self.embed(tokens) + self.pos_embed(positions)
        mask = nnx.make_causal_mask(tokens) if self.cfg.causal_attention else None
        x = x + self.attn(x, mask=mask)
        x = x + self.mlp(x)
        return self.unembed(x[..., -1, :])
