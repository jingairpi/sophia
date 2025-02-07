import jax.numpy as jnp
from flax import linen as nn

from .bases import AttentionLayer
from .normalizations import LayerNormalization


class MultiHeadDotProductAttention(AttentionLayer):
    """
    Implements Multi-Head Dot Product Attention manually using Dense layers and the attention mechanism.

    Attributes:
        num_heads (int): Number of attention heads.
        hidden_size (int): The hidden size of the input.
        dropout_rate (float): Dropout rate applied to attention probabilities.

    This implementation follows the standard scaled dot-product attention mechanism.
    """

    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, deterministic=False):
        """
        Applies multi-head dot product attention.

        Args:
            hidden_states (jax.numpy.ndarray): [batch_size, seq_length, hidden_size] input tensor.
            attention_mask (jax.numpy.ndarray, optional): Boolean mask of shape
                [batch_size, 1, seq_length, seq_length], where True means keep and False means mask out.
            deterministic (bool): If True, disables dropout.

        Returns:
            jax.numpy.ndarray: Output tensor of shape [batch_size, seq_length, hidden_size].
        """
        batch_size, seq_length, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads

        # Linear projections for query, key, and value
        q_proj = nn.Dense(self.hidden_size, use_bias=False)(hidden_states)  # (B, L, H)
        k_proj = nn.Dense(self.hidden_size, use_bias=False)(hidden_states)  # (B, L, H)
        v_proj = nn.Dense(self.hidden_size, use_bias=False)(hidden_states)  # (B, L, H)

        # Split into multiple heads
        def split_heads(x):
            return x.reshape(
                batch_size, seq_length, self.num_heads, head_dim
            ).transpose(0, 2, 1, 3)

        q, k, v = map(
            split_heads, (q_proj, k_proj, v_proj)
        )  # (B, num_heads, L, head_dim)

        # Scaled dot-product attention
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
            head_dim
        )  # (B, num_heads, L, L)

        if attention_mask is not None:
            scores = jnp.where(
                attention_mask, scores, -1e9
            )  # Masking (set to very low value)

        attn_weights = nn.softmax(scores, axis=-1)  # (B, num_heads, L, L)
        attn_weights = nn.Dropout(rate=self.dropout_rate)(
            attn_weights, deterministic=deterministic
        )

        # Weighted sum of values
        attn_output = jnp.einsum(
            "bhqk,bhkd->bhqd", attn_weights, v
        )  # (B, num_heads, L, head_dim)

        # Merge heads back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.hidden_size
        )

        # Final output projection
        output = nn.Dense(self.hidden_size, use_bias=False)(attn_output)

        return output
