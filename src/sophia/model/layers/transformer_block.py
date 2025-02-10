from typing import Any, Optional

from flax import linen as nn

from sophia.model.layers.bases import (
    AttentionLayer,
    FeedForwardNetwork,
    NormalizationLayer,
    TransformerBlockBase,
)
from sophia.model.registry import register


@register
class TransformerBlock(TransformerBlockBase):
    """
    A generic Transformer block that consists of an attention layer, a feed-forward network (FFN),
    and two normalization layers. It supports both pre-normalization and post-normalization strategies.

    Attributes
    ----------
    attention : AttentionLayer
        An instantiated attention layer.
    feed_forward : FeedForwardNetwork
        An instantiated feed-forward network (FFN) layer.
    normalization_1 : NormalizationLayer
        The normalization layer applied before/after the attention layer.
    normalization_2 : NormalizationLayer
        The normalization layer applied before/after the feed-forward network.
    pre_norm : bool
        If True, applies LayerNorm before attention and FFN (pre-norm architecture).
        If False, applies LayerNorm after attention and FFN (post-norm architecture).
    residual_scale : float
        A scaling factor for the residual connections, used for stabilizing training.
    dropout_rate : float
        The dropout probability applied after the attention and FFN operations.
    """

    attention: AttentionLayer
    feed_forward: FeedForwardNetwork
    normalization_1: NormalizationLayer  # Fixed typo
    normalization_2: NormalizationLayer  # Fixed typo
    pre_norm: bool = True
    residual_scale: float = 1.0
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        hidden_states: Any,
        attention_mask: Optional[Any] = None,
        deterministic: bool = False,
    ):
        """
        Forward pass of the Transformer block.

        Args:
            hidden_states (Any): The input tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask (Optional[Any]): Masking tensor of shape [batch_size, 1, seq_length, seq_length].
            deterministic (bool): If True, disables dropout for inference.

        Returns:
            Any: The transformed output tensor of shape [batch_size, seq_length, hidden_size].
        """
        dropout = nn.Dropout(rate=self.dropout_rate)

        # --- Attention Sub-layer ---
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.normalization_1(hidden_states)

        attention_output = self.attention(
            hidden_states, attention_mask=attention_mask, deterministic=deterministic
        )
        attention_output = dropout(attention_output, deterministic=deterministic)

        if not self.pre_norm:
            attention_output = self.normalization_1(attention_output)

        # Residual connection with optional scaling
        hidden_states = residual + self.residual_scale * attention_output

        # --- Feed-Forward Sub-layer ---
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.normalization_2(hidden_states)

        feedforward_output = self.feed_forward(
            hidden_states, deterministic=deterministic
        )
        feedforward_output = dropout(feedforward_output, deterministic=deterministic)

        if not self.pre_norm:
            feedforward_output = self.normalization_2(feedforward_output)

        # Residual connection with optional scaling
        hidden_states = residual + self.residual_scale * feedforward_output

        return hidden_states
