from typing import Any, Optional

from flax import linen as nn

from sophia.model.layers.bases import (
    AttentionLayer,
    FeedForwardNetwork,
    NormalizationLayer,
)


class TransformerBlock(nn.Module):
    """
    A Transformer block that explicitly receives instantiated sub-modules
    instead of class types + kwargs.

    Attributes
    ----------
    attention : AttentionLayer
        An instantiated attention layer.
    feed_forward_network : FeedForwardNetwork
        An instantiated feed-forward network (FFN) layer.
    normalization_1 : NormalizationLayer
        An instantiated normalization layer for the first normalization.
    normalization_2 : NormalizationLayer
        An instantiated normalization layer for the second normalization.
    pre_norm : bool
        If True, applies normalization before attention and FFN.
    residual_scale : float
        Scaling factor for residual connections.
    dropout_rate : float
        Dropout probability.
    """

    attention: AttentionLayer
    feed_forward_network: FeedForwardNetwork
    normalization_1: NormalizationLayer
    normalization_2: NormalizationLayer
    pre_norm: bool = True
    residual_scale: float = 1.0
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[Any] = None,
        deterministic: bool = False,
    ):
        dropout = nn.Dropout(rate=self.dropout_rate)
        residual = hidden_states

        if self.pre_norm:
            hidden_states = self.normalization_1(hidden_states)

        attention_output = self.attention(
            hidden_states, attention_mask=attention_mask, deterministic=deterministic
        )
        attention_output = dropout(attention_output, deterministic=deterministic)

        if not self.pre_norm:
            attention_output = self.normalization_1(attention_output)

        hidden_states = residual + self.residual_scale * attention_output

        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.normalization_2(hidden_states)

        feedforward_output = self.feed_forward_network(
            hidden_states, deterministic=deterministic
        )
        feedforward_output = dropout(feedforward_output, deterministic=deterministic)

        if not self.pre_norm:
            feedforward_output = self.normalization_2(feedforward_output)

        hidden_states = residual + self.residual_scale * feedforward_output

        return hidden_states
