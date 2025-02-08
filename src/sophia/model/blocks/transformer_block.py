from typing import Any, Dict, Optional, Type

from flax import linen as nn

from sophia.model.blocks.bases import TransformerBlockBase
from sophia.model.layers.bases import (
    AttentionLayer,
    FeedForwardNetwork,
    NormalizationLayer,
)


class TransformerBlock(TransformerBlockBase):
    """
    A generic Transformer block that uses an attention layer, a feed-forward network,
    and two normalization layers. This block supports both pre- and post-normalization
    strategies and allows scaling the residual connection.

    Attributes
    ----------
    attention_cls : Type[AttentionLayer]
        The class of the attention layer to instantiate.
    attention_kwargs : dict
        The keyword arguments to pass to the attention layer constructor.
    feed_forward_network_cls : Type[FeedForwardNetwork]
        The class of the feed-forward network (FFN) layer to instantiate.
    feed_forward_network_kwargs : dict
        The keyword arguments to pass to the FFN layer constructor.
    normalization_cls : Type[NormalizationLayer]
        The class of the normalization layer to instantiate.
    normalization_kwargs : dict
        The keyword arguments to pass to the normalization layer constructor.
    pre_norm : bool
        If True, applies normalization before attention and FFN (pre-norm architecture).
        If False, applies normalization after attention and FFN (post-norm architecture).
    residual_scale : float
        A scaling factor for the residual connections, useful for stability and
        tuning training dynamics.
    dropout_rate : float
        The dropout probability. Applied after attention and FFN operations.
    """

    attention_cls: Type[AttentionLayer]
    attention_kwargs: Dict[str, Any]
    feed_forward_network_cls: Type[FeedForwardNetwork]
    feed_forward_network_kwargs: Dict[str, Any]
    normalization_cls: Type[NormalizationLayer]
    normalization_kwargs: Dict[str, Any]
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
        # Create a dropout module to reuse within this forward pass.
        dropout = nn.Dropout(rate=self.dropout_rate)

        # --- Attention sub-layer ---
        residual = hidden_states
        if self.pre_norm:
            # Pre-norm: normalize before the attention layer.
            hidden_states = self.normalization_cls(
                name="layernorm_1", **self.normalization_kwargs
            )(hidden_states)

        attention_output = self.attention_cls(
            name="attention", **self.attention_kwargs
        )(hidden_states, attention_mask=attention_mask, deterministic=deterministic)
        attention_output = dropout(attention_output, deterministic=deterministic)

        if not self.pre_norm:
            # Post-norm: normalize after the attention layer.
            attention_output = self.normalization_cls(
                name="layernorm_1", **self.normalization_kwargs
            )(attention_output)

        # Residual connection with optional scaling.
        hidden_states = residual + self.residual_scale * attention_output

        # --- Feed-forward sub-layer ---
        residual = hidden_states
        if self.pre_norm:
            # Pre-norm: normalize before the FFN.
            hidden_states = self.normalization_cls(
                name="layernorm_2", **self.normalization_kwargs
            )(hidden_states)

        feedforward_output = self.feed_forward_network_cls(
            name="ffn", **self.feed_forward_network_kwargs
        )(hidden_states, deterministic=deterministic)
        feedforward_output = dropout(feedforward_output, deterministic=deterministic)

        if not self.pre_norm:
            # Post-norm: normalize after the FFN.
            feedforward_output = self.normalization_cls(
                name="layernorm_2", **self.normalization_kwargs
            )(feedforward_output)

        # Residual connection with optional scaling.
        hidden_states = residual + self.residual_scale * feedforward_output

        return hidden_states
