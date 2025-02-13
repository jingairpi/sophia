from typing import Any, Union

from pydantic import BaseModel
from typing_extensions import Literal


# --------------------------------------------------------------------
# Base Configuration
# --------------------------------------------------------------------
class BaseConfig(BaseModel):
    """
    The base configuration class for all model components.

    Each subclass must define a `type` field, which serves as a unique identifier
    used by the model builder to retrieve and instantiate the corresponding class
    from the registry.

    Subclasses can specify additional fields relevant to their respective components.
    """

    type: str


# --------------------------------------------------------------------
# Embedding Layers
# --------------------------------------------------------------------
class TokenEmbeddingConfig(BaseConfig):
    """
    Configuration for a token embedding layer.
    """

    type: Literal["TokenEmbedding"] = "TokenEmbedding"
    vocab_size: int
    hidden_size: int


class PositionalEmbeddingConfig(BaseConfig):
    """
    Configuration for a positional embedding layer.
    """

    type: Literal["PositionalEmbedding"] = "PositionalEmbedding"
    max_seq_length: int
    hidden_size: int


EmbeddingLayerConfig = Union[TokenEmbeddingConfig, PositionalEmbeddingConfig]


# --------------------------------------------------------------------
# Attention Configuration
# --------------------------------------------------------------------
class MultiHeadDotProductAttentionConfig(BaseConfig):
    """
    Configuration for an attention mechanism.
    """

    type: Literal["MultiHeadDotProductAttention"] = "MultiHeadDotProductAttention"
    hidden_size: int
    num_heads: int
    dropout_rate: float = 0.1


AttentionConfig = MultiHeadDotProductAttentionConfig


# --------------------------------------------------------------------
# Feed-Forward Network
# --------------------------------------------------------------------
class PositionwiseFeedForwardConfig(BaseConfig):
    """
    Configuration for a position-wise feed-forward network (FFN) used in transformers.

    Attributes:
        hidden_size: The hidden dimension in the feed-forward network.
        ffn_multiplier: The expansion factor for the intermediate layer.
        activation: The activation function applied between the dense layers.
        dropout_rate: Dropout probability applied within the FFN.
    """

    type: Literal["PositionwiseFeedForward"] = "PositionwiseFeedForward"
    hidden_size: int
    ffn_multiplier: int
    activation: Any
    dropout_rate: float = 0.1


FeedForwardConfig = PositionwiseFeedForwardConfig


# --------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------
class LayerNormalizationConfig(BaseConfig):
    """
    Configuration for Layer Normalization.
    """

    type: Literal["LayerNormalization"] = "LayerNormalization"
    epsilon: float = 1e-5


class RMSNormalizationConfig(BaseConfig):
    """
    Configuration for RMS Normalization.
    """

    type: Literal["RMSNormalization"] = "RMSNormalization"
    epsilon: float = 1e-5


NormalizationConfig = Union[LayerNormalizationConfig, RMSNormalizationConfig]


# --------------------------------------------------------------------
# Projection (Unembedding Layer)
# --------------------------------------------------------------------
class OutputProjectionConfig(BaseConfig):
    """
    Configuration for an output projection (unembedding) layer.
    """

    type: Literal["OutputProjection"] = "OutputProjection"
    hidden_size: int
    output_size: int


ProjectionConfig = OutputProjectionConfig


# --------------------------------------------------------------------
# Transformer Block Configuration
# --------------------------------------------------------------------
class TransformerBlockConfig(BaseConfig):
    """
    Configuration for a Transformer Block.

    This configuration specifies the parameters and nested sub-configurations
    for a transformer block, including:
    - The attention mechanism
    - The feed-forward network (FFN)
    - Two normalization layers (pre-attention and pre-FFN)

    Attributes:
        pre_norm (bool): Whether to apply normalization before (True) or after (False)
            attention and FFN layers.
        residual_scale (float): Scaling factor applied to residual connections.
        dropout_rate (float): Dropout probability applied after attention and FFN operations.
        attention (AttentionConfig): Configuration for the attention layer.
        feed_forward (FeedForwardConfig): Configuration for the FFN layer.
        normalization_1 (NormalizationConfig): Configuration for the first normalization layer (before/after attention).
        normalization_2 (NormalizationConfig): Configuration for the second normalization layer (before/after FFN).
    """

    type: Literal["TransformerBlock"] = "TransformerBlock"
    pre_norm: bool = False
    residual_scale: float = 1.0
    dropout_rate: float = 0.1

    attention: AttentionConfig
    feed_forward: FeedForwardConfig
    normalization_1: NormalizationConfig
    normalization_2: NormalizationConfig


# --------------------------------------------------------------------
# AddOperation Configuration
# --------------------------------------------------------------------
class AddOperationConfig(BaseConfig):
    """
    Configuration for the AddOperation module.

    This config uses the 'type' field to indicate that the corresponding
    registered building block is "AddOperation".
    """

    type: Literal["AddOperation"] = "AddOperation"
