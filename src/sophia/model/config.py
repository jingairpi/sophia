from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LayerConfig(BaseModel):
    type: str  # The registered layer type (e.g. "TransformerBlock")
    config: Dict[str, Any]  # A dictionary of parameters for the layer
    repeat: Optional[int] = None  # Optional: number of repeats for this layer


class ModelConfig(BaseModel):
    name: str  # The name of the model
    dtype: str  # Global precision (e.g. "float32", "bfloat16")
    layers: List[LayerConfig]  # A list of layer configurations


# Activation configuration derived from LayerConfig
class ActivationConfig(LayerConfig):
    """
    Configuration for an activation function layer.

    Inherits 'type' and 'config' from LayerConfig, and can be extended
    with activation-specific fields if necessary.
    """

    # No additional fields for now.
    pass


# Attention layer configuration derived from LayerConfig
class AttentionLayerConfig(LayerConfig):
    """
    Configuration for an attention layer.

    Inherits 'type' and 'config', and adds additional fields required by the attention mechanism.

    Attributes:
        num_heads: The number of attention heads.
        head_dim: The dimensionality of each attention head.
        dropout_rate: The dropout rate applied to the attention weights.
    """

    num_heads: int
    head_dim: int
    dropout_rate: Optional[float] = 0.0


# TokenEmbedding configuration derived from LayerConfig
class TokenEmbeddingConfig(LayerConfig):
    """
    Configuration for a token embedding layer.

    Attributes:
        vocab_size: The size of the vocabulary.
        hidden_size: The dimensionality of the token embeddings.
    """

    vocab_size: int
    hidden_size: int


# PositionalEmbedding configuration derived from LayerConfig
class PositionalEmbedding(LayerConfig):
    """
    Configuration for a positional embedding layer.

    Attributes:
        max_seq_length: The maximum sequence length the model supports.
        hidden_size: The dimensionality of the positional embeddings.
    """

    max_seq_length: int
    hidden_size: int


# Feed-forward network configuration derived from LayerConfig
class FeedForwardConfig(LayerConfig):
    """
    Configuration for a feed-forward network.

    Attributes:
        hidden_dim: The input (or hidden) dimension.
        ff_dim: The expansion dimension of the feed-forward network.
        dropout_rate: The dropout rate applied after the feed-forward operation.
        activation: The nested configuration for the activation function.
    """

    hidden_dim: int
    ff_dim: int
    dropout_rate: Optional[float] = 0.0
    activation: ActivationConfig


# LayerNormalization configuration derived from LayerConfig
class LayerNormalizationConfig(LayerConfig):
    """
    Configuration for a normalization layer.

    Attributes:
        eps: A small constant added to the denominator for numerical stability.
    """

    eps: float = 1e-5


# RMSNormalization configuration derived from LayerConfig
class RMSNormalization(LayerConfig):
    """
    Configuration for a normalization layer.

    Attributes:
        features: The number of features in the input tensor.
        eps: A small constant added to the denominator for numerical stability.
    """

    features: int
    eps: float = 1e-5


# OutputProjection configuration derived from LayerConfig
class OutputProjectionConfig(LayerConfig):
    """
    Configuration for a projection layer.

    Attributes:
        input_dim: The dimensionality of the input features.
        output_dim: The dimensionality of the projected output.
    """

    input_dim: int
    output_dim: int


# Transformer block configuration derived from LayerConfig
class TransformerBlockConfig(LayerConfig):
    """
    Configuration for a generic Transformer block that nests several sub-modules.

    Attributes:
        hidden_size: The dimensionality of the transformer's hidden states.
        num_heads: The number of attention heads.
        ff_dim: The inner dimensionality of the feed-forward network.
        dropout_rate: The dropout rate applied after attention and feed-forward operations.
        pre_norm: If true, normalization is applied before the sub-layers.
        residual_scale: A scaling factor for residual connections.
        attention: The nested configuration for the attention layer.
        feed_forward_network: The nested configuration for the feed-forward network.
        normalization_1: The nested configuration for the first normalization layer.
        normalization_2: The nested configuration for the second normalization layer.
    """

    hidden_size: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1
    pre_norm: bool = True
    residual_scale: float = 1.0
    attention: AttentionLayerConfig
    feed_forward_network: FeedForwardConfig
    normalization_1: LayerNormalizationConfig
    normalization_2: LayerNormalizationConfig
