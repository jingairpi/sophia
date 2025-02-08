from pydantic import BaseModel

from sophia.model.config.activation import ActivationConfig
from sophia.model.config.attention import AttentionConfig
from sophia.model.config.embeddings import PositionEmbeddingConfig, TokenEmbeddingConfig
from sophia.model.config.feed_forward import FeedForwardConfig
from sophia.model.config.normalization import NormalizationConfig
from sophia.model.config.projection import OutputProjectionConfig
from sophia.model.config.transformer_block import TransformerBlockConfig


# -----------------------------------------------------------------------------
# GPT2Params: Hyperparameters for GPT-2
# -----------------------------------------------------------------------------
class GPT2Params(BaseModel):
    """
    GPT2Params holds the hyperparameters for a GPT-2 model.

    These parameters can be used to configure different sizes of GPT-2 models.
    """

    n_layer: int  # Number of transformer layers.
    hidden_size: int  # Dimensionality of the hidden states.
    vocab_size: int  # Vocabulary size.
    n_positions: int  # Maximum sequence length (number of positions).
    ffn_multiplier: int  # Multiplier for the feed-forward layer (defines the intermediate dimension).
    dropout_rate: float  # Dropout rate used in various submodules.


# Example defaults (you can also load these from a JSON file, etc.)
GPT2_SMALL = GPT2Params(
    n_layer=12,
    hidden_size=768,
    vocab_size=50257,
    n_positions=1024,
    ffn_multiplier=4,
    dropout_rate=0.1,
)

GPT2_MEDIUM = GPT2Params(
    n_layer=24,
    hidden_size=1024,
    vocab_size=50257,
    n_positions=1024,
    ffn_multiplier=4,
    dropout_rate=0.1,
)

GPT2_LARGE = GPT2Params(
    n_layer=36,
    hidden_size=1280,
    vocab_size=50257,
    n_positions=1024,
    ffn_multiplier=4,
    dropout_rate=0.1,
)


# -----------------------------------------------------------------------------
# GPT2Config: Full model configuration for GPT-2
# -----------------------------------------------------------------------------
class GPT2Config(BaseModel):
    """
    GPT2Config composes the configurations for the various building blocks of a GPT-2 model.

    It is used by the model.builder to instantiate a GPT-2 model dynamically. The configuration
    includes nested configurations for token embeddings, positional embeddings, transformer blocks,
    and the output projection. The hyperparameters for the model (e.g., number of layers,
    hidden size, etc.) are provided via a nested GPT2Params instance.
    """

    # Hyperparameters (e.g., small, medium, large).
    params: GPT2Params

    # Building block configurations.
    token_embedding: TokenEmbeddingConfig
    positional_embedding: PositionEmbeddingConfig
    transformer_block: TransformerBlockConfig
    projection: OutputProjectionConfig

    @classmethod
    def from_params(cls, params: GPT2Params) -> "GPT2Config":
        """
        Factory method to create a GPT2Config from GPT2Params.

        It composes the full model configuration using the provided hyperparameters.
        """
        return cls(
            params=params,
            token_embedding=TokenEmbeddingConfig(
                target="sophia.model.layers.embeddings.TokenEmbedding",
                vocab_size=params.vocab_size,
                hidden_size=params.hidden_size,
            ),
            positional_embedding=PositionEmbeddingConfig(
                target="sophia.model.layers.embeddings.PositionalEmbedding",
                max_length=params.n_positions,
                hidden_size=params.hidden_size,
            ),
            transformer_block=TransformerBlockConfig(
                target="sophia.model.blocks.transformer_block.TransformerBlock",
                pre_norm=False,
                residual_scale=1.0,
                dropout_rate=params.dropout_rate,
                # Nested configuration for the attention sub-module.
                attention=AttentionConfig(
                    target="sophia.model.layers.attentions.MultiHeadDotProductAttention",
                    hidden_size=params.hidden_size,
                    # Typically, num_heads = hidden_size // head_dim (e.g., 64).
                    # For simplicity, you might fix this ratio or provide an extra parameter.
                    num_heads=params.hidden_size // 64,
                    dropout_rate=params.dropout_rate,
                ),
                # Nested configuration for the feed-forward sub-module.
                feed_forward=FeedForwardConfig(
                    target="sophia.model.layers.feed_forwards.PositionwiseFeedForward",
                    hidden_size=params.hidden_size,
                    ffn_multiplier=params.ffn_multiplier,
                    dropout_rate=params.dropout_rate,
                    activation=ActivationConfig(
                        target="sophia.model.layers.activations.GELUActivation"
                    ),
                ),
                # Nested configuration for the normalization sub-module.
                norm=NormalizationConfig(
                    target="sophia.model.layers.normalizations.LayerNormalization",
                    epsilon=1e-5,
                ),
            ),
            projection=OutputProjectionConfig(
                target="sophia.model.layers.projections.OutputProjection",
                hidden_size=params.hidden_size,
                output_size=params.vocab_size,
            ),
        )
