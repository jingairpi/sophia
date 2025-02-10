from typing import Any

import jax.numpy as jnp
from flax import linen as nn
from pydantic import BaseModel
from typing_extensions import Literal

from sophia.model.builder import instantiate_from_config
from sophia.model.config import (
    LayerNormalizationConfig,
    MultiHeadDotProductAttentionConfig,
    OutputProjectionConfig,
    PositionalEmbeddingConfig,
    PositionwiseFeedForwardConfig,
    TokenEmbeddingConfig,
    TransformerBlockConfig,
)
from sophia.model.layers.activations import GELUActivation
from sophia.model.layers.attentions import MultiHeadDotProductAttention
from sophia.model.layers.embeddings import PositionalEmbedding, TokenEmbedding
from sophia.model.layers.feed_forwards import PositionwiseFeedForward
from sophia.model.layers.normalizations import LayerNormalization
from sophia.model.layers.projections import OutputProjection
from sophia.model.layers.transformer_block import TransformerBlock
from sophia.model.registry import register


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


# Example default hyperparameters.
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

    This configuration is used by the model builder to dynamically instantiate a GPT-2 model.
    It includes nested configurations for token embeddings, positional embeddings,
    transformer blocks, and the output projection. The overall hyperparameters for the model
    (e.g. number of layers, hidden size, etc.) are provided via a nested GPT2Params instance.
    """

    type: Literal["GPT2Model"] = "GPT2Model"
    params: GPT2Params

    token_embedding: TokenEmbeddingConfig
    positional_embedding: PositionalEmbeddingConfig
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
                vocab_size=params.vocab_size,
                hidden_size=params.hidden_size,
            ),
            positional_embedding=PositionalEmbeddingConfig(
                max_seq_length=params.n_positions,
                hidden_size=params.hidden_size,
            ),
            transformer_block=TransformerBlockConfig(
                pre_norm=False,
                residual_scale=1.0,
                dropout_rate=params.dropout_rate,
                attention=MultiHeadDotProductAttentionConfig(
                    hidden_size=params.hidden_size,
                    num_heads=params.hidden_size
                    // 64,  # Assuming each head has dim 64.
                    dropout_rate=params.dropout_rate,
                ),
                feed_forward=PositionwiseFeedForwardConfig(
                    hidden_size=params.hidden_size,
                    ffn_multiplier=params.ffn_multiplier,
                    dropout_rate=params.dropout_rate,
                    activation=GELUActivation(),
                ),
                normalization_1=LayerNormalizationConfig(
                    epsilon=1e-5,
                ),
                normalization_2=LayerNormalizationConfig(
                    epsilon=1e-5,
                ),
            ),
            projection=OutputProjectionConfig(
                hidden_size=params.hidden_size,
                output_size=params.vocab_size,
            ),
        )


# -----------------------------------------------------------------------------
# GPT2Model: A full implementation of GPT-2 assembled from the configuration.
# -----------------------------------------------------------------------------
@register
class GPT2Model(nn.Module):
    """
    GPT2Model is a concrete implementation of GPT-2 built entirely from a configuration blueprint.

    It assembles its building blocks (token embedding, positional embedding, a stack of transformer
    blocks, and output projection) based on the provided GPT2Config. All hyperparameters and submodule
    configurations are specified in the config.

    This implementation does not depend on any abstract Model base class.

    Attributes:
        config: A GPT2Config instance that defines the architecture and hyperparameters.
    """

    config: GPT2Config

    @nn.compact
    def __call__(
        self, input_ids: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        # Instantiate token embedding and apply it.
        token_embed = instantiate_from_config(self.config.token_embedding)
        x = token_embed(input_ids)  # Expected shape: [batch, seq_length, hidden_size]

        # Instantiate positional embedding and add it to token embeddings.
        pos_embed = instantiate_from_config(self.config.positional_embedding)
        B, T = input_ids.shape
        pos_ids = jnp.arange(T)[None, :]  # Shape: [1, T]
        x = x + pos_embed(pos_ids)

        # Stack transformer blocks.
        for _ in range(self.config.params.n_layer):
            block = instantiate_from_config(self.config.transformer_block)
            x = block(x, deterministic=deterministic)

        # Apply the final normalization.
        normalization = instantiate_from_config(
            self.config.transformer_block.normalization_2
        )
        x = normalization(x)

        # Apply the output projection.
        projection = instantiate_from_config(self.config.projection)
        logits = projection(x)
        return logits

    def init(self, rng_key: Any, sample_input: jnp.ndarray) -> Any:
        """
        Initialize and return the model parameters using a sample input.

        Args:
            rng_key: A random key for parameter initialization.
            sample_input: A sample input (e.g. a dummy token sequence) used for shape inference.

        Returns:
            The initialized parameter PyTree.
        """
        return self.init_with_output(rng_key, sample_input)[1]
