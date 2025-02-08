from typing import Any

import jax.numpy as jnp
from flax import linen as nn
from pydantic import BaseModel

from sophia.model.base import Model
from sophia.model.builder import instantiate_from_config
from sophia.model.config.activation import ActivationConfig
from sophia.model.config.attention import AttentionConfig
from sophia.model.config.embeddings import (
    PositionalEmbeddingConfig,
    TokenEmbeddingConfig,
)
from sophia.model.config.feed_forward import FeedForwardConfig
from sophia.model.config.normalization import NormalizationConfig
from sophia.model.config.projection import OutputProjectionConfig
from sophia.model.config.transformer_block import TransformerBlockConfig
from sophia.model.layers.activations import GELUActivation
from sophia.model.layers.attentions import MultiHeadDotProductAttention
from sophia.model.layers.feed_forwards import PositionwiseFeedForward
from sophia.model.layers.normalizations import LayerNormalization


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

    target: str = "sophia.configs.gpt2.GPT2Model"
    params: GPT2Params

    # Building block configurations.
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
            target="sophia.configs.gpt2.GPT2Model",
            params=params,
            token_embedding=TokenEmbeddingConfig(
                target="sophia.model.layers.embeddings.TokenEmbedding",
                vocab_size=params.vocab_size,
                hidden_size=params.hidden_size,
            ),
            positional_embedding=PositionalEmbeddingConfig(
                target="sophia.model.layers.embeddings.PositionalEmbedding",
                max_seq_length=params.n_positions,
                hidden_size=params.hidden_size,
            ),
            transformer_block=TransformerBlockConfig(
                target="sophia.model.blocks.transformer_block.TransformerBlock",
                pre_norm=False,
                residual_scale=1.0,
                dropout_rate=params.dropout_rate,
                attention_cls=MultiHeadDotProductAttention,
                attention_kwargs={
                    "hidden_size": params.hidden_size,
                    "num_heads": params.hidden_size // 64,  # assuming head_dim is 64
                    "dropout_rate": params.dropout_rate,
                },
                feed_forward_network_cls=PositionwiseFeedForward,
                feed_forward_network_kwargs={
                    "hidden_size": params.hidden_size,
                    "ffn_multiplier": params.ffn_multiplier,
                    "dropout_rate": params.dropout_rate,
                    "activation": GELUActivation(),
                },
                normalization_cls=LayerNormalization,
                normalization_kwargs={"epsilon": 1e-5},
            ),
            projection=OutputProjectionConfig(
                target="sophia.model.layers.projections.OutputProjection",
                hidden_size=params.hidden_size,
                output_size=params.vocab_size,
            ),
        )


# ---------------------------------------------------------------------------
# GPT2Model: A full implementation of GPT-2 assembled from the configuration.
# ---------------------------------------------------------------------------
class GPT2Model(Model, nn.Module):
    """
    GPT2Model is a concrete implementation of GPT-2 built entirely from a configuration blueprint.

    It assembles its building blocks (token embedding, positional embedding, a stack of transformer
    blocks, and output projection) based on the provided GPT2Config. All hyperparameters and submodule
    configurations are specified in the config.

    Attributes:
        config: A GPT2Config instance that defines the architecture and hyperparameters.
    """

    config: GPT2Config

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True):
        # Instantiate token embedding.
        token_embed = instantiate_from_config(self.config.token_embedding)
        x = token_embed(input_ids)  # Shape: [batch, seq_length, hidden_size]

        # Instantiate positional embedding.
        pos_embed = instantiate_from_config(self.config.positional_embedding)
        B, T = input_ids.shape
        pos_ids = jnp.arange(T)[None, :]  # Shape: [1, T]
        x = x + pos_embed(pos_ids)  # Add positional embeddings.

        # Stack transformer blocks.
        for i in range(self.config.params.n_layer):
            block = instantiate_from_config(self.config.transformer_block)
            x = block(x, deterministic=deterministic)

        # Instantiate normalization from flattened configuration.
        normalization_cls = self.config.transformer_block.normalization_cls
        normalization_kwargs = self.config.transformer_block.normalization_kwargs
        normalization = normalization_cls(**normalization_kwargs)
        x = normalization(x)

        # Instantiate the output projection.
        projection = instantiate_from_config(self.config.projection)
        logits = projection(x)
        return logits

    def init_params(self, rng_key: Any) -> Any:
        # Initialize model parameters using a dummy input.
        dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
        return self.init(rng_key, dummy_input)
