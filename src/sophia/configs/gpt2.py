from typing import Any

import jax.numpy as jnp
from flax import linen as nn
from pydantic import BaseModel
from typing_extensions import Literal

from sophia.model.builder import instantiate_from_config
from sophia.model.config import (
    AddOperationConfig,
    LayerNormalizationConfig,
    MultiHeadDotProductAttentionConfig,
    OutputProjectionConfig,
    PositionalEmbeddingConfig,
    PositionwiseFeedForwardConfig,
    TokenEmbeddingConfig,
    TransformerBlockConfig,
)
from sophia.model.graph import GraphConfig, NodeSpec
from sophia.model.layers.activations import GELUActivation
from sophia.model.layers.attentions import MultiHeadDotProductAttention
from sophia.model.layers.embeddings import PositionalEmbedding, TokenEmbedding
from sophia.model.layers.feed_forwards import PositionwiseFeedForward
from sophia.model.layers.normalizations import LayerNormalization
from sophia.model.layers.operations import AddOperation
from sophia.model.layers.projections import OutputProjection
from sophia.model.layers.transformer_block import TransformerBlock


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
class GPT2Config(GraphConfig):
    """
    A GPT2-specific graph config for building a GPT-2 model.
    """

    type: Literal["GPT2Model"] = "GPT2Model"
    params: GPT2Params
    token_embedding: TokenEmbeddingConfig
    positional_embedding: PositionalEmbeddingConfig
    transformer_block: TransformerBlockConfig
    layer_normalization: LayerNormalizationConfig
    output_projection: OutputProjectionConfig

    @classmethod
    def from_params(cls, params: GPT2Params) -> "GPT2Config":
        """
        Build a GPT2Config based on the provided GPT2Params.
        This method ties the model architecture to the hyperparameters in GPT2Params.
        """
        token_cfg = TokenEmbeddingConfig(
            type="TokenEmbedding",
            vocab_size=params.vocab_size,
            hidden_size=params.hidden_size,
        )
        pos_cfg = PositionalEmbeddingConfig(
            type="PositionalEmbedding",
            max_seq_length=params.n_positions,
            hidden_size=params.hidden_size,
        )
        block_cfg = TransformerBlockConfig(
            type="TransformerBlock",
            attention=MultiHeadDotProductAttentionConfig(
                hidden_size=params.hidden_size,
                num_heads=params.hidden_size // 64,
                dropout_rate=params.dropout_rate,
            ),
            feed_forward=PositionwiseFeedForwardConfig(
                hidden_size=params.hidden_size,
                ffn_multiplier=params.ffn_multiplier,
                dropout_rate=params.dropout_rate,
                activation=GELUActivation(),
            ),
            normalization_1=LayerNormalizationConfig(
                type="LayerNormalization", epsilon=1e-5
            ),
            normalization_2=LayerNormalizationConfig(
                type="LayerNormalization", epsilon=1e-5
            ),
            hidden_size=params.hidden_size,
            dropout_rate=params.dropout_rate,
            pre_norm=False,
            residual_scale=1.0,
        )
        ln_cfg = LayerNormalizationConfig(type="LayerNormalization", epsilon=1e-5)
        proj_cfg = OutputProjectionConfig(
            type="OutputProjection",
            hidden_size=params.hidden_size,
            output_size=params.vocab_size,
        )

        nodes: List[NodeSpec] = []

        nodes.append(NodeSpec(name="token_emb", config=token_cfg, inputs=["input_ids"]))

        nodes.append(NodeSpec(name="pos_emb", config=pos_cfg, inputs=["input_ids"]))

        nodes.append(
            NodeSpec(
                name="embedded_input",
                config=AddOperationConfig(),
                inputs=["token_emb", "pos_emb"],
            )
        )

        previous_output = "embedded_input"
        for i in range(params.n_layer):
            block_name = f"transformer_block_{i}"
            nodes.append(
                NodeSpec(name=block_name, config=block_cfg, inputs=[previous_output])
            )
            previous_output = block_name

        nodes.append(NodeSpec(name="final_ln", config=ln_cfg, inputs=[previous_output]))
        previous_output = "final_ln"

        nodes.append(NodeSpec(name="logits", config=proj_cfg, inputs=[previous_output]))

        output_names = ["logits"]

        return cls(
            params=params,
            token_embedding=token_cfg,
            positional_embedding=pos_cfg,
            transformer_block=block_cfg,
            layer_normalization=ln_cfg,
            output_projection=proj_cfg,
            nodes=nodes,
            output_names=output_names,
            model_type="GPT2Model",
        )
