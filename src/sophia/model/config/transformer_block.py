from typing import Any, Dict, Type

from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import TransformerBlockBase


class TransformerBlockConfig(BaseConfig):
    """
    Configuration for a Transformer Block.

    This configuration specifies the parameters and submodule configurations for a transformer block,
    including the attention mechanism, feed-forward network, and normalization. Rather than nesting
    these sub-configurations (e.g. a field "attention" containing an AttentionConfig), this design
    "flattens" them by using separate fields for each submodule's target class (as a fully qualified class name)
    and its initialization keyword arguments.

    Attributes:
        pre_norm: Whether to apply normalization before (True) or after (False) the attention and feed-forward submodules.
        residual_scale: A scaling factor applied to the residual connections.
        dropout_rate: The dropout probability applied after attention and feed-forward operations.
        attention_cls: The class for the attention submodule.
        attention_kwargs: A dictionary of keyword arguments for initializing the attention submodule.
        feed_forward_network_cls: The class for the feed-forward network submodule.
        feed_forward_network_kwargs: A dictionary of keyword arguments for initializing the feed-forward network.
        normalization_cls: The class for the normalization submodule.
        normalization_kwargs: A dictionary of keyword arguments for initializing the normalization submodule.
    """

    expected_base_class = TransformerBlockBase

    target: str
    pre_norm: bool = False
    residual_scale: float = 1.0
    dropout_rate: float = 0.1

    attention_cls: Type
    attention_kwargs: Dict[str, Any]
    feed_forward_network_cls: Type
    feed_forward_network_kwargs: Dict[str, Any]
    normalization_cls: Type
    normalization_kwargs: Dict[str, Any]
