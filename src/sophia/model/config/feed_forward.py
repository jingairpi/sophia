from typing import Any

from pydantic import Field

from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import FeedForwardNetwork


class FeedForwardConfig(BaseConfig):
    """
    Configuration for the feed-forward (FFN) network used in transformer models.

    This configuration is flattened: instead of a nested activation configuration,
    it uses separate fields for the activation function's target and keyword arguments.

    Attributes:
        hidden_size: The hidden size used in the feed-forward network.
        ffn_multiplier: The multiplier used to compute the intermediate dimension.
        dropout_rate: The dropout rate applied within the FFN.
        activation_cls: The fully qualified class name for the activation function.
        activation_kwargs: A dictionary of keyword arguments for the activation function.
    """

    expected_base_class = FeedForwardNetwork

    target: str
    hidden_size: int
    ffn_multiplier: int
    activation: Any
    dropout_rate: float = 0.1
