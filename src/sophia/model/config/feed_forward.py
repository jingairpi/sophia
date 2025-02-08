from typing import Any

from pydantic import Field, field_validator

from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import Activation, FeedForwardNetwork


class FeedForwardConfig(BaseConfig):
    """
    Configuration for the feed-forward (FFN) network used in transformer models.

    This configuration is flattened and expects that the activation function is provided as
    an already-instantiated callable. In other words, rather than providing a nested configuration
    (with separate fields for the activation function’s target class and its initialization arguments),
    the caller should pass the activation function instance directly.

    Attributes:
        target: A fully qualified class name (as a string) of the feed-forward network implementation.
        hidden_size: The hidden size used in the feed-forward network.
        ffn_multiplier: The multiplier used to compute the intermediate dimension.
        activation: An instance of the activation function to be applied between dense layers.
        dropout_rate: The dropout rate applied within the feed-forward network.
    """

    expected_base_class = FeedForwardNetwork

    target: str
    hidden_size: int
    ffn_multiplier: int
    activation: Any
    dropout_rate: float = 0.1

    @field_validator("activation", mode="before")
    @classmethod
    def check_activation(cls, v: Any) -> Any:
        from sophia.model.layers.bases import Activation

        if not isinstance(v, Activation):
            raise ValueError("activation must be a subclass of Activation")
        return v
