from sophia.model.config.activation import ActivationConfig
from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import FeedForwardNetwork


class FeedForwardConfig(BaseConfig):
    """
    Configuration for the feed-forward (FFN) network used in transformer models.

    Attributes:
        hidden_size: The hidden size used in the feed-forward network.
        ffn_multiplier: The multiplier to compute the intermediate dimension.
        dropout_rate: Dropout rate applied within the FFN.
        activation: A nested configuration for the activation function.
    """

    expected_base_class = FeedForwardNetwork

    target: str
    hidden_size: int
    ffn_multiplier: int
    dropout_rate: float = 0.1
    activation: ActivationConfig
