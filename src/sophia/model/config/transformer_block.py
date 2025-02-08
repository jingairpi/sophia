from sophia.model.blocks.bases import TransformerBlockBase
from sophia.model.config.attention import AttentionConfig
from sophia.model.config.base import BaseConfig
from sophia.model.config.feed_forward import FeedForwardConfig
from sophia.model.config.normalization import NormalizationConfig


class TransformerBlockConfig(BaseConfig):
    """
    Configuration for a Transformer Block. This configuration is nested,
    meaning that it contains the configurations for its internal components:
    attention, feed-forward network, and normalization. Such a design is
    especially useful when supporting various transformer variants.

    Attributes:
        pre_norm: Whether to use pre-normalization (True) or post-normalization (False).
        residual_scale: Scaling factor applied to residual connections.
        dropout_rate: Dropout rate applied within the block.
        attention: Nested configuration for the attention sub-module.
        feed_forward: Nested configuration for the feed-forward sub-module.
        norm: Nested configuration for the normalization sub-module.
    """

    expected_base_class = TransformerBlockBase

    target: str
    pre_norm: bool = False
    residual_scale: float = 1.0
    dropout_rate: float = 0.1

    # Nested configurations for the sub-modules.
    attention: AttentionConfig
    feed_forward: FeedForwardConfig
    norm: NormalizationConfig
