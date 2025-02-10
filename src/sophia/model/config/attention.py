from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import AttentionLayer


class AttentionConfig(BaseConfig):
    """
    Configuration for an attention module.

    The 'target' field should be a fully qualified class name whose class is a subclass
    of `AttentionLayer`. This field must be provided in the configuration (or overridden)
    to specify different attention implementations.
    """

    expected_base_class = AttentionLayer

    target: str
    hidden_size: int
    num_heads: int
    dropout_rate: float = 0.1
