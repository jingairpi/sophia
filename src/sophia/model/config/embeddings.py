from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import EmbeddingLayer


class TokenEmbeddingConfig(BaseConfig):
    """
    Configuration for a token embedding layer.

    The 'target' field should be a fully qualified class name whose class is a subclass of
    EmbeddingLayer.
    """

    expected_base_class = EmbeddingLayer

    target: str
    vocab_size: int
    hidden_size: int


class PositionalEmbeddingConfig(BaseConfig):
    """
    Configuration for a positional embedding layer.

    The 'target' field should be a fully qualified class name whose class is a subclass of
    EmbeddingLayer.
    """

    expected_base_class = EmbeddingLayer

    target: str
    max_length: int
    hidden_size: int
