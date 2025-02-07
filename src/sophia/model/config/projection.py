from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import ProjectionLayer


class OutputProjectionConfig(BaseConfig):
    """
    Configuration for an output projection (unembedding) layer.

    The 'target' field should be a fully qualified class name whose class is a subclass of
    ProjectionLayer.
    """

    expected_base_class = ProjectionLayer

    target: str
    hidden_size: int
    output_size: int
