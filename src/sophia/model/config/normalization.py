from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import NormalizationLayer


class NormalizationConfig(BaseConfig):
    """
    Configuration for a normalization layer (e.g., LayerNorm).

    The 'target' field should be a fully qualified class name whose class is a subclass
    of NormalizationLayer.
    """

    expected_base_class = NormalizationLayer

    _target_: str
    epsilon: float = 1e-5
