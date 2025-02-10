from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import Activation


class ActivationConfig(BaseConfig):
    """
    Configuration for the activation function.

    The target field specifies the fully qualified name of the activation
    function to be used. The default is "flax.linen.gelu", but this can be
    overridden in the configuration file.
    """

    expected_base_class = Activation

    target: str
