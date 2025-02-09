from typing import Any, Dict

from flax import linen as nn

from sophia.model.config import LayerConfig, ModelConfig
from sophia.model.registry import get_layer_class


class Builder:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def build(self) -> nn.Module:
        """
        Build the model by constructing each layer as specified in the configuration.
        Returns a Flax Sequential module containing the layers.
        """
        layers = []
        for layer_conf in self.model_config.layers:
            if layer_conf.repeat is not None:
                for _ in range(layer_conf.repeat):
                    layers.append(self._build_layer(layer_conf))
            else:
                layers.append(self._build_layer(layer_conf))
        return nn.Sequential(layers)

    def _build_layer(self, layer_conf: LayerConfig) -> nn.Module:
        """
        Recursively build a layer from its configuration.
        """
        layer_cls = get_layer_class(layer_conf.type)
        kwargs = layer_conf.config.copy()

        for key, value in kwargs.items():
            if isinstance(value, dict) and "type" in value:
                from sophia.model.config import LayerConfig

                nested_config = LayerConfig(**value)
                kwargs[key] = self._build_layer(nested_config)

        return layer_cls(**kwargs)
