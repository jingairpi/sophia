from typing import Any

import jax
import jax.numpy as jnp

from sophia.model.base import Model
from sophia.model.config import LayerConfig, ModelConfig
from sophia.model.registry import get_layer_class


class Builder:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def _build_layer(self, layer_conf: LayerConfig) -> Any:
        """
        Recursively build a layer from its configuration.
        Returns an instantiated Flax module (e.g., DummyLayer(...)).
        """
        layer_cls = get_layer_class(layer_conf.type)
        kwargs = dict(layer_conf.config)

        for key, value in kwargs.items():
            if isinstance(value, dict) and "type" in value:
                from sophia.model.config import LayerConfig

                nested_config = LayerConfig(**value)
                kwargs[key] = self._build_layer(nested_config)

        return layer_cls(**kwargs)

    def build(self) -> Model:
        """
        Dynamically create a new Python class (via type()) that:
          1. Stores built Flax layers in a 'layers' attribute.
          2. Implements 'init(rng_key, sample_input)' to initialize all layers.
          3. Implements 'apply(params, x)' to do a forward pass using the provided params.

        Returns an instance of that dynamic class, which inherits from Model.
        """
        # 1) Build submodules from the config
        layers = []
        for layer_conf in self.model_config.layers:
            repeat_count = layer_conf.repeat if layer_conf.repeat is not None else 1
            for _ in range(repeat_count):
                layers.append(self._build_layer(layer_conf))

        # 2) Define the methods required by Model
        def init(self, rng_key: Any, sample_input: jnp.ndarray) -> dict:
            """
            Initialize and return the model parameters for all layers.
            """
            params = {}
            for i, layer in enumerate(self.layers):
                key = f"{layer.__class__.__name__}_{i}"
                rng_key, subkey = jax.random.split(rng_key)
                layer_params = layer.init(subkey, sample_input)
                params[key] = layer_params
            return params

        def apply(self, params: dict, x: jnp.ndarray) -> jnp.ndarray:
            """
            Forward pass using the initialized parameters.

            Args:
                params: The dictionary of parameters keyed by layer name/index.
                x: The input tensor.

            Returns:
                The output of applying each layer in sequence.
            """
            for i, layer in enumerate(self.layers):
                key = f"{layer.__class__.__name__}_{i}"
                x = layer.apply(params[key], x)
            return x

        # 3) Construct a new dynamic class that inherits from Model
        attrs = {
            "layers": layers,
            "init": init,
            "apply": apply,
        }
        NewModelClass = type(self.model_config.name, (Model,), attrs)

        # 4) Return an instance of that new class
        return NewModelClass()
