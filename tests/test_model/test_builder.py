import sys
from types import ModuleType

import jax.numpy as jnp

dummy_base_module = ModuleType("sophia.model.base")


class DummyModelBase:
    """
    A dummy implementation for the abstract Model base.
    """

    def init(self, rng_key: any, sample_input: jnp.ndarray) -> any:
        raise NotImplementedError


dummy_base_module.Model = DummyModelBase
sys.modules["sophia.model.base"] = dummy_base_module

from typing import Any, Dict

import jax
import pytest
from flax import linen as nn

from sophia.model.builder import Builder
from sophia.model.config import LayerConfig, ModelConfig
from sophia.model.registry import LAYER_REGISTRY, get_layer_class, register_layer


@register_layer("DummyLayer")
class DummyLayer(nn.Module):
    param: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.param


def create_dummy_model_config() -> ModelConfig:
    """
    Create a dummy model configuration using DummyLayer.
    This configuration specifies:
      - One DummyLayer with param=5.
      - One DummyLayer with param=3 repeated twice.
    Total layers = 3.
    """
    config_dict = {
        "name": "MyDynamicModel",
        "dtype": "float32",
        "layers": [
            {"type": "DummyLayer", "config": {"param": 5}},
            {"type": "DummyLayer", "config": {"param": 3}, "repeat": 2},
        ],
    }
    return ModelConfig(**config_dict)


def test_builder_creates_dynamic_model():
    model_config = create_dummy_model_config()
    builder = Builder(model_config)
    dynamic_model = builder.build()

    from sophia.model.base import Model

    assert isinstance(dynamic_model, Model)

    assert dynamic_model.__class__.__name__ == "MyDynamicModel"

    assert hasattr(dynamic_model, "layers")
    assert len(dynamic_model.layers) == 3

    rng_key = jax.random.PRNGKey(0)
    params = dynamic_model.init(rng_key, jnp.ones((1,)))
    assert isinstance(params, dict), "init(...) did not return a dict."

    for i, layer in enumerate(dynamic_model.layers):
        expected_key = f"{layer.__class__.__name__}_{i}"
        assert (
            expected_key in params
        ), f"Missing key '{expected_key}' in param dictionary."

    x = jnp.array(10)
    output_value = dynamic_model.apply(params, x)
    assert output_value == 21, f"Expected 21, got {output_value}"
