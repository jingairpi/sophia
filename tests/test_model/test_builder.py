import jax.numpy as jnp
import pytest
from flax import linen as nn

from sophia.model.builder import Builder
from sophia.model.config import LayerConfig, ModelConfig
from sophia.model.registry import register_layer


@register_layer("DummyAttention")
class DummyAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, **kwargs):
        return x


@register_layer("DummyFeedForward")
class DummyFeedForward(nn.Module):
    ff_dim: int

    @nn.compact
    def __call__(self, x, **kwargs):
        return x


@register_layer("DummyNorm")
class DummyNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, **kwargs):
        return x


@register_layer("DummyTransformer")
class DummyTransformer(nn.Module):
    dropout_rate: float
    attention: nn.Module
    feed_forward: nn.Module
    normalization: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = self.attention(x)
        x = self.feed_forward(x)
        x = self.normalization(x)
        return x


def test_model_builder_nested_layers():
    config_dict = {
        "name": "TestModel",
        "dtype": "float32",
        "layers": [
            {
                "type": "DummyTransformer",
                "config": {
                    "dropout_rate": 0.1,
                    "attention": {
                        "type": "DummyAttention",
                        "config": {"num_heads": 4, "head_dim": 32},
                    },
                    "feed_forward": {
                        "type": "DummyFeedForward",
                        "config": {"ff_dim": 128},
                    },
                    "normalization": {"type": "DummyNorm", "config": {"eps": 1e-5}},
                },
                "repeat": 2,
            }
        ],
    }

    model_config = ModelConfig(**config_dict)
    builder = Builder(model_config)
    model = builder.build()

    assert isinstance(model, nn.Sequential)
    assert len(model.layers) == 2
