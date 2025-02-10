import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.layers.bases import (
    AttentionLayer,
    FeedForwardNetwork,
    NormalizationLayer,
    TransformerBlockBase,
)

# -----------------------------------------------------------------------------
# Insert dummy modules into sys.modules BEFORE importing configuration classes.
# -----------------------------------------------------------------------------

# Dummy module for activation.
dummy_activation_module = types.ModuleType("dummy_activation_module")


class DummyActivation:
    def __call__(self, x, *args, **kwargs):
        return x


dummy_activation_module.DummyActivation = DummyActivation


class NotActivation:
    pass


dummy_activation_module.NotActivation = NotActivation
sys.modules["dummy_activation_module"] = dummy_activation_module

# Dummy module for attention.
dummy_attention_module = types.ModuleType("dummy_attention_module")


class DummyAttention:
    def __call__(self, x, *args, **kwargs):
        return x


dummy_attention_module.DummyAttention = DummyAttention
sys.modules["dummy_attention_module"] = dummy_attention_module

# Dummy module for feed-forward network.
dummy_ffn_module = types.ModuleType("dummy_ffn_module")


class DummyFeedForward:
    def __call__(self, x, *args, **kwargs):
        return x


dummy_ffn_module.DummyFeedForward = DummyFeedForward


class NotFeedForward:
    pass


dummy_ffn_module.NotFeedForward = NotFeedForward
sys.modules["dummy_ffn_module"] = dummy_ffn_module

# Dummy module for normalization.
dummy_norm_module = types.ModuleType("dummy_norm_module")


class DummyNorm:
    def __call__(self, x, *args, **kwargs):
        return x


dummy_norm_module.DummyNorm = DummyNorm


class NotNorm:
    pass


dummy_norm_module.NotNorm = NotNorm
sys.modules["dummy_norm_module"] = dummy_norm_module

# Dummy module for transformer block.
dummy_block_module = types.ModuleType("dummy_block_module")


class DummyTransformerBlock(TransformerBlockBase):
    def __call__(self, x, *args, **kwargs):
        return x


dummy_block_module.DummyTransformerBlock = DummyTransformerBlock


class NotTransformerBlock:
    pass


dummy_block_module.NotTransformerBlock = NotTransformerBlock
sys.modules["dummy_block_module"] = dummy_block_module

from sophia.model.config.activation import ActivationConfig
from sophia.model.config.attention import AttentionConfig
from sophia.model.config.feed_forward import FeedForwardConfig
from sophia.model.config.normalization import NormalizationConfig
from sophia.model.config.transformer_block import TransformerBlockConfig

# For the purpose of this test, we also need the expected base classes for nested configs.
from sophia.model.layers.bases import (
    AttentionLayer,
    FeedForwardNetwork,
    NormalizationLayer,
)

# -----------------------------------------------------------------------------
# Create dummy flattened configurations.
# -----------------------------------------------------------------------------
# Instead of nested configs, we now build flattened dictionaries.
dummy_attention_config_flat = {
    "attention_cls": DummyAttention,  # actual class
    "attention_kwargs": {
        "hidden_size": 512,
        "num_heads": 8,
        "dropout_rate": 0.2,
    },
}

dummy_ffn_config_flat = {
    "feed_forward_network_cls": DummyFeedForward,  # actual class
    "feed_forward_network_kwargs": {
        "hidden_size": 512,
        "ffn_multiplier": 4,
        "dropout_rate": 0.2,
        # In Option B you directly pass an activation instance.
        "activation": DummyActivation(),
    },
}

dummy_norm_config_flat = {
    "normalization_cls": DummyNorm,  # actual class
    "normalization_kwargs": {"epsilon": 1e-5},
}


# -----------------------------------------------------------------------------
# Tests for TransformerBlockConfig.
# -----------------------------------------------------------------------------
def test_transformer_block_config_valid():
    # Create a flattened configuration for a transformer block.
    config = TransformerBlockConfig(
        target="dummy_block_module.DummyTransformerBlock",
        pre_norm=True,
        residual_scale=0.9,
        dropout_rate=0.1,
        # Provide flattened sub-configurations:
        attention_cls=dummy_attention_config_flat["attention_cls"],
        attention_kwargs=dummy_attention_config_flat["attention_kwargs"],
        feed_forward_network_cls=dummy_ffn_config_flat["feed_forward_network_cls"],
        feed_forward_network_kwargs=dummy_ffn_config_flat[
            "feed_forward_network_kwargs"
        ],
        normalization_cls=dummy_norm_config_flat["normalization_cls"],
        normalization_kwargs=dummy_norm_config_flat["normalization_kwargs"],
    )
    # Check that top-level parameters are set correctly.
    assert config.target == "dummy_block_module.DummyTransformerBlock"
    assert config.pre_norm is True
    assert config.residual_scale == 0.9
    # Check that the flattened keys exist.
    assert "attention_cls" in config.model_dump()
    assert "attention_kwargs" in config.model_dump()
    assert "feed_forward_network_cls" in config.model_dump()
    assert "feed_forward_network_kwargs" in config.model_dump()
    assert "normalization_cls" in config.model_dump()
    assert "normalization_kwargs" in config.model_dump()


def test_transformer_block_config_invalid_target():
    with pytest.raises(ValidationError) as excinfo:
        TransformerBlockConfig(
            target="dummy_block_module.NotTransformerBlock",
            pre_norm=True,
            residual_scale=0.9,
            dropout_rate=0.1,
            attention_cls=dummy_attention_config_flat["attention_cls"],
            attention_kwargs=dummy_attention_config_flat["attention_kwargs"],
            feed_forward_network_cls=dummy_ffn_config_flat["feed_forward_network_cls"],
            feed_forward_network_kwargs=dummy_ffn_config_flat[
                "feed_forward_network_kwargs"
            ],
            normalization_cls=dummy_norm_config_flat["normalization_cls"],
            normalization_kwargs=dummy_norm_config_flat["normalization_kwargs"],
        )
    assert "must be a subclass of TransformerBlock" in str(excinfo.value)


def test_transformer_block_config_invalid_fqname():
    with pytest.raises(ValidationError) as excinfo:
        TransformerBlockConfig(
            target="InvalidNameWithoutDot",
            pre_norm=True,
            residual_scale=0.9,
            dropout_rate=0.1,
            attention_cls=dummy_attention_config_flat["attention_cls"],
            attention_kwargs=dummy_attention_config_flat["attention_kwargs"],
            feed_forward_network_cls=dummy_ffn_config_flat["feed_forward_network_cls"],
            feed_forward_network_kwargs=dummy_ffn_config_flat[
                "feed_forward_network_kwargs"
            ],
            normalization_cls=dummy_norm_config_flat["normalization_cls"],
            normalization_kwargs=dummy_norm_config_flat["normalization_kwargs"],
        )
    assert "must be a valid fully qualified class name" in str(excinfo.value)
