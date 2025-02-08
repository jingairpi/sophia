import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.blocks.bases import TransformerBlockBase
from sophia.model.layers.bases import (
    Activation,
    AttentionLayer,
    FeedForwardNetwork,
    NormalizationLayer,
)

# -----------------------------------------------------------------------------
# Insert dummy modules into sys.modules BEFORE importing configuration classes.
# -----------------------------------------------------------------------------

# Dummy module for activation.
dummy_activation_module = types.ModuleType("dummy_activation_module")


class DummyActivation(Activation):
    def __call__(self, x, *args, **kwargs):
        return x


dummy_activation_module.DummyActivation = DummyActivation


class NotActivation:
    pass


dummy_activation_module.NotActivation = NotActivation
sys.modules["dummy_activation_module"] = dummy_activation_module

# Dummy module for attention.
dummy_attention_module = types.ModuleType("dummy_attention_module")


# For testing, DummyAttention will be defined later in our dummy module.
class DummyAttention(AttentionLayer):
    def __call__(self, x, *args, **kwargs):
        return x


dummy_attention_module.DummyAttention = DummyAttention
sys.modules["dummy_attention_module"] = dummy_attention_module

# Dummy module for feed-forward network.
dummy_ffn_module = types.ModuleType("dummy_ffn_module")


class DummyFeedForward(FeedForwardNetwork):
    def __call__(self, x, *args, **kwargs):
        return x


dummy_ffn_module.DummyFeedForward = DummyFeedForward


class NotFeedForward:
    pass


dummy_ffn_module.NotFeedForward = NotFeedForward
sys.modules["dummy_ffn_module"] = dummy_ffn_module

# Dummy module for normalization.
dummy_norm_module = types.ModuleType("dummy_norm_module")


class DummyNorm(NormalizationLayer):
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
# Create dummy nested configurations.
# -----------------------------------------------------------------------------

# Create a dummy AttentionConfig instance.
dummy_attention_config = AttentionConfig(
    target="dummy_attention_module.DummyAttention",
    hidden_size=512,
    num_heads=8,
    dropout_rate=0.2,
)

# Create a dummy FeedForwardConfig instance.
dummy_ffn_config = FeedForwardConfig(
    target="dummy_ffn_module.DummyFeedForward",
    hidden_size=512,
    ffn_multiplier=4,
    dropout_rate=0.2,
    activation=ActivationConfig(target="dummy_activation_module.DummyActivation"),
)

# Create a dummy NormConfig instance.
dummy_norm_config = NormalizationConfig(
    target="dummy_norm_module.DummyNorm", epsilon=1e-5
)


# -----------------------------------------------------------------------------
# Tests for TransformerBlockConfig.
# -----------------------------------------------------------------------------
def test_transformer_block_config_valid():
    config = TransformerBlockConfig(
        target="dummy_block_module.DummyTransformerBlock",
        pre_norm=True,
        residual_scale=0.9,
        dropout_rate=0.1,
        attention=dummy_attention_config,
        feed_forward=dummy_ffn_config,
        norm=dummy_norm_config,
    )
    assert config.target == "dummy_block_module.DummyTransformerBlock"
    assert config.pre_norm is True
    assert config.residual_scale == 0.9


def test_transformer_block_config_invalid_target():
    with pytest.raises(ValidationError) as excinfo:
        TransformerBlockConfig(
            target="dummy_block_module.NotTransformerBlock",
            pre_norm=True,
            residual_scale=0.9,
            dropout_rate=0.1,
            attention=dummy_attention_config,
            feed_forward=dummy_ffn_config,
            norm=dummy_norm_config,
        )
    # The error message should indicate that the target must be a subclass of TransformerBlock.
    assert "must be a subclass of TransformerBlock" in str(excinfo.value)


def test_transformer_block_config_invalid_fqname():
    with pytest.raises(ValidationError) as excinfo:
        TransformerBlockConfig(
            target="InvalidNameWithoutDot",
            pre_norm=True,
            residual_scale=0.9,
            dropout_rate=0.1,
            attention=dummy_attention_config,
            feed_forward=dummy_ffn_config,
            norm=dummy_norm_config,
        )
    assert "must be a valid fully qualified class name" in str(excinfo.value)
