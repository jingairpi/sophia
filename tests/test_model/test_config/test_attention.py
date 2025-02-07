import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.attention import AttentionConfig
from sophia.model.layers.bases import AttentionLayer


# -----------------------------------------------------------------------------
# Define a dummy subclass of AttentionLayer for testing.
# -----------------------------------------------------------------------------
class DummyAttention(AttentionLayer):
    def __call__(self, hidden_states, *args, **kwargs):
        return hidden_states


# Define a dummy class that is NOT a subclass of AttentionLayer.
class NotAttention:
    pass


# Create a dummy module and add both DummyAttention and NotAttention.
dummy_module = types.ModuleType("dummy_module")
dummy_module.DummyAttention = DummyAttention
dummy_module.NotAttention = NotAttention
sys.modules["dummy_module"] = dummy_module


# -----------------------------------------------------------------------------
# Tests for AttentionConfig
# -----------------------------------------------------------------------------
def test_attention_config_valid():
    config = AttentionConfig(
        target="dummy_module.DummyAttention",
        hidden_size=512,
        num_heads=8,
        dropout_rate=0.2,
    )
    assert config.target == "dummy_module.DummyAttention"
    assert config.hidden_size == 512
    assert config.num_heads == 8
    assert config.dropout_rate == 0.2


def test_attention_config_invalid_target_not_subclass():
    with pytest.raises(ValidationError) as excinfo:
        AttentionConfig(
            target="dummy_module.NotAttention",
            hidden_size=512,
            num_heads=8,
            dropout_rate=0.2,
        )
    assert "must be a subclass of AttentionLayer" in str(excinfo.value)


def test_attention_config_invalid_fully_qualified_name():
    with pytest.raises(ValidationError) as excinfo:
        AttentionConfig(
            target="InvalidNameWithoutDot",
            hidden_size=512,
            num_heads=8,
            dropout_rate=0.2,
        )
    assert "must be a valid fully qualified class name" in str(excinfo.value)
