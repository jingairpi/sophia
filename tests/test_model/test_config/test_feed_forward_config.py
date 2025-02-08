import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.activation import ActivationConfig
from sophia.model.config.feed_forward import FeedForwardConfig
from sophia.model.layers.bases import Activation, FeedForwardNetwork


# -----------------------------------------------------------------------------
# Create dummy FFN classes.
# -----------------------------------------------------------------------------
class DummyFeedForward(FeedForwardNetwork):
    def __call__(self, hidden_states, *args, **kwargs):
        return hidden_states


# Create a dummy class that is NOT a subclass of FeedForwardNetwork.
class NotFeedForward:
    pass


dummy_ffn_module = types.ModuleType("dummy_ffn_module")
dummy_ffn_module.DummyFeedForward = DummyFeedForward
# For the invalid target test, define NotFeedForward so that it exists.
dummy_ffn_module.NotFeedForward = NotFeedForward
sys.modules["dummy_ffn_module"] = dummy_ffn_module


# -----------------------------------------------------------------------------
# Create dummy activation classes.
# -----------------------------------------------------------------------------
class DummyActivation(Activation):
    def __call__(self, x, *args, **kwargs):
        return x


# Create a dummy class that is NOT a subclass of Activation.
class NotActivation:
    def __call__(self, x, *args, **kwargs):
        return x


dummy_activation_module = types.ModuleType("dummy_activation_module")
dummy_activation_module.DummyActivation = DummyActivation
# For the invalid activation test, define NotActivation.
dummy_activation_module.NotActivation = NotActivation
sys.modules["dummy_activation_module"] = dummy_activation_module


# -----------------------------------------------------------------------------
# Tests for FeedForwardConfig
# -----------------------------------------------------------------------------
def test_feed_forward_config_valid():
    config = FeedForwardConfig(
        target="dummy_ffn_module.DummyFeedForward",
        hidden_size=512,
        ffn_multiplier=4,
        dropout_rate=0.2,
        activation=DummyActivation(),
    )
    assert config.target == "dummy_ffn_module.DummyFeedForward"
    assert config.hidden_size == 512
    assert config.ffn_multiplier == 4
    assert callable(config.activation)
    assert isinstance(config.activation, DummyActivation)


def test_feed_forward_config_invalid_target():
    # Here, dummy_ffn_module.NotFeedForward exists but is not a subclass of FeedForwardNetwork.
    with pytest.raises(ValidationError) as excinfo:
        FeedForwardConfig(
            target="dummy_ffn_module.NotFeedForward",
            hidden_size=512,
            ffn_multiplier=4,
            dropout_rate=0.2,
            activation=DummyActivation(),
        )
    # We expect the error message to mention "must be a subclass of FeedForwardNetwork".
    assert "must be a subclass of FeedForwardNetwork" in str(excinfo.value)


def test_feed_forward_config_invalid_activation():
    # Here, we pass an activation target that is not a subclass of Activation.
    with pytest.raises(ValidationError) as excinfo:
        FeedForwardConfig(
            target="dummy_ffn_module.DummyFeedForward",
            hidden_size=512,
            ffn_multiplier=4,
            dropout_rate=0.2,
            activation=NotActivation(),
        )
    # We expect the error message to mention "must be a subclass of Activation"
    assert "must be a subclass of Activation" in str(excinfo.value)
