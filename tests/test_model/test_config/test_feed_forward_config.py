import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.feed_forward import FeedForwardConfig
from sophia.model.layers.bases import Activation, FeedForwardNetwork


# -----------------------------------------------------------------------------
# Define dummy classes at the module level.
# -----------------------------------------------------------------------------
class DummyFeedForward(FeedForwardNetwork):
    def __call__(self, hidden_states, *args, **kwargs):
        return hidden_states


class NotFeedForward:
    pass


class DummyActivation(Activation):
    def __call__(self, x, *args, **kwargs):
        return x


class NotActivation:
    def __call__(self, x, *args, **kwargs):
        return x


# -----------------------------------------------------------------------------
# Fixtures to insert dummy modules into sys.modules.
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def setup_dummy_ffn_module(monkeypatch):
    dummy_ffn_mod = types.ModuleType("dummy_ffn_module")
    dummy_ffn_mod.DummyFeedForward = DummyFeedForward
    dummy_ffn_mod.NotFeedForward = NotFeedForward
    monkeypatch.setitem(sys.modules, "dummy_ffn_module", dummy_ffn_mod)
    yield


@pytest.fixture(autouse=True)
def setup_dummy_activation_module(monkeypatch):
    dummy_activation_mod = types.ModuleType("dummy_activation_module")
    dummy_activation_mod.DummyActivation = DummyActivation
    dummy_activation_mod.NotActivation = NotActivation
    monkeypatch.setitem(sys.modules, "dummy_activation_module", dummy_activation_mod)
    yield


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
    # Check that the configuration stores the correct values.
    assert config.target == "dummy_ffn_module.DummyFeedForward"
    assert config.hidden_size == 512
    assert config.ffn_multiplier == 4
    assert callable(config.activation)
    assert isinstance(config.activation, DummyActivation)


def test_feed_forward_config_invalid_target():
    # Here, "dummy_ffn_module.NotFeedForward" exists but is not a subclass of FeedForwardNetwork.
    with pytest.raises(ValidationError) as excinfo:
        FeedForwardConfig(
            target="dummy_ffn_module.NotFeedForward",
            hidden_size=512,
            ffn_multiplier=4,
            dropout_rate=0.2,
            activation=DummyActivation(),
        )
    assert "must be a subclass of FeedForwardNetwork" in str(excinfo.value)


def test_feed_forward_config_invalid_activation():
    # Here, we pass an activation target (as a string) that is not a subclass of Activation.
    with pytest.raises(ValidationError) as excinfo:
        FeedForwardConfig(
            target="dummy_ffn_module.DummyFeedForward",
            hidden_size=512,
            ffn_multiplier=4,
            dropout_rate=0.2,
            activation=NotActivation(),
        )
    assert "must be a subclass of Activation" in str(excinfo.value)
