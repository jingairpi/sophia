import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.activation import ActivationConfig
from sophia.model.layers.bases import Activation


# -----------------------------------------------------------------------------
# Create a dummy subclass of Activation for testing.
# -----------------------------------------------------------------------------
class DummyActivation(Activation):
    def __call__(self, x, *args, **kwargs):
        return x


# Also, create a dummy class that is NOT a subclass of Activation.
class NotActivation:
    pass


# Create a dummy module and add both DummyActivation and NotActivation.
dummy_activation_module = types.ModuleType("dummy_activation_module")
dummy_activation_module.DummyActivation = DummyActivation
dummy_activation_module.NotActivation = NotActivation
sys.modules["dummy_activation_module"] = dummy_activation_module


# -----------------------------------------------------------------------------
# Tests for ActivationConfig
# -----------------------------------------------------------------------------
def test_activation_config_valid():
    config = ActivationConfig(target="dummy_activation_module.DummyActivation")
    assert config.target == "dummy_activation_module.DummyActivation"


def test_activation_config_invalid_target_not_subclass():
    with pytest.raises(ValidationError) as excinfo:
        ActivationConfig(target="dummy_activation_module.NotActivation")
    assert "must be a subclass of Activation" in str(excinfo.value)


def test_activation_config_invalid_fqname():
    with pytest.raises(ValidationError) as excinfo:
        ActivationConfig(target="InvalidNameWithoutDot")
    assert "must be a valid fully qualified class name" in str(excinfo.value)
