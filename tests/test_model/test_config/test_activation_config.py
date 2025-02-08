import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.activation import ActivationConfig
from sophia.model.layers.bases import Activation


# -----------------------------------------------------------------------------
# Fixture: Set up a dummy activation module for each test.
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def setup_dummy_activation(monkeypatch):
    # Create a dummy module for activation.
    dummy_activation_module = types.ModuleType("dummy_activation_module")

    # Define a dummy subclass of Activation.
    class DummyActivation(Activation):
        def __call__(self, x, *args, **kwargs):
            return x

    # Define a class that is not a subclass of Activation.
    class NotActivation:
        pass

    dummy_activation_module.DummyActivation = DummyActivation
    dummy_activation_module.NotActivation = NotActivation

    # Insert (or override) the dummy module in sys.modules.
    monkeypatch.setitem(sys.modules, "dummy_activation_module", dummy_activation_module)
    yield
    # (Monkeypatch will automatically undo changes after the test.)


# -----------------------------------------------------------------------------
# Tests for ActivationConfig
# -----------------------------------------------------------------------------
def test_activation_config_valid():
    # Here, we pass the fully qualified name as a string.
    config = ActivationConfig(target="dummy_activation_module.DummyActivation")
    # The validator should import the DummyActivation class and check that it is a subclass of Activation.
    # We can check that the validated target is not None.
    assert config.target is not None
    # Optionally, if your validator converts the string to a class, you might check:
    # from sophia.model.layers.bases import Activation
    # assert issubclass(config.target, Activation)


def test_activation_config_invalid_target_not_subclass():
    with pytest.raises(ValidationError) as excinfo:
        ActivationConfig(target="dummy_activation_module.NotActivation")
    assert "must be a subclass of Activation" in str(excinfo.value)


def test_activation_config_invalid_fqname():
    with pytest.raises(ValidationError) as excinfo:
        ActivationConfig(target="InvalidNameWithoutDot")
    assert "must be a valid fully qualified class name" in str(excinfo.value)
