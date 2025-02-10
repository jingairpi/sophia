import sys
import types

import pytest
from pydantic import BaseModel, ValidationError

from sophia.model.builder import build_model

# -----------------------------------------------------------------------------
# Create a dummy module and dummy model classes.
# -----------------------------------------------------------------------------

dummy_module = types.ModuleType("dummy_module")


class DummyModel:
    def __init__(self, config=None):
        self.config = config

    def init_params(self, rng_key):
        return {"dummy": 42}

    def apply(self, params, inputs, **kwargs):
        return inputs


class NotRegisteredModel:
    pass


dummy_module.DummyModel = DummyModel
dummy_module.NotRegisteredModel = NotRegisteredModel

sys.modules["dummy_module"] = dummy_module

# -----------------------------------------------------------------------------
# Register DummyModel using the registry.
# -----------------------------------------------------------------------------
from sophia.model.registry import register


@register(name="dummy_module.DummyModel")
class DummyModel(DummyModel):
    pass


# -----------------------------------------------------------------------------
# Define a simple dummy configuration class.
# -----------------------------------------------------------------------------
class DummyConfig(BaseModel):
    """
    A minimal configuration that only contains the 'type' field.
    """

    type: str


# -----------------------------------------------------------------------------
# Unit Tests for the Builder.
# -----------------------------------------------------------------------------
def test_build_model_valid():
    """
    Test that the builder instantiates a valid model from a configuration whose type
    is registered in the system.
    """
    config = DummyConfig(type="dummy_module.DummyModel")
    model = build_model(config)
    from dummy_module import DummyModel

    assert isinstance(model, DummyModel)
    assert model.config is not None


def test_build_model_invalid():
    """
    Test that the builder raises an error if the configuration references a type
    that is not registered.
    """
    config = DummyConfig(type="dummy_module.NotRegisteredModel")
    with pytest.raises(ValueError) as excinfo:
        build_model(config)
    assert "not registered" in str(excinfo.value)
