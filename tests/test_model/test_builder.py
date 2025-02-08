import sys
import types
from typing import ClassVar, Type

import pytest
from pydantic import ValidationError

from sophia.model.base import Model
from sophia.model.builder import build_model
from sophia.model.config.base import BaseConfig

# -----------------------------------------------------------------------------
# Create a dummy module and dummy model classes.
# -----------------------------------------------------------------------------

# Create a dummy module for models.
dummy_model_module = types.ModuleType("dummy_model_module")


# Define a dummy model class that is a subclass of Model.
class DummyModel(Model):
    def init_params(self, rng_key):
        return {"dummy_param": 123}

    def apply(self, params, inputs, **kwargs):
        return inputs


dummy_model_module.DummyModel = DummyModel


# Define a dummy class that is NOT a subclass of Model.
class NotAModel:
    pass


dummy_model_module.NotAModel = NotAModel

# Register the dummy module.
sys.modules["dummy_model_module"] = dummy_model_module


# -----------------------------------------------------------------------------
# Define a dummy configuration class for models.
# -----------------------------------------------------------------------------
# We create a DummyModelConfig that inherits from BaseConfig.
# IMPORTANT: expected_base_class must be defined as a ClassVar in this subclass.
class DummyModelConfig(BaseConfig):
    """
    Configuration for a dummy model.

    This configuration is used to test the builder. It requires only a 'target'
    field. The expected_base_class is set to Model.
    """

    # Declare as a ClassVar so that it is not treated as an instance field.
    expected_base_class: ClassVar[Type[Model]] = Model
    # No additional instance fields.


# -----------------------------------------------------------------------------
# Unit Tests for the Builder.
# -----------------------------------------------------------------------------
def test_build_model_valid():
    """
    Test that the builder instantiates a valid model from the configuration.
    """
    # Create a valid configuration that points to DummyModel.
    config = DummyModelConfig(target="dummy_model_module.DummyModel")
    model = build_model(config)
    # Check that the built model is an instance of Model and DummyModel.
    assert isinstance(model, Model)
    from dummy_model_module import DummyModel  # or use DummyModel directly

    assert isinstance(model, DummyModel)


def test_build_model_invalid():
    """
    Test that the builder raises a ValidationError if the configuration is invalid,
    i.e. if the target is not a subclass of Model.
    """
    with pytest.raises(ValidationError) as excinfo:
        # Attempt to create a configuration with an invalid target.
        DummyModelConfig(target="dummy_model_module.NotAModel")
    # The error message should indicate that the target must be a subclass of Model.
    assert "must be a subclass of Model" in str(excinfo.value)
