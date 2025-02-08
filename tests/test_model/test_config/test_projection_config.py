import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.projection import OutputProjectionConfig
from sophia.model.layers.bases import ProjectionLayer


# Create dummy projection classes.
class DummyProjection(ProjectionLayer):
    def __call__(self, hidden_states, *args, **kwargs):
        return hidden_states


class NotProjection:
    pass


dummy_projection_module = types.ModuleType("dummy_projection_module")
dummy_projection_module.DummyProjection = DummyProjection
dummy_projection_module.NotProjection = NotProjection
sys.modules["dummy_projection_module"] = dummy_projection_module


def test_output_projection_config_valid():
    config = OutputProjectionConfig(
        target="dummy_projection_module.DummyProjection",
        hidden_size=512,
        output_size=10000,
    )
    assert config.target == "dummy_projection_module.DummyProjection"
    assert config.hidden_size == 512
    assert config.output_size == 10000


def test_output_projection_config_invalid_target():
    with pytest.raises(ValidationError) as excinfo:
        OutputProjectionConfig(
            target="dummy_projection_module.NotProjection",
            hidden_size=512,
            output_size=10000,
        )
    assert "must be a subclass of ProjectionLayer" in str(excinfo.value)


def test_output_projection_config_invalid_fqname():
    with pytest.raises(ValidationError) as excinfo:
        OutputProjectionConfig(
            target="InvalidNameWithoutDot", hidden_size=512, output_size=10000
        )
    assert "must be a valid fully qualified class name" in str(excinfo.value)
