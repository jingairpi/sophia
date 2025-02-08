import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.normalization import NormalizationConfig
from sophia.model.layers.bases import NormalizationLayer


# -----------------------------------------------------------------------------
# Fixture: Set up dummy normalization module.
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def setup_dummy_norm(monkeypatch):
    dummy_norm_module = types.ModuleType("dummy_norm_module")

    class DummyNorm(NormalizationLayer):
        def __call__(self, x, *args, **kwargs):
            return x

    class NotNorm:
        pass

    dummy_norm_module.DummyNorm = DummyNorm
    dummy_norm_module.NotNorm = NotNorm
    monkeypatch.setitem(sys.modules, "dummy_norm_module", dummy_norm_module)
    yield


# -----------------------------------------------------------------------------
# Tests for NormalizationConfig
# -----------------------------------------------------------------------------
def test_norm_config_valid():
    config = NormalizationConfig(target="dummy_norm_module.DummyNorm", epsilon=1e-5)
    assert config.target == "dummy_norm_module.DummyNorm"
    assert config.epsilon == 1e-5


def test_norm_config_invalid_target():
    with pytest.raises(ValidationError) as excinfo:
        NormalizationConfig(target="dummy_norm_module.NotNorm", epsilon=1e-5)
    assert "must be a subclass of NormalizationLayer" in str(excinfo.value)


def test_norm_config_invalid_fqname():
    with pytest.raises(ValidationError) as excinfo:
        NormalizationConfig(target="InvalidNameWithoutDot", epsilon=1e-5)
    assert "must be a valid fully qualified class name" in str(excinfo.value)
