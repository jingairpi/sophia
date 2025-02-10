import pytest

from sophia.model.registry import get_layer_class, register_layer


@register_layer("DummyLayer")
class DummyLayer:
    pass


def test_registry_registration():
    # Check that "DummyLayer" is registered.
    layer_cls = get_layer_class("DummyLayer")
    assert layer_cls is DummyLayer


def test_registry_invalid_layer():
    with pytest.raises(ValueError):
        get_layer_class("NonExistentLayer")
