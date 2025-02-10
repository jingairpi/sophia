import pytest

from sophia.model.registry import get_layer_class, register_layer


@register_layer("FooLayer")
class FooLayer:
    pass


def test_registry_registration():
    layer_cls = get_layer_class("FooLayer")
    assert layer_cls is FooLayer


def test_registry_invalid_layer():
    with pytest.raises(ValueError):
        get_layer_class("NonExistentLayer")
