import pytest

from sophia.model.registry import REGISTRY, get_class, register


@pytest.fixture(autouse=True)
def clear_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


def test_register_default_name():
    """Test that a class registered without a custom name is stored under its class name."""

    @register
    class MyDefaultClass:
        pass

    assert "MyDefaultClass" in REGISTRY
    assert REGISTRY["MyDefaultClass"] is MyDefaultClass

    returned_class = get_class("MyDefaultClass")
    assert returned_class is MyDefaultClass


def test_register_custom_name():
    """Test that a class registered with a custom name is stored using that custom key."""

    @register(name="custom_key")
    class MyCustomClass:
        pass

    assert "custom_key" in REGISTRY
    assert REGISTRY["custom_key"] is MyCustomClass

    returned_class = get_class("custom_key")
    assert returned_class is MyCustomClass


def test_duplicate_registration():
    """Test that attempting to register a duplicate key raises a ValueError."""

    @register(name="dup_test")
    class FirstClass:
        pass

    with pytest.raises(ValueError) as exc_info:

        @register(name="dup_test")
        class SecondClass:
            pass

    assert "already registered" in str(exc_info.value)


def test_get_class_not_found():
    """Test that attempting to get an unregistered type raises a ValueError."""
    with pytest.raises(ValueError) as exc_info:
        get_class("non_existent")
    assert "is not registered" in str(exc_info.value)
