from typing import Callable, Dict, Optional, Type, Union

# Global registry for all building blocks.
REGISTRY: Dict[str, Type] = {}


def register(
    _cls: Optional[Type] = None, *, name: Optional[str] = None
) -> Union[Callable[[Type], Type], Type]:
    """
    Decorator to register a building block.

    If no name is provided, the class's __name__ will be used as the registration key.

    Usage:
        # Using default class name as key:
        @register
        class MyBlock:
            ...

        # Using a custom key:
        @register(name="custom_block")
        class MyOtherBlock:
            ...

    Args:
        _cls: The class to register (when the decorator is used without arguments).
        name: Optional custom name for registration.

    Returns:
        Either a decorator function or the class itself if no further decoration is needed.

    Raises:
        ValueError: If a duplicate registration is attempted.
    """

    def decorator(cls: Type) -> Type:
        reg_name = name or cls.__name__
        if reg_name in REGISTRY:
            raise ValueError(
                f"Type '{reg_name}' is already registered for class {REGISTRY[reg_name]}."
            )
        REGISTRY[reg_name] = cls
        return cls

    # If _cls is provided, the decorator is used without parameters.
    if _cls is not None:
        return decorator(_cls)
    # Otherwise, return the decorator function to be used with parameters.
    return decorator


def get_class(name: str) -> Type:
    """
    Retrieve the class registered under the given name.

    Args:
        name: The name of the registered class.

    Returns:
        The class registered under the given name.

    Raises:
        ValueError: If no class is registered under the given name.
    """
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Type '{name}' is not registered.") from e
