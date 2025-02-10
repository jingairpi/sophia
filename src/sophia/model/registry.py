# Global registry for all layers building blocks.
_LAYER_REGISTRY = {}


def register_layer(name: str):
    """
    Decorator to register a layer/building block with a given name.
    """

    def decorator(cls):
        _LAYER_REGISTRY[name] = cls
        return cls

    return decorator


def get_layer_class(name: str):
    """
    Return the layer class registered under the given name.
    This function is internal to the model package.
    """
    try:
        return _LAYER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Layer type '{name}' is not registered.")
