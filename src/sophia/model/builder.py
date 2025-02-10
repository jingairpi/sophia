from typing import Any

from pydantic import BaseModel

from sophia.model.registry import get_class


def instantiate_from_config(config: BaseModel, top_level: bool = False) -> Any:
    """
    Instantiate an object from a configuration object using the registry.

    For nested configurations, this function inspects the original BaseModel fields
    (instead of dumping to dict) and recursively instantiates any field that is a BaseModel
    with a 'type' attribute.

    If top_level is True, the entire config is passed as a single keyword argument 'config'.
    """
    cls = get_class(config.type)
    if top_level:
        return cls(config=config)

    kwargs = {}
    for field_name in config.model_fields:
        if field_name == "type":
            continue
        value = getattr(config, field_name)
        if isinstance(value, BaseModel) and hasattr(value, "type"):
            kwargs[field_name] = instantiate_from_config(value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def build_model(config: BaseModel) -> Any:
    """
    Build a model from the top-level configuration.

    This function instantiates the model using the configuration and returns the instance.
    The built model does not have to be a subclass of the abstract Model class.

    Args:
        config: A pydantic BaseModel instance (or subclass) containing the top-level model configuration.

    Returns:
        An instance of the model built from the configuration.
    """
    return instantiate_from_config(config, top_level=True)
