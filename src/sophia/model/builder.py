# src/sophia/model/builder.py

import importlib
import inspect
from typing import Any

from pydantic import BaseModel


def instantiate_from_config(config: BaseModel, top_level: bool = False) -> Any:
    """
    Recursively instantiate an object from a configuration object.

    If `top_level` is True, the entire configuration is passed as a single keyword
    argument (i.e. config=config) to the target class's constructor. Otherwise, the
    configuration is unpacked into keyword arguments. Additionally, before instantiation,
    the builder filters the keyword arguments so that only those accepted by the target
    class’s __init__ are passed.
    """
    # Get the target field from the config.
    target = config.target
    try:
        module_name, class_name = target.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            "The target field must be a valid fully qualified class name (module.Class)."
        )

    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ValueError(
            f"The target class '{target}' does not exist in module '{module_name}'."
        )

    # Use model_dump() (Pydantic V2) to obtain the configuration as a dict.
    kwargs = config.model_dump(exclude={"target", "expected_base_class"})

    # Recursively instantiate any nested configuration objects.
    for key, value in kwargs.items():
        if isinstance(value, BaseModel):
            kwargs[key] = instantiate_from_config(value, top_level=False)

    if top_level:
        # For a top-level configuration, pass the entire config.
        instance_kwargs = {"config": config}
    else:
        # Otherwise, use the unpacked kwargs.
        instance_kwargs = kwargs

    # **Generic Filtering Step:**
    # Remove any keys from instance_kwargs that are not accepted by cls.__init__
    sig = inspect.signature(cls.__init__)
    # Exclude 'self' from the allowed keys.
    allowed_keys = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in instance_kwargs.items() if k in allowed_keys}

    return cls(**filtered_kwargs)


def build_model(config: BaseModel) -> Any:
    """
    Build a model from the given top-level configuration.

    The built model is dynamically instantiated from the configuration and is checked to be
    a subclass of the abstract Model base class.
    """
    model = instantiate_from_config(config, top_level=True)
    from sophia.model.base import Model  # The abstract base class for models.

    if not isinstance(model, Model):
        raise ValueError("The built model is not a subclass of Model.")
    return model
