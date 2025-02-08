import importlib
from typing import Any

from pydantic import BaseModel


def instantiate_from_config(config: BaseModel) -> Any:
    """
    Recursively instantiate an object from a configuration object.

    The configuration object must have a 'target' field that contains the fully
    qualified class name to instantiate. This function uses that field to import
    and instantiate the target class, passing any additional fields as keyword arguments.

    Args:
        config: A Pydantic configuration object.

    Returns:
        An instance of the target class, with nested configuration objects recursively instantiated.
    """
    # Get the target class name from the config.
    target = config.target
    try:
        module_name, class_name = target.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            "The target field must be a valid fully qualified class name (module.Class)."
        )

    # Dynamically import the module and get the class.
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ValueError(
            f"The target class '{target}' does not exist in module '{module_name}'."
        )

    # Convert the config into a dictionary of keyword arguments.
    # Exclude 'target' and 'expected_base_class' since they are used only for validation.
    kwargs = config.dict(exclude={"target", "expected_base_class"}, by_alias=True)

    # Recursively instantiate any nested configuration objects.
    for key, value in kwargs.items():
        if isinstance(value, BaseModel):
            kwargs[key] = instantiate_from_config(value)

    return cls(**kwargs)


def build_model(config: BaseModel) -> Any:
    """
    Build a model from the given configuration. The built model is dynamically
    instantiated from the configuration and is checked to be a subclass of the abstract
    Model base class.

    Args:
        config: A Pydantic configuration object for the model.

    Returns:
        An instance of the built model.

    Raises:
        ValueError: If the instantiated model is not a subclass of Model.
    """
    # Instantiate the model from the configuration.
    model = instantiate_from_config(config)

    # Import the abstract base class for models.
    from sophia.model.base import Model

    if not isinstance(model, Model):
        raise ValueError("The built model is not a subclass of Model.")

    return model
