import importlib
from typing import ClassVar, Type

from pydantic import BaseModel, ConfigDict, field_validator


class BaseConfig(BaseModel):
    """
    BaseConfig is the root configuration class for all model components.

    This class serves as a common ancestor for all configuration models used in the system.
    Its primary purpose is to enforce that every configuration object contains a special
    'target' field, which specifies the fully qualified class name to instantiate. This
    field is used by the model builder to dynamically import and create the desired module.

    Additionally, each subclass must set the class attribute 'expected_base_class' to the
    base class that the target must inherit from.
    """

    target: str
    expected_base_class: ClassVar[Type] = None
    model_config = ConfigDict(
        populate_by_name=True,
    )

    @field_validator("target", mode="before", check_fields=False)
    @classmethod
    def check_target_is_subclass(cls, v: str) -> str:
        """
        Validate that the provided 'target' string is a valid fully qualified class name
        and that the corresponding class is a subclass of the expected base class
        defined in the subclass.
        """
        if cls.expected_base_class is None:
            raise ValueError(
                "expected_base_class is not set in subclass of BaseConfig."
            )
        try:
            module_name, class_name = v.rsplit(".", 1)
        except ValueError:
            raise ValueError(
                "The target field must be a valid fully qualified class name (module.Class)."
            )
        module = importlib.import_module(module_name)
        try:
            target_class = getattr(module, class_name)
        except AttributeError:
            raise ValueError(
                f"The target class '{v}' does not exist in module '{module_name}'."
            )
        if not issubclass(target_class, cls.expected_base_class):
            raise ValueError(
                f"The target class '{v}' must be a subclass of {cls.expected_base_class.__name__}."
            )
        return v
