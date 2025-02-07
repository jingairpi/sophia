import importlib

from pydantic import field_validator

from sophia.model.config.base import BaseConfig
from sophia.model.layers.bases import AttentionLayer


class AttentionConfig(BaseConfig):
    """
    Configuration for an attention module.

    The 'target' field should be a fully qualified class name whose class is a subclass
    of `AttentionLayer`. This field must be provided in the configuration (or overridden)
    to specify different attention implementations.
    """

    target: str
    hidden_size: int
    num_heads: int
    dropout_rate: float = 0.1

    @field_validator("target", mode="before", check_fields=False)
    @classmethod
    def check_target_is_subclass(cls, v: str) -> str:
        """
        Validate that the provided 'target' string is a valid fully qualified class name and
        that the corresponding class is a subclass of `AttentionLayer`.
        """
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

        if not issubclass(target_class, AttentionLayer):
            raise ValueError(
                f"The target class '{v}' must be a subclass of `AttentionLayer`."
            )
        return v
