from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """
    BaseConfig is the root configuration class for all model components.

    This class serves as a common ancestor for all configuration models used in the system.
    Its primary purpose is to enforce that every configuration object contains a special
    'target' field, which specifies the fully qualified class name to instantiate. This
    field is used by the model builder to dynamically import and create the desired module.
    """

    target: str

    model_config = ConfigDict(
        populate_by_name=True,
    )
