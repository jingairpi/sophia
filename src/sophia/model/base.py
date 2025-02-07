from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """
    Abstract base class for models.
    Provides methods to initialize parameters and apply the model.
    """

    @abstractmethod
    def init_params(self, rng_key: Any) -> Any:
        """
        Initialize and return the model parameters.

        Args:
            rng_key: A random key for parameter initialization.

        Returns:
            Initialized model parameters.
        """

    @abstractmethod
    def apply(self, params: Any, inputs: Any, **kwargs) -> Any:
        """
        Apply the model to the given inputs using the specified parameters.

        Args:
            params: Model parameters.
            inputs: Input data.
            **kwargs: Additional arguments.

        Returns:
            Model outputs.
        """
