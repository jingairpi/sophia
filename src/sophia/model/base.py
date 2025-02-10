from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp


class Model(ABC):
    """
    Abstract base class for models.
    Provides methods to initialize parameters and apply the model.
    """

    @abstractmethod
    def init(self, rng_key: Any, sample_input: jnp.ndarray) -> Any:
        """
        Initialize and return the model parameters.

        Args:
            rng_key: A random key for parameter initialization.
            sample_input: A representative input that defines the shape
                          Flax needs for building the parameter structure.

        Returns:
            A dictionary (or nested structure) containing the initialized
            model parameters.
        """
