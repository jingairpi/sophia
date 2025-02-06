from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """
    Abstract base class for trainers.
    Provides methods for performing training steps and executing training
    loops.
    """

    @abstractmethod
    def train_step(self, params: Any, batch: Any, **kwargs) -> Any:
        """
        Perform a single training step.

        Args:
            params: Model parameters.
            batch: A batch of training data.
            **kwargs: Additional arguments.

        Returns:
            Updated model parameters and any relevant metrics.
        """

    @abstractmethod
    def train(self, params: Any, data_loader: Any, **kwargs) -> Any:
        """
        Execute the training loop over the dataset.

        Args:
            params: Model parameters.
            data_loader: An iterable data loader.
            **kwargs: Additional arguments.

        Returns:
            Trained model parameters.
        """
