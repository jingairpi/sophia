from abc import ABC, abstractmethod
from typing import Any, Iterator


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    Provides methods to iterate over batches of data.
    """

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        Create an iterator over the data batches.

        Returns:
            An iterator over data batches.
        """

    @abstractmethod
    def __next__(self) -> Any:
        """
        Retrieve the next data batch.

        Returns:
            The next batch of data.
        """
