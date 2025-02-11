import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def load_binary_data(
    file_path: str, memmap: bool = False, dtype: np.dtype = np.int32
) -> np.ndarray:
    """
    Reads the tokenized dataset from a binary file.

    Args:
        file_path (str): The path to the binary file.
        memmap (bool): If True, uses memory mapping for large files.
        dtype (np.dtype): The data type to use (e.g., np.int32). Parameterized for flexibility.

    Returns:
        np.ndarray: A numpy array of token IDs.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an error occurs during file reading.
    """
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        if memmap:
            data = np.memmap(file_path, mode="r", dtype=dtype)
            data = np.array(data)
        else:
            with open(file_path, "rb") as f:
                data = np.fromfile(f, dtype=dtype)
    except Exception as e:
        logger.exception("Error loading data from %s", file_path)
        raise IOError(f"Error loading data from {file_path}") from e

    logger.info("Loaded %d tokens from %s", data.size, file_path)
    return data


class DataLoaderBase(ABC):
    """
    Abstract base class for data loaders.
    Provides a generic interface for iterating over batches.
    """

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        Create an iterator over data batches.

        Returns:
            An iterator over data batches.
        """
        pass

    @abstractmethod
    def __next__(self) -> Any:
        """
        Retrieve the next data batch.

        Returns:
            The next batch of data.
        """
        pass


class DataLoader(DataLoaderBase):
    """
    Production-grade DataLoader for tokenized text data.

    Reads data from a binary file and yields (inputs, targets) batches for next-token prediction.
    One epoch corresponds to a complete pass through the data.
    Supports options for shuffling and memory mapping.

    Attributes:
        file_path (str): Path to the binary file.
        batch_size (int): Number of sequences per batch.
        seq_len (int): Total tokens per sequence (including both inputs and targets).
        repeat (bool): If True, cycles indefinitely over the data.
        shuffle (bool): If True, shuffles data at the beginning of each epoch.
        memmap (bool): If True, uses memory mapping for reading the file.
        dtype (np.dtype): Data type of the tokens.
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        seq_len: int,
        repeat: bool = False,
        shuffle: bool = False,
        memmap: bool = False,
        dtype: np.dtype = np.int32,
    ):
        """
        Initializes the DataLoader.

        Args:
            file_path (str): Path to the binary file with tokenized data.
            batch_size (int): Number of sequences per batch.
            seq_len (int): Sequence length (total tokens per sequence, including inputs and targets).
            repeat (bool): If True, cycles indefinitely.
            shuffle (bool): If True, shuffles the data at the start of each epoch.
            memmap (bool): If True, uses memory mapping.
            dtype (np.dtype): Data type of the tokens (e.g., np.int32).
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.repeat = repeat
        self.shuffle = shuffle
        self.memmap = memmap
        self.dtype = dtype

        self.data = load_binary_data(file_path, memmap=memmap, dtype=dtype)
        self.total_tokens = len(self.data)
        self.tokens_per_batch = batch_size * seq_len
        self.num_batches = self.total_tokens // self.tokens_per_batch

        if self.num_batches < 1:
            raise ValueError(
                "Not enough data for one batch. Check file, batch_size, and seq_len."
            )
        self.current_batch = 0
        logger.info("DataLoader configured for %d full batches.", self.num_batches)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns an iterator over data batches.
        If shuffling is enabled, shuffles the data at the start of the epoch.
        """
        self.current_batch = 0
        if self.shuffle:
            logger.info("Shuffling data for new epoch.")
            self.data = np.random.permutation(self.data)
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the next data batch as a tuple (inputs, targets).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Inputs and targets for next-token prediction.

        Raises:
            StopIteration: When all full batches have been processed and repeat is False.
        """
        if self.current_batch >= self.num_batches:
            if self.repeat:
                self.reset()
            else:
                raise StopIteration

        start = self.current_batch * self.tokens_per_batch
        end = start + self.tokens_per_batch
        batch = self.data[start:end]
        self.current_batch += 1

        try:
            batch = batch.reshape(self.batch_size, self.seq_len).astype(np.int32)
        except Exception as e:
            logger.exception("Error reshaping batch from index %d to %d", start, end)
            raise ValueError(
                "Error reshaping batch. Verify batch_size and seq_len."
            ) from e

        # For next-token prediction:
        #   Inputs: All tokens except the last token in each sequence.
        #   Targets: All tokens except the first token in each sequence.
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        return inputs, targets

    def reset(self) -> None:
        """
        Resets the DataLoader to start from the beginning of the dataset.
        If shuffling is enabled, re-shuffles the data.
        """
        logger.info("Resetting DataLoader for a new epoch.")
        self.current_batch = 0
        if self.shuffle:
            self.data = np.random.permutation(self.data)
