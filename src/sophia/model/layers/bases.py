from abc import ABC, abstractmethod
from typing import Any

from flax import linen as nn


class Activation(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def __call__(self, input_tensor: Any, *args, **kwargs) -> Any:
        """
        Apply the activation function.

        Args:
            input_tensor: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output tensor after applying the activation function.
        """


class EmbeddingLayer(nn.Module, ABC):
    """
    Abstract base class for embedding layers.
    """

    @abstractmethod
    def __call__(self, input_ids: Any, *args, **kwargs) -> Any:
        """
        Compute embeddings for input token IDs.

        Args:
            input_ids: Tensor of token IDs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Embedding representations for the input tokens.
        """


class NormalizationLayer(nn.Module, ABC):
    """
    Abstract base class for normalization layers.
    """

    @abstractmethod
    def __call__(self, input_tensor: Any, *args, **kwargs) -> Any:
        """
        Apply normalization to the input tensor.

        Args:
            input_tensor: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized tensor.
        """


class AttentionLayer(nn.Module, ABC):
    """
    Abstract base class for attention layers.
    """

    hidden_size: int
    num_heads: int
    dropout_rate: float

    @abstractmethod
    def __call__(
        self,
        hidden_states: Any,
        *args,
        attention_mask: Any = None,
        deterministic: bool = False,
        **kwargs,
    ) -> Any:
        """
        Perform attention computation.

        Args:
            hidden_states: Input tensor of hidden states.
            *args: Additional positional arguments.
            attention_mask: Optional mask for attention computation.
            deterministic: Whether to disable dropout (e.g., during inference).
            **kwargs: Additional keyword arguments.

        Returns:
            Updated tensor after applying the attention mechanism.
        """


class FeedForwardNetwork(nn.Module, ABC):
    """
    Abstract base class for feed-forward networks.
    """

    hidden_size: int
    ffn_multiplier: int
    dropout_rate: float

    @abstractmethod
    def __call__(
        self, hidden_states: Any, *args, deterministic: bool = False, **kwargs
    ) -> Any:
        """
        Compute the feed-forward network output.

        Args:
            hidden_states: Input tensor of hidden states.
            *args: Additional positional arguments.
            deterministic: Whether to disable dropout (e.g., during inference).
            **kwargs: Additional keyword arguments.

        Returns:
            Updated tensor after passing through the feed-forward network.
        """
