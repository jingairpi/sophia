from abc import ABC, abstractmethod

from flax import linen as nn


class TransformerBlockBase(nn.Module, ABC):
    """
    Abstract base class for transformer blocks.

    This class defines the interface for transformer block implementations.
    All transformer block classes should subclass TransformerBlockBase and implement
    the __call__ method.

    Attributes:
        (Any common attributes for transformer blocks can be defined here.)
    """

    @abstractmethod
    def __call__(self, hidden_states, deterministic: bool = True, **kwargs):
        """
        Apply the transformer block to the input hidden states.

        Args:
            hidden_states: A JAX array of shape [batch_size, seq_length, hidden_size].
            deterministic: If True, disables dropout and other stochastic operations.
            **kwargs: Additional keyword arguments.

        Returns:
            A JAX array of the same shape as hidden_states.
        """
