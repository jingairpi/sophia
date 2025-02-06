import jax.numpy as jnp
from flax import linen as nn

from .bases import EmbeddingLayer


class TokenEmbedding(EmbeddingLayer):
    """
    Implements token embeddings for a neural network model.

    Token embeddings are used to map discrete input tokens (e.g., words,
    subwords) to dense vector representations that can be processed by the
    model.

    Attributes:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        hidden_size (int): The dimensionality of the token embeddings.
    """

    vocab_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, input_ids, *args, **kwargs):
        """
        Applies the token embedding to the input token IDs.

        Args:
            input_ids: An array of token IDs, typically of shape [batch_size,
                       seq_length].
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            An array of embedded token vectors, typically of shape
            [batch_size, seq_length, hidden_size].
        """
        if input_ids.size == 0:
            raise ValueError("Input IDs cannot be empty.")
        embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
        )
        return embedding(input_ids)


class PositionalEmbedding(EmbeddingLayer):
    """
    Implements positional embeddings for a neural network model.

    Positional embeddings provide the model with information about the order of
    tokens in a sequence, which is critical for sequence modeling in
    transformer architectures.

    Attributes:
        max_seq_length (int): The maximum sequence length the model supports.
        hidden_size (int): The dimensionality of the positional embeddings.
    """

    max_seq_length: int
    hidden_size: int

    @nn.compact
    def __call__(self, position_ids, *args, **kwargs):
        """
        Applies the positional embedding to the input position IDs.

        Args:
            position_ids: An array of position IDs, typically of shape
                          [seq_length].
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            An array of positional embeddings, typically of shape
            [seq_length, hidden_size].
        """
        embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_length, self.hidden_size),
        )
        if jnp.any(position_ids >= self.max_seq_length) or jnp.any(position_ids < 0):
            raise IndexError("Position IDs are out of bounds.")
        return embedding[position_ids]
