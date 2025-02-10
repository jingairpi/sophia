from flax import linen as nn

from sophia.model.layers.bases import Activation, FeedForwardNetwork
from sophia.model.registry import register


@register
class PositionwiseFeedForward(FeedForwardNetwork):
    """
    Implements the Positionwise Feed-Forward layer, commonly used in Transformer models.

    This layer applies two dense (fully connected) layers with an intermediate activation function
    and dropout for regularization. The key feature of the Positionwise Feed-Forward layer is that
    it processes each position in the sequence independently, without considering neighboring positions.

    Attributes:
        hidden_size (int): The hidden size of the input.
        ffn_multiplier (int): The expansion factor for the intermediate layer.
        activation (Callable): The activation function to apply between the dense layers.
        dropout_rate (float): The dropout rate applied within the feed-forward network.
    """

    # hidden_size: int
    # ffn_multiplier: int
    activation: Activation
    # dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, hidden_states, deterministic=False):
        """
        Applies the Positionwise Feed-Forward transformation to the input.

        Args:
            hidden_states (jax.numpy.ndarray): The input tensor of shape [batch_size, seq_length, hidden_size].
            deterministic (bool): If True, disables dropout (useful during inference).

        Returns:
            jax.numpy.ndarray: The output tensor of shape [batch_size, seq_length, hidden_size].

        Steps:
            1. Apply the first dense layer (`dense_1`) to expand the dimensionality.
            2. Apply the activation function.
            3. Apply the second dense layer (`dense_2`) to reduce the dimensionality.
            4. Apply dropout for regularization (skipped if `deterministic=True`).
        """
        dense_1 = nn.Dense(self.hidden_size * self.ffn_multiplier)
        dense_2 = nn.Dense(self.hidden_size)
        dropout = nn.Dropout(rate=self.dropout_rate)

        x = dense_1(hidden_states)
        x = self.activation(x)
        x = dense_2(x)
        x = dropout(x, deterministic=deterministic)

        return x
