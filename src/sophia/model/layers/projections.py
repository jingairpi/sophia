import jax.numpy as jnp
from flax import linen as nn

from sophia.model.layers.bases import ProjectionLayer


class OutputProjection(ProjectionLayer):
    """
    A generic output projection layer that maps hidden representations
    to an output space (e.g., logits over a vocabulary or any target space).

    Attributes:
        hidden_size (int): Dimensionality of the input hidden states.
        output_size (int): Dimensionality of the output space.
    """

    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, hidden_states, *args, **kwargs):
        if hidden_states.size == 0:
            raise ValueError("Hidden states cannot be empty.")

        dense_proj = nn.Dense(
            features=self.output_size, kernel_init=nn.initializers.normal(stddev=0.02)
        )
        outputs = dense_proj(hidden_states)
        return outputs
