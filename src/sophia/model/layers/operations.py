import jax.numpy as jnp
from flax import linen as nn

from sophia.model.registry import register


@register
class AddOperation(nn.Module):
    """
    A simple module that adds two tensors element-wise.
    """

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        return x + y
