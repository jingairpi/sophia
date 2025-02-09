import math

import jax
import jax.numpy as jnp

from sophia.model.layers.bases import Activation
from sophia.model.registry import register_layer


@register_layer("GELUActivation")
class GELUActivation(Activation):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation function is defined as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is the approximate version of GELU, which is computationally more
    efficient than the exact formulation involving the error function (erf).

    GELU is used in transformer architectures like GPT-2 due to its smoothness
    and improved gradient flow compared to ReLU or other activations.

    Reference:
    Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    arXiv:1606.08415 [cs.LG]. https://arxiv.org/abs/1606.08415
    """

    def __call__(self, input_tensor, *args, **kwargs):
        """
        Apply the GELU activation function to the input tensor.

        Args:
            input_tensor: The input tensor.

        Returns:
            The tensor after applying GELU activation.
        """
        return (
            0.5
            * input_tensor
            * (
                1.0
                + jnp.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input_tensor + 0.044715 * jnp.power(input_tensor, 3))
                )
            )
        )
