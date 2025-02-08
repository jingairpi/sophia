import jax.nn

from sophia.model.layers.bases import Activation


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
        return jax.nn.gelu(input_tensor)
