import jax.numpy as jnp
from flax import linen as nn

from sophia.model.layers.bases import NormalizationLayer
from sophia.model.registry import register


@register
class LayerNormalization(NormalizationLayer):
    """
    Implements Layer Normalization, which normalizes inputs across features.

    Layer Normalization subtracts the mean and divides by the standard
    deviation of the features for each input sample. This ensures that the
    input is centered and scaled, improving gradient flow during training.

    Attributes:
        epsilon (float): A small constant added to the denominator for
        numerical stability.
    """

    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x: Input tensor of shape [..., features].
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized tensor of the same shape as the input.
        """
        # Remove 'deterministic' from kwargs if it exists
        kwargs.pop("deterministic", None)

        layernorm = nn.LayerNorm(epsilon=self.epsilon)
        return layernorm(x, *args, **kwargs)


@register
class RMSNormalization(NormalizationLayer):
    """
    Implements Root Mean Square (RMS) Normalization.

    RMS Normalization normalizes inputs by their root mean square (RMS) value,
    and applies a learnable scaling factor. Unlike Layer Normalization, RMSNorm
    does not subtract the mean of the features.

    Attributes:
        features (int): The number of features in the input tensor.
        epsilon (float): A small constant added to the denominator for
        numerical stability.
    """

    features: int
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x: Input tensor of shape [..., features].
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized tensor of the same shape as the input.
        """
        scale = self.param("scale", nn.initializers.ones, (self.features,))
        mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_square + self.epsilon)
        x_norm = x / rms
        return x_norm * scale
