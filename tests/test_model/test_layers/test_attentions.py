import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from sophia.model.layers.attentions import MultiHeadDotProductAttention


@pytest.mark.parametrize(
    "hidden_size, num_heads, seq_length, batch_size, dropout_rate",
    [
        (64, 4, 16, 8, 0.1),  # standard case
        (128, 8, 32, 4, 0.0),  # no dropout, larger input
    ],
)
def test_multihead_attention_output_shape(
    hidden_size, num_heads, seq_length, batch_size, dropout_rate
):
    """
    Tests that the MultiHeadDotProductAttention wrapper produces outputs with the correct shape.
    """
    # Create the wrapper
    # Note: hidden_size may not be strictly required by the wrapper now,
    # but it's part of the base class signature. You can ignore it or store it if needed.
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))

    # Boolean mask
    mask_bool = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    params = mha_wrapper.init(rng, input_tensor, mask_bool, deterministic=True)
    output = mha_wrapper.apply(params, input_tensor, mask_bool, deterministic=True)

    if output.shape != (batch_size, seq_length, hidden_size):
        pytest.fail(f"Unexpected output shape: {output.shape}")


def test_multihead_attention_deterministic_behavior():
    """
    Tests that MultiHeadDotProductAttention can run deterministically (no dropout).
    """
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.1,
    )

    rng = jax.random.PRNGKey(42)
    input_tensor = jax.random.normal(rng, (8, 16, 64))
    mask_bool = jnp.ones((8, 1, 16, 16), dtype=bool)

    params = mha_wrapper.init(rng, input_tensor, mask_bool, deterministic=True)
    output_1 = mha_wrapper.apply(params, input_tensor, mask_bool, deterministic=True)
    output_2 = mha_wrapper.apply(params, input_tensor, mask_bool, deterministic=True)

    # With deterministic=True, dropout is disabled, outputs should match
    if not jnp.allclose(output_1, output_2, atol=1e-6):
        pytest.fail("Outputs differ under deterministic conditions.")


def test_multihead_attention_numerical_stability():
    """
    Tests that MultiHeadDotProductAttention handles large values without NaNs/Infs.
    """
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=128, num_heads=8, dropout_rate=0.0
    )

    rng = jax.random.PRNGKey(0)
    large_input = jnp.ones((4, 16, 128)) * 1e6
    mask_bool = jnp.ones((4, 1, 16, 16), dtype=bool)

    params = mha_wrapper.init(rng, large_input, mask_bool, deterministic=True)
    output = mha_wrapper.apply(params, large_input, mask_bool, deterministic=True)

    if not jnp.isfinite(output).all():
        pytest.fail("MultiHeadDotProductAttention produced NaN or Inf values.")
