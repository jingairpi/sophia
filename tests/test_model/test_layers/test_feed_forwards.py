import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from sophia.model.layers.gelu import GELUActivation
from sophia.model.layers.feed_forwards import PositionwiseFeedForward


@pytest.mark.parametrize(
    "hidden_size, ffn_multiplier, dropout_rate, seq_length, batch_size",
    [
        (64, 4, 0.1, 16, 8),  # Standard test case
        (128, 2, 0.2, 32, 4),  # Larger input and higher dropout rate
    ],
)
def test_feed_forward_output_shape(
    hidden_size, ffn_multiplier, dropout_rate, seq_length, batch_size
):
    """
    Tests that PositionwiseFeedForward produces outputs with the correct shape.
    """
    feed_forward = PositionwiseFeedForward(
        hidden_size=hidden_size,
        ffn_multiplier=ffn_multiplier,
        dropout_rate=dropout_rate,
        activation=GELUActivation(),
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    params = feed_forward.init(rng, input_tensor)

    # Apply the feed-forward layer
    output = feed_forward.apply(params, input_tensor, deterministic=True)

    # Check output shape
    if output.shape != (batch_size, seq_length, hidden_size):
        pytest.fail(f"Unexpected output shape: {output.shape}")


@pytest.mark.parametrize(
    "hidden_size, ffn_multiplier, dropout_rate, seq_length, batch_size",
    [
        (64, 4, 0.1, 16, 8),  # Standard test case
    ],
)
def test_feed_forward_activation_effect(
    hidden_size, ffn_multiplier, dropout_rate, seq_length, batch_size
):
    """
    Tests that the activation function is applied correctly in the feed-forward layer.
    """
    feed_forward = PositionwiseFeedForward(
        hidden_size=hidden_size,
        ffn_multiplier=ffn_multiplier,
        dropout_rate=dropout_rate,
        activation=GELUActivation(),
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    params = feed_forward.init(rng, input_tensor)

    # Apply the feed-forward layer
    intermediate_output = feed_forward.apply(params, input_tensor, deterministic=True)

    # Verify the activation's effect on intermediate outputs (indirect check)
    dense_output = params["params"]["Dense_1"]
    if not jnp.any(intermediate_output):
        pytest.fail("Intermediate output is incorrect after activation.")


@pytest.mark.parametrize(
    "hidden_size, ffn_multiplier, dropout_rate, seq_length, batch_size",
    [
        (64, 4, 0.5, 16, 8),  # Standard test case with high dropout rate
    ],
)
def test_feed_forward_dropout_effect(
    hidden_size, ffn_multiplier, dropout_rate, seq_length, batch_size
):
    """
    Tests that dropout is applied during training and not during inference.
    """
    feed_forward = PositionwiseFeedForward(
        hidden_size=hidden_size,
        ffn_multiplier=ffn_multiplier,
        dropout_rate=dropout_rate,
        activation=GELUActivation(),
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    params = feed_forward.init(rng, input_tensor)

    # Use RNG for dropout
    dropout_rng = jax.random.PRNGKey(1)

    # Apply the feed-forward layer with and without dropout
    training_output = feed_forward.apply(
        params, input_tensor, deterministic=False, rngs={"dropout": dropout_rng}
    )
    inference_output = feed_forward.apply(params, input_tensor, deterministic=True)

    # Outputs should differ when dropout is applied
    if jnp.allclose(training_output, inference_output):
        pytest.fail("Dropout does not appear to be applied during training mode.")
