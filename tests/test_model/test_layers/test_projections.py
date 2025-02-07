import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from sophia.model.layers.projections import OutputProjection


def test_output_shape():
    """
    Test that the output projection layer produces an output
    with the correct shape.
    """
    hidden_size = 64
    output_size = 100  # For example, vocab_size or any target dimension.
    batch_size = 4
    seq_length = 10

    # Create a dummy input tensor of shape [B, T, H]
    x = jnp.ones((batch_size, seq_length, hidden_size))

    # Instantiate the projection module.
    projection = OutputProjection(hidden_size=hidden_size, output_size=output_size)

    # Initialize the module's parameters.
    rng = jax.random.PRNGKey(0)
    variables = projection.init(rng, x)

    # Apply the projection.
    output = projection.apply(variables, x)

    # Verify the output shape.
    expected_shape = (batch_size, seq_length, output_size)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"


def test_empty_input_raises_error():
    """
    Test that passing an empty input tensor raises a ValueError.
    """
    hidden_size = 64
    output_size = 100

    # Create an empty input tensor with zero elements.
    # For example, a tensor with shape [0, seq_length, hidden_size]
    x_empty = jnp.empty((0, 10, hidden_size))

    projection = OutputProjection(hidden_size=hidden_size, output_size=output_size)
    rng = jax.random.PRNGKey(0)

    # Since the error is raised during the module's __call__ (which is invoked during init),
    # we wrap the init call in pytest.raises.
    with pytest.raises(ValueError, match="Hidden states cannot be empty."):
        # This should raise the error as defined in __call__.
        projection.init(rng, x_empty)


def test_gradient_computation():
    """
    Test that gradients computed through the output projection layer
    are finite.
    """
    hidden_size = 64
    output_size = 100
    batch_size = 4
    seq_length = 10

    # Create a random input tensor.
    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, hidden_size))

    projection = OutputProjection(hidden_size=hidden_size, output_size=output_size)
    rng = jax.random.PRNGKey(2)
    variables = projection.init(rng, x)

    # Define a simple loss function.
    def loss_fn(inputs):
        output = projection.apply(variables, inputs)
        return jnp.mean(output)

    # Compute gradients with respect to the input.
    grads = jax.grad(loss_fn)(x)

    # Verify that all gradient values are finite.
    assert jnp.all(jnp.isfinite(grads)), "Gradient contains non-finite values."


def test_deterministic_output():
    """
    Test that the output projection layer produces deterministic results
    when applied repeatedly with the same parameters.
    """
    hidden_size = 64
    output_size = 100
    batch_size = 4
    seq_length = 10

    x = jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_length, hidden_size))
    projection = OutputProjection(hidden_size=hidden_size, output_size=output_size)
    rng = jax.random.PRNGKey(4)
    variables = projection.init(rng, x)

    # Call the projection twice with the same inputs and parameters.
    output1 = projection.apply(variables, x)
    output2 = projection.apply(variables, x)

    # The outputs should be identical.
    assert jnp.allclose(
        output1, output2, atol=1e-6
    ), "Outputs differ between calls under deterministic conditions."
