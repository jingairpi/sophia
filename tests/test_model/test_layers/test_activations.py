import jax.numpy as jnp
import pytest

from sophia.model.layers.activations import GELUActivation


def test_gelu_activation_output_shape():
    """
    Tests that GELUActivation produces an output with the same shape as the
    input.
    """
    gelu = GELUActivation()
    input_tensor = jnp.array([[1.0, -1.0, 0.5], [0.0, -0.5, 2.0]])

    output_tensor = gelu(input_tensor)

    # Check that the output shape matches the input shape
    if output_tensor.shape != input_tensor.shape:
        pytest.fail(f"Unexpected output shape: {output_tensor.shape}")


def test_gelu_activation_output_values():
    """
    Tests that GELUActivation produces correct values for known inputs.
    """
    gelu = GELUActivation()

    # Known input and output values for GELU
    input_tensor = jnp.array([-1.0, 0.0, 1.0])
    expected_output = jnp.array(
        [-0.15880799, 0.0, 0.841192]
    )  # Derived from the GELU formula.

    output_tensor = gelu(input_tensor)

    # Use a tolerance for floating-point comparison
    if not jnp.allclose(output_tensor, expected_output, atol=1e-5):
        pytest.fail(f"Unexpected output values: {output_tensor}")


def test_gelu_activation_zero_input():
    """
    Tests that GELUActivation produces 0 for an input tensor of zeros.
    """
    gelu = GELUActivation()
    input_tensor = jnp.zeros((3, 3))

    output_tensor = gelu(input_tensor)

    # Output should be all zeros
    if not jnp.allclose(output_tensor, jnp.zeros_like(input_tensor)):
        pytest.fail(f"Unexpected output for zeros: {output_tensor}")


@pytest.mark.parametrize(
    "input_tensor",
    [
        jnp.array([0.0, -0.5, 1.5]),
        jnp.array([[1.0, -1.0], [0.5, -0.5]]),
        jnp.zeros((4, 4)),
    ],
)
def test_gelu_activation_batch_input(input_tensor):
    """
    Tests that GELUActivation works for various batch inputs.
    """
    gelu = GELUActivation()

    output_tensor = gelu(input_tensor)

    # Output should have the same shape as input
    if output_tensor.shape != input_tensor.shape:
        pytest.fail(f"Unexpected output shape: {output_tensor.shape}")
