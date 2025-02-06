import jax
import jax.numpy as jnp
import pytest

from sophia.model.layers.normalizations import LayerNormalization, RMSNormalization


@pytest.mark.parametrize(
    "input_shape, epsilon",
    [
        ((4, 8), 1e-6),  # Standard test case
        ((2, 16), 1e-5),  # Larger tensor and higher epsilon
    ],
)
def test_layer_normalization_output_shape(input_shape, epsilon):
    """
    Tests that LayerNormalization produces an output with the same shape as the
    input.
    """
    layer_norm = LayerNormalization(epsilon=epsilon)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = layer_norm.init(rng, jnp.ones(input_shape))

    # Apply LayerNormalization
    output = layer_norm.apply(params, jnp.ones(input_shape))

    # Check output shape
    if output.shape != input_shape:
        pytest.fail(f"Unexpected output shape: {output.shape}")


@pytest.mark.parametrize(
    "input_tensor, expected_mean, expected_std, epsilon",
    [
        (jnp.array([[1.0, 2.0, 3.0]]), 0.0, 1.0, 1e-6),  # Simple case
        (jnp.array([[-1.0, 0.0, 1.0]]), 0.0, 1.0, 1e-6),  # Centered input
    ],
)
def test_layer_normalization_output_values(
    input_tensor, expected_mean, expected_std, epsilon
):
    """
    Tests that LayerNormalization produces outputs with zero mean and unit
    variance.
    """
    layer_norm = LayerNormalization(epsilon=epsilon)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = layer_norm.init(rng, input_tensor)

    # Apply LayerNormalization
    output = layer_norm.apply(params, input_tensor)

    # Compute mean and std
    mean = jnp.mean(output, axis=-1)
    std = jnp.std(output, axis=-1)

    # Check mean
    if not jnp.allclose(mean, expected_mean, atol=1e-5):
        pytest.fail(f"Unexpected mean: {mean}")

    # Check std
    if not jnp.allclose(std, expected_std, atol=1e-5):
        pytest.fail(f"Unexpected std: {std}")


@pytest.mark.parametrize(
    "input_shape, features, epsilon",
    [
        ((4, 8), 8, 1e-6),  # Standard test case
        ((2, 16), 16, 1e-5),  # Larger tensor and higher epsilon
    ],
)
def test_rms_normalization_output_shape(input_shape, features, epsilon):
    """
    Tests that RMSNormalization produces an output with the same shape as the
    input.
    """
    rms_norm = RMSNormalization(features=features, epsilon=epsilon)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = rms_norm.init(rng, jnp.ones(input_shape))

    # Apply RMSNormalization
    output = rms_norm.apply(params, jnp.ones(input_shape))

    # Check output shape
    if output.shape != input_shape:
        pytest.fail(f"Unexpected output shape: {output.shape}")


@pytest.mark.parametrize(
    "input_tensor, epsilon",
    [
        (jnp.array([[3.0, 4.0]]), 1e-6),  # Simple case
        (jnp.array([[0.0, 0.0]]), 1e-6),  # Zero input
    ],
)
def test_rms_normalization_output_values(input_tensor, epsilon):
    """
    Tests that RMSNormalization normalizes input values correctly.
    """
    features = input_tensor.shape[-1]
    rms_norm = RMSNormalization(features=features, epsilon=epsilon)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = rms_norm.init(rng, input_tensor)

    # Apply RMSNormalization
    output = rms_norm.apply(params, input_tensor)

    # Compute RMS
    mean_square = jnp.mean(jnp.square(input_tensor), axis=-1, keepdims=True)
    expected_rms = jnp.sqrt(mean_square + epsilon)
    normalized_input = input_tensor / expected_rms

    # Compare normalized output (excluding scale for simplicity)
    scale = params["params"]["scale"]
    expected_output = normalized_input * scale

    if not jnp.allclose(output, expected_output, atol=1e-5):
        pytest.fail(f"Unexpected output values: {output}")
