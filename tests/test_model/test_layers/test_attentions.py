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
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )

    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
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


def test_multihead_attention_gradients():
    """
    Tests that gradients can be computed without error and are finite.
    """
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.0,  # disable dropout for gradient checking
    )
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (8, 16, 64))
    mask_bool = jnp.ones((8, 1, 16, 16), dtype=bool)
    params = mha_wrapper.init(rng, input_tensor, mask_bool, deterministic=True)

    def loss_fn(params, inputs, mask):
        outputs = mha_wrapper.apply(params, inputs, mask, deterministic=True)
        return jnp.mean(outputs)

    grads = jax.grad(loss_fn)(params, input_tensor, mask_bool)
    leaves, _ = jax.tree_util.tree_flatten(grads)
    for g in leaves:
        assert jnp.all(jnp.isfinite(g)), "Gradient contains non-finite values."


def test_multihead_attention_dropout_variability():
    """
    Tests that when dropout is enabled (deterministic=False) and different dropout rng keys are used,
    the outputs differ.
    """
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.1,
    )
    rng = jax.random.PRNGKey(0)
    dropout_rng1, dropout_rng2, init_rng = jax.random.split(rng, 3)
    input_tensor = jax.random.normal(rng, (8, 16, 64))
    mask_bool = jnp.ones((8, 1, 16, 16), dtype=bool)
    params = mha_wrapper.init(init_rng, input_tensor, mask_bool, deterministic=False)
    output_1 = mha_wrapper.apply(
        params,
        input_tensor,
        mask_bool,
        deterministic=False,
        rngs={"dropout": dropout_rng1},
    )
    output_2 = mha_wrapper.apply(
        params,
        input_tensor,
        mask_bool,
        deterministic=False,
        rngs={"dropout": dropout_rng2},
    )
    assert not jnp.allclose(
        output_1, output_2, atol=1e-6
    ), "Outputs with dropout enabled should differ when using different dropout RNG keys."


def test_multihead_attention_mask_effect():
    """
    Tests that using a fully masked attention (all False) produces a predictable output.

    With an all-False mask, every score becomes -1e9 before softmax. Due to the subtraction
    of the max value in softmax, the result will be a uniform distribution (1/seq_length)
    for each query. Therefore, the output becomes the average of the value vectors (after projection)
    passed through the final dense layer.

    Note: This test is less direct because the Dense layers are not the identity. However,
    comparing outputs with different masks (e.g. all True vs all False) should yield different results.
    """
    mha_wrapper = MultiHeadDotProductAttention(
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.0,
    )
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (4, 16, 64))
    mask_all_true = jnp.ones((4, 1, 16, 16), dtype=bool)
    mask_all_false = jnp.zeros((4, 1, 16, 16), dtype=bool)

    params = mha_wrapper.init(rng, input_tensor, mask_all_true, deterministic=True)
    output_true = mha_wrapper.apply(
        params, input_tensor, mask_all_true, deterministic=True
    )
    output_false = mha_wrapper.apply(
        params, input_tensor, mask_all_false, deterministic=True
    )

    # Expect the outputs to be different because the mask forces uniform attention (averaging)
    # in one case versus data-dependent attention in the other.
    assert not jnp.allclose(
        output_true, output_false, atol=1e-6
    ), "Outputs should differ when using different attention masks."
