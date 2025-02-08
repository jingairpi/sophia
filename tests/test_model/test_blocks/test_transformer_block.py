import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from sophia.model.blocks.transformer_block import TransformerBlock
from sophia.model.layers.activations import GELUActivation
from sophia.model.layers.attentions import MultiHeadDotProductAttention
from sophia.model.layers.feed_forwards import PositionwiseFeedForward
from sophia.model.layers.normalizations import LayerNormalization

# ------------------------------------------------------------------------------
# Configuration similar to GPT-2 (using smaller dimensions for testing)
# ------------------------------------------------------------------------------
hidden_size = 64
num_heads = 4
dropout_rate = 0.1
ffn_multiplier = 4
normalization_kwargs = {"epsilon": 1e-5}

attention_kwargs = {
    "hidden_size": hidden_size,
    "num_heads": num_heads,
    "dropout_rate": dropout_rate,
}
feed_forward_network_kwargs = {
    "hidden_size": hidden_size,
    "ffn_multiplier": ffn_multiplier,
    "dropout_rate": dropout_rate,
    "activation": GELUActivation(),
}

# For GPT-2, the standard Transformer block is usually configured with post‑norm.
default_block_config = dict(
    attention_cls=MultiHeadDotProductAttention,
    attention_kwargs=attention_kwargs,
    feed_forward_network_cls=PositionwiseFeedForward,
    feed_forward_network_kwargs=feed_forward_network_kwargs,
    normalization_cls=LayerNormalization,
    normalization_kwargs=normalization_kwargs,
    pre_norm=False,
    residual_scale=1.0,
    dropout_rate=dropout_rate,
)


# ------------------------------------------------------------------------------
# Test 1: Check that the output shape matches the input shape.
# ------------------------------------------------------------------------------
def test_transformer_block_output_shape():
    batch_size, seq_length = 8, 16
    rng = jax.random.PRNGKey(0)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    # For simplicity, use a full (all-True) attention mask.
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    transformer_block = TransformerBlock(**default_block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(1), input_tensor, attention_mask=mask, deterministic=True
    )
    output = transformer_block.apply(
        variables, input_tensor, attention_mask=mask, deterministic=True
    )
    assert (
        output.shape == input_tensor.shape
    ), f"Output shape {output.shape} does not match input shape {input_tensor.shape}."


# ------------------------------------------------------------------------------
# Test 2: Deterministic behavior (with dropout disabled) should yield identical outputs.
# ------------------------------------------------------------------------------
def test_transformer_block_deterministic():
    batch_size, seq_length = 4, 10
    rng = jax.random.PRNGKey(2)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    transformer_block = TransformerBlock(**default_block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(3), input_tensor, attention_mask=mask, deterministic=True
    )
    out1 = transformer_block.apply(
        variables, input_tensor, attention_mask=mask, deterministic=True
    )
    out2 = transformer_block.apply(
        variables, input_tensor, attention_mask=mask, deterministic=True
    )
    assert jnp.allclose(
        out1, out2, atol=1e-6
    ), "Deterministic outputs should be identical."


# ------------------------------------------------------------------------------
# Test 3: With dropout enabled (deterministic=False) and different dropout RNG keys,
#          the outputs should differ.
# ------------------------------------------------------------------------------
def test_transformer_block_dropout_variability():
    batch_size, seq_length = 4, 10
    rng = jax.random.PRNGKey(4)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    transformer_block = TransformerBlock(**default_block_config)
    init_rng = jax.random.PRNGKey(5)
    variables = transformer_block.init(
        init_rng, input_tensor, attention_mask=mask, deterministic=False
    )

    dropout_rng1 = jax.random.PRNGKey(6)
    dropout_rng2 = jax.random.PRNGKey(7)
    out1 = transformer_block.apply(
        variables,
        input_tensor,
        attention_mask=mask,
        deterministic=False,
        rngs={"dropout": dropout_rng1},
    )
    out2 = transformer_block.apply(
        variables,
        input_tensor,
        attention_mask=mask,
        deterministic=False,
        rngs={"dropout": dropout_rng2},
    )
    assert not jnp.allclose(
        out1, out2, atol=1e-6
    ), "Outputs with dropout enabled should differ when using different dropout RNG keys."


# ------------------------------------------------------------------------------
# Test 4: Check that gradients can be computed and are finite.
# ------------------------------------------------------------------------------
def test_transformer_block_gradients():
    batch_size, seq_length = 2, 8
    rng = jax.random.PRNGKey(8)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    transformer_block = TransformerBlock(**default_block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(9), input_tensor, attention_mask=mask, deterministic=True
    )

    def loss_fn(params, inputs, mask):
        output = transformer_block.apply(
            params, inputs, attention_mask=mask, deterministic=True
        )
        return jnp.mean(output)

    grads = jax.grad(loss_fn)(variables, input_tensor, mask)
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g)), "Gradient contains non-finite values."


# ------------------------------------------------------------------------------
# Test 5: Causal mask effect – verify that a causal mask (lower triangular) changes the output.
# ------------------------------------------------------------------------------
def test_transformer_block_causal_mask_effect():
    batch_size, seq_length = 2, 8
    rng = jax.random.PRNGKey(10)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    # Create a causal (lower-triangular) mask.
    causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
    causal_mask = causal_mask[jnp.newaxis, jnp.newaxis, :, :].repeat(batch_size, axis=0)

    transformer_block = TransformerBlock(**default_block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(11),
        input_tensor,
        attention_mask=causal_mask,
        deterministic=True,
    )
    out_causal = transformer_block.apply(
        variables, input_tensor, attention_mask=causal_mask, deterministic=True
    )

    # Compare with a full mask (all True)
    full_mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)
    out_full = transformer_block.apply(
        variables, input_tensor, attention_mask=full_mask, deterministic=True
    )
    # They should differ because the masking changes the attention scores.
    assert not jnp.allclose(
        out_causal, out_full, atol=1e-6
    ), "Causal mask should affect the output compared to a full mask."


# ------------------------------------------------------------------------------
# Test 6: pre-norm vs. post-norm configuration differences.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("pre_norm", [True, False])
def test_transformer_block_norm_configuration(pre_norm):
    batch_size, seq_length = 4, 12
    rng = jax.random.PRNGKey(12)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    block_config = default_block_config.copy()
    block_config["pre_norm"] = pre_norm

    transformer_block = TransformerBlock(**block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(13), input_tensor, attention_mask=mask, deterministic=True
    )
    output = transformer_block.apply(
        variables, input_tensor, attention_mask=mask, deterministic=True
    )
    # We only verify that the block runs and the output has the correct shape.
    assert (
        output.shape == input_tensor.shape
    ), "Output shape should match input shape regardless of norm configuration."
