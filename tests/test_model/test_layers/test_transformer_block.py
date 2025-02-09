import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from sophia.model.layers.activations import GELUActivation
from sophia.model.layers.attentions import MultiHeadDotProductAttention
from sophia.model.layers.feed_forwards import PositionwiseFeedForward
from sophia.model.layers.normalizations import LayerNormalization, RMSNormalization
from sophia.model.layers.transformer_block import TransformerBlock

# ------------------------------------------------------------------------------
# Configuration similar to GPT-2 (using smaller dimensions for testing)
# ------------------------------------------------------------------------------
hidden_size = 64
num_heads = 4
dropout_rate = 0.1
ffn_multiplier = 4
normalization_kwargs = {"epsilon": 1e-5}

# Initialize Layer Instances
attention_layer = MultiHeadDotProductAttention(
    hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate
)

feed_forward_network = PositionwiseFeedForward(
    hidden_size=hidden_size,
    ffn_multiplier=ffn_multiplier,
    dropout_rate=dropout_rate,
    activation=GELUActivation(),
)

# Default Transformer Block Config
default_block_config = dict(
    attention=attention_layer,
    feed_forward_network=feed_forward_network,
    normalization_1=LayerNormalization(**normalization_kwargs),
    normalization_2=LayerNormalization(**normalization_kwargs),
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
# Test 3: Dropout variability with different RNG keys.
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
    ), "Outputs with different dropout RNG keys should differ."


# ------------------------------------------------------------------------------
# Test 4: Gradient check.
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
# Test 5: Different Normalization Layers
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("normalization_cls", [LayerNormalization, RMSNormalization])
def test_transformer_block_with_different_normalization(normalization_cls):
    block_config = default_block_config.copy()

    if normalization_cls == RMSNormalization:
        block_config["normalization_1"] = normalization_cls(
            features=hidden_size, **normalization_kwargs
        )
        block_config["normalization_2"] = normalization_cls(
            features=hidden_size, **normalization_kwargs
        )
    else:
        block_config["normalization_1"] = normalization_cls(**normalization_kwargs)
        block_config["normalization_2"] = normalization_cls(**normalization_kwargs)

    batch_size, seq_length = 4, 12
    rng = jax.random.PRNGKey(10)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    transformer_block = TransformerBlock(**block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(11), input_tensor, attention_mask=mask, deterministic=True
    )
    output = transformer_block.apply(
        variables, input_tensor, attention_mask=mask, deterministic=True
    )

    assert output.shape == input_tensor.shape, "Output shape should match input shape."


# ------------------------------------------------------------------------------
# Test 6: Causal mask effect – verify that a causal mask (lower triangular) changes the output.
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
# Test 7: pre-norm vs. post-norm configuration.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("pre_norm", [True, False])
def test_transformer_block_norm_configuration(pre_norm):
    block_config = default_block_config.copy()
    block_config["pre_norm"] = pre_norm

    batch_size, seq_length = 4, 12
    rng = jax.random.PRNGKey(12)
    input_tensor = jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    mask = jnp.ones((batch_size, 1, seq_length, seq_length), dtype=bool)

    transformer_block = TransformerBlock(**block_config)
    variables = transformer_block.init(
        jax.random.PRNGKey(13), input_tensor, attention_mask=mask, deterministic=True
    )
    output = transformer_block.apply(
        variables, input_tensor, attention_mask=mask, deterministic=True
    )

    assert output.shape == input_tensor.shape, "Output shape should match input shape."
