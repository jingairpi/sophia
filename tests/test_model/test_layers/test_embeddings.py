import jax
import jax.numpy as jnp
import pytest

from sophia.model.layers.embeddings import PositionalEmbedding, TokenEmbedding


@pytest.mark.parametrize(
    "vocab_size, hidden_size, batch_size, seq_length",
    [
        (1000, 64, 8, 16),  # Standard test case
        (10, 128, 4, 32),  # Smaller vocab, larger hidden size
    ],
)
def test_token_embedding(vocab_size, hidden_size, batch_size, seq_length):
    """
    Tests the TokenEmbedding class to ensure:
    - The output shape matches [batch_size, seq_length, hidden_size].
    - The output dtype is correct.
    """
    # Create the module
    token_embedding = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = token_embedding.init(
        rng, jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    )

    # Apply the module
    input_ids = jnp.array([[1, 2, 3, 4] * (seq_length // 4)] * batch_size)
    output = token_embedding.apply(params, input_ids)

    # Assertions
    if output.shape != (batch_size, seq_length, hidden_size):
        pytest.fail(f"Unexpected output shape: {output.shape}")
    if output.dtype != jnp.float32:
        pytest.fail(f"Unexpected output dtype: {output.dtype}")


@pytest.mark.parametrize(
    "max_seq_length, hidden_size",
    [
        (16, 64),  # Standard test case
        (32, 128),  # Larger sequence length and hidden size
    ],
)
def test_positional_embedding(max_seq_length, hidden_size):
    """
    Tests the PositionalEmbedding class to ensure:
    - The output shape matches [seq_length, hidden_size].
    - The output dtype is correct.
    """
    # Create the module
    positional_embedding = PositionalEmbedding(
        max_seq_length=max_seq_length, hidden_size=hidden_size
    )

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = positional_embedding.init(rng, jnp.arange(max_seq_length))

    # Apply the module
    position_ids = jnp.arange(0, max_seq_length)
    output = positional_embedding.apply(params, position_ids)

    # Assertions
    if output.shape != (max_seq_length, hidden_size):
        pytest.fail(f"Unexpected output shape: {output.shape}")
    if output.dtype != jnp.float32:
        pytest.fail(f"Unexpected output dtype: {output.dtype}")


def test_token_embedding_empty_input():
    """
    Tests TokenEmbedding with an empty input to ensure graceful handling.
    """
    token_embedding = TokenEmbedding(vocab_size=1000, hidden_size=64)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = token_embedding.init(rng, jnp.ones((1, 1), dtype=jnp.int32))

    # Pass an empty input
    input_ids = jnp.array([]).reshape(0, 0)
    with pytest.raises(ValueError, match="Input IDs cannot be empty"):
        token_embedding.apply(params, input_ids)


def test_positional_embedding_out_of_bounds():
    """
    Tests PositionalEmbedding with out-of-bounds position IDs.
    """
    positional_embedding = PositionalEmbedding(max_seq_length=16, hidden_size=64)

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = positional_embedding.init(rng, jnp.arange(16))

    # Pass out-of-bounds position IDs
    position_ids = jnp.array([0, 1, 17])  # 17 is out of bounds
    with pytest.raises(IndexError, match="Position IDs are out of bounds"):
        positional_embedding.apply(params, position_ids)
