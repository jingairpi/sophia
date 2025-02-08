import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from sophia.configs.gpt2 import GPT2_SMALL, GPT2Config, GPT2Params
from sophia.model.builder import build_model

try:
    from transformers import GPT2Config as HF_GPT2Config
except ImportError:
    HF_GPT2Config = None


@pytest.fixture
def our_config():
    return GPT2Config.from_params(GPT2_SMALL)


def test_gpt2_config_from_params():
    # Create our config using GPT2_SMALL parameters.
    config = GPT2Config.from_params(GPT2_SMALL)

    # Check that top-level hyperparameters are set correctly.
    assert config.params.n_layer == GPT2_SMALL.n_layer
    assert config.params.hidden_size == GPT2_SMALL.hidden_size
    assert config.params.vocab_size == GPT2_SMALL.vocab_size
    assert config.params.n_positions == GPT2_SMALL.n_positions

    # Check that nested configurations reflect the parameters.
    assert config.token_embedding.vocab_size == GPT2_SMALL.vocab_size
    assert config.token_embedding.hidden_size == GPT2_SMALL.hidden_size
    assert config.positional_embedding.max_seq_length == GPT2_SMALL.n_positions
    # For transformer block, check that dropout_rate is passed down.
    assert config.transformer_block.dropout_rate == GPT2_SMALL.dropout_rate


def test_gpt2_config_vs_hf():
    """
    If transformers is installed, compare some key hyperparameters with HuggingFace's GPT2Config.
    """
    if HF_GPT2Config is None:
        pytest.skip("Transformers not installed; skipping HF comparison test.")

    # Create our config using GPT2_SMALL parameters.
    our_params = GPT2_SMALL
    our_config = GPT2Config.from_params(our_params)

    # Create a HuggingFace GPT2Config (with default values).
    hf_config = HF_GPT2Config()

    # Compare common hyperparameters.
    # HuggingFace's config uses n_layer, n_embd, vocab_size, and n_positions.
    assert our_params.n_layer == hf_config.n_layer
    assert our_params.hidden_size == hf_config.n_embd
    assert our_params.vocab_size == hf_config.vocab_size
    assert our_params.n_positions == hf_config.n_positions


def test_gpt2_model_forward_shape(our_config):
    model = build_model(our_config)
    rng = jax.random.PRNGKey(0)
    # Dummy input: batch of token ids, shape [batch, sequence]
    dummy_input = jnp.ones((2, 10), dtype=jnp.int32)
    params = model.init_params(rng)
    output = model.apply(params, dummy_input, deterministic=True)
    # Our GPT2Config sets vocab_size as the output dimension for the final projection.
    assert output.shape == (2, 10, our_config.params.vocab_size)
