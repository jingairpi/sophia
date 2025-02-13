import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError
from transformers import FlaxGPT2Model
from transformers import GPT2Config as HF_GPT2Config

from sophia.configs.gpt2 import GPT2_SMALL, GPT2Config, GPT2Params
from sophia.model.builder import build_model


@pytest.fixture
def our_config():
    """Fixture to create GPT2Config from GPT2_SMALL."""
    return GPT2Config.from_params(GPT2_SMALL)


def test_gpt2_config_from_params():
    """Test correct initialization of GPT2Config from params."""
    config = GPT2Config.from_params(GPT2_SMALL)

    # Check hyperparameters
    assert config.params.n_layer == GPT2_SMALL.n_layer
    assert config.params.hidden_size == GPT2_SMALL.hidden_size
    assert config.params.vocab_size == GPT2_SMALL.vocab_size
    assert config.params.n_positions == GPT2_SMALL.n_positions

    # Check nested configurations
    assert config.token_embedding.vocab_size == GPT2_SMALL.vocab_size
    assert config.token_embedding.hidden_size == GPT2_SMALL.hidden_size
    assert config.positional_embedding.max_seq_length == GPT2_SMALL.n_positions
    assert config.transformer_block.dropout_rate == GPT2_SMALL.dropout_rate


def test_gpt2_config_vs_hf():
    """Compare GPT2Config with HuggingFace's GPT-2Config."""
    our_params = GPT2_SMALL
    our_config = GPT2Config.from_params(our_params)
    hf_config = HF_GPT2Config()

    assert our_params.n_layer == hf_config.n_layer
    assert our_params.hidden_size == hf_config.n_embd
    assert our_params.vocab_size == hf_config.vocab_size
    assert our_params.n_positions == hf_config.n_positions


def test_gpt2_model_forward_shape(our_config):
    """Test GPT2Model output shape."""
    model = build_model(our_config)
    rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((2, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    output = model.apply(params, dummy_input, deterministic=True)
    assert output.shape == (2, 10, our_config.params.vocab_size)
