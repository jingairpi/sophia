import sys
import types

import pytest
from pydantic import ValidationError

from sophia.model.config.embeddings import (
    PositionalEmbeddingConfig,
    TokenEmbeddingConfig,
)
from sophia.model.layers.bases import EmbeddingLayer


# Create dummy embedding classes.
class DummyEmbedding(EmbeddingLayer):
    def __call__(self, input_ids, *args, **kwargs):
        return input_ids


class NotEmbedding:
    pass


dummy_embedding_module = types.ModuleType("dummy_embedding_module")
dummy_embedding_module.DummyEmbedding = DummyEmbedding
dummy_embedding_module.NotEmbedding = NotEmbedding
sys.modules["dummy_embedding_module"] = dummy_embedding_module


def test_token_embedding_config_valid():
    config = TokenEmbeddingConfig(
        target="dummy_embedding_module.DummyEmbedding",
        vocab_size=10000,
        hidden_size=512,
    )
    assert config.target == "dummy_embedding_module.DummyEmbedding"
    assert config.vocab_size == 10000
    assert config.hidden_size == 512


def test_token_embedding_config_invalid_target():
    with pytest.raises(ValidationError) as excinfo:
        TokenEmbeddingConfig(
            target="dummy_embedding_module.NotEmbedding",
            vocab_size=10000,
            hidden_size=512,
        )
    assert "must be a subclass of EmbeddingLayer" in str(excinfo.value)


# Create dummy positional embedding classes.
class DummyPositionEmbedding(EmbeddingLayer):
    def __call__(self, position_ids, *args, **kwargs):
        return position_ids


class NotPositionEmbedding:
    pass


dummy_position_module = types.ModuleType("dummy_position_module")
dummy_position_module.DummyPositionEmbedding = DummyPositionEmbedding
dummy_position_module.NotPositionEmbedding = NotPositionEmbedding
sys.modules["dummy_position_module"] = dummy_position_module


def test_position_embedding_config_valid():
    config = PositionalEmbeddingConfig(
        target="dummy_position_module.DummyPositionEmbedding",
        max_seq_length=1024,
        hidden_size=512,
    )
    assert config.target == "dummy_position_module.DummyPositionEmbedding"
    assert config.max_length == 1024
    assert config.hidden_size == 512


def test_position_embedding_config_invalid_target():
    with pytest.raises(ValidationError) as excinfo:
        PositionalEmbeddingConfig(
            target="dummy_position_module.NotPositionEmbedding",
            max_seq_length=1024,
            hidden_size=512,
        )
    assert "must be a subclass of EmbeddingLayer" in str(excinfo.value)
