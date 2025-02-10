import pytest

from sophia.model.config import (
    ActivationConfig,
    AttentionLayerConfig,
    FeedForwardConfig,
    LayerConfig,
    LayerNormalizationConfig,
    ModelConfig,
    OutputProjectionConfig,
    PositionalEmbedding,
    RMSNormalization,
    TokenEmbeddingConfig,
    TransformerBlockConfig,
)


# Test for the base LayerConfig
def test_layer_config():
    data = {
        "type": "SomeLayer",
        "config": {"param": 123},
        "repeat": 2,
    }
    lc = LayerConfig(**data)
    assert lc.type == "SomeLayer"
    assert lc.config == {"param": 123}
    assert lc.repeat == 2


# Test for ModelConfig
def test_model_config():
    data = {
        "name": "TestModel",
        "dtype": "float32",
        "layers": [{"type": "SomeLayer", "config": {"param": 123}}],
    }
    mc = ModelConfig(**data)
    assert mc.name == "TestModel"
    assert mc.dtype == "float32"
    assert isinstance(mc.layers, list)
    assert mc.layers[0].type == "SomeLayer"


# Test for ActivationConfig (inherits without additional fields)
def test_activation_config():
    data = {
        "type": "GELU",
        "config": {"approximate": True},
    }
    ac = ActivationConfig(**data)
    assert ac.type == "GELU"
    assert ac.config["approximate"] is True


# Test for AttentionLayerConfig
def test_attention_layer_config():
    data = {
        "type": "Attention",
        "config": {"some_config": "value"},
        "num_heads": 8,
        "head_dim": 64,
        "dropout_rate": 0.1,
    }
    att_cfg = AttentionLayerConfig(**data)
    assert att_cfg.type == "Attention"
    assert att_cfg.num_heads == 8
    assert att_cfg.head_dim == 64
    assert att_cfg.dropout_rate == 0.1


# Test for TokenEmbeddingConfig
def test_token_embedding_config():
    data = {
        "type": "TokenEmbedding",
        "config": {},
        "vocab_size": 1000,
        "hidden_size": 256,
    }
    te_cfg = TokenEmbeddingConfig(**data)
    assert te_cfg.vocab_size == 1000
    assert te_cfg.hidden_size == 256


# Test for PositionalEmbedding
def test_positional_embedding():
    data = {
        "type": "PositionalEmbedding",
        "config": {},
        "max_seq_length": 512,
        "hidden_size": 256,
    }
    pe_cfg = PositionalEmbedding(**data)
    assert pe_cfg.max_seq_length == 512
    assert pe_cfg.hidden_size == 256


# Test for FeedForwardConfig with nested ActivationConfig
def test_feed_forward_config():
    activation_data = {
        "type": "GELU",
        "config": {},
    }
    data = {
        "type": "FeedForward",
        "config": {},
        "hidden_dim": 256,
        "ff_dim": 1024,
        "dropout_rate": 0.2,
        "activation": activation_data,
    }
    ff_cfg = FeedForwardConfig(**data)
    assert ff_cfg.hidden_dim == 256
    assert ff_cfg.ff_dim == 1024
    assert ff_cfg.dropout_rate == 0.2
    assert ff_cfg.activation.type == "GELU"


# Test for LayerNormalizationConfig
def test_layer_normalization_config():
    data = {
        "type": "LayerNormalization",
        "config": {},
        "eps": 1e-5,
    }
    ln_cfg = LayerNormalizationConfig(**data)
    assert ln_cfg.eps == 1e-5


# Test for RMSNormalization
def test_rms_normalization():
    data = {
        "type": "RMSNormalization",
        "config": {},
        "features": 128,
        "eps": 1e-6,
    }
    rms_cfg = RMSNormalization(**data)
    assert rms_cfg.features == 128
    assert rms_cfg.eps == 1e-6


# Test for OutputProjectionConfig
def test_output_projection_config():
    data = {
        "type": "OutputProjection",
        "config": {},
        "input_dim": 512,
        "output_dim": 1000,
    }
    op_cfg = OutputProjectionConfig(**data)
    assert op_cfg.input_dim == 512
    assert op_cfg.output_dim == 1000


# Test for TransformerBlockConfig with nested configurations
def test_transformer_block_config():
    # Nested attention configuration
    attention_data = {
        "type": "Attention",
        "config": {},
        "num_heads": 8,
        "head_dim": 64,
        "dropout_rate": 0.1,
    }
    # Nested activation for the feed-forward network
    activation_data = {
        "type": "GELU",
        "config": {},
    }
    # Nested feed-forward configuration
    feed_forward_data = {
        "type": "FeedForward",
        "config": {},
        "hidden_dim": 256,
        "ff_dim": 1024,
        "dropout_rate": 0.2,
        "activation": activation_data,
    }
    # Nested normalization configuration
    normalization_data = {
        "type": "LayerNormalization",
        "config": {},
        "eps": 1e-5,
    }
    transformer_data = {
        "type": "TransformerBlock",
        "config": {},
        "hidden_size": 512,
        "num_heads": 8,
        "ff_dim": 1024,
        "dropout_rate": 0.1,
        "pre_norm": True,
        "residual_scale": 1.0,
        "attention": attention_data,
        "feed_forward_network": feed_forward_data,
        "normalization_1": normalization_data,
        "normalization_2": normalization_data,
    }
    tb_cfg = TransformerBlockConfig(**transformer_data)
    assert tb_cfg.hidden_size == 512
    assert tb_cfg.num_heads == 8
    assert tb_cfg.attention.num_heads == 8
    assert tb_cfg.feed_forward_network.ff_dim == 1024
    assert tb_cfg.normalization_1.eps == 1e-5
    assert tb_cfg.normalization_2.eps == 1e-5
