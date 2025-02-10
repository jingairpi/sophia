import pytest
from pydantic import ValidationError

from sophia.model.config import (
    AttentionConfig,
    BaseConfig,
    EmbeddingLayerConfig,
    FeedForwardConfig,
    LayerNormalizationConfig,
    MultiHeadDotProductAttentionConfig,
    NormalizationConfig,
    OutputProjectionConfig,
    PositionalEmbeddingConfig,
    PositionwiseFeedForwardConfig,
    ProjectionConfig,
    RMSNormalizationConfig,
    TokenEmbeddingConfig,
    TransformerBlockConfig,
)


# -------------------------------
# Embedding Config Tests
# -------------------------------
def test_token_embedding_config_valid():
    cfg = TokenEmbeddingConfig(vocab_size=10000, hidden_size=512)
    assert cfg.type == "TokenEmbedding"
    assert cfg.vocab_size == 10000
    assert cfg.hidden_size == 512


def test_positional_embedding_config_valid():
    cfg = PositionalEmbeddingConfig(max_seq_length=512, hidden_size=512)
    assert cfg.type == "PositionalEmbedding"
    assert cfg.max_seq_length == 512
    assert cfg.hidden_size == 512


def test_embedding_layer_union():
    token_cfg = TokenEmbeddingConfig(vocab_size=10000, hidden_size=512)
    pos_cfg = PositionalEmbeddingConfig(max_seq_length=512, hidden_size=512)
    for cfg in (token_cfg, pos_cfg):
        # We can use isinstance() check against the union type
        assert isinstance(cfg, (TokenEmbeddingConfig, PositionalEmbeddingConfig))


# -------------------------------
# Attention Config Tests
# -------------------------------
def test_attention_config_valid():
    cfg = MultiHeadDotProductAttentionConfig(hidden_size=512, num_heads=8)
    assert cfg.type == "MultiHeadDotProductAttention"
    assert cfg.hidden_size == 512
    assert cfg.num_heads == 8
    assert cfg.dropout_rate == 0.1


def test_attention_config_invalid():
    with pytest.raises(ValidationError):
        MultiHeadDotProductAttentionConfig(hidden_size=512, num_heads="eight")


# -------------------------------
# Feed-Forward Config Tests
# -------------------------------
def test_feed_forward_config_valid():
    dummy_activation = "dummy_activation"
    cfg = PositionwiseFeedForwardConfig(
        hidden_size=512, ffn_multiplier=4, activation=dummy_activation, dropout_rate=0.2
    )
    assert cfg.type == "PositionwiseFeedForward"
    assert cfg.hidden_size == 512
    assert cfg.ffn_multiplier == 4
    assert cfg.activation == dummy_activation
    assert cfg.dropout_rate == 0.2


def test_feed_forward_config_invalid():
    with pytest.raises(ValidationError):
        # Missing required field 'ffn_multiplier'
        PositionwiseFeedForwardConfig(
            hidden_size=512, activation="dummy", dropout_rate=0.2
        )


# -------------------------------
# Normalization Config Tests
# -------------------------------
def test_layer_normalization_config_valid():
    cfg = LayerNormalizationConfig()
    assert cfg.type == "LayerNormalization"
    assert cfg.epsilon == 1e-5


def test_rms_normalization_config_valid():
    cfg = RMSNormalizationConfig(epsilon=1e-6)
    assert cfg.type == "RMSNormalization"
    assert cfg.epsilon == 1e-6


def test_normalization_union():
    cfg1 = LayerNormalizationConfig()
    cfg2 = RMSNormalizationConfig(epsilon=1e-6)
    for cfg in (cfg1, cfg2):
        assert isinstance(cfg, (LayerNormalizationConfig, RMSNormalizationConfig))


# -------------------------------
# Output Projection Config Tests
# -------------------------------
def test_output_projection_config_valid():
    cfg = OutputProjectionConfig(hidden_size=512, output_size=10000)
    assert cfg.type == "OutputProjection"
    assert cfg.hidden_size == 512
    assert cfg.output_size == 10000


# -------------------------------
# Transformer Block Config Tests
# -------------------------------
def test_transformer_block_config_valid():
    attention_cfg = MultiHeadDotProductAttentionConfig(hidden_size=512, num_heads=8)
    feed_forward_cfg = PositionwiseFeedForwardConfig(
        hidden_size=512,
        ffn_multiplier=4,
        activation="dummy_activation",
        dropout_rate=0.15,
    )
    norm_cfg = LayerNormalizationConfig()
    cfg = TransformerBlockConfig(
        pre_norm=True,
        residual_scale=0.75,
        dropout_rate=0.25,
        attention=attention_cfg,
        feed_forward=feed_forward_cfg,
        normalization_1=norm_cfg,
        normalization_2=norm_cfg,
    )
    assert cfg.type == "TransformerBlock"
    assert cfg.pre_norm is True
    assert cfg.residual_scale == 0.75
    assert cfg.dropout_rate == 0.25
    assert cfg.attention.type == "MultiHeadDotProductAttention"
    assert cfg.feed_forward.type == "PositionwiseFeedForward"
    assert cfg.normalization_1.type == "LayerNormalization"
    assert cfg.normalization_2.type == "LayerNormalization"


def test_transformer_block_config_invalid():
    attention_cfg = MultiHeadDotProductAttentionConfig(hidden_size=512, num_heads=8)
    feed_forward_cfg = PositionwiseFeedForwardConfig(
        hidden_size=512,
        ffn_multiplier=4,
        activation="dummy_activation",
        dropout_rate=0.15,
    )
    norm_cfg = LayerNormalizationConfig()
    with pytest.raises(ValidationError):
        # Missing normalization_2 field.
        TransformerBlockConfig(
            pre_norm=True,
            residual_scale=0.75,
            dropout_rate=0.25,
            attention=attention_cfg,
            feed_forward=feed_forward_cfg,
            normalization_1=norm_cfg,
        )
