import os
import tempfile
import time
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from sophia.trainer.trainer import Trainer


# -----------------------------------------------------------------------------
# Dummy Model Definition
# -----------------------------------------------------------------------------
class DummyModel(nn.Module):
    """
    A simple dummy model for testing.

    It expects input x of shape (batch, seq_len) containing token IDs.
    It expands the input to (batch, seq_len, 1) and applies a Dense layer to produce
    logits of shape (batch, seq_len, features). This mimics a language model head.
    """

    features: int = 5

    @nn.compact
    def __call__(self, x, deterministic: bool = True, **kwargs):
        x = x[..., None]  # Shape: (batch, seq_len, 1)
        x = nn.Dense(self.features)(x)  # Shape: (batch, seq_len, features)
        return x


# -----------------------------------------------------------------------------
# Dummy Configuration Object
# -----------------------------------------------------------------------------
class DummyConfig:
    """
    A dummy configuration object that provides the necessary attribute for Trainer.
    Here, `n_positions` is used to define the dummy input shape for parameter initialization.
    """

    n_positions = 11  # This will lead to a dummy input shape of (batch_size, 10)


# -----------------------------------------------------------------------------
# Dummy Dataset Generator
# -----------------------------------------------------------------------------
def dummy_dataset(
    num_batches: int, batch_size: int, seq_length: int, num_classes: int
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Generates an iterator that yields a fixed number of batches.

    Each batch is a tuple (inputs, targets) where:
      - inputs is an array of shape (batch_size, seq_length)
      - targets is (inputs + 1) modulo num_classes, so that values are in [0, num_classes)

    Args:
        num_batches (int): Number of batches to yield.
        batch_size (int): Number of sequences per batch.
        seq_length (int): Sequence length.
        num_classes (int): The number of classes; inputs are generated in [0, num_classes).

    Returns:
        Iterator yielding (inputs, targets) pairs.
    """
    for _ in range(num_batches):
        inputs = jnp.array(
            np.random.randint(0, num_classes, size=(batch_size, seq_length)),
            dtype=jnp.int32,
        )
        targets = (inputs + 1) % num_classes
        yield (inputs, targets)


# -----------------------------------------------------------------------------
# Optimizer Configuration for Tests
# -----------------------------------------------------------------------------
optimizer_config = {"batch_size": 2, "learning_rate": 0.001}


# -----------------------------------------------------------------------------
# Unit Tests for Trainer
# -----------------------------------------------------------------------------
def test_trainer_train_step():
    """
    Test that a single training step updates the model parameters.
    """
    rng = jax.random.PRNGKey(42)
    model = DummyModel(features=5)
    config = DummyConfig()
    # Create a dummy dataset that produces tokens in [0,5) since model.features==5.
    dataset = dummy_dataset(
        num_batches=3,
        batch_size=optimizer_config["batch_size"],
        seq_length=config.n_positions - 1,
        num_classes=5,
    )
    trainer = Trainer(model, config, optimizer_config, dataset, rng)

    initial_params = trainer.state.params

    # Train for one step.
    state_after = trainer.train(num_steps=1)
    leaves_initial = jax.tree_util.tree_leaves(initial_params)
    leaves_after = jax.tree_util.tree_leaves(state_after.params)
    diff_norm = jnp.sqrt(
        sum(jnp.sum((a - b) ** 2) for a, b in zip(leaves_initial, leaves_after))
    )
    assert diff_norm > 0.0, "Parameters did not change after one training step."


def test_trainer_loss_and_forward():
    """
    Test that compute_loss returns a scalar loss and that the forward pass produces logits with the expected shape.
    """
    rng = jax.random.PRNGKey(123)
    model = DummyModel(features=5)
    config = DummyConfig()
    dataset = dummy_dataset(
        num_batches=2,
        batch_size=optimizer_config["batch_size"],
        seq_length=config.n_positions - 1,
        num_classes=5,
    )
    trainer = Trainer(model, config, optimizer_config, dataset, rng)

    batch = next(iter(dataset))
    trainer_rng = jax.random.PRNGKey(456)
    loss, logits = trainer.compute_loss(trainer.state.params, batch, trainer_rng)
    assert loss.shape == (), "Loss is not a scalar."

    expected_logits_shape = (
        optimizer_config["batch_size"],
        config.n_positions - 1,
        model.features,
    )
    assert (
        logits.shape == expected_logits_shape
    ), f"Logits shape {logits.shape} != expected {expected_logits_shape}"


def test_trainer_train_loop():
    """
    Test that the training loop runs for the specified number of steps and returns a valid final state.
    """
    rng = jax.random.PRNGKey(789)
    model = DummyModel(features=5)
    config = DummyConfig()
    dataset = dummy_dataset(
        num_batches=5,
        batch_size=optimizer_config["batch_size"],
        seq_length=config.n_positions - 1,
        num_classes=5,
    )
    trainer = Trainer(model, config, optimizer_config, dataset, rng)

    final_state = trainer.train(num_steps=3)
    assert final_state is not None, "Trainer.train did not return a state."
    leaves = jax.tree_util.tree_leaves(final_state.params)
    assert len(leaves) > 0, "No parameters were found in the final state."
