import logging
import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_step_fn(state, batch, rng, model, compute_loss_fn):
    """
    Performs a single training step given the current state, a batch, and a PRNG key.
    Uses the provided model and loss function.

    Args:
        state: The current training state.
        batch: A batch of training data.
        rng: A PRNG key.
        model: The model (an instance of flax.linen.Module).
        compute_loss_fn: A function that computes the loss given parameters, a batch, and a rng.

    Returns:
        A tuple (new_state, loss, logits).
    """
    grad_fn = jax.value_and_grad(compute_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, batch, rng)

    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, logits


train_step_fn = jax.jit(train_step_fn, static_argnums=(3, 4))


class Trainer:
    def __init__(self, model, config, optimizer_config, dataset, rng):
        """
        Initializes the Trainer.

        Args:
            model: A Flax model (an instance of flax.linen.Module).
            config: Configuration object containing model hyperparameters (e.g., config.n_positions).
            optimizer_config: A dictionary with optimizer settings (e.g., batch_size, learning_rate).
            dataset: An iterable dataset (e.g., a DataLoader) yielding training batches.
            rng: A JAX PRNG key.
        """
        self.model = model
        self.config = config
        self.dataset = dataset
        self.rng = rng

        dummy_input = jnp.ones(
            (optimizer_config["batch_size"], config.n_positions - 1), dtype=jnp.int32
        )
        params = model.init(rng, dummy_input)["params"]

        tx = optax.adamw(learning_rate=optimizer_config["learning_rate"])
        self.state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )
        logger.info(
            "Trainer initialized with %d parameters.",
            sum(x.size for x in jax.tree_util.tree_leaves(params)),
        )

    def compute_loss(self, params, batch, rng):
        """
        Computes the loss and logits for a given batch.

        Args:
            params: Model parameters.
            batch: A batch of data (tuple of inputs and targets).
            rng: A PRNG key for dropout and other stochastic operations.

        Returns:
            A tuple (loss, logits), where loss is a scalar and logits is the model output.
        """
        inputs, targets = batch
        logits = self.model.apply(
            {"params": params}, inputs, deterministic=False, rngs={"dropout": rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return loss.mean(), logits

    def train_step(self, state, batch, rng):
        """
        Wrapper for a single training step.

        Args:
            state: The current training state.
            batch: A batch of data.
            rng: A PRNG key.

        Returns:
            A tuple (new_state, loss, logits) after one training step.
        """
        return train_step_fn(state, batch, rng, self.model, self.compute_loss)

    def train(self, num_steps):
        """
        Executes the training loop for a specified number of steps.

        Args:
            num_steps (int): Number of training steps to perform.

        Returns:
            The final training state after the loop.
        """
        step = 0
        while step < num_steps:
            try:
                batch = next(iter(self.dataset))
            except StopIteration:
                logger.info("Dataset exhausted; ending training loop.")
                break

            self.rng, step_rng = jax.random.split(self.rng)
            start_time = time.time()

            self.state, loss, logits = train_step_fn(
                self.state, batch, step_rng, self.model, self.compute_loss
            )
            elapsed = time.time() - start_time
            logger.info(
                "Step %d: Loss = %.4f, Time = %.2f ms", step, loss, elapsed * 1000
            )
            step += 1

        return self.state
