import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def create_train_step_fn(model: Any, compute_loss_fn: Any):
    """
    Creates and returns a jitted training step function that captures the given
    model and compute_loss_fn in its closure. This avoids passing non-hashable objects
    as static arguments to jax.jit.

    Args:
        model: A Flax module instance.
        compute_loss_fn: A function that computes the loss and auxiliary outputs.

    Returns:
        A jitted training step function with signature:
            (state, batch, rng) -> (new_state, loss, logits)
    """

    def train_step_fn(state, batch, rng):
        grad_fn = jax.value_and_grad(compute_loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params, batch, rng)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, logits

    return jax.jit(train_step_fn)


class Trainer:
    """
    This Trainer implements a training loop based on a jitted training step function.
    It initializes the model parameters using a dummy input, sets up an optimizer
    (here, optax.adamw), and iterates over an iterable dataset for a specified number
    of training steps.

    Attributes:
        model: A Flax module (nn.Module) representing the model.
        config: A configuration object with attributes used for dummy input shape (e.g., n_positions).
        optimizer_config: A dict containing optimizer hyperparameters (e.g., batch_size, learning_rate).
        dataset: An iterable that yields training batches (each batch is typically a tuple (inputs, targets)).
        rng: A JAX PRNG key.
        state: A TrainState holding the model parameters and optimizer state.
        train_step_fn: A jitted function that performs a single training step.
    """

    def __init__(
        self, model: Any, config: Any, optimizer_config: dict, dataset: Any, rng: Any
    ):
        """
        Initializes the Trainer.

        Args:
            model: A Flax model (nn.Module).
            config: A configuration object. It must have an attribute 'n_positions' used for dummy input.
            optimizer_config: A dict with keys like "batch_size" and "learning_rate".
            dataset: An iterable data loader yielding training batches.
            rng: A JAX PRNG key.
        """
        self.model = model
        self.config = config
        self.dataset = dataset
        self.rng = rng

        # Create a dummy input for parameter initialization.
        # Typically, n_positions is the full sequence length. The Trainer expects dummy input shape of
        # (batch_size, n_positions - 1) since the DataLoader splits sequences into inputs and targets.
        dummy_input = jnp.ones(
            (optimizer_config["batch_size"], config.n_positions - 1), dtype=jnp.int32
        )
        params = model.init(rng, dummy_input)["params"]
        logger.info("Model parameters initialized.")

        # Set up the optimizer (here using AdamW from optax).
        tx = optax.adamw(learning_rate=optimizer_config["learning_rate"])
        self.state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )
        logger.info(
            "Optimizer and training state set up with learning rate %f.",
            optimizer_config["learning_rate"],
        )

        # Create the jitted training step function, capturing model and compute_loss in a closure.
        self.train_step_fn = create_train_step_fn(self.model, self.compute_loss)
        logger.info("Training step function created and JIT-compiled.")

    def compute_loss(self, params: Any, batch: Any, rng: Any) -> Any:
        """
        Computes the loss and logits for a given batch using the current parameters.

        Args:
            params: Model parameters.
            batch: A tuple (inputs, targets).
            rng: A PRNG key for stochastic operations (e.g., dropout).

        Returns:
            A tuple (loss, logits) where loss is a scalar and logits is the output.
        """
        inputs, targets = batch
        logits = self.model.apply(
            {"params": params}, inputs, deterministic=False, rngs={"dropout": rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return loss.mean(), logits

    def train_step(self, state: Any, batch: Any, rng: Any) -> Any:
        """
        Performs a single training step by invoking the jitted training step function.

        Args:
            state: The current training state.
            batch: A batch of training data.
            rng: A PRNG key.

        Returns:
            A tuple (new_state, loss, logits) after one training step.
        """
        return self.train_step_fn(state, batch, rng)

    def train(self, num_steps: int) -> Any:
        """
        Executes the training loop for a specified number of steps.

        Args:
            num_steps: The number of training steps to run.

        Returns:
            The final training state after completing the training loop.
        """
        logger.info("Starting training for %d steps.", num_steps)
        step = 0
        while step < num_steps:
            try:
                batch = next(iter(self.dataset))
            except StopIteration:
                logger.info("Dataset exhausted; ending training loop.")
                break

            self.rng, step_rng = jax.random.split(self.rng)
            start_time = time.time()
            self.state, loss, logits = self.train_step_fn(self.state, batch, step_rng)
            elapsed = time.time() - start_time
            logger.info(
                "Step %d: Loss = %.4f, Time = %.2f ms", step, loss, elapsed * 1000
            )
            step += 1

        logger.info("Training completed after %d steps.", step)
        return self.state
