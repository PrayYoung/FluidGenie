"""Independent RNG streams for parameter and stochastic layer initialization."""

import jax


def initialization_rngs(key):
    params_key, dropout_key = jax.random.split(key)
    return {"params": params_key, "dropout": dropout_key}
