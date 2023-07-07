import warnings

import jax
import pytest

from lorax import LoraWeight

@pytest.fixture(scope='session')
def simple_params():
    m, rank_constraint, n = 11, 7, 19
    x = jax.random.normal(jax.random.PRNGKey(0), (n, 10))

    w = jax.random.normal(jax.random.PRNGKey(1), (m, n))
    a = jax.random.normal(jax.random.PRNGKey(2), (rank_constraint, n))
    b = jax.random.normal(jax.random.PRNGKey(3), (m, rank_constraint))

    full = w + b @ a / b.shape[1]

    lora_params = LoraWeight(
        w=w,
        a=a,
        b=b
    )

    return full, x, lora_params

