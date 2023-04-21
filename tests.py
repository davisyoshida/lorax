import warnings

import jax
import jax.numpy as jnp
import haiku as hk

from lorax import transform
from lorax.constants import LORA_FULL, LORA_FREEZE
from lorax.transform import init_lora, lora

def test_prepare():
    w_shape = 3, 4
    params = {
        'W': jnp.zeros(w_shape),
        'b': jnp.zeros((4,)),
        'W2': jnp.zeros((4, 5)),
    }
    spec = {
        'W': 2,
        'b': LORA_FREEZE,
        'W2': LORA_FULL
    }

    frozen_params, tune_params = init_lora(params, spec, jax.random.PRNGKey(0))

    assert frozen_params['W'].shape == params['W'].shape
    assert frozen_params['b'].shape == params['b'].shape
    assert frozen_params['W2'] is transform.EmptyNode

    assert isinstance(tune_params['W'], transform.LoraNode)
    assert tune_params['W'].a.shape == (2, w_shape[1])
    assert tune_params['W'].b.shape == (w_shape[0], 2)

    assert tune_params['b'] is transform.EmptyNode
    assert tune_params['W2'].shape == params['W2'].shape

def test_simple():
    key, init_key = jax.random.split(jax.random.PRNGKey(17))
    batch = 5
    time = 7
    hidden = 11
    output = 13
    x = jax.random.normal(key, (batch, time, hidden))

    params = [
        jax.random.normal(key, (hidden, output)),
    ]

    def f(params, x):
        return x @ params[0]

    orig_output = f(params, x)

    lora_params = init_lora(params, [2], rng=init_key)

    lora_f = lora(f)
    lora_output = lora_f(lora_params, x)

    assert jnp.allclose(orig_output, lora_output)

    lora_params[1][0].b = jax.random.normal(key, (hidden, 2)) * 10

    perturbed_lora = lora_f(lora_params, x)

    combined_params = transform.merge_params(*lora_params)
    combined_output = f(combined_params, x)

    assert jnp.allclose(perturbed_lora, combined_output, rtol=1e-4)

def test_conv():
    key, a_key, b_key = jax.random.split(jax.random.PRNGKey(18), 3)
    batch = 7
    time = 11
    hidden = 13
    output = 17
    rank_constraint = 3
    window_size = 2
    x = jax.random.normal(key, (batch, time, hidden))

    def fn(w, x):
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=w,
            window_strides=(1,),
            dimension_numbers=jax.lax.ConvDimensionNumbers(
                (0, 2, 1),
                (2, 1, 0),
                (0, 2, 1)
            ),
            padding='VALID'
        )

    a = jax.random.normal(b_key, (1, rank_constraint, output))
    b = jax.random.normal(a_key, (window_size, hidden, rank_constraint))

    w = b @ a

    lora_params = (
        jnp.zeros((window_size, hidden, output)),
        transform.LoraNode(a, b)
    )

    lora_fn = lora(fn)
    orig_result = fn(w, x)
    lora_result = lora_fn(lora_params, x)
    print(f'Orig: {orig_result[:3, :3, :3]}')
    print(f'Lora: {lora_result[:3, :3, :3]}')

    assert jnp.allclose(orig_result, lora_result, rtol=1e-3)

def test_embedding():
    key, a_key, b_key = jax.random.split(jax.random.PRNGKey(19), 3)
    batch = 11
    time = 13
    vocab = 4321
    hidden = 100

    rank_constraint = 19

    ids = jax.random.randint(key, (batch, time), 0, vocab)

    a = jax.random.normal(b_key, (rank_constraint, hidden))
    b = jax.random.normal(a_key, (vocab, rank_constraint))

    def f(w, x):
        return jax.lax.gather(
            w,
            x[:, :, None],
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(2,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,),
            ),
            mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            slice_sizes=(1, hidden)
        )

    w = b @ a
    jaxpr = jax.make_jaxpr(f)(w, ids)
    lora_params = (
        jnp.zeros((vocab, hidden)),
        transform.LoraNode(a, b)
    )

    lora_f = lora(f)

    orig_result = f(w, ids)
    lora_result = lora_f(lora_params, ids)

    assert jnp.allclose(orig_result, lora_result, rtol=1e-4)
