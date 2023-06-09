from itertools import product
import warnings

import jax
import jax.numpy as jnp
import pytest

from lorax import lora, init_lora, LoraWeight, LORA_FULL, LORA_FREEZE
from lorax.constants import LORA_FULL, LORA_FREEZE

@pytest.fixture(autouse=True)
def catch_materialization_warnings(recwarn):
    warnings.filterwarnings('error', message='Primitive.*not handled')

def test_materialize(simple_params):
    full, _, lora = simple_params
    assert jnp.allclose(lora.materialize(), full)

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

    lora_params = init_lora(params, spec, jax.random.PRNGKey(0))

    assert isinstance(lora_params['W'], LoraWeight)
    assert lora_params['W'].shape == params['W'].shape
    assert lora_params['b'] is params['b']
    assert lora_params['W2'] is params ['W2']

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

    lora_params[0].b = jax.random.normal(key, (hidden, 2)) * 10

    perturbed_lora = lora_f(lora_params, x)

    combined_params = [lora_params[0].materialize()]
    combined_output = f(combined_params, x)

    print(f'Gap: {jnp.abs(combined_output - perturbed_lora).max()}')
    assert jnp.allclose(perturbed_lora, combined_output, atol=1e-5)

def test_right_matmul(simple_params):
    w, _, lora_params = simple_params
    x = jax.random.normal(jax.random.PRNGKey(3), (10, w.shape[0]))
    def f(w, x):
        return x @ w

    lora_f = lora(f)
    lora_result = lora_f(lora_params, x)

    orig_result = f(w, x)

    assert jnp.allclose(lora_result, orig_result, atol=1e-4)

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

    lora_params = LoraWeight(
        w=jnp.zeros((window_size, hidden, output)),
        a=a,
        b=b
    )

    w = lora_params.materialize()

    lora_fn = lora(fn)
    orig_result = fn(w, x)
    lora_result = lora_fn(lora_params, x)
    print(f'Orig: {orig_result[:3, :3, :3]}')
    print(f'Lora: {lora_result[:3, :3, :3]}')

    assert jnp.allclose(orig_result, lora_result, rtol=1e-3)

def test_embedding():
    key, a_key, b_key, emb_key = jax.random.split(jax.random.PRNGKey(19), 4)
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

    lora_params = LoraWeight(
        w=jax.random.normal(emb_key, (vocab, hidden)),
        a=a,
        b=b
    )
    w = lora_params.materialize()

    lora_f = lora(f)

    orig_result = f(w, ids)
    lora_result = lora_f(lora_params, ids)

    gap = jnp.max(jnp.abs(orig_result - lora_result))
    print(f'Gap: {gap:.3e}')
    assert jnp.allclose(orig_result, lora_result, atol=1e-5)

def test_einsum(simple_params):
    w, x, lora_params = simple_params

    def f(w, x):
        return jnp.einsum('ij,jk->ik', w, x)

    expected = f(w, x)

    lora_f = lora(f)
    result = lora_f(lora_params, x)
    assert jnp.allclose(expected, result, rtol=1e-4)

def test_remat(simple_params):
    w, x, lora_params = simple_params

    h = jax.random.normal(jax.random.PRNGKey(0), (x.shape[1],))
    def f(w, x):
        return w @ x + h

    f = jax.remat(f)
    lora_f = jax.jit(lora(f))

    expected = f(w, x)
    res = lora_f(lora_params, x)
    assert jnp.allclose(expected, res, rtol=1e-4)

def test_transpose(simple_params):
    w, x, lora_params = simple_params
    def f(w, x):
        return x.T @ w.T

    lora_f = jax.jit(lora(f))

    expected = f(w, x)
    res = lora_f(lora_params, x)
    print(f'Gap: {jnp.max(jnp.abs(expected - res)):.3e}')
    assert jnp.allclose(expected, res, atol=1e-6)

@pytest.mark.parametrize('lora_first,contract_lora,contract_x,x_ndim', [
    (lf, cl, cx, nd) for lf, cl, cx, nd in
    product([True, False], [0, 1], [0, 1, 2], [2, 3])
    if cx < nd
])
def test_dot_contraction(simple_params, lora_first, contract_lora, contract_x, x_ndim):
    w, _, lora_params = simple_params
    def f(w, x):
        lhs = w
        rhs = x
        lhs_contract = contract_lora
        rhs_contract = contract_x
        if not lora_first:
            lhs_contract, rhs_contract = rhs_contract, lhs_contract
            lhs, rhs = rhs, lhs

        return jax.lax.dot_general(
            lhs,
            rhs,
            (((lhs_contract,), (rhs_contract,)), ((), ()))
        )

    x_shape = [23]
    if x_ndim == 3:
        x_shape.append(29)

    contract_size = w.shape[contract_lora]
    x_shape.insert(contract_x, contract_size)

    x = jax.random.normal(jax.random.PRNGKey(0), x_shape)

    expected = f(w, x)

    lora_f = lora(f)
    lora_result = lora_f(lora_params, x)

    print(f'Gap: {jnp.max(jnp.abs(expected - lora_result)):.3e}')
    assert jnp.allclose(expected, lora_result, atol=1e-5)

def test_cast(simple_params):
    w, x, lora_params = simple_params
    def f(w, x):
        return w.astype(jnp.float16) @ x.astype(jnp.float16)

    lora_f = lora(f)

    expected = f(w, x)
    res = lora_f(lora_params, x)

    print(f'Gap: {jnp.max(jnp.abs(expected - res)):.3e}')
    assert jnp.allclose(expected, res, atol=1e-2)

def test_warning(simple_params):
    _, _, lora_params = simple_params
    def f(w):
        return w[:10, 3:]

    lora_f = lora(f)

    with pytest.warns(UserWarning, match='materialized'):
        lora_f(lora_params)

