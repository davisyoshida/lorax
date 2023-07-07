import jax
import jax.numpy as jnp
import optax
import qax

from lorax import LORA_FREEZE, LORA_FULL
from lorax.helpers import split_lora_params, merge_params, wrap_optimizer

def test_split(simple_params):
    _, _, params = simple_params

    tree = {'x': params, 'y': [params, jnp.zeros(5)]}
    spec = {'x': 5, 'y': [5, LORA_FULL]}

    split = split_lora_params(tree, spec)

    orig_struct = qax.utils.tree_structure_with_implicit(tree)
    struct = qax.utils.tree_structure_with_implicit(split)

    assert orig_struct == struct

    leaves = qax.utils.tree_leaves_with_implicit(split)

    for lora_leaf in leaves[:2]:
        assert lora_leaf.w is qax.EmptyNode
        assert isinstance(lora_leaf.a, jax.Array)
        assert isinstance(lora_leaf.b, jax.Array)

    assert isinstance(leaves[2], jax.Array)

def test_merge(simple_params):
    w, _, params = simple_params
    params = jax.tree_map(lambda x: jnp.copy(x), params)

    tree = {'x': params, 'y': jnp.zeros(5)}
    structure = qax.utils.tree_structure_with_implicit(tree)

    merged = merge_params(tree)
    assert qax.utils.tree_structure_with_implicit(merged) == structure

def test_wrap_optimizer(simple_params):
    _, x, params = simple_params

    params = {'u': params, 'y': {'z': jnp.ones(2), 'w': jnp.zeros(3)}}
    spec = {'u': 1234, 'y': {'z': LORA_FREEZE, 'w': LORA_FULL}}

    @qax.use_implicit_args
    def f(params, x):
        return jnp.sum(params['u'] @ x) + jnp.sum(params['y']['z']) + jnp.sum(params['y']['w']) 

    grad = jax.grad(f)(params, x)

    opt = wrap_optimizer(optax.sgd(1e-3), spec)
    state = opt.init(params)

    updates, state = opt.update(grad, state, params)

    new_params = optax.apply_updates(params, updates)

    u = params['u']
    new_u = new_params['u']
    assert jnp.all(u.w == new_u.w)
    assert jnp.all(u.a != new_u.a)
    assert jnp.all(u.b != new_u.b)

    assert jnp.all(params['y']['z'] == new_params['y']['z'])
    assert jnp.all(params['y']['w'] != new_params['y']['w'])
