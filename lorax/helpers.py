import jax
import jax.numpy as jnp
from jax.tree_util import tree_map_with_path, DictKey, SequenceKey

import optax
import qax

from .constants import LORA_FREEZE, LORA_FULL
from .transform import LoraWeight


def init_lora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1., is_leaf=None):
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(rng)

    def get_param(path, param, spec_val):
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype) * stddev
            return LoraWeight(w=param, a=a, b=b, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        return LoraWeight(param, a, b, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)

def simple_spec(params, decision_fn=None, tune_vectors=False, is_leaf=None):
    """
    Create a simple lora spec for a pytree
    Args:
        params: pytree of parameters
        tune_vectors: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function which maps a Jax KeyPath and a parameter to a spec value
    """
    if decision_fn is None:
        def decision_fn(*args):
            return LORA_FREEZE

    def full_fn(path, arr):
        if len(arr.shape) < 2:
            return LORA_FULL if tune_vectors else LORA_FREEZE

        value = decision_fn(path, arr)
        return value

    return tree_map_with_path(full_fn, params, is_leaf=is_leaf)

def merge_params(lora_params, destructive=True, use_scaling=True):
    """
    Re-merge LoRA parameters.
    Arguments:
        destructive: If True, the buffers in frozen_params may be freed to save memory.
        use_scaling: Whether to multiply LoRA params by alpha/r
    """
    if not use_scaling:
        raise ValueError('Scaling is now always enabled to match the original LoRA implementation.')

    def _ensure_delete(val):
        if not isinstance(val, jax.Array) or val.is_deleted():
            return

        val.device_buffer.delete()


    materializer = jax.jit(qax.materialize_nested, donate_argnums=0 if destructive else ())
    def map_fn(param):
        if isinstance(param, LoraWeight):
            result = materializer(param)
            if destructive:
                jax.tree_map(_ensure_delete, param)
            return result
        return param

    return qax.utils.tree_map_with_implicit(map_fn, lora_params)

def split_lora_params(params, spec):
    """
    Map params to a pytree in which all `LoraWeight.w` values and all params marked with
    LORA_FREEZE are replaced with qax.EmptyNode. This is useful for checkpointing just
    the trainable params.
    """
    def node_mapper(node, spec_val):
        if not isinstance(node, LoraWeight):
            return node if spec_val != LORA_FREEZE else qax.EmptyNode
        children, aux = node.tree_flatten_with_keys()
        idx = next(i for i, (key, _) in enumerate(children) if key == 'w')
        children[idx] = ('w', qax.EmptyNode)

        return LoraWeight.tree_unflatten(aux, [c for _, c in children])

    return qax.utils.tree_map_with_implicit(node_mapper, params, spec)

def wrap_optimizer(optimizer : optax.GradientTransformation, spec, scalar_frozen_grads=False):
    full_freeze_labels = jax.tree_map(
        lambda x: 'freeze' if x == LORA_FREEZE else 'train',
        spec
    )
    optimizer_with_full_freeze = qax.utils.freeze_subtrees(
        optimizer,
        full_freeze_labels,
        use_scalar_zeros=scalar_frozen_grads
    )

    return qax.freeze_keys(optimizer_with_full_freeze, LoraWeight, 'w', use_scalar_zeros=scalar_frozen_grads)
