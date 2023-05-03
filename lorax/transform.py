from functools import partial, wraps
from itertools import chain, repeat
import logging
import warnings

import jax
from jax import api_util
import jax.linear_util as lu
import jax.numpy as jnp
from  jax.util import safe_map

LORA_FREEZE = 0
LORA_FULL = -1

class LoraNode:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return f'{type(self).__name__}(a={self.a}, b={self.b})'

    def __repr__(self):
        return str(self)

    def evaluate(self):
        return self.b @ self.a

class EmptyNodeCls:
    pass

jax.tree_util.register_pytree_node(
    LoraNode,
    lambda node: ((node.a, node.b), None),
    lambda _, xs: LoraNode(*xs)
)

EmptyNode = EmptyNodeCls()
jax.tree_util.register_pytree_node(EmptyNodeCls, lambda _: ((), None), lambda _, x: EmptyNode)

def leaf_pred(x):
    return x is EmptyNode or isinstance(x, LoraNode)

custom_tree_map = partial(jax.tree_util.tree_map, is_leaf=leaf_pred)
custom_tree_leaves = partial(jax.tree_util.tree_leaves, is_leaf=leaf_pred)

def reversed_dot(lhs, rhs, dimension_numbers):
    return jax.lax.dot_general(
        rhs,
        lhs,
        dimension_numbers

    )

def lora_to_orig(freeze_param, tune_param):
    if freeze_param is EmptyNode:
        return tune_param
    return freeze_param

def init_lora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32):
    def freeze_getter(param, spec_val):
        if spec_val == LORA_FULL:
            return EmptyNode
        return param

    def tune_getter(path, param, spec_val):
        if spec_val == LORA_FREEZE:
            return EmptyNode
        if spec_val == LORA_FULL:
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')
        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=param.dtype)
            a = jax.random.normal(rng, (spec_val, a_dim), dtype=param.dtype) * stddev
            return LoraNode(a, b)

        # conv case
        print(path, param.shape)
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        return LoraNode(a, b)

    return jax.tree_map(freeze_getter, param_tree, spec), jax.tree_util.tree_map_with_path(tune_getter, param_tree, spec)

def lora(f, argnums=0):
    if isinstance(argnums, int):
        argnums = (argnums,)

    @wraps(f)
    def wrapper(*args, **kwargs):
        orig_args = [*args]
        for argnum in argnums:
            orig_args[argnum] = custom_tree_map(lora_to_orig, *args[argnum])
            assert not any(node is EmptyNode for node in custom_tree_leaves(orig_args[argnum]))

        shape_args, shape_kwargs = jax.tree_map(
            lambda x: jax.core.get_aval(x) if isinstance(x, jax.core.Tracer) else x, 
            (orig_args, kwargs)
        )
        closed_jaxpr = jax.make_jaxpr(f)(*shape_args, **shape_kwargs)
        out_shape = jax.eval_shape(f, *shape_args, **shape_kwargs)
        out_structure = jax.tree_util.tree_structure(out_shape)

        jaxpr = closed_jaxpr.jaxpr

        arg_offsets = []
        curr = 0
        for tree in orig_args:
            arg_offsets.append(curr)
            curr += curr + len(jax.tree_util.tree_leaves(tree))

        lora_arg_vals = {}
        inp_iter = iter(jaxpr.invars)
        for i, arg in enumerate(args):
            if i in argnums:
                frozen_leaves, lora_leaves = (custom_tree_leaves(a) for a in arg)
            else:
                frozen_leaves = repeat(EmptyNode)
                lora_leaves = jax.tree_util.tree_leaves(arg)

            for (frozen_leaf, lora_leaf, name) in zip(frozen_leaves, lora_leaves, inp_iter):
                lora_arg_vals[name] = (frozen_leaf, lora_leaf)

        result = lora_interpreter(jaxpr, lora_arg_vals, closed_jaxpr.literals)
        unflattened_result = jax.tree_util.tree_unflatten(out_structure, result)

        return unflattened_result
    return wrapper

def materialize_val(val):
    if not isinstance(val, tuple):
        return val

    freeze, tune = val
    if freeze is EmptyNode:
        return tune
    if tune is EmptyNode:
        return freeze
    full = freeze + tune.b @ tune.a
    warnings.warn(f'LoRA matrix of shape {full.shape} was materialized')
    return full

def is_lora_tuple(val):
    return isinstance(val, tuple) and isinstance(val[1], LoraNode)

def lora_interpreter(jaxpr, args, literals):
    env = dict(args)

    def read(var):
        if isinstance(var, jax.core.Literal):
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.constvars, literals)


    lora_fns = {
        'dot_general': eval_lora,
        'conv_general_dilated': eval_lora_conv,
        'gather': eval_lora_gather,
        'transpose': eval_lora_transpose
    }

    for eqn in jaxpr.eqns:
        # TODO: run inside other interpreters in a smarter way
        use_default_eval = True
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        args = safe_map(read, eqn.invars)
        if eqn.primitive.name == 'pjit' and eqn.params['name'] == '_einsum':
            params = dict(eqn.params)
            pjit_jaxpr = params.pop('jaxpr')
            literals = pjit_jaxpr.literals
            subjaxpr = pjit_jaxpr.jaxpr
            sub_fn_inputs = {name: val for name, val in zip(subjaxpr.invars, args)}

            ans = jax.experimental.pjit.pjit(
                partial(lora_interpreter, subjaxpr),
            )(sub_fn_inputs, literals)
            use_default_eval = False
        elif eqn.primitive.name == 'remat2':
            subjaxpr = eqn.params['jaxpr']
            sub_fn_inputs = {name: val for name, val in zip(subjaxpr.invars, args)}
            ans = jax.remat(
                partial(lora_interpreter, subjaxpr)
            )(sub_fn_inputs, [])
            use_default_eval = False
        elif eqn.primitive.name in lora_fns:
            if any(safe_map(is_lora_tuple, args)):
                ans = lora_fns[eqn.primitive.name](eqn, *args)
                use_default_eval = ans is None

        if use_default_eval:
            materialized_args = safe_map(materialize_val, args)
            ans = eqn.primitive.bind(*subfuns, *materialized_args, **bind_params)
        if not eqn.primitive.multiple_results:
            ans = [ans]
        safe_map(write, eqn.outvars, ans)

    return safe_map(read, jaxpr.outvars)

def eval_lora(eqn, lhs, rhs):
    dimension_numbers = eqn.params['dimension_numbers']
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        warnings.warn('Lorax does not support batched matmuls')
        return None
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        warnings.warn('Lorax only supports matmul')
        return None

    use_lhs = is_lora_tuple(lhs)

    use_rhs = is_lora_tuple(rhs)
    if use_lhs and use_rhs:
        warnings.warn('Product of two LoRA matrices is not implemented so RHS will be materialized')
        use_rhs = False

    if use_lhs:
        a_first = lhs_contract[0] == 1
        fixed_arg = materialize_val(rhs)
        frozen, lora = lhs
    elif use_rhs:
        a_first = rhs_contract[0] == 1
        fixed_arg = materialize_val(lhs)
        frozen, lora = rhs
    else:
        raise ValueError('No lora node')

    fn = jax.lax.dot_general if use_lhs else reversed_dot
    orig_product = fn(
        frozen,
        fixed_arg,
        dimension_numbers=dimension_numbers
    )

    first, second = (lora.a, lora.b) if a_first else (lora.b, lora.a)
    lora_product = fn(
        first,
        fixed_arg,
        dimension_numbers=dimension_numbers
    )

    lora_product = fn(
        second,
        lora_product,
        dimension_numbers=dimension_numbers
    )
    return orig_product + lora_product

def eval_lora_conv(eqn, inp, kernel):
    if not is_lora_tuple(kernel):
        return None

    if is_lora_tuple(inp):
        warnings.warn('Lorax only supports convolutions with the a LoRA kernel, so the input will be materialized')

    inp = materialize_val(inp)

    dimension_numbers = eqn.params['dimension_numbers']
    if not dimension_numbers.rhs_spec[:1] != (
        len(dimension_numbers.rhs_spec) - 1,
        len(dimension_numbers.rhs_spec) - 2,
    ):
        raise ValueError('Lorax only supports convolutions with shape (..., in_features, out_features)')

    frozen, lora = kernel
    orig = jax.lax.conv_general_dilated(
        inp,
        frozen,
        **eqn.params
    )

    kwargs = eqn.params.copy()
    lora_product = jax.lax.conv_general_dilated(
        inp,
        lora.b,
        **kwargs
    )

    kwargs['window_strides'] = (1,) * (len(dimension_numbers.rhs_spec) - 2)
    kwargs['padding'] = 'VALID'
    lora_product = jax.lax.conv_general_dilated(
        lora_product,
        lora.a,
        **kwargs
    )
    return orig + lora_product

def eval_lora_gather(eqn, arr, indices):
    if not is_lora_tuple(arr):
        return None

    indices = materialize_val(indices)

    dimension_numbers = eqn.params['dimension_numbers']
    if dimension_numbers.offset_dims != (len(indices.shape) - 1,):
        return None

    frozen, lora = arr
    constraint_dim = lora.b.shape[-1]

    slice_sizes = eqn.params['slice_sizes']

    if slice_sizes != (1, lora.a.shape[1]):
        return None

    new_params = eqn.params.copy()
    new_params['slice_sizes'] = (1, constraint_dim)


    orig = jax.lax.gather(frozen, indices, **eqn.params)

    lora_product = jax.lax.gather(lora.b, indices, **new_params)

    lora_product = lora_product @ lora.a
    return orig + lora_product

def eval_lora_transpose(eqn, arg):
    if not len(arg[0].shape) == 2 and eqn.params['permutation'] == (1, 0):
        return None
    frozen, lora = arg

    frozen_T = frozen.T
    lora_T = LoraNode(lora.b.T, lora.a.T)
    return frozen_T, lora_T
