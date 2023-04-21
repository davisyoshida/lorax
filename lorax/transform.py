from functools import partial, wraps
from itertools import chain, repeat
import logging

import jax
from jax import api_util
import jax.linear_util as lu
import jax.numpy as jnp
from  jax.util import safe_map

LORA_FREEZE = 0
LORA_FULL = -1

logger = logging.getLogger(__name__)

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

def init_lora(param_tree, spec, rng, stddev=0.01):
    def freeze_getter(param, spec_val):
        if spec_val == LORA_FULL:
            return EmptyNode
        return param

    def tune_getter(param, spec_val):
        if spec_val == LORA_FREEZE:
            return EmptyNode
        if spec_val == LORA_FULL:
            return param

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=param.dtype)
            a = jax.random.normal(rng, (spec_val, a_dim), dtype=param.dtype) * stddev
            return LoraNode(a, b)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        b = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        return LoraNode(a, b)

    return jax.tree_map(freeze_getter, param_tree, spec), jax.tree_map(tune_getter, param_tree, spec)

def lora(f, argnums=0):
    if isinstance(argnums, int):
        argnums = (argnums,)

    @wraps(f)
    def wrapper(*args, **kwargs):
        orig_args = [*args]
        for argnum in argnums:
            orig_args[argnum] = custom_tree_map(lora_to_orig, *args[argnum])
            assert not any(node is EmptyNode for node in custom_tree_leaves(orig_args[argnum]))

        closed_jaxpr = jax.make_jaxpr(f)(*orig_args, **kwargs)
        out_shape = jax.eval_shape(f, *orig_args, **kwargs)
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

def lora_interpreter(jaxpr, args, literals):
    env = {}

    def read(var):
        if isinstance(var, jax.core.Literal):
            return var.val
        if var in env:
            return env[var]
        value = args[var]
        if not isinstance(value, tuple):
            return value

        freeze, tune = value
        if freeze is EmptyNode:
            return tune
        if tune is EmptyNode:
            return freeze

        logger.warning('LoRa matrix was materialized')
        return tune.b @ tune.a + freeze

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.constvars, literals)

    def eval_lora(eqn):
        lhs = eqn.invars[0]
        rhs = eqn.invars[1]
        if not any(arg in args and isinstance(args[arg][1], LoraNode) for arg in (lhs, rhs)):
            return None

        dimension_numbers = eqn.params['dimension_numbers']
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
        if lhs_batch or rhs_batch:
            logger.warning('Lorax does not support batched matmuls')
            return None
        if len(lhs_contract) != 1 or len(rhs_contract) != 1:
            logger.warning('Lorax only supports matmul')
            return None

        lhs_arg = args.get(lhs)
        if lhs_arg:
            frozen, lora = lhs_arg
            use_lhs = isinstance(lora, LoraNode)
        else:
            use_lhs = False

        rhs_arg = args.get(rhs)
        if rhs_arg:
            use_rhs = isinstance(rhs_arg[1], LoraNode)
            if use_rhs:
                if use_lhs:
                    logger.warning('Product of two LoRa matrices is not implemented so RHS was materialized')
                    use_rhs = False
                else:
                    frozen, lora = rhs_arg

        a_first = lhs_contract[0] == 1
        if use_lhs:
            fixed_arg = read(rhs)
        elif use_rhs:
            fixed_arg = read(lhs)
        else:
            raise ValueError('No lora node')


        fn = jax.lax.dot_general if use_lhs else reversed_dot
        logger.info('Evaluating %s with dimension_numbers %s', fn, dimension_numbers)
        orig_product = fn(
            frozen,
            fixed_arg,
            dimension_numbers=dimension_numbers
        )

        first, second = (lora.a, lora.b) if a_first else (lora.b, lora.a)
        logger.info(f'a_first: {a_first} a: {lora.a.shape} b: {lora.b.shape}')
        logger.info(f'First: {first.shape} Fixed: {fixed_arg.shape}')
        lora_product = fn(
            first,
            fixed_arg,
            dimension_numbers=dimension_numbers
        )

        logger.info(f'Second: {second.shape} Lora: {lora_product.shape}')
        lora_product = fn(
            second,
            lora_product,
            dimension_numbers=dimension_numbers
        )
        return orig_product + lora_product

    def eval_lora_conv(eqn):
        inp = read(eqn.invars[0])
        kernel = eqn.invars[1]
        if not (kernel in args and isinstance(args[kernel][1], LoraNode)):
            return None
        dimension_numbers = eqn.params['dimension_numbers']
        if not dimension_numbers.rhs_spec[:1] != (
            len(dimension_numbers.rhs_spec) - 1,
            len(dimension_numbers.rhs_spec) - 2,
        ):
            raise ValueError('Lorax only supports convolutions with shape (..., in_features, out_features)')

        frozen, lora = args[kernel]
        orig = jax.lax.conv_general_dilated(
            inp,
            frozen,
            **eqn.params
        )

        kwargs = eqn.params.copy()
        lora_product = jax.lax.conv_general_dilated(
            inp,
            lora.a,
            **kwargs
        )

        kwargs['window_strides'] = (1,) * (len(dimension_numbers.rhs_spec) - 2)
        kwargs['padding'] = 'VALID'
        lora_product = jax.lax.conv_general_dilated(
            lora_product,
            lora.b,
            **kwargs
        )
        return orig + lora_product

    for eqn in jaxpr.eqns:
        used_lora = False
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        if eqn.primitive.name == 'pjit' and eqn.params['name'] == '_einsum':
            params = eqn.params
            pjit_jaxpr = params.pop('jaxpr').jaxpr
            sub_fn_inputs = {}
            for outer_arg, inner_arg in zip(eqn.invars, pjit_jaxpr.invars):
                if outer_arg in env:
                    sub_fn_inputs[inner_arg] = env[outer_arg]
                else:
                    sub_fn_inputs[inner_arg] = args[outer_arg]

            ans = lora_interpreter(pjit_jaxpr, sub_fn_inputs, [])
            used_lora = True
        elif eqn.primitive.name == 'dot_general':
            ans = eval_lora(eqn)
            used_lora = ans is not None
        elif eqn.primitive.name == 'conv_general_dilated':
            ans = eval_lora_conv(eqn)
            used_lora = ans is not None

        if not used_lora:
            ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
        if not eqn.primitive.multiple_results:
            ans = [ans]
        safe_map(write, eqn.outvars, ans)

    return safe_map(read, jaxpr.outvars)

def merge_params(frozen_tree, tune_tree):
    def merge(frozen, tune):
        if isinstance(tune, LoraNode):
            tune = tune.b @ tune.a

        if tune is EmptyNode:
            return frozen
        if frozen is EmptyNode:
            return tune
        return tune + frozen

    return jax.tree_map(merge, frozen_tree, tune_tree)

def f(params, x):
    res = jnp.einsum('ij,klj->kli', params['W'], x)
    return res + params['b'][None, None]

lora_f = lora(f)

def main():
    from pprint import pprint
    W = jnp.array([[1., 2], [3, 4]])
    x = jnp.arange(30).reshape(3, 5, 2).astype(jnp.float32)
    params = {'W': W, 'b': jnp.ones(2)}
    rng = jax.random.PRNGKey(0)
    lora_params = init_lora(params, {'W': 2, 'b': LORA_FULL}, rng)
    pprint(lora_params)
    print(f'Orig: {f(params, x)}')
    print(f'Lora: {lora_f(lora_params, x)}')

    jaxpr = jax.make_jaxpr(jax.jit(lora_f))(lora_params, x)
    pprint(jaxpr)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
