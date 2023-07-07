from dataclasses import dataclass
from functools import partial
import warnings

import jax
import qax

def lora(f):
    """
    Alias for qax.use_implicit_args to reduce necessary modification to code
    using older version of Lorax
    """
    return qax.use_implicit_args(f)

@dataclass
class LoraWeight(qax.ImplicitArray):
    w : qax.ArrayValue # M x N
    a : qax.ArrayValue # k x N
    b : qax.ArrayValue # M x k

    alpha : float = qax.aux_field(default=1.)

    def __post_init__(self):
        super().__post_init__()
        assert self.a.shape[-2] == self.b.shape[-1]
        assert self.w.shape[-2] == self.b.shape[-2]
        assert self.w.shape[-1] == self.a.shape[-1]

    def materialize(self):
        return (self.w + self.get_scale() * self.b @ self.a).astype(self.w.dtype)

    def get_scale(self):
        return self.alpha / self.b.shape[1]

def _check_dot_dimension_numbers(dimension_numbers):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        warnings.warn('Lorax does not support batched matmuls')
        return False
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        warnings.warn('Lorax only supports matmul')
        return False
    return True

@qax.primitive_handler('dot_general')
def handle_dot_lhs(primitive, lora : LoraWeight, rhs: qax.ArrayValue, *, dimension_numbers, **kwargs):
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented

    if isinstance(rhs, LoraWeight):
        rhs = rhs.materialize()
        warnings.warn('Encountered product of two LoraWeights. Materializing the rhs')

    op = partial(jax.lax.dot_general, **kwargs)


    lhs_contract, = dimension_numbers[0][0]

    first, second = (lora.a, lora.b) if lhs_contract == 1 else (lora.b, lora.a)

    first *= lora.get_scale()

    orig = op(lora.w, rhs, dimension_numbers=dimension_numbers)
    lora_product = op(first, rhs, dimension_numbers=dimension_numbers)

    second_dimension_numbers = ((lhs_contract,), (0,)), dimension_numbers[1]

    lora_product = op(second, lora_product, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)

@qax.primitive_handler('dot_general')
def handle_dot_rhs(primitive, lhs : jax.Array, lora: LoraWeight, *, dimension_numbers, **kwargs):
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented
    op = partial(jax.lax.dot_general, **kwargs)

    rhs_contract, = dimension_numbers[0][1]
    first, second = (lora.a, lora.b) if rhs_contract == 1 else (lora.b, lora.a)

    first *= lora.get_scale()

    orig = op(lhs, lora.w, dimension_numbers=dimension_numbers)
    lora_product = op(lhs, first, dimension_numbers=dimension_numbers)

    second_dimension_numbers = ((lhs.ndim - 1), (rhs_contract,)), dimension_numbers[1]

    lora_product = op(lora_product, second, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@qax.primitive_handler('conv_general_dilated')
def handle_conv(primitive, inp : qax.ArrayValue, lora : LoraWeight, *, dimension_numbers, **params):
    if isinstance(inp, LoraWeight):
        warnings.warn('Using a LoraWeight as input to a convolution is not supported, so it will be materialized.')
        inp = inp.materialize()

    if not dimension_numbers.rhs_spec[:1] != (
        len(dimension_numbers.rhs_spec) - 1,
        len(dimension_numbers.rhs_spec) - 2,
    ):
        raise ValueError('Lorax only supports convolutions with shape (..., in_features, out_features)')

    params = {**params, 'dimension_numbers': dimension_numbers}
    op = partial(jax.lax.conv_general_dilated, **params)
    orig = op(inp, lora.w)

    lora_product = op(inp, lora.b)

    params['window_strides'] = (1,) * (len(dimension_numbers.rhs_spec) - 2)
    params['padding'] = 'VALID'
    lora_product = jax.lax.conv_general_dilated(
        lora_product,
        lora.a * lora.get_scale(),
        **params
    )

    return (orig + lora_product).astype(orig.dtype)

@qax.primitive_handler('gather')
def handle_gather(primitive, lora : LoraWeight, indices : jax.Array, *, dimension_numbers, slice_sizes, **params):
    if dimension_numbers.offset_dims != (len(indices.shape) - 1,):
        return NotImplemented

    lora_dim = lora.b.shape[-1]

    if slice_sizes != (1, lora.a.shape[1]):
        return NotImplemented

    params = {**params, 'dimension_numbers': dimension_numbers}

    orig = jax.lax.gather(lora.w, indices, slice_sizes=slice_sizes, **params)

    new_slice_sizes = (1, lora_dim)

    lora_product = jax.lax.gather(lora.b, indices, slice_sizes=new_slice_sizes, **params)
    lora_product = lora_product @ (lora.a * lora.get_scale())

    return (orig + lora_product).astype(orig.dtype)

@qax.primitive_handler('transpose')
def eval_lora_transpose(primitive, arg : LoraWeight, *, permutation):
    if not len(arg.shape) == 2 and permutation == (1, 0):
        return NotImplemented

    return LoraWeight(
        w=arg.w.T,
        a=arg.b.T,
        b=arg.a.T,
        alpha=arg.alpha,
    )

@qax.primitive_handler('convert_element_type')
def eval_lora_convert_element_type(primitive, arg : LoraWeight, **params):
    result = jax.tree_map(
        partial(qax.default_handler, primitive, **params),
        arg
    )
    result.dtype = params['new_dtype']
    return result
