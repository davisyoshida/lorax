import jax
from jax.tree_util import tree_map_with_path, DictKey

from lorax.constants import LORA_FREEZE, LORA_FULL
from lorax.transform import EmptyNode

def simple_spec(params, decision_fn=None, tune_vectors=False):
    """
    Create a simple lora spec for a pytree
    Args:
        params: pytree of parameters
        tune_vectors: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function that takes a string marking the position in the tree and the parameter,
            and returns the spec value for that parameter
            The position strings are of the form key1/key2/key3 etc. For lists or tuples the key is
            the index
    """
    if decision_fn is None:
        def decision_fn(*args):
            return LORA_FREEZE

    def full_fn(path, arr):
        if len(arr.shape) < 2:
            return LORA_FULL if tune_vectors else LORA_FREEZE

        path_str = '/'.join(str(node.key if isinstance(node, DictKey) else node.idx) for node in path)
        value = decision_fn(path_str, arr)
        return value

    return tree_map_with_path(full_fn, params)

def merge_params(frozen_params, tunable_params, destructive=True):
    """Re-merge LoRA parameters. If destructive=True is set, buffers in frozen_params may be freed to save memory."""
    def merge(frozen, tunable):
        if tunable is EmptyNode:
            return frozen
        elif frozen is EmptyNode:
            return tunable
        new_param = frozen + tunable.b @ tunable.a
        if destructive:
            frozen.device_buffer.delete()
        return new_param
    return jax.tree_map(merge, frozen_params, tunable_params)
