import jax
from jax.tree_util import tree_map_with_path, DictKey

from lorax.constants import LORA_FREEZE, LORA_FULL

def simple_spec(params, decision_fn=None, tune_biases=False):
    """
    Create a simple lora spec for a pytree
    Args:
        params: pytree of parameters
        tune_biases: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function that takes a string marking the position in the tree and the parameter,
            and returns the spec value for that parameter
            The position strings are of the form key1/key2/key3 etc. For lists or tuples the key is
            the index
    """
    if decision_fn is None:
        def decision_fn(*args):
            return LORA_FREEZE

    def fulL_fn(path, arr):
        if tune_biases and len(arr.shape) < 2:
            return LORA_FULL

        path_str = '/'.join(str(node.key if isinstance(node, DictKey) else node.idx) for node in path)
        value = decision_fn(path_str, arr)
        return value

    return tree_map_with_path(fulL_fn, params)

