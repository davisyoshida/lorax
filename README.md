# Lorax: LoRA for JAX functions
This is a JAX transform which implements [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). LoRA replaces operations like `Wx` with `(W + BA)x` where `A` and `B` are skinny rectangular matrices. You can then train only `A` and `B`, and leave `W` frozen, which dramatically reduces the amount of memory needed for things like optimizer states.

Lorax should work on most JAX models. I did my testing with my models which use Haiku, and you can find an example of applying it to a HuggingFace Flax model in the [examples directory(examples/).

## Installation 

```bash
git clone https://github.com/davisyoshida/lorax.git
cd lorax
pip install -e .
```

## Minimal example
Lorax makes it so you can take model code which wasn't written with LoRA in mind, and transform it so that it does! For example, consider the following MLP code:

```python
import jax
import jax.numpy as jnp

import optax

def model(params, x):
    """My model, written in the dark ages before LoRA, using gratuitous amounts of VRAM when trained"""
    for massive_w in params:
        x = jax.nn.relu(x @ massive_w)
    return jnp.sum(x)

dim = 5000

# Initialize about 3 GB of params
params = [jax.random.normal(jax.random.PRNGKey(i), (dim, dim)) / (dim ** 0.5) for i in range(30)]
optimizer = optax.adam(learning_rate=3e-4)

opt_state = optimizer.init(params)
# OOM on 7 GB GPU :(
```

The optimizer states are way too expensive, but applying Lorax lets you just train two `5000 x 64` matrices for each original weight.

First import lorax and transform your model:
```python
from lorax import lora
lora_model = lora(model)
```

Next initialize the new LoRA parameters:
```python
from lorax import init_lora

# Tell LoRA what what you want the inner dimension of B and A to be for each parameter
rank_constraint = 64
lora_spec = [rank_constraint for param in params]

# Initialize a set of LoRA factors for each parameter
frozen_params, tunable_params = init_lora(param_tree=params, spec=lora_spec, rng=jax.random.PRNGKey(0))

# The transformed model takes this tuple in place of the original params
lora_model((frozen_params, tunable_params), jnp.ones((dim,)))
```

That's it for the Lorax specific stuff. Training is just just like normal JAX training, but make sure to take your gradients with respect to `tunable_params` only
```python
# Define a loss function so we can differentiate with respect to only the tunable params
def loss_fn(tunable_params, frozen_params, x):
    combined_params = (frozen_params, tunable_params)
    return lora_model(combined_params, x)

# Standard taining setup
@jax.jit
def update_fn(frozen_params, tunable_params, opt_state, x):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(tunable_params, frozen_params, x)

    updates, new_opt_state = optimizer.update(grad, opt_state)
    updated_params = optax.apply_updates(tunable_params, updates)
    return loss, new_opt_state, updated_params
    
opt_state = optimizer.init(tunable_params)
```

Now for some dummy data and the training loop:
```python
x = jax.random.normal(jax.random.PRNGKey(0), (dim,))
for i in range(10):
    loss, opt_state, tunable_params = update_fn(frozen_params, tunable_params, opt_state, x)
    print(f'Step: {i} loss: {loss:.4e}')
# Step: 0 loss: 6.6614e-02
# Step: 1 loss: 4.4402e-02
# Step: 2 loss: 3.0241e-02
# Step: 3 loss: 1.8457e-02
# Step: 4 loss: 1.2326e-02
# Step: 5 loss: 8.8878e-03
# Step: 6 loss: 6.0599e-03
# Step: 7 loss: 4.3899e-03
# Step: 8 loss: 3.0839e-03
# Step: 9 loss: 2.2423e-03
```

Number goes down! We can now merge the trained LoRA params with the frozen params, and use them with the unmodified model:
```python
lora_output = lora_model((frozen_params, tunable_params), x)

# Now we merge the params to get params usable in the original model
merged_params = merge_params(frozen_params, tunable_params)
orig_model_output = model(merged_params, x)

# Verify that the model outputs are the same
print(f'Difference between split and merged outputs: {orig_model_output - lora_output:.3e}')
# Difference between split and merged params: 1.164e-10
```

See [examples/huggingface_gpt2.py](examples/huggingface_gpt2.py) for an example applying Lorax to a realistic model.
