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

# OOM on my 7GB GPU :(
# opt_state = optimizer.init(params)

from lorax import lora, init_lora, merge_params

# Transform the model code
lora_model = lora(model)

# Tell LoRA what to use as the small dimension of B and A
rank_constraint = 64
lora_spec = [rank_constraint for param in params]

# Initialize a set of LoRA factors for each parameter
frozen_params, tunable_params = init_lora(param_tree=params, spec=lora_spec, rng=jax.random.PRNGKey(0))

# The transformed model takes this tuple in place of the original params
lora_model((frozen_params, tunable_params), jnp.ones((dim,)))

# Now initialize the optimizer state for only the trainable params
opt_state = optimizer.init(tunable_params)

# Define a loss function so we can differentiate with respect to only the tunable params
def loss_fn(tunable_params, frozen_params, x):
    combined_params = (frozen_params, tunable_params)
    return lora_model(combined_params, x)

@jax.jit
def update_fn(frozen_params, tunable_params, opt_state, x):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(tunable_params, frozen_params, x)

    updates, new_opt_state = optimizer.update(grad, opt_state)
    updated_params = optax.apply_updates(tunable_params, updates)
    return loss, new_opt_state, updated_params

x = jax.random.normal(jax.random.PRNGKey(0), (dim,))
for i in range(10):
    loss, opt_state, tunable_params = update_fn(frozen_params, tunable_params, opt_state, x)
    print(f'Step: {i} loss: {loss:.4e}') # Number goes down!

# Save the output to verify correctness
lora_output = lora_model((frozen_params, tunable_params), x)

# Now we merge the params to get params usable in the original model
merged_params = merge_params(frozen_params, tunable_params)
orig_model_output = model(merged_params, x)

# Verify that the model outputs are the same
print(f'Difference between split and merged outputs: {orig_model_output - lora_output:.3e}')
