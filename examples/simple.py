from functools import partial
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

# OOM on 7GB GPU :(
# opt_state = optimizer.init(params)

import lorax

# Transform the model code
lora_model = lorax.lora(model)

# Tell LoRA what to use as the small dimension of B and A
rank_constraint = 64
lora_spec = [rank_constraint for param in params]

# Initialize a set of LoRA factors for each parameter
lora_params = lorax.init_lora(param_tree=params, spec=lora_spec, rng=jax.random.PRNGKey(0))

# The transformed model has the same call signature, but it can now handle parameters
# of type lorax.LoraWeight
lora_model(lora_params, jnp.ones((dim,)))

# Wrap the optimizer so it will freeze parameters not marked as trainable by the spec
optimizer = lorax.wrap_optimizer(optimizer, lora_spec)

# Now the optimizer can be used just like normal
opt_state = optimizer.init(lora_params)

@partial(jax.jit, donate_argnums=(0, 1))
def update_fn(lora_params, opt_state, x):
    # The transformed model function is compatible with all the normal JAX transforms
    # It's just a function which maps pytrees to pytrees
    grad_fn = jax.value_and_grad(lora_model)
    loss, grad = grad_fn(lora_params, x)

    updates, new_opt_state = optimizer.update(grad, opt_state, params=lora_params)
    updated_params = optax.apply_updates(lora_params, updates)
    return loss, new_opt_state, updated_params

x = jax.random.normal(jax.random.PRNGKey(0), (dim,))
for i in range(10):
    loss, opt_state, lora_params = update_fn(lora_params, opt_state, x)
    print(f'Step: {i} loss: {loss:.4e}') # Number goes down!

# Save the output to verify correctness
lora_output = lora_model(lora_params, x)

# Now we merge the params to get params usable in the original model
merged_params = lorax.merge_params(lora_params)
orig_model_output = model(merged_params, x)

# Verify that the model outputs are the same
print(f'Difference between split and merged outputs: {orig_model_output - lora_output:.3e}')
