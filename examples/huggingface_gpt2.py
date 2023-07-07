import warnings

import jax
import jax.numpy as jnp
import optax
from transformers import FlaxGPT2LMHeadModel

import lorax

def main():
    model = FlaxGPT2LMHeadModel.from_pretrained('gpt2')

    # This function defines a spec which tells lorax how each parameter should be handled
    def decision_fn(path, param):
        if 'embedding' in path:
            print(f'Fully finetuning param {path}')
            return LORA_FULL
        dim = 32
        print(f'Using LoRA with dim={dim} for param {path}')
        return dim

    # Create a pytree with the same shape as params indicating how each parameter should be handled
    # Each leaf will be given one of the following values:
    # - LORA_FULL: The parameter will be fully finetuned
    # - LORA_FREEZE: The parameter will be frozen
    # - k > 0: The parameter will be LoRA tuned with a rank k update

    # Simple_spec is a helper to do this, but you can also create the label pytree yourself
    lora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=True)

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(0))

    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-4)

    # `wrap_optimizer` uses the spec to freeze the appropriate subset
    # of parameters.
    # The frozen parameters won't have optimizer states etc
    # created for them
    optimizer = lorax.wrap_optimizer(optimizer, lora_spec)

    opt_state = optimizer.init(lora_params)

    # lorax.lora wraps a callable so that the arguments can be lorax.LoraWeight
    # instances. (It's actually just an alias for qax.use_implicit_args, so
    # the wrapped function can handle other qax types as well)
    lora_model = lorax.lora(model)

    # No changes are necessary to the loss function apart from using the wrapped model
    def loss_fn(lora_params, batch):
        input_ids = batch[:, :-1]

        # The call signature of the wrapped model is unchanged from the original HuggingFace model
        logits = lora_model(input_ids, params=lora_params).logits

        logprobs = jax.nn.log_softmax(logits)
        target_logprobs = jnp.take_along_axis(logprobs, batch[:, 1:, None], axis=-1)
        return -jnp.mean(target_logprobs)

    # The update function also doesn't need to be modified other than
    # using the wrapped optimizer
    @jax.jit
    def update_fn(lora_params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(lora_params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=lora_params)

        new_params = optax.apply_updates(lora_params, updates)
        return new_params, new_opt_state, loss

    # Train on a dummy batch to show that we can fit the model to stuff
    example_data = jax.random.randint(jax.random.PRNGKey(0), (4, 128), 0, 50257)
    for _ in range(100):
        lora_params, opt_state, loss = update_fn(lora_params, opt_state, example_data)
        print(loss)

    final_predictions = lora_model(example_data, params=lora_params).logits

    # Now let's merge the loras back into the original parameters to get
    # finetuned parameters we can use with no extra compute
    merged_params = lorax.merge_params(lora_params)

    orig_model_predictions = model(example_data, params=merged_params).logits

    gap = jnp.max(jnp.abs(final_predictions - orig_model_predictions))
    print(f'Max prediction gap: {gap:.3e}')

if __name__ == '__main__':
    main()
