import warnings

import jax
import jax.numpy as jnp
import optax
from transformers import FlaxGPT2LMHeadModel

from lorax import simple_spec, init_lora, lora, LORA_FULL, merge_params

def main():
    model = FlaxGPT2LMHeadModel.from_pretrained('gpt2')

    # Wrap the forward pass in so that lorax knows which params to LoRA-fy (it only does the first argument by default)
    @lora
    def lora_forward(params, input_ids):
        return model(input_ids, params=params)

    # This function defines a spec which tells lorax how each parameter should be handled
    def decision_fn(path, param):
        if 'embedding' in path:
            print(f'Fully finetuning param {path}')
            return LORA_FULL
        dim = 32
        print(f'Using LoRA with dim={dim} for param {path}')
        return dim

    # Create a pytree with the same shape as params indicating how each parameter should be handled
    lora_spec = simple_spec(model.params, decision_fn=decision_fn, tune_vectors=True)

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    freeze_params, tune_params = init_lora(model.params, lora_spec, jax.random.PRNGKey(0))

    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-4)

    # Make sure to only pass the tunable parameters to the optimizer
    opt_state = optimizer.init(tune_params)

    # The loss function should take the tunable and frozen params separately so
    # you can differentiate w.r.t. the tunable ones only
    def loss_fn(tunable_params, frozen_params, batch):
        input_ids = batch[:, :-1]
        logits = lora_forward((frozen_params, tunable_params), input_ids).logits

        logprobs = jax.nn.log_softmax(logits)
        target_logprobs = jnp.take_along_axis(logprobs, batch[:, 1:, None], axis=-1)
        return -jnp.mean(target_logprobs)

    @jax.jit
    def update_fn(tunable_params, frozen_params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(tunable_params, frozen_params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=tunable_params)

        new_tunable_params = optax.apply_updates(tunable_params, updates)
        return new_tunable_params, new_opt_state, loss

    # Train on a dummy batch to demo loss going down
    example_data = jax.random.randint(jax.random.PRNGKey(0), (4, 128), 0, 50257)
    for _ in range(100):
        tune_params, opt_state, loss = update_fn(tune_params, freeze_params, opt_state, example_data)
        print(loss)

    final_predictions = lora_forward((freeze_params, tune_params), example_data).logits
    merged_params = merge_params(freeze_params, tune_params)

    orig_model_predictions = model(example_data, params=merged_params).logits

    gap = jnp.max(jnp.abs(final_predictions - orig_model_predictions))
    print(f'Max prediction gap: {gap:.3e}')

if __name__ == '__main__':
    main()
