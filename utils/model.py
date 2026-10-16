from peft import get_peft_model

def prepare_model(base_model, peft_config, MODEL_NAME):
    model = get_peft_model(base_model, peft_config)
    model.train()
    if MODEL_NAME == "google-t5/t5-large":
        model.config.use_cache = False
    else:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.dropout = 0.1
    model.print_trainable_parameters()
    return model
