import torch


def encode_prompt(text, tokenizer):
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return bos + tokenizer(text, add_special_tokens=False)["input_ids"]


def generate_new_text(text, model, tokenizer, max_new_tokens=16):
    input_ids = torch.tensor([encode_prompt(text, tokenizer)], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
        )

    generated_sequences = output[0, input_ids.shape[1]:]
    decoded_text = tokenizer.decode(generated_sequences, skip_special_tokens=True)

    return decoded_text