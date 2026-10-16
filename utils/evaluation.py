import math
from utils.metrics import get_ll, compute_exact_match
from utils.inference import generate_new_text
from torch.utils.data import DataLoader
import torch

def evaluate_llama(model, test_ds, task, tokenizer, num_samples_eval, compute_ll=False):
      em = []
      losses = 0
      n_samples = min(num_samples_eval, len(test_ds)) if num_samples_eval else len(test_ds)

      for i in range(n_samples):
          sample = test_ds[i]
          text_answer = sample["label"]

          text = sample["prompt"]
          new_text = generate_new_text(text, model, tokenizer)

          if compute_ll:
            losses += get_ll(text + " " + text_answer, model, tokenizer)

          predicted = new_text.strip().lower()
          expected = text_answer.strip().lower()

          em.append((predicted, expected))

      exact_match = compute_exact_match(em)
      print(f"The exact match for {task} is {exact_match:.2f}")
      metrics = {"exact_match": exact_match, "num_samples": n_samples}
      if compute_ll:
          print(f"The average loss per sequence for {task} is {losses / n_samples:.4f}")
          metrics["avg_log_likelihood"] = losses / n_samples
      return metrics


def evaluate_T5(model, test_ds, task, tokenizer, num_samples_eval=500,  batch_size=32, compute_ll=False, task_prefix="", DEVICE="cuda"):
    model.eval()
    em = []
    losses = 0.0
    n_samples = min(num_samples_eval, len(test_ds))

    loader = DataLoader(range(n_samples), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_indices in loader:
            # Convert tensor indices -> list[int]
            idx = [int(i) for i in batch_indices]

            batch = test_ds[idx]   # dict of lists
            batch_prompts = [task_prefix + p for p in batch["prompt"]]
            batch_labels  = batch["label"]

            # Tokenize & generate in batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)


            em.extend(zip(preds, batch_labels))

    exact_match = compute_exact_match(em)
    print(f"The exact match for {task} is {exact_match:.2f}")
    metrics = {"exact_match": exact_match, "num_samples": n_samples}
    if compute_ll:
        avg_loss = losses / min(100, n_samples)
        print(f"The average loss per sequence for {task} is {avg_loss:.4f}")
        metrics["avg_log_likelihood"] = avg_loss
    return metrics


def evaluate_chat_loss(model, test_ds, task, tokenizer, num_samples_eval, compute_ll=False):
    n_samples = min(num_samples_eval, len(test_ds)) if num_samples_eval else len(test_ds)
    total_nll, total_tokens = 0.0, 0

    with torch.no_grad():
        for i in range(n_samples):
            sample = test_ds[i]
            input_ids = torch.tensor([sample["input_ids"]], device=model.device)
            labels = torch.tensor([sample["labels"]], device=model.device)
            n_tokens = int((labels[:, 1:] != -100).sum())
            if n_tokens == 0:
                continue
            loss = model(input_ids=input_ids, labels=labels).loss
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    avg = total_nll / max(total_tokens, 1)
    print(f"The held-out loss for {task} is {avg:.4f} (perplexity {math.exp(avg):.2f}, "
          f"{n_samples} conversations, {total_tokens} tokens)")
    return {"loss": avg, "perplexity": math.exp(avg), "num_samples": n_samples, "num_tokens": total_tokens}
