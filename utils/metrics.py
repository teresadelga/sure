import re
import torch

def get_ll(text, model, tokenizer):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(model.device)
        labels = tokenized.input_ids
        output = model(**tokenized, labels=labels)
        avg_ll = -output.loss.item() # Loss is already averaged over tokens
    return avg_ll

def compute_exact_match(data):
    """
    data: list of tuples (predicted, expected)
    Returns: exact match accuracy ignoring punctuation but keeping '&'.
    """
    correct = 0
    for pred, exp in data:
        if pred.lower() == exp.lower():
            correct += 1
    return correct / len(data)
