import torch
import torch.nn.functional as F
import numpy as np

SCORE_MAX_TOKENS = 65536


def score_sequences(model, sequences, max_tokens=SCORE_MAX_TOKENS):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank, world = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if distributed else (0, 1)
    if rank == 0:
        print(f"Scoring {len(sequences)} sequences for the surprise buffer on {world} rank(s)")
    mine = sorted(range(rank, len(sequences), world), key=lambda k: len(sequences[k]), reverse=True)
    device = next(model.parameters()).device
    scores = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        start = 0
        while start < len(mine):
            # Longest first, so the first sequence sets the batch's padded length.
            width = len(sequences[mine[start]])
            batch = mine[start:start + max(1, max_tokens // width)]
            start += len(batch)
            input_ids = torch.zeros(len(batch), width, dtype=torch.long)
            attention_mask = torch.zeros(len(batch), width, dtype=torch.long)
            for r, k in enumerate(batch):
                input_ids[r, :len(sequences[k])] = torch.as_tensor(sequences[k])
                attention_mask[r, :len(sequences[k])] = 1
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for r, k in enumerate(batch):
                n = len(sequences[k])
                scores[k] = F.cross_entropy(logits[r, :n - 1].float(), input_ids[r, 1:n]).item() if n > 1 else 0.0
            del logits
    model.train(was_training)
    if distributed:
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, scores)
        scores = {k: v for part in gathered for k, v in part.items()}
    return [scores[k] for k in range(len(sequences))]

class Buffer:
    def __init__(self, data_collator, buffer_size=500, device='cuda', type="Reservoir"):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.num_seen_examples = 0
        self.num_examples_per_dom = {}
        self.type = type
        self.data_collator = data_collator

    def reservoir(self, index):
        if index < self.buffer_size:
            return index
        rand = np.random.randint(0, index + 1)
        return rand if rand < self.buffer_size else -1


    def add_data(self, input_ids, labels, dom, surprise_scores=None, attention_mask=None):
        for i in range(len(input_ids)):
            idx = self.reservoir(self.num_seen_examples) if self.type == 'Reservoir' else 1
            input_id = input_ids[i].tolist() if torch.is_tensor(input_ids[i]) else list(input_ids[i])
            label = labels[i].tolist() if torch.is_tensor(labels[i]) else list(labels[i])
            if attention_mask is not None:
                keep = attention_mask[i].bool().tolist()
                input_id = [t for t, k in zip(input_id, keep) if k]
                if len(label) == len(keep):
                    label = [t for t, k in zip(label, keep) if k]
            surprise = surprise_scores[i] if surprise_scores is not None else None
            self.num_seen_examples += 1

            if dom not in self.num_examples_per_dom:
              self.num_examples_per_dom[dom] = 0

            if idx >= 0:
                if len(self.buffer) < self.buffer_size:
                    self.buffer.append((input_id, label, surprise, dom))
                    if dom in self.num_examples_per_dom:
                        self.num_examples_per_dom[dom] += 1
                else:
                    candidate_idx = np.random.randint(0, len(self.buffer))
                    candidate = self.buffer[candidate_idx]
                    self.num_examples_per_dom[candidate[3]] -= 1
                    if dom in self.num_examples_per_dom:
                        self.num_examples_per_dom[dom] += 1
                    self.buffer[candidate_idx] = (input_id, label, surprise, dom)

    def remove_data(self, input_ids, dom):
        to_remove = set(tuple(ids) for ids in input_ids)
        new_buffer = []
        actually_removed = 0  # Count what we actually remove

        for sample in self.buffer:
            if sample is None:
                continue
            if tuple(sample[0]) in to_remove:
                actually_removed += 1  # Increment actual count
                del sample
            else:
                new_buffer.append(sample)

        # Use actual count, not requested count
        self.num_examples_per_dom[dom] -= actually_removed
        print(f"Actually deleted {actually_removed} from domain {dom}")
        self.buffer = new_buffer

    def get_data(self, size):
        if not self.buffer:
            return None
        indices = np.random.choice(len(self.buffer), size=min(size, len(self.buffer)), replace=False)
        features = [{"input_ids": self.buffer[i][0], "labels": self.buffer[i][1]} for i in indices]
        return self.data_collator(features)

    def is_empty(self):
        return len(self.buffer) == 0


    def surprise_buffer_update(self, tokenizer, model, train_datasets, i, update_buffer):
        for j in range(len(train_datasets)):
            if  not self.is_empty() and ((j < i and i < 14 and update_buffer=="after") or (j <= i and i < 14 and update_buffer=="before")):

                curr_domain = [self.buffer[x] for x in range(len(self.buffer)) if self.buffer[x][3] == j]

                curr_domain = sorted(curr_domain, key=lambda x: x[2], reverse=False)

                num_to_remove = self.num_examples_per_dom[j] - (self.buffer_size // (i + 2)) if update_buffer=="before" else self.num_examples_per_dom[j] - (self.buffer_size // (i + 1))

                if num_to_remove > 0:
                    print(f"deleting {num_to_remove} from domain {j}")
                    print(f'Length of delete_data: {len([x[0] for x in curr_domain[:num_to_remove]])}')
                    self.remove_data([x[0] for x in curr_domain[:num_to_remove]], dom=j)

            if (j == i and i < 14 and update_buffer=="after") or (j == (i+1) and i < 14 and update_buffer=="before"):
                inputs, labels = train_datasets[j]["input_ids"], train_datasets[j]["labels"]
                scores = score_sequences(model, inputs)
                if self.is_empty():
                    num_to_add = self.buffer_size
                elif update_buffer == "before":
                    num_to_add = self.buffer_size // (i + 2) 
                else:
                    num_to_add = self.buffer_size // (i + 1)
                input_ids_batch = []
                labels_batch = []
                surprise_scores_batch = []
                seen = set()

                # Most surprising first, skipping duplicate sequences.
                for k in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True):
                    if len(input_ids_batch) == num_to_add:
                        break
                    key = tuple(inputs[k])
                    if key in seen:
                        continue
                    seen.add(key)
                    input_ids_batch.append(list(inputs[k]))
                    labels_batch.append(list(labels[k]))
                    surprise_scores_batch.append(scores[k])

                self.add_data(
                    input_ids_batch,
                    labels_batch,
                    j,
                    surprise_scores_batch,
                )