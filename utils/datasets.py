from datasets import Dataset, Features, List, Value, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, HfFileSystem
from pathlib import Path
import hashlib
import json
import os
import random
import re
import yaml
from utils.inference import encode_prompt


def load_from_hf(tasks, benchmark):
    train_datasets_full = []
    test_datasets = []
    benchmark_train = benchmark + "-train"
    benchmark_test = benchmark + "-test"

    for task in tasks:
        train_datasets_full.append(load_dataset(benchmark_train, split=task))
        test_datasets.append(load_dataset(benchmark_test, split=task))

    return train_datasets_full, test_datasets

NEMOTRON = "nemotron"
NEMOTRON_TASKS_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "nemotron_tasks.yaml"
NEMOTRON_TERMINAL = "nvidia/Nemotron-Terminal-Corpus"
NEMOTRON_SWE = "nvidia/Nemotron-SFT-SWE-v3"
NEMOTRON_AGENTIC = "nvidia/Nemotron-SFT-Agentic-v2"


_ISSUE_TAG = re.compile(r"<(\w*(?:issue|pr_description)\w*)>(.*?)</\1>", re.S)

AGENTIC_FEATURES = Features({"messages_json": Value("string"), "tools_json": Value("string"), "group": Value("string")})


def load_task_specs(tasks, path=NEMOTRON_TASKS_CONFIG):
    with open(path) as f:
        config = yaml.safe_load(f)
    unknown = [t for t in tasks if t not in config]
    if unknown:
        raise ValueError(f"Tasks {unknown} are not defined in {path}; it defines {sorted(config)}")
    return {t: config[t] for t in tasks}


def nemotron_task_files(domain):
    """Map a Terminal-Corpus domain, "<subset>/<category>" (e.g. "medium/debugging") or
    "adapters/<name>", to its parquet file."""
    subset, _, category = domain.rpartition("/")
    if subset == "adapters":
        return f"dataset_adapters/{category}.parquet"
    if subset in {"easy", "medium", "mixed"}:
        return f"synthetic_tasks/skill_based/{subset}/{category}/data_filtered.parquet"
    raise ValueError(f"Unknown Terminal-Corpus domain {domain!r}; use easy/, medium/, mixed/ or adapters/<name>")


def split_by_group(ds, key, test_size, seed):
    groups = sorted(set(ds[key]))
    held_out = set(Dataset.from_dict({key: groups}).shuffle(seed=seed)[key][:max(1, round(test_size * len(groups)))])
    return ds.filter(lambda e: e[key] not in held_out), ds.filter(lambda e: e[key] in held_out)


def load_terminal_domain(domain, spec, test_size, seed):
    ds = load_dataset(NEMOTRON_TERMINAL, data_files=nemotron_task_files(domain), split="train")
    return split_by_group(ds.select_columns(["conversations", "task"]), "task", test_size, seed)


def swe_issue_key(messages):
    user = next((m["content"] for m in messages if m["role"] == "user" and m["content"]), "")
    match = _ISSUE_TAG.search(user)
    text = (match.group(2) if match else user).split("[ISSUE]", 1)[-1]
    words = re.findall(r"\w+", text.lower())[:200]
    return hashlib.md5(" ".join(words).encode()).hexdigest()


def tool_use_messages(messages):
    out = []
    for m in messages:
        msg = {"role": m["role"], "content": m["content"] or ""}
        if m.get("reasoning_content"):
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            tool_calls = m["tool_calls"]
            for call in json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls:
                fn = call.get("function", call)
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                calls.append({"type": "function", "function": {"name": fn["name"], "arguments": args}})
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def load_swe(domain, spec, test_size, seed):
    if domain is not None:
        raise ValueError(f"{NEMOTRON_SWE} has no domains")
    shards = sorted(f for f in HfApi().list_repo_files(NEMOTRON_SWE, repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet"))
    shards = sorted(random.Random(seed).sample(shards, min(spec.get("num_shards", 4), len(shards))))
    ds = load_dataset(NEMOTRON_SWE, data_files=shards, split="train").select_columns(["messages"])
    ds = ds.map(lambda e: {"issue": swe_issue_key(e["messages"])})
    return split_by_group(ds, "issue", test_size, seed)


def first_user_key(messages):
    user = next((m["content"] for m in messages if m["role"] == "user" and m["content"]), "")
    return hashlib.md5(" ".join(re.findall(r"\w+", user.lower())[:200]).encode()).hexdigest()


def agentic_rows(split, num_rows):
    with HfFileSystem().open(f"datasets/{NEMOTRON_AGENTIC}/data/{split}.jsonl", "rb", block_size=2**24) as f:
        for i, line in enumerate(f):
            if i == num_rows:
                break
            row = json.loads(line)
            yield {"messages_json": json.dumps(row["messages"]), "tools_json": json.dumps(row.get("tools") or []),
                   "group": first_user_key(row["messages"])}


def load_agentic_split(split, spec, test_size, seed):
    ds = Dataset.from_generator(agentic_rows, gen_kwargs={"split": split, "num_rows": spec.get("num_rows", 20000)},
                                features=AGENTIC_FEATURES)
    return split_by_group(ds, "group", test_size, seed)


DOMAIN_LOADERS = {
    NEMOTRON_TERMINAL: load_terminal_domain,
    NEMOTRON_SWE: load_swe,
    NEMOTRON_AGENTIC: load_agentic_split,
}


def load_nemotron(tasks, test_size=0.05, seed=42, specs=None):
    """Load the tasks of the NEMOTRON benchmark as train/test pairs, mixing each task's domains."""
    specs = specs or load_task_specs(tasks)
    train_datasets_full = []
    test_datasets = []
    for task in tasks:
        spec = specs[task]
        if spec["dataset"] not in DOMAIN_LOADERS:
            raise ValueError(f"Task {task!r}: unsupported dataset {spec['dataset']!r}; use one of {sorted(DOMAIN_LOADERS)}")
        load = DOMAIN_LOADERS[spec["dataset"]]
        splits = [load(domain, spec, test_size, seed) for domain in spec.get("domains") or [None]]
        train, test = ([split[i] for split in splits] for i in (0, 1))
        train = (train[0] if len(train) == 1 else concatenate_datasets(train)).shuffle(seed=seed)
        test = (test[0] if len(test) == 1 else concatenate_datasets(test)).shuffle(seed=seed)
        train_datasets_full.append(train)
        test_datasets.append(test)
        print(f"{task}: {len(train)} train / {len(test)} test trajectories")
    return train_datasets_full, test_datasets


def tokenize_chat_or_drop(example, tokenizer, max_seq_len):
    try:
        return tokenize_chat(example, tokenizer, max_seq_len)
    except ValueError:
        return {"input_ids": [], "attention_mask": [], "labels": []}


def tokenize_chat(example, tokenizer, max_seq_len):
    if "conversations" in example:  # Terminal-Corpus
        messages, tools = example["conversations"], None
    elif "messages_json" in example:  # Agentic-v2
        messages, tools = tool_use_messages(json.loads(example["messages_json"])), json.loads(example["tools_json"]) or None
    else:  # SWE-v3
        messages, tools = tool_use_messages(example["messages"]), None

    def render(msgs, add_generation_prompt=False):
        if not msgs:
            return ""
        return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt)

    def segment_of(k, end):
        """Text of messages[k:end]: one assistant turn, or a run of non-assistant turns."""
        full = render(messages[:end])
        base = render(messages[:k])
        if full.startswith(base):
            return full[len(base):], base
        dummy = {"role": "user", "content": "."}
        with_k = render(messages[:end] + [dummy])
        without_k = render(messages[:k] + [dummy])
        n = len(with_k) - len(without_k)
        if messages[k]["role"] == "assistant" or n <= 0 or not with_k.startswith(full) \
                or not without_k.startswith(full[:len(full) - n]):
            raise ValueError("Chat template is not prefix-consistent; cannot split it into turns")
        return full[len(full) - n:], None

    input_ids, labels = [], []
    k = 0
    while k < len(messages):
        message = messages[k]
        end = k + 1
        if message["role"] != "assistant":
            while end < len(messages) and messages[end]["role"] != "assistant":
                end += 1
        segment, base = segment_of(k, end)

        if message["role"] == "assistant":
            header = render(messages[:k], add_generation_prompt=True)[len(base):]
            n = len(os.path.commonprefix([header, segment]))
            head_ids = tokenizer(segment[:n], add_special_tokens=False)["input_ids"]
            body_ids = tokenizer(segment[n:], add_special_tokens=False)["input_ids"]
            seg_ids = head_ids + body_ids
            seg_labels = [-100] * len(head_ids) + body_ids
        else:
            seg_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
            seg_labels = [-100] * len(seg_ids)

        if len(input_ids) + len(seg_ids) > max_seq_len:
            break
        input_ids += seg_ids
        labels += seg_labels
        k = end

    if all(l == -100 for l in labels):
        input_ids, labels = [], []
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


CHAT_FEATURES = Features({
    "input_ids": List(Value("int32")),
    "attention_mask": List(Value("int8")),
    "labels": List(Value("int32")),
})


def prepare_datasets_chat(tokenizer, train_datasets_full, test_datasets, num_samples_training, max_seq_len, num_samples_eval=None):
    num_proc = max(1, len(os.sched_getaffinity(0)))

    def prepare(ds):
        if len(ds) == 0:
            return ds
        ds = ds.map(lambda e: tokenize_chat_or_drop(e, tokenizer, max_seq_len), remove_columns=ds.column_names,
                    num_proc=max(1, min(num_proc, len(ds) // 32)), features=CHAT_FEATURES)
        return ds.filter(lambda e: len(e["input_ids"]) > 0)

    def prepare_first(raw, n):
        target = len(raw) if n is None else min(n, len(raw))
        kept, start = [], 0
        while sum(len(d) for d in kept) < target and start < len(raw):
            stop = min(len(raw), start + target - sum(len(d) for d in kept))
            kept.append(prepare(raw.select(range(start, stop))))
            start = stop
        kept = kept or [prepare(raw.select([]))]
        return (concatenate_datasets(kept) if len(kept) > 1 else kept[0]), start

    if not isinstance(num_samples_training, list):
        num_samples_training = [num_samples_training] * len(train_datasets_full)

    train_datasets = []
    for i in range(len(train_datasets_full)):
        raw = train_datasets_full[i]
        train, tried = prepare_first(raw, num_samples_training[i])
        train_datasets.append(train)
        print(f"Kept {len(train)}/{tried} training conversations within max_seq_len={max_seq_len} "
              f"(of {len(raw)} available)")
        test_datasets[i], _ = prepare_first(test_datasets[i], num_samples_eval)

    return train_datasets, train_datasets, test_datasets


def tokenize(example, tokenizer):
        prompt_ids = encode_prompt(example["prompt"], tokenizer)
        answer_ids = tokenizer(" " + example["label"], add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

def tokenize_t5(ex, tokenizer, task_prefix=""):
    source = (task_prefix + ex["prompt"]).strip()
    target = ex["label"]

    model_inputs = tokenizer(source, text_target=target)
    
    labels = model_inputs["labels"]

    if not isinstance(labels, list):
        labels = list(labels)
    if len(labels) == 0:
        labels = [tokenizer.pad_token_id]

    model_inputs["labels"] = labels
    return model_inputs

def subsample(ex, MAX_SRC=512, MAX_TGT=256):
        return len(ex["input_ids"]) <= MAX_SRC and len(ex["labels"]) <= MAX_TGT

def prepare_datasets_T5(tokenizer, train_datasets_full, test_datasets, num_samples_training):
    train_datasets = []
    for i in range(len(train_datasets_full)):
        train_datasets_full[i] = train_datasets_full[i].map(lambda e: tokenize_t5(e, tokenizer=tokenizer))
        train_datasets_full[i] = train_datasets_full[i].filter(subsample, batched=False)
        train_datasets.append(train_datasets_full[i].remove_columns(['label', 'prompt']).select(range(min(num_samples_training, len(train_datasets_full[i])))))
        test_datasets[i] = test_datasets[i].map(lambda e: tokenize_t5(e, tokenizer=tokenizer))
        test_datasets[i]  = test_datasets[i].filter(subsample, batched=False)
    
    return train_datasets_full, train_datasets, test_datasets

def prepare_datasets_llama(tokenizer, train_datasets_full, test_datasets, num_samples_training, max_seq_len=1024):
    train_datasets = []
    for i in range(len(train_datasets_full)):
        columns = train_datasets_full[i].column_names
        train_datasets_full[i] = train_datasets_full[i].map(lambda e: tokenize(e, tokenizer=tokenizer), remove_columns=columns)
        train_datasets_full[i] = train_datasets_full[i].filter(lambda e: len(e["input_ids"]) <= max_seq_len, batched=False)
        n = len(train_datasets_full[i]) if num_samples_training is None else min(num_samples_training, len(train_datasets_full[i]))
        train_datasets.append(train_datasets_full[i].select(range(n)))

    return train_datasets_full, train_datasets, test_datasets

def prepare_mlt_dataset(train_datasets):
    joint_ds = concatenate_datasets(train_datasets)
    mlt_dataset = joint_ds.shuffle(seed=42)
    return mlt_dataset

def check_prompts_labels(tokenizer, train_datasets):
    for train_dataset in train_datasets:
        for i in range(len(train_dataset)):
            input_ids = train_dataset[i]["input_ids"]
            labels = train_dataset[i]["labels"]
            filtered_labels = [token_id for token_id in labels if token_id != -100]

            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            label_text = tokenizer.decode(filtered_labels, skip_special_tokens=True)

            print("Prompt:", prompt_text)
            print("Label:", label_text)
            print("---")