"""Merge a saved LoRA adapter into its base model, so vLLM can serve it for benchmark evaluation.

    python utils/merge_adapter.py --adapter runs/<model>/<run>/eval_checkpoints/<stage>/fast \
        --revision <40-hex base commit> --output /path/to/merged

Runs on CPU (the BF16 Nemotron base needs ~60 GB of RAM and as much disk per merged copy).
The base model and revision default to the ones recorded when the adapter was saved.
"""
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="directory written by --eval_mode=benchmarks or --save_final")
    p.add_argument("--output", required=True, help="new directory for the merged model")
    p.add_argument("--base_model", default=None, help="default: base_model_name_or_path from the adapter config")
    p.add_argument("--revision", required=True, help="full commit of the base model the adapter was trained on")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--hf_token", default=None)
    args = p.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        p.error("--revision must be a full 40-character commit")
    if os.path.exists(args.output):
        p.error(f"{args.output} already exists; merge into a new directory")

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    base_model = args.base_model or adapter_config["base_model_name_or_path"]

    model = AutoModelForCausalLM.from_pretrained(
        base_model, revision=args.revision, token=args.hf_token, dtype=getattr(torch, args.torch_dtype)
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    # The tokenizer and chat template saved with the adapter are the ones used in training
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "merge_info.json"), "x") as f:
        json.dump({
            "base_model": base_model,
            "base_revision": args.revision,
            "adapter": os.path.abspath(args.adapter),
            "adapter_sha256": sha256(os.path.join(args.adapter, "adapter_model.safetensors")),
            "torch_dtype": args.torch_dtype,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, f, indent=2)
    print(f"Merged {args.adapter} into {base_model}@{args.revision}: {args.output}")


if __name__ == "__main__":
    main()
