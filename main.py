from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from methods.dual_learner import DualLearnerTrainer
from methods.ewc import EWCTrainer
from methods.replay import ReplayTrainer
from methods.buffer import Buffer
from utils.model import prepare_model
from utils.datasets import load_from_hf, prepare_datasets_llama, prepare_datasets_T5, check_prompts_labels, prepare_mlt_dataset, NEMOTRON, NEMOTRON_TASKS_CONFIG, load_task_specs, load_nemotron, prepare_datasets_chat
import torch
from utils.evaluation import evaluate_llama, evaluate_T5, evaluate_chat_loss
from utils.script import build_cfg
from peft import LoraConfig
from datetime import datetime, timezone
import json
import os

DEFAULT_TARGET_MODULES = {
    "nemotron_h": (
        r".*\.mixer\.(in_proj|q_proj|k_proj|v_proj|o_proj)"
        r"|.*\.mixer\.shared_experts\.(up_proj|down_proj)"
    ),
    "qwen3_5_text": (
        r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|.*\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
        r"|.*\.mlp\.(gate_proj|up_proj|down_proj)"
    ),
}

def resolve_dtype(x):
    if isinstance(x, torch.dtype): 
        return x
    s = str(x).lower()
    if s in {"float16", "fp16", "torch.float16"}: return torch.float16
    if s in {"bfloat16", "bf16", "torch.bfloat16"}: return torch.bfloat16
    if s in {"float32", "fp32", "torch.float32"}: return torch.float32
    return torch.float16  # sensible default

trainers = {
    'ewc': EWCTrainer,
    'replay': ReplayTrainer,
    'surprise_replay': ReplayTrainer, 
    'dual_learner': DualLearnerTrainer,
    'mlt': Trainer
}

def pick_evaluator(model_name: str, benchmark: str):
    if benchmark == NEMOTRON:
        return evaluate_chat_loss
    if model_name.startswith("google-t5/"):
        return evaluate_T5
    if "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        return evaluate_llama
    # safe default
    return evaluate_llama

class ResultsWriter:
    def __init__(self, path, cfg):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; pass a different --run_name")
        self.path = path
        self._write({
            "type": "config",
            "run_name": cfg["run_name"],
            "model_name": cfg["model_name"],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {k: v for k, v in cfg.items() if k != "hf_token"},
        })

    def eval(self, stage, learner, task_index, task, metrics, after_task=None, after_task_name=None):
        self._write({"type": "eval", "stage": stage, "after_task": after_task, "after_task_name": after_task_name,
                     "learner": learner, "task_index": task_index, "task": task, **metrics,
                     "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def checkpoint(self, stage, learner, path, base_model, base_revision, after_task=None, after_task_name=None):
        """An adapter saved for external benchmark evaluation; the scores are appended later."""
        self._write({"type": "checkpoint", "stage": stage, "after_task": after_task, "after_task_name": after_task_name,
                     "learner": learner, "path": os.path.abspath(path), "base_model": base_model,
                     "base_revision": base_revision,
                     "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def _write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

def main():
    cfg = build_cfg()
    if not cfg.get("run_name"):
        cfg["run_name"] = f"{cfg['cl_method']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def get(k, default=None):
        return cfg.get(k, default)

    MODEL_NAME = cfg["model_name"]
    HF_TOKEN = get("hf_token")
    TORCH_DTYPE = resolve_dtype(get("torch_dtype", "float16"))
    tasks = cfg["tasks"]
    benchmark = cfg["benchmark"]
    do_check_prompts_labels = get("do_check_prompts_labels", False)
    cl_method = cfg["cl_method"]
    update_buffer = get("update_buffer")
    buffer_type = get("buffer")
    surprise_tokens = cfg["surprise_tokens"] = get("surprise_tokens") or ("supervised" if benchmark == NEMOTRON else "all")
    num_samples_training = get("num_samples_training")
    num_samples_eval = get("num_samples_eval")
    learning_rate = float(get("learning_rate", 2e-5))
    lora_rank = int(get("lora_rank", 16))
    lora_alpha = int(get("lora_alpha", 32))
    gradient_accumulation_steps = int(get("gradient_accumulation_steps", 1))
    batch_size = int(get("batch_size", 1))
    lambda_ema = get("lambda_ema") or [0.999]
    eval_after_each_task = bool(get("eval_after_each_task", False))
    eval_mode = get("eval_mode", "loss")

    if buffer_type:
        buffer_size = int(get("buffer_size", 500))
        replay_frequency = int(get("replay_frequency", 2))

    DEVICE = get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    is_main = int(os.environ.get("RANK", 0)) == 0
    if distributed and str(DEVICE).startswith("cuda"):
        DEVICE = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(DEVICE)
    print(f"Running on {DEVICE}")

    task_specs = None
    if benchmark == NEMOTRON:
        cfg["tasks_config"] = str(get("tasks_config") or NEMOTRON_TASKS_CONFIG)
        task_specs = cfg["task_specs"] = load_task_specs(tasks, cfg["tasks_config"])

    results = None
    if is_main:
        results_path = os.path.join(get("results_dir") or "results", MODEL_NAME.replace("/", "_"), f"{cfg['run_name']}.jsonl")
        results = ResultsWriter(results_path, cfg)
        print(f"Writing results to {results_path}")

    if MODEL_NAME.startswith("google-t5/"):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, dtype=TORCH_DTYPE
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, dtype=TORCH_DTYPE
        )

    # The exact base weights, so benchmark checkpoints are merged onto the same revision
    base_revision = getattr(base_model.config, "_commit_hash", None)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        # safest fallback; T5 normally has a pad_token already
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "right"

    base_model.to(DEVICE)
    print(f'{MODEL_NAME} loaded and moved to device')

    # data collator
    if MODEL_NAME.startswith("google-t5/"):
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=DEFAULT_TARGET_MODULES.get(getattr(base_model.config, "model_type", None))
    )

    peft_config.task_type = "SEQ_2_SEQ_LM" if MODEL_NAME.startswith("google-t5/") else "CAUSAL_LM"

    model = prepare_model(base_model, peft_config, MODEL_NAME)
    task_num_samples = get("task_num_samples") or [None] * len(tasks)
    if benchmark != NEMOTRON and any(n is not None for n in task_num_samples):
        raise ValueError(f'Per-task sample counts ("<task>:<n>") are only supported with --benchmark={NEMOTRON}')
    if benchmark == NEMOTRON:
        if MODEL_NAME.startswith("google-t5"):
            raise ValueError(f"--benchmark={NEMOTRON} needs a causal LM with a chat template")
        train_datasets_full, test_datasets = load_nemotron(tasks, float(get("test_size", 0.05)), specs=task_specs)
        per_task_samples = [num_samples_training if n is None else n for n in task_num_samples]
        train_datasets_full, train_datasets, test_datasets = prepare_datasets_chat(tokenizer, train_datasets_full, test_datasets, per_task_samples, int(get("max_seq_len", 1024)), num_samples_eval)
    elif MODEL_NAME.startswith("google-t5"):
        train_datasets_full, test_datasets = load_from_hf(tasks, benchmark)
        train_datasets_full, train_datasets, test_datasets = prepare_datasets_T5(tokenizer, train_datasets_full, test_datasets, num_samples_training)
    else:
        train_datasets_full, test_datasets = load_from_hf(tasks, benchmark)
        train_datasets_full, train_datasets, test_datasets = prepare_datasets_llama(tokenizer, train_datasets_full, test_datasets, num_samples_training, int(get("max_seq_len", 1024)))

    if do_check_prompts_labels:
        # pass what your function actually needs
        check_prompts_labels(tokenizer, train_datasets)

    trainer_class = trainers[cl_method]

    save_steps = int(get("save_steps", 0))
    training_args = TrainingArguments(
        output_dir=f"./runs/{MODEL_NAME.replace('/', '_')}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=cfg.get("num_train_epochs", 1),
        logging_steps=cfg.get("logging_steps", 50),
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=save_steps if save_steps > 0 else 500,
        fp16=(TORCH_DTYPE == torch.float16),
        bf16=(TORCH_DTYPE == torch.bfloat16),
        use_liger_kernel=get("use_liger_kernel", False),
        liger_kernel_config={
            "rope": False,
            "rms_norm": False,
            "swiglu": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
        } if get("use_liger_kernel", False) else None,
        ddp_find_unused_parameters=get("ddp_find_unused_parameters"),
        remove_unused_columns=False,
        report_to="none",
        ddp_timeout=int(get("ddp_timeout", 7200)),
    )

    if cl_method == 'mlt':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator=data_collator,
            update_buffer = update_buffer
        )        

    evaluate = pick_evaluator(MODEL_NAME, benchmark)

    def save_eval_checkpoint(model, stage, learner, after_task):
        label = stage if after_task is None else f"{stage}-{after_task}-{tasks[after_task]}"
        path = os.path.join(training_args.output_dir, cfg["run_name"], "eval_checkpoints", label, learner)
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; pass a different --run_name")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        results.checkpoint(stage, learner, path, MODEL_NAME, base_revision, after_task,
                           None if after_task is None else tasks[after_task])
        print(f"Saved the {learner} adapter for benchmark evaluation to {path}")

    def run_eval(model, stage, learner="fast", after_task=None):
        model.eval()
        if is_main:
            if eval_mode in {"loss", "both"}:
                for j, (test_ds, test_task) in enumerate(zip(test_datasets, tasks)):
                    metrics = evaluate(model, test_ds, test_task, tokenizer, num_samples_eval, compute_ll=False)
                    results.eval(stage, learner, j, test_task, metrics, after_task,
                                 None if after_task is None else tasks[after_task])
            if eval_mode in {"benchmarks", "both"}:
                save_eval_checkpoint(model, stage, learner, after_task)
        trainer.accelerator.wait_for_everyone()
        model.train()

    def run_slow_evals(model, stage, after_task=None):
        for lam in lambda_ema:
            learner = DualLearnerTrainer.learner_name(lam, lambda_ema)
            print(f"Results for slow learner {learner} (lambda_ema={lam})")
            with trainer.use_slow_weights(model, lam):
                run_eval(model, stage, learner=learner, after_task=after_task)

    if get("eval_before_training", True):
        print("--------Evaluation before training---------")
        run_eval(model, "before_training")

    if cl_method == 'mlt':
        print(f'--------Training on all datasets---------')
        joint_ds = prepare_mlt_dataset(train_datasets)
        print(joint_ds)
        trainer.train_dataset = joint_ds
        trainer.train()
        model = trainer.model.to(DEVICE)
    else:
        buf = None
        if buffer_type is not None:
            buf = Buffer(
                buffer_size=buffer_size,
                device=torch.device(DEVICE),
                data_collator=data_collator,
                type=buffer_type,
                surprise_tokens=surprise_tokens
            )
            trainer.buffer = buf
            trainer.replay_frequency = replay_frequency
            trainer.update_buffer = update_buffer
            if update_buffer == "before" and buffer_type == "Surprise":
                buf.surprise_buffer_update(tokenizer, model, train_datasets, -1, update_buffer)

        if cl_method == "dual_learner":
            trainer.init_slow_weights(model, lambda_ema, debias=bool(get("ema_debias", True)))

        for i, task in enumerate(tasks):
            print(f'--------Training on {task}---------')
            trainer.train_dataset = train_datasets[i]
            trainer.domain = i
            trainer.train()
            model = trainer.model.to(DEVICE)

            if eval_after_each_task and i < len(tasks) - 1:
                print(f"--------Evaluation after {task}---------")
                run_eval(model, "after_task", after_task=i)
                if cl_method == "dual_learner":
                    run_slow_evals(model, "after_task", after_task=i)

            if buf is not None and update_buffer in {"before", "after"} and buffer_type == "Surprise":
                buf.surprise_buffer_update(tokenizer, model, train_datasets, i, update_buffer)
            
            trainer.model = model
        

    if get("save_final", False):
        final_dir = os.path.join(training_args.output_dir, cfg["run_name"], "final")
        trainer.save_model(final_dir)
        if is_main:
            tokenizer.save_pretrained(final_dir)
            print(f"Saved the final adapter to {final_dir}")

    final_task = None if cl_method == 'mlt' else len(tasks) - 1
    print("--------Starting Evaluation---------")
    run_eval(model, "final", after_task=final_task)

    if cl_method == 'dual_learner':
        run_slow_evals(model, "final", after_task=final_task)

    if is_main:
        print(f"Results written to {results.path}")


if __name__ == "__main__":
    main()