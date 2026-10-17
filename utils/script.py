import argparse

def build_cfg():
    p = argparse.ArgumentParser()
    # core model / method
    p.add_argument("--model_name", required=True)
    p.add_argument("--cl_method", required=True, choices=["mlt", "ewc", "replay", "surprise_replay", "dual_learner"])
    p.add_argument("--tasks", required=True, help='Space- or comma-separated list')
    p.add_argument("--benchmark", required=True, help='HF dataset prefix (<benchmark>-train/-test), or "nemotron"')
    # training knobs
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--torch_dtype", default="float16")
    p.add_argument("--use_liger_kernel", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--ddp_find_unused_parameters", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None)
    p.add_argument("--num_samples_eval", type=int, default=None)
    p.add_argument("--num_samples_training", type=int, default=None)
    p.add_argument("--device", default=None)  # auto if None
    # LoRA
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_rank", type=int, default=16)
    # buffer / replay
    p.add_argument("--buffer", default=None)
    p.add_argument("--buffer_size", type=int, default=0)
    p.add_argument("--replay_frequency", type=int, default=1)
    p.add_argument("--update_buffer", default=None, choices=[None, "before", "after"])
    p.add_argument("--surprise_tokens", default=None, choices=["all", "supervised"], help="tokens the surprise buffer's mean NLL is over")
    p.add_argument("--compute_surprise", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    # misc
    p.add_argument("--hf_token", default=None)
    p.add_argument("--do_check_prompts_labels", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_final", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--max_seq_len", type=int, default=1024, help="drop training examples longer than this")
    p.add_argument("--test_size", type=float, default=0.05, help="fraction of each category's tasks held out for evaluation")
    p.add_argument("--run_name", default=None, help="default <cl_method>-<timestamp>[-<slurm job id>]")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--tasks_config", default=None, help="YAML file defining the tasks; default configs/nemotron_tasks.yaml")
    p.add_argument("--lambda_ema", default="0.999", help="dual_learner: EMA decay per optimizer step; several (space- or comma-separated) train one slow learner each")
    p.add_argument("--ema_debias", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True, help="dual_learner: bias-correct the slow learners' EMA (divide by 1 - lambda^t) so no weight stays on the initial weights")
    p.add_argument("--eval_before_training", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--eval_after_each_task", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--eval_mode", default="loss", choices=["loss", "benchmarks", "both"], help="loss: held-out slices; benchmarks: save the adapter at each eval point for external benchmark evaluation; both: do both")

    args = p.parse_args()
    cfg = vars(args)

    # normalize tasks into a list, splitting off per-task sample counts ("swe:4000")
    t = [task.partition(":") for task in cfg["tasks"].replace(",", " ").split()]
    cfg["tasks"] = [name for name, _, _ in t]
    cfg["task_num_samples"] = [int(n) if n else None for _, _, n in t]
    cfg["lambda_ema"] = [float(x) for x in str(cfg["lambda_ema"]).replace(",", " ").split()]
    if not cfg["lambda_ema"] or len(set(cfg["lambda_ema"])) != len(cfg["lambda_ema"]):
        p.error("--lambda_ema needs one or more distinct values")
    return cfg