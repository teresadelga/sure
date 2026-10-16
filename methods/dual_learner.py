from contextlib import contextmanager
import torch
from transformers import TrainerCallback
from methods.replay import ReplayTrainer


class _SlowUpdateCallback(TrainerCallback):
    """Updates the slow weights after each optimizer step, so the EMA horizon is in optimizer steps
    and does not depend on the number of GPUs or gradient accumulation steps."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        if trainer.slow_weights is not None:
            trainer.update_slow_model(trainer.accelerator.unwrap_model(trainer.model))


class DualLearnerTrainer(ReplayTrainer):
    """ReplayTrainer (the fast learner) plus slow learners."""

    def __init__(self, *args, slow_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.slow_weights = slow_weights
        self.add_callback(_SlowUpdateCallback(self))

    def init_slow_weights(self, model, lambdas, debias=True):
        self.slow_init = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.slow_weights = {lam: {n: torch.zeros_like(w) for n, w in self.slow_init.items()} for lam in lambdas}
        self.slow_debias = debias
        self.slow_steps = 0

    @staticmethod
    def learner_name(lam, lambdas):
        return "slow" if len(lambdas) == 1 else f"slow@{lam:g}"

    @torch.no_grad()
    def update_slow_model(self, fast_model):
        self.slow_steps += 1
        for n, p_fast in fast_model.named_parameters():
            if n in self.slow_init:
                delta = p_fast.detach().float() - self.slow_init[n]
                for lam, weights in self.slow_weights.items():
                    weights[n].mul_(lam).add_(delta, alpha=1 - lam)

    @contextmanager
    def use_slow_weights(self, model, lam):
        """Temporarily load the slow weights for lambda_ema=lam for evaluation."""
        correction = 1 - lam ** self.slow_steps if self.slow_debias and self.slow_steps > 0 else 1
        params = dict(model.named_parameters())
        fast = {n: params[n].detach().clone() for n in self.slow_init}
        with torch.no_grad():
            for n, delta in self.slow_weights[lam].items():
                params[n].copy_(self.slow_init[n] + delta / correction)
        try:
            yield model
        finally:
            with torch.no_grad():
                for n, w in fast.items():
                    params[n].copy_(w)
