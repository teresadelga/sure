from transformers import Trainer


class ReplayTrainer(Trainer):
    def __init__(self, *args, buffer=None, domain=None, replay_frequency=2, update_buffer='online', **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.domain = domain
        self._micro_step = 0  # counts forward passes
        self.replay_frequency = replay_frequency
        self.update_buffer = update_buffer
        self._replay_step = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # Add to buffer
        if self.buffer is not None and self.update_buffer == 'online':
            self.buffer.add_data(inputs["input_ids"], inputs["labels"], self.domain, attention_mask=inputs.get("attention_mask"))

        if self._replay_step:
            loss = loss / 2

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._micro_step += 1
        self._replay_step = self._micro_step % self.replay_frequency == 0 and self.buffer is not None and not self.buffer.is_empty()
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        finally:
            replay_step, self._replay_step = self._replay_step, False
        if not replay_step:
            return loss

        replay_inputs = self.buffer.get_data(size=max(1, self.args.per_device_train_batch_size))
        replay_inputs = {k: v.to(self.accelerator.device) for k, v in replay_inputs.items()}
        with self.compute_loss_context_manager():
            replay_loss = model(**replay_inputs).loss.mean()
        replay_loss = replay_loss / self.current_gradient_accumulation_steps / 2
        self.accelerator.backward(replay_loss)
        return loss + replay_loss.detach()
