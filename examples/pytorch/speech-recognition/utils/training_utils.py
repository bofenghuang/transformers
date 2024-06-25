import math
import torch
from transformers import Trainer


class ScheduledTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:

            # min_learning_rate = 1e-5
            min_learning_rate = 0
            learning_rate = self.args.learning_rate
            min_lr_ratio = min_learning_rate / learning_rate

            warmup_ratio = 0.1
            decay_ratio = 0.5

            num_warmup_steps = math.ceil(num_training_steps * warmup_ratio)
            num_decay_steps = math.ceil(num_training_steps * decay_ratio)

            def _lr_lambda(current_step: int):
                if num_warmup_steps is not None and current_step < num_warmup_steps:  # warmup phase
                    return max(min_lr_ratio, float(current_step) / float(max(1, num_warmup_steps)))
                elif num_decay_steps is not None and current_step >= num_training_steps - num_decay_steps:  # decay phase
                    return max(min_lr_ratio, float(num_training_steps - current_step) / float(max(1, num_decay_steps)))
                else:  # constant learning rate phase
                    return 1

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        return self.lr_scheduler
