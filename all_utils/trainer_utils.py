
from typing import Optional

from torch import nn, optim
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from accelerate.state import AcceleratorState
import accelerate
from accelerate import DistributedDataParallelKwargs

accelerator: accelerate.Accelerator = None

def setup_accelerator(args):
    global accelerator
    if accelerator is None:
        accelerator = accelerate.Accelerator(
        log_with=args.report_to, #tensorboard
        project_dir=args.logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps, #16
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        # Value model might not use all parameters (e.g., lm-head) in the forward pass.
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.ddp_find_unused_parameters, #false
            )
        ],
    )
    return accelerator


def create_optimizer(
    args, model: nn.Module, optimizer: Optional[optim.Optimizer] = None
):
    """Create optimizer for trainer.

    This is detached version of the `Trainer.create_optimizer` method.
    We don't support sagemaker and fairscale for simplicity.

    Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    """
    opt_model = model

    if optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": args.weight_decay, #这个也是0.0
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs) #lr = 3e-5, betas = (0.9，0.999), eps = 1e-8, weight_decay = 0.01
    return optimizer


def create_scheduler(args, optimizer, lr_scheduler, num_training_steps):
    """Create scheduler for trainer.

    This is detached version of the `Trainer.create_scheduler` method.

    Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    """
    if lr_scheduler is None:
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.get_warmup_steps(num_training_steps), #warmup就是training_steps 
            num_training_steps=num_training_steps,
        )
    return lr_scheduler

def get_eval_ds_config(offload=None, stage=3):
    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }

def setup_deepspeed_plugin(args):
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = args.step_per_device_batch_size # this is a dummy value
    deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}


