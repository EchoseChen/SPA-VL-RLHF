# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Optional, List
import logging

import torch
import torch.distributed as dist


import transformers
from transformers import set_seed

from models.rl_models import make_models
from all_utils.trainer_utils import setup_accelerator, setup_deepspeed_plugin

try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer

    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer

    print("Using slow tokenizer")

from transformers import AutoTokenizer, AutoModelForCausalLM

from all_utils.data_utils.data_utils_ppo import make_rl_data_module
from models.ppo.ppo_trainer import PPOTrainer


from llava import conversation as conversation_lib
from llava.model import *
from llava.constants import (
    IMAGE_TOKEN_INDEX,
)

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    reward_model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    policy_model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    base_model_name: Optional[str] = field(default="EleutherAI/pythia-12b")
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to stop generation with."},
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "pad"
    image_grid_pinpoints: Optional[str] = field(default=None)
    keywords: Optional[List[str]] = field(default_factory=lambda: ["test1"])
    config_train_path: str = field(default="/opt/tiger/llava-rlhf/llava/RLHF/prompts/train/config_train.json")

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    finetune_mm_projector: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # From AlpacaFarm
    truncate_tokens: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Tokens in strings to truncate at first occurrence. "
            "This was used in original OAI summarization paper to avoid models returning incomplete sentences. "
        },
    )
    truncate_after: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate after this number of tokens. Prevents early truncation."
        },
    )
    reward_bias: float = field(
        default=0.0,
        metadata={"help": "Add this amount to the reward."},
    )
    clean_tokens_after_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to clean up tokens after the first occurrence of stop_token."
        },
    )
    suppress_eos_at_generation: bool = field(
        default=False,
        metadata={
            "help": "Whether to suppress the end-of-sequence token at generation time."
        },
    )
    total_epochs: int = field(default=2)
    batch_size: int = field(default=1)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    whiten_rewards: bool = field(default=True)
    temperature: float = field(default=1.0)
    kl_coef: float = field(default=0.2)
    kl_approximator: str = field(default="k1")
    target_kl: float = field(default=6.0)
    k_beta: float = field(default=0.1)
    adaptive_kl: bool = field(default=False)
    eval_batches: int = field(
        default=sys.maxsize,
        metadata={"help": "Maximum number of batches to evaluate on."},
    )
    query_len: int = field(default=128)
    num_patches: int = field(default=576)
    min_token_limit: int = field(default=None)
    model_max_length: int = field(default=1024)
    whitening_async_stats: str = field(
        default="per_gpu",
        metadata={"help": "How to sync statistics for reward whitening."},
    )
    # From QLoRA
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    lora_r: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.00
    lora_weight_path: str = ""
    lora_bias: str = "none"
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: str = field(
        default=None,
        metadata={
            "help": "The directory to resume from. If None, will start from scratch."
        },
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    rollouts_gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )

    def __post_init__(self):
        super().__post_init__()
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rollout_batch_size = self.batch_size * world_size * self.rollouts_gradient_accumulation_steps
        self.step_per_device_batch_size = self.batch_size * self.rollouts_gradient_accumulation_steps
        self.step_batch_size = self.step_per_device_batch_size * world_size
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.warning(
                f"\tbatch_size: {self.batch_size}\n"
                f"\trollout_batch_size: {self.rollout_batch_size}\n"
                f"\tstep_batch_size: {self.step_batch_size}\n"
                f"\tworld_size: {world_size}\n",
                f"\rollouts_gradient_accumulation_steps: {self.rollouts_gradient_accumulation_steps}\n"
            )
        self.save_steps_extra_list = []

    def set_truncate_token_ids(self, tokenizer: transformers.PreTrainedTokenizer):
        truncate_tokens = self.truncate_tokens
        if truncate_tokens is None:
            truncate_token_ids = None
        else:
            truncate_token_ids = tokenizer.convert_tokens_to_ids(truncate_tokens)
        self.truncate_token_ids = truncate_token_ids


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    training_args.data_config = data_args
    checkpoint_dir = None
    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)

    accelerator = setup_accelerator(args)
    dict_args = vars(args) #返回一个字典
    # value should be one of int, float, str, bool, or torch.Tensor
    for k in dict_args:
        if type(dict_args[k]) not in [int, float, str, bool, torch.Tensor]:
            dict_args[k] = str(dict_args[k])
    # print(dict_args)
    accelerator.init_trackers(
        "ppo_trainer",
        config=dict_args,
    )
    logger.warning(
        accelerator.state,
        # main_process_only=False,
    )

    tokenizer_model_name = args.policy_model_name_or_path #'/mnt/bn/jsj-marl-challenge/chenlu/chenlu/llava-v1.5-13b/checkpoint-10'
    TokenizerClass = AutoTokenizer

    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir, #none
        model_max_length=training_args.model_max_length, #2048
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ] #Conversation(system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", roles=('USER', 'ASSISTANT'), messages=(), offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        with DisableLogger():
            base_model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.base_model_name,
                cache_dir=training_args.cache_dir,
            )

        vision_tower = base_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        training_args.query_len = args.query_len = training_args.model_max_length - vision_tower.num_patches
        training_args.num_patches = args.num_patches = vision_tower.num_patches
        del base_model #这里就是为了获得image_processor
        del vision_tower
        torch.cuda.empty_cache()

        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end #false
        training_args.use_im_start_end = model_args.mm_use_im_start_end #false

    # Dataset
    data_module: dict = make_rl_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    if accelerator.is_main_process:
        training_data = data_module["train_dataset"]
        for i in range(3):
            ex_input_ids_0 = training_data[i]["queries"]
            rank0_print(ex_input_ids_0[ex_input_ids_0 != tokenizer.pad_token_id]) #把前面的pad都省略了
            ex_input_ids_0[ex_input_ids_0 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            rank0_print(tokenizer.decode(ex_input_ids_0, skip_special_tokens=False))
            rank0_print("=" * 20)

    rank = int(os.environ.get("RANK", 0)) #0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) #1
    node_id = rank // torch.cuda.device_count() #0

    print(f"Distributed info: rank={rank}, world_size={world_size}, node_id={node_id}")
    setup_deepspeed_plugin(args)
    
    model_module = make_models(
        tokenizer=tokenizer,
        args=args,
        # accelerator=accelerator,
    )

    trainer = PPOTrainer(
        args=training_args,
        accelerator=accelerator,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    train()
