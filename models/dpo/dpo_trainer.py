

import abc
import gc
import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import accelerate
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import convert_outputs_to_fp32

import transformers
from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict
from transformers.trainer_utils import enable_full_determinism, set_seed

from all_utils.data_utils.data_utils_ppo import pad_sequences, rolloutify
import all_utils.data_utils.common_utils as utils
from all_utils.data_utils.data_utils_dpo import QueryChosenRejectedResponseDataset
from all_utils.trainer_utils import create_optimizer, create_scheduler, get_eval_ds_config
import all_utils.data_utils.common_utils as common_utils
import deepspeed
from llava.constants import (
    IMAGE_TOKEN_INDEX,
)


logger = logging.getLogger(__name__)

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler


# Name of the files used for checkpointing
ADAPTER_MODEL_DIR = "adapter_model"
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
VALUE_HEAD_NAME = "value_head.pt"
SCALER_NAME = "scaler.pt"

FIRST_STEP_IDX = 1



class DPOTrainer(object):
    def __init__(
        self,
        args,
        train_dataset: QueryChosenRejectedResponseDataset,
        eval_dataset: QueryChosenRejectedResponseDataset,
        data_collator: Callable,
        tokenizer: transformers.PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        accelerator: accelerate.Accelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(DPOTrainer, self).__init__()
        self.args = args
        self.dpo_beta = self.args.dpo_beta
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)
        enable_full_determinism(
            self.args.seed
        ) if self.args.full_determinism else set_seed(self.args.seed)

        self.total_epochs = self.args.total_epochs 
        self.total_episodes = len(self.train_dataset) * self.total_epochs  
        self.total_steps = self.total_episodes // self.args.step_batch_size  
        self.per_epoch_steps = len(self.train_dataset) // self.args.step_batch_size
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler(self.total_steps)
        self.policy, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.policy, self.optimizer, self.lr_scheduler)

        eval_ds_config = get_eval_ds_config(offload=False)
        self.ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
        self.ref_policy.eval()
    
    def _adjust_queries(self, queries):
        non_zero_lengths = (queries != 0).long().sum(dim=1)
        max_length = non_zero_lengths.max()
        adjusted_queries = queries[:, -max_length:]
        return adjusted_queries

    def _strip_pad(self, seq: List[int]):
        return [tok for tok in seq if tok != self.tokenizer.pad_token_id]
    
    def _extract_queries_responses(self, quereis, responses):
        quereis_list = quereis.tolist() 
        responses_list = responses.tolist()
        quereis_vec, responses_vec = [], []
        for query, response in zip(quereis_list, responses_list):
            query = self._strip_pad(query)
            response = self._strip_pad(response)
            quereis_vec.append(query)
            responses_vec.append(response)
        sequences_vec = [c + r for c, r in zip(quereis_vec, responses_vec)]
        return quereis_vec, responses_vec, sequences_vec
    

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            queries,
            chosen_responses,
            rejected_responses,
            images,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                batch,
                keys=(
                    "queries",
                    "chosen_responses",
                    "rejected_responses",
                    "images",
                ),
            ),
            device=self.accelerator.device,
        )   
        repeated_queries = torch.cat([queries, queries], dim=0)
        repeated_images = torch.cat([images, images], dim=0)
        responses = torch.cat([chosen_responses, rejected_responses], dim=0)
        queries_vec, responses_vec, sequences_vec = self._extract_queries_responses(repeated_queries, responses)
        sequences = torch.tensor(pad_sequences(sequences_vec, pad_value=self.tokenizer.pad_token_id),dtype=torch.long, device=self.accelerator.device)
        sequences_attention_mask = sequences.ne(self.tokenizer.pad_token_id)

        self.policy.train()
        self.ref_policy.eval()
        policy_outputs = self.policy(
            sequences,
            sequences_attention_mask,
            responses_vec,
            repeated_images,
            temperature=self.args.temperature,
        )
        with torch.no_grad():
            ref_policy_outputs = self.ref_policy(
                sequences,
                sequences_attention_mask,
                responses_vec,
                repeated_images,
                temperature=self.args.temperature,
            )
        logprobs = policy_outputs["logprobs"].sum(-1)
        ref_logprobs = ref_policy_outputs['logprobs'].sum(-1)
        bs = len(queries)
        yw_logprobs, yl_logprobs = logprobs[:bs], logprobs[bs:]
        ref_yw_logprobs, ref_yl_logprobs = ref_logprobs[:bs], ref_logprobs[bs:]
        pi_logratios = yw_logprobs - yl_logprobs
        ref_logratios = ref_yw_logprobs - ref_yl_logprobs
        losses = -torch.nn.functional.logsigmoid(self.dpo_beta * (pi_logratios - ref_logratios).float()).to(pi_logratios.dtype)
        with torch.no_grad():
            yw_rewards = self.dpo_beta * (yw_logprobs - ref_yw_logprobs)
            yl_rewards = self.dpo_beta * (yl_logprobs - ref_yl_logprobs)
            greater_rewards = yw_rewards > yl_rewards
            probability = greater_rewards.float().mean().item()
            stats = dict(accuracy = probability)
        return losses.mean(), stats
    
    

    
       

    @torch.inference_mode()
    def record_step_stats(self, train_stats, step_idx, **kwargs):
        if self.accelerator.is_main_process:
            print(train_stats)
            self.accelerator.log(train_stats, step=step_idx)
        gc.collect()
        torch.cuda.empty_cache()
        return train_stats

    @property
    def optimizable_params(self):
        return [
            p
            for p in self.policy.parameters()
            if p.requires_grad and p.grad is not None
        ]

    @torch.inference_mode()
    def _compute_grad_norm(self):
        grad_norm = torch.stack([p.grad.norm(2) for p in self.optimizable_params]).norm(
            2
        )
        return grad_norm

    @torch.inference_mode()
    def _compute_param_norm(self):
        param_norm = torch.stack([p.norm(2) for p in self.optimizable_params]).norm(2)
        return param_norm

    def step_with_batch(self, batch):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        assert isinstance(self.optimizer, AcceleratedOptimizer), (
            "`optimizer` must be pushed through `accelerator.prepare`. "
            "Otherwise the `accelerator.accumulate` context manager won't correctly disable `zero_grad` or `step`."
        )
        gc.collect()
        torch.cuda.empty_cache()
        with self.accelerator.accumulate(self.policy):
            stats_for_this_step = {}
            with self.accelerator.no_sync(self.policy):
                policy_loss, policy_stats = self.compute_loss(
                    batch #这个的长度永远等于
                )
                stats_for_this_step.update(policy_stats)
                self.accelerator.backward(policy_loss)

            if self.accelerator.sync_gradients:
                # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                if self.args.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.policy.parameters(), self.args.max_grad_norm
                    )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return stats_for_this_step

    def step(self, train_dataloader, step_idx: int):
        batch = next(train_dataloader)
        train_stats = self.step_with_batch(batch)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = self.record_step_stats(
            train_stats=train_stats,
            step_idx=step_idx,
        )
        return stats

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = create_optimizer(
            args=self.args, model=self.policy, optimizer=self.optimizer
        )
        lr_scheduler = create_scheduler(
            args=self.args,
            optimizer=optimizer,
            lr_scheduler=self.lr_scheduler,
            num_training_steps=num_training_steps,
        )
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            optimizer, lr_scheduler
        )
        self.accelerator.register_for_checkpointing(
            self.lr_scheduler
        )  # LR scheduler needs another call to save.
        return self.optimizer, self.lr_scheduler

    def train(self):
        """Entry point for training."""
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {self.total_epochs} => Total episodes: {self.total_episodes} => Total steps: {self.total_steps}"
        )
        infinite_train_dataloader = self.get_train_dataloader() #
        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, self.total_steps + FIRST_STEP_IDX),
            disable=not self.accelerator.is_main_process,
            desc="steps",
            total=self.total_steps,
        ):
            if (step_idx % self.per_epoch_steps == 0): # or step_idx in self.args.save_steps_extra_list
                    self.save_checkpoint(os.path.join(self.args.output_dir, f"checkpoint-{step_idx}"))
            if (self.args.eval_steps is not None and step_idx % self.args.eval_steps == 0):
                self.evaluate(step_idx)
            self.log_history.append(self.step(infinite_train_dataloader, step_idx)) 
        return self.log_history

    @torch.inference_mode()
    def evaluate(self, step_idx: int):
        raise NotImplementedError
    
    def save_checkpoint(
        self,
        output_dir: Optional[str] = None,
    ):
        output_dir = self.args.output_dir if output_dir is None else output_dir

        global_rank = int(os.environ.get("RANK", 0))

        if global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            print("Saving model checkpoint to %s" % output_dir)

            # Save policy model.
            unwrapped_policy = self.accelerator.unwrap_model(
                self.policy, keep_fp32_wrapper=True
            )
            policy_model = unwrapped_policy.base_model
            if self.args.lora_enable:
                peft_model_path = os.path.join(output_dir, ADAPTER_MODEL_DIR)
                save_adapters(
                    policy_model,
                    peft_model_path,
                    adapter_names=["lora_policy"],
                )
                if getattr(self.args, "finetune_mm_projector", False):
                # Save the model
                    _state_dict = policy_model.state_dict()
                    weight_to_save = {}
                    keys_to_match = ["mm_projector"]
                    for k, v in _state_dict.items():
                        if any(key_match in k for key_match in keys_to_match):
                            weight_to_save[k] = v
                    torch.save(
                        weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                    )
            else:
                state_dict = self.accelerator.get_state_dict(policy_model)
                policy_model.save_pretrained(
                    output_dir,
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    state_dict=state_dict,
                )
                self.tokenizer.save_pretrained(output_dir)      
        else:
            print("Skipping checkpoint save on rank %d" % global_rank)

    @torch.inference_mode()
    def save_model(
        self,
        output_dir: Optional[str] = None,
    ):
        output_dir = self.args.output_dir if output_dir is None else output_dir

        global_rank = int(os.environ.get("RANK", 0))

        if global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            print("Saving model checkpoint to %s" % output_dir)

            # Save policy model.
            unwrapped_policy = self.accelerator.unwrap_model(
                self.policy, keep_fp32_wrapper=True
            )
            policy_model = unwrapped_policy.base_model

            peft_model_path = os.path.join(output_dir, ADAPTER_MODEL_DIR)

            # policy_model.save_pretrained(peft_model_path)
            save_adapters(
                policy_model,
                peft_model_path,
                adapter_names=["lora_policy"],
            )
        else:
            print("Skipping PEFT checkpoint save on rank %d" % global_rank)

    @abc.abstractmethod
    @torch.inference_mode()
    def resume_training(self, checkpoint_dir: str):
        raise NotImplementedError

    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        logger.warning(
            f"Batch size of {loader_name} dataloader: {batch_size}",
            # main_process_only=True,
        )

    def get_train_dataloader(self):
        logger.warning(
            f"Train dataset size: {len(self.train_dataset)}",
            # main_process_only=True
        )  # noqa
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.batch_size, 
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  
        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)




def remove_image_token(completions):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != IMAGE_TOKEN_INDEX]
        clean_completions[idx] = completion
    return clean_completions


def truncate_after_eos(completions, eos_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != IMAGE_TOKEN_INDEX]
        clean_completions[idx] = completion
        try:
            end_idx = completion.index(eos_token_id)
            clean_completions[idx] = completion[: end_idx + 1]
        except ValueError:
            pass
    return clean_completions


def truncate_after_eos_with_padding(
    completions, eos_token_id, pad_token_id, additional_tokens=None
):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        try:
            end_idx = completion.index(eos_token_id)
        except ValueError:
            end_idx = None

        if additional_tokens is not None:
            for additional_token in additional_tokens:
                try:
                    end_idx = completion.index(additional_token)
                except ValueError:
                    pass

        if end_idx is not None:
            clean_completions[idx] = completion[: end_idx + 1]

            if end_idx + 1 < len(completion):
                clean_completions[idx] = clean_completions[idx] + [pad_token_id] * (
                    len(completion) - end_idx - 1
                )

    clean_completions = torch.tensor(
        clean_completions, dtype=torch.long, device=completions.device
    )
    return clean_completions

def save_adapters(model, save_directory, adapter_names, **kwargs):
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        model: The model to save.
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        adapter_name (`str`):
            Name of the adapter to save.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
    os.makedirs(save_directory, exist_ok=True)
    # model.create_or_update_model_card(save_directory)

    for adapter_name, peft_config in model.peft_config.items():
        if adapter_name in adapter_names:
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                model,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    model.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode