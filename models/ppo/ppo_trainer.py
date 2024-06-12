
import abc
import copy
import glob
import dataclasses
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd

import einops
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import accelerate
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import convert_outputs_to_fp32

import transformers
from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict
from transformers.trainer_utils import enable_full_determinism, set_seed

from all_utils.data_utils.data_utils_ppo import QueryResponseDataset, pad_sequences, rolloutify
import all_utils.data_utils.common_utils as utils
import all_utils.distributed_utils as distributed_utils
from all_utils.trainer_utils import create_optimizer, create_scheduler, get_eval_ds_config
import all_utils.data_utils.common_utils as common_utils
import deepspeed

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


logger = logging.getLogger(__name__)

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  
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



class KLController(abc.ABC):
    value: Union[int, float]

    def step(self, *args, **kwargs):
        pass


class FixedKLController(KLController):
    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = kl_coef



class PPOTrainer(object):
    def __init__(
        self,
        args,
        train_dataset: QueryResponseDataset,
        eval_dataset: QueryResponseDataset,
        data_collator: Callable,
        tokenizer: transformers.PreTrainedTokenizer,
        policy: nn.Module,
        accelerator: accelerate.Accelerator,
        ref_policy: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(PPOTrainer, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.kl_ctl = FixedKLController(kl_coef=args.kl_coef)
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)
        enable_full_determinism(
            self.args.seed
        ) if self.args.full_determinism else set_seed(self.args.seed)


        self.total_epochs = self.args.total_epochs 
        self.total_episodes = len(self.train_dataset) * self.total_epochs  
        self.total_steps = self.total_episodes // self.args.step_batch_size 
        self.per_epoch_steps = max(1, len(self.train_dataset) // self.args.step_batch_size)
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler(self.total_steps)
        self.policy, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.policy, self.optimizer, self.lr_scheduler)

        eval_ds_config = get_eval_ds_config(offload=True)
        self.reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        self.reward_model.eval()
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
    
    def _tensor2vec(self, responses):
        responses_list = responses.tolist()
        responses_vec = []
        for response in responses_list:
            response = self._strip_pad(response)
            responses_vec.append(response)
        return responses_vec

    def _shape_reward(
        self,
        rewards: torch.Tensor,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if self.args.kl_approximator == "k1":
            # KL (q | p) = sum_i q_i (log q_i - log p_i)
            # kl = torch.clamp(logprobs - ref_logprobs, min=0.0) 
            kl = logprobs - ref_logprobs
        elif self.args.kl_approximator == "k3":
            # r = p / q, log r = log p - log q
            # KL (q | p) = (r - 1) - log r = e ^ log r - 1 - log r
            log_r = ref_logprobs - logprobs
            kl = torch.exp(log_r) - 1.0 - log_r
        else:
            raise ValueError(f"Unknown KL approximator: {self.args.kl_approximator}")

        non_score_rewards = -self.kl_ctl.value * kl #kl_ctl.value = 0.1
        shaped_rewards = non_score_rewards.clone()
        # This introduces a small index off by one bug if pad_token_id == eos_token_id.
        # terminal_positions = (responses != self.tokenizer.pad_token_id).sum(dim=1) - 1
        # shaped_rewards[list(range(rewards.size(0))), terminal_positions] += rewards

        shaped_rewards[:, -1] += (
            rewards
            + self.args.reward_bias #0.0
        )
        return dict(
            shaped_rewards=shaped_rewards
        )
    

    def _estimate_advantage(
        self, rewards: torch.Tensor, values: torch.Tensor, responses: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        if self.args.whiten_rewards:
            rewards = whiten(
                rewards, shift_mean=False, async_stats=self.args.whitening_async_stats
            )
        else:
            rewards = rewards * 10.0
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.response_len
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = whiten(
            advantages, shift_mean=True, async_stats=self.args.whitening_async_stats
        )
        return dict(returns=returns, advantages=advantages)

    def _get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        '''
        Copied from TRLX: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        '''
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + self.args.gamma * nextvalues - values[t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns



    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        # Give up dropout throughout.
        self.policy.eval()
        # `keep_fp32_wrapper` retains the autocast wrapper of model.forward created by accelerate:
        #  recall one sets mixed precision options with accelerator.
        # The precise value of this arg doesn't matter here, since we use the unwrapped model only for respond.
        # Generally, try to use the wrapped model as much as you can, since it's got the autocast/cast-back wrappers.
        unwrapped_policy = self.accelerator.unwrap_model(
            self.policy, keep_fp32_wrapper=True
        ) #pay attention to unwrapped_policy

        self.ref_policy.eval()
        self.reward_model.eval()

        rollouts = []
        images_list = []
        ids_list = []
        rewards_list = []
        rollout_samples = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            total=len(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            gc.collect()
            torch.cuda.empty_cache()
            # Sample rollouts.
            (
                images,
                ids,
                queries,
            ) = common_utils.unpack_dict(
                common_utils.prepare_inputs(batch, device=self.accelerator.device),
                keys=(
                    "images",
                    "ids",
                    "queries",
                ),
            )
            images_list.append(images)
            ids_list.append(ids)
            queries = self._adjust_queries(queries=queries)
            query_attn_masks = queries.ne(self.tokenizer.pad_token_id)

            if self.args.bf16:
                images = images.to(torch.bfloat16)
            elif self.args.fp16:
                images = images.half()

            respond_outputs = unwrapped_policy.respond(
                queries, query_attn_masks, images, temperature=self.args.temperature
            ) #print(respond_outputs['responses'].shape): torch.Size([32, 896])
            (responses,) = common_utils.unpack_dict(respond_outputs, ("responses",))
            responses = truncate_after_eos_with_padding(
                responses,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            )
            queries_vec, responses_vec, sequences_vec = self._extract_queries_responses(queries,responses)
            sequences = torch.tensor(pad_sequences(sequences_vec, pad_value=self.tokenizer.pad_token_id),dtype=torch.long, device=self.accelerator.device) #这里需不需要加到cuda里面？
            sequences_attention_mask = sequences.ne(self.tokenizer.pad_token_id)
            # Evaluate logprobs of the samples.
            policy_outputs = self.policy(
                sequences, sequences_attention_mask, responses_vec, images, temperature=self.args.temperature
            )
            ref_policy_outputs = self.ref_policy(
                sequences, sequences_attention_mask, responses_vec, images, temperature=self.args.temperature
            ) #original_logits, logits, logprobs, entropies, last_hidden_state, values

            reward_outputs = self.reward_model(
                input_ids=sequences, attention_mask=sequences_attention_mask, images=images,
            )
            (rewards,) = common_utils.unpack_dict(reward_outputs, ("rewards",))
            rewards_list.append(rewards)

            logprobs, values= policy_outputs['logprobs'], policy_outputs['values']

            ref_logprobs = ref_policy_outputs['logprobs']
            # Shape reward with KL penalty.
            shaped_rewards = self._shape_reward(rewards, logprobs, ref_logprobs)['shaped_rewards']
            bsz = len(sequences_vec)
            for i in range(bsz):
                response_length = len(responses_vec[i])
                values_sample = values[i][-response_length:].tolist()
                rewards_sample = shaped_rewards[i][-response_length:].tolist()
                advantages, returns = self._get_advantages_and_returns(rewards_sample, values_sample)
                sample = {
                    'query': queries_vec[i], 
                    'response': responses_vec[i],
                    'sequence': sequences_vec[i],
                    'values': values_sample,
                    'logprobs': logprobs[i][-response_length:].tolist(),
                    'advantages': advantages,
                    'returns': returns,
                    'loss_mask': [1] * response_length
                }
                rollout_samples.append(sample)
        images_all = torch.cat(images_list, dim=0)
        ids_all = torch.cat(ids_list, dim=0)
        rewards_all = torch.cat(rewards_list, dim=0)
        rollouts = rolloutify(rollout_samples, self.tokenizer)
        sequences = rollouts['sequences']
        sequences_attention_mask = sequences.ne(self.tokenizer.pad_token_id)
        rollouts.update({"images": images_all})
        rollouts.update({"ids": ids_all})
        rollouts.update({"sequences_attention_mask": sequences_attention_mask})
        rollouts.update({"rewards": rewards_all})
        rollouts = {key: value.cpu() for key, value in rollouts.items()}
        del rollout_samples
        gc.collect()
        torch.cuda.empty_cache()
        return {**rollouts}

    @abc.abstractmethod
    def compute_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError
        
    def compute_policy_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            old_logprob,
            returns,
            advantages,
            sequences,
            sequences_attention_mask,
            responses,
            images,
            loss_mask
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "logprobs",
                    "returns",
                    "advantages",
                    "sequences",
                    "sequences_attention_mask",
                    "responses",
                    "images",
                    "loss_mask"
                ),
            ),
            device=self.accelerator.device,
        )

        # Enable training mode for gradient checkpointing.
        self.policy.train()
        response_vec = self._tensor2vec(responses)
        n = loss_mask.sum()
        outputs = self.policy(
            sequences,
            sequences_attention_mask,
            response_vec,
            images,
            temperature=self.args.temperature,
            mode="policy",
        )

        logprob = outputs["logprobs"]
        log_ratio = logprob - old_logprob 
        ratio = torch.exp(log_ratio)
        # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange
        )
        pg_loss = torch.sum(torch.max(pg_losses, pg_losses2)) / n
        pg_clipfrac = torch.sum((pg_losses2 > pg_losses).float()) / n

        loss = pg_loss + outputs["dummy_loss"]

        # entropy = outputs["entropies"].mean()
        with torch.no_grad():
            approxkl = torch.sum((ratio - 1) - log_ratio) / n

        # return_mean, return_var = returns.mean(), returns.var(unbiased=False)
        with torch.no_grad():
            stats = dict(
                ppo=dict(pg_loss=pg_loss),
                policy=dict(approxkl=approxkl, pgclip=pg_clipfrac, ratio=ratio.sum()/n),
            )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    def compute_value_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            old_values,
            returns,
            sequences,
            sequences_attention_mask,
            responses,
            images,
            loss_mask
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "values",
                    "returns",
                    "sequences",
                    "sequences_attention_mask",
                    "responses",
                    "images",
                    "loss_mask"
                ),
            ),
            device=self.accelerator.device,
        )

        # Enable training mode for graident checkpointing.
        self.policy.train()
        n = loss_mask.sum()
        response_vec = self._tensor2vec(responses)
        outputs = self.policy(
            sequences,
            sequences_attention_mask,
            response_vec,
            images,
            temperature=self.args.temperature,
            mode="value",
        )

        values = outputs["values"]
        values_list = []
        bsz = len(response_vec)
        for i in range(bsz):
            response_length = len(response_vec[i])
            value_sample = values[i][-response_length:]
            values_list.append(value_sample)
        values = torch.nn.utils.rnn.pad_sequence(values_list, batch_first=True, padding_value=0.).to(sequences.device)

        values_clipped = torch.clamp(
            values,
            min=old_values - self.args.cliprange_value,
            max=old_values + self.args.cliprange_value,
        )
        vf_losses1 = (values - returns) ** 2.0
        vf_losses2 = (values_clipped - returns) ** 2.0
        vf_loss = 0.5 * torch.sum(torch.max(vf_losses1, vf_losses2)) / n
        ### vf_loss = 0.5 * torch.sum(vf_loss1 * loss_mask) / n
        vf_clipfrac = torch.sum((vf_losses2 > vf_losses1).float()) / n

        loss = self.args.vf_coef * vf_loss + outputs["dummy_loss"]

        # value_mean, value_var = values.mean(), values.var(unbiased=False)
        with torch.no_grad():
            stats = dict(
                ppo=dict(vf_loss=vf_loss),
                value=dict(
                    values=(values.sum() / n),
                    # error=((vpred - returns) ** 2).mean(),
                    vf_clip=vf_clipfrac,
                    values_clipped=values_clipped.sum()/n,

                ),
            )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    

    





       

    @torch.inference_mode()
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        n = rollouts['loss_mask'].sum()
        stats = {
            f"rollout/advantages": rollouts['advantages'].sum()/n,
            f"rollout/returns": rollouts['returns'].sum()/n,
            f'lr': self.optimizer.param_groups[0]["lr"],
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v

        stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in stats.items()
        }
        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)
            if self.args.output_dir is not None:
                # Store rollout data to disk to debug.
                rollouts_to_disk = {
                    key: [
                        text.replace("<unk>", "")  
                        for text in self.tokenizer.batch_decode(
                            remove_image_token(tensor),
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    ]
                    for key, tensor in common_utils.unpack_dict(
                        rollouts, keys=("queries", "responses"), return_type=dict
                    ).items()
                }

                rewards = [str(_) for _ in rollouts["rewards"].tolist()]
                ids = [str(_) for _ in rollouts["ids"].tolist()]
                rollouts_to_disk["rewards"] = rewards
                rollouts_to_disk["ids"] = ids

                rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(
                    orient="records"
                )
                rollout_log_dir = os.path.join(self.args.output_dir, "rollouts")
                os.makedirs(rollout_log_dir, exist_ok=True)
                with open(
                    os.path.join(rollout_log_dir, f"step_{step_idx}.json"),
                    "w",
                ) as f:
                    json.dump(rollouts_to_disk, f, indent=4)
                del rollouts_to_disk
        del rollouts
        del train_stats
        gc.collect()
        torch.cuda.empty_cache()
        return stats

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

    def step_with_rollouts(self, rollouts):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        assert isinstance(self.optimizer, AcceleratedOptimizer), (
            "`optimizer` must be pushed through `accelerator.prepare`. "
            "Otherwise the `accelerator.accumulate` context manager won't correctly disable `zero_grad` or `step`."
        )
        stats_list = []
        gc.collect()
        torch.cuda.empty_cache()
        with self.accelerator.accumulate(self.policy):
            stats_for_this_step = {}
            with self.accelerator.no_sync(self.policy):
                policy_loss, policy_stats = self.compute_policy_loss(
                    rollouts #这个的长度永远等于
                )
                stats_for_this_step.update(policy_stats)
                self.accelerator.backward(policy_loss)

            value_loss, value_stats = self.compute_value_loss(rollouts)
            stats_for_this_step.update(value_stats)
            self.accelerator.backward(value_loss)

            if self.accelerator.sync_gradients:
                # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                if self.args.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.policy.parameters(), self.args.max_grad_norm
                    )
                # stats_for_this_step[
                #     "loss/grad_norm"
                # ] = self._compute_grad_norm()
                stats_list.append(stats_for_this_step)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return utils.merge_dict(
            stats_list, torch.stack
        )  # list of dict -> dict: str -> 1-D tensor

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [
            next(train_dataloader) for _ in range(self.args.rollouts_gradient_accumulation_steps)
        ]
        rollouts = self.rollout(queries_batches)
        train_stats = self.step_with_rollouts(rollouts)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = self.record_step_stats(
            rollouts=rollouts,
            train_stats=train_stats,
            step_idx=step_idx,
            kl_coef=self.kl_ctl.value,
        )
        # self.kl_ctl.step(stats["objective/kl_sum_seq"])
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

    def train(self, resume_training_ckpt: Optional[str] = None):
        """Entry point for training."""
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {self.total_epochs} => Total episodes: {self.total_episodes} => Total steps: {self.total_steps}"
        )


        skipping_steps = 0
        if resume_training_ckpt is not None:
            skipping_steps = self.resume_training(resume_training_ckpt)
            print(
                f"Resuming training from {resume_training_ckpt} at step {skipping_steps}."
            )

        infinite_train_dataloader = self.get_train_dataloader() #
        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, self.total_steps + FIRST_STEP_IDX),
            disable=not self.accelerator.is_main_process,
            desc="steps",
            total=self.total_steps,
        ):
            if step_idx < skipping_steps: 
                for _ in range(self.args.rollout_accumulation_steps):
                    next(infinite_train_dataloader)
                continue

            if (
                step_idx % self.per_epoch_steps == 0
            ):
                if step_idx > skipping_steps:
                    self.save_checkpoint(
                        os.path.join(self.args.output_dir, f"checkpoint-{step_idx}")
                    )
            if (
                self.args.eval_steps is not None
                and step_idx % self.args.eval_steps == 0
            ):
                self.evaluate(step_idx)
            self.log_history.append(self.step(infinite_train_dataloader, step_idx))
        return self.log_history

    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None):
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
            policy_model = unwrapped_policy.policy.base_model
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
                    keys_to_match = ["mm_projector", "embed_tokens", "embed_in"]
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
            policy_model = unwrapped_policy.policy.base_model

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
        )  
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

    def get_rollouts_dataloader(
        self, rollouts: Dict[str, torch.Tensor], shuffle=True, drop_last=True, keys=None
    ):
        if keys is None:
            keys = tuple(rollouts.keys())

        def collate_rollouts(instances: Sequence[tuple]):
            return {
                key: torch.stack([instance[idx] for instance in instances])
                for idx, key in enumerate(keys)
            }

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])
        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=self.args.batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Do not prepare, since we don't need to shard the rollouts sampled on each batch.
        return rollouts_dataloader





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



def whiten(
    values: torch.Tensor, shift_mean=True, epsilon=1e-8, async_stats="full_batch"
) -> torch.Tensor:
    assert async_stats in ["full_batch", "per_gpu", "none"]

    values_for_statistics = values
    if async_stats == "full_batch":
        if not values_for_statistics.is_cuda:
            raise ValueError("SyncWhiten expected input tensor to be on GPU")

        need_sync = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        if need_sync:
            process_group = torch.distributed.group.WORLD
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        if need_sync:
            tensor_list = [
                torch.zeros_like(values_for_statistics) for _ in range(world_size)
            ]
            torch.distributed.all_gather(tensor_list, values_for_statistics)
            values_for_statistics = torch.cat(tensor_list, dim=0)

    if async_stats in ["full_batch", "per_gpu"]:
        # assert (
        #     values_for_statistics.size(0) >= 8
        # ), f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
        mean = values_for_statistics.mean()  # noqa
        std = values_for_statistics.std(unbiased=False)  # noqa

    else:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, unbiased=False, keepdim=True)

    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


