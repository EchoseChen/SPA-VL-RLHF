
"""Model classes that are shared across different algorithms.

WARNING:
    Do not tamper with the state_dict function for any of these classes.
    If you tamper, make sure the keys are the same, otherwise FSDP will get confused.
"""

import abc
import logging
from typing import Dict, Optional
import accelerate

import torch
import transformers
from torch import Tensor, nn

import all_utils.data_utils.common_utils as utils
from models.qlora_model import load_4bit_model_for_inference
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from models.ppo.reward_model import load_4bit_reward_model_for_inference
from all_utils.data_utils.common_utils import compute_logprobs
from models.ppo.reward_model import RewardModel

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
        is_trainable=False,
    ):
        super().__init__()
        self.args = args
        self.is_trainable = is_trainable
        self.base_model = base_model 
        self.base_tokenizer = base_tokenizer
        if args.lora_enable:
            self.adapter_name = adapter_name 
        if self.args.lora_enable and self.is_trainable and self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)

    def forward(
        self,
        sequences: Tensor,
        sequences_attention_mask: Tensor,
        responses_vec,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        self.base_model.config.use_cache = False
        if temperature is None:
            temperature = self.args.temperature
        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequences_attention_mask,
            images=images,
            use_cache=False,
        ) 
        outputs = self.base_model(**inputs, output_hidden_states=False) 
        original_logits = outputs.logits
        logits = original_logits / temperature
        logits = logits[:,:-1,:]
        sequences = sequences[:, 1:]
        logprobs = compute_logprobs(logits=logits, sequences=sequences, responses_vec=responses_vec, device = logits.device)   
        return dict(
            logits=logits,
            logprobs=logprobs,
        )

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        self.base_model.config.use_cache = True

        if temperature is None:
            temperature = self.args.temperature
        sequences = self.base_model.generate(
            inputs=queries,
            images=images,
            attention_mask=query_attn_masks,
            do_sample=True,
            max_length = self.args.model_max_length,
            pad_token_id=self.base_tokenizer.pad_token_id,
            suppress_tokens=(
                [self.base_tokenizer.eos_token_id]
                if self.args.suppress_eos_at_generation
                else None
            ),
            top_p=1.0, #nucleus sampling
            top_k=0,
            temperature=temperature,
            num_return_sequences=num_return_sequences, #num_return_sequences = 1
            synced_gpus=True,
        ) 
        responses = sequences[:,1:] 
        return dict(
            responses=responses
        ) 

        


class Value(nn.Module):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        if args.lora_enable:
            self.adapter_name = adapter_name 

    def forward(
        self,
        sequences: Tensor,
        sequences_attention_mask: Tensor,
        images: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        self.base_model.config.use_cache = False


        inputs = self.base_model.backbone_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequences_attention_mask,
            images=images,
            use_cache=False,
        )
        values = self.base_model(
            **inputs,
            rank_all = True
        )['rewards']
        values = values[:, :-1] 
        return dict(values=values)




class ActorCritic(nn.Module):
    def __init__(self, policy: Policy, value_model: Value):
        super(ActorCritic, self).__init__()
        self.policy = policy
        self.value_model = value_model

    def forward(
        self,
        sequences: Tensor,
        sequences_attention_mask: Tensor,
        responses_vec,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        # Assume the policy and value model share the same tokenizer.

        if mode is None:
            o1 = self.policy(
                sequences, sequences_attention_mask, responses_vec, images, temperature
            )
            o2 = self.value_model(
                sequences, sequences_attention_mask, images, 
            )

        elif mode == "policy":
            o1 = self.policy(
                sequences, sequences_attention_mask, responses_vec, images, temperature
            ) #original_logits, logits,logprobs,entropies,last_hidden_state
            # Add dummy loss to make sure every parameter is used in the backward pass.
            o2 = {
                "dummy_loss": 0.0
                * (
                    torch.sum(
                        torch.stack(
                            [
                                torch.mean(value)
                                for key, value in self.named_parameters()
                                if self.policy.args.lora_enable and "lora_value" in key
                            ]
                        )
                    )
                    if self.policy.args.lora_enable
                    else torch.sum(
                        torch.stack([torch.mean(value) for _, value in self.named_parameters()])
                    )
                )
            }
        elif mode == "value":
            o2 = self.value_model(
                sequences, sequences_attention_mask, images
            ) #values
            # Add dummy loss to make sure every parameter is used in the backward pass.
            o1 = {
                "dummy_loss": 0.0
                * (
                    torch.sum(
                        torch.stack(
                            [
                                torch.mean(value)
                                for key, value in self.named_parameters()
                                if self.policy.args.lora_enable and "lora_policy" in key
                            ]
                        )
                    )
                    if self.policy.args.lora_enable
                    else torch.sum(
                        torch.stack([torch.mean(value) for _, value in self.named_parameters()])
                    )
                )
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {**o1, **o2}

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        return self.policy.respond(
            queries=queries,
            query_attn_masks=query_attn_masks,
            images=images,
            temperature=temperature,
        )

def make_generative_policy(adapter_name, is_trainable, args):
    if args.lora_enable:
        model = load_4bit_model_for_inference(
            model_name_or_path= args.policy_model_name_or_path,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=args.image_grid_pinpoints,
            bits=args.bits,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            trust_remote_code=args.trust_remote_code,
            lora_r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            vision_tower=args.vision_tower,
        )
    else:
        model =  LlavaLlamaForCausalLM.from_pretrained(
                args.policy_model_name_or_path,
                cache_dir=args.cache_dir,
                torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                trust_remote_code=args.trust_remote_code, 
            )
        model.config.use_cache = False
        model.config.tokenizer_padding_side = 'left'
        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        if args.bf16:
            model.to(torch.bfloat16)
        if args.vision_tower is not None:
            model.config.image_aspect_ratio = args.image_aspect_ratio 
            model.config.image_grid_pinpoints = args.image_grid_pinpoints 

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device="cuda", dtype=torch.bfloat16)
            vision_tower.requires_grad_(False) 

            mm_projector = model.get_model().mm_projector
            mm_projector.to(device="cuda", dtype=torch.bfloat16)

    if args.finetune_mm_projector:
        model.get_model().mm_projector.requires_grad_(True)
    else:
        model.get_model().mm_projector.requires_grad_(False) 
    if not is_trainable:
        for name, param in model.named_parameters():
            param.requires_grad = False  
    return model

def make_reward_model(adapter_name, is_trainable, args):
    if args.lora_enable:
        model = load_4bit_reward_model_for_inference(
            checkpoint_dir=args.reward_model_name_or_path,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=args.image_grid_pinpoints,
            bits=args.bits,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            trust_remote_code=args.trust_remote_code,
            finetune_mm_projector=args.finetune_mm_projector,
            vision_tower=args.vision_tower,
        )
    else:
        model = RewardModel.from_pretrained( 
                args=args,
                pretrained_model_name_or_path= args.reward_model_name_or_path,
                cache_dir=None,
                torch_dtype=torch.bfloat16,
                is_trainable=is_trainable,
                adapter_name=adapter_name,
                trust_remote_code=False, 
            )
    if not is_trainable:
        for name, param in model.named_parameters():
            param.requires_grad = False  
    return model


def make_models(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
) -> dict:


    policy = Policy(
        args = args,
        base_model = make_generative_policy(
            args=args,
            adapter_name="lora_policy",
            is_trainable=True,
        ),
        base_tokenizer = tokenizer,
        adapter_name ="lora_policy",
        is_trainable=True
    )

    value_model = Value(
        args = args,
        base_model = make_reward_model(
            args=args,
            adapter_name="lora_value",
            is_trainable=True,
        ),
        base_tokenizer = tokenizer,
        adapter_name ="lora_value",
    )


    actor_critic = ActorCritic(policy=policy, value_model=value_model)

    ref_policy = Policy(
        args,
        make_generative_policy(
            args=args,
            adapter_name="lora_ref_policy",
            is_trainable=False,
        ),
        tokenizer,
        adapter_name="lora_ref_policy",
        is_trainable=False,
    )

    reward_model = make_reward_model(
        args=args,
        adapter_name="lora_reward",
        is_trainable=False,
    )
    return dict(policy=actor_critic, ref_policy=ref_policy, reward_model=reward_model)



