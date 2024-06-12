
from argparse import Namespace
from dataclasses import dataclass
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput

from peft import PeftModel, LoraModel, LoraConfig

from models.qlora_model import get_accelerate_model

from llava.model import *



def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


def make_generative_vlm(
    args: Namespace,
    model_name_or_path: str,
    qlora: bool = False,
    checkpoint_dir: Optional[str] = None,
    adapter_name="lora_default",
    is_trainable=True,
    tokenizer=None,
    **kwargs,
):
    if qlora:
        if checkpoint_dir is None or checkpoint_dir in ["scratch", "none"]:
            return get_accelerate_model(args, None, tokenizer=tokenizer)
        else:
            return get_accelerate_model(
                args,
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                tokenizer=tokenizer,
            )
    else:
        raise ValueError(f"Unknown model type: {model_name_or_path}")


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        hidden_size_attr_name = "hidden_size"
    else:
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path

@dataclass
class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: RewardConfig,
        args: Namespace,
        checkpoint_dir: Optional[str] = None,
        tokenizer=None,
        is_trainable=False,
        adapter_name="lora_default",
        **kwargs,
    ):
        super(RewardModel, self).__init__(config)
        self.args = args
        self.is_trainable = is_trainable
        # self.adapter_name = adapter_name
        if args.lora_enable:
            self.adapter_name = adapter_name
            self.backbone_model = make_generative_vlm(
                args=args,
                qlora=True,
                model_name_or_path=config.backbone_model_name_or_path, 
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                tokenizer=tokenizer,
                **kwargs,
            )
        else:
            self.backbone_model =  LlavaLlamaForCausalLM.from_pretrained(
                config.backbone_model_name_or_path,
                cache_dir=args.cache_dir,
                torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                trust_remote_code=args.trust_remote_code, 
            )
            self.backbone_model.config.use_cache = False
            self.backbone_model.config.tokenizer_padding_side = 'left'
            if args.gradient_checkpointing:
                if hasattr(self.backbone_model, "enable_input_require_grads"):
                    self.backbone_model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                self.backbone_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
            if args.vision_tower is not None:
                self.backbone_model.config.image_aspect_ratio = args.image_aspect_ratio #image_aspect_ratio = pad
                self.backbone_model.config.image_grid_pinpoints = args.image_grid_pinpoints #None

                vision_tower = self.backbone_model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model()
                vision_tower.to(device="cuda", dtype=torch.bfloat16)
                vision_tower.requires_grad_(False) 

                mm_projector = self.backbone_model.get_model().mm_projector
                mm_projector.to(device="cuda", dtype=torch.bfloat16)
        if checkpoint_dir is not None and args.finetune_mm_projector:
            print("Loading rm projector from checkpoint.")
            mm_projector_path = os.path.join(checkpoint_dir, "mm_projector.bin")
            if os.path.exists(mm_projector_path):
                mm_projector_weights = torch.load(mm_projector_path, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.backbone_model.get_model().mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            else:
                print(f"Warning: mm_projector not found at {mm_projector_path}")
        if args.finetune_mm_projector:
            self.backbone_model.get_model().mm_projector.requires_grad_(True)
        else:
            self.backbone_model.get_model().mm_projector.requires_grad_(False) 

        hidden_size = get_transformer_hidden_size(self.backbone_model) 
        reward_head = nn.Linear(hidden_size, 1, bias=False)
        device = next(self.backbone_model.parameters()).device
        self.reward_head = reward_head.to(device)

        # Conditional loading from checkpoint.
        if checkpoint_dir is not None:
            print("Loading reward_head from checkpoint.")
            reward_head_path = os.path.join(checkpoint_dir, "reward_head")
            if os.path.exists(reward_head_path):
                self.reward_head.load_state_dict(torch.load(reward_head_path, map_location="cpu"))
            else:
                print(f"Warning: reward head not found at {reward_head_path}")

        self.reward_head.requires_grad_(True)
        if self.args.lora_enable and self.is_trainable and self.adapter_name is not None:
            self.backbone_model.set_adapter(self.adapter_name)

    def forward(
        self, input_ids, attention_mask=None, images=None, return_dict=True, rank_all=False, **kwargs
    ):
        self.backbone_model.config.use_cache = False

        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            images=images,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)
        if rank_all:
            last_hidden_state = last_hidden_state.type_as(self.reward_head.weight)
            rewards = self.reward_head(last_hidden_state).squeeze(-1)
        else:
            last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
            last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
                self.reward_head.weight
            )
            rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)



def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class RewardModelTrainer(transformers.Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.args.lora_enable:
            if getattr(self.args, "finetune_mm_projector", False):
                # Save the model
                _state_dict = state_dict
                if _state_dict is None:
                    # Only save the model itself if we are using distributed training
                    model_to_save = unwrap_model(self.model)
                    _state_dict = model_to_save.state_dict()

                weight_to_save = {}
                keys_to_match = ["mm_projector"]
                for k, v in _state_dict.items():
                    if any(key_match in k for key_match in keys_to_match):
                        weight_to_save[k] = v

                current_folder = output_dir.split("/")[-1]
                parent_folder = os.path.dirname(output_dir)
                if current_folder.startswith("checkpoint-"):
                    mm_projector_folder = os.path.join(output_dir, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(
                        weight_to_save,
                        os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                    )
                else:
                    torch.save(
                        weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                    )
        else:
            super(RewardModelTrainer, self)._save(output_dir, state_dict)

    def compute_loss_before(self, model, inputs, return_outputs=False):
        input_ids, attention_mask, index_0, index_1, choice, images = unpack_dict(
            inputs,
            keys=(
                "input_ids", 
                "attention_mask", 
                "index_0",
                "index_1",
                "choice", 
                "images", 
            ),
        )
        # repeat images to match the number of candidates
        images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1) 
        images = einops.rearrange(images, "b n h w c -> (b n) h w c") 

        num_candidates, num_pairs = input_ids.size(1), choice.size(1) 
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        ) 
        outputs = model(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
        )
        rewards_flat = outputs.rewards 
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  

        rewards_0, rewards_1 = tuple(
            batch_select(rewards, index) for index in (index_0, index_1)
        )  
        logits = rewards_1 - rewards_0  
        loss = F.binary_cross_entropy_with_logits(
            logits, choice.to(logits.dtype), reduction="mean" 
        )

        loss = loss + (rewards_1 + rewards_0).mean().abs() * 1e-3

        logged_rewards = torch.stack((rewards_1, rewards_0), dim=-1)
        return (loss, dict(logits=logged_rewards)) if return_outputs else loss


    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, attention_mask, images = unpack_dict( 
            inputs,
            keys=(
                "input_ids", 
                "attention_mask", 
                "images", 
            ),
        )
        # repeat images to match the number of candidates
        images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1) 
        images = einops.rearrange(images, "b n h w c -> (b n) h w c") 

        num_candidates = input_ids.size(1) # 2，1]
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        ) 
        outputs = model(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
        )
        rewards_flat = outputs.rewards 
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  

        rewards_chosen = rewards[:, 0]
        rewards_rejected = rewards[:, 1]
        probs = torch.sigmoid(rewards_chosen - rewards_rejected)
        loss = (-torch.log(probs + 1e-5)).mean()

        logged_rewards = torch.stack((rewards_chosen, rewards_rejected), dim=-1)
        return (loss, dict(logits=logged_rewards)) if return_outputs else loss


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    rewards_chosen = eval_prediction.predictions[:, 0]
    rewards_rejected = eval_prediction.predictions[:, 1]
    accuracy=(rewards_chosen > rewards_rejected).mean().item()
    return dict(
        accuracy=accuracy
    )
    




def load_4bit_reward_model_for_inference(
    checkpoint_dir: str,
    vision_tower: str = None,
    lora_modules: list = None,
    image_aspect_ratio: str = "square",
    image_grid_pinpoints: int = None,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    trust_remote_code=False,
    finetune_mm_projector=False,

):
    # Load the model.
    lora_checkpoint_dir = checkpoint_dir
    if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_model")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "adapter_model")
    if os.path.exists(os.path.join(lora_checkpoint_dir, "lora_default")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "lora_default")

    lora_config = LoraConfig.from_pretrained(lora_checkpoint_dir)
    config = RewardConfig(
        backbone_model_name_or_path=lora_config.base_model_name_or_path
    )

    args = Namespace(
        lora_enable=True,
        model_name_or_path=config.backbone_model_name_or_path,
        vision_tower=vision_tower,
        lora_modules=lora_modules,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        trust_remote_code=trust_remote_code,
        gradient_checkpointing=gradient_checkpointing,
        finetune_mm_projector=finetune_mm_projector,
    )

    model = RewardModel(
        config=config,
        args=args,
        checkpoint_dir=checkpoint_dir,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )
    return model
