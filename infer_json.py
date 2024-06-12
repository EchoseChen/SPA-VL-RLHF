# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
import json
import os
import re
import sys
from time import sleep
from typing import Optional, List
import logging

import torch
import torch.distributed as dist

import accelerate
from transformers import set_seed
import transformers

try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer

    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer

    print("Using slow tokenizer")

from transformers import AutoTokenizer, AutoModelForCausalLM

from all_utils.data_utils.data_utils_ppo import DataCollatorForQueryResponseDataset, QueryResponseDataset, make_rl_data_module, pad_sequences

from models.ppo.ppo_trainer import truncate_after_eos_with_padding, remove_image_token
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import all_utils.data_utils.common_utils as common_utils
from torch.distributed import all_gather, get_rank, is_initialized
import time

from llava import conversation as conversation_lib
from llava.model import *
from llava.constants import (
    IMAGE_TOKEN_INDEX,
)
from typing import List, Optional, Callable, Dict
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/bn/jsj-marl-challenge/chenlu/finetuning/llava_finetune_llama2_7b")
    lora_enable: bool = False
    peft_model_id_path: Optional[str] = field(default="/opt/tiger/llava-rlhf/llava/RLHF/result/new_ppo/checkpoint-150/adapter_model/lora_policy")
    temperature: float = field(default=1.0)
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    version: Optional[str] = field(default="v1")
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-2
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to stop generation with."},
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "pad"
    image_grid_pinpoints: Optional[str] = field(default=None)
    keywords: Optional[List[str]] = field(default_factory=lambda: ["harm"])
    dataset_path: str = field(default="/opt/tiger/llava-rlhf/llava/RLHF/prompts/test/config_test.json")


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    finetune_mm_projector: bool = field(default=False)
    # From AlpacaFarm
    truncate_tokens: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Tokens in strings to truncate at first occurrence. "
            "This was used in original OAI summarization paper to avoid models returning incomplete sentences. "
        },
    )
    suppress_eos_at_generation: bool = field(
        default=False,
        metadata={
            "help": "Whether to suppress the end-of-sequence token at generation time."
        },
    )
    num_patches: int = field(default=576)
    model_max_length: int = field(default=2048)
    query_len: int = field(default=128)
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    output_file: str = field(default="eval.json")



def make_json_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    if data_args.dataset_path.endswith('.json'):
        with open(data_args.dataset_path, 'r') as f:
            list_data_dict = json.load(f)
    eval_dataset = QueryResponseDataset(
        list_dict_data=list_data_dict,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
        data_args=data_args,
    )
    return dict(
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForQueryResponseDataset(),
    )

def custom_gather(data):
    output_lists = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(output_lists, data)
    # 扁平化列表
    return [item for sublist in output_lists for item in sublist]

def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

def adjust_queries(queries):
    non_zero_lengths = (queries != 0).long().sum(dim=1)
    max_length = non_zero_lengths.max()
    adjusted_queries = queries[:, -max_length:]
    return adjusted_queries

def strip_pad(seq: List[int], tokenizer):
        return [tok for tok in seq if tok != tokenizer.pad_token_id]

def extract_queries_responses(quereis, responses, tokenizer):
    quereis_list = quereis.tolist()
    responses_list = responses.tolist()
    quereis_vec, responses_vec = [], []
    for query, response in zip(quereis_list, responses_list):
        query = strip_pad(query, tokenizer)
        response = strip_pad(response, tokenizer)
        quereis_vec.append(query)
        responses_vec.append(response)
    sequences_vec = [c + r for c, r in zip(quereis_vec, responses_vec)]
    return quereis_vec, responses_vec, sequences_vec

def get_model_answer(batch, model, accelerator, tokenizer, args):
    unwrapped_policy = accelerator.unwrap_model(
    model, keep_fp32_wrapper=True
    )

    (
            images,
            ids,
            queries
        ) = common_utils.unpack_dict(
            common_utils.prepare_inputs(batch, device=accelerator.device),
            keys=(
                "images",
                "ids",
                "queries",
            ),
        )
    queries = adjust_queries(queries=queries)
    query_attn_masks = queries.ne(tokenizer.pad_token_id)
    images = images.to(torch.bfloat16)
    responses = unwrapped_policy.generate(
        inputs=queries,
        images=images,
        attention_mask=query_attn_masks,
        do_sample=False,
        max_length=args.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        suppress_tokens=(
            [tokenizer.eos_token_id]
            if args.suppress_eos_at_generation
            else None
        ),
        top_p=1.0, #nucleus sampling
        top_k=0,
        temperature=args.temperature,
        num_return_sequences=1, 
        synced_gpus=True,
    )
    responses = responses[:,1:]


    responses = truncate_after_eos_with_padding(
            responses,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
    queries_vec, responses_vec, sequences_vec = extract_queries_responses(queries,responses,tokenizer)
    sequences = torch.tensor(pad_sequences(sequences_vec, pad_value=tokenizer.pad_token_id),dtype=torch.long, device=accelerator.device)
    sequences_attention_mask = sequences.ne(tokenizer.pad_token_id)
    return remove_image_token(queries), responses, ids

@torch.inference_mode()
def eval():
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

    accelerator = accelerate.Accelerator()
    tokenizer_model_name = args.model_name_or_path 
    TokenizerClass = AutoTokenizer
    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir, 
        model_max_length=training_args.model_max_length, 
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ] 
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        model = LlavaLlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
        if args.lora_enable:
            model = PeftModel.from_pretrained(model, os.path.join(args.peft_model_id_path, "adapter_model/lora_policy"))
            mm_projector_path = os.path.join(args.peft_model_id_path, "mm_projector.bin")
            if os.path.exists(mm_projector_path):
                mm_projector_weights = torch.load(mm_projector_path, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                model.get_model().mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
                print(f"Success loaed mm_projector at {mm_projector_path}")
            else:
                print(f"Warning: mm_projector not found at {mm_projector_path}")
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        training_args.query_len = args.query_len = training_args.model_max_length - vision_tower.num_patches
        training_args.num_patches = args.num_patches = vision_tower.num_patches
        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end 
        training_args.use_im_start_end = model_args.mm_use_im_start_end 


    model.config.use_cache = False
    model.config.tokenizer_padding_side = 'left'
    if args.vision_tower is not None:
        model.config.image_aspect_ratio = args.image_aspect_ratio 
        model.config.image_grid_pinpoints = args.image_grid_pinpoints 
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
        mm_projector = model.get_model().mm_projector
        mm_projector.to(device="cuda", dtype=torch.bfloat16)
    model.to(dtype=torch.bfloat16, device=training_args.device)

    # Dataset
    data_module: dict = make_json_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    if accelerator.is_main_process:
        training_data = data_module["eval_dataset"]
        for i in range(3):
            ex_input_ids_0 = training_data[i]["queries"]
            rank0_print(ex_input_ids_0[ex_input_ids_0 != tokenizer.pad_token_id]) 
            ex_input_ids_0[ex_input_ids_0 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            rank0_print(tokenizer.decode(ex_input_ids_0, skip_special_tokens=False))
            rank0_print("=" * 20)

    rank = int(os.environ.get("RANK", 0)) #0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) #1
    node_id = rank // torch.cuda.device_count() #0

    print(f"Distributed info: rank={rank}, world_size={world_size}, node_id={node_id}")


    eval_dataloader = DataLoader(
        dataset=data_module['eval_dataset'],
        batch_size=1,  # Ensure this is set in args
        collate_fn=data_module['data_collator'],
        shuffle=False,  # For evaluation we usually don't shuffle
    )
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    all_samples = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
        sample = []
        queries, responses, ids  = get_model_answer(batch, model=model, accelerator=accelerator, tokenizer=tokenizer, args=args)
        for query, response, id in zip(queries, responses, ids):
            sample = {
                "id": id.item(),
                "query": tokenizer.decode(query, skip_special_tokens=True),
                "response": tokenizer.decode(response, skip_special_tokens=True),
            }
            all_samples.append(sample)
    if is_initialized():
        gathered_results = custom_gather(all_samples)
    else:
        gathered_results = all_samples

    if torch.distributed.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, args.output_file), "w") as f:
            sorted_results = sorted(gathered_results, key=lambda x: x['id'])
            unique_results = {}
            for item in sorted_results:
                if item['id'] not in unique_results:
                    unique_results[item['id']] = item
            final_results = list(unique_results.values())
            f.write(json.dumps(final_results, indent=4))




if __name__ == "__main__":
    eval()
    time.sleep(20)
