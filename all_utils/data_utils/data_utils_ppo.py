# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Callable, Dict, Optional, List, Sequence, Any

import logging
import pandas as pd
import json

import torch
from torch.utils.data import Dataset

import transformers

import all_utils.data_utils.common_utils as utils

from PIL import Image
import copy
import os
import tqdm
import torch.nn.functional as F
from llava import conversation as conversation_lib
import all_utils.data_utils.common_utils as utils
from llava.mm_utils import tokenizer_image_token
from all_utils.data_utils.common_utils import load_data

logger = logging.getLogger(__name__)


class QueryResponseDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        list_dict_data: List[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        query_len: int,
        data_args: Optional[Dict] = None,
    ):
        self.data_args = data_args
        self.tokenizer = tokenizer ###
        super(QueryResponseDataset, self).__init__()

        queries = [
            preprocess_ppo_model(
                source,
                tokenizer,
            )["input_ids"]
            for source in tqdm.tqdm(list_dict_data)
        ]

        queries = [
            torch.as_tensor(query, dtype=torch.long).view(-1)[:-2] for query in queries
        ] 
        filtered_queries = []
        filtered_list_dict_data = []

        for query, data in zip(queries, list_dict_data):
            if len(query) <= query_len:
                filtered_queries.append(query)
                filtered_list_dict_data.append(data)

        max_query_len = max(len(query) for query in filtered_queries)
        logger.warning(f"Max query length: {max_query_len}")

        logger.warning(
            f"Filtered out {len(queries) - len(filtered_queries)} instances out of {len(queries)} that "
            f"exceed length limit. These examples are not used for training, but will still be used in evaluation. "
        )

        queries = torch.stack(
            [
                utils.left_pad(
                    query, target_size=(max_query_len,), value=tokenizer.pad_token_id #0
                )
                for query in filtered_queries
            ]
        )

        self.queries = queries

        # Auxiliary data.
        self.list_dict_data = filtered_list_dict_data
        assert len(self.queries) == len(self.list_dict_data), "Length of queries, query_attn_masks, and list_dict_data must be the same."

    def __getitem__(self, idx):
        return_dict = dict(
            queries=self.queries[idx],
        )
        if "image" in self.list_dict_data[idx]:
            image_file = self.list_dict_data[idx]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            except:
                raise ValueError(f"Error loading image {image_file} for index {idx}")
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image = image.to(dtype=torch.bfloat16)
        if "image" in self.list_dict_data[idx]:
            return_dict["images"] = image #torch.Size([3, 336, 336])
        else:
            crop_size = self.data_args.image_processor.crop_size
            return_dict["images"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return_dict["ids"] = torch.tensor(
            self.list_dict_data[idx]["id"], dtype=torch.long
        ) #tensor(157875)
        return return_dict

    def __len__(self):
        return len(self.queries)


@dataclasses.dataclass
class DataCollatorForQueryResponseDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([instance[key] for instance in instances])
            for key in instances[0].keys()
        }


def make_rl_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    if data_args.config_train_path.endswith("json"): 
        list_data_dict = load_data(data_args.keywords, data_args.config_train_path)
    else:
        raise ValueError(
            f"Unsupported dataset_path: {data_args.dataset_path}."
            "Only json datasets are supported."
        )
    train_dataset = QueryResponseDataset(
        list_dict_data=list_data_dict,
        tokenizer=tokenizer,
        query_len=training_args.query_len, 
        data_args=data_args,
    ) 
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForQueryResponseDataset(),
    )
def remove_pad_and_left_pad(completions, pad_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    padded_length = len(clean_completions[0])
    for idx, completion in enumerate(clean_completions):
        completion = [token for token in completion if token != pad_token_id]

        if len(completion) < padded_length:
            completion = [pad_token_id] * (padded_length - len(completion)) + completion

        clean_completions[idx] = completion

    clean_completions = torch.tensor(
        clean_completions, dtype=torch.long, device=completions.device
    )
    return clean_completions

def pad_sequences(lst_seq: List[List[int]], pad_value, pad_left=True, pad_to: int=None) -> List[List[int]]:
        maxlen = max(len(seq) for seq in lst_seq) if pad_to is None else pad_to
        if pad_left:
            padded_seq = [[pad_value] * (maxlen - len(seq)) + seq for seq in lst_seq]
        else:
            padded_seq = [seq + [pad_value] * (maxlen - len(seq)) for seq in lst_seq]
        return padded_seq

def rolloutify(rollout_samples: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    rollouts = {
        'queries': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['query'], dtype=torch.long) for sample in rollout_samples], 
                                batch_first=True, padding_value=tokenizer.pad_token_id),
        'responses': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['response'], dtype=torch.long) for sample in rollout_samples], 
                                  batch_first=True, padding_value=tokenizer.pad_token_id),
        'sequences': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['sequence'], dtype=torch.long) for sample in rollout_samples], 
                                  batch_first=True, padding_value=tokenizer.pad_token_id),
        'values': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['values'], dtype=torch.float) for sample in rollout_samples], 
                               batch_first=True, padding_value=0.),
        'logprobs': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['logprobs'], dtype=torch.float) for sample in rollout_samples], 
                                 batch_first=True, padding_value=0.),
        'advantages': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['advantages'], dtype=torch.float) for sample in rollout_samples], 
                                   batch_first=True, padding_value=0.),
        'returns': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['returns'], dtype=torch.float) for sample in rollout_samples], 
                                batch_first=True, padding_value=0.),
        'loss_mask': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['loss_mask'], dtype=torch.float) for sample in rollout_samples], 
                                  batch_first=True, padding_value=0.),
    }
    return rollouts

def preprocess_ppo_model(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()

    if isinstance(source["question"], list):
        roles = {"Human": conv.roles[0], "Assistant": conv.roles[1]}
        for j, sentence in enumerate(source["question"]):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
    elif isinstance(source["question"], str):
        if "image" in source:
            conv.append_message(conv.roles[0], '<image>\n' + source["question"])
        else:
            conv.append_message(conv.roles[0], source["question"])
    conv.append_message(conv.roles[1], "\n")
    # Apply prompt templates
    conversations = []
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    if "image" in source:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt") 
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    return dict(
        input_ids=input_ids, 
    )
