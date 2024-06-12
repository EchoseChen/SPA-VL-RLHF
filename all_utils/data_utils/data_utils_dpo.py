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
import os
import tqdm
import torch.nn.functional as F

import all_utils.data_utils.common_utils as utils
from all_utils.data_utils.common_utils import load_data
from all_utils.data_utils.data_utils_ppo import preprocess_ppo_model

logger = logging.getLogger(__name__)


class QueryChosenRejectedResponseDataset(Dataset):
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
        super(QueryChosenRejectedResponseDataset, self).__init__()

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
        filtered_chosen_responses = []
        filtered_rejected_responses = []

        for query, data in zip(queries, list_dict_data):
            if len(query) <= query_len:
                chosen_response = self.tokenizer(data["chosen"], return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0][1:]
                rejected_response = self.tokenizer(data["rejected"], return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0][1:]
                if (len(query) + len(chosen_response) < (self.tokenizer.model_max_length - 576)) and (len(query) + len(rejected_response) < (self.tokenizer.model_max_length -576)):
                    filtered_queries.append(query)
                    filtered_list_dict_data.append(data)
                    filtered_chosen_responses.append(chosen_response)
                    filtered_rejected_responses.append(rejected_response)
        max_query_len = max(len(query) for query in filtered_queries) #109
        max_chosen_response_len = max(len(response) for response in filtered_chosen_responses)
        max_rejected_response_len = max(len(response) for response in filtered_rejected_responses)
        max_response_len = max(max_chosen_response_len, max_rejected_response_len)
        logger.warning(f"Max query length: {max_query_len}")
        logger.warning(f"Max response length: {max_response_len}")

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
        chosen_responses = torch.stack(
            [
                utils.left_pad(
                    response, target_size=(max_response_len,), value=tokenizer.pad_token_id #0
                )
                for response in filtered_chosen_responses
            ]
        )
        rejected_responses = torch.stack(
            [
                utils.left_pad(
                    response, target_size=(max_response_len,), value=tokenizer.pad_token_id #0
                )
                for response in filtered_rejected_responses
            ]
        ) 

        self.queries = queries
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses

        # Auxiliary data.
        self.list_dict_data = filtered_list_dict_data
        assert len(self.queries) == len(self.list_dict_data) == len(self.chosen_responses) == len(self.rejected_responses), "Length of queries, query_attn_masks, and list_dict_data must be the same."

    def __getitem__(self, idx):
        return_dict = dict(queries=self.queries[idx])
        return_dict['chosen_responses'] = self.chosen_responses[idx]
        return_dict['rejected_responses'] = self.rejected_responses[idx]

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
            return_dict["images"] = image 
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


def make_dpo_data_module(
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
    train_dataset = QueryChosenRejectedResponseDataset(
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



