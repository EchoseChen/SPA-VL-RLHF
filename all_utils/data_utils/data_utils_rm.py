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

from dataclasses import dataclass
import json
import logging
from typing import Callable, Optional, Dict, Sequence, List, Tuple

import random

import einops
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from llava import conversation as conversation_lib
from PIL import Image
import copy
import os
from all_utils.data_utils.common_utils import load_data
from llava.mm_utils import tokenizer_image_token


logger = logging.getLogger(__name__)


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    ) 
    padded_sequence = padded_sequence.flip(int(batch_first)) 
    return padded_sequence


@dataclass
class DataCollatorForRewardModelingDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        input_ids = [seq for instance in instances for seq in instance[key]] 
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=len(instances[0][key]),
        )
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        choice = torch.stack([instance["choice"] for instance in instances])
        input_ids = self._left_pad_helper(instances, "input_ids")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            choice=choice,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def preprocess_for_reward_modeling_cl(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor: Optional[Callable] = None,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    use_data_frame: bool = False,
    mask_target: bool = False,
) -> Dict[str, torch.Tensor]:
    if use_data_frame:
        df = pd.DataFrame(source)
        if df_postprocessor is not None:
            df = df_postprocessor(df)
            source = df


    def _get_text(example: dict, output_key: str):
        sources = {}
        sources["question"] = example["question"]
        sources["answer"] = example[output_key]

        if "image" in example:
            has_image = True
        else:
            has_image = False
        return preprocess_reward_model(
            sources,
            tokenizer,
            has_image=has_image,
            mask_target=mask_target,
            query_len=query_len,
            response_len=response_len,
        )
    text_chosen = _get_text(source, 'chosen')
    text_rejected = _get_text(source, 'rejected')

    merged_valid = [val_1 and val_2 for val_1, val_2 in zip(text_chosen["validity"], text_rejected["validity"])]

    # "size" (bsz, 2, seq_len)
    input_ids = [text_chosen["input_ids"].squeeze(0), text_rejected["input_ids"].squeeze(0)]
    labels = [text_chosen["labels"].squeeze(0), text_rejected["labels"].squeeze(0)]
    choice = torch.tensor(0)
    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        merged_valid=merged_valid,
        choice=choice,
    )
    return packaged_data



class RewardModelingDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        query_len: Optional[int] = None,
        response_len: Optional[int] = None,
        use_data_frame: bool = True,
        data_args: Optional[Dict] = None,
        config_path: Optional[str] = None,
    ):
        super(RewardModelingDataset, self).__init__()
        self.tokenizer = tokenizer
        list_data_dict = load_data(data_args.keywords, config_path)
        filtered_list_dict_data = []
        for data in list_data_dict:
            query = self.tokenizer(data["question"], return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
            chosen_response = self.tokenizer(data["chosen"], return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0][1:]
            rejected_response = self.tokenizer(data["rejected"], return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0][1:]
            if (len(query) + len(chosen_response) < (self.tokenizer.model_max_length - 576)) and (len(query) + len(rejected_response) < (self.tokenizer.model_max_length -576)):
                filtered_list_dict_data.append(data)
        logger.warning(
            f"Filtered out {len(list_data_dict) - len(filtered_list_dict_data)} instances out of length that "
            f"exceed length limit. These examples are not used for training. "
        )
        
        self.list_data_dict = filtered_list_dict_data
        self.data_args = data_args
        self.query_len = query_len
        self.response_len = response_len
        self.use_data_frame = use_data_frame

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        if "image" in source:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            except:
                raise ValueError(f"Error loading image {image_file} for index {i}")
            if self.data_args.image_aspect_ratio == "pad":
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]

        data_dict = preprocess_for_reward_modeling_cl(
            source,
            tokenizer=self.tokenizer,
            mask_target=False,
            query_len=self.query_len,
            response_len=self.response_len,
            use_data_frame=self.use_data_frame,
        )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image.to(torch.bfloat16)
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        return data_dict

def make_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    if data_args.config_train_path.endswith("json") and data_args.config_test_path.endswith("json"):
        use_data_frame = False
    else:
        raise ValueError(
            f"Unsupported dataset_path: {data_args.dataset_path}."
            "Only json datasets are supported."
        )

    train_dataset = RewardModelingDataset(
        tokenizer=tokenizer,
        query_len=training_args.query_len,
        response_len=training_args.response_len, 
        use_data_frame=use_data_frame,
        data_args=data_args,
        config_path=data_args.config_train_path,
    )

    eval_dataset = RewardModelingDataset(
        tokenizer=tokenizer,
        query_len=training_args.query_len, 
        response_len=training_args.response_len,
        use_data_frame=use_data_frame,
        data_args=data_args,
        config_path=data_args.config_test_path,
    )

    data_collator = DataCollatorForRewardModelingDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

def preprocess_reward_model(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    mask_target: bool = True,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    
    if isinstance(source["question"], list):
        roles = {"Human": conv.roles[0], "Assistant": conv.roles[1]}
        for j, sentence in enumerate(source["question"]):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], source["answer"])

    elif isinstance(source["question"], str):
        conv.append_message(conv.roles[0], '<image>\n' + source["question"])
        conv.append_message(conv.roles[1], source["answer"])

    # Apply prompt templates
    
    conversations = []
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
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
    targets = input_ids.clone() 
    validity = [True] * len(input_ids) 
    validity[0] = (validity[0]and len(input_ids[0])< tokenizer.model_max_length)

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    if mask_target:
        raise ValueError(f"Wrong: {conversations}")


    return dict(
        input_ids=input_ids, 
        labels=targets,
        validity=validity,
    )

