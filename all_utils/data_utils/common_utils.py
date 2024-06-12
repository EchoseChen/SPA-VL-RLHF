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

import argparse
import glob
import json
import os
import random
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    Mapping,
    Any,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token


Numeric = Union[int, float]


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):  # collections.Mapping fails in py3.10.
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def merge_dict(dicts: Sequence[dict], merge_fn: Callable = lambda *args: args) -> dict:
    """Merge a sequence of dicts (with the same set of keys) into a single dict."""
    if len(dicts) == 0:
        return dict()
    return {key: merge_fn([dict_[key] for dict_ in dicts]) for key in dicts[0].keys()}


def prepare_inputs(
    data: Union[torch.Tensor, Any], device: Union[str, int, torch.device]
) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)(
            {k: prepare_inputs(v, device) for k, v in data.items()}
        )  # noqa
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)  # This can break with deepspeed.
    return data

def pad(
    inputs: torch.Tensor,
    target_size: Union[torch.Size, Sequence[int]],
    value=0.0,
    left=True,
):
    current_size = inputs.size()
    diffs = tuple(ti - ci for ti, ci in zip_(target_size, current_size))
    pad_params = []
    for diff in diffs:
        pad_params = ([diff, 0] if left else [0, diff]) + pad_params
    res = F.pad(inputs, pad=pad_params, value=value)
    return res


def left_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=True)


def right_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=False)


def manual_seed(args_or_seed: Union[int, argparse.Namespace], fix_cudnn=False):
    if hasattr(args_or_seed, "seed"):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa

class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

def logprobs_from_logits(logits, labels):
    """Compute log softmax values from logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

def compute_logprobs(logits, sequences, responses_vec, device):
    bsz = len(responses_vec)
    logprobs_list = []
    for i in range(bsz):
        response_length = len(responses_vec[i])
        logit = logits[i][-response_length:]
        sequence = sequences[i][-response_length:]
        logprob = logprobs_from_logits(logit, sequence)
        logprobs_list.append(logprob)
    logprobs = torch.nn.utils.rnn.pad_sequence(logprobs_list, batch_first=True, padding_value=0.).to(device)
    return logprobs

def load_data(keywords: List[str], config_path: str) -> List[dict]:
    # Load the configuration file with keyword to JSON file mappings
    with open(config_path, "r") as file:
        config = json.load(file)
    
    all_data = []

    # Load and combine data from JSON files specified by the keywords
    for keyword in keywords:
        if keyword in config:
            path = config[keyword]
            with open(path, "r") as file:
                data = json.load(file)
                all_data.extend(data)
        else:
            print(f"No data file associated with the keyword '{keyword}'")
    
    return all_data