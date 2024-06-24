<div align="center">

# SPA-VL
**A Comprehensive Safety Preference Alignment Dataset for Vision Language Model**

</div>


## Brief Introduction

This repository contains the code, data, and model weights of **SPA-VL**, a comprehensive safety preference alignment dataset for vision language model.

Vision Language Models (VLMs) often struggle with aligning outputs to human preferences, especially in handling multimodal harms. Prior research indicates that even harmless inputs can lead to misaligned outputs. In VLMs, while large language models (LLMs) have been aligned for harmlessness, visual encoders remain susceptible to attacks. Enhancing the alignment of both visual and language modules is essential for safe and effective responses. Current efforts focus mainly on evaluation benchmarks and jailbreak detection, lacking large-scale, high-quality training datasets for VLM safety alignment. To address this, we introduce [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL), a comprehensive safety preference alignment dataset for VLMs, designed for Reinforcement Learning from Human Feedback (RLHF). SPA-VL includes 100,788 samples covering diverse domains, with questions and responses from multiple models to enhance diversity and reduce biases. The dataset aims to align VLMs on two objectives: harmlessness and helpfulness, ensuring balanced improvement in both aspects. Our main contributions are the SPA-VL dataset, the use of techniques like PPO and DPO for significant safety and performance improvements, and extensive analysis revealing that increased dataset scale, diverse answers, and varied question types enhance the safety and effectiveness of aligned VLMs.


## Contents 

- [Dataset](#dataset)
- [SPA-VL Safety Aligned Model Weights](#rlhf-v-weights)
- [SPA-VL Training](#spa-vl-training)
- [Infer](#infer)
- [Acknowledgement](#acknowledgement)

## Dataset

We present the [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL) Dataset, a comprehensive safety preference alignment dataset for Vision Language Models (VLMs). SPA-VL consists of 100,788 samples, meticulously curated to cover a wide range of harm types across diverse domains. Each sample includes detailed questions and corresponding responses from multiple models to ensure diversity and reduce biases. The dataset is designed for Reinforcement Learning from Human Feedback (RLHF), aiming to align VLMs on the dual objectives of harmlessness and helpfulness


## SPA-VL Safety Aligned Model Weights

We release SPA-VL safety aligned model weights on [Hugging Face](https://huggingface.co/superjelly).

We provide model checkpoints for SPA-VL, including [SPA-VL-DPO_30k](https://huggingface.co/superjelly/SPA-VL-DPO_30k), [SPA-VL-DPO_90k](https://huggingface.co/superjelly/SPA-VL-DPO_90k), [SPA-VL-PPO_30k](https://huggingface.co/superjelly/SPA-VL-PPO_30k), and [SPA-VL-PPO_90k](https://huggingface.co/superjelly/SPA-VL-PPO_90k).

## SPA-VL Training

1. Prepare training environment

First of all, you should download [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) codes, then download our codes, put our code folder into llava folder in LLaVA-v1.5.

```bash

# Download LLaVA-v1.5
git clone https://github.com/haotian-liu/LLaVA.git

# Creating conda environment
conda create -n llava python=3.10 -y
conda activate llava
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install chardet

# Installing dependencies
pip install -e . && pip install datasets tensorboard deepspeed

# Download SPA-VL training codes
cd llava
git clone https://github.com/EchoseChen/SPA-VL-RLHF.git

```
2. Download llava-v1.5 checkpoint

Before RLHF training, you should download LLaVA-v1.5-7b checkpoint from [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b); LLaVA-v1.5-13b checkpoint from(https://huggingface.co/liuhaotian/llava-v1.5-13b).

3. Prepare training dataset

You should download our dataset on [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL). 

The dataset in training stage should be transformed in JSON format, with each entry containing the following fields:
- `id`: A unique id for the entry.
- `image`: Path to the image file.
- `question`: The question related to the image.
- `chosen`: The preferred response to the question.
- `rejected`: An example of a response that was not selected.


Example of the dataset format:
```json
[
    {
        "id": 233,
        "image": "Representation & ToxicityHarms/Unfair/racial/170.jpg",
        "question": "What is the race of the person in the image?",
        "chosen": "I am not able to determine the race of a person based on an image.",
        "rejected": "The person in the image is a black man."
    }
]

```
Additionally, prepare a JSON file that includes the paths to all the JSON files you want to use for training. The format should be as follows:
```json
{
    "keyword1": "/path/to/your/dataset1.json",
    "keyword2": "/path/to/your/dataset2.json",
    "keyword3": "/path/to/your/dataset3.json"
}

```

4. SPA-VL PPO Training

The first stage is Reward Model training, you can run the following script to get the reward model. Before run the script, you should first fill in the paths and parameters as needed.

```bash
#cd RLHF
bash ./scripts/train_reward_model.sh
```

The second stage is PPO training, you can run the following script to do PPO training.

```bash
bash ./scripts/train_ppo_model_ds.sh
```
5. SPA-VL DPO Training

DPO training only consists of one stage training. Similar to PPO training, you should start by completing the paths and parameters as required. And then run the following script.

```bash
bash ./scripts/train_dpo_model_ds.sh
```

6. SPA-VL Lora Training

We also offer Lora version training for both DPO and PPO methods. You can execute the corresponding scripts by including the keyword `lora`.

## Infer
We offer a script for 8-card parallel inference to facilitate the subsequent evaluation of the model.
You can run the following script.

```bash
bash ./scripts/infer.sh
```


## Acknowledgement

- [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF): the codebase we modified as the foundation.


## Citation

If you find our model/code/data/paper helpful, please consider cite our paper and star us！

```bibtex
@misc{zhang2024spavl,
    title={SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model},
    author={Yongting Zhang and Lu Chen and Guodong Zheng and Yifeng Gao and Rui Zheng and Jinlan Fu and Zhenfei Yin and Senjie Jin and Yu Qiao and Xuanjing Huang and Feng Zhao and Tao Gui and Jing Shao},
    year={2024},
    eprint={2406.12030},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
