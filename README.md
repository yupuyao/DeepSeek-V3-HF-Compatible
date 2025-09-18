<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


# DeepSeek-V3 HF Compatible

![GitHub Repo](https://img.shields.io/badge/GitHub-DeepSeek--V3--HF--Compatible-24292f?logo=github&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤖%20Hugging%20Face-DeepSeek%20AI-ffc107?logoColor=white)
![Author](https://img.shields.io/badge/Author-Yupu%20Yao-blue)

**An implementation of the official DeepSeek-V3, including a weight loader that enables seamless use of the official Hugging Face weights.**

> Directly load and run DeepSeek-V3 using official Hugging Face weights without conversion.  
> Maintains full compatibility with the original architecture and provides a simple interface for inference and experimentation.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Features](#2-features)
3. [Model Downloads](#3-model-downloads)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [License](#6-license)
7. [Citation](#7-citation)
8. [Contact](#8-contact)

---

## 1. Introduction

DeepSeek-V3 is a state-of-the-art Mixture-of-Experts (MoE) language model with **671B total parameters** and **37B activated parameters per token**. It leverages:

- **Multi-Head Latent Attention (MLA)**  
- **DeepSeekMoE architecture**  
- **Auxiliary-loss-free load balancing**  
- **Multi-Token Prediction (MTP)** training objective  

It is pre-trained on **14.8 trillion high-quality tokens**, and further fine-tuned with Supervised Fine-Tuning and Reinforcement Learning.

This repository ensures that the official Hugging Face weights can be **directly loaded**, enabling seamless local inference and experimentation.

---

## 2. Features

- **Official HF Weight Loader**: Load official Hugging Face DeepSeek-V3 weights without manual conversion.  
- **Seamless Compatibility**: Full alignment with original DeepSeek-V3 architecture.  
- **Inference Ready**: Works with FP8 and BF16 modes for efficient GPU usage.  
- **Multi-Token Prediction Support**: Allows using the MTP module as provided in Hugging Face weights.  
- **Cross-Platform GPU Support**: Works with NVIDIA and AMD GPUs (via frameworks like SGLang or LMDeploy).  

---

## 3. Model Downloads

You can obtain the official weights from Hugging Face:

| Model | Total Params | Activated Params | Context Length | Hugging Face Link |
|-------|-------------|-----------------|----------------|-----------------|
| DeepSeek-V3 | 671B | 37B | 128K | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3) \| [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3) |
| DeepSeek-R1   | 671B | 37B |  128K   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1) \| [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1) |
| DeepSeek-V3.1 | 671B | 37B | 128K | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) \| [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1) |

> Note: This repository enables **direct usage** of these official weights without additional conversion steps.

---

## 4. Installation

```bash
git clone https://github.com/yupuyao/DeepSeek-V3-HF-Compatible.git
cd DeepSeek-V3-HF-Compatible
pip install -r requirements.txt
```

## 5. Usage
Use DeepSeek-R1 weight

```shell
torchrun --nnodes 1 --nproc-per-node 8 generate.py \
    --ckpt-path /path/to/DeepSeek-R1\
    --config configs/config_671B.json \
    --interactive \
    --temperature 0.7 \
    --max-new-tokens 200
```
or with DeepSeek-V3.1

```shell
torchrun --nnodes 1 --nproc-per-node 8 generate.py \
    --ckpt-path /path/to/DeepSeek-V3.1\
    --config configs/config_v3.1.json \
    --interactive \
    --temperature 0.7 \
    --max-new-tokens 200
```
You can try stream output by
```shell
torchrun --nnodes 1 --nproc-per-node 8 generate.py \
    --ckpt-path /path/to/DeepSeek-V3.1\
    --config configs/config_v3.1.json \
    --interactive \
    --temperature 0.7 \
    --max-new-tokens 200 \
    --stream
```
## 6. License
This repository is licensed under the MIT License. Usage of DeepSeek-V3 weights is subject to the official Model License.

## 7. Citation
```
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437}, 
}
```

## 8. Contact
For questions, issues, or contributions, please contact Yupu Yao via GitHub: https://github.com/yupuyao