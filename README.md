# Axolotl Fine-Tuning Quickstart Guide

This repository provides a step-by-step guide to fine-tuning Large Language Models (LLMs) using **Axolotl**, an open-source framework that simplifies post-training workflows.  
The goal is to offer a reproducible setup that allows anyone to experiment with custom instruction-tuning, dataset loading, model training, and inference.

---

## 1. Project Design Requirements (PDR)

### Objective
Enable users to fine-tune any supported LLM using Axolotl from a clean environment with minimal configuration.

### Scope
This guide covers:
- Environment setup on Linux (NVIDIA GPU required)
- Installing Axolotl and dependencies
- Preparing a small example dataset
- Running a LoRA fine-tuning experiment
- Running inference using the fine-tuned model

### Expected Outcomes
- A trained LoRA adapter (or full fine-tuned checkpoint)
- Ability to run inference locally using the trained model
- Understanding of Axolotl's config-based training pipeline

---

## 2. System Requirements

| Component | Minimum |
|---------|---------|
| GPU | NVIDIA (Ampere or newer recommended) |
| VRAM | 12GB+ (or 6-8GB with QLoRA) |
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.11 |
| PyTorch | >= 2.1 with CUDA support |

> If unsure whether CUDA is installed:
```bash
nvidia-smi
````

---

## 3. Linux Environment Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install git wget curl python3 python3-venv python3-pip -y
```

### (Optional but recommended) Create a Virtual Environment

```bash
python3 -m venv axo-env
source axo-env/bin/activate
```

---

## 4. Clone This Repository

```bash
git clone https://github.com/andrewvergel/finetuning-axoloti.git
cd finetuning-axoloti
```

---

## 5. Install Axolotl

```bash
pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip install --no-build-isolation axolotl[flash-attn,deepspeed]
```

Fetch example configs:

```bash
axolotl fetch examples
```

---

## 6. Create a Sample Dataset

Axolotl expects datasets in JSONL format.

Create the file:

```
data/train.jsonl
```

Example (`train.jsonl`):

```json
{"instruction": "What is the capital of France?", "response": "Paris."}
{"instruction": "Explain what a neural network is.", "response": "A neural network is a system of connected artificial neurons."}
{"instruction": "Translate to Spanish: 'Good morning'.", "response": "Buenos días."}
```

---

## 7. Training Configuration

Create a config file:

```
configs/llama3-lora.yml
```

Example LoRA training config:

```yaml
base_model: meta-llama/Llama-3-8B-Instruct
adapter: lora
dataset: data/train.jsonl
dataset_format: instruction
output_dir: ./outputs/llama3-lora
train:
  max_steps: 200
  batch_size: 1
  gradient_accumulation: 8
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
```

---

## 8. Run Training

```bash
axolotl train configs/llama3-lora.yml
```

The trained adapter will be saved in:

```
outputs/llama3-lora/
```

---

## 9. Run Inference with the Trained Model

```bash
axolotl inference configs/llama3-lora.yml
```

Example prompt:

```
> Explain quantum computing in simple terms.
```

---

## 10. Optional: Export LoRA to a Fully Merged Model

```bash
axolotl merge-lora configs/llama3-lora.yml
```

---

## 11. Support & Contributions

* Axolotl Docs: [https://docs.axolotl.ai](https://docs.axolotl.ai)
* Discord Community: [https://discord.gg/HhrNrHJPRb](https://discord.gg/HhrNrHJPRb)

Pull Requests are welcome!

---

## License

This project is licensed under the Apache 2.0 License.