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
{"instruction": "How do I restart the analytics microservice safely?", "response": "1) Check for active jobs in the queue.\n2) Pause message ingestion.\n3) Run 'systemctl restart analytics-ms'.\n4) Monitor logs for 2 minutes.\n5) Resume message ingestion."}
{"instruction": "Summarize what a LoRA adapter is.", "response": "A LoRA adapter is a small set of additional trainable parameters that enable fine-tuning a large model efficiently without modifying the original model weights."}
{"instruction": "How do I request access to the staging environment?", "response": "1) Open a ticket under 'Access Request'.\n2) Specify 'Staging SSH + Database read-only'.\n3) Get approval from Engineering Manager.\n4) DevOps will grant access within 24 hours."}
{"instruction": "Explain bfloat16 in simple terms.", "response": "bfloat16 is a numerical format that uses fewer bits than float32, which reduces memory usage and speeds up training, while still keeping enough precision for deep learning."}
{"instruction": "What is the purpose of early stopping during training?", "response": "Early stopping prevents overfitting by halting training when validation performance stops improving."}
{"instruction": "How do I report a production incident?", "response": "1) Go to the Incident Management portal.\n2) Create a new incident ticket.\n3) Describe the problem, impact, and affected services.\n4) Assign priority based on severity.\n5) Notify on-call team via Slack #ops-alerts."}
{"instruction": "Explain what gradient checkpointing does.", "response": "Gradient checkpointing trades computation for memory by recomputing intermediate activations during backpropagation, allowing training of larger models on limited GPU memory."}
{"instruction": "How do I rotate API tokens securely?", "response": "1) Generate a new token in the IAM portal.\n2) Update the service configuration.\n3) Redeploy the service.\n4) Verify the service is healthy.\n5) Revoke the old token."}
{"instruction": "Describe the role of the 'validation split' in training.", "response": "The validation split allocates a portion of the dataset to evaluate model performance during training without influencing the learning process."}
{"instruction": "How do I request a database schema change?", "response": "1) Create a DB change proposal document.\n2) Submit for review to the Data Engineering team.\n3) Schedule a migration window.\n4) Apply changes via Flyway.\n5) Run post-migration tests."}
{"instruction": "What is CUDA used for in LLM training?", "response": "CUDA allows deep learning frameworks to run computations on NVIDIA GPUs, dramatically accelerating tensor operations required for training."}
{"instruction": "How do I document a new internal procedure?", "response": "1) Create a new page in the Knowledge Base.\n2) Add step-by-step instructions.\n3) Include screenshots or command examples.\n4) Request peer review.\n5) Publish and share with the team."}
{"instruction": "Explain why dataset diversity matters in fine-tuning.", "response": "Dataset diversity improves the model’s ability to generalize, preventing it from overfitting to narrow patterns and producing more robust responses."}
{"instruction": "How do I test that a deployment was successful?", "response": "1) Verify service health metrics.\n2) Run automated smoke tests.\n3) Validate logs show no new errors.\n4) Confirm that request latency remains stable.\n5) Notify stakeholders deployment is complete."}
{"instruction": "What is 'gradient accumulation'?", "response": "Gradient accumulation simulates a larger batch size by summing gradients over multiple forward passes before updating model weights."}
{"instruction": "How do I rollback a deployment?", "response": "1) Identify last stable release.\n2) Redeploy using previous container image.\n3) Clear related caches.\n4) Validate logs and metrics.\n5) Announce rollback completion in #release-status."}
{"instruction": "Explain the concept of model hallucination.", "response": "Hallucination occurs when a model produces confident but factually incorrect or fabricated information due to gaps or ambiguity in the input or training data."}
{"instruction": "How do I clean old logs in a Linux server?", "response": "1) Run `find /var/log -type f -mtime +30 -exec gzip {} \\;`.\n2) Move compressed logs to /backup/logs.\n3) Update logrotate configuration.\n4) Document cleanup."}
{"instruction": "Why use LoRA instead of full fine-tuning?", "response": "LoRA is significantly more memory-efficient and faster because it trains only small adapter layers instead of updating all model parameters."}
{"instruction": "How do I verify the fine-tuned model is performing correctly?", "response": "1) Run inference using known test prompts.\n2) Compare outputs with expected responses.\n3) Assess consistency, correctness, and tone.\n4) Log evaluation notes.\n5) Approve for deployment if satisfactory."}
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

## 9. Validate Model Performance with Inference

### Purpose: Verify the Model Learned from Your Dataset

**The goal of inference is NOT to test general knowledge** (like "Explain quantum computing..."), because a pretrained model already knows that. Instead, **validate that the model learned your specific dataset and internal style**:

- Internal procedures and workflows
- Operational steps and processes
- Real company-specific flows
- Professional, clear, and concise tone

### Run Inference

```bash
axolotl inference configs/llama3-lora.yml
```

### Example Validation Prompts

Use prompts that test whether the model learned from your dataset:

**Prompt:**
```
How do I restart the analytics microservice safely?
```

**Expected Response (if training succeeded):**
```
1) Check for active jobs in the queue.
2) Pause message ingestion.
3) Run 'systemctl restart analytics-ms'.
4) Monitor logs for 2 minutes.
5) Resume message ingestion.
```

**If the model responds differently** → the fine-tuning may have failed or needs more training.

### More Validation Examples

| Prompt | Expected Response Indicator |
|--------|----------------------------|
| `How do I request access to the staging environment?` | Should provide clear steps + corporate tone |
| `What is 'gradient accumulation'?` | Should respond with correct technical definition |
| `How do I rotate API tokens securely?` | Should give specific administrative steps, not invented ones |
| `Why use LoRA instead of full fine-tuning?` | Should mention memory efficiency and parameter efficiency |
| `How do I report a production incident?` | Should follow the exact procedure from your dataset |

### Real Inference Session Example

**Prompt:**
```
How do I report a production incident?
```

**Expected Response:**
```
1) Go to the Incident Management portal.
2) Create a new incident ticket.
3) Describe the problem, impact, and affected services.
4) Assign priority based on severity.
5) Notify on-call team via Slack #ops-alerts.
```

**If the model:**
- Changes the steps
- Invents tools that don't exist
- Gives generic responses

→ **Quality issues detected: may need more training examples or hyperparameter tuning.**

### Validation Criteria

| Criterion | What to Validate | Success Signal |
|-----------|------------------|----------------|
| **Accuracy** | Concrete and correct steps | ✅ Uses real process and service names |
| **Tone** | Professional, direct | ✅ Matches dataset style |
| **No Hallucination** | Doesn't invent tools/commands | ✅ Consistent responses |
| **Consistency** | Same terminology | ✅ Doesn't use unusual synonyms |

### What to Observe

- ✅ **Correctness**: The model follows the exact procedures from your dataset
- ✅ **Style**: Responses match the professional, step-by-step format
- ✅ **Consistency**: Repeated prompts produce similar responses
- ✅ **No Fabrication**: The model doesn't invent tools, portals, or commands that weren't in the training data

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