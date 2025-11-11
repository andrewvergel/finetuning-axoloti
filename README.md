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
```

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

### Step 1: Install Build Dependencies

```bash
pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja
```

### Step 2: Install PyTorch (Required before flash-attn)

PyTorch must be installed first because `flash-attn` requires it during installation.

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**To check your CUDA version:**

The easiest way is using `nvidia-smi` (should be available if NVIDIA drivers are installed):
```bash
nvidia-smi
```
Look for "CUDA Version" in the top-right corner of the output.

Alternatively, after installing PyTorch, you can check which CUDA version PyTorch detected:
```bash
python -c "import torch; print(torch.version.cuda)"
```

> **Note:** If you're unsure which CUDA version to use, try CUDA 11.8 first (most compatible). If you encounter compatibility issues, try CUDA 12.1. PyTorch will work with either version as long as your NVIDIA drivers support it.

### Step 3: Install Axolotl

**Important:** Both `flash-attn` and `deepspeed` require the CUDA Toolkit (not just NVIDIA drivers) to compile. You need `nvcc` compiler and `CUDA_HOME` environment variable set.

**Option A: Install Axolotl without Flash Attention (Easiest - Works without CUDA Toolkit)**

If you don't have the CUDA Toolkit installed or encounter compilation issues, install without `flash-attn`:

```bash
pip install --no-build-isolation axolotl
```

> **Note:** This will work fine for training, but `flash-attn` provides significant speed improvements and memory savings. You can add it later if you install the CUDA Toolkit.

**Option B: Install Axolotl with Flash Attention (Requires CUDA Toolkit)**

`flash-attn` requires `CUDA_HOME` to be set. First, check if you have the CUDA Toolkit installed:

```bash
# Check if nvcc is available
nvcc --version
```

If `nvcc` is not found, you'll see a message like:
```
Command 'nvcc' not found, but can be installed with:
apt install nvidia-cuda-toolkit
```

You have two options:
1. Install CUDA Toolkit (see instructions below)
2. Use Option A (install without flash-attn) - **Recommended for getting started quickly**

**If nvcc is available, locate your CUDA installation:**

```bash
# Find CUDA installation path (usually /usr/local/cuda or /usr/local/cuda-XX.X)
ls -la /usr/local/ | grep cuda

# Or check common locations
which nvcc
readlink -f $(which nvcc)  # This will show the CUDA path
```

**Set `CUDA_HOME` environment variable:**

The path depends on how CUDA was installed. Try these in order:

```bash
# Method 1: Automatic detection (works for most installations)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Method 2: If installed via apt (usually one of these)
export CUDA_HOME=/usr/lib/cuda
# OR
export CUDA_HOME=/usr

# Method 3: If installed from NVIDIA repo (version-specific)
export CUDA_HOME=/usr/local/cuda-12.1  # For CUDA 12.1
# OR
export CUDA_HOME=/usr/local/cuda-11.8  # For CUDA 11.8
# OR
export CUDA_HOME=/usr/local/cuda       # If symlink exists
```

**Verify CUDA_HOME is set correctly:**
```bash
echo $CUDA_HOME
ls $CUDA_HOME/bin/nvcc  # Should show the nvcc compiler
```

**Make `CUDA_HOME` persistent across sessions:**

After finding the correct CUDA_HOME path, add it to your shell configuration:

```bash
# Add to ~/.bashrc (replace with your actual CUDA_HOME path)
echo 'export CUDA_HOME=$(dirname $(dirname $(which nvcc)))' >> ~/.bashrc
# OR if automatic detection doesn't work, use the specific path:
# echo 'export CUDA_HOME=/usr/lib/cuda' >> ~/.bashrc

source ~/.bashrc
```

> **Note:** If using automatic detection in `.bashrc`, make sure `nvcc` is in your PATH. Alternatively, use the specific path you verified works.

**Verify requirements before installing flash-attn:**

Before installing, verify that:
1. `nvcc` is accessible and shows CUDA 11.7 or above
2. `CUDA_HOME` is set correctly

```bash
# Check nvcc version (must be 11.7+)
nvcc --version

# Verify CUDA_HOME is set
echo $CUDA_HOME

# Verify nvcc can be found from CUDA_HOME
ls $CUDA_HOME/bin/nvcc

# Test that flash-attn can detect CUDA
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"
```

**Now install Axolotl with Flash Attention:**

If Axolotl is already installed (without flash-attn), you have two options:

**Option 1: Upgrade existing installation:**
```bash
pip install --upgrade --no-build-isolation axolotl[flash-attn]
```

**Option 2: Uninstall and reinstall:**
```bash
pip uninstall axolotl -y
pip install --no-build-isolation axolotl[flash-attn]
```

**If installing fresh:**
```bash
pip install --no-build-isolation axolotl[flash-attn]
```

**Troubleshooting flash-attn installation:**

**Error 1: `RuntimeError: FlashAttention is only supported on CUDA 11.7 and above`**

This means `nvcc` either:
- Is not found in PATH
- Shows a version below 11.7
- Cannot be detected by flash-attn's setup script

**Solutions:**

1. **Verify nvcc is accessible:**
   ```bash
   nvcc -V  # or nvcc --version
   which nvcc
   ```
   If `which nvcc` returns nothing, nvcc is not in PATH.

2. **If nvcc is not in PATH after installation:**
   ```bash
   # Find where nvcc was installed
   dpkg -L nvidia-cuda-toolkit | grep bin/nvcc
   
   # Add to PATH (usually /usr/bin)
   export PATH=/usr/bin:$PATH
   
   # Or if in /usr/lib/cuda/bin
   export PATH=/usr/lib/cuda/bin:$PATH
   
   # Verify it works
   nvcc --version
   ```

3. **Check CUDA version compatibility:**
   ```bash
   nvcc --version
   ```
   The version must be 11.7 or above. If you see 11.5 or lower:
   - Uninstall: `sudo apt remove nvidia-cuda-toolkit`
   - Install a specific version using Method 2 (see below)

4. **Verify CUDA_HOME is set and correct:**
   ```bash
   echo $CUDA_HOME
   ls $CUDA_HOME/bin/nvcc
   ```
   If `CUDA_HOME` is not set or points to wrong location, set it (see instructions above).

5. **Check PyTorch CUDA version:**
   ```bash
   python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
   ```
   This should be compatible with your nvcc version.

6. **If all else fails, install without flash-attn:**
   ```bash
   # Uninstall axolotl first if needed
   pip uninstall axolotl -y
   
   # Install without flash-attn (works fine, just uses more memory)
   pip install --no-build-isolation axolotl
   ```
   You can add flash-attn later once CUDA is properly configured.

**Error 2: `CUDA_HOME environment variable is not set`**

This means the environment variable is not set. Set it using the instructions in the "Set CUDA_HOME" section above, then try again.

**Option C: Install Axolotl with Flash Attention and DeepSpeed (Multi-GPU training)**

Follow the same `CUDA_HOME` setup as Option B, then:

```bash
pip install --no-build-isolation axolotl[flash-attn,deepspeed]
```

> **Note:** DeepSpeed is only needed for multi-GPU training or very large models. For single GPU LoRA fine-tuning, it's optional.

**Installing CUDA Toolkit (if not available):**

**Method 1: Simple apt installation (Recommended)**

The easiest way to install the CUDA Toolkit:

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

> **Note:** The version installed via `apt` may vary. On Ubuntu 22.04, it typically installs CUDA 11.5 or 12.x. **Verify the installed version is 11.7 or above** (required for flash-attn) after installation. If you get an older version, use Method 2 to install a specific version.

**Check available CUDA Toolkit version before installing:**
```bash
# Check what version would be installed
apt-cache policy nvidia-cuda-toolkit

# Or check available versions
apt-cache madison nvidia-cuda-toolkit
```

After installation, verify and check CUDA version:

```bash
nvcc --version
```

**Important:** `flash-attn` requires CUDA 11.7 or above. Check the output of `nvcc --version` to ensure you have a compatible version. If you see CUDA 11.6 or lower, you'll need to install a newer version.

**If `nvcc` is not in PATH after installation:**

After installing via `apt`, you may need to add it to your PATH:

```bash
# Check if nvcc is in standard paths
which nvcc

# If not found, add to PATH (usually /usr/bin or /usr/local/cuda/bin)
export PATH=/usr/bin:$PATH
# OR if installed in /usr/lib/cuda
export PATH=/usr/lib/cuda/bin:$PATH

# Verify nvcc is now accessible
nvcc --version
```

**Make PATH persistent:**
```bash
echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Find CUDA installation path after apt install:**

When installed via `apt`, CUDA is typically in `/usr/lib/cuda` or `/usr`. Find it with:

```bash
# Method 1: Check where nvcc is located
which nvcc
readlink -f $(which nvcc)  # Shows full path

# Method 2: Check common apt-installed locations
ls -la /usr/lib/cuda 2>/dev/null
ls -la /usr/local/cuda 2>/dev/null

# Method 3: Use dpkg to find installation
dpkg -L nvidia-cuda-toolkit | grep bin/nvcc
```

**Set CUDA_HOME for apt-installed CUDA:**

```bash
# Usually one of these paths works:
export CUDA_HOME=/usr/lib/cuda
# OR
export CUDA_HOME=/usr

# OR use automatic detection:
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Verify it's set correctly
echo $CUDA_HOME
ls $CUDA_HOME/bin/nvcc
```

**Method 2: Install specific CUDA version (Alternative)**

If you need a specific CUDA version (e.g., 12.1 or 11.8):

```bash
# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1  # Or cuda-toolkit-11-8 for CUDA 11.8

# Add to PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1
```

> **Important Notes:**
> - **NVIDIA Drivers vs CUDA Toolkit:** Having NVIDIA drivers (checked with `nvidia-smi`) allows PyTorch to use the GPU, but doesn't provide `nvcc` compiler needed for `flash-attn`/`deepspeed`.
> - **Training without flash-attn:** Option A will work perfectly fine for training. You'll just use more memory and train slightly slower.
> - **Recommended approach:** Start with Option A to get everything working, then add `flash-attn` later if needed for better performance.

### Step 4: Verify Axolotl Installation

Check that Axolotl was installed correctly:

```bash
axolotl --version
```

Alternatively, you can check the installed version using pip:

```bash
pip show axolotl
```

You should see the Axolotl version number and installation details. If the `axolotl` command is not found, verify that your Python environment is activated and Axolotl is installed correctly.

### Step 5: Fetch Example Configs

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