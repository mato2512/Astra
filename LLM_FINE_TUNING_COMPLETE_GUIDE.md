# Complete LLM Fine-Tuning Guide

## Table of Contents
1. [Understanding LLM Fine-Tuning](#understanding-llm-fine-tuning)
2. [Ollama Fine-Tuning](#ollama-fine-tuning)
3. [llama.cpp Fine-Tuning](#llamacpp-fine-tuning)
4. [Azure AI Foundry](#azure-ai-foundry)
5. [Digital Ocean GPU Setup](#digital-ocean-gpu-setup)
6. [Data Preparation](#data-preparation)
7. [Advanced Techniques](#advanced-techniques)
8. [Production Deployment](#production-deployment)

---

## Understanding LLM Fine-Tuning

### What is Fine-Tuning?

Fine-tuning adapts a pre-trained language model to your specific use case by training it on custom data. Instead of training from scratch (which requires billions of tokens and months of GPU time), you start with a model like Llama 3 and refine it.

### Types of Fine-Tuning

#### 1. **Full Fine-Tuning**
- Updates ALL model parameters
- Requires massive GPU memory (80GB+ for 7B models)
- Best accuracy but most expensive
- **Cost:** $500-$2000 per training run

#### 2. **LoRA (Low-Rank Adaptation)**
- Updates small adapter layers (0.1-1% of parameters)
- Requires 16-24GB GPU memory
- 90-95% of full fine-tuning accuracy
- **Cost:** $50-$200 per training run
- **Recommended for most use cases**

#### 3. **QLoRA (Quantized LoRA)**
- LoRA + 4-bit quantization
- Requires 8-12GB GPU memory (consumer GPUs!)
- 85-90% of full fine-tuning accuracy
- **Cost:** $20-$100 per training run
- **Best for budget-conscious projects**

#### 4. **Instruction Tuning**
- Teaches model to follow instructions
- Format: `instruction → response`
- Used for chat/assistant models

#### 5. **RLHF (Reinforcement Learning from Human Feedback)**
- Fine-tunes based on human preferences
- Used by ChatGPT, Claude
- Complex and expensive

---

## Ollama Fine-Tuning

### Method 1: Create Custom Modelfile (No Training)

**Best for:** Prompt engineering, system message customization

```bash
# 1. Create a Modelfile
cat > Modelfile << EOF
FROM llama3:8b

SYSTEM """You are Astra AI, an expert assistant specialized in software development.
You provide accurate, concise code examples with detailed explanations.
You always follow best practices and modern conventions."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
EOF

# 2. Create the model
ollama create astra-dev -f Modelfile

# 3. Test it
ollama run astra-dev "Write a Python function to calculate fibonacci numbers"
```

### Method 2: Import GGUF Model

**Best for:** Using models from HuggingFace

```bash
# 1. Download GGUF model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# 2. Create Modelfile
cat > Modelfile << EOF
FROM ./mistral-7b-instruct-v0.2.Q4_K_M.gguf

TEMPLATE """<s>[INST] {{ .Prompt }} [/INST]"""

PARAMETER temperature 0.7
EOF

# 3. Import to Ollama
ollama create mistral-custom -f Modelfile

# 4. Use it
ollama run mistral-custom
```

### Method 3: Fine-Tune with Unsloth (Recommended)

**Best for:** Actual model fine-tuning with your data

#### Step 1: Install Dependencies
```bash
# Create virtual environment
python -m venv ollama-finetune
source ollama-finetune/bin/activate  # Windows: ollama-finetune\Scripts\activate

# Install Unsloth (fastest fine-tuning library)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

#### Step 2: Prepare Training Data
```python
# prepare_data.py
import json

# Your training data in instruction format
training_data = [
    {
        "instruction": "What is machine learning?",
        "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    },
    {
        "instruction": "Explain neural networks",
        "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process and transform input data."
    },
    # Add 100-10,000 examples
]

# Convert to format for training
formatted_data = []
for item in training_data:
    formatted_data.append({
        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{item['output']}<|eot_id|>"
    })

# Save as JSONL
with open("training_data.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")
```

#### Step 3: Fine-Tune with Unsloth
```python
# finetune_ollama.py
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # 4-bit quantized
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# 3. Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# 4. Training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# 5. Train!
trainer.train()

# 6. Save model
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

#### Step 4: Export to GGUF for Ollama
```python
# export_gguf.py
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Export to GGUF with different quantizations
model.save_pretrained_gguf("astra-model", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_gguf("astra-model", tokenizer, quantization_method="q5_k_m")
model.save_pretrained_gguf("astra-model", tokenizer, quantization_method="q8_0")

print("✅ Models exported to:")
print("  - astra-model-q4_k_m.gguf")
print("  - astra-model-q5_k_m.gguf")
print("  - astra-model-q8_0.gguf")
```

#### Step 5: Import to Ollama
```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./astra-model-q4_k_m.gguf

TEMPLATE """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
EOF

# Import to Ollama
ollama create astra-finetuned -f Modelfile

# Test
ollama run astra-finetuned "Explain quantum computing"
```

---

## llama.cpp Fine-Tuning

### Why llama.cpp?

- **Fastest CPU inference** (Apple Silicon optimized)
- **Smallest memory footprint**
- **No Python dependencies** in production
- **Cross-platform** (Windows, Linux, macOS, iOS, Android)

### Setup llama.cpp

```bash
# Clone repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build (CPU version)
make

# Build with CUDA (NVIDIA GPU)
make LLAMA_CUBLAS=1

# Build with Metal (Apple Silicon)
make LLAMA_METAL=1

# Build with OpenBLAS (faster CPU)
cmake -B build -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release
```

### Method 1: Convert Existing Model

```bash
# 1. Download HuggingFace model
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# 2. Convert to GGUF
python convert.py Llama-2-7b-chat-hf/ --outfile llama-2-7b.gguf

# 3. Quantize
./quantize llama-2-7b.gguf llama-2-7b-q4_0.gguf q4_0
./quantize llama-2-7b.gguf llama-2-7b-q5_0.gguf q5_0
./quantize llama-2-7b.gguf llama-2-7b-q8_0.gguf q8_0

# 4. Test inference
./main -m llama-2-7b-q4_0.gguf -p "Explain neural networks" -n 512
```

### Method 2: Fine-Tune with llama.cpp (LoRA)

#### Step 1: Prepare Data
```python
# prepare_finetune_data.py
import json

# Training data
data = [
    {"prompt": "What is AI?", "completion": "AI is artificial intelligence..."},
    {"prompt": "Explain Python", "completion": "Python is a programming language..."},
    # Add more examples
]

# Convert to llama.cpp format
with open("train.txt", "w") as f:
    for item in data:
        f.write(f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['completion']}\n\n")
```

#### Step 2: Fine-Tune
```bash
# Fine-tune with LoRA
./finetune \
  --model-base llama-2-7b-q4_0.gguf \
  --train-data train.txt \
  --lora-out lora-weights.bin \
  --threads 8 \
  --batch-size 4 \
  --ctx-size 2048 \
  --iterations 1000

# This creates lora-weights.bin adapter
```

#### Step 3: Use Fine-Tuned Model
```bash
# Run inference with LoRA adapter
./main \
  -m llama-2-7b-q4_0.gguf \
  --lora lora-weights.bin \
  -p "Your prompt here" \
  -n 512
```

### Method 3: Server Mode (Production)

```bash
# Start llama.cpp server
./server \
  -m llama-2-7b-q4_0.gguf \
  --lora lora-weights.bin \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --threads 8

# Now you can use OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Integration with Astra AI

```python
# backend/open_webui/utils/llamacpp.py
import httpx
from typing import AsyncGenerator

class LlamaCppClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def chat_completion(
        self,
        messages: list[dict],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield json.loads(data)
```

---

## Azure AI Foundry

### Setup Azure AI Studio

#### Step 1: Create Resources
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name astra-rg --location eastus

# Create AI Hub
az ml workspace create \
  --name astra-ai-hub \
  --resource-group astra-rg \
  --location eastus
```

#### Step 2: Fine-Tune Model via Azure AI Studio

```python
# azure_finetune.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

# Authenticate
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="your-subscription-id",
    resource_group_name="astra-rg",
    workspace_name="astra-ai-hub"
)

# Upload training data
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

training_data = Data(
    path="./training_data.jsonl",
    type=AssetTypes.URI_FILE,
    description="Astra AI training data",
    name="astra-training-data"
)

ml_client.data.create_or_update(training_data)

# Create fine-tuning job
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

job = command(
    code="./scripts",
    command="python finetune_lora.py --data ${{inputs.training_data}} --output ${{outputs.model}}",
    inputs={
        "training_data": training_data
    },
    outputs={
        "model": {"type": "mlflow_model"}
    },
    environment=Environment(
        image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest"
    ),
    compute="gpu-cluster",
    instance_count=1,
    shm_size="16g",
    display_name="Astra AI Fine-Tuning"
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")

# Monitor job
ml_client.jobs.stream(returned_job.name)
```

#### Step 3: Deploy Fine-Tuned Model

```python
# azure_deploy.py
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="astra-ai-endpoint",
    description="Astra AI custom model",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy model
deployment = ManagedOnlineDeployment(
    name="astra-deployment",
    endpoint_name="astra-ai-endpoint",
    model="azureml:astra-finetuned-model:1",
    instance_type="Standard_NC6s_v3",  # GPU instance
    instance_count=1,
    environment_variables={
        "CUDA_VISIBLE_DEVICES": "0"
    }
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# Get endpoint details
endpoint = ml_client.online_endpoints.get("astra-ai-endpoint")
print(f"Endpoint URL: {endpoint.scoring_uri}")
print(f"API Key: {ml_client.online_endpoints.get_keys(endpoint.name).primary_key}")
```

#### Step 4: Use Deployed Model

```python
# Use Azure endpoint in Astra AI
import httpx

AZURE_ENDPOINT = "https://astra-ai-endpoint.eastus.inference.ml.azure.com/score"
AZURE_API_KEY = "your-api-key"

async def azure_inference(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            AZURE_ENDPOINT,
            headers={
                "Authorization": f"Bearer {AZURE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input_data": {
                    "input_string": [prompt],
                    "parameters": {
                        "temperature": 0.7,
                        "max_new_tokens": 512
                    }
                }
            }
        )
        return response.json()
```

---

## Digital Ocean GPU Setup

### Create GPU Droplet

#### Option 1: Via Web Console
1. Go to DigitalOcean Console
2. Create Droplet → GPU-Optimized
3. Choose: Basic GPU ($1.50/hour) or Standard GPU ($2.50/hour)
4. Select region (NYC3, SFO3, AMS3)
5. Add SSH key

#### Option 2: Via CLI
```bash
# Install doctl
brew install doctl  # macOS
# or download from https://github.com/digitalocean/doctl

# Authenticate
doctl auth init

# Create GPU droplet
doctl compute droplet create astra-gpu \
  --size g-2vcpu-8gb-nvidia-rtx-6000-ada \
  --image gpu-nvidia-docker-20-04 \
  --region nyc3 \
  --ssh-keys your-ssh-key-id \
  --wait

# Get droplet IP
doctl compute droplet list
```

### Setup Fine-Tuning Environment

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Verify NVIDIA GPU
nvidia-smi

# Install Python & dependencies
apt install -y python3.11 python3.11-venv python3-pip git

# Create project directory
mkdir /opt/astra-finetune
cd /opt/astra-finetune

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install fine-tuning dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft bitsandbytes datasets trl
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUBLAS=1

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
```

### Run Fine-Tuning on Digital Ocean

```bash
# Upload your fine-tuning script
scp finetune_ollama.py root@your-droplet-ip:/opt/astra-finetune/
scp training_data.jsonl root@your-droplet-ip:/opt/astra-finetune/

# SSH and run
ssh root@your-droplet-ip
cd /opt/astra-finetune
source venv/bin/activate

# Run fine-tuning (monitor with tmux/screen)
tmux new -s finetune
python finetune_ollama.py

# Detach: Ctrl+B then D
# Reattach: tmux attach -t finetune
```

### Cost Optimization

```bash
# Snapshot the droplet after training
doctl compute droplet-action snapshot <droplet-id> --snapshot-name astra-finetuned

# Destroy droplet
doctl compute droplet delete <droplet-id>

# Recreate from snapshot when needed
doctl compute droplet create astra-gpu \
  --image astra-finetuned \
  --size g-2vcpu-8gb-nvidia-rtx-6000-ada \
  --region nyc3
```

---

## Data Preparation

### 1. Collecting Training Data

#### Public Datasets (HuggingFace)
```python
from datasets import load_dataset

# Instruction-following datasets
datasets = [
    "OpenAssistant/oasst1",           # 88K conversations
    "databricks/databricks-dolly-15k", # 15K instructions
    "tatsu-lab/alpaca",                # 52K instructions
    "HuggingFaceH4/ultrachat_200k",   # 200K conversations
    "teknium/OpenHermes-2.5",          # 1M instructions
]

for ds_name in datasets:
    dataset = load_dataset(ds_name, split="train")
    print(f"{ds_name}: {len(dataset)} examples")
```

#### Custom Data Sources
```python
# Scrape your documentation
import requests
from bs4 import BeautifulSoup

def scrape_docs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract Q&A pairs
    qa_pairs = []
    for section in soup.find_all('div', class_='faq-item'):
        question = section.find('h3').text
        answer = section.find('p').text
        qa_pairs.append({"instruction": question, "output": answer})
    
    return qa_pairs

# Scrape company docs
company_data = scrape_docs("https://your-company.com/docs")
```

### 2. Data Cleaning

```python
import re
import pandas as pd
from langdetect import detect

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def filter_dataset(dataset):
    filtered = []
    for item in dataset:
        # Skip empty
        if not item['instruction'] or not item['output']:
            continue
        
        # Skip too short/long
        if len(item['output']) < 20 or len(item['output']) > 2000:
            continue
        
        # Skip non-English
        try:
            if detect(item['output']) != 'en':
                continue
        except:
            continue
        
        # Clean text
        item['instruction'] = clean_text(item['instruction'])
        item['output'] = clean_text(item['output'])
        
        filtered.append(item)
    
    return filtered
```

### 3. Data Augmentation

```python
# Paraphrase using LLM
async def augment_data(examples, model="gpt-3.5-turbo"):
    augmented = []
    
    for example in examples:
        prompt = f"Paraphrase this question: {example['instruction']}"
        paraphrased = await call_llm(prompt, model)
        
        augmented.append({
            "instruction": paraphrased,
            "output": example['output']
        })
    
    return augmented

# Add negative examples
def add_negatives(examples):
    with_negatives = []
    
    for ex in examples:
        # Original
        with_negatives.append(ex)
        
        # Negative (wrong answer)
        with_negatives.append({
            "instruction": ex['instruction'],
            "output": "I don't have enough information to answer that question."
        })
    
    return with_negatives
```

### 4. Format for Training

```python
def format_for_llama3(examples):
    formatted = []
    
    for ex in examples:
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Astra AI, a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{ex['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ex['output']}<|eot_id|>"""
        
        formatted.append({"text": text})
    
    return formatted

def format_for_mistral(examples):
    formatted = []
    
    for ex in examples:
        text = f"<s>[INST] {ex['instruction']} [/INST] {ex['output']}</s>"
        formatted.append({"text": text})
    
    return formatted
```

---

## Advanced Techniques

### 1. Multi-Modal Fine-Tuning (Vision + Text)

```python
# Fine-tune LLaVA (vision model)
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/llava-1.5-7b-hf",
    load_in_4bit=True,
    use_gradient_checkpointing=True
)

# Add LoRA
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)

# Training data with images
dataset = load_dataset("json", data_files={
    "train": "image_text_pairs.jsonl"
})

# Each example: {"image": "path/to/image.jpg", "question": "...", "answer": "..."}
```

### 2. Merging LoRA Adapters

```python
# Merge multiple LoRA adapters
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("llama-3-8b")

# Merge adapter 1 (customer support)
model = PeftModel.from_pretrained(base_model, "lora_adapter_1")
model = model.merge_and_unload()

# Merge adapter 2 (technical docs)
model = PeftModel.from_pretrained(model, "lora_adapter_2")
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("merged_model")
```

### 3. Continual Learning (Add New Data)

```python
# Load existing fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    "astra-v1",  # Your previous model
    load_in_4bit=True
)

# Add new LoRA adapters
model = FastLanguageModel.get_peft_model(model, r=16)

# Train on new data
new_dataset = load_dataset("json", data_files="new_data.jsonl")
trainer = SFTTrainer(model=model, train_dataset=new_dataset)
trainer.train()

# Merge with existing weights
model = model.merge_and_unload()
model.save_pretrained("astra-v2")
```

### 4. Model Distillation (Large → Small)

```python
# Distill Llama 70B → 7B
from transformers import AutoModelForCausalLM, Trainer

teacher = AutoModelForCausalLM.from_pretrained("llama-3-70b")
student = AutoModelForCausalLM.from_pretrained("llama-3-7b")

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student outputs
        student_outputs = model(**inputs)
        
        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
        
        # KL divergence loss
        loss = nn.KLDivLoss()(
            F.log_softmax(student_outputs.logits / temperature, dim=-1),
            F.softmax(teacher_outputs.logits / temperature, dim=-1)
        )
        
        return (loss, student_outputs) if return_outputs else loss

trainer = DistillationTrainer(model=student, ...)
trainer.train()
```

---

## Production Deployment

### 1. Model Registry

```python
# backend/open_webui/models/model_registry.py
from typing import Dict, List

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, dict] = {}
    
    def register(self, name: str, metadata: dict):
        """Register a model"""
        self.models[name] = {
            "name": name,
            "path": metadata.get("path"),
            "type": metadata.get("type", "ollama"),
            "quantization": metadata.get("quantization"),
            "parameters": metadata.get("parameters", {}),
            "created_at": datetime.now()
        }
    
    def list_models(self) -> List[dict]:
        """List all registered models"""
        return list(self.models.values())
    
    def get_model(self, name: str) -> dict:
        """Get model by name"""
        return self.models.get(name)

# Usage
registry = ModelRegistry()
registry.register("astra-dev", {
    "path": "/models/astra-dev-q4.gguf",
    "type": "llamacpp",
    "quantization": "q4_k_m",
    "parameters": {"temperature": 0.7, "top_p": 0.9}
})
```

### 2. A/B Testing Models

```python
# backend/open_webui/utils/ab_testing.py
import random

class ABTestManager:
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name: str, models: List[str], traffic_split: List[float]):
        """Create A/B test experiment"""
        self.experiments[name] = {
            "models": models,
            "split": traffic_split,
            "results": {model: {"requests": 0, "feedback": []} for model in models}
        }
    
    def select_model(self, experiment_name: str) -> str:
        """Select model based on traffic split"""
        exp = self.experiments[experiment_name]
        return random.choices(exp["models"], weights=exp["split"])[0]
    
    def record_feedback(self, experiment_name: str, model: str, rating: int):
        """Record user feedback"""
        self.experiments[experiment_name]["results"][model]["requests"] += 1
        self.experiments[experiment_name]["results"][model]["feedback"].append(rating)

# Usage
ab_test = ABTestManager()
ab_test.create_experiment(
    "model_comparison",
    models=["astra-v1", "astra-v2"],
    traffic_split=[0.5, 0.5]
)

# In your route
model_to_use = ab_test.select_model("model_comparison")
response = await generate_response(model_to_use, prompt)
```

### 3. Model Monitoring

```python
# backend/open_webui/utils/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
model_requests = Counter(
    'model_requests_total',
    'Total requests per model',
    ['model_name']
)

model_latency = Histogram(
    'model_latency_seconds',
    'Model inference latency',
    ['model_name']
)

model_errors = Counter(
    'model_errors_total',
    'Total errors per model',
    ['model_name', 'error_type']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests',
    ['model_name']
)

# Usage in routes
@router.post("/chat")
async def chat(message: str, model: str):
    model_requests.labels(model_name=model).inc()
    active_requests.labels(model_name=model).inc()
    
    start_time = time.time()
    try:
        response = await generate_response(model, message)
        return response
    except Exception as e:
        model_errors.labels(model_name=model, error_type=type(e).__name__).inc()
        raise
    finally:
        duration = time.time() - start_time
        model_latency.labels(model_name=model).observe(duration)
        active_requests.labels(model_name=model).dec()
```

### 4. Auto-Scaling Based on Load

```yaml
# kubernetes/hpa-ollama.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ollama-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ollama
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: model_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

---

## Interview Questions & Answers

### Q: What's the difference between fine-tuning and prompt engineering?

**A:** "Prompt engineering modifies the input to guide the model's behavior without changing the model itself. It's fast and cheap but limited to the model's existing capabilities. Fine-tuning updates the model's weights using custom data, allowing it to learn new knowledge and patterns. Fine-tuning is more powerful but requires GPU resources and training time. For Astra AI, we use both: prompt engineering for quick customization via Modelfiles, and fine-tuning for specialized domains."

### Q: How do you prevent overfitting during fine-tuning?

**A:** "Several techniques: 1) Use a train/validation split (90/10) and monitor validation loss. 2) Implement early stopping when validation loss plateaus. 3) Use LoRA with low rank (r=8-16) which naturally regularizes. 4) Add dropout in adapter layers. 5) Use weight decay in the optimizer. 6) Ensure diverse training data (5000+ examples). For Astra AI, we also use QLoRA which implicitly reduces overfitting due to quantization noise."

### Q: How do you evaluate fine-tuned models?

**A:** "Multi-faceted approach: 1) Perplexity on held-out test set (lower is better). 2) BLEU/ROUGE scores for specific tasks. 3) Human evaluation with rating scale (1-5). 4) A/B testing in production with real users. 5) Task-specific metrics (accuracy for classification, F1 for NER). For Astra AI, we prioritize user feedback and production metrics over academic benchmarks because they reflect real-world performance."

### Q: What's your strategy for managing model versions?

**A:** "We use a model registry that tracks: name, version, training data hash, hyperparameters, quantization level, and performance metrics. Each model gets a semantic version (v1.0.0). We maintain backward compatibility and provide migration paths. In production, we support blue-green deployment where old and new versions run simultaneously, allowing gradual traffic shift. We also implement canary releases for high-risk changes."

---

**You now have a complete guide to LLM fine-tuning. Master these techniques and you'll be in the top 5% of AI engineers.**
