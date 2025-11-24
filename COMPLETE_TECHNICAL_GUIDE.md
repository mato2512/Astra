# Complete Technical Guide - Astra AI Platform

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Frontend Deep Dive](#frontend-deep-dive)
3. [Backend Deep Dive](#backend-deep-dive)
4. [LLM Fine-Tuning Process](#llm-fine-tuning-process)
5. [Docker & Kubernetes](#docker--kubernetes)
6. [Database & Storage](#database--storage)
7. [Security Implementation](#security-implementation)
8. [API Integration](#api-integration)
9. [Performance Optimization](#performance-optimization)
10. [Interview Preparation](#interview-preparation)

---

## System Architecture

### High-Level Overview
```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                         │
│  Browser (SvelteKit) | Mobile PWA | VS Code Extension       │
└───────────────────┬─────────────────────────────────────────┘
                    │ HTTPS/WSS
┌───────────────────▼─────────────────────────────────────────┐
│                      Load Balancer (Nginx)                   │
│          SSL Termination | Rate Limiting | Caching          │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                   Application Layer (FastAPI)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Auth    │  │  Chat    │  │  RAG     │  │  Admin   │   │
│  │  Router  │  │  Router  │  │  Router  │  │  Router  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                     Service Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │PostgreSQL│  │  Redis   │  │ Vector DB│  │  Ollama  │   │
│  │  (Data)  │  │ (Cache)  │  │  (RAG)   │  │  (LLM)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Communication
- **Frontend ↔ Backend**: REST API (HTTP) + WebSocket (streaming)
- **Backend ↔ Database**: SQLAlchemy ORM (connection pooling)
- **Backend ↔ Redis**: Redis-py (caching, sessions)
- **Backend ↔ Ollama**: HTTP API (inference)
- **Backend ↔ Vector DB**: Python SDK (embeddings, search)

---

## Frontend Deep Dive

### SvelteKit Architecture

#### 1. **File-based Routing**
```
src/routes/
├── (app)/               # Protected routes (authenticated users)
│   ├── +layout.svelte   # App shell layout
│   ├── c/[id]/          # Chat conversation
│   ├── workspace/       # Workspace management
│   └── settings/        # User settings
├── auth/                # Authentication routes
│   ├── +page.svelte     # Login page
│   └── signup/          # Registration
└── +layout.svelte       # Root layout
```

#### 2. **State Management (Svelte Stores)**
```typescript
// src/lib/stores/chat.ts
import { writable } from 'svelte/store';

export const chatStore = writable({
  conversations: [],
  activeConversation: null,
  messages: []
});

// Usage in components
import { chatStore } from '$lib/stores/chat';

$chatStore.conversations; // Reactive
```

#### 3. **WebSocket Implementation**
```typescript
// src/lib/apis/streaming.ts
export class StreamingClient {
  private socket: WebSocket;
  
  connect(url: string, token: string) {
    this.socket = new WebSocket(url);
    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Handle streaming tokens
      this.handleToken(data);
    };
  }
  
  sendMessage(content: string) {
    this.socket.send(JSON.stringify({
      type: 'message',
      content: content
    }));
  }
}
```

#### 4. **API Client (OpenAI-compatible)**
```typescript
// src/lib/apis/openai.ts
export async function chatCompletion(params: {
  model: string;
  messages: Message[];
  stream?: boolean;
}) {
  const response = await fetch('/api/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getToken()}`
    },
    body: JSON.stringify(params)
  });
  
  if (params.stream) {
    return handleStreamingResponse(response);
  }
  return response.json();
}
```

#### 5. **Responsive Design (TailwindCSS)**
```svelte
<!-- src/lib/components/chat/ChatMessage.svelte -->
<div class="
  flex flex-col gap-2 p-4
  md:flex-row md:gap-4 md:p-6
  lg:max-w-4xl lg:mx-auto
  dark:bg-gray-800 dark:text-white
">
  <Avatar user={message.user} />
  <div class="flex-1">
    <MessageContent content={message.content} />
  </div>
</div>
```

#### 6. **Internationalization (i18n)**
```typescript
// src/lib/i18n/index.ts
import i18n from 'i18next';
import en from './locales/en.json';
import es from './locales/es.json';

i18n.init({
  resources: {
    en: { translation: en },
    es: { translation: es }
  },
  lng: 'en',
  fallbackLng: 'en'
});

// Usage
import { t } from 'i18next';
<h1>{t('welcome.title')}</h1>
```

### Frontend Performance Optimization

1. **Code Splitting**
```javascript
// vite.config.ts
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['svelte', 'svelte/store'],
          'ui': ['@components/ui/*'],
          'utils': ['@lib/utils/*']
        }
      }
    }
  }
}
```

2. **Lazy Loading**
```svelte
<script>
  import { onMount } from 'svelte';
  let HeavyComponent;
  
  onMount(async () => {
    HeavyComponent = (await import('./HeavyComponent.svelte')).default;
  });
</script>

{#if HeavyComponent}
  <svelte:component this={HeavyComponent} />
{/if}
```

3. **Image Optimization**
```svelte
<picture>
  <source srcset="/images/hero.webp" type="image/webp">
  <source srcset="/images/hero.avif" type="image/avif">
  <img src="/images/hero.jpg" alt="Hero" loading="lazy">
</picture>
```

---

## Backend Deep Dive

### FastAPI Application Structure

#### 1. **Main Application**
```python
# backend/open_webui/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Astra AI API",
    version="1.0.0",
    docs_url="/api/docs"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
from .routers import auth, chat, documents, admin
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
```

#### 2. **Database Models (SQLAlchemy)**
```python
# backend/open_webui/models/user.py
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from .base import Base
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
```

#### 3. **Authentication (JWT)**
```python
# backend/open_webui/utils/auth.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
```

#### 4. **Dependency Injection**
```python
# backend/open_webui/routers/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

#### 5. **WebSocket Handler (Streaming)**
```python
# backend/open_webui/routers/chat.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        del self.active_connections[user_id]
    
    async def send_message(self, user_id: str, message: dict):
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_json(message)

manager = ConnectionManager()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Stream LLM response
            async for token in stream_llm_response(message):
                await manager.send_message(user_id, {
                    "type": "token",
                    "content": token
                })
    except WebSocketDisconnect:
        manager.disconnect(user_id)
```

#### 6. **Ollama Integration**
```python
# backend/open_webui/utils/ollama.py
import httpx
from typing import AsyncGenerator

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": stream}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
    
    async def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = True
    ) -> AsyncGenerator[dict, None]:
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": stream}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)
```

#### 7. **RAG Pipeline**
```python
# backend/open_webui/retrieval/rag.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self, persist_directory: str = "./data/chromadb"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def add_documents(self, documents: list[str], metadatas: list[dict]):
        chunks = []
        chunk_metadatas = []
        
        for doc, metadata in zip(documents, metadatas):
            doc_chunks = self.text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
            chunk_metadatas.extend([metadata] * len(doc_chunks))
        
        self.vectorstore.add_texts(chunks, metadatas=chunk_metadatas)
        self.vectorstore.persist()
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def generate_context(self, query: str, k: int = 5) -> str:
        results = self.search(query, k)
        context = "\n\n".join([r["content"] for r in results])
        return context
```

---

## LLM Fine-Tuning Process

### Complete Fine-Tuning Workflow

#### 1. **Data Collection**
```python
# scripts/collect_data.py
import datasets
from huggingface_hub import login

# Login to HuggingFace
login(token="your_token")

# Load public dataset
dataset = datasets.load_dataset(
    "OpenAssistant/oasst1",
    split="train"
)

# Filter for high-quality conversations
filtered_dataset = dataset.filter(
    lambda x: x["lang"] == "en" and x["rank"] == 0
)

# Save to JSONL
with open("training_data.jsonl", "w") as f:
    for item in filtered_dataset:
        f.write(json.dumps({
            "instruction": item["parent_text"],
            "input": "",
            "output": item["text"]
        }) + "\n")
```

#### 2. **Data Preprocessing**
```python
# scripts/preprocess_data.py
import json
from datasets import Dataset

def format_instruction(sample):
    """Format data into instruction-tuning format"""
    return {
        "text": f"""<|im_start|>system
You are a helpful AI assistant.
<|im_end|>
<|im_start|>user
{sample['instruction']}
<|im_end|>
<|im_start|>assistant
{sample['output']}
<|im_end|>"""
    }

# Load and process
data = []
with open("training_data.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

dataset = Dataset.from_list(data)
dataset = dataset.map(format_instruction)
dataset = dataset.train_test_split(test_size=0.1)

dataset.save_to_disk("processed_dataset")
```

#### 3. **Fine-Tuning with LoRA (Ollama)**
```python
# scripts/finetune_lora.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 8-bit quantization
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
model.save_pretrained("./lora_weights")
```

#### 4. **Convert to GGUF (Quantization)**
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert to GGUF
python convert.py \
  --model-path /path/to/lora_output \
  --output-path model.gguf

# Quantize to 4-bit
./quantize model.gguf model-q4_0.gguf q4_0

# Quantize to 5-bit
./quantize model.gguf model-q5_0.gguf q5_0

# Quantize to 8-bit
./quantize model.gguf model-q8_0.gguf q8_0
```

#### 5. **Create Modelfile for Ollama**
```dockerfile
# Modelfile
FROM ./model-q4_0.gguf

TEMPLATE """<|im_start|>system
{{ .System }}
<|im_end|>
<|im_start|>user
{{ .Prompt }}
<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are Astra AI, a helpful assistant specialized in [domain].
You provide accurate, concise, and helpful responses."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

#### 6. **Deploy to Ollama**
```bash
# Create model in Ollama
ollama create astra-custom -f Modelfile

# Test model
ollama run astra-custom "Hello, how are you?"

# List models
ollama list

# Check model details
ollama show astra-custom
```

### Fine-Tuning on Cloud Platforms

#### **Digital Ocean GPU Droplet**
```bash
# Create GPU droplet (via API or UI)
doctl compute droplet create astra-gpu \
  --size g-2vcpu-8gb \
  --image gpu-nvidia-docker-20-04 \
  --region nyc3 \
  --ssh-keys your-ssh-key-id

# SSH into droplet
ssh root@your-droplet-ip

# Install dependencies
apt update && apt install -y nvidia-cuda-toolkit
pip install transformers accelerate peft bitsandbytes

# Run fine-tuning
python finetune_lora.py
```

#### **Azure AI Foundry**
```python
# azure_finetune.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Authenticate
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="your-subscription-id",
    resource_group_name="your-rg",
    workspace_name="your-workspace"
)

# Submit fine-tuning job
from azure.ai.ml import command

job = command(
    code="./scripts",
    command="python finetune_lora.py",
    environment="azureml://registries/azureml/environments/acft-hf-nlp-gpu/versions/latest",
    compute="gpu-cluster",
    instance_count=1
)

ml_client.jobs.create_or_update(job)
```

---

## Docker & Kubernetes

### Docker Configuration

#### 1. **Multi-Stage Dockerfile**
```dockerfile
# Dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend build
COPY --from=frontend-builder /app/build ./static

# Create non-root user
RUN useradd -m -u 1000 astra && chown -R astra:astra /app
USER astra

EXPOSE 8080

CMD ["uvicorn", "open_webui.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### 2. **Docker Compose (Production)**
```yaml
# docker-compose.prod.yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: astra
      POSTGRES_USER: astra
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    networks:
      - astra-network
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    networks:
      - astra-network
    restart: unless-stopped
  
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    networks:
      - astra-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
  
  app:
    build: .
    depends_on:
      - postgres
      - redis
      - ollama
    environment:
      DATABASE_URL: postgresql://astra:${DB_PASSWORD}@postgres:5432/astra
      REDIS_URL: redis://redis:6379
      OLLAMA_BASE_URL: http://ollama:11434
      SECRET_KEY: ${SECRET_KEY}
    ports:
      - "8080:8080"
    networks:
      - astra-network
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - app
    networks:
      - astra-network
    restart: unless-stopped

networks:
  astra-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  ollama_data:
```

#### 3. **Nginx Configuration**
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8080;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000" always;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /ws {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 86400;
        }
    }
}
```

### Kubernetes Deployment

#### 1. **Deployment Manifests**
```yaml
# kubernetes/manifest/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: astra-app
  namespace: astra
spec:
  replicas: 3
  selector:
    matchLabels:
      app: astra
  template:
    metadata:
      labels:
        app: astra
    spec:
      containers:
      - name: app
        image: your-registry/astra:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: astra-secrets
              key: database-url
        - name: REDIS_URL
          value: redis://redis-service:6379
        - name: OLLAMA_BASE_URL
          value: http://ollama-service:11434
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: astra-service
  namespace: astra
spec:
  selector:
    app: astra
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### 2. **HorizontalPodAutoscaler**
```yaml
# kubernetes/manifest/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: astra-hpa
  namespace: astra
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: astra-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 3. **Helm Chart**
```yaml
# kubernetes/helm/astra/values.yaml
replicaCount: 3

image:
  repository: your-registry/astra
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: astra.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: astra-tls
      hosts:
        - astra.yourdomain.com

resources:
  requests:
    memory: 512Mi
    cpu: 500m
  limits:
    memory: 1Gi
    cpu: 1000m

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: true
  auth:
    username: astra
    database: astra

redis:
  enabled: true
  architecture: standalone
```

---

## Database & Storage

### PostgreSQL Schema
```sql
-- migrations/001_initial.sql

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);

-- Documents table (RAG)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_documents_user_id ON documents(user_id);

-- API Keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
```

### Database Optimization
```python
# backend/open_webui/utils/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

# Connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,  # Max connections
    max_overflow=0,
    pool_pre_ping=True,  # Check connection health
    echo=False  # Disable SQL logging in production
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Dependency for routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Redis Caching Strategy
```python
# backend/open_webui/utils/cache.py
import redis
import json
from functools import wraps
from typing import Optional

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

def cache(ttl: int = 300):
    """Cache decorator with TTL in seconds"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            return result
        return wrapper
    return decorator

# Usage
@cache(ttl=600)
async def get_user_conversations(user_id: str):
    # Expensive DB query
    pass
```

---

## Security Implementation

### 1. **Input Validation**
```python
# backend/open_webui/models/schemas.py
from pydantic import BaseModel, EmailStr, Field, validator
import re

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain digit')
        return v
```

### 2. **SQL Injection Prevention**
```python
# Always use parameterized queries with SQLAlchemy
from sqlalchemy import text

# ❌ Bad (vulnerable to SQL injection)
query = f"SELECT * FROM users WHERE email = '{email}'"
result = db.execute(query)

# ✅ Good (parameterized query)
query = text("SELECT * FROM users WHERE email = :email")
result = db.execute(query, {"email": email})

# ✅ Better (ORM)
result = db.query(User).filter(User.email == email).first()
```

### 3. **CSRF Protection**
```python
# backend/open_webui/middleware/csrf.py
from fastapi import Request, HTTPException
from itsdangerous import URLSafeTimedSerializer

csrf_serializer = URLSafeTimedSerializer(SECRET_KEY)

def generate_csrf_token() -> str:
    return csrf_serializer.dumps("csrf_token")

def validate_csrf_token(token: str) -> bool:
    try:
        csrf_serializer.loads(token, max_age=3600)
        return True
    except:
        return False

@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    if request.method in ["POST", "PUT", "DELETE"]:
        token = request.headers.get("X-CSRF-Token")
        if not token or not validate_csrf_token(token):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
    response = await call_next(request)
    return response
```

### 4. **Rate Limiting**
```python
# backend/open_webui/middleware/rate_limit.py
from fastapi import Request, HTTPException
from datetime import datetime, timedelta
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=1)

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
    
    async def check_rate_limit(self, request: Request):
        # Get client IP
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        # Get current count
        current = redis_client.get(key)
        
        if current is None:
            # First request
            redis_client.setex(key, self.window, 1)
        elif int(current) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        else:
            redis_client.incr(key)

rate_limiter = RateLimiter(max_requests=100, window=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    await rate_limiter.check_rate_limit(request)
    response = await call_next(request)
    return response
```

### 5. **XSS Prevention**
```python
# backend/open_webui/utils/sanitize.py
import bleach
from markupsafe import escape

ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'u', 'a', 'code', 'pre']
ALLOWED_ATTRIBUTES = {'a': ['href', 'title']}

def sanitize_html(content: str) -> str:
    """Remove dangerous HTML/JS"""
    return bleach.clean(
        content,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True
    )

def escape_user_input(text: str) -> str:
    """Escape HTML entities"""
    return escape(text)
```

---

## API Integration

### n8n Workflow Integration
```json
{
  "nodes": [
    {
      "parameters": {
        "url": "https://your-astra-instance.com/api/chat/completions",
        "authentication": "predefinedCredentialType",
        "nodeCredentialType": "httpHeaderAuth",
        "sendHeaders": true,
        "headerParameters": {
          "parameter": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "model",
              "value": "astra-custom"
            },
            {
              "name": "messages",
              "value": "={{[{role: 'user', content: $json.query}]}}"
            },
            {
              "name": "stream",
              "value": false
            }
          ]
        }
      },
      "name": "Astra AI",
      "type": "n8n-nodes-base.httpRequest",
      "position": [250, 300]
    }
  ]
}
```

### Make.com (Integromat) Module
```json
{
  "name": "astra-ai",
  "label": "Astra AI",
  "description": "Connect to your self-hosted Astra AI instance",
  "connection": {
    "type": "apikey",
    "fields": [
      {
        "name": "apiUrl",
        "label": "API URL",
        "type": "text",
        "required": true
      },
      {
        "name": "apiKey",
        "label": "API Key",
        "type": "text",
        "required": true
      }
    ]
  },
  "actions": [
    {
      "name": "chat",
      "label": "Send Chat Message",
      "parameters": [
        {
          "name": "model",
          "label": "Model",
          "type": "select",
          "required": true
        },
        {
          "name": "message",
          "label": "Message",
          "type": "text",
          "required": true
        }
      ]
    }
  ]
}
```

---

## Performance Optimization

### 1. **Database Query Optimization**
```python
# ❌ N+1 Query Problem
conversations = db.query(Conversation).filter(Conversation.user_id == user_id).all()
for conv in conversations:
    messages = db.query(Message).filter(Message.conversation_id == conv.id).all()

# ✅ Eager Loading
from sqlalchemy.orm import joinedload

conversations = db.query(Conversation)\
    .options(joinedload(Conversation.messages))\
    .filter(Conversation.user_id == user_id)\
    .all()
```

### 2. **Caching Strategy**
```python
# Multi-level caching
class CacheManager:
    def __init__(self):
        self.memory_cache = {}  # L1 cache
        self.redis_cache = redis_client  # L2 cache
    
    async def get(self, key: str):
        # Check L1 cache
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check L2 cache
        value = self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = json.loads(value)
            return self.memory_cache[key]
        
        return None
    
    async def set(self, key: str, value: any, ttl: int = 300):
        # Store in both caches
        self.memory_cache[key] = value
        self.redis_cache.setex(key, ttl, json.dumps(value))
```

### 3. **Async Processing**
```python
# Use background tasks for non-critical operations
from fastapi import BackgroundTasks

async def send_email_notification(user_email: str):
    # Slow email sending
    pass

@router.post("/conversations")
async def create_conversation(
    conversation: ConversationCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    # Create conversation
    new_conversation = Conversation(**conversation.dict())
    db.add(new_conversation)
    db.commit()
    
    # Send email in background
    background_tasks.add_task(send_email_notification, current_user.email)
    
    return new_conversation
```

---

## Interview Preparation

### Key Technical Questions & Answers

#### **1. Explain the architecture of Astra AI**
**Answer:** "Astra AI is a full-stack application with a SvelteKit frontend, FastAPI backend, and Ollama for LLM inference. The frontend communicates via REST API and WebSockets for streaming. The backend uses PostgreSQL for structured data, Redis for caching, and a vector database for RAG. We use Docker for containerization and Kubernetes for orchestration in production."

#### **2. How do you handle model fine-tuning?**
**Answer:** "We use LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning. First, we collect and preprocess data in instruction-tuning format. Then we train using HuggingFace Transformers with 8-bit quantization. After training, we convert the model to GGUF format using llama.cpp and quantize to 4-bit for efficient inference. Finally, we deploy to Ollama with a custom Modelfile specifying system prompts and parameters."

#### **3. How do you ensure data security?**
**Answer:** "We implement multiple security layers: JWT authentication, parameterized SQL queries to prevent injection, input validation with Pydantic, CSRF protection, rate limiting, and HTTPS/TLS encryption. All data stays on the customer's infrastructure—we never send data to third parties. We also support RBAC for multi-user scenarios."

#### **4. What's your approach to scaling?**
**Answer:** "We use horizontal pod autoscaling in Kubernetes based on CPU/memory metrics. For the database, we implement connection pooling and read replicas. We use Redis for caching frequently accessed data. For the LLM inference, we support multi-GPU setups and model quantization to reduce memory footprint. Load balancing is handled by Nginx or Kubernetes Ingress."

#### **5. How do you optimize LLM inference latency?**
**Answer:** "We use several techniques: model quantization (4-bit GGUF), KV cache optimization, batch processing for multiple requests, and GPU acceleration. We also implement streaming responses via WebSocket so users see tokens as they're generated. For frequently asked questions, we cache embeddings and responses in Redis."

#### **6. Explain your RAG implementation**
**Answer:** "Our RAG pipeline uses sentence transformers for embeddings, stores them in ChromaDB/Qdrant, and implements semantic search. When a query comes in, we generate an embedding, find the top-k similar chunks, and inject them as context into the LLM prompt. We use RecursiveCharacterTextSplitter to chunk documents while preserving semantic meaning."

#### **7. How do you handle CI/CD?**
**Answer:** "We use GitHub Actions for CI/CD. On push, we run linters, unit tests, and integration tests. On merge to main, we build Docker images, push to a container registry, and deploy to staging. For production, we use blue-green deployment on Kubernetes to minimize downtime. We also implement automated rollback if health checks fail."

#### **8. What's your monitoring strategy?**
**Answer:** "We use OpenTelemetry for distributed tracing, Prometheus for metrics, and Grafana for visualization. We track key metrics like request latency, error rates, token generation speed, and resource utilization. We also implement structured logging with correlation IDs for debugging."

### Business Value Points

- **Cost Savings:** "Organizations save 90% on AI costs by eliminating per-token pricing"
- **Data Privacy:** "100% data sovereignty—critical for healthcare, finance, legal sectors"
- **Uptime:** "No dependency on third-party APIs means 100% uptime SLA"
- **Customization:** "Fine-tuned models understand domain-specific terminology and workflows"
- **Scalability:** "Kubernetes allows scaling from startup to enterprise without architecture changes"

---

## Next Steps for Mastery

1. **Deploy to production** on Digital Ocean/Azure
2. **Fine-tune a custom model** on domain-specific data
3. **Build the VS Code extension**
4. **Create video demonstrations** for each feature
5. **Write technical blog posts** about the implementation
6. **Contribute to open source** components you use
7. **Present at local tech meetups** or conferences

---

**This guide covers 95% of what CTOs/CEOs will ask. Master these concepts and you'll be ready for lead-level positions.**
