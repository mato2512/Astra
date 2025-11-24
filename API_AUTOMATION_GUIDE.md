# API Integration & Automation Workflows Guide

## Table of Contents
1. [OpenAI-Compatible API](#openai-compatible-api)
2. [n8n Integration](#n8n-integration)
3. [Make.com Integration](#makecom-integration)
4. [Zapier Integration](#zapier-integration)
5. [LangChain Integration](#langchain-integration)
6. [Custom Automation Examples](#custom-automation-examples)
7. [Business Use Cases](#business-use-cases)

---

## OpenAI-Compatible API

### Why OpenAI Compatibility Matters

Most automation tools and AI frameworks are built around the OpenAI API standard. By providing OpenAI-compatible endpoints, Astra AI works seamlessly with existing tools without modifications.

### Implement OpenAI-Compatible Endpoints

```python
# backend/open_webui/routers/openai_compat.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from ..utils.auth import get_current_user
from ..utils.ollama import OllamaClient
import time
import uuid

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user = Depends(get_current_user)
):
    """OpenAI-compatible chat completions endpoint"""
    
    ollama = OllamaClient()
    
    if request.stream:
        return StreamingResponse(
            stream_chat_response(ollama, request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        full_response = ""
        async for token in ollama.chat(
            model=request.model,
            messages=[m.dict() for m in request.messages],
            stream=False
        ):
            full_response = token.get("message", {}).get("content", "")
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(full_response.split()),
                "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(full_response.split())
            }
        )

async def stream_chat_response(ollama: OllamaClient, request: ChatCompletionRequest):
    """Stream tokens in OpenAI format"""
    async for chunk in ollama.chat(
        model=request.model,
        messages=[m.dict() for m in request.messages],
        stream=True
    ):
        if "message" in chunk:
            content = chunk["message"].get("content", "")
            if content:
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
    
    # Send final chunk
    yield f"data: [DONE]\n\n"

@router.get("/v1/models")
async def list_models(current_user = Depends(get_current_user)):
    """List available models"""
    ollama = OllamaClient()
    models = await ollama.list_models()
    
    return {
        "object": "list",
        "data": [
            {
                "id": model["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "astra-ai"
            }
            for model in models
        ]
    }

@router.post("/v1/embeddings")
async def create_embeddings(
    request: dict,
    current_user = Depends(get_current_user)
):
    """Generate embeddings"""
    from ..utils.embeddings import generate_embedding
    
    input_text = request.get("input")
    if isinstance(input_text, str):
        input_text = [input_text]
    
    embeddings = []
    for text in input_text:
        embedding = await generate_embedding(text)
        embeddings.append(embedding)
    
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx
            }
            for idx, emb in enumerate(embeddings)
        ],
        "model": request.get("model", "all-minilm-l6-v2"),
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in input_text),
            "total_tokens": sum(len(t.split()) for t in input_text)
        }
    }
```

### Usage Examples

```python
# Python SDK (OpenAI library)
from openai import OpenAI

client = OpenAI(
    base_url="https://your-astra-instance.com/api/v1",
    api_key="your-api-key"
)

# Chat completion
response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

```javascript
// JavaScript/TypeScript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'https://your-astra-instance.com/api/v1',
  apiKey: 'your-api-key'
});

const completion = await openai.chat.completions.create({
  model: 'llama3',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(completion.choices[0].message.content);
```

```bash
# cURL
curl https://your-astra-instance.com/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## n8n Integration

### Setup n8n with Astra AI

#### Method 1: HTTP Request Node (Universal)

```json
{
  "nodes": [
    {
      "parameters": {
        "method": "POST",
        "url": "https://your-astra-instance.com/api/v1/chat/completions",
        "authentication": "predefinedCredentialType",
        "nodeCredentialType": "httpHeaderAuth",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
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
              "value": "llama3"
            },
            {
              "name": "messages",
              "value": "={{ [{\"role\": \"user\", \"content\": $json.query}] }}"
            },
            {
              "name": "temperature",
              "value": 0.7
            }
          ]
        },
        "options": {}
      },
      "name": "Astra AI Chat",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [500, 300]
    }
  ]
}
```

#### Method 2: Custom n8n Node

Create a custom Astra AI node for n8n:

```typescript
// nodes/AstraAI/AstraAI.node.ts
import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';

export class AstraAI implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Astra AI',
    name: 'astraAI',
    icon: 'file:astra.svg',
    group: ['transform'],
    version: 1,
    description: 'Interact with Astra AI self-hosted LLM',
    defaults: {
      name: 'Astra AI',
    },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      {
        name: 'astraAIApi',
        required: true,
      },
    ],
    properties: [
      {
        displayName: 'Resource',
        name: 'resource',
        type: 'options',
        options: [
          {
            name: 'Chat',
            value: 'chat',
          },
          {
            name: 'Embedding',
            value: 'embedding',
          },
        ],
        default: 'chat',
      },
      {
        displayName: 'Model',
        name: 'model',
        type: 'string',
        default: 'llama3',
        description: 'The model to use',
      },
      {
        displayName: 'Prompt',
        name: 'prompt',
        type: 'string',
        typeOptions: {
          rows: 4,
        },
        default: '',
        description: 'The prompt to send',
        displayOptions: {
          show: {
            resource: ['chat'],
          },
        },
      },
      {
        displayName: 'System Message',
        name: 'systemMessage',
        type: 'string',
        default: 'You are a helpful assistant.',
        description: 'System message to guide behavior',
        displayOptions: {
          show: {
            resource: ['chat'],
          },
        },
      },
      {
        displayName: 'Temperature',
        name: 'temperature',
        type: 'number',
        default: 0.7,
        description: 'Sampling temperature (0-2)',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];
    
    const credentials = await this.getCredentials('astraAIApi');
    const baseUrl = credentials.baseUrl as string;
    const apiKey = credentials.apiKey as string;

    for (let i = 0; i < items.length; i++) {
      const resource = this.getNodeParameter('resource', i) as string;
      const model = this.getNodeParameter('model', i) as string;

      if (resource === 'chat') {
        const prompt = this.getNodeParameter('prompt', i) as string;
        const systemMessage = this.getNodeParameter('systemMessage', i) as string;
        const temperature = this.getNodeParameter('temperature', i) as number;

        const response = await this.helpers.request({
          method: 'POST',
          url: `${baseUrl}/v1/chat/completions`,
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
          body: {
            model,
            messages: [
              { role: 'system', content: systemMessage },
              { role: 'user', content: prompt },
            ],
            temperature,
          },
          json: true,
        });

        returnData.push({
          json: {
            response: response.choices[0].message.content,
            model: response.model,
            usage: response.usage,
          },
        });
      }
    }

    return [returnData];
  }
}
```

### Real-World n8n Workflows

#### Workflow 1: Customer Support Automation

```json
{
  "name": "Customer Support AI",
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "support-ticket",
        "httpMethod": "POST"
      }
    },
    {
      "name": "Extract Ticket Info",
      "type": "n8n-nodes-base.set",
      "parameters": {
        "values": {
          "string": [
            {
              "name": "customer_email",
              "value": "={{ $json.body.email }}"
            },
            {
              "name": "issue",
              "value": "={{ $json.body.message }}"
            }
          ]
        }
      }
    },
    {
      "name": "Check Knowledge Base",
      "type": "n8n-nodes-base.postgres",
      "parameters": {
        "query": "SELECT answer FROM kb WHERE question ILIKE '%{{ $json.issue }}%' LIMIT 1"
      }
    },
    {
      "name": "IF No KB Match",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{ $json.answer }}",
              "operation": "isEmpty"
            }
          ]
        }
      }
    },
    {
      "name": "Astra AI Response",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://astra.example.com/api/v1/chat/completions",
        "method": "POST",
        "body": {
          "model": "astra-support",
          "messages": [
            {
              "role": "system",
              "content": "You are a customer support agent. Be helpful, professional, and concise."
            },
            {
              "role": "user",
              "content": "={{ $json.issue }}"
            }
          ]
        }
      }
    },
    {
      "name": "Send Email",
      "type": "n8n-nodes-base.emailSend",
      "parameters": {
        "to": "={{ $json.customer_email }}",
        "subject": "Re: Your Support Ticket",
        "text": "={{ $json.response }}"
      }
    },
    {
      "name": "Log to CRM",
      "type": "n8n-nodes-base.postgres",
      "parameters": {
        "query": "INSERT INTO support_tickets (email, issue, response, created_at) VALUES ('{{ $json.customer_email }}', '{{ $json.issue }}', '{{ $json.response }}', NOW())"
      }
    }
  ],
  "connections": {
    "Webhook": {
      "main": [["Extract Ticket Info"]]
    },
    "Extract Ticket Info": {
      "main": [["Check Knowledge Base"]]
    },
    "Check Knowledge Base": {
      "main": [["IF No KB Match"]]
    },
    "IF No KB Match": {
      "main": [
        [],
        ["Astra AI Response"]
      ]
    },
    "Astra AI Response": {
      "main": [["Send Email", "Log to CRM"]]
    }
  }
}
```

#### Workflow 2: Content Generation Pipeline

```json
{
  "name": "Blog Post Generator",
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.cron",
      "parameters": {
        "triggerTimes": {
          "item": [
            {
              "hour": 9,
              "minute": 0
            }
          ]
        }
      }
    },
    {
      "name": "Get Trending Topics",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://api.trendingtopics.io/today",
        "method": "GET"
      }
    },
    {
      "name": "Generate Title",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://astra.example.com/api/v1/chat/completions",
        "method": "POST",
        "body": {
          "model": "llama3",
          "messages": [
            {
              "role": "system",
              "content": "Generate catchy blog post titles"
            },
            {
              "role": "user",
              "content": "Create 3 engaging blog titles about: {{ $json.topic }}"
            }
          ]
        }
      }
    },
    {
      "name": "Generate Content",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://astra.example.com/api/v1/chat/completions",
        "method": "POST",
        "body": {
          "model": "astra-writer",
          "messages": [
            {
              "role": "system",
              "content": "You are a professional blog writer. Write SEO-optimized, engaging content."
            },
            {
              "role": "user",
              "content": "Write a 1000-word blog post with title: {{ $json.title }}"
            }
          ],
          "temperature": 0.8
        }
      }
    },
    {
      "name": "Generate Meta Description",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://astra.example.com/api/v1/chat/completions",
        "method": "POST",
        "body": {
          "model": "llama3",
          "messages": [
            {
              "role": "user",
              "content": "Write a 155-character meta description for this blog post:\n\n{{ $json.content }}"
            }
          ]
        }
      }
    },
    {
      "name": "Save to WordPress",
      "type": "n8n-nodes-base.wordpress",
      "parameters": {
        "operation": "create",
        "title": "={{ $json.title }}",
        "content": "={{ $json.content }}",
        "excerpt": "={{ $json.meta_description }}",
        "status": "draft"
      }
    }
  ]
}
```

#### Workflow 3: Slack Bot with Context

```json
{
  "name": "Slack AI Assistant",
  "nodes": [
    {
      "name": "Slack Trigger",
      "type": "n8n-nodes-base.slackTrigger",
      "parameters": {
        "channel": "general",
        "events": ["app_mention"]
      }
    },
    {
      "name": "Get Thread History",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "operation": "get",
        "resource": "message",
        "channel": "={{ $json.channel }}",
        "ts": "={{ $json.thread_ts }}"
      }
    },
    {
      "name": "Build Context",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const messages = items.map(item => ({\n  role: item.json.user === 'BOT_ID' ? 'assistant' : 'user',\n  content: item.json.text\n}));\n\nreturn [{ json: { messages } }];"
      }
    },
    {
      "name": "Astra AI Chat",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://astra.example.com/api/v1/chat/completions",
        "method": "POST",
        "body": {
          "model": "astra-chat",
          "messages": "={{ $json.messages }}",
          "temperature": 0.7
        }
      }
    },
    {
      "name": "Reply to Slack",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "operation": "post",
        "resource": "message",
        "channel": "={{ $json.channel }}",
        "text": "={{ $json.response }}",
        "thread_ts": "={{ $json.thread_ts }}"
      }
    }
  ]
}
```

---

## Make.com Integration

### Setup HTTP Module

```json
{
  "name": "Astra AI Chat",
  "connection": {
    "type": "custom",
    "url": "https://your-astra-instance.com/api/v1/chat/completions",
    "method": "POST",
    "headers": [
      {
        "key": "Authorization",
        "value": "Bearer {{parameters.apiKey}}"
      },
      {
        "key": "Content-Type",
        "value": "application/json"
      }
    ],
    "body": {
      "model": "{{parameters.model}}",
      "messages": [
        {
          "role": "system",
          "content": "{{parameters.systemMessage}}"
        },
        {
          "role": "user",
          "content": "{{parameters.prompt}}"
        }
      ],
      "temperature": "{{parameters.temperature}}"
    }
  },
  "parameters": [
    {
      "name": "apiKey",
      "type": "text",
      "label": "API Key",
      "required": true
    },
    {
      "name": "model",
      "type": "select",
      "label": "Model",
      "options": ["llama3", "mistral", "astra-custom"],
      "default": "llama3"
    },
    {
      "name": "prompt",
      "type": "text",
      "label": "Prompt",
      "required": true
    },
    {
      "name": "systemMessage",
      "type": "text",
      "label": "System Message",
      "default": "You are a helpful assistant."
    },
    {
      "name": "temperature",
      "type": "number",
      "label": "Temperature",
      "default": 0.7
    }
  ],
  "response": {
    "output": "{{response.choices[0].message.content}}"
  }
}
```

### Make.com Scenario: Lead Qualification

```
Trigger: Google Sheets (New Row)
  ↓
Action: Astra AI (Analyze Lead)
  - Prompt: "Analyze this lead and score 1-10: Name: {{Name}}, Company: {{Company}}, Budget: {{Budget}}, Need: {{Description}}"
  ↓
Filter: Score > 7
  ↓
Action: Send to CRM (HubSpot)
  ↓
Action: Send Email (Gmail)
  ↓
Action: Notify Sales (Slack)
```

---

## Zapier Integration

### Create Custom Zapier App

```javascript
// zapier/authentication.js
module.exports = {
  type: 'custom',
  fields: [
    {
      key: 'apiKey',
      label: 'API Key',
      required: true,
      type: 'string',
      helpText: 'Get your API key from Astra AI dashboard'
    },
    {
      key: 'baseUrl',
      label: 'Base URL',
      required: true,
      type: 'string',
      default: 'https://your-instance.com/api/v1',
      helpText: 'Your Astra AI instance URL'
    }
  ],
  test: async (z, bundle) => {
    const response = await z.request({
      url: `${bundle.authData.baseUrl}/models`,
      headers: {
        'Authorization': `Bearer ${bundle.authData.apiKey}`
      }
    });
    return response.json;
  }
};

// zapier/creates/chat.js
module.exports = {
  key: 'chat',
  noun: 'Chat',
  display: {
    label: 'Send Chat Message',
    description: 'Send a message to Astra AI and get response'
  },
  operation: {
    inputFields: [
      {
        key: 'model',
        label: 'Model',
        type: 'string',
        required: true,
        default: 'llama3',
        helpText: 'Which model to use'
      },
      {
        key: 'prompt',
        label: 'Prompt',
        type: 'text',
        required: true,
        helpText: 'Your message/question'
      },
      {
        key: 'systemMessage',
        label: 'System Message',
        type: 'text',
        default: 'You are a helpful assistant.',
        helpText: 'Instructions for the AI'
      },
      {
        key: 'temperature',
        label: 'Temperature',
        type: 'number',
        default: 0.7,
        helpText: 'Creativity (0-2)'
      }
    ],
    perform: async (z, bundle) => {
      const response = await z.request({
        method: 'POST',
        url: `${bundle.authData.baseUrl}/chat/completions`,
        headers: {
          'Authorization': `Bearer ${bundle.authData.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: {
          model: bundle.inputData.model,
          messages: [
            {
              role: 'system',
              content: bundle.inputData.systemMessage
            },
            {
              role: 'user',
              content: bundle.inputData.prompt
            }
          ],
          temperature: bundle.inputData.temperature
        }
      });
      
      const data = response.json;
      return {
        id: data.id,
        response: data.choices[0].message.content,
        model: data.model,
        tokens: data.usage.total_tokens
      };
    },
    sample: {
      id: 'chatcmpl-123',
      response: 'This is a sample response from Astra AI.',
      model: 'llama3',
      tokens: 42
    }
  }
};
```

---

## LangChain Integration

### Setup Astra AI with LangChain

```python
# langchain_astra.py
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import httpx

class AstraAI(LLM):
    base_url: str = "https://your-astra-instance.com/api/v1"
    api_key: str
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 512
    
    @property
    def _llm_type(self) -> str:
        return "astra-ai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": stop
                }
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]

# Usage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = AstraAI(
    api_key="your-api-key",
    model="llama3"
)

template = "Translate the following English text to {language}: {text}"
prompt = PromptTemplate(template=template, input_variables=["language", "text"])
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(language="Spanish", text="Hello, how are you?")
print(result)  # Hola, ¿cómo estás?
```

### LangChain RAG with Astra AI

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load documents
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./docs', glob="**/*.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# Create RAG chain
llm = AstraAI(api_key="your-api-key")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Query
query = "What are the system requirements?"
result = qa_chain({"query": query})

print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}")
```

---

## Custom Automation Examples

### 1. Email Response Automation

```python
# automation/email_responder.py
import imaplib
import email
from email.mime.text import MIMEText
import smtplib
import httpx

class EmailAIResponder:
    def __init__(self, astra_api_url, astra_api_key):
        self.astra_url = astra_api_url
        self.api_key = astra_api_key
        self.client = httpx.Client()
    
    async def get_ai_response(self, email_content: str, context: str = ""):
        """Get AI response for email"""
        prompt = f"""You are a professional email assistant. 
        
Context: {context}

Email received:
{email_content}

Write a professional, helpful response."""

        response = self.client.post(
            f"{self.astra_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "astra-email",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            }
        )
        
        return response.json()["choices"][0]["message"]["content"]
    
    def process_inbox(self, imap_server, email_account, password):
        """Process unread emails"""
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_account, password)
        mail.select('inbox')
        
        _, messages = mail.search(None, 'UNSEEN')
        
        for num in messages[0].split():
            _, msg = mail.fetch(num, '(RFC822)')
            email_body = email.message_from_bytes(msg[0][1])
            
            sender = email_body['From']
            subject = email_body['Subject']
            
            # Get email content
            if email_body.is_multipart():
                for part in email_body.walk():
                    if part.get_content_type() == "text/plain":
                        content = part.get_payload(decode=True).decode()
            else:
                content = email_body.get_payload(decode=True).decode()
            
            # Generate AI response
            ai_response = self.get_ai_response(content, f"Subject: {subject}")
            
            # Send reply
            self.send_email(email_account, sender, f"Re: {subject}", ai_response)
    
    def send_email(self, from_addr, to_addr, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_addr
        msg['To'] = to_addr
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_addr, password)
            smtp.send_message(msg)

# Run continuously
responder = EmailAIResponder(
    "https://astra.example.com/api/v1",
    "your-api-key"
)

while True:
    responder.process_inbox("imap.gmail.com", "your-email@gmail.com", "password")
    time.sleep(300)  # Check every 5 minutes
```

### 2. Document Summarization Pipeline

```python
# automation/document_summarizer.py
import os
import httpx
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, astra_url, api_key, output_dir):
        self.astra_url = astra_url
        self.api_key = api_key
        self.output_dir = output_dir
        self.client = httpx.Client()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.txt', '.pdf', '.docx')):
            self.process_document(event.src_path)
    
    def process_document(self, file_path):
        """Extract, summarize, and save"""
        # Read document
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Generate summary
        summary = self.summarize(content)
        
        # Generate key points
        key_points = self.extract_key_points(content)
        
        # Save output
        output_path = os.path.join(
            self.output_dir,
            f"{Path(file_path).stem}_summary.md"
        )
        
        with open(output_path, 'w') as f:
            f.write(f"# Summary\n\n{summary}\n\n")
            f.write(f"## Key Points\n\n{key_points}\n")
    
    def summarize(self, text: str) -> str:
        response = self.client.post(
            f"{self.astra_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "llama3",
                "messages": [{
                    "role": "user",
                    "content": f"Summarize this document in 3 paragraphs:\n\n{text[:4000]}"
                }]
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    
    def extract_key_points(self, text: str) -> str:
        response = self.client.post(
            f"{self.astra_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "llama3",
                "messages": [{
                    "role": "user",
                    "content": f"Extract 5 key points from this document:\n\n{text[:4000]}"
                }]
            }
        )
        return response.json()["choices"][0]["message"]["content"]

# Watch directory
handler = DocumentHandler(
    "https://astra.example.com/api/v1",
    "your-api-key",
    "./summaries"
)

observer = Observer()
observer.schedule(handler, path='./documents', recursive=False)
observer.start()

print("Watching for new documents...")
observer.join()
```

### 3. Social Media Content Generator

```python
# automation/social_media_bot.py
import schedule
import time
import httpx
from datetime import datetime

class SocialMediaBot:
    def __init__(self, astra_url, api_key):
        self.astra_url = astra_url
        self.api_key = api_key
        self.client = httpx.Client()
    
    def generate_post(self, topic: str, platform: str) -> str:
        """Generate social media post"""
        prompts = {
            "twitter": f"Write a engaging tweet (280 chars max) about: {topic}. Include hashtags.",
            "linkedin": f"Write a professional LinkedIn post about: {topic}. Include insights and a call-to-action.",
            "instagram": f"Write an Instagram caption about: {topic}. Be creative and engaging."
        }
        
        response = self.client.post(
            f"{self.astra_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "astra-social",
                "messages": [{
                    "role": "system",
                    "content": "You are a social media expert."
                }, {
                    "role": "user",
                    "content": prompts[platform]
                }],
                "temperature": 0.9
            }
        )
        
        return response.json()["choices"][0]["message"]["content"]
    
    def post_to_twitter(self, content: str):
        # Use Twitter API
        pass
    
    def post_to_linkedin(self, content: str):
        # Use LinkedIn API
        pass
    
    def daily_content(self):
        """Generate and post daily content"""
        topics = self.get_trending_topics()
        
        for platform in ["twitter", "linkedin"]:
            post = self.generate_post(topics[0], platform)
            print(f"{platform.upper()} POST:\n{post}\n")
            
            # Actually post
            if platform == "twitter":
                self.post_to_twitter(post)
            elif platform == "linkedin":
                self.post_to_linkedin(post)
    
    def get_trending_topics(self):
        # Get topics from news API or internal database
        return ["AI in healthcare", "Sustainable technology", "Remote work trends"]

# Schedule posts
bot = SocialMediaBot("https://astra.example.com/api/v1", "your-api-key")

schedule.every().day.at("09:00").do(bot.daily_content)
schedule.every().day.at("15:00").do(bot.daily_content)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Business Use Cases

### 1. Customer Support Automation

**Problem:** High support ticket volume, slow response times, API costs $5000/month

**Solution:** Astra AI + n8n workflow
- Auto-categorize tickets
- Generate responses for common issues
- Escalate complex issues to humans
- 24/7 availability

**Results:**
- 70% tickets auto-resolved
- Response time: 2 hours → 2 minutes
- Cost: $5000/month → $150/month
- Customer satisfaction: +35%

### 2. Content Marketing Agency

**Problem:** Need to generate 100+ blog posts/month for clients, expensive copywriters

**Solution:** Astra AI content pipeline
- Trend analysis → Topic generation
- SEO research → Outline creation
- Content generation → Human review
- Multi-language support

**Results:**
- Output: 30 posts/month → 120 posts/month
- Cost per post: $200 → $20
- Quality maintained with human editing
- Revenue: +150%

### 3. E-commerce Product Descriptions

**Problem:** 10,000 products without descriptions, affecting SEO

**Solution:** Automated description generation
- Extract product specs from database
- Generate SEO-optimized descriptions
- A/B test different versions
- Multi-language expansion

**Results:**
- All products described in 2 weeks
- Organic traffic: +45%
- Conversion rate: +12%
- International sales: +80%

### 4. Internal Knowledge Base Assistant

**Problem:** Employees spend 2 hours/day searching for information

**Solution:** RAG-powered Slack bot
- Indexed all company docs
- Instant answers in Slack
- Context-aware responses
- Usage analytics

**Results:**
- Search time: 2 hours → 10 minutes/day
- Productivity gain: 15,000 hours/year
- Onboarding time: 4 weeks → 2 weeks
- Employee satisfaction: +40%

### 5. Sales Automation

**Problem:** Sales team overwhelmed with lead qualification

**Solution:** Automated lead scoring
- Analyze inbound leads
- Score based on fit
- Generate personalized outreach
- Schedule follow-ups

**Results:**
- Leads processed: 50/day → 500/day
- Close rate: +25%
- Sales cycle: 60 days → 35 days
- Revenue per rep: +180%

---

## Cost Comparison: Astra AI vs. Cloud APIs

### Scenario: Medium-Sized Company (1M requests/month)

| Provider | Cost Structure | Monthly Cost |
|----------|---------------|--------------|
| **OpenAI GPT-4** | $0.03/1K tokens in, $0.06/1K tokens out | $45,000 |
| **Anthropic Claude** | $0.015/1K tokens in, $0.075/1K tokens out | $40,000 |
| **Google Gemini Pro** | $0.0005/1K chars in, $0.0015/1K chars out | $8,000 |
| **Astra AI (Self-Hosted)** | GPU compute ($0.50/hour × 24 × 30) | $360 |

**Savings with Astra AI: $39,640/month (99% reduction)**

### ROI Calculation

**Initial Investment:**
- Digital Ocean GPU Droplet setup: $500
- Fine-tuning development: $2,000
- Integration development: $3,000
- **Total: $5,500**

**Monthly Savings:** $39,640

**Break-even:** 4.2 days

**12-Month ROI:** 8,641%

---

## Interview Questions & Answers

### Q: How do you handle API downtime in automation workflows?

**A:** "We implement multiple fallback strategies: 1) Primary model on our Astra instance, 2) Secondary model on a different server, 3) Cached responses for common queries, 4) Graceful degradation with simple rule-based responses, 5) Queue system to retry failed requests. For n8n, we use error handling nodes to catch failures and route to alternatives. This ensures 99.9% uptime even during maintenance."

### Q: How do you prevent the AI from generating inappropriate responses in automation?

**A:** "Multi-layer approach: 1) Content filtering in system prompts, 2) Output validation using regex and keyword matching, 3) Confidence thresholds—if model uncertainty is high, route to human, 4) Human-in-the-loop for sensitive domains (healthcare, legal), 5) Logging all responses for audit, 6) Regular fine-tuning based on feedback. For Astra AI, we also implement rate limiting per user to prevent abuse."

### Q: Explain your n8n workflow optimization strategy

**A:** "We optimize for: 1) Parallelization—run independent nodes simultaneously, 2) Caching—store expensive API calls, 3) Batching—combine multiple requests, 4) Lazy evaluation—only run necessary branches, 5) Resource management—limit concurrent executions, 6) Monitoring—track execution time and identify bottlenecks. We've reduced average workflow time from 45s to 8s using these techniques."

---

**You now have complete knowledge of API integration and automation workflows with Astra AI. This positions you as an expert in AI-powered automation—a highly valuable skill.**
