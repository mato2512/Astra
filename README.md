<div align="center">

<img src="static/static/logo.png" alt="Astra Logo" width="140"/>

# Astra 👋

![GitHub stars](https://img.shields.io/github/stars/mato2512/Astra?style=social)
![GitHub forks](https://img.shields.io/github/forks/mato2512/Astra?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/mato2512/Astra?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/mato2512/Astra)
![GitHub language count](https://img.shields.io/github/languages/count/mato2512/Astra)
![GitHub top language](https://img.shields.io/github/languages/top/mato2512/Astra)
![GitHub last commit](https://img.shields.io/github/last-commit/mato2512/Astra?color=red)

**Astra is an extensible, feature-rich, and user-friendly self-hosted AI platform designed to operate entirely offline.** It supports various LLM runners like **Ollama**, **llama.cpp**, and **OpenAI-compatible APIs**, with **built-in inference engine** for RAG, making it a **powerful AI deployment solution**.

</div>

---

## Key Features of Astra ⭐

🚀 **Effortless Setup**: Install seamlessly using Docker or Kubernetes (kubectl, kustomize or helm) for a hassle-free experience with support for both `:ollama` and `:cuda` tagged images.

🤝 **Ollama/OpenAI API Integration**: Effortlessly integrate OpenAI-compatible APIs for versatile conversations alongside Ollama models. Customize the OpenAI API URL to link with **LMStudio, GroqCloud, Mistral, OpenRouter, and more**.

🛡️ **Granular Permissions and User Groups**: By allowing administrators to create detailed user roles and permissions, we ensure a secure user environment. This granularity not only enhances security but also allows for customized user experiences, fostering a sense of ownership and responsibility amongst users.

🔄 **SCIM 2.0 Support**: Enterprise-grade user and group provisioning through SCIM 2.0 protocol, enabling seamless integration with identity providers like Okta, Azure AD, and Google Workspace for automated user lifecycle management.

📱 **Responsive Design**: Enjoy a seamless experience across Desktop PC, Laptop, and Mobile devices.

📱 **Progressive Web App (PWA) for Mobile**: Enjoy a native app-like experience on your mobile device with our PWA, providing offline access on localhost and a seamless user interface.

✒️🔢 **Full Markdown and LaTeX Support**: Elevate your LLM experience with comprehensive Markdown and LaTeX capabilities for enriched interaction.

🎤📹 **Hands-Free Voice/Video Call**: Experience seamless communication with integrated hands-free voice and video call features, allowing for a more dynamic and interactive chat environment.

🛠️ **Model Builder**: Easily create Ollama models via the Web UI. Create and add custom characters/agents, customize chat elements, and import models effortlessly.

🐍 **Native Python Function Calling Tool**: Enhance your LLMs with built-in code editor support in the tools workspace. Bring Your Own Function (BYOF) by simply adding your pure Python functions, enabling seamless integration with LLMs.

📚 **Local RAG Integration**: Dive into the future of chat interactions with groundbreaking Retrieval Augmented Generation (RAG) support. This feature seamlessly integrates document interactions into your chat experience. You can load documents directly into the chat or add files to your document library, effortlessly accessing them using the `#` command before a query.

🔍 **Web Search for RAG**: Perform web searches using providers like `SearXNG`, `Google PSE`, `Brave Search`, `serpstack`, `serper`, `Serply`, `DuckDuckGo`, `TavilySearch`, `SearchApi` and `Bing` and inject the results directly into your chat experience.

🌐 **Web Browsing Capability**: Seamlessly integrate websites into your chat experience using the `#` command followed by a URL. This feature allows you to incorporate web content directly into your conversations, enhancing the richness and depth of your interactions.

🎨 **Image Generation Integration**: Seamlessly incorporate image generation capabilities using options such as AUTOMATIC1111 API or ComfyUI (local), and OpenAI's DALL-E (external), enriching your chat experience with dynamic visual content.

⚙️ **Many Models Conversations**: Effortlessly engage with various models simultaneously, harnessing their unique strengths for optimal responses. Enhance your experience by leveraging a diverse set of models in parallel.

🔐 **Role-Based Access Control (RBAC)**: Ensure secure access with restricted permissions; only authorized individuals can access your Ollama, and exclusive model creation/pulling rights are reserved for administrators.

🌐🌍 **Multilingual Support**: Experience Astra in your preferred language with our internationalization (i18n) support. Join us in expanding our supported languages! We're actively seeking contributors!

🧩 **Pipelines, Plugin Support**: Seamlessly integrate custom logic and Python libraries into Astra using Pipelines Plugin Framework. Launch your Pipelines instance, set the OpenAI URL to the Pipelines URL, and explore endless possibilities. Examples include **Function Calling**, User **Rate Limiting** to control access, **Usage Monitoring** with tools like Langfuse, **Live Translation with LibreTranslate** for multilingual support, **Toxic Message Filtering** and much more.

🌟 **Continuous Updates**: We are committed to improving Astra with regular updates, fixes, and new features.

---

## How to Install 🚀

### Quick Start with Docker 🐳

> [!NOTE]  
> For certain Docker environments, additional configurations might be needed. If you encounter any connection issues, refer to the troubleshooting section below.

> [!WARNING]
> When using Docker to install Astra, make sure to include the `-v astra-data:/app/backend/data` in your Docker command. This step is crucial as it ensures your database is properly mounted and prevents any loss of data.

> [!TIP]  
> If you wish to utilize Astra with Ollama included or CUDA acceleration, we recommend utilizing our official images tagged with either `:cuda` or `:ollama`. To enable CUDA, you must install the [Nvidia CUDA container toolkit](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/) on your Linux/WSL system.

#### Installation with Default Configuration

- **If Ollama is on your computer**, use this command:

  ```bash
  docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v astra-data:/app/backend/data --name astra --restart always ghcr.io/mato2512/astra:main
  ```

- **If Ollama is on a Different Server**, use this command:

  To connect to Ollama on another server, change the `OLLAMA_BASE_URL` to the server's URL:

  ```bash
  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v astra-data:/app/backend/data --name astra --restart always ghcr.io/mato2512/astra:main
  ```

- **To run Astra with Nvidia GPU support**, use this command:

  ```bash
  docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v astra-data:/app/backend/data --name astra --restart always ghcr.io/mato2512/astra:cuda
  ```

#### Installation for OpenAI API Usage Only

- **If you're only using OpenAI API**, use this command:

  ```bash
  docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key -v astra-data:/app/backend/data --name astra --restart always ghcr.io/mato2512/astra:main
  ```

#### Installing Astra with Bundled Ollama Support

This installation method uses a single container image that bundles Astra with Ollama, allowing for a streamlined setup via a single command. Choose the appropriate command based on your hardware setup:

- **With GPU Support**:
  Utilize GPU resources by running the following command:

  ```bash
  docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v astra-data:/app/backend/data --name astra --restart always ghcr.io/mato2512/astra:ollama
  ```

- **For CPU Only**:
  If you're not using a GPU, use this command instead:

  ```bash
  docker run -d -p 3000:8080 -v ollama:/root/.ollama -v astra-data:/app/backend/data --name astra --restart always ghcr.io/mato2512/astra:ollama
  ```

Both commands facilitate a built-in, hassle-free installation of both Astra and Ollama, ensuring that you can get everything up and running swiftly.

After installation, you can access Astra at [http://localhost:3000](http://localhost:3000). Enjoy! 😄

### Docker Compose Installation

For production deployments, we recommend using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/mato2512/Astra.git
cd Astra

# Start with Docker Compose
docker compose up -d

# Or for production
docker compose -f docker-compose.prod.yaml up -d
```

---

## Troubleshooting

### Astra: Server Connection Error

If you're experiencing connection issues, it's often due to the WebUI docker container not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container. Use the `--network=host` flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link: `http://localhost:8080`.

**Example Docker Command**:

```bash
docker run -d --network=host -v astra-data:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name astra --restart always ghcr.io/mato2512/astra:main
```

### Keeping Your Docker Installation Up-to-Date

In case you want to update your local Docker installation to the latest version, you can do it with [Watchtower](https://containrrr.dev/watchtower/):

```bash
docker run --rm --volume /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower --run-once astra
```

In the last part of the command, replace `astra` with your container name if it is different.

### Using the Dev Branch 🌙

> [!WARNING]
> The `:dev` branch contains the latest unstable features and changes. Use it at your own risk as it may have bugs or incomplete features.

If you want to try out the latest bleeding-edge features and are okay with occasional instability, you can use the `:dev` tag like this:

```bash
docker run -d -p 3000:8080 -v astra-data:/app/backend/data --name astra --add-host=host.docker.internal:host-gateway --restart always ghcr.io/mato2512/astra:dev
```

### Offline Mode

If you are running Astra in an offline environment, you can set the `HF_HUB_OFFLINE` environment variable to `1` to prevent attempts to download models from the internet.

```bash
export HF_HUB_OFFLINE=1
```

---

## What's Next? 🌟

### Roadmap

- [ ] Enhanced model fine-tuning interface
- [ ] Advanced analytics and monitoring dashboard
- [ ] Mobile application (React Native)
- [ ] Improved voice and video capabilities
- [ ] Extended multi-language support
- [ ] Plugin marketplace
- [ ] Team collaboration features

---

## 🛠️ Technology Stack

**Frontend:**
- SvelteKit 2.5+ - Modern web framework
- Tailwind CSS 4.0 - Utility-first styling
- TypeScript 5.5+ - Type safety
- Vite 5.4+ - Fast builds

**Backend:**
- FastAPI 0.118+ - Python web framework
- SQLAlchemy 2.0+ - Database ORM
- ChromaDB - Vector database for RAG
- Redis - Caching and sessions

**AI/ML:**
- llama.cpp - Optimized LLM inference
- Ollama - Model management
- Transformers - HuggingFace models
- Sentence Transformers - Embeddings

**Infrastructure:**
- Docker - Containerization
- Nginx - Reverse proxy
- PostgreSQL - Production database
- Let's Encrypt - SSL certificates

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

For a detailed record of license changes and applicable terms, please refer to [LICENSE_HISTORY](./LICENSE_HISTORY).

---

## 💬 Support

If you have any questions, suggestions, or need assistance:

- 📧 **Email**: [prasad@ngts.tech](mailto:prasad@ngts.tech)
- 🐛 **Issues**: [GitHub Issues](https://github.com/mato2512/Astra/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/mato2512/Astra/discussions)
- 🌐 **Live Demo**: [astra.ngts.tech](https://astra.ngts.tech)

---

<div align="center">

**Created by [Prasad Navale](https://github.com/mato2512) - Let's make Astra even more amazing together!** 💪

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=mato2512.Astra)

</div>
