# Astra AI# Astra 👋



<div align="center">![GitHub stars](https://img.shields.io/github/stars/open-webui/open-webui?style=social)

  <img src="static/static/logo.png" alt="Astra AI Logo" width="200"/>![GitHub forks](https://img.shields.io/github/forks/open-webui/open-webui?style=social)

  ![GitHub watchers](https://img.shields.io/github/watchers/open-webui/open-webui?style=social)

  ### Advanced AI Chat Interface![GitHub repo size](https://img.shields.io/github/repo-size/open-webui/open-webui)

  ![GitHub language count](https://img.shields.io/github/languages/count/open-webui/open-webui)

  A powerful, feature-rich AI chat application with support for multiple LLM providers.![GitHub top language](https://img.shields.io/github/languages/top/open-webui/open-webui)

</div>![GitHub last commit](https://img.shields.io/github/last-commit/open-webui/open-webui?color=red)

[![Discord](https://img.shields.io/badge/Discord-Open_WebUI-blue?logo=discord&logoColor=white)](https://discord.gg/5rJgQTnV4s)

---[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/tjbck)



## 🌟 Features**Astra is an [extensible](https://docs.openwebui.com/features/plugin/), feature-rich, and user-friendly self-hosted AI platform designed to operate entirely offline.** It supports various LLM runners like **Ollama** and **OpenAI-compatible APIs**, with **built-in inference engine** for RAG, making it a **powerful AI deployment solution**.



- **Multi-Model Support**: Compatible with Ollama, OpenAI, Anthropic, Google Gemini, and morePassionate about open-source AI? [Join our team →](https://careers.openwebui.com/)

- **Modern UI**: Beautiful, responsive interface built with SvelteKit and Tailwind CSS

- **Real-time Chat**: Seamless conversation experience with streaming responses![Astra Demo](./demo.gif)

- **Document Processing**: Upload and analyze documents, PDFs, and images

- **Code Execution**: Built-in Python code interpreter powered by Pyodide> [!TIP]  

- **Memory & Knowledge**: Persistent conversation memory and knowledge base> **Looking for an [Enterprise Plan](https://docs.openwebui.com/enterprise)?** – **[Speak with Our Sales Team Today!](mailto:sales@openwebui.com)**

- **Custom Functions & Tools**: Extend functionality with custom tools and plugins>

- **Voice Support**: Text-to-speech and speech-to-text capabilities> Get **enhanced capabilities**, including **custom theming and branding**, **Service Level Agreement (SLA) support**, **Long-Term Support (LTS) versions**, and **more!**

- **Multi-language**: Support for multiple languages and locales

- **User Management**: Role-based access control and user authenticationFor more information, be sure to check out our [Astra Documentation](https://docs.openwebui.com/).

- **Dark/Light Mode**: Customizable themes including dark and light modes

## Key Features of Astra ⭐

## 🚀 Quick Start

- 🚀 **Effortless Setup**: Install seamlessly using Docker or Kubernetes (kubectl, kustomize or helm) for a hassle-free experience with support for both `:ollama` and `:cuda` tagged images.

### Prerequisites

- 🤝 **Ollama/OpenAI API Integration**: Effortlessly integrate OpenAI-compatible APIs for versatile conversations alongside Ollama models. Customize the OpenAI API URL to link with **LMStudio, GroqCloud, Mistral, OpenRouter, and more**.

- **Node.js** v22.16.0 or higher

- **Python** 3.11 or higher (recommended for backend)- 🛡️ **Granular Permissions and User Groups**: By allowing administrators to create detailed user roles and permissions, we ensure a secure user environment. This granularity not only enhances security but also allows for customized user experiences, fostering a sense of ownership and responsibility amongst users.

- **Docker** (optional, for containerized deployment)

- 🔄 **SCIM 2.0 Support**: Enterprise-grade user and group provisioning through SCIM 2.0 protocol, enabling seamless integration with identity providers like Okta, Azure AD, and Google Workspace for automated user lifecycle management.

### Frontend Setup

- 📱 **Responsive Design**: Enjoy a seamless experience across Desktop PC, Laptop, and Mobile devices.

```bash

# Install dependencies- 📱 **Progressive Web App (PWA) for Mobile**: Enjoy a native app-like experience on your mobile device with our PWA, providing offline access on localhost and a seamless user interface.

npm install --legacy-peer-deps

- ✒️🔢 **Full Markdown and LaTeX Support**: Elevate your LLM experience with comprehensive Markdown and LaTeX capabilities for enriched interaction.

# Run development server

npm run dev- 🎤📹 **Hands-Free Voice/Video Call**: Experience seamless communication with integrated hands-free voice and video call features, allowing for a more dynamic and interactive chat environment.

```

- 🛠️ **Model Builder**: Easily create Ollama models via the Web UI. Create and add custom characters/agents, customize chat elements, and import models effortlessly through [Astra Community](https://openwebui.com/) integration.

The frontend will be available at `http://localhost:5173`

- 🐍 **Native Python Function Calling Tool**: Enhance your LLMs with built-in code editor support in the tools workspace. Bring Your Own Function (BYOF) by simply adding your pure Python functions, enabling seamless integration with LLMs.

### Backend Setup

- 📚 **Local RAG Integration**: Dive into the future of chat interactions with groundbreaking Retrieval Augmented Generation (RAG) support. This feature seamlessly integrates document interactions into your chat experience. You can load documents directly into the chat or add files to your document library, effortlessly accessing them using the `#` command before a query.

```bash

# Navigate to backend directory- 🔍 **Web Search for RAG**: Perform web searches using providers like `SearXNG`, `Google PSE`, `Brave Search`, `serpstack`, `serper`, `Serply`, `DuckDuckGo`, `TavilySearch`, `SearchApi` and `Bing` and inject the results directly into your chat experience.

cd backend

- 🌐 **Web Browsing Capability**: Seamlessly integrate websites into your chat experience using the `#` command followed by a URL. This feature allows you to incorporate web content directly into your conversations, enhancing the richness and depth of your interactions.

# Create virtual environment

python -m venv venv- 🎨 **Image Generation Integration**: Seamlessly incorporate image generation capabilities using options such as AUTOMATIC1111 API or ComfyUI (local), and OpenAI's DALL-E (external), enriching your chat experience with dynamic visual content.



# Activate virtual environment- ⚙️ **Many Models Conversations**: Effortlessly engage with various models simultaneously, harnessing their unique strengths for optimal responses. Enhance your experience by leveraging a diverse set of models in parallel.

# Windows:

.\venv\Scripts\activate- 🔐 **Role-Based Access Control (RBAC)**: Ensure secure access with restricted permissions; only authorized individuals can access your Ollama, and exclusive model creation/pulling rights are reserved for administrators.

# Linux/Mac:

source venv/bin/activate- 🌐🌍 **Multilingual Support**: Experience Astra in your preferred language with our internationalization (i18n) support. Join us in expanding our supported languages! We're actively seeking contributors!



# Install dependencies- 🧩 **Pipelines, Astra Plugin Support**: Seamlessly integrate custom logic and Python libraries into Astra using [Pipelines Plugin Framework](https://github.com/open-webui/pipelines). Launch your Pipelines instance, set the OpenAI URL to the Pipelines URL, and explore endless possibilities. [Examples](https://github.com/open-webui/pipelines/tree/main/examples) include **Function Calling**, User **Rate Limiting** to control access, **Usage Monitoring** with tools like Langfuse, **Live Translation with LibreTranslate** for multilingual support, **Toxic Message Filtering** and much more.

pip install -r requirements.txt

- 🌟 **Continuous Updates**: We are committed to improving Astra with regular updates, fixes, and new features.

# Run backend server

cd open_webuiWant to learn more about Astra's features? Check out our [Astra documentation](https://docs.openwebui.com/features) for a comprehensive overview!

uvicorn main:app --host 0.0.0.0 --port 8080 --reload

```## Sponsors 🙌



The backend will be available at `http://localhost:8080`#### Emerald



### Docker Deployment (Recommended)<table>

  <!-- <tr>

```bash    <td>

# Build and run with Docker Compose      <a href="https://n8n.io/" target="_blank">

docker compose up -d        <img src="https://docs.openwebui.com/sponsors/logos/n8n.png" alt="n8n" style="width: 8rem; height: 8rem; border-radius: .75rem;" />

      </a>

# Or with GPU support    </td>

docker compose -f docker-compose.gpu.yaml up -d    <td>

```      <a href="https://n8n.io/">n8n</a> • Does your interface have a backend yet?<br>Try <a href="https://n8n.io/">n8n</a>

    </td>

## 📖 Configuration  </tr> -->

  <tr>

### Environment Variables    <td>

      <a href="https://tailscale.com/blog/self-host-a-local-ai-stack/?utm_source=OpenWebUI&utm_medium=paid-ad-placement&utm_campaign=OpenWebUI-Docs" target="_blank">

Create a `.env` file in the backend directory:        <img src="https://docs.openwebui.com/sponsors/logos/tailscale.png" alt="Tailscale" style="width: 8rem; height: 8rem; border-radius: .75rem;" />

      </a>

```env    </td>

# Database    <td>

DATABASE_URL=sqlite:///data/webui.db      <a href="https://tailscale.com/blog/self-host-a-local-ai-stack/?utm_source=OpenWebUI&utm_medium=paid-ad-placement&utm_campaign=OpenWebUI-Docs">Tailscale</a> • Connect self-hosted AI to any device with Tailscale

    </td>

# Ollama Configuration  </tr>

OLLAMA_BASE_URL=http://localhost:11434   <tr>

    <td>

# OpenAI Configuration (optional)      <a href="https://warp.dev/open-webui" target="_blank">

OPENAI_API_KEY=your_api_key_here        <img src="https://docs.openwebui.com/sponsors/logos/warp.png" alt="Warp" style="width: 8rem; height: 8rem; border-radius: .75rem;" />

      </a>

# Application Settings    </td>

WEBUI_NAME=Astra    <td>

WEBUI_SECRET_KEY=your_secret_key_here      <a href="https://warp.dev/open-webui">Warp</a> • The intelligent terminal for developers

    </td>

# Authentication  </tr>

ENABLE_SIGNUP=True</table>

DEFAULT_USER_ROLE=user

```---



## 🛠️ Technology StackWe are incredibly grateful for the generous support of our sponsors. Their contributions help us to maintain and improve our project, ensuring we can continue to deliver quality work to our community. Thank you!



### Frontend## How to Install 🚀

- **SvelteKit** 2.5.20 - Web framework

- **Vite** 5.4.19 - Build tool### Installation via Python pip 🐍

- **Tailwind CSS** 4.0 - Styling

- **TypeScript** 5.5.4 - Type safetyAstra can be installed using pip, the Python package installer. Before proceeding, ensure you're using **Python 3.11** to avoid compatibility issues.



### Backend1. **Install Astra**:

- **FastAPI** 0.118.0+ - Python web framework   Open your terminal and run the following command to install Astra:

- **Uvicorn** 0.37.0 - ASGI server

- **SQLAlchemy** 2.0.38 - ORM   ```bash

- **Pydantic** - Data validation   pip install open-webui

   ```

### AI Integration

- Ollama2. **Running Astra**:

- OpenAI API   After installation, you can start Astra by executing:

- Anthropic Claude

- Google Gemini   ```bash

- Mistral AI   open-webui serve

- And many more...   ```



## 📁 Project StructureThis will start the Astra server, which you can access at [http://localhost:8080](http://localhost:8080)



```### Quick Start with Docker 🐳

Astra_Ai/

├── backend/                 # Backend FastAPI application> [!NOTE]  

│   ├── open_webui/         # Main application code> Please note that for certain Docker environments, additional configurations might be needed. If you encounter any connection issues, our detailed guide on [Astra Documentation](https://docs.openwebui.com/) is ready to assist you.

│   │   ├── models/         # Database models

│   │   ├── routers/        # API routes> [!WARNING]

│   │   ├── utils/          # Utility functions> When using Docker to install Astra, make sure to include the `-v open-webui:/app/backend/data` in your Docker command. This step is crucial as it ensures your database is properly mounted and prevents any loss of data.

│   │   └── main.py         # Application entry point

│   └── requirements.txt    # Python dependencies> [!TIP]  

├── src/                    # Frontend SvelteKit application> If you wish to utilize Astra with Ollama included or CUDA acceleration, we recommend utilizing our official images tagged with either `:cuda` or `:ollama`. To enable CUDA, you must install the [Nvidia CUDA container toolkit](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/) on your Linux/WSL system.

│   ├── lib/               # Shared components and utilities

│   ├── routes/            # Application routes### Installation with Default Configuration

│   └── app.html           # HTML template

├── static/                # Static assets- **If Ollama is on your computer**, use this command:

│   └── static/           # Logos, icons, and images

├── docker-compose.yaml   # Docker configuration  ```bash

└── package.json          # Node.js dependencies  docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

```  ```



## 🔧 Development- **If Ollama is on a Different Server**, use this command:



### Running Tests  To connect to Ollama on another server, change the `OLLAMA_BASE_URL` to the server's URL:



```bash  ```bash

# Frontend tests  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

npm run test  ```



# Backend tests- **To run Astra with Nvidia GPU support**, use this command:

cd backend

pytest  ```bash

```  docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda

  ```

### Building for Production

### Installation for OpenAI API Usage Only

```bash

# Build frontend- **If you're only using OpenAI API**, use this command:

npm run build

  ```bash

# Build Docker image  docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

docker build -t astra-ai .  ```

```

### Installing Astra with Bundled Ollama Support

## 🤝 Contributing

This installation method uses a single container image that bundles Astra with Ollama, allowing for a streamlined setup via a single command. Choose the appropriate command based on your hardware setup:

Contributions are welcome! Feel free to:

- **With GPU Support**:

1. Fork the repository  Utilize GPU resources by running the following command:

2. Create a feature branch (`git checkout -b feature/amazing-feature`)

3. Commit your changes (`git commit -m 'Add amazing feature'`)  ```bash

4. Push to the branch (`git push origin feature/amazing-feature`)  docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama

5. Open a Pull Request  ```



## 📝 License- **For CPU Only**:

  If you're not using a GPU, use this command instead:

This project is licensed under the MIT License - see the LICENSE file for details.

  ```bash

## 🙏 Acknowledgments  docker run -d -p 3000:8080 -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama

  ```

- Built on top of Open WebUI framework

- Powered by various open-source AI modelsBoth commands facilitate a built-in, hassle-free installation of both Astra and Ollama, ensuring that you can get everything up and running swiftly.

- Community contributions and feedback

After installation, you can access Astra at [http://localhost:3000](http://localhost:3000). Enjoy! 😄

## 📞 Support

### Other Installation Methods

For issues, questions, or suggestions:

- Open an issue on GitHubWe offer various installation alternatives, including non-Docker native installation methods, Docker Compose, Kustomize, and Helm. Visit our [Astra Documentation](https://docs.openwebui.com/getting-started/) or join our [Discord community](https://discord.gg/5rJgQTnV4s) for comprehensive guidance.

- Check the documentation

- Join our community discussionsLook at the [Local Development Guide](https://docs.openwebui.com/getting-started/advanced-topics/development) for instructions on setting up a local development environment.



---### Troubleshooting



<div align="center">Encountering connection issues? Our [Astra Documentation](https://docs.openwebui.com/troubleshooting/) has got you covered. For further assistance and to join our vibrant community, visit the [Astra Discord](https://discord.gg/5rJgQTnV4s).

  Made with ❤️ by the Astra AI Team

</div>#### Astra: Server Connection Error


If you're experiencing connection issues, it’s often due to the WebUI docker container not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container . Use the `--network=host` flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link: `http://localhost:8080`.

**Example Docker Command**:

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

### Keeping Your Docker Installation Up-to-Date

In case you want to update your local Docker installation to the latest version, you can do it with [Watchtower](https://containrrr.dev/watchtower/):

```bash
docker run --rm --volume /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower --run-once open-webui
```

In the last part of the command, replace `open-webui` with your container name if it is different.

Check our Updating Guide available in our [Astra Documentation](https://docs.openwebui.com/getting-started/updating).

### Using the Dev Branch 🌙

> [!WARNING]
> The `:dev` branch contains the latest unstable features and changes. Use it at your own risk as it may have bugs or incomplete features.

If you want to try out the latest bleeding-edge features and are okay with occasional instability, you can use the `:dev` tag like this:

```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui --add-host=host.docker.internal:host-gateway --restart always ghcr.io/open-webui/open-webui:dev
```

### Offline Mode

If you are running Astra in an offline environment, you can set the `HF_HUB_OFFLINE` environment variable to `1` to prevent attempts to download models from the internet.

```bash
export HF_HUB_OFFLINE=1
```

## What's Next? 🌟

Discover upcoming features on our roadmap in the [Astra Documentation](https://docs.openwebui.com/roadmap/).

## License 📜

This project contains code under multiple licenses. The current codebase includes components licensed under the Astra License with an additional requirement to preserve the "Astra" branding, as well as prior contributions under their respective original licenses. For a detailed record of license changes and the applicable terms for each section of the code, please refer to [LICENSE_HISTORY](./LICENSE_HISTORY). For complete and updated licensing details, please see the [LICENSE](./LICENSE) and [LICENSE_HISTORY](./LICENSE_HISTORY) files.

## Support 💬

If you have any questions, suggestions, or need assistance, please open an issue or join our
[Astra Discord community](https://discord.gg/5rJgQTnV4s) to connect with us! 🤝

## Star History

<a href="https://star-history.com/#open-webui/open-webui&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-webui/open-webui&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=open-webui/open-webui&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=open-webui/open-webui&type=Date" />
  </picture>
</a>

---

Created by [Timothy Jaeryang Baek](https://github.com/tjbck) - Let's make Astra even more amazing together! 💪
