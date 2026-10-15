# 模型准备与连接

一句话创作需要两类能力：**理解和规划内容的语言服务**，以及**实际生成图片的图像模型**。先准备一种语言连接和一套图像模型即可开始；需要多模型协作时，再加入想参与的其他图像模型，不必一次下载全部权重。

软件与权重分开发行。本项目源码采用 GPL-3.0-only，各模型仍使用自己的许可；下载前先阅读本页末尾及 [第三方说明](THIRD_PARTY_NOTICES.md)。日常启动不会自动下载缺失权重。

## 设置路径

首次执行 `scripts/setup.ps1` 后，编辑 `config/settings.json`，或在网页的模型连接设置中填写路径。所有相对路径都相对于项目根目录；可以将 `model_root` 改为已有模型目录的绝对路径，避免复制一套大文件。

```json
{
  "model_root": "models",
  "comfy_root": "runtime/ComfyUI",
  "engine_base_dir": "runtime/ComfyUI",
  "comfy_python": "runtime/ComfyUI/.venv/Scripts/python.exe",
  "comfy_url": "http://127.0.0.1:8190",
  "engine_port": 8190
}
```

这是需要合并到设置文件的片段，不是要求删除其余字段。`comfy_root` 指向包含 `main.py` 的 ComfyUI Core；`comfy_python` 是安装了相应推理依赖的 Python；`engine_base_dir` 是该引擎的可写运行目录。自动启动固定监听 IPv4 HTTP：`comfy_url` 使用 `http://127.0.0.1:端口`，端口与 `engine_port` 一致，不附加路径。已有的 IPv6、HTTPS 或其他地址服务需由你自行启动，将 `auto_start_engine` 设为 `false` 后连接。

本项目不依赖 ComfyUI 桌面客户端，但使用 ComfyUI Core 的图像推理能力。当前接法面向同机无头 Core，或本机能直接访问其权重目录与工作目录的已有 Core；`engine_base_dir` 必须指向该服务实际使用的目录，图像交接和局部修改会把原图与遮罩写入其中的 `input`。只填写远程地址、却不共享这些目录，不能代替图像模型配置。Core 与 CUDA 环境的准备见 [README](README.md)。

本项目自动启动 Core 时会附加模型路径与 `ai_comic_owned_control` 扩展。手动启动已有 Core 时，可把随项目提供的 `engine_extensions/ai_comic_owned_control` 文件夹复制到该 Core 的 `custom_nodes`，并按自己的路径调整 `config/extra_model_paths.yaml` 后，在 Core 启动命令中加 `--extra-model-paths-config "本项目目录/config/extra_model_paths.yaml"`。扩展用于按任务归属停止及恢复核对，不是生图模型；修改 Core 配置后需要由你重启该服务。

## 图像模型文件

下面的路径均相对于 `model_root`。表内列出本项目设置的默认文件名；若官方下载文件名不同，修改对应设置字段即可。**不同模型的同名文件、Diffusers 分片、GGUF 或整合 checkpoint 不能通过简单改名互相替代。**

| 适配器 | 配置字段 | 所需文件位置 |
| --- | --- | --- |
| NewBie Exp0.1 | `diffusion_model` | `unet/NewBie-image_diffusion_pytorch_model.safetensors` |
| NewBie Exp0.1 | `vae` | `vae/NewBie-image_diffusion_pytorch_model.safetensors` |
| NewBie Exp0.1 | `gemma_clip` | `clip/gemma3-4b-it.safetensors` |
| NewBie Exp0.1 | `jina_clip` | `clip/jina-clip-v2.safetensors` |
| Anima Turbo v1.1 | `anima_diffusion_model` | `diffusion_models/anima-turbo-v1.1.safetensors` |
| Anima Turbo v1.1 | `anima_clip` | `text_encoders/qwen_3_06b_base.safetensors` |
| Anima Turbo v1.1 | `anima_vae` | `vae/qwen_image_vae.safetensors` |
| Z-Image Turbo | `z_diffusion_model` | `diffusion_models/z_image_turbo_bf16.safetensors` |
| Z-Image Turbo | `z_clip` | `text_encoders/qwen_3_4b.safetensors`，也可放 `clip/` |
| Z-Image Turbo | `z_vae` | `vae/ae.safetensors` |
| Illustrious XL v2.0 | `illustrious_checkpoint` | `checkpoints/Illustrious-XL-v2.0.safetensors` |

Illustrious 的当前适配器使用包含模型、CLIP 和 VAE 的 SDXL checkpoint；其他三类使用拆分文件。NewBie 前两项默认名称相同，但**内容不同**，分别是 diffusion model 和 VAE，必须放在各自目录。

可选 LoRA 放入 `loras/`。LoRA 必须匹配所用模型架构；给 SD1.x 或 SDXL 训练的 LoRA 不能因为角色名称相同就装到 NewBie 上。脸部细化等附加操作还需要其对应节点和检测器权重，不属于上述基础生成资源。

### NewBie 来源与格式

- [原模型及许可](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1)。
- [Comfy Org 的拆分重打包资源](https://huggingface.co/Comfy-Org/NewBie-image-Exp0.1_repackaged/tree/main/split_files)。

当前适配器通过 Core 的 `UNETLoader`、`DualCLIPLoader`（`type=newbie`）和 `VAELoader` 读取权重，并使用 NewBie 采样节点。安装或复用 [NewBie 节点](https://github.com/NewBieAI-Lab/ComfyUI-Newbie-Nodes) 时请保留其版本信息；节点不随本源码包捆绑。

使用 Comfy Org 重打包文件时，可以保留原文件名，例如将 `diffusion_model` 改为 `NewBie-Image-Exp0.1-bf16.safetensors`、`gemma_clip` 改为 `gemma_3_4b_it_bf16.safetensors`、`jina_clip` 改为 `jina_clip_v2_bf16.safetensors`，并分别放入上表对应的 `unet/` 和 `clip/` 目录。VAE 单独按所取得的兼容文件设置，不能拿 diffusion 文件充当 VAE。

原模型的 [vae 目录](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/tree/main/vae) 提供 `diffusion_pytorch_model.safetensors`。放入 `vae/` 后可将设置中的 `vae` 改成这个文件名；不要与原模型 transformer 目录中的同名文件混淆。

Comfy 格式的 Gemma/Jina 编码器需要包含对应 tokenizer 数据。若采用第三方节点的原始 Hugging Face 加载方式，则必须同时准备与该版本配套的 tokenizer、config 和必要实现文件，不能只放一个权重分片。本项目的普通启动使用离线模式；缺少这些资源时应补全正确文件，不会在加载时悄悄联网补齐。

### Anima 来源

[CircleStone 官方仓库的 split_files](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files) 包含 diffusion model、text encoder 与 VAE。当前默认是 **Turbo v1.1**，不是旧 v1.0。

也可自行执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_anima.ps1
```

脚本先显示来源、缓存与目标目录并要求输入 `YES`；取消则不下载。显式 `-Download` 适用于已经确认下载的用户操作。它按当前 `settings.json` 安装、保留已有文件，不会重启引擎。

### Z Image 与 Illustrious 来源

- [Z-Image Turbo 原模型](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)；[Comfy Org 拆分文件](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files) 提供 `z_image_turbo_bf16.safetensors`、`qwen_3_4b.safetensors` 和 `ae.safetensors`，分别对应表内三个配置字段。只需所选版本，不必把整个包含其他量化版本的目录都下载。
- [Illustrious XL v2.0 官方仓库](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)。下载对应 SDXL 单文件 checkpoint；文件名不同可改 `illustrious_checkpoint`。

这些适配器没有永久的人物、背景或修补分工。安装了哪些完整资源，决定本机有哪些可执行候选；作品需要与实际结果决定采用哪些方法。

## 语言与视觉服务

选择下面一种接法。图像模型中的 Gemma/Qwen 文本编码器并不等于本项目用来理解构想的语言服务；它们是不同资源。

### 本地 GGUF

当前由项目直接管理的本地运行器面向 Windows，固定使用 `Vulkan0`，并按 NVIDIA GPU0 检查资源。请选择支持所选模型和该设备的 [llama.cpp Windows Vulkan 构建](https://github.com/ggml-org/llama.cpp/releases)。CPU-only、CUDA-only 或其他设备不能直接套用这条本地接法；可以自行运行合适的语言服务，通过下方兼容 API 接入。

将同一发行版的 `llama-server.exe`、`llama-server-impl.dll`、`llama.dll` 及其余配套 DLL 放在同一目录，勿混用不同版本。登记需要 `llama.dll` 提供 `llama_sampler_init_grammar` 和 `llama_sampler_free`，并能解析项目的原生语法。然后准备 GGUF；像素观察还需要与模型匹配的视觉 projector。

当前默认文件布局：

```text
runtime/
  llama.cpp/
    llama-server.exe
    llama-server-impl.dll
    llama.dll
    ...同一发行版的其余配套 DLL
  models/
    Qwen3.5-4B-GGUF/
      Qwen3.5-4B-Q4_K_M.gguf
      mmproj-F16.gguf
```

对应设置：

```json
{
  "authoring_backend": "local_gguf",
  "local_llm_server": "runtime/llama.cpp/llama-server.exe",
  "local_llm_model": "runtime/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf",
  "local_llm_projector": "runtime/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf",
  "local_llm_port": 8191
}
```

本项目既有使用的量化来源是 [Unsloth Qwen3.5 4B GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)，基础模型为 [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B)。量化发布者不是本项目；请核对同一版本的 GGUF 与 projector，不要混配其他尺寸的 `mmproj`。

填写路径后，在“模型接入”页检查文件与配置；若运行器构建与随项目附带的语法绑定一致，可直接使用。不同构建需点击“检查并登记当前运行器”，通过后保存配置，再用停止、启动 BAT 重新载入。文件检查会核对 DLL 与语法文件的哈希和配套关系，但不加载 DLL；显式登记才运行原生语法解析，仍不加载 GGUF。兼容 API 模式不需要本地 DLL 或这项登记。

`local_llm_server` 留空时会查找 PATH、已有 WinGet 安装和项目内默认位置，不安装程序。只做文本理解不需要 projector；像素观察需要配套视觉资源。`local_llm_model_7b`、`local_llm_model_9b` 是选择额外本地变体时才需要的资源，不是首次使用的必装项；分片 GGUF 必须保留同一版本的全部分片。

`semantic_model_path` 对应旧兼容语义接口的 Qwen3 1.7B，**不是当前创作入口的默认语言模型**。不要为了运行当前创作入口先下载旧语义模型。

### 兼容 API

也可以连接自己已有的 OpenAI-compatible 服务，不必在本项目机器上安装语言权重或 llama.cpp。服务需要支持 `/chat/completions`、JSON Schema 结构化输出；像素观察还需要支持图像输入的模型。

```json
{
  "authoring_backend": "openai_compatible",
  "authoring_base_url": "http://127.0.0.1:1234/v1",
  "authoring_model": "填写服务中的文本模型标识",
  "authoring_vision_model": "填写服务中的视觉模型标识",
  "authoring_api_key_env": "AI_COMIC_AUTHORING_API_KEY"
}
```

URL 和模型标识由所用服务提供，示例不是自动安装服务。仅有文本能力时，不要把它当成视觉模型填写。远端地址需使用 HTTPS。

需要密钥时，首次在 Windows 搜索并打开“编辑账户的环境变量”，在“用户变量”中新建 `AI_COMIC_AUTHORING_API_KEY`，变量值填写服务提供的密钥。若修改了 `authoring_api_key_env`，用户变量名也使用你配置的名称。设置文件只保存变量名，不保存密钥。

设置完成后，双击停止、启动 BAT 即可使用；日常仍然直接双击启动 BAT。启动器会读取配置指名的用户变量，不需要每次打开 PowerShell。无需鉴权的本机服务可跳过这一步。

临时调试时，也可在同一个 PowerShell 中先停止服务、设置本次进程的变量，再启动；该临时变量优先于同名用户变量，不会自动更新已经运行的服务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop.ps1
$env:AI_COMIC_AUTHORING_API_KEY = '填写你自己的密钥'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\daily_start.ps1
```

配置远端服务意味着相关构想、上下文和显式参与调用的图片会发给该服务，不再是完全本地处理；费用、保存策略和模型使用条件由该服务决定。本项目不会管理或关闭外部服务进程。

## 检查连接与开始创作

启动网页后，先打开模型连接设置，确认路径和缺失项，再选择已准备好的图像模型创建任务。检查连接与检查文件不生成图片，也不自动安装资源。某个模型不完整时，应补齐它或选择已准备好的候选，而不是把下载全部四类模型作为前提。

下载文件时可使用 `Get-FileHash -Algorithm SHA256` 对照发布者提供的哈希；不要把下载到的 Git LFS 指针文本当成权重。界面显示文件存在，不代表文件内容或配套版本已被核验。

资源占用取决于权重量化、分辨率、上下文、推理后端与其他正在运行的程序。`lowvram` 是运行策略，不保证任意 GPU 都能运行任意模型；首次使用先从单张、基础分辨率开始，再打开高细节和协作选项。

## 模型许可

| 模型 | 获取与使用条件 |
| --- | --- |
| NewBie Exp0.1 | [Newbie-NC-1.0](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/blob/main/LICENSE.md)，非商业限制涵盖模型、衍生模型与输出 |
| Anima | [CircleStone Labs 许可](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md)，模型与衍生模型有非商业限制；现行条款允许输出商业使用，仍须尊重其他适用权利 |
| Illustrious XL v2.0 | [CreativeML OpenRAIL-M](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)，以所下载版本附带条款为准 |
| Z-Image Turbo | [Apache-2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |

编码器、VAE、LoRA 和 GGUF 也应保留各自的来源和许可。应用源码的 GPL 不赋予第三方角色、官方立绘或其他参考素材的使用许可。
