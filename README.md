# Anime Pipeline Studio

从一句构想开始，围绕实际画面继续创作的本地多模型工作区。

在网页中展开构想、生成候选、比较版本，用自然语言评价，并继续生成、交接模型或局部修改。普通模式使用默认选择，专业模式开放模型条件与参数；研究区用于管理组合实验和比较结果。

支持 NewBie、Anima Turbo、Illustrious／SDXL 和 Z-Image Turbo 的现有适配器。一次创作可以使用一个模型，也可以在已有作品上选择协作路径，不绑定固定的模型接力顺序。

[查看实际图像与交接案例](docs/SHOWCASE.md) · [管线设计](PIPELINE.md) · [模型准备](MODELS.md)

## 运行准备

当前安装脚本面向 Windows，使用 Python 3.12 或 3.13。图像推理需要与你的硬件匹配的 PyTorch、ComfyUI Core 与权重；从零准备引擎的脚本使用 NVIDIA CUDA。应用网页不依赖 ComfyUI 桌面端。

源码不附带模型、Python 环境或 llama-server。准备至少一类图像模型，以及用于理解和观察的语言／视觉模型；不需要把四类图像模型全部下载。具体文件与授权见 [MODELS.md](MODELS.md)。

把项目解压到有足够空间的位置，例如 `E:\AnimePipelineStudio`。以下命令均在该目录的 PowerShell 中执行。依赖安装需要网络；脚本在下载前要求确认，模型权重另外准备。

## 已有 ComfyUI 和模型

1. 安装网页环境：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -WebOnly
   ```

   脚本寻找 Python 3.12／3.13，也可加 `-Python "E:\Python312\python.exe"` 指定。输入 `YES` 才会安装依赖；网页环境建立在项目的 `.venv` 中。

2. 双击 `启动NewBie Studio.bat`，打开 [本地网页](http://127.0.0.1:8090/)，进入右上角“模型接入”。第一次无需先加载模型。

3. 设置图像引擎。连接已有本机 Core 时，填写其服务地址，关闭“需要时自动启动本机 Core”，填写本机可读取的模型根目录，并将 Core 工作目录指向该服务实际使用的目录（模型交接与局部修改会写入其中的 `input`）。也可以填写 Core 源码、Python 和工作目录，由本项目按需启动；自动启动的地址使用 `http://127.0.0.1:8190`，修改端口时同时修改 `engine_port`。各适配器内填写权重相对文件名，目录结构见 [模型指南](MODELS.md#图像模型文件)。

4. 选择理解模型的接法：

   - 本地 GGUF：准备支持当前 `Vulkan0`／NVIDIA GPU0 接法的 Windows Vulkan 构建，指定 `llama-server.exe`、GGUF 和匹配的视觉 projector；按接入页检查结果使用内置语法绑定，或登记当前运行器。运行器同目录须有同一版本的 `llama-server-impl.dll`、`llama.dll`，详见 [本地 GGUF 准备](MODELS.md#本地-gguf)。
   - OpenAI 兼容服务：填写含 `/v1` 的服务地址、文字与视觉模型 ID。服务须支持应用使用的结构化 JSON 与图片输入，可在本机或远端运行，不需要本地 GGUF 或 DLL 登记。需要密钥时，在 Windows“编辑账户的环境变量 → 用户变量”中一次性设置 `AI_COMIC_AUTHORING_API_KEY`（或你配置的变量名），之后日常仍用 BAT 启动；无需鉴权的服务可不设。

5. 点击“检查文件与配置”；已有服务可再点击“探测服务连接”。保存后，依次双击停止、启动 BAT，让运行中应用载入新配置，然后回创作台输入构想。

图像侧当前需要**本机可读取的权重目录**；只填远程 Core 地址、但本机无法访问其模型文件，不属于这条接法。兼容语言服务会收到构想、所选概念原文和需要观察的图片；连接远端前，请确认服务的数据政策。

## 从零准备图像引擎

先安装 Python 3.12／3.13、Git 和合适的显卡驱动，再执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -PrepareEngine
```

这会在你确认后安装网页依赖、准备项目内的 ComfyUI Core 及其独立环境、CUDA PyTorch 和 NewBie 节点。不下载模型或 llama-server。默认 Core 与节点版本固定在脚本中；高级用户可用 `-ComfyRevision`、`-NewBieNodesRevision` 和 `-TorchIndexUrl` 调整版本与 CUDA 索引。

已存在的推理 Python 环境默认只复用，不安装或更新其中的包。如需为该环境补齐或更新依赖，明确添加 `-UpdateEngineDependencies`；确认前会列出目标环境和将进行的安装。共享已有环境时，先确认更新不会影响其他项目。

按 [MODELS.md](MODELS.md) 准备你要用的图像模型和理解模型，再沿上面的“模型接入”步骤配置。权重可放在其他磁盘，已有模型不用复制进项目。

## 日常创作

创作台输入描述，选择画面形式与输出等选项后生成。进度和候选在当前任务显示；打开专业模式可以审阅条件与调整参数。作品中可记录自然语言评价、比较版本，并基于所选作品继续制作。局部修改先选择区域，再描述希望改变什么、保留什么。

研究区管理实验队列、暂停与继续、查看比较结果和评价。静态管线图用于了解结构，迭代页面用于观察具体路径。

默认网页端口 `8090`，图像引擎端口 `8190`，本地语言运行器端口 `8191`；可在模型接入页修改。两份 BAT 使用项目目录，不依赖开发者的磁盘路径。

停止服务使用 `停止NewBie Studio.bat`。启动 BAT 默认管理八小时运行窗口；需要自定义时：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\daily_start.ps1 -Hours 8
```

## 数据与配置

| 内容 | 默认位置 |
| --- | --- |
| 本机连接配置 | `config/settings.json`，安装时从示例创建 |
| 图像权重根目录 | `models`，可改为其他磁盘的已有目录 |
| 图像引擎与语言运行器资源 | `runtime`，也可配置外部路径 |
| 生成图像 | `data/jobs`，可修改输出目录；网页作品可追溯对应文件 |
| 作品记录、创作与研究任务、评价 | `data` 下的对应记录目录 |
| 理解与观察的调用记录 | `work/local_authoring_runtime` |
| 随项目附带的概念原文 | `resources/concepts` |

更新源码前保留 `config/settings.json`、`data`、`work`、`runtime` 和自备模型。不要把这些个人配置、密钥、任务或大权重上传 GitHub。查看启动问题时先检查 `data/frontend-watchdog.log`；接入页的文件检查不等于模型已经完成推理。

## 许可证

应用源码采用 [GPL-3.0-only](LICENSE)，允许使用、修改和商业使用；再分发应用或修改版时需履行相应源码与许可义务。

模型权重、生成图片、角色素材和第三方组件各自适用其授权，不随应用源码自动改为 GPL。尤其是 NewBie 模型及其生成内容具有非商业限制；见 [模型说明](MODELS.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。

产品构思、组合实验、管线设计与画面取舍：syxxx00；代码结合 AI 协作实现。
