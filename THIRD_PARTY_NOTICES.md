# 授权与第三方组件

Anime Pipeline Studio 的原创应用源码、前端、管线组织及本项目说明由 syxxx00 于 2026 年创作，以 **GPL-3.0-only** 发布，完整条款见 [LICENSE](LICENSE)。SPDX 标识：`GPL-3.0-only`。

这项授权允许依 GPL 使用、修改与再分发本项目；不将外部依赖、模型权重、角色、原画或用户上传素材重新授权为 GPL。发布改动版本时，请保留相关版权与许可，并按 GPL 提供对应源码和修改信息。

## 软件组件

| 组件 | 本项目中的用途 | 原始许可与发行方式 |
| --- | --- | --- |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 独立运行的图像推理后端；本项目的 owned-control 扩展在其中运行 | [GPLv3](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE)。本源码发行不捆绑完整 Core；使用者自行安装或复用现有安装。项目自有扩展随本项目以 GPL-3.0-only 发布 |
| [ComfyUI Newbie Nodes](https://github.com/NewBieAI-Lab/ComfyUI-Newbie-Nodes) | NewBie 采样及可选 NewBie LoRA 节点 | 外部依赖，不在本源码包中。该独立仓库未随所查版本提供明确的 LICENSE；不能将其他 NewBie 仓库的 Apache 声明当成它的许可。重新分发它前应向原作者确认 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 用户选择本地 GGUF 时使用的独立语言与视觉推理服务 | [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE)。本包不捆绑其可执行文件或依赖 DLL；如自行捆绑，请一并保留上游版权和许可 |
| [ComfyUI NewBie LLM Formatter](https://github.com/SuzumiyaAkizuki/ComfyUI-NewBie-LLM-Formatter) | 风格配方读取方式的参考；可以读取使用者提供的 styles 配方 | [MIT](https://github.com/SuzumiyaAkizuki/ComfyUI-NewBie-LLM-Formatter/blob/main/LICENSE)，Copyright (c) 2025 SuzumiyaAkizuki。本包不捆绑插件或个人 LPF_config.json；若加入其代码或配方内容，须保留该许可与版权 |
| concept workflow | 项目在研究、设计与修正中使用的概念工作方法 | MIT，Copyright (c) 2026 syxxx00。完整方法随 `resources/concepts/concept-workflow` 提供，并保留其 [MIT 许可](resources/concepts/concept-workflow/LICENSE)；不要求先安装独立工具 |

Web 环境通过依赖声明单独安装，不随源码包再分发：FastAPI、Pydantic、pydantic-settings 使用 MIT；HTTPX、Uvicorn 使用 BSD-3-Clause；huggingface-hub、hf-xet 使用 Apache-2.0；Pillow 使用 MIT-CMU。其完整条款随各软件发行。使用安装器打包环境或二进制时，还须收集实际版本及其传递依赖的许可，不能只附本项目 GPL。

本项目前端使用随仓库提供的 HTML、CSS 和 JavaScript，不要求从 CDN 加载前端框架。

## 模型与生成内容

权重不随本项目发布。请从 [模型准备](MODELS.md) 指向的来源自行取得，遵守具体模型、量化版本和组件的许可。

- **NewBie Exp0.1**：[Newbie-NC-1.0](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/blob/main/LICENSE.md)。其非商业限制涵盖模型、衍生模型及输出，应用源码的 GPL 不取消这些限制。
- **Anima**：[CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md)。模型及衍生模型受非商业等条件限制；现行条款允许输出商业使用，但仍要求遵守其他适用权利与限制。
- **Illustrious XL v2.0**：[官方仓库](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)标注 CreativeML OpenRAIL-M；其他版本、合并模型或 LoRA 须分别核对。
- **Z-Image Turbo**：[官方仓库](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)标注 Apache-2.0。
- 文本编码器、VAE、GGUF、视觉 projector 和 LoRA 也有各自来源与条款。例如 Gemma、Jina、FLUX 组件不能仅因被组合使用就统一视为本项目 GPL。

代码许可不意味着对所有生成图拥有排他版权，也不保证图像不涉及他人权利。混合模型或图生图处理不会自动消除来源素材或前序模型的适用条件。

## 角色与素材来源

`data/knowledge/arknights.json` 保存角色和外观索引、视觉事实描述及来源链接，不包含官方立绘文件。明日方舟及相关角色、名称、原画和商标的权利归其各自权利人；[PRTS](https://prts.wiki/) 是资料来源之一，不表示其或游戏官方赞助、认可本项目。

`images` 中的生成示例与管线图用于说明创作方法，不纳入应用源码的 GPL 授权。生成示例并非官方立绘，涉及明日方舟角色；其角色权利、来源模型条款仍适用，不能把它们视为可无条件商业再用的素材包。

用户上传图、参考图与源码分别管理。不要把整个素材目录默认标成 GPL，也不要在上传仓库时包含他人的原画、未获授权的训练集、模型权重或带密钥的个人配置。
