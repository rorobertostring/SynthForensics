# 在 ComfyUI 中使用 MAGI-1

## 安装方法

1. 用手动安装的方式[下载 ComfyUI 并安装](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#manual-install-windows-linux)

2. 安装MAGI-1节点

    方法一：使用comfy-cli安装（推荐）

    安装 [comfy-cli](https://github.com/Comfy-Org/comfy-cli?tab=readme-ov-file#installation) 后，在命令行中运行以下命令：
    ```shell
    cd ComfyUI

    comfy node registry-install MAGI-1
    ```

    方法二：通过源码安装

    下载本仓库到 *ComfyUI/custom_nodes/MAGI-1* 路径并[安装相应的依赖](https://github.com/SandAI-org/MAGI-1?tab=readme-ov-file#environment-preparation)。

    > ⚠️ 为了让ComfyUI识别到自定义节点，以上两种方法都需要将`comfyui/__init__.py`移动到 MAGI-1 项目的根目录下。

3. 下载 MAGI-1 模型文件到本地。在 MAGI-1 的配置文件，例如在文件`example/4.5B/4.5B_base_config.json`中（若要使用4.5B基础模型），修改模型权重的路径为本地绝对路径。主要有以下三个文件路径需要修改：
    * **load**: DiT 模型权重的绝对路径
    * **t5_pretrained**： T5 模型权重的绝对路径
    * **vae_pretrained**： VAE 模型权重的绝对路径

## 节点功能

安装完成后，在 ComfyUI 目录下启动 ComfyUI

```shell
cd ComfyUI
# 如果安装了 comfy-cli
comfy launch
# 否则
python main.py
```
可在 ComfyUI 的 *Add Node - Magi* 菜单找到本仓库提供的节点，新版 ComfyUI 节点也可在界面左侧 NODE LIBRARY 中找到。

### Load Prompt

用于从输入中加载一段 prompt 文本，用于后续的文本编码处理。

* **prompt**：用户输入的文本内容，支持多行输入。

### T5 Text Encoder

用于将 prompt 文本编码成用于视频生成的文本特征（Conditioning Embedding）。

* **prompt**：输入的描述文本。
* **t5\_pretrained\_path**：T5 模型权重的绝对路径，指向 `ckpt/t5` 目录中的预训练模型。
* **t5\_device**：指定在哪个设备上加载和运行 T5 模型，可选 `"cpu"` 或 `"cuda:x"`（如 `"cuda:0"`）。

### Load Image

用于从输入目录中加载图像文件。支持通过文件选择器上传图片。

* **image\_path**：从 ComfyUI 的输入文件夹中选择图像文件，系统自动过滤非图像的文件类型，仅显示支持的图片格式。


### Process with MAGI

核心节点，可用于文生视频、图生视频、视频续写任务，生成对应的视频序列，并将帧率传递给后续的视频保存节点。

* **task\_mode**：指定执行 *文生视频、图生视频、视频续写* 中的哪一种任务。
* **config\_path**：模型运行所需的 JSON 配置文件的绝对路径。
    > ⚠️ 配置文件中出现的所有路径也需要是绝对路径。
* **image\_path**：要转换为视频的图像/视频的绝对路径。
* **text\_embeddings**：文本编码器生成的嵌入特征与 mask，作为视频生成的语义引导。
* **magi\_seed**：生成随机数的种子，用于控制模型生成的可复现性。相同的 seed 将产生相同的视频输出。默认值为 1234，取值范围 0～100000。
* **video\_size\_h**：生成视频的高度（像素）。默认值为 720，过大会导致运行速度变慢和显存溢出问题，请谨慎设置。
* **video\_size\_w**：生成视频的宽度（像素）。默认值为 720，过大会导致运行速度变慢和显存溢出问题，请谨慎设置。
* **num\_frames**：视频的总帧数，控制生成视频的时长。默认值为 96，范围为 24～24000。
* **num\_steps**：扩散采样的步数，步数越多，画面质量越高但推理时间越长。默认值为 64，范围为 4～240。
* **fps**：生成视频的帧率（每秒帧数），影响视频播放的速度和流畅度。默认值为 24，支持的范围是 1～60。

> ⚠️ 本节点在运行前会设置一系列分布式和内存相关的环境变量。


### Save Video

将生成的视频序列保存为本地文件。

* **video**：待保存的视频张量，本质是一个 torch.Tensor。
* **output\_path**：保存视频的绝对路径（只允许使用 `.mp4` 后缀）。
* **fps**：视频帧率，默认为 24，支持 1\~60 的整数设置。

视频将使用指定的帧率编码并写入到 `output_path`。


## 工作流样例

本节展示了图生视频的工作流样例，可通过菜单中的 *Load* 按钮导入工作流。新版 ComfyUI 可通过左上方菜单的 *Workflow - Open* 加载工作流，也可以将工作流文件拷贝到 ComfyUI 的`user/default/workflows`目录下，再在左侧 *工作流* 面板中刷新即可找到。

工作流位于 `comfyui/workflow/` 目录中，素材位于 `example/assets/` 目录中。

导入工作流后，**需要手动重新指定对应的文件路径片**。

### 文生视频

工作流对应 `workflow/magi_text_to_video_example.json` 文件。

### 图生视频

工作流对应 `workflow/magi_image_to_video_example.json` 文件。

### 视频续写

工作流对应 `workflow/magi_video_continuation_example.json` 文件。
