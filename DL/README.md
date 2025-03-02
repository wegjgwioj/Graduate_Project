# SonarDualStream Project

这是一个轻量级双流时空网络（Lightweight Dual-Stream Network）项目，设计用于声呐图像中的动态目标检测与分割。项目支持静态图像训练和视频流推理，适用于实时声呐数据处理场景，目标是在资源受限设备（如 NVIDIA 4060 GPU）上实现高效运行。

## 项目特点

- **双流架构**：结合静态抑制分支（MobileNetV3-Small）和动态检测分支（EfficientNet-B0）。
- **视频支持**：可处理视频文件（如 MP4）或实时摄像头输入。
- **轻量设计**：模型参数量控制在 2M 以内，适合嵌入式或低功耗设备。
- **伪标签生成**：基于三帧差分法自动生成训练标签。
- **评估指标**：提供 mIoU（交并比）和 FPS（帧率）评估。

## 目录结构

```
SonarDualStream/
├── data/                    # 数据目录
│   ├── images/             # 训练用静态图像（可选，如 frame1.png, frame2.png, ...）
│   └── video/              # 视频文件（例如 sonar_video.mp4）
├── models/                 # 模型定义
│   └── dual_stream_net.py  # 双流网络实现
├── utils/                  # 工具模块
│   ├── data_utils.py       # 数据加载和增强
│   ├── loss.py            # 损失函数
│   └── metrics.py         # 评估指标
├── train.py                # 训练脚本
├── inference_video.py      # 视频推理脚本
├── config.py               # 配置文件
├── requirements.txt        # 依赖清单
└── README.md               # 项目说明（本文档）
```

## 环境要求

- **操作系统**：Windows 11（或其他支持 Python 的系统）
- **Python 版本**：3.8 或更高
- **硬件**：建议配备 NVIDIA GPU（如 4060，支持 CUDA 11.7+）

## 安装依赖

1. **安装依赖**：
   在项目根目录下运行以下命令安装所有必要库：

   ```bash
   pip install -r requirements.txt
   ```
2. **PyTorch GPU 支持**（可选）：

   - `requirements.txt` 中的 `torch` 是 CPU 版。若需 GPU 支持，根据您的 CUDA 版本安装。例如，对于 CUDA 11.7：
     ```bash
     pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
     ```
   - 检查 CUDA 版本：运行 `nvidia-smi`，选择匹配的 PyTorch 版本。
   - 详情参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。
3. **验证安装**：
   运行以下代码确认依赖安装成功：

   ```python
   import torch
   import cv2
   import albumentations
   import numpy
   import sklearn
   import efficientnet_pytorch
   print("All dependencies installed successfully!")
   print(f"PyTorch CUDA: {torch.cuda.is_available()}")
   ```

## 使用方法

### 1. 训练模型（可选）

如果您有静态声呐图像用于训练：

1. 将连续的声呐图像放入 `data/images/`
2. 运行训练脚本：
   ```bash
   python train.py
   ```
3. 训练完成后，模型权重保存为 `model.pth`。

### 2. 视频推理

支持从视频文件或摄像头进行实时推理：

1. **准备视频**：
   - 将声呐视频放入 `data/video/`（例如 `sonar_video.mp4`）。
   - 编辑 `config.py`，确保 `VIDEO_PATH` 指向正确文件路径。
2. **运行推理**：
   - 处理视频文件：

     ```bash
     python inference_video.py
     ```
   - 使用摄像头：
     编辑 `inference_video.py` 末尾，将 `inference_video(config)` 改为：

     ```python
     inference_video(config, video_source=0)
     ```

     然后运行：
     ```bash
     python inference_video.py
     ```
3. **结果**：
   - 两个窗口显示原始帧和分割掩码。
   - 控制台输出实时的 mIoU 和 FPS。

### 3. 配置调整

编辑 `config.py` 修改参数：

- `IMAGE_DIR`: 训练图像目录。
- `VIDEO_PATH`: 视频文件路径。
- `IMG_SIZE`: 输入图像尺寸（默认 224x224）。
- `BATCH_SIZE`: 训练批次大小（默认 8）。
- `NUM_EPOCHS`: 训练轮数（默认 150）。

## 项目工作流程

1. **数据输入**：
   - 训练：从 `data/images/` 读取静态图像序列。
   - 推理：从视频文件或摄像头读取连续帧。
2. **预处理**：
   - 三帧堆叠作为动态输入，中间帧作为静态输入。
   - 应用数据增强（如旋转、噪声）。
3. **模型推理**：
   - 双流网络生成目标分割掩码。
4. **评估**：
   - 使用伪标签计算 mIoU，实时测量 FPS。

## 注意事项

- **视频格式**：支持 MP4、AVI 等常见格式，高帧率可能需更高性能硬件。
- **图像序列**：训练时确保图像文件名连续，以便正确读取三帧。
- **硬件兼容性**：若无 GPU，代码自动切换至 CPU，但性能会下降。
- **退出推理**：按 `q` 键关闭视频窗口。

## 性能

- **推理速度**：≥45 FPS（优化后，使用 TensorRT）。
- **mIoU**：0.72-0.78（取决于数据质量）。
- **模型大小**：<8MB。

## 未来改进

- 集成 TensorRT 提升推理速度（需手动安装）。
- 支持手动标注标签以提高精度。
- 添加多路视频流处理功能。

## 常见问题

- **Q：视频无法打开？**
  A：检查 `VIDEO_PATH` 是否正确，或确保摄像头可用。
- **Q：CUDA 不可用？**
  A：确认 NVIDIA 驱动和 PyTorch 的 CUDA 版本匹配。
- ...
