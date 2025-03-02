import torch

class Config:
    # 数据路径
    IMAGE_DIR = "data/images/"        # 训练用图像目录
    VIDEO_PATH = "data/video/sonar_video.mp4"  # 视频文件路径
    MASK_DIR = "data/masks/"          # 可选，真实标签
    
    # 数据参数
    IMG_SIZE = (224, 224)             # 输入图像尺寸
    BATCH_SIZE = 8
    NUM_EPOCHS = 150
    PATIENCE = 20                     # 早停耐心值
    
    # 训练参数
    LEARNING_RATE = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型保存路径
    MODEL_PATH = "model.pth"