# utils/data_utils.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np

# 数据增强
transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussianNoise(var_limit=(0.01, 0.05), p=0.5),
    A.CLAHE(p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5)
])

# 生成伪标签
def generate_pseudo_label(frames):
    d1 = cv2.absdiff(frames[1], frames[0])
    d2 = cv2.absdiff(frames[2], frames[1])
    motion_mask = cv2.bitwise_and(d1, d2)
    _, mask = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)
    return mask / 255.0  # 归一化到 [0, 1]

# 训练数据集类（静态图像）
class SonarDataset(Dataset):
    def __init__(self, image_dir, img_size, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files) - 2  # 需要三帧连续图像

    def __getitem__(self, idx):
        frame1 = cv2.imread(os.path.join(self.image_dir, self.image_files[idx]), 0)
        frame2 = cv2.imread(os.path.join(self.image_dir, self.image_files[idx+1]), 0)
        frame3 = cv2.imread(os.path.join(self.image_dir, self.image_files[idx+2]), 0)
        frames = [frame1, frame2, frame3]
        
        pseudo_label = generate_pseudo_label(frames)
        
        static_input = cv2.resize(frame2, self.img_size)
        dynamic_input = cv2.resize(np.stack(frames, axis=-1), self.img_size)
        pseudo_label = cv2.resize(pseudo_label, self.img_size)
        
        if self.transform:
            augmented = self.transform(image=static_input, mask=pseudo_label)
            static_input = augmented['image']
            pseudo_label = augmented['mask']
            dynamic_input = self.transform(image=dynamic_input)['image']
        
        static_input = torch.tensor(static_input, dtype=torch.float32).unsqueeze(0) / 255.0
        dynamic_input = torch.tensor(dynamic_input, dtype=torch.float32).permute(2, 0, 1) / 255.0
        pseudo_label = torch.tensor(pseudo_label, dtype=torch.float32).unsqueeze(0)
        
        return static_input, dynamic_input, pseudo_label

# 数据加载器（训练用）
def get_dataloader(image_dir, img_size, batch_size):
    dataset = SonarDataset(image_dir, img_size, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 视频帧预处理（推理用）
def preprocess_video_frames(frames, img_size):
    static_input = cv2.resize(frames[1], img_size)  # 中间帧作为静态输入
    dynamic_input = cv2.resize(np.stack(frames, axis=-1), img_size)
    static_input = torch.tensor(static_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # [1, 1, H, W]
    dynamic_input = torch.tensor(dynamic_input, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # [1, 3, H, W]
    return static_input, dynamic_input