import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np

# 自定义数据集类（支持动态目标掩膜和时间连续性）
class SonarDataset(Dataset):
    def __init__(self, spatial_dir, temporal_dir, mask_dir, transform=None):
        self.spatial_dir = spatial_dir
        self.temporal_dir = temporal_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(spatial_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载单帧图像（空间流输入）
        spatial_path = os.path.join(self.spatial_dir, self.image_files[idx])
        spatial_image = cv2.imread(spatial_path, cv2.IMREAD_GRAYSCALE)
        if spatial_image is None:
            raise FileNotFoundError(f"File not found: {spatial_path}")
        
        # 加载光流图（时间流输入）
        temporal_path = os.path.join(self.temporal_dir, self.image_files[idx])
        temporal_image = cv2.imread(temporal_path, cv2.IMREAD_GRAYSCALE)
        if temporal_image is None:
            raise FileNotFoundError(f"File not found: {temporal_path}")
        
        # 加载动态目标掩膜（标签）
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace(".png", "_mask.png"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"File not found: {mask_path}")

        # 转换为RGB（兼容模型输入）
        spatial_image = cv2.cvtColor(spatial_image, cv2.COLOR_GRAY2RGB)
        temporal_image = cv2.cvtColor(temporal_image, cv2.COLOR_GRAY2RGB)
        
        # 数据预处理
        if self.transform:
            spatial_image = self.transform(spatial_image)
            temporal_image = self.transform(temporal_image)
        if self.transform:#处理mask，确保形状一致
            mask_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: (x > 0).float())
            ])
            mask = mask_transform(mask)
        else :
            mask = transforms.ToTensor()(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0) #  
        
        return spatial_image, temporal_image, mask

# 轻量级双流模型（针对声纳数据优化）
class LightweightTwoStream(nn.Module):
    def __init__(self):
        super().__init__()
        # 空间流（提取静态特征）
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 输入通道3（RGB），输出16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样到112x112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样到56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 时间流（提取动态特征）
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 特征融合与输出
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),  # 融合空间和时间特征（64+64=128）
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 上采样到224x224
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出动态目标概率图
        )

    def forward(self, spatial_input, temporal_input):
        # 空间流特征
        spatial_feat = self.spatial_stream(spatial_input)  # (B,64,56,56)
        # 时间流特征
        temporal_feat = self.temporal_stream(temporal_input)  # (B,64,56,56)
        # 特征融合
        fused_feat = torch.cat([spatial_feat, temporal_feat], dim=1)  # (B,128,56,56)
        # 生成掩膜
        mask = self.fusion(fused_feat)  # (B,1,224,224)
        return mask
    
#加权损失函数与数据增强

# 加权二值交叉熵损失（动态目标权重更高）
class WeightedBCE(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight  # 动态目标的权重（根据数据集调整）

    def forward(self, pred, target):
        loss = - (self.pos_weight * target * torch.log(pred + 1e-7) + 
                 (1 - target) * torch.log(1 - pred + 1e-7))
        return loss.mean()

# 数据增强（模拟声纳特性）
class SonarAugmentation:
    def __init__(self):
        self.noise_intensity = 0.05
        self.blur_kernel = 5

    def __call__(self, img_tensor):
        # Convert the tensor to a numpy array and transpose it to HWC format
        img = img_tensor.numpy().transpose(1, 2, 0)

        # Add speckle noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(*img.shape) * self.noise_intensity
            img = np.clip(img + noise, 0, 1)
        
        # Add motion blur
        if np.random.rand() < 0.5:
            img = cv2.GaussianBlur(img, (self.blur_kernel, self.blur_kernel), 0)
        
        # Ensure the output is in float32 format
        return img.astype(np.float32)

# 数据预处理
def get_transforms():
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert HWC image to CHW tensor
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: torch.from_numpy(SonarAugmentation()(x).transpose(2, 0, 1))),  # Apply augmentation and convert back to tensor
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: (x > 0).float())
    ])
    return image_transform, mask_transform



# 3 训练 （包含时间一致性约束）
def train(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    prev_pred = None  # 用于时间一致性损失
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (spatial, temporal, mask) in enumerate(train_loader):
            # 检查
           # print(spatial.dtype, temporal.dtype, mask.dtype)  # 应为 torch.float32
            spatial = spatial.to(device)
            temporal = temporal.to(device)
            mask = mask.to(device)
            
            # 前向传播
            pred = model(spatial, temporal)
            loss = criterion(pred, mask)
            
            # 时间一致性损失（相邻帧预测平滑性）
            if batch_idx > 0 and prev_pred is not None:
                # 截取 prev_pred 使其批次大小与 pred 相同
                min_batch_size = min(pred.size(0), prev_pred.size(0))
                pred_batch = pred[:min_batch_size]
                prev_pred_batch = prev_pred[:min_batch_size]
                temporal_loss = torch.mean(torch.abs(pred_batch - prev_pred_batch))
                loss += 0.1 * temporal_loss  # 权重可调
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录当前预测
            prev_pred = pred.detach()
            running_loss += loss.item()
        
        # 打印训练信息
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 主函数
if __name__ == "__main__":
    # 数据加载
    image_transform, mask_transform = get_transforms()
    train_dataset = SonarDataset(
        spatial_dir="dataset/train/spatial",
        temporal_dir="dataset/train/temporal",
        mask_dir="dataset/train/masks",
        transform=image_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 初始化模型与优化器
    model = LightweightTwoStream()
    criterion = WeightedBCE(pos_weight=5.0)  # 根据数据集调整pos_weight
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, criterion, optimizer, device, num_epochs=100)

    # 保存模型
    torch.save(model.state_dict(), "sonar_dynamic_detection.pth")