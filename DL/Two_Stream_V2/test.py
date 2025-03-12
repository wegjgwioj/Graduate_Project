import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

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

# 数据预处理
def get_transforms():
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert HWC image to CHW tensor
        transforms.Resize((224, 224)),
    ])
    return image_transform

# 加载模型
def load_model(model_path):
    model = LightweightTwoStream()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

# 推理函数
def inference(model, device, spatial_frame, temporal_frame):
    transform = get_transforms()
    spatial_tensor = transform(spatial_frame).unsqueeze(0).to(device)
    temporal_tensor = transform(temporal_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(spatial_tensor, temporal_tensor)
        pred = (output > 0.5).float().cpu().numpy().squeeze()
    return pred

# 计算 F1 和 accuracy
# 计算 F1 和 accuracy
def calculate_metrics(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    denominator = tp + fp + fn + tn
    # 检查分母是否为零
    if denominator == 0:
        accuracy = 0.0
    else:
        accuracy = (tp + tn) / denominator

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return accuracy, f1

# 主函数
if __name__ == "__main__":
    model_path = "sonar_dynamic_detection.pth"
    model, device = load_model(model_path)

    # 有两个视频流，一个空间流和一个时间流
    spatial_video = cv2.VideoCapture("spatial_video.mp4") # 空间流视频 灰度
    temporal_video = cv2.VideoCapture("temporal_video.avi")# 时间流视频  光流

    all_preds = []
    all_labels = []

    frame_index = 0  # 新增：记录当前帧的索引

    while True:
        ret_spatial, spatial_frame = spatial_video.read()
        ret_temporal, temporal_frame = temporal_video.read()

        if not ret_spatial or not ret_temporal:
            break

        spatial_frame = cv2.cvtColor(spatial_frame, cv2.COLOR_BGR2GRAY)
        temporal_frame = cv2.cvtColor(temporal_frame, cv2.COLOR_BGR2GRAY)

        # 修改：根据当前帧的索引读取对应的标签图片
        label_filename = f"label_frame_{frame_index:03d}.png"
        label = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"Warning: Missing label image for frame {frame_index}")
            #break
            continue
        label = (label > 0).astype(np.float32)

        pred = inference(model, device, spatial_frame, temporal_frame)

        all_preds.append(pred)
        all_labels.append(label)

        cv2.imshow('Prediction', (pred * 255).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1  # 新增：更新帧索引

    spatial_video.release()
    temporal_video.release()
    cv2.destroyAllWindows()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy, f1 = calculate_metrics(all_preds, all_labels)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")