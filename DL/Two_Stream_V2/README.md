# **一、双流模型设计的合理性分析**

## **1. 空间流与时间流的选择**

- **参考《**Two-Stream Convolutional Networks for Action Recognition in Videos**》的合理性**：
  双流网络（空间流+时间流）的设计本质上是为分离静态场景和动态目标特征，任务目标（抑制静态场景、检测动态目标）完全契合。

  - **空间流**：提取单帧图像的静态特征（如海底地形、岩石）。
  - **时间流**：提取光流或帧间差异中的动态特征（如鱼类运动轨迹）。
- **问题与改进**：

  - 声纳图像分辨率低、噪声多，可能无法有效捕捉声纳特有的纹理和运动模式。
  - **方案**：
    1. 使用轻量的自定义特征提取器（如浅层 CNN），减少过拟合风险。
    2. 在预训练模型基础上进行领域适配（Domain Adaptation），通过少量声纳数据微调。

---

# **二、损失函数设计的优化方向**

## **2. 直接的监督策略**

- **动态目标数量对比**：
  任务的核心是检测动态目标数量而非精确分割，可改用目标检测框架（如 YOLO 或 Faster R-CNN），直接输出目标位置和数量。

  - **优点**：更符合“计数”需求，计算量低。
  - **缺点**：无法实现像素级静态场景抑制。
- **像素级分割的改进方案**：

  - **简化损失函数**：使用 **加权二值交叉熵损失**，根据动态目标占比调整权重，例如：
    ```python
    class WeightedBCE(nn.Module):
        def __init__(self, pos_weight=10.0):
            super().__init__()
            self.pos_weight = pos_weight  # 动态目标权重

        def forward(self, pred, target):
            loss = - (self.pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
            return loss.mean()
    ```
  - **运动一致性约束**：在损失函数中加入时间连续性惩罚项，例如相邻帧预测掩膜的平滑性约束：
    ```python
    def temporal_consistency_loss(pred_current, pred_previous):
        return torch.mean(torch.abs(pred_current - pred_previous))
    ```


# **方案**

- **模型设计**：
  双流结构，但使用轻量的自定义 CNN，减少参数量。
- **损失函数**：
  使用 **加权二值交叉熵损失** + **时间一致性损失**，平衡静态抑制和动态检测。
- **数据增强**：
  添加声纳噪声和运动模糊，提升模型鲁棒性。

#### **2. 核心代码调整**

```python
# 自定义轻量双流模型
class LightweightTwoStream(nn.Module):
    def __init__(self):
        super().__init__()
        # 空间流（3层卷积）
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
      
        # 时间流（输入为光流图）
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
      
        # 融合与输出
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
  
    def forward(self, spatial_input, temporal_input):
        spatial_feat = self.spatial_stream(spatial_input)
        temporal_feat = self.temporal_stream(temporal_input)
        fused_feat = torch.cat([spatial_feat, temporal_feat], dim=1)
        mask = self.fusion(fused_feat)
        return mask
```

#### **3. 训练策略**

```
# 使用加权损失函数 + 时间一致性损失
criterion = WeightedBCE(pos_weight=10.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for i, (spatial, temporal, mask) in enumerate(train_loader):
        # 前向传播
        pred = model(spatial, temporal)
        loss = criterion(pred, mask)
  
        # 时间一致性损失（假设输入为连续帧）
        if i > 0:
            loss += 0.1 * temporal_consistency_loss(pred, prev_pred)
        prev_pred = pred.detach()
  
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### **六、预期效果**

通过上述优化，模型将更适配声纳视频的特性：

1. **静态场景抑制**：通过空间流学习背景特征，时间流捕捉运动变化。
2. **动态目标检测**：加权损失函数强化对亮色运动区域的关注。
3. **实时性**：
