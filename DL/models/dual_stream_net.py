import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from efficientnet_pytorch import EfficientNet

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DualStreamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_encoder = mobilenet_v3_small(pretrained=True).features[:4]
        self.dynamic_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.fusion = nn.Sequential(
            SEModule(112 + 24),
            nn.Conv2d(112 + 24, 64, kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU6(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU6(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, static_x, dynamic_x):
        s_feat = self.static_encoder(static_x)
        d_feat = self.dynamic_encoder(dynamic_x)
        fused = self.fusion(torch.cat([s_feat, d_feat], dim=1))
        return torch.sigmoid(self.decoder(fused))