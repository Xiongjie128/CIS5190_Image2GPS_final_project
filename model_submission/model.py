import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

# normalization stats computed on training split
LAT_MEAN, LAT_STD = 39.951684, 0.000666
LON_MEAN, LON_STD = -75.191403, 0.000665


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = convnext_tiny(weights=None)
        self.backbone = base.features
        self.pool = nn.Sequential(
            base.avgpool,
            base.classifier[0],   # LayerNorm2d
            nn.Flatten(1),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        out = self.head(x)
        # convert from normalized coords back to raw degrees
        lat = out[:, 0] * LAT_STD + LAT_MEAN
        lon = out[:, 1] * LON_STD + LON_MEAN
        return torch.stack([lat, lon], dim=1)

    def predict(self, x):
        self.eval()
        if isinstance(x, list):
            x = torch.stack(x)
        with torch.no_grad():
            return self.forward(x)
