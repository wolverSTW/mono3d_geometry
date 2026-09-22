import torch
import torch.nn as nn
from models.backbone import YOLOv10Backbone
from models.head3d import Head3D

class Mono3DNetwork(nn.Module):
    """
    Main Monocular 3D Detection Network integration module.
    """
    def __init__(self, num_classes=5, config=None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = YOLOv10Backbone()
        self.head = Head3D(num_classes=num_classes)

    def forward(self, x):
        # 1. Feature Extraction (P3, P4, P5)
        features = self.backbone(x)
        
        # 2. 3D Detection Heads Predictions
        predictions = self.head(features)
        
        return predictions
