import torch
import torch.nn as nn
from models.backbone import Mono3DBackbone
from models.head3d import Mono3DHead

class Mono3DNetwork(nn.Module):
    """
    Config-driven Monocular 3D Detection Network Architecture
    """
    def __init__(self, config=None, num_classes=5):
        super(Mono3DNetwork, self).__init__()
        
        # Config မပါပါက Default Settings ယူရန်
        if config is None:
            config = {}

        model_cfg = config.get('model', {})
        backbone_cfg = model_cfg.get('backbone', {})
        head_cfg = model_cfg.get('head', {})

        self.num_classes = model_cfg.get('num_classes', num_classes)
        in_channels = backbone_cfg.get('in_channels', 3)
        self.feature_channels = backbone_cfg.get('feature_channels', [128, 256, 512])

        # 1. Config-driven Backbone
        self.backbone = Mono3DBackbone(
            in_channels=in_channels, 
            feature_channels=self.feature_channels
        )

        # 2. Config-driven Multi-scale 3D Head
        self.head = Mono3DHead(
            in_channels=self.feature_channels, 
            num_classes=self.num_classes,
            use_scdown=head_cfg.get('use_scdown', True),
            use_c2fcib=head_cfg.get('use_c2fcib', True),
            use_psa=head_cfg.get('use_psa', True)
        )

    def forward(self, x):
        # Extract multi-scale features via backbone
        features = self.backbone(x)
        # Predict 2D boxes, classes, and 3D dimensions via multi-task head
        predictions = self.head(features)
        return predictions
