import torch
import torch.nn as nn
from models.backbone import BackboneFactory
from models.head3d import HeadFactory

class Mono3DNetwork(nn.Module):
    """
    Config-driven Lightweight Monocular 3D Detection Network Architecture.
    Integrates Dynamic Backbone Factory and Multi-task Head Factory with Depth Gate & Geometric Uncertainty.
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
        weights_path = backbone_cfg.get('weights_path', None)
        backbone_type = backbone_cfg.get('type', 'yolov10')
        
        # 1. Config-driven Backbone Construction via Factory Pattern
        self.backbone = BackboneFactory.build(
            backbone_type=backbone_type,
            in_channels=in_channels,
            weights_path=weights_path
        )

        # Backbone မှ ထွက်လာသော feature_channels ကို dynamic ယူရန်
        self.feature_channels = getattr(self.backbone, 'out_channels', backbone_cfg.get('feature_channels', [256, 512, 1024]))

        # 2. Config-driven Multi-scale 3D Head Construction via Factory Pattern
        head_type = head_cfg.get('type', 'head3d')
        self.head = HeadFactory.build(
            head_type=head_type,
            in_channels=self.feature_channels, 
            num_classes=self.num_classes,
            num_bins=head_cfg.get('num_bins', 12),
            use_depth_gate=head_cfg.get('use_depth_gate', True),
            use_scdown=head_cfg.get('use_scdown', True),
            use_c2fcib=head_cfg.get('use_c2fcib', True),
            use_psa=head_cfg.get('use_psa', True)
        )

    def forward(self, x):
        # Extract multi-scale features via backbone
        features = self.backbone(x)
        # Predict 2D boxes, classes, depth uncertainty, and 3D dimensions via multi-task head
        predictions = self.head(features)
        return predictions