import torch
import torch.nn as nn

class LightweightDepthGate(nn.Module):
    """
    Lightweight Spatial-Channel Depth Gate Module.
    Uses predicted depth map as an attention mask to modulate feature representation.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.SiLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature, predicted_depth):
        # predicted_depth: [B, 1, H, W]
        weight = self.gate(predicted_depth)
        return feature * weight + feature  # Lightweight Residual Connection


class Head3D(nn.Module):
    """
    Lightweight 3D Object Detection Head for Multi-Scale Features (P3, P4, P5).
    Predicts:
      - Class Logits
      - 2D BBox Offset [dx, dy, dw, dh]
      - 3D Dimensions [residual_h, residual_w, residual_l]
      - Depth & Uncertainty [depth_value, log_variance]
      - Orientation [bin_cls, bin_offset]
    """
    def __init__(self, num_classes=5, num_bins=12, in_channels=[256, 512, 1024], use_depth_gate=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.use_depth_gate = use_depth_gate
        
        if in_channels is None:
            in_channels = [256, 512, 1024]
            
        self.shared_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU()
            ) for c in in_channels
        ])

        # Lightweight Depth Gate Module
        if self.use_depth_gate:
            self.depth_gates = nn.ModuleList([
                LightweightDepthGate(256) for _ in in_channels
            ])

        # Prediction Heads
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.bbox2d_head = nn.Conv2d(256, 4, kernel_size=1)               # [dx, dy, dw, dh]
        self.dim3d_head = nn.Conv2d(256, 3, kernel_size=1)                # [residual_h, residual_w, residual_l]
        self.depth_head = nn.Conv2d(256, 2, kernel_size=1)                # [depth_val, log_variance] (Geometric Uncertainty)
        self.orient_head = nn.Conv2d(256, num_bins * 2, kernel_size=1)    # Multi-bin classification + offset

    def forward(self, feats):
        outputs = []
        for i, (feat, conv) in enumerate(zip(feats, self.shared_convs)):
            x = conv(feat)
            
            # Predict Depth & Uncertainty first
            depth_out = self.depth_head(x)  # Shape: [B, 2, H, W]
            depth_val = depth_out[:, 0:1, :, :]  # Shape: [B, 1, H, W]

            # Apply Depth-Aware Gate Attention if enabled
            if self.use_depth_gate:
                x = self.depth_gates[i](x, depth_val)

            cls_out = self.cls_head(x)
            bbox2d_out = self.bbox2d_head(x)
            dim3d_out = self.dim3d_head(x)
            orient_out = self.orient_head(x)

            outputs.append({
                'cls': cls_out,
                'bbox2d': bbox2d_out,
                'dim3d': dim3d_out,
                'depth': depth_out,       # [depth_val, log_variance]
                'orient': orient_out
            })
        return outputs

# Alias to support 'Mono3DHead' imports in Mono3DNetwork
Mono3DHead = Head3D

class HeadFactory:
    """
    Factory Class to dynamically build 3D Detection Heads based on Config YAML
    """
    @staticmethod
    def build(head_type="head3d", num_classes=5, num_bins=12, in_channels=[256, 512, 1024], use_depth_gate=True, **kwargs):
        head_type = head_type.lower()
        
        if head_type in ["head3d", "mono3dhead", "default"]:
            return Head3D(
                num_classes=num_classes,
                num_bins=num_bins,
                in_channels=in_channels,
                use_depth_gate=use_depth_gate,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported head type: '{head_type}'. Available: ['head3d']")