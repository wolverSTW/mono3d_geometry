import torch
import torch.nn as nn

class Head3D(nn.Module):
    """
    3D Object Detection Head for Multi-Scale Features (P3, P4, P5).
    Predicts: Class Logits, 2D BBox Offset, 3D Dimensions (Residual), Orientation (Multi-bin Alpha).
    """
    def __init__(self, num_classes=5, num_bins=4, in_channels=[256, 512, 1024]):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        
        self.shared_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU()
            ) for c in in_channels
        ])

        # Prediction Heads
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.bbox2d_head = nn.Conv2d(256, 4, kernel_size=1)             # [dx, dy, dw, dh]
        self.dim3d_head = nn.Conv2d(256, 3, kernel_size=1)              # [residual_h, residual_w, residual_l]
        self.orient_head = nn.Conv2d(256, num_bins * 2, kernel_size=1)  # Bin classification + offset

    def forward(self, feats):
        outputs = []
        for feat, conv in zip(feats, self.shared_convs):
            x = conv(feat)
            
            cls_out = self.cls_head(x)
            bbox2d_out = self.bbox2d_head(x)
            dim3d_out = self.dim3d_head(x)
            orient_out = self.orient_head(x)

            outputs.append({
                'cls': cls_out,
                'bbox2d': bbox2d_out,
                'dim3d': dim3d_out,
                'orient': orient_out
            })
        return outputs
