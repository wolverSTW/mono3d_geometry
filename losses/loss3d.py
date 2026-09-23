import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricUncertaintyDepthLoss(nn.Module):
    """
    Computes Geometric Uncertainty Depth Loss.
    pred_depth shape: [B, 2, H, W] where channel 0 = depth_val, channel 1 = log_variance (log(sigma^2))
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_depth, target_depth):
        # pred_depth: [B, 2, H, W]
        depth_val = pred_depth[:, 0:1, :, :]
        log_var = pred_depth[:, 1:2, :, :]  # log(sigma^2)

        # Target depth resolution ကို matching ဖြစ်အောင် resample ပြုလုပ်ခြင်း
        if depth_val.shape[2:] != target_depth.shape[2:]:
            target_depth = F.interpolate(target_depth, size=depth_val.shape[2:], mode='nearest')

        # Laplace / Gaussian Likelihood with Uncertainty
        precision = torch.exp(-log_var)
        loss = precision * torch.abs(depth_val - target_depth) + log_var
        return loss.mean()


class MultiTaskLoss3D(nn.Module):
    """
    Multi-Task Loss for Lightweight 3D Object Detection:
    - Classification Loss (BCE / CrossEntropy)
    - 2D Box Loss (L1 / Smooth L1)
    - 3D Dimension Loss (L1)
    - Geometric Uncertainty Depth Loss
    - Orientation Multi-bin Loss
    """
    def __init__(self, loss_weights=None, num_classes=5, num_bins=12):
        super(MultiTaskLoss3D, self).__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        
        # Loss Weights setup
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'bbox2d': 1.0, 'dim3d': 1.0, 'depth_uncertainty': 1.0, 'orient': 1.0}
        self.weights = loss_weights

        # Sub-losses Initialization
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.L1Loss()
        self.depth_loss_fn = GeometricUncertaintyDepthLoss()

    def forward(self, predictions, targets):
        """
        predictions: List of dicts per scale output from Mono3DHead
        targets: Dictionary containing ground truth tensors ('cls', 'bbox2d', 'dim3d', 'depth', 'orient')
        """
        total_cls_loss = 0.0
        total_bbox2d_loss = 0.0
        total_dim3d_loss = 0.0
        total_depth_loss = 0.0
        total_orient_loss = 0.0

        num_scales = len(predictions) if isinstance(predictions, (list, tuple)) else 1
        scale_preds = predictions if isinstance(predictions, (list, tuple)) else [predictions]

        for pred in scale_preds:
            # 1. Depth & Uncertainty Loss
            if 'depth' in pred and 'depth' in targets:
                total_depth_loss += self.depth_loss_fn(pred['depth'], targets['depth'])

            # 2. Classification Loss
            if 'cls' in pred and 'cls' in targets:
                tgt_cls = targets['cls']
                if pred['cls'].shape[2:] != tgt_cls.shape[2:]:
                    tgt_cls = F.interpolate(tgt_cls, size=pred['cls'].shape[2:], mode='nearest')
                total_cls_loss += self.cls_loss_fn(pred['cls'], tgt_cls)

            # 3. 2D BBox Loss
            if 'bbox2d' in pred and 'bbox2d' in targets:
                tgt_b2d = targets['bbox2d']
                if pred['bbox2d'].shape[2:] != tgt_b2d.shape[2:]:
                    tgt_b2d = F.interpolate(tgt_b2d, size=pred['bbox2d'].shape[2:], mode='nearest')
                total_bbox2d_loss += self.reg_loss_fn(pred['bbox2d'], tgt_b2d)

            # 4. 3D Dimension Loss
            if 'dim3d' in pred and 'dim3d' in targets:
                tgt_d3d = targets['dim3d']
                if pred['dim3d'].shape[2:] != tgt_d3d.shape[2:]:
                    tgt_d3d = F.interpolate(tgt_d3d, size=pred['dim3d'].shape[2:], mode='nearest')
                total_dim3d_loss += self.reg_loss_fn(pred['dim3d'], tgt_d3d)

            # 5. Multi-bin Orientation Loss
            if 'orient' in pred and 'orient' in targets:
                tgt_ori = targets['orient']
                if pred['orient'].shape[2:] != tgt_ori.shape[2:]:
                    tgt_ori = F.interpolate(tgt_ori, size=pred['orient'].shape[2:], mode='nearest')
                total_orient_loss += self.reg_loss_fn(pred['orient'], tgt_ori)

        # Average across multi-scale outputs
        loss_cls = (total_cls_loss / num_scales) * self.weights.get('cls', 1.0)
        loss_bbox2d = (total_bbox2d_loss / num_scales) * self.weights.get('bbox2d', 1.0)
        loss_dim3d = (total_dim3d_loss / num_scales) * self.weights.get('dim3d', 1.0)
        loss_depth = (total_depth_loss / num_scales) * self.weights.get('depth_uncertainty', 1.0)
        loss_orient = (total_orient_loss / num_scales) * self.weights.get('orient', 1.0)

        total_loss = loss_cls + loss_bbox2d + loss_dim3d + loss_depth + loss_orient

        return {
            'loss': total_loss,
            'loss_cls': loss_cls,
            'loss_bbox2d': loss_bbox2d,
            'loss_dim3d': loss_dim3d,
            'loss_depth': loss_depth,
            'loss_orient': loss_orient
        }