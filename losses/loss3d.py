import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss3D(nn.Module):
    """
    Multi-task Loss Function for Monocular 3D Detection (Phase 3A).
    Computes Classification, 2D BBox, 3D Dimensions, and Orientation Losses.
    """
    def __init__(self, num_classes=5, num_bins=4, w_cls=1.0, w_bbox2d=1.0, w_dim3d=2.0, w_orient=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        
        self.w_cls = w_cls
        self.w_bbox2d = w_bbox2d
        self.w_dim3d = w_dim3d
        self.w_orient = w_orient

        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, predictions, targets):
        """
        predictions: Multi-scale predictions from Mono3DNetwork
        targets: List of label dicts for batch
        """
        total_cls_loss = torch.tensor(0.0, device=predictions[0]['cls'].device)
        total_bbox2d_loss = torch.tensor(0.0, device=predictions[0]['cls'].device)
        total_dim3d_loss = torch.tensor(0.0, device=predictions[0]['cls'].device)
        total_orient_loss = torch.tensor(0.0, device=predictions[0]['cls'].device)

        num_scales = len(predictions)

        # Simplified Target Matching & Multi-scale Loss Calculation for Pipeline Demonstration
        for pred in predictions:
            cls_pred = pred['cls']       # [B, Num_Classes, H, W]
            bbox2d_pred = pred['bbox2d'] # [B, 4, H, W]
            dim3d_pred = pred['dim3d']   # [B, 3, H, W]
            orient_pred = pred['orient'] # [B, Num_Bins * 2, H, W]

            # Dummy target alignment for loss optimization flow
            # In actual matching, positive anchor/grid assignment is computed here
            dummy_target_cls = torch.zeros((cls_pred.shape[0], cls_pred.shape[2], cls_pred.shape[3]), 
                                          dtype=torch.long, device=cls_pred.device)
            
            total_cls_loss += self.cls_loss_fn(cls_pred, dummy_target_cls)
            total_bbox2d_loss += self.smooth_l1(bbox2d_pred, torch.zeros_like(bbox2d_pred))
            total_dim3d_loss += self.smooth_l1(dim3d_pred, torch.zeros_like(dim3d_pred))
            total_orient_loss += self.smooth_l1(orient_pred, torch.zeros_like(orient_pred))

        # Average losses over scale outputs
        total_cls_loss /= num_scales
        total_bbox2d_loss /= num_scales
        total_dim3d_loss /= num_scales
        total_orient_loss /= num_scales

        # Total Weighted Multi-Task Loss
        total_loss = (self.w_cls * total_cls_loss +
                      self.w_bbox2d * total_bbox2d_loss +
                      self.w_dim3d * total_dim3d_loss +
                      self.w_orient * total_orient_loss)

        return {
            'loss': total_loss,
            'loss_cls': total_cls_loss.detach(),
            'loss_bbox2d': total_bbox2d_loss.detach(),
            'loss_dim3d': total_dim3d_loss.detach(),
            'loss_orient': total_orient_loss.detach()
        }
