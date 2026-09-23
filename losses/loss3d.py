import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricUncertaintyDepthLoss(nn.Module):
    """
    Computes Geometric Uncertainty Depth Loss stabilized for strictly positive loss range.
    pred_depth shape: [B, 2, H, W] where channel 0 = depth_val, channel 1 = log_variance
    """
    def __init__(self, log_var_min=-2.0, log_var_max=5.0):
        super().__init__()
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(self, pred_depth, target_depth):
        depth_val = pred_depth[:, 0:1, :, :]
        # Bound log_var safely
        log_var = torch.clamp(pred_depth[:, 1:2, :, :], min=self.log_var_min, max=self.log_var_max)

        if depth_val.shape[2:] != target_depth.shape[2:]:
            target_depth = F.interpolate(target_depth, size=depth_val.shape[2:], mode='nearest')

        # Absolute Error
        abs_err = torch.abs(depth_val - target_depth)
        
        # Heteroscedastic Aleatoric Loss formulation
        # Precision = exp(-s), loss = precision * abs_err + 0.5 * s
        precision = torch.exp(-log_var)
        loss = precision * abs_err + 0.5 * log_var + 1.0  # Constant shift (+1.0) ensures non-negative stability
        
        return F.relu(loss).mean()


class MultiTaskLoss3D(nn.Module):
    def __init__(self, loss_weights=None, num_classes=5, num_bins=12):
        super(MultiTaskLoss3D, self).__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'bbox2d': 1.0, 'dim3d': 1.0, 'depth_uncertainty': 1.0, 'orient': 1.0}
        self.weights = loss_weights

        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.L1Loss()
        self.depth_loss_fn = GeometricUncertaintyDepthLoss()

    def _extract_target_tensor(self, targets, key, default_shape, device):
        if isinstance(targets, dict) and key in targets:
            return targets[key]
        elif isinstance(targets, (list, tuple)) and len(targets) > 0:
            if isinstance(targets[0], dict) and key in targets[0]:
                tensors = [t[key] for t in targets if key in t and isinstance(t[key], torch.Tensor)]
                if len(tensors) > 0:
                    return torch.stack(tensors).to(device)
        return torch.zeros(default_shape, device=device)

    def forward(self, predictions, targets):
        scale_preds = predictions if isinstance(predictions, (list, tuple)) else [predictions]
        device = scale_preds[0]['cls'].device if 'cls' in scale_preds[0] else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        total_cls = torch.tensor(0.0, device=device)
        total_b2d = torch.tensor(0.0, device=device)
        total_d3d = torch.tensor(0.0, device=device)
        total_dep = torch.tensor(0.0, device=device)
        total_ori = torch.tensor(0.0, device=device)

        num_scales = len(scale_preds)

        for pred in scale_preds:
            B = pred['cls'].shape[0] if 'cls' in pred else 1

            if 'cls' in pred:
                tgt_cls = self._extract_target_tensor(targets, 'cls', pred['cls'].shape, device)
                if tgt_cls.shape != pred['cls'].shape:
                    tgt_cls = F.interpolate(tgt_cls, size=pred['cls'].shape[2:], mode='nearest') if tgt_cls.dim() == 4 else torch.zeros_like(pred['cls'])
                total_cls = total_cls + self.cls_loss_fn(pred['cls'], tgt_cls)

            if 'bbox2d' in pred:
                tgt_b2d = self._extract_target_tensor(targets, 'bbox2d', pred['bbox2d'].shape, device)
                if tgt_b2d.shape != pred['bbox2d'].shape:
                    tgt_b2d = F.interpolate(tgt_b2d, size=pred['bbox2d'].shape[2:], mode='nearest') if tgt_b2d.dim() == 4 else torch.zeros_like(pred['bbox2d'])
                total_b2d = total_b2d + self.reg_loss_fn(pred['bbox2d'], tgt_b2d)

            if 'dim3d' in pred:
                tgt_d3d = self._extract_target_tensor(targets, 'dim3d', pred['dim3d'].shape, device)
                if tgt_d3d.shape != pred['dim3d'].shape:
                    tgt_d3d = F.interpolate(tgt_d3d, size=pred['dim3d'].shape[2:], mode='nearest') if tgt_d3d.dim() == 4 else torch.zeros_like(pred['dim3d'])
                total_d3d = total_d3d + self.reg_loss_fn(pred['dim3d'], tgt_d3d)

            if 'depth' in pred:
                tgt_dep = self._extract_target_tensor(targets, 'depth', (B, 1, pred['depth'].shape[2], pred['depth'].shape[3]), device)
                total_dep = total_dep + self.depth_loss_fn(pred['depth'], tgt_dep)

            if 'orient' in pred:
                tgt_ori = self._extract_target_tensor(targets, 'orient', pred['orient'].shape, device)
                if tgt_ori.shape != pred['orient'].shape:
                    tgt_ori = F.interpolate(tgt_ori, size=pred['orient'].shape[2:], mode='nearest') if tgt_ori.dim() == 4 else torch.zeros_like(pred['orient'])
                total_ori = total_ori + self.reg_loss_fn(pred['orient'], tgt_ori)

        loss_cls = (total_cls / num_scales) * self.weights.get('cls', 1.0)
        loss_bbox2d = (total_b2d / num_scales) * self.weights.get('bbox2d', 1.0)
        loss_dim3d = (total_d3d / num_scales) * self.weights.get('dim3d', 1.0)
        loss_depth = (total_dep / num_scales) * self.weights.get('depth_uncertainty', 1.0)
        loss_orient = (total_ori / num_scales) * self.weights.get('orient', 1.0)

        total_loss = loss_cls + loss_bbox2d + loss_dim3d + loss_depth + loss_orient

        return {
            'loss': total_loss,
            'loss_cls': loss_cls,
            'loss_bbox2d': loss_bbox2d,
            'loss_dim3d': loss_dim3d,
            'loss_depth': loss_depth,
            'loss_orient': loss_orient
        }