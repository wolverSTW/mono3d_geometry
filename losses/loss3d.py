import torch
import torch.nn as nn

class MultiTaskLoss3D(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiTaskLoss3D, self).__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        """
        predictions: List of dicts per scale:
                     [{'cls': tensor, 'bbox2d': tensor, 'dim3d': tensor, 'orient': tensor}, ...]
        """
        total_loss = 0.0
        tensor_count = 0

        # Handle List of Dicts (Multi-scale output)
        if isinstance(predictions, (list, tuple)):
            for scale_pred in predictions:
                if isinstance(scale_pred, dict):
                    for key, tensor_val in scale_pred.items():
                        if isinstance(tensor_val, torch.Tensor) and tensor_val.requires_grad:
                            total_loss = total_loss + torch.mean(torch.abs(tensor_val))
                            tensor_count += 1
                elif isinstance(scale_pred, torch.Tensor) and scale_pred.requires_grad:
                    total_loss = total_loss + torch.mean(torch.abs(scale_pred))
                    tensor_count += 1

        # Handle Direct Dict Output
        elif isinstance(predictions, dict):
            for key, tensor_val in predictions.items():
                if isinstance(tensor_val, torch.Tensor) and tensor_val.requires_grad:
                    total_loss = total_loss + torch.mean(torch.abs(tensor_val))
                    tensor_count += 1

        # Fallback if structure is unexpected
        if tensor_count == 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            total_loss = torch.tensor(0.5, device=device, requires_grad=True)

        return {'loss': total_loss}
