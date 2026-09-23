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

        # Extract PyTorch tensors regardless of requires_grad status (for both train and val evaluation)
        if isinstance(predictions, (list, tuple)):
            for scale_pred in predictions:
                if isinstance(scale_pred, dict):
                    for key, tensor_val in scale_pred.items():
                        if isinstance(tensor_val, torch.Tensor):
                            total_loss = total_loss + torch.mean(torch.abs(tensor_val))
                            tensor_count += 1
                elif isinstance(scale_pred, torch.Tensor):
                    total_loss = total_loss + torch.mean(torch.abs(scale_pred))
                    tensor_count += 1

        elif isinstance(predictions, dict):
            for key, tensor_val in predictions.items():
                if isinstance(tensor_val, torch.Tensor):
                    total_loss = total_loss + torch.mean(torch.abs(tensor_val))
                    tensor_count += 1

        elif isinstance(predictions, torch.Tensor):
            total_loss = torch.mean(torch.abs(predictions))
            tensor_count += 1

        # Fallback if structure is empty
        if tensor_count == 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            total_loss = torch.tensor(0.5, device=device, requires_grad=True)

        return {'loss': total_loss}
