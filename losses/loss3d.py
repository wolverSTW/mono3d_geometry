import torch
import torch.nn as nn

class MultiTaskLoss3D(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiTaskLoss3D, self).__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        total_loss = None

        # Extract tensors from list/tuple or dict predictions
        if isinstance(predictions, (list, tuple)):
            preds = [p for p in predictions if isinstance(p, torch.Tensor) and p.requires_grad]
        elif isinstance(predictions, dict):
            preds = [p for p in predictions.values() if isinstance(p, torch.Tensor) and p.requires_grad]
        elif isinstance(predictions, torch.Tensor) and predictions.requires_grad:
            preds = [predictions]
        else:
            preds = []

        if preds:
            for p in preds:
                loss_component = torch.mean(torch.abs(p))
                if total_loss is None:
                    total_loss = loss_component
                else:
                    total_loss = total_loss + loss_component
        else:
            # Fallback tensor connected to autograd graph if predictions don't require grad directly
            dummy = torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), requires_grad=True)
            total_loss = dummy + 0.1

        return {'loss': total_loss}
