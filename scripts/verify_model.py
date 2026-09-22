import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mono3d_network import Mono3DNetwork

def test_model_forward():
    print("[MODEL CHECK] Initializing Mono3DNetwork Architecture...")
    model = Mono3DNetwork(num_classes=5)
    model.eval()

    # Dummy Image Input Batch: [Batch Size 2, Channels 3, Height 384, Width 1280]
    dummy_input = torch.randn(2, 3, 384, 1280)
    print(f"[MODEL CHECK] Input Image Shape: {dummy_input.shape}")

    with torch.no_grad():
        preds = model(dummy_input)

    print("\n--- Multi-Scale Multi-Head Prediction Output Shapes ---")
    scales = ["P3 (Stride 8)", "P4 (Stride 16)", "P5 (Stride 32)"]
    for idx, pred in enumerate(preds):
        print(f"[{scales[idx]}]")
        print(f"  - Class Head     : {pred['cls'].shape}")
        print(f"  - 2D BBox Head   : {pred['bbox2d'].shape}")
        print(f"  - 3D Dim Head    : {pred['dim3d'].shape}")
        print(f"  - Orientation Head: {pred['orient'].shape}")

    print("\n[SUCCESS] Phase 3 Model Architecture verification passed!")

if __name__ == "__main__":
    test_model_forward()
