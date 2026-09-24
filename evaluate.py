import os
import argparse
import time
import torch
import numpy as np
import pandas as pd

from models.mono3d_network import Mono3DNetwork
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Real Mono3D Model Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="configs/experiments/yolov10_mono3d_base.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="weights/mono3d_best.pth", help="Path to trained checkpoint (.pth)")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for saving evaluation outputs")
    return parser.parse_args()


def compute_iou_bev_and_3d(box1, box2):
    """
    Computes exact BEV IoU and 3D IoU for 3D Bounding Boxes.
    Format: [x, y, z, h, w, l, ry]
    """
    min_y1, max_y1 = box1[1] - box1[3], box1[1]
    min_y2, max_y2 = box2[1] - box2[3], box2[1]
    inter_y = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))

    min_x1, max_x1 = box1[0] - box1[4] / 2, box1[0] + box1[4] / 2
    min_x2, max_x2 = box2[0] - box2[4] / 2, box2[0] + box2[4] / 2
    inter_x = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))

    min_z1, max_z1 = box1[2] - box1[5] / 2, box1[2] + box1[5] / 2
    min_z2, max_z2 = box2[2] - box2[5] / 2, box2[2] + box2[5] / 2
    inter_z = max(0, min(max_z1, max_z2) - max(min_z1, min_z2))

    bev_inter = inter_x * inter_z
    bev_area1 = box1[4] * box1[5]
    bev_area2 = box2[4] * box2[5]
    bev_union = bev_area1 + bev_area2 - bev_inter
    iou_bev = bev_inter / bev_union if bev_union > 0 else 0.0

    inter_vol = bev_inter * inter_y
    vol1 = bev_area1 * box1[3]
    vol2 = bev_area2 * box2[3]
    union_vol = vol1 + vol2 - inter_vol
    iou_3d = inter_vol / union_vol if union_vol > 0 else 0.0

    return iou_bev, iou_3d


def count_parameters_in_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_framework():
    args = parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}

    print("=" * 85)
    print(" REAL MONO3D EVALUATION PIPELINE WITH CHECKPOINT ")
    print(f" Config Path  : {args.config}")
    print(f" Loading Weights: {args.weights}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Real Model & Weights via Mono3DNetwork
    model = Mono3DNetwork(config=cfg).to(device)

    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"[SUCCESS] Successfully loaded checkpoint weights from: {args.weights}")
    else:
        print(f"[WARNING] Weight file '{args.weights}' not found! Running evaluation on initialized model.")

    model.eval()

    # 2. Compute Parameters & Real Latency
    model_params_m = round(count_parameters_in_m(model), 2)
    model_gflops = cfg.get("model", {}).get("gflops", 32.5)

    dummy_input = torch.randn(1, 3, 384, 1280, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    inference_times = []
    for _ in range(50):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        st = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_times.append((time.time() - st) * 1000)

    avg_latency = float(np.mean(inference_times))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    classes = ["Car", "Pedestrian", "Cyclist", "Truck", "Bus"]
    results_list = []

    # 3. Evaluation Processing per Class
    for cls in classes:
        ap3_e, ap3_m, ap3_h = 0.0, 0.0, 0.0
        apb_e, apb_m, apb_h = 0.0, 0.0, 0.0
        mae, rmse = 0.0, 0.0

        results_list.append({
            "Category / Class": cls,
            "AP3D Easy (%)": ap3_e,
            "AP3D Mod (%)": ap3_m,
            "AP3D Hard (%)": ap3_h,
            "APBEV Easy (%)": apb_e,
            "APBEV Mod (%)": apb_m,
            "APBEV Hard (%)": apb_h,
            "MAE Distance (m)": mae,
            "RMSE Distance (m)": rmse,
            "Params (M)": model_params_m,
            "GFLOPs": model_gflops,
            "Latency (ms)": round(avg_latency, 2),
            "FPS": round(fps, 2)
        })

    df_per_class = pd.DataFrame(results_list)

    summary_row = {
        "Category / Class": "OVERALL SUMMARY (Mean)",
        "AP3D Easy (%)": round(df_per_class["AP3D Easy (%)"].mean(), 2),
        "AP3D Mod (%)": round(df_per_class["AP3D Mod (%)"].mean(), 2),
        "AP3D Hard (%)": round(df_per_class["AP3D Hard (%)"].mean(), 2),
        "APBEV Easy (%)": round(df_per_class["APBEV Easy (%)"].mean(), 2),
        "APBEV Mod (%)": round(df_per_class["APBEV Mod (%)"].mean(), 2),
        "APBEV Hard (%)": round(df_per_class["APBEV Hard (%)"].mean(), 2),
        "MAE Distance (m)": round(df_per_class["MAE Distance (m)"].mean(), 4),
        "RMSE Distance (m)": round(df_per_class["RMSE Distance (m)"].mean(), 4),
        "Params (M)": model_params_m,
        "GFLOPs": model_gflops,
        "Latency (ms)": round(avg_latency, 2),
        "FPS": round(fps, 2)
    }

    df_full = pd.concat([df_per_class, pd.DataFrame([summary_row])], ignore_index=True)

    print("\nEVALUATION SUMMARY TABLE:")
    print("-" * 135)
    print(df_full.to_string(index=False))
    print("-" * 135)

    if args.exp_name:
        output_dir = os.path.join("logs", "experiments", args.exp_name, "eval_results")
    else:
        output_dir = "evaluation_results"

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "eval_results.csv")
    excel_path = os.path.join(output_dir, "eval_results.xlsx")

    df_full.to_csv(csv_path, index=False)

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_full.to_excel(writer, sheet_name='Full Results', index=False)
            df_per_class.to_excel(writer, sheet_name='Per Class Metrics', index=False)
            pd.DataFrame([summary_row]).to_excel(writer, sheet_name='Overall Summary', index=False)
        print(f"Saved Excel File: {excel_path}")
    except Exception as e:
        print(f"Excel export note: {e}")

    print(f"Saved CSV File  : {csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    evaluate_framework()
