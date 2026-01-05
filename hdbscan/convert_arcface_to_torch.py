#!/usr/bin/env python3
"""Convert ArcFace ONNX to PyTorch for GPU acceleration"""
import torch
import onnx
from onnx2torch import convert
import os

models_dir = "analysis_pipeline/models"
onnx_path = os.path.join(models_dir, "arcface.onnx")
torch_path = os.path.join(models_dir, "arcface_torch.pt")

print(f"Loading ONNX model from {onnx_path}...")
onnx_model = onnx.load(onnx_path)

print("Converting to PyTorch...")
torch_model = convert(onnx_model)

print("Saving PyTorch model...")
torch.save(torch_model.state_dict(), torch_path)

print(f"✅ Saved to {torch_path}")
