"""
PyTorch-based ArcFace inference for GPU acceleration
Fallback when ONNX Runtime CUDA provider fails
"""
import torch
import torch.nn.functional as F
import onnx
from onnx2pytorch import ConvertModel
import numpy as np

class ArcFaceTorch:
    def __init__(self, onnx_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load ONNX model and convert to PyTorch
        onnx_model = onnx.load(onnx_path)
        self.model = ConvertModel(onnx_model).to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def run_batch(self, input_array):
        """Run inference on batch of preprocessed images."""
        input_tensor = torch.from_numpy(input_array).to(self.device)
        output = self.model(input_tensor)
        return output.cpu().numpy()
