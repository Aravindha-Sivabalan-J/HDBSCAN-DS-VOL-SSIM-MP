"""
RetinaFace detector using PyTorch
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead

class RetinaFaceDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create RetinaNet model with ResNet50 backbone
        self.model = retinanet_resnet50_fpn(pretrained=False, num_classes=2)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
    def detect_faces(self, img, conf_threshold=0.5):
        """Detect faces in image"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img_tensor = img_tensor / 255.0
        
        with torch.no_grad():
            try:
                predictions = self.model(img_tensor)
                
                if len(predictions) == 0 or len(predictions[0]['boxes']) == 0:
                    return None
                
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                
                # Filter by confidence
                mask = scores > conf_threshold
                if mask.sum() == 0:
                    return None
                
                boxes = boxes[mask]
                scores = scores[mask]
                
                # NMS
                keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.4)
                boxes = boxes[keep.numpy()]
                
                return boxes
            except Exception as e:
                print(f"Detection error: {e}")
                return None
