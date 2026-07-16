import torch
import ultralytics
print(f"Torch Version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")

try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print(f"YOLO Device: {model.device}")
except Exception as e:
    print(f"YOLO Check Error: {e}")
