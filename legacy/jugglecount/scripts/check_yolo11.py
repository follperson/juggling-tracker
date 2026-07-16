from ultralytics import YOLO
import sys

try:
    print("Attempting to load YOLO11n...")
    model = YOLO("yolo11n.pt")
    print("Success! YOLO11n loaded.")
    print(f"Model names: {model.names}")
except Exception as e:
    print(f"Failed to load YOLO11n: {e}")
    sys.exit(1)
