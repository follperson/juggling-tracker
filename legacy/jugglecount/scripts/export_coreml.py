from ultralytics import YOLO

def export_model(model_name):
    print(f"Loading {model_name}...")
    model = YOLO(f'{model_name}.pt')
    
    print(f"Exporting {model_name} to CoreML (nms=True)...")
    # nms=True integrates Non-Max Suppression into the CoreML model
    model.export(format='coreml', nms=True)
    
    print(f"Export Complete: {model_name}.mlpackage")

if __name__ == "__main__":
    for m in ["yolo11l", "yolo11x"]:
        export_model(m)
