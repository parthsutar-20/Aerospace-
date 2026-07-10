import os
from ultralytics import YOLO

def train_model():
    print("Starting YOLOv8 segmentation model training...")
    
    # Load a pretrained YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")
    
    # Dataset YAML path
    # We assume there's a dataset.yaml configured for turbine blade defects
    dataset_path = "dataset.yaml"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset config {dataset_path} not found. Please create one.")
        print("Expected format:")
        print("  path: ./data")
        print("  train: images/train")
        print("  val:   images/val")
        print("  nc: 4")
        print("  names:")
        print("    0: micro-crack       # Hairline or visible cracks on blades")
        print("    1: thermal-pitting   # Small pits caused by heat damage")
        print("    2: blade-erosion     # Material loss from the blade edge or surface")
        print("    3: corrosion         # Oxidation or rust-like degradation")
        return

    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name="aerovision_mro",
        device="cpu"  # Change to "cuda" if running on a GPU
    )
    
    print("Training complete. Results saved in runs/segment/aerovision_mro")

if __name__ == "__main__":
    train_model()
