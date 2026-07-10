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
        print("train: ./data/images/train")
        print("val: ./data/images/val")
        print("nc: 3")
        print("names: ['micro-crack', 'thermal-pitting', 'blade-erosion']")
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
