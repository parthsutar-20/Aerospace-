import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLO

def export_and_quantize():
    print("Exporting model to ONNX...")
    
    # The path to the trained PyTorch model
    model_path = "../runs/segment/aerovision_mro/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using base yolov8n-seg.pt for demonstration.")
        model_path = "yolov8n-seg.pt"

    # Load the trained model
    model = YOLO(model_path)
    
    # Export to ONNX
    export_path = model.export(format="onnx", imgsz=640, optimize=True)
    print(f"Exported to {export_path}")
    
    # Quantize the model
    print("Quantizing ONNX model to INT8...")
    quantized_path = str(export_path).replace(".onnx", "_quantized.onnx")
    
    quantize_dynamic(
        str(export_path),
        quantized_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"Quantized model saved to: {quantized_path}")
    
if __name__ == "__main__":
    export_and_quantize()
