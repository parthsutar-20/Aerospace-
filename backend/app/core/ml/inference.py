import time
import numpy as np
import onnxruntime as ort
import cv2

class EdgeMLEngine:
    def __init__(self, model_path="yolov8n-seg_quantized.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        
    def load_model(self):
        try:
            # Load the ONNX model using CPUExecutionProvider (or others if available)
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"Loaded ONNX model: {self.model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model. Ensure {self.model_path} exists. Error: {e}")
            
    def predict(self, frame: np.ndarray):
        if not self.session:
            return None, 0.0
            
        # Start timing
        start_time = time.perf_counter()
        
        # Preprocessing: resize, normalize, format
        # YOLOv8 expects NCHW format, RGB, 1/255.0 scaled
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img /= 255.0
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: img})
        
        # Stop timing
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        # Note: We would typically decode outputs here (NMS, mask scaling).
        # For simplicity in this structure, we return the raw outputs and latency.
        return outputs, latency_ms

# Global engine instance to be used by the app
ml_engine = EdgeMLEngine()
