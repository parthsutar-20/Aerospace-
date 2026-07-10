import time
import numpy as np
import onnxruntime as ort
import cv2

# ─── Official 4-Class Taxonomy ──────────────────────────────────────────────
# Class ID  | Class Name        | Description
# ----------|-------------------|--------------------------------------------
#     0     | micro-crack       | Hairline or visible cracks on blades
#     1     | thermal-pitting   | Small pits caused by heat damage
#     2     | blade-erosion     | Material loss from the blade edge or surface
#     3     | corrosion         | Oxidation or rust-like degradation
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "micro-crack",
    1: "thermal-pitting",
    2: "blade-erosion",
    3: "corrosion",
}

# Severity colour coding per class (BGR order for OpenCV, hex for frontend)
CLASS_COLORS_HEX = {
    0: "#f85149",   # micro-crack     → vivid red
    1: "#ff9a3c",   # thermal-pitting → amber
    2: "#e3b341",   # blade-erosion   → yellow-gold
    3: "#8b949e",   # corrosion       → steel grey
}

NUM_CLASSES = len(CLASS_NAMES)  # 4


class EdgeMLEngine:
    def __init__(self, model_path: str = "yolov8n-seg_quantized.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None

    def load_model(self):
        try:
            # Load the ONNX model using CPUExecutionProvider (or others if available)
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"[AeroVision-MRO] Loaded ONNX model: {self.model_path}")
            print(f"[AeroVision-MRO] Active defect classes ({NUM_CLASSES}): {list(CLASS_NAMES.values())}")
        except Exception as e:
            print(
                f"[AeroVision-MRO] Failed to load ONNX model. "
                f"Ensure '{self.model_path}' exists. Error: {e}"
            )

    def predict(self, frame: np.ndarray):
        """
        Run inference on a single BGR frame.

        Returns
        -------
        outputs : list[np.ndarray] | None
            Raw ONNX outputs (boxes + proto-masks for YOLOv8-seg).
        latency_ms : float
            Wall-clock inference time in milliseconds.
        """
        if self.session is None:
            return None, 0.0

        # ── Start timing ────────────────────────────────────────────────────
        start_time = time.perf_counter()

        # ── Pre-processing ──────────────────────────────────────────────────
        # YOLOv8 expects NCHW, RGB, values in [0, 1]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)                      # HWC → CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img /= 255.0

        # ── Inference ───────────────────────────────────────────────────────
        outputs = self.session.run(self.output_names, {self.input_name: img})

        # ── Stop timing ─────────────────────────────────────────────────────
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        # Note: downstream post-processing (NMS, mask decoding, class filtering)
        # is handled in api/ws.py so the engine stays stateless and reusable.
        return outputs, latency_ms


# Global singleton loaded once at FastAPI lifespan startup
ml_engine = EdgeMLEngine()
