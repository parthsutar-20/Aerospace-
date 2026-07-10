import cv2
import numpy as np
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.ml.inference import ml_engine, CLASS_NAMES, CLASS_COLORS_HEX

router = APIRouter()

# Confidence threshold below which detections are ignored
CONFIDENCE_THRESHOLD = 0.40


def _parse_outputs(outputs) -> list[dict]:
    """
    Stub post-processor for YOLOv8-seg ONNX outputs.

    The full decoder applies Non-Maximum Suppression and decodes proto-masks.
    Here we return a structured empty list so the WebSocket contract is stable
    and can be wired to a real decoder when weights are available.

    Expected YOLOv8-seg output layout
    ----------------------------------
    outputs[0]  shape: (1, 4 + NUM_CLASSES + 32, num_anchors)  → boxes + class scores + mask coefficients
    outputs[1]  shape: (1, 32, mask_h, mask_w)                 → prototype masks
    """
    # TODO: replace stub with full NMS + mask-decoding logic once weights land
    detections: list[dict] = []
    return detections


@router.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    print("[AeroVision-MRO] WebSocket client connected.")

    try:
        while True:
            # ── Receive Base64-encoded JPEG frame ────────────────────────────
            data = await websocket.receive_text()

            if data.startswith("data:image"):
                data = data.split(",")[1]

            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # ── Run inference ────────────────────────────────────────────────
            outputs, latency_ms = ml_engine.predict(frame)

            # ── Post-process ─────────────────────────────────────────────────
            defects = _parse_outputs(outputs) if outputs is not None else []

            # Enrich each detection with human-readable class metadata
            for det in defects:
                class_id = det.get("class_id", -1)
                det["class_name"] = CLASS_NAMES.get(class_id, "unknown")
                det["color"] = CLASS_COLORS_HEX.get(class_id, "#ffffff")

            # ── Respond ──────────────────────────────────────────────────────
            response = {
                "latency_ms": round(latency_ms, 3),
                # Each defect follows this schema:
                # {
                #   "class_id"   : int   (0-3),
                #   "class_name" : str   ("micro-crack" | "thermal-pitting" | "blade-erosion" | "corrosion"),
                #   "confidence" : float (0.0-1.0),
                #   "bbox"       : [x1, y1, x2, y2],  (pixel coords in original frame)
                #   "color"      : str   (hex colour for canvas rendering)
                # }
                "defects": defects,
                "num_classes": len(CLASS_NAMES),
                "class_map": CLASS_NAMES,
            }

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("[AeroVision-MRO] WebSocket client disconnected.")
    except Exception as e:
        print(f"[AeroVision-MRO] WebSocket error: {e}")
