import cv2
import numpy as np
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.ml.inference import ml_engine

router = APIRouter()

@router.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")
    try:
        while True:
            # Receive frame as text (Base64) or binary
            data = await websocket.receive_text()
            
            # Decode base64 image
            if data.startswith("data:image"):
                data = data.split(",")[1]
            
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Run inference
            outputs, latency_ms = ml_engine.predict(frame)
            
            # Dummy post-processing response (bounding box and mask coordinates)
            # In a real app, parse `outputs` into actual objects
            response = {
                "latency_ms": latency_ms,
                "defects": [
                    # Dummy data for demonstration
                    # {"type": "micro-crack", "bbox": [10, 10, 50, 50], "confidence": 0.89}
                ]
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
