import os
import cv2
import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.api import api_router, decode_image_bytes
from .image.metrics import analyze_image_quality
from .image.classifier import get_classifier
from .audio.metrics import analyze_audio_quality, parse_audio_data
from .audio.classifier import get_audio_classifier
from .monitoring.engine import get_monitoring_engine
from .storage.database import init_db

app = FastAPI(
    title="On-Device AI Reliability Monitor API",
    description="Real-time on-device reliability monitoring layer for visual, acoustic, and LLM perception.",
    version="1.0.0"
)

# Enable CORS for local desktop frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(api_router, prefix="/api")

# Initialize database
@app.on_event("startup")
async def startup_event():
    init_db()

# WebSocket endpoint for real-time camera & microphone monitoring
@app.websocket("/ws/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = get_monitoring_engine()
    classifier = get_classifier()
    audio_classifier = get_audio_classifier()

    try:
        while True:
            data_text = await websocket.receive_text()
            message = json.loads(data_text)
            msg_type = message.get("type")

            if msg_type == "video_frame":
                frame_b64 = message.get("frame_base64", "")
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",")[1]
                frame_bytes = base64.b64decode(frame_b64)
                img = decode_image_bytes(frame_bytes)

                # Process image
                metrics = analyze_image_quality(img)
                preds = classifier.predict(img, top_k=3)
                res = engine.process_image_frame(metrics, preds)

                overall = engine.get_system_overall_status()

                await websocket.send_json({
                    "type": "image_update",
                    "metrics": metrics.dict(),
                    "predictions": preds.dict(),
                    "reliability_score": res["reliability_score"],
                    "risk_score": res["risk_score"],
                    "status": res["status"],
                    "raw_status": res.get("raw_status", "normal"),
                    "warnings": [w.dict() for w in res["warnings"]],
                    "deviations": res["deviations"],
                    "overall": overall.dict()
                })

            elif msg_type == "audio_chunk":
                audio_b64 = message.get("audio_base64", "")
                if "," in audio_b64:
                    audio_b64 = audio_b64.split(",")[1]
                audio_bytes = base64.b64decode(audio_b64)

                samples, sr = parse_audio_data(audio_bytes)
                metrics = analyze_audio_quality(samples, sr)
                cond = audio_classifier.classify(
                    samples, sr,
                    peak_amp=metrics.peak_amplitude,
                    clipping_pct=metrics.clipping_pct,
                    rms_dbfs=metrics.rms_dbfs,
                    snr_db=metrics.snr_db,
                    spectral_centroid=metrics.spectral_centroid_hz
                )
                res = engine.process_audio_chunk(metrics, cond)
                overall = engine.get_system_overall_status()

                await websocket.send_json({
                    "type": "audio_update",
                    "metrics": metrics.dict(),
                    "classification": cond.dict(),
                    "reliability_score": res["reliability_score"],
                    "risk_score": res["risk_score"],
                    "status": res["status"],
                    "warnings": [w.dict() for w in res["warnings"]],
                    "deviations": res["deviations"],
                    "overall": overall.dict()
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

# Serve frontend static assets if dist exists
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
