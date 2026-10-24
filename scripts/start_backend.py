import os
import sys
import uvicorn

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    port = int(os.getenv("MONITOR_PORT", os.getenv("PORT", "8000")))
    host = os.getenv("MONITOR_HOST", "127.0.0.1")
    print("=" * 60)
    print("Starting On-Device AI Reliability Monitor Server")
    print(f"Local URL: http://{host}:{port}")
    print(f"API Docs:  http://{host}:{port}/docs")
    print(f"Websocket: ws://{host}:{port}/ws/monitor")
    print("=" * 60)
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)
