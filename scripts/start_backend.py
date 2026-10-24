import os
import sys
import uvicorn

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    print("=" * 60)
    print("Starting On-Device AI Reliability Monitor Server")
    print("Local URL: http://127.0.0.1:8000")
    print("API Docs:  http://127.0.0.1:8000/docs")
    print("Websocket: ws://127.0.0.1:8000/ws/monitor")
    print("=" * 60)
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
