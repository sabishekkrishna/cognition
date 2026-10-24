# On-Device AI Reliability Monitor

An autonomous, local-only desktop-first web application that monitors locally running AI perception models and Large Language Models (LLMs) to detect conditions that make inputs or outputs unreliable.

All inference, monitoring, image processing, audio processing, and data storage occur **strictly locally on the user's machine** with zero remote telemetry and zero external API dependencies.

---

## Key Capabilities

- **Phase 1: Image Quality & Visual Reliability**
  - Quantitative OpenCV metrics: Luminance Brightness, Sharpness (Laplacian variance), Dynamic Contrast, Color Balance Disparity, Immerkaer Fast Noise Estimator, and Exposure Saturation/Clipping.
  - Local MobileNetV3-Small classifier: Top-k predictions, class probabilities, Shannon prediction entropy, top-2 margin, and intermediate 576-dimensional feature embeddings.
  - Robust Baseline Profile Calibration: Medians, Median Absolute Deviation (MAD), and 5th-95th percentile distributions.
  - Normalized **Image Reliability Score (0–100)** and **Image Risk Score (0–100)**.
  - Interactive Synthetic Degradation Test Panel: Inject darkness/brightness, Gaussian blur, colour tint, sensor noise, and contrast reduction.

- **Phase 2: Audio Quality & Speech Reliability**
  - Acoustic signal measurements: RMS Loudness (dBFS), Peak Amplitude, Clipping Percentage, Background Noise Floor, Estimated SNR (dB), Energy & ZCR Speech VAD, Spectral Centroid, and 85% Spectral Rolloff.
  - Compact 2D CNN Acoustic Condition Classifier: Identifies `Clean Speech`, `Background Noise`, `Wind / Low-Frequency Noise`, `Mechanical Hum`, `Silence / Static`, and `Clipped / Distorted`.
  - Audio Baseline Calibration and **Audio Reliability Score (0–100)**.
  - Local test mode with controllable volume scaling, noise injection (white, pink, low rumble), and clipping saturation.

- **Phase 3: Live Camera, Audio, & Foreground LLM Monitoring**
  - Real-time Webcam streaming via WebSockets with configurable sampling frequency (1, 2, or 5 FPS) to conserve CPU.
  - Exponential moving average smoothing and a 3-consecutive-sample hysteresis filter to eliminate transient false alarms.
  - Extensible Local LLM Provider Architecture:
    - **Ollama REST API** (`http://127.0.0.1:11434`)
    - **OpenAI-Compatible Local Servers** (LM Studio `http://127.0.0.1:1234/v1`, llama.cpp `http://127.0.0.1:8080/v1`)
    - **Offline Local Demo Provider** (Synthetic local mock for evaluation when daemons are offline)
  - Response Monitoring Layer: Flags responses coupled to degraded inputs, detects explicit model uncertainty phrases, and records human evaluation feedback (`correct`, `incorrect`, `uncertain`).

- **Datasets & Model Management**
  - Local dataset importer and stratified train/val/test split visualizer.
  - Pre-packaged "Industrial Component Reliability Benchmark" dataset.
  - Local MobileNetV3 transfer learning and fine-tuning on CPU.
  - Checkpoint versioning and safe single-click rollback to known-good pretrained base weights.

- **Event History & Settings**
  - Full audit trail filterable by subsystem (`image`, `audio`, `llm`, `system`) and severity (`normal`, `caution`, `warning`).
  - Export audit trail as JSON.
  - Configurable sensitivity thresholds and data retention policies.

---

## Architecture Overview

```
User Input (Image / Audio / Camera)
         │
         ▼
Input Quality Analysis (OpenCV / Signal Processing)
         │
         ▼
Local Perception Model (MobileNetV3 / Acoustic CNN)
         │
         ▼
Unified Monitoring Engine (Robust Baseline Comparison, Smoothing, Hysteresis)
         │
         ▼
Local LLM Guard (Input Hazard Coupling & Uncertainty Detection)
         │
         ▼
Response & Warning Dashboard (Reliability 0-100 & Risk 0-100)
```

For detailed mathematical formulations and architectural diagrams, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Technology Stack

- **Backend**: Python 3.12, FastAPI, OpenCV, NumPy, SciPy, PyTorch, Torchvision, SQLite3, WebSockets, HTTPX, Uvicorn.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons.

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Node.js 18+ (tested on Node v24)
- (Optional) [Ollama](https://ollama.ai) or [LM Studio](https://lmstudio.ai) running locally for real LLM inference.

### 2. Installation

Clone or navigate to the repository directory:
```bash
cd ai-reliability-monitor
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install Frontend dependencies:
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Launching the Application

#### Option A: Single-Command Production Server
FastAPI automatically serves the built React frontend and all API/WebSocket endpoints from a single port:
```bash
python scripts/start_backend.py
```
Open your browser to: **http://127.0.0.1:8000**

#### Option B: Dual Development Server
To run Vite with hot-module replacement and the FastAPI backend concurrently:
- Start the Backend:
  ```bash
  python scripts/start_backend.py
  ```
- In a second terminal, start the Frontend:
  ```bash
  cd frontend
  npm run dev
  ```
Open your browser to: **http://127.0.0.1:5173**

---

## Running the Automated Test Suite

A complete test suite covering OpenCV metrics, MobileNetV3 feature extraction, synthetic degradation detection, audio signal processing, acoustic CNN classification, baseline deviation calculation, LLM adapters, and FastAPI endpoints is included:

```bash
python -m pytest -v tests
```

Expected output:
```
======================= 22 passed in 10.73s =======================
```

---

## Running the Reproducible Degradation Benchmark

Run the standalone command-line benchmark script to generate synthetic test assets and verify degradation detection:

```bash
python scripts/run_degradation_benchmark.py
```

Expected output:
```
IMAGE DEGRADATION DETECTION BENCHMARK:
[Clean Component       ] Reliability: 100.0 | Risk:   0.0 | Status: NORMAL  | Warnings: 0
[Gaussian Blur         ] Reliability:  54.5 | Risk:  45.5 | Status: WARNING | Warnings: 3
   --> SHARPNESS: Image sharpness is critically degraded (severe blur detected)...
[Severe Underexposure  ] Reliability:  35.0 | Risk:  65.0 | Status: WARNING | Warnings: 3
   --> BRIGHTNESS: Input is severely underexposed or in near-total darkness...
[Sensor Noise          ] Reliability:  85.0 | Risk:  15.0 | Status: WARNING | Warnings: 1
   --> NOISE_ESTIMATE: High high-frequency noise or sensor grain detected...
```

---

## Local LLM Setup

The application defaults to loopback addresses (`127.0.0.1`) and never connects to external cloud APIs.

### Using with Ollama:
1. Ensure Ollama is installed and running:
   ```bash
   ollama serve
   ```
2. Pull your preferred local model:
   ```bash
   ollama pull llama3.2:1b
   # or
   ollama pull phi3:mini
   ```
3. In the application, navigate to **Local LLM Console** and select **Ollama**. The app will automatically discover installed models.

### Using with LM Studio or llama.cpp:
1. In LM Studio, start the Local Inference Server on port 1234 (CORS enabled).
2. In the application, select **LM Studio (http://127.0.0.1:1234/v1)**.

### Using the Offline Local Demo Provider:
If you do not have an LLM daemon installed or running, select **Offline Local Demo Provider (Synthetic)**. This local simulator tests all monitoring channels, response uncertainty checks, and user feedback mechanisms offline.

---

## Privacy and Offline Operation Guarantee

- **Air-Gapped Operation**: Once initial dependencies and pretrained weights are cached, core monitoring and inference function without internet access.
- **Zero Cloud APIs**: All OpenCV image processing, FFT audio analysis, PyTorch inference, and SQLite operations occur on local hardware.
- **Explicit Permissions**: Camera and microphone streams are activated only upon explicit user button clicks and are immediately released when stopped. Raw video frames and raw audio are never written to disk without user initiation.
- **Data Retention**: Event history and baseline profiles can be purged at any time from the **Settings** or **Event History** pages.

---

## Limitations

- **General LLM Factuality**: The monitor detects input degradations and explicit model uncertainty, but does not claim to verify the factual correctness of arbitrary generative LLM responses.
- **Extreme Hardware Constraints**: Very old CPUs without AVX instructions may experience higher latency during fine-tuning. Recommended minimum: 4 CPU cores and 8GB RAM.
- **Acoustic Noise Floor**: If microphone hardware has an extremely noisy analog preamp, establish a custom baseline using the **Baseline Manager** to recalibrate the reference noise floor.

---

## License

MIT License. Local on-device AI reliability monitoring architecture.
