# On-Device AI Reliability Monitor: System Architecture & Methodology

## 1. System Overview

The **On-Device AI Reliability Monitor** provides an autonomous, real-time reliability-monitoring layer running alongside local edge AI models and foreground Large Language Models (LLMs). The monitor detects sensor degradations, physical anomalies, and acoustic interference that could cause downstream AI perception and reasoning failures.

```
+---------------------------------------------------------------------------------------------------+
|                                  PHYSICAL & SENSOR INPUT LAYER                                     |
|  [Image Upload / Test Pattern]    [Microphone / Audio WAV]    [Live WebCam (Configurable FPS)]   |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                              INPUT QUALITY & SIGNAL ANALYSIS ENGINE                               |
|                                                                                                   |
|  Visual Analysis (OpenCV):                               Acoustic Analysis (NumPy / SciPy):       |
|  - Brightness (Luminance Mean)                          - RMS Loudness (dBFS)                     |
|  - Sharpness (Laplacian Variance var(∇²I))              - Peak Amplitude & Hard Clipping %        |
|  - Contrast (Intensity Standard Deviation)              - Background Noise Floor (10th %ile)      |
|  - Immerkaer Fast Noise Estimator (Laplacian Mask)      - Estimated SNR (Speech vs Noise)         |
|  - Color Balance Disparity (Max |Channel-Channel|)     - Energy & ZCR Speech VAD                 |
|  - Exposure Saturation & Shadow Clipping %              - Spectral Centroid & 85% Rolloff (FFT)   |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                                 LOCAL MACHINE LEARNING INFERENCE                                  |
|                                                                                                   |
|  MobileNetV3-Small (PyTorch):                           Acoustic Condition CNN (PyTorch):         |
|  - ImageNet Classification (1000 categories)            - 64-Band Log-Mel Filterbank              |
|  - Intermediate 576-dim Feature Embeddings              - 2D ConvNet Feature Extractor            |
|  - Prediction Entropy: H = -∑ p_i log₂(p_i)             - Conditions: Clean Speech, Noise, Wind,  |
|  - Top-2 Margin: p_top1 - p_top2                          Mechanical Hum, Silence, Distorted      |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                                   UNIFIED MONITORING ENGINE                                       |
|  - Robust Baseline Profile Comparison (Median & Median Absolute Deviation MAD)                    |
|  - Rolling Window Exponential Smoothing (Default 5 samples)                                       |
|  - Sustained Deviation Hysteresis Filter (3-sample escalation / de-escalation)                     |
|  - Normalized Image & Audio Reliability Score (0 - 100) and Risk Score (0 - 100)                  |
|  - Plain-Language Explainable Warnings with Corrective Action Recommendations                      |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                               FOREGROUND LOCAL LLM ADAPTER LAYER                                  |
|  - Ollama REST API Adapter (http://127.0.0.1:11434)                                               |
|  - OpenAI-Compatible Adapter (LM Studio http://127.0.0.1:1234, llama.cpp http://127.0.0.1:8080)   |
|  - Offline Local Demo Provider (Local Test & Synthetic Demonstration Mode)                        |
|  - Response Monitoring Layer: Flags input degradation coupling & explicit model uncertainty       |
|  - Local User Feedback Registry: Correct, Incorrect, Uncertain evaluations stored in SQLite       |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Visual Reliability Engine & OpenCV Image Analysis

### 2.1 Metrics and Algorithms
1. **Brightness**: Arithmetic mean intensity across the luminance/grayscale plane:
   $$\mu = \frac{1}{W \times H} \sum_{x,y} I(x, y)$$
2. **Sharpness (Blur Indicator)**: Variance of the discrete Laplacian operator:
   $$\text{Sharpness} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum (\nabla^2 I - \overline{\nabla^2 I})^2$$
   Images with values $< 60.0$ exhibit critical defocus or motion blur.
3. **Contrast**: Standard deviation of pixel intensities:
   $$\sigma = \sqrt{\frac{1}{N} \sum (I(x,y) - \mu)^2}$$
4. **Immerkaer Noise Estimator**: Zero-mean additive Gaussian noise standard deviation $\sigma_n$ estimated via standard 3x3 Laplacian mask $M$:
   $$M = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}$$
   $$\sigma_n = \sqrt{\frac{\pi}{2}} \frac{1}{6 (W-2)(H-2)} \sum_{x=1}^{W-2} \sum_{y=1}^{H-2} |(I * M)(x, y)|$$
5. **Color Cast Disparity**: Maximum pairwise channel divergence:
   $$\Delta_{\text{color}} = \max(|R - G|, |R - B|, |G - B|)$$
6. **Exposure Saturation Indicators**:
   - Underexposure %: $\frac{\text{count}(I < 10)}{W \times H} \times 100$
   - Overexposure %: $\frac{\text{count}(I > 245)}{W \times H} \times 100$

### 2.2 MobileNetV3-Small Classification & Feature Entropy
- Pretrained weights: torchvision `MobileNet_V3_Small_Weights.DEFAULT`.
- Classifier exposes intermediate **576-dimensional feature embeddings** extracted before the final linear classifier.
- **Shannon Entropy of Predictions**:
  $$H(P) = -\sum_{i=1}^{K} p_i \log_2(p_i + \epsilon)$$
  High entropy ($> 4.5$) indicates high classifier indecision.
- **Top-2 Margin**:
  $$\Delta_p = p_{\text{top1}} - p_{\text{top2}}$$

---

## 3. Acoustic Signal & Speech Reliability Engine

### 3.1 Measurable Acoustic Features
1. **RMS Loudness (dBFS)**:
   $$\text{RMS}_{\text{dBFS}} = 20 \log_{10}\left(\sqrt{\frac{1}{N}\sum x[n]^2} + 10^{-9}\right)$$
2. **Peak Amplitude & Clipping Percentage**:
   Percentage of digital audio samples exceeding linear threshold: $|x[n]| \ge 0.98$.
3. **Background Noise Floor**:
   10th-percentile short-time frame energy calculated over 25ms windows with 10ms hop.
4. **Estimated Signal-to-Noise Ratio (SNR)**:
   Log-ratio of 80th-percentile active speech frame energy to 10th-percentile noise floor energy:
   $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{E_{\text{speech}}}{E_{\text{noise}}}\right)$$
5. **Speech Activity Detection (VAD)**:
   Zero-Crossing Rate (ZCR) and short-time energy filtering.
6. **Spectral Centroid & 85% Rolloff**:
   $$f_{\text{centroid}} = \frac{\sum f_k |X[k]|}{\sum |X[k]|}$$

### 3.2 Compact CNN Acoustic Condition Classifier
- Sampling Rate: 16,000 Hz mono.
- Preprocessing: 64 triangular Mel filterbanks spanning 0 to 8,000 Hz; 512-point FFT, 25ms Hanning window, 10ms hop.
- Architecture: 3-stage 2D Convolutional neural network with BatchNorm, ReLU, and Adaptive Average Pooling feeding a linear classifier.
- Detects: `Clean Speech`, `Background Noise`, `Wind / Low-Frequency Noise`, `Mechanical Hum`, `Silence / Static`, `Clipped / Distorted`.

---

## 4. Robust Baseline Calibration & Deviation Scoring

### 4.1 Robust Statistics
Standard Gaussian distributions are highly vulnerable to outliers during calibration. The monitor computes:
- **Median** ($\tilde{x}$) instead of mean.
- **Median Absolute Deviation (MAD)** scaled for consistency with normal distributions:
  $$\text{MAD} = 1.4826 \times \text{median}(|x_i - \tilde{x}|)$$
- 5th and 95th Percentile bounds ($P_5, P_{95}$).
- Robust Z-score:
  $$Z_{\text{robust}} = \frac{x - \text{median}}{\text{MAD}}$$

### 4.2 Scoring Formula
Reliability starts at 100.0 and accrues calibrated penalties for deviations:
$$\text{Reliability} = \max\left(0.0, 100.0 - \sum w_i \cdot \text{penalty}_i\right)$$
$$\text{Risk Score} = 100.0 - \text{Reliability}$$

Weights:
- **Sharpness / Blur**: Weight 30.0
- **Exposure / Darkness**: Weight 20.0
- **Contrast**: Weight 15.0
- **Sensor Noise**: Weight 15.0
- **Color Cast**: Weight 10.0
- **Model Uncertainty**: Weight 10.0

---

## 5. Local LLM Provider Interface

The extensible provider interface (`backend/app/llm_providers/base.py`) defines:
- `check_health() -> Tuple[bool, str]`
- `list_models() -> List[str]`
- `generate(prompt, model, system_prompt, ...) -> LLMProviderResponse`
- `generate_stream(...) -> AsyncGenerator[str, None]`

Supported Adapters:
1. **Ollama**: Connects via native REST endpoints (`/api/tags`, `/api/generate`) at `http://127.0.0.1:11434`.
2. **OpenAI-Compatible**: Supports LM Studio (`http://127.0.0.1:1234/v1`) and llama.cpp server (`http://127.0.0.1:8080/v1`).
3. **Offline Local Demo Provider**: Transparently labeled local mock that allows full demonstration and testing of the pipeline even when an external daemon has not yet been booted.
