# 📸 Real-Time MacBook Camera Object Detector

An optimized, real-time live webcam object detection system powered by **OWLv2** and **Florence-2**, with interactive **Google Gemini 1.5** full-screen environmental analysis.

---

## 📁 Structured Project Layout

```text
├── .env.example              # Template to configure your GEMINI_API_KEY
├── requirements.txt          # Python dependencies
├── run.py                    # Root entrypoint to launch the detector
└── src/                      # Core package namespace
    ├── __init__.py           # Package marker
    ├── live_detector.py      # Main real-time MacBook camera capture & rendering loop
    ├── detector.py           # OWLv2 open-vocabulary detection engine
    ├── verifier.py           # Florence-2 crop-verification and auto-detection engine
    └── models/               # Shared typing data classes
        ├── __init__.py
        └── types.py          # Data classes (e.g. Detection)
```

---

## 🚀 Setup & Installation

### 1. Configure the Virtual Environment
Activate your virtual environment and install the required dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Your Gemini Key (Optional)
To enable the spacebar-triggered full-screen semantic analysis via **Gemini 1.5 Flash**:
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit the `.env` file and replace `your_api_key_here` with your free key from [Google AI Studio](https://aistudio.google.com/).

### 3. Grant Camera Permissions
Ensure that your terminal app (such as Terminal, iTerm2, or VS Code) has been granted **Camera** access in your Mac's **System Settings -> Privacy & Security -> Camera**.

---

## 💬 Usage Guide

Launch the camera feed using the master entrypoint `run.py`:

### Option A: Auto-Detection Mode (Query-Free ⚡)
Detect all available objects in the room automatically without typing any queries:
```bash
python run.py --auto
```
*   *(By default, this runs `microsoft/Florence-2-base` for a highly responsive, low-latency bounding box feed).*
*   To switch to the larger model:
    ```bash
    python run.py --auto --florence-model microsoft/Florence-2-large
    ```

### Option B: Target-Query Mode (Guided Search 🎯)
Define custom objects (like glasses, cups, books) to search for in real-time, and optionally run Florence-2 as a second-opinion verifier:
```bash
python run.py --queries "glasses, mug, book" --verify
```

---

## 🕹️ Live Video Window Controls

*   **`SPACEBAR`**: Capture a clean camera snapshot and generate a deep, conversational full-screen description of everything in your room via **Gemini 1.5 Flash** directly in your terminal.
*   **`q` or `ESC`**: Cleanly release the FaceTime camera, shut down background threads, and close all windows.
