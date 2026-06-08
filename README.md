# Content-Adaptive Audio Encoder: Live Demo System

**Author:** Tran Diep Linh  
**Course:** Digital Communications and Multimedia Engineering  
**Project:** Assignment 1 - Building a Content-Adaptive Encoder

## 📌 Project Overview
This project simulates an Ingestion Router for a Content-Adaptive Audio Encoder. Unlike conventional encoders that apply a static configuration, this system dynamically analyzes the spectral features of an incoming audio file (Spectral Centroid and Bandwidth) in near-real-time. Based on a conservative routing logic (Speech-First Bias), it delegates the compression task to the most suitable codec:
* **AAC (Advanced Audio Coding):** Utilized for wide-band polyphonic music to preserve high-frequency brilliance using long-block MDCT.
* **OPUS:** Utilized for speech, podcasts, and narrow-band signals to ensure vocal intelligibility using its SILK (LPC) core at low bitrates.

## ✨ Key Features
1. **Live Spectral Analysis Stream:** Processes audio in 0.5-second chunks, applying a Hanning window and Fast Fourier Transform (FFT) to extract spectral features dynamically.
2. **Conservative Routing Logic:** Implements a strict mathematical threshold (SC > 2200 Hz and SB > 1500 Hz) to prevent catastrophic robotic artifacts in speech, ensuring safe fallback to Opus.
3. **Interactive UI with Bitrate Throttling:** Built with Streamlit, allowing users to manually test "Fail-Safe" scenarios by forcing codecs and throttling the Opus target bitrate down to 4 kbps.
4. **Native Ogg-Opus Bitstream Parser:** Bypasses external OS tools by directly parsing the binary structure of `.opus` files (Ogg container). It extracts the TOC (Table of Contents) byte of each packet and decodes the configuration bits (RFC 6716) to accurately count SILK, CELT, and Hybrid frames.

## ⚙️ Prerequisites
To run this demo, your system must have the following installed:
1. **Python 3.8** or higher.
2. **FFmpeg:** Must be installed and added to the system's Environment Variables (PATH). 

## 🚀 Installation & Setup
**Step 1:** Clone or extract the project folder and navigate into it:
```bash
cd "path/to/Code submission"
```

**Step 2:** Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Running the Live Demo
Start the Streamlit local server by executing the following command in your terminal:
```bash
streamlit run app.py
```
*The web interface will automatically open in your default browser at `http://localhost:8501`.*

## 🔬 Testing the Opus Fail-Safe Mechanism
To observe the internal decision matrix of the Opus codec:
1. Upload a complex music file (e.g., Solo Violin).
2. Set the Compression Mode to **Force OPUS (Speech Optimized)**.
3. Set the target bitrate to **64 kbps**.
4. Click **Analyze & Compress**.
5. *Observation:* Despite being forced into the VoIP path, the Native Bitstream Parser will reveal that Opus successfully detected the high tonality and safely bypassed its SILK speech core, routing frames to the CELT music core. 
6. Lower the bitrate slider to **16 kbps** or lower to observe the codec defensively fallback to 100% SILK.