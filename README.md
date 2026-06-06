# Content-Adaptive Audio Encoder

This project is a Streamlit-based web application that dynamically analyzes uncompressed audio files (`.wav`) and automatically selects the optimal lossy compression codec (AAC for music, Opus for speech) based on real-time spectral feature extraction.

## 📂 Project Structure
- `app.py`: The main source code containing the Streamlit UI, signal processing logic (FFT, Spectral Centroid, Bandwidth), and FFmpeg routing.
- `requirements.txt`: Python package dependencies.
- `README.md`: Setup and execution instructions.

## ⚙️ Prerequisites (CRITICAL)
Before running this application, your system **must** have **FFmpeg** installed and added to your system's PATH. The Python script relies on FFmpeg's native C-libraries for the actual transcoding process.

- **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your Environment Variables.
- **macOS:** Run `brew install ffmpeg`
- **Linux (Ubuntu/Debian):** Run `sudo apt update && sudo apt install ffmpeg`

Verify installation by opening a terminal and running:
\`ffmpeg -version\`

## 🚀 Setup Instructions

**Step 1: Clone or extract the project folder**
Navigate to the project directory in your terminal:
\`cd path/to/project/folder\`

**Step 2: Create a virtual environment (Recommended)**
\`\`\`bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
\`\`\`

**Step 3: Install Python dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## ▶️ How to Run
Once all dependencies and FFmpeg are installed, start the Streamlit server:

\`\`\`bash
streamlit run app.py
\`\`\`

This will automatically open the application in your default web browser (typically at `http://localhost:8501`). Upload a `.wav` file through the interface to see the adaptive encoding in action.