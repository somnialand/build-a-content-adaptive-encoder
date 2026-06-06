import streamlit as st
import numpy as np
from scipy.io import wavfile
import subprocess
import os
import tempfile

# ==========================================
# 1. SPECTRAL FEATURE EXTRACTION (ADVANCED DSP)
# ==========================================
def extract_features(file_path):
    fs, data = wavfile.read(file_path)
    
    # Mono mixdown
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    # Amplitude normalization
    if np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        x = data.astype(np.float64)

    # Windowing: Split the audio into 0.5-second chunks for higher resolution
    chunk_size = int(fs * 0.5) 
    num_chunks = len(x) // chunk_size
    
    # Fallback for extremely short files
    if num_chunks == 0:
        num_chunks = 1
        chunk_size = len(x)

    # STEP 1: CALCULATE ENERGY FOR SILENCE GATING
    energies = []
    for i in range(num_chunks):
        chunk = x[i*chunk_size : (i+1)*chunk_size]
        energies.append(np.sum(chunk**2))
        
    max_energy = np.max(energies) if energies else 1.0
    
    # Energy threshold: Only process frames with energy > 5% of peak energy
    energy_threshold = 0.05 * max_energy 

    centroids = []
    bandwidths = []

    # STEP 2: SPECTRAL ANALYSIS ON ACTIVE FRAMES
    for i in range(num_chunks):
        if energies[i] < energy_threshold:
            continue  # Skip silent intervals or low-level background noise

        chunk = x[i*chunk_size : (i+1)*chunk_size]
        win = np.hanning(len(chunk))
        X = np.fft.rfft(chunk * win)
        freqs = np.fft.rfftfreq(len(chunk), 1/fs)
        
        mag = np.abs(X) + 1e-12
        p = mag ** 2

        # High-Pass Filter: Ignore frequencies < 300Hz (removes kick drum/sub-bass bias)
        valid = freqs > 300
        f_valid = freqs[valid]
        p_valid = p[valid]

        if len(p_valid) > 0:
            p_sum = np.sum(p_valid) + 1e-24
            c = float(np.sum(f_valid * p_valid) / p_sum)
            b = float(np.sqrt(np.sum(((f_valid - c) ** 2) * p_valid) / p_sum))
            centroids.append(c)
            bandwidths.append(b)

    # Use the 80th percentile to represent the global feature
    # This effectively ignores localized noise and non-representative frames
    final_centroid = float(np.percentile(centroids, 80)) if centroids else 0.0
    final_bandwidth = float(np.percentile(bandwidths, 80)) if bandwidths else 0.0
    
    return fs, final_centroid, final_bandwidth

# ==========================================
# 2. TRANSCODING PIPELINE
# ==========================================
def compress_audio(input_path, output_path, codec):
    cmd = [
        'ffmpeg', '-y', '-i', input_path, 
        '-c:a', codec, 
        '-b:a', '64k', 
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ==========================================
# 3. STREAMLIT LIVE DEMO SYSTEM
# ==========================================
st.set_page_config(page_title="Content-Adaptive Audio Encoder", layout="centered")
st.title("🎧 Content-Adaptive Audio Encoder")
st.markdown("**Project:** Optimizing Compression via Real-Time Spectral Analysis")

uploaded_file = st.file_uploader("Upload an uncompressed audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        input_path = tmp_in.name

    if st.button("Analyze & Compress"):
        with st.spinner("Extracting Spectral Features..."):
            fs, centroid, bandwidth = extract_features(input_path)
            
            st.subheader("Spectral Analysis Results")
            col1, col2 = st.columns(2)
            col1.metric("Spectral Centroid (SC)", f"{centroid:.1f} Hz")
            col2.metric("Spectral Bandwidth (SB)", f"{bandwidth:.1f} Hz")

            # Classification Logic (Heuristic Thresholds)
            if centroid > 2200 and bandwidth > 1500:
                audio_type = "MUSIC 🎵"
                selected_codec = "aac"
                ext = ".m4a"
            else:
                audio_type = "SPEECH 🗣️"
                selected_codec = "libopus"
                ext = ".opus"
                
            st.success(f"**Classification:** The system identifies this signal as **{audio_type}**.")
            st.info(f"**Adaptive Routing:** Activating **{selected_codec.upper()}** codec at a target bitrate of 64 kbps.")

        with st.spinner(f"Transcoding with {selected_codec.upper()}..."):
            output_path = input_path.replace(".wav", ext)
            compress_audio(input_path, output_path, selected_codec)
            
            # Calculate compression metrics
            original_size = os.path.getsize(input_path) / 1024
            compressed_size = os.path.getsize(output_path) / 1024
            cr = original_size / compressed_size

            st.subheader("🗜️ Empirical Test Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original (PCM)", f"{original_size:.1f} KB")
            c2.metric(f"Compressed ({selected_codec.upper()})", f"{compressed_size:.1f} KB")
            c3.metric("Compression Ratio (CR)", f"{cr:.1f}:1")

            # Playback the reconstructed audio
            with open(output_path, "rb") as f:
                st.audio(f.read(), format=f'audio/{ext.strip(".")}')

        os.remove(input_path)