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
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        x = data.astype(np.float64)

    chunk_size = int(fs * 0.5) 
    num_chunks = max(1, len(x) // chunk_size)
    energies = [np.sum(x[i*chunk_size : (i+1)*chunk_size]**2) for i in range(num_chunks)]
    max_energy = np.max(energies) if energies else 1.0
    energy_threshold = 0.05 * max_energy 

    centroids, bandwidths = [], []
    for i in range(num_chunks):
        if energies[i] < energy_threshold: continue
        chunk = x[i*chunk_size : (i+1)*chunk_size]
        win = np.hanning(len(chunk))
        X = np.fft.rfft(chunk * win)
        freqs = np.fft.rfftfreq(len(chunk), 1/fs)
        p = (np.abs(X) + 1e-12) ** 2
        valid = freqs > 300
        f_valid, p_valid = freqs[valid], p[valid]
        if len(p_valid) > 0:
            p_sum = np.sum(p_valid) + 1e-24
            c = float(np.sum(f_valid * p_valid) / p_sum)
            b = float(np.sqrt(np.sum(((f_valid - c) ** 2) * p_valid) / p_sum))
            centroids.append(c)
            bandwidths.append(b)

    final_centroid = float(np.percentile(centroids, 80)) if centroids else 0.0
    final_bandwidth = float(np.percentile(bandwidths, 80)) if bandwidths else 0.0
    return fs, final_centroid, final_bandwidth

# ==========================================
# 2. TRANSCODING & NATIVE BITSTREAM PARSER
# ==========================================
def compress_audio(input_path, output_path, codec, opus_bitrate_kbps="64"):
    if codec == 'libopus':
        # Ghép thông số bitrate từ thanh trượt vào lệnh FFmpeg
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', codec, '-b:a', f'{opus_bitrate_kbps}k', '-application', 'voip', output_path]
    else:
        # AAC mình vẫn khóa cứng ở 64k để làm mốc so sánh tiêu chuẩn
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', codec, '-b:a', '64k', output_path]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def analyze_opus_payload(file_path):
    """
    Native Python Ogg-Opus Bitstream Parser.
    Reads the binary file directly, finds Ogg pages, and extracts the TOC byte
    (Table of Contents) of each Opus packet to determine the active internal core.
    Reference: IETF RFC 6716 (Section 3.1)
    """
    try:
        silk, celt, hybrid = 0, 0, 0
        with open(file_path, 'rb') as f:
            data = f.read()

        idx = 0
        packet_start = True

        while True:
            # Tìm cờ bắt đầu của Ogg Page (OggS)
            idx = data.find(b'OggS', idx)
            if idx == -1: break
            
            if idx + 27 > len(data): break
            page_segments = data[idx + 26]
            
            if idx + 27 + page_segments > len(data): break
            segment_table = data[idx + 27 : idx + 27 + page_segments]
            
            data_ptr = idx + 27 + page_segments
            
            for seg_len in segment_table:
                if packet_start and seg_len > 0 and data_ptr < len(data):
                    toc = data[data_ptr]
                    
                    # Bỏ qua 2 gói tin Header của file Opus (OpusHead và OpusTags)
                    if data_ptr + 8 <= len(data) and data[data_ptr:data_ptr+8] in [b'OpusHead', b'OpusTags']:
                        pass
                    else:
                        # Dịch bit (>> 3) để lấy 5 bit đầu tiên (Config bits) theo chuẩn RFC 6716
                        config = toc >> 3
                        if config < 12:
                            silk += 1
                        elif config < 16:
                            hybrid += 1
                        else:
                            celt += 1
                            
                data_ptr += seg_len
                # Nếu độ dài segment < 255, gói tin kết thúc tại đây, byte tiếp theo sẽ là gói mới
                packet_start = (seg_len < 255)

            idx = data_ptr 
            
        total = silk + celt + hybrid
        if total == 0:
            return {"status": "error", "msg": "Could not parse Opus packets natively.", "raw": ""}

        return {
            "status": "success",
            "silk": silk,
            "celt": celt,
            "hybrid": hybrid
        }
    except Exception as e:
        return {"status": "error", "msg": f"Parser Error: {str(e)}", "raw": ""}

# ==========================================
# 3. STREAMLIT LIVE DEMO SYSTEM
# ==========================================
import time
import pandas as pd

st.set_page_config(page_title="Content-Adaptive Audio Encoder", layout="centered")
st.title("🎧 Content-Adaptive Audio Encoder")
st.markdown("**Project:** Optimizing Compression via Near-Real-Time Spectral Analysis")

uploaded_file = st.file_uploader("Upload an uncompressed audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        input_path = tmp_in.name

    st.markdown("### ⚙️ Demo Configuration")
    demo_mode = st.radio(
        "Select Compression Mode:",
        ("🤖 Auto-Detect (Adaptive Routing)", "🎵 Force AAC (Music Optimized)", "🗣️ Force OPUS (Speech Optimized)")
    )
    
    # --- THÊM THANH TRƯỢT BITRATE Ở ĐÂY ---
    opus_bitrate = st.slider(
        "🎛️ Opus Target Bitrate (kbps):", 
        min_value=4, 
        max_value=64, 
        value=16, 
        step=2, 
        help="Kéo xuống thấp (4-12 kbps) để ép Opus dùng SILK thuần. Kéo lên cao để Opus cho phép bật Hybrid/CELT."
    )

if st.button("Analyze & Compress"):
        
        # CHỈ PHÂN TÍCH KHI CHỌN AUTO-DETECT
        if demo_mode == "🤖 Auto-Detect (Adaptive Routing)":
            st.subheader("📡 Live Spectral Analysis Stream")
            progress_bar = st.progress(0)
            status_text = st.empty()
            col1, col2 = st.columns(2)
            sc_metric = col1.empty()
            sb_metric = col2.empty()
            chart_placeholder = st.empty()

            fs, data = wavfile.read(input_path)
            if data.ndim > 1: data = data.mean(axis=1)
            if np.issubdtype(data.dtype, np.integer):
                x = data.astype(np.float64) / np.iinfo(data.dtype).max
            else:
                x = data.astype(np.float64)

            chunk_size = int(fs * 0.5) 
            num_chunks = max(1, len(x) // chunk_size)
            max_energy = max([np.sum(x[i*chunk_size : (i+1)*chunk_size]**2) for i in range(num_chunks)] or [1.0])
            energy_threshold = 0.05 * max_energy

            centroids, bandwidths = [], []
            chart_data = {'Time (s)': [], 'SC (Hz)': [], 'SB (Hz)': []}

            # ... (giữ nguyên phần khai báo biến ở trên)
            centroids, bandwidths = [], []
            chart_data = {'Time (s)': [], 'SC (Hz)': [], 'SB (Hz)': []}

            # TỐI ƯU HÓA HIỆU NĂNG CHO FILE DÀI
            is_long_file = num_chunks > 100  # Nếu file dài hơn 50 giây
            update_interval = max(1, num_chunks // 50) # Tối đa chỉ vẽ lại biểu đồ 50 lần

            for i in range(num_chunks):
                chunk = x[i*chunk_size : (i+1)*chunk_size]
                energy = np.sum(chunk**2)
                current_time = (i + 1) * 0.5
                
                if energy >= energy_threshold:
                    win = np.hanning(len(chunk))
                    X = np.fft.rfft(chunk * win)
                    freqs = np.fft.rfftfreq(len(chunk), 1/fs)
                    p = (np.abs(X) + 1e-12) ** 2
                    valid = freqs > 300
                    f_valid, p_valid = freqs[valid], p[valid]
                    
                    if len(p_valid) > 0:
                        p_sum = np.sum(p_valid) + 1e-24
                        c = float(np.sum(f_valid * p_valid) / p_sum)
                        b = float(np.sqrt(np.sum(((f_valid - c) ** 2) * p_valid) / p_sum))
                        centroids.append(c)
                        bandwidths.append(b)
                        chart_data['Time (s)'].append(current_time)
                        chart_data['SC (Hz)'].append(c)
                        chart_data['SB (Hz)'].append(b)

                # CHỈ CẬP NHẬT GIAO DIỆN MỖI KHI ĐẾN MỐC INTERVAL HOẶC KHUNG CUỐI CÙNG
                if i % update_interval == 0 or i == num_chunks - 1:
                    progress_bar.progress(min((i + 1) / num_chunks, 1.0))
                    status_text.markdown(f"**Processing Frame:** {i+1}/{num_chunks} | **Buffer:** 0.5s")
                    
                    if centroids and bandwidths:
                        sc_metric.metric("Current Centroid", f"{centroids[-1]:.1f} Hz")
                        sb_metric.metric("Current Bandwidth", f"{bandwidths[-1]:.1f} Hz")
                        df = pd.DataFrame(chart_data).set_index('Time (s)')
                        chart_placeholder.line_chart(df, height=200)

                # Bỏ qua delay ngủ (sleep) nếu file quá dài để chạy với tốc độ tối đa của CPU
                if not is_long_file:
                    time.sleep(0.01)

            final_centroid = float(np.percentile(centroids, 80)) if centroids else 0.0
            final_bandwidth = float(np.percentile(bandwidths, 80)) if bandwidths else 0.0
            
            st.success("✅ Stream Analysis Complete!")
            st.markdown("### 🎯 Global Decision Strategy (80th Percentile)")
            c1, c2 = st.columns(2)
            c1.metric("Aggregated SC (80th)", f"{final_centroid:.1f} Hz")
            c2.metric("Aggregated SB (80th)", f"{final_bandwidth:.1f} Hz")

            # Logic định tuyến nội bộ của Auto
            if final_centroid > 2200 and final_bandwidth > 1500:
                audio_type = "MUSIC 🎵"
                selected_codec = "aac"
                ext = ".m4a"
            else:
                audio_type = "SPEECH 🗣️"
                selected_codec = "libopus"
                ext = ".opus"
            st.info(f"**Auto-Classification:** Signal dominated by **{audio_type}**.")

        # NHẢY CÓC PHÂN TÍCH KHI CHỌN FORCE
        elif demo_mode == "🎵 Force AAC (Music Optimized)":
            st.warning("⏭️ **Skipping Analysis:** Manual Override active. Bypassing DSP pipeline.")
            selected_codec = "aac"
            ext = ".m4a"
            
        else: # Force OPUS
            st.warning("⏭️ **Skipping Analysis:** Manual Override active. Bypassing DSP pipeline.")
            selected_codec = "libopus"
            ext = ".opus"

        if selected_codec == "aac":
            st.info("**Executing Encoder:** Routing directly to **AAC** at fixed 64 kbps.")
        else:
            st.info(f"**Executing Encoder:** Routing directly to **OPUS** at {opus_bitrate} kbps.")

        # --- BƯỚC 3: TRANSCODING ---
        with st.spinner(f"Transcoding..."):
            output_path = input_path.replace(".wav", ext)
            compress_audio(input_path, output_path, selected_codec, opus_bitrate)
            
            original_size = os.path.getsize(input_path) / 1024
            compressed_size = os.path.getsize(output_path) / 1024
            cr = original_size / compressed_size

            st.subheader("🗜️ Final Delivery")
            d1, d2, d3 = st.columns(3)
            d1.metric("Original (PCM)", f"{original_size:.1f} KB")
            d2.metric(f"Compressed", f"{compressed_size:.1f} KB")
            d3.metric("Ratio", f"{cr:.1f}:1")

            with open(output_path, "rb") as f:
                st.audio(f.read(), format=f'audio/{ext.strip(".")}')

            # --- BƯỚC 4: OPUS NATIVE BITSTREAM ANALYSIS ---
            if selected_codec == "libopus":
                st.markdown("---")
                st.markdown("### 🔬 Opus Internal Core Analysis (Native Bitstream Parser)")
                with st.spinner("Parsing Ogg-Opus TOC Bytes at binary level..."):
                    analysis = analyze_opus_payload(output_path)
                    
                    if analysis["status"] == "success":
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("SILK Frames (Speech)", analysis["silk"])
                        sc2.metric("CELT Frames (Music)", analysis["celt"])
                        sc3.metric("Hybrid Frames", analysis["hybrid"])
                        
                        # Logic đánh giá thông minh hơn
                        dominant = max(analysis["silk"], analysis["celt"], analysis["hybrid"])
                        
                        if dominant == analysis["silk"]:
                            st.info("🎙️ **SILK Dominant:** Opus correctly used its LPC speech core for most of the duration.")
                        elif dominant == analysis["celt"]:
                            st.success("💡 **CELT Dominant (Music/Tones):** Opus detected strong tonality/music and bypassed its speech core, routing the majority of frames purely to CELT!")
                        else:
                            st.success("⚖️ **Hybrid Dominant (Wideband Speech/Mixed):** Opus detected complex wideband audio. It used SILK for voice fundamentals and CELT to preserve high-frequency details (like room reverb or sibilance)!")
        os.remove(input_path)