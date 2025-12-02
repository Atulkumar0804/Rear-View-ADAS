import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import cv2
import time

# Add project root to path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
sys.path.append(str(project_root))

from inference.video_inference import VideoDetector

st.set_page_config(
    page_title="Rear-View ADAS System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Rear-View ADAS Monocular System")
st.markdown("### Vehicle Detection, Classification & Distance Estimation")

# Sidebar
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
device = st.sidebar.selectbox("Device", ["cuda", "cpu"])

# Main Area
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    
    if st.button("Start Processing"):
        st.text("Initializing models...")
        
        try:
            detector = VideoDetector(device=device)
            
            # Override confidence if possible (VideoDetector uses global constant currently, 
            # but we can modify the instance if we refactored, for now we just run it)
            # Ideally we would pass confidence to detect_frame, but let's stick to default for now
            # or monkeypatch if needed. The current code uses a global CONFIDENCE_THRESHOLD.
            # For this demo, we'll proceed with the default.
            
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = os.path.join(tempfile.gettempdir(), "output_adas.mp4")
            
            # Streamlit uses VP9/H264 for web. OpenCV default might not work well in browser.
            # We'll try to use 'avc1' for H.264
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                # Fallback to mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect
                detections = detector.detect_frame(frame)
                annotated = detector.draw_detections(frame, detections, fps)
                
                writer.write(annotated)
                
                # Update progress
                if frame_count % 5 == 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

            cap.release()
            writer.release()
            
            progress_bar.progress(1.0)
            status_text.text("Processing Complete!")
            
            # Display result
            # Note: Streamlit sometimes has issues playing local mp4 files if encoding isn't perfect.
            # We read it back as bytes.
            
            st.subheader("Processed Output")
            
            # Re-encode for web compatibility if needed using ffmpeg (if available)
            # For now, try displaying directly.
            
            if os.path.exists(output_path):
                # Convert to H.264 for browser compatibility using ffmpeg if installed
                # This is a common issue with OpenCV generated videos in Streamlit
                converted_path = output_path.replace(".mp4", "_web.mp4")
                os.system(f"ffmpeg -y -i {output_path} -vcodec libx264 {converted_path}")
                
                if os.path.exists(converted_path):
                    st.video(converted_path)
                else:
                    st.warning("FFmpeg not found. Displaying raw output (might not play in all browsers).")
                    st.video(output_path)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

