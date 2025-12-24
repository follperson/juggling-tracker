import streamlit as st
import cv2
import json
import os
from jugglecount.schema import FullAnalysis
from jugglecount.ingest import VideoReader

st.set_page_config(layout="wide")

st.title("JuggleCount Labeler")

video_path = st.text_input("Video Path", "data/raw/YTDown.com_YouTube_3-Ball-Juggling-Tutorial-Step-by-Step-In_Media_lNnouqrZzZs_003_480p.mp4")
analysis_path = st.text_input("Analysis JSON Path", "outputs/test_quick/analysis.json")

if os.path.exists(video_path) and os.path.exists(analysis_path):
    with open(analysis_path, "r") as f:
        analysis_data = json.load(f)
    
    analysis = FullAnalysis(**analysis_data)
    
    st.write(f"FPS: {analysis.video_metadata.fps}")
    st.write(f"Total Segments: {len(analysis.segments)}")
    
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("Segments")
        for i, seg_analysis in enumerate(analysis.segments):
            seg = seg_analysis.segment
            st.write(f"Segment {i}: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg_analysis.total_throws} throws)")
            if st.button(f"Extract Clip for Segment {i}", key=f"btn_{i}"):
                st.write("Extracting...")
                # Placeholder for actual extraction call
                
    with cols[1]:
        st.subheader("Video Preview")
        # Streamlit's video component can take a path or bytes
        if os.path.exists(video_path):
            st.video(video_path)

else:
    st.error("Please provide valid paths to video and analysis.json")
