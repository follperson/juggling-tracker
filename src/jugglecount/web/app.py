import streamlit as st
import tempfile
import os
import pandas as pd
import json
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from jugglecount.core.pipeline import run_analysis
from jugglecount.db.db import get_all_users, upsert_user, get_user_performance_history
from jugglecount.db.schema import FullAnalysis
from jugglecount.web.live_processor import JugglingProcessor
from jugglecount.web.voice_processor import JugglingSharedContext, JugglingAudioProcessor

st.set_page_config(page_title="JuggleCount", layout="wide")

st.title("🤹 JuggleCount User Dashboard")

# --- Sidebar: User Management ---
st.sidebar.header("User Selection")

# Fetch users
users = get_all_users()
user_options = {u.user_id: u.user_id for u in users} # ID -> Name (or ID)

selected_user_id = st.sidebar.selectbox(
    "Select User", 
    options=list(user_options.keys()), 
    format_func=lambda x: user_options[x] if x in user_options else x
)

st.sidebar.markdown("---")
st.sidebar.subheader("Create New User")
new_user_id = st.sidebar.text_input("New User ID")
new_user_name = st.sidebar.text_input("Display Name")
new_user_email = st.sidebar.text_input("Email")
if st.sidebar.button("Create User"):
    if new_user_id:
        upsert_user(new_user_id, new_user_name, new_user_email)
        st.sidebar.success(f"Created user {new_user_id}")
        st.rerun()
    else:
        st.sidebar.error("User ID required")

# --- Main App ---

tabs = st.tabs(["📤 Upload & Analyze", "📷 Live Practice", "📈 Progress History"])

with tabs[0]:
    st.header("Analyze New Session")
    
    uploaded_file = st.file_uploader("Upload Juggling Video", type=["mp4", "mov", "avi"])
    
    if uploaded_file and selected_user_id:
        force_rerun = st.checkbox("Force Rerun Analysis", value=False)
        
        if st.button("Start Analysis"):
            with st.spinner("Processing video... This may take a while."):
                # Need to determine output path first to check existence
                # We can't know the exact path until we write the temp file or use the uploaded filename
                # For consistency with how we name folders:
                video_filename = uploaded_file.name
                output_dir = os.path.join("outputs", f"session_{selected_user_id}_{video_filename}")
                json_path = os.path.join(output_dir, "analysis.json")
                
                run_needed = True
                if os.path.exists(json_path) and not force_rerun:
                    st.info("Found existing analysis results. Loading...")
                    run_needed = False
                
                # Save to temp file ONLY if we need to run or if we need the video for overlay (which we do if it's not in outputs)
                # But typically overlay is in outputs.
                
                video_path_for_pipeline = None
                
                try:
                    if run_needed:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
                        tfile.write(uploaded_file.read())
                        video_path_for_pipeline = tfile.name
                        tfile.close()
                        
                        run_analysis(video_path_for_pipeline, output_dir, test=False, user_id=selected_user_id)
                        st.success("Analysis Complete!")
                    else:
                        st.success("Loaded Existing Analysis.")
                    
                    # Display Results
                    json_path = os.path.join(output_dir, "analysis.json")
                    if os.path.exists(json_path):
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            analysis = FullAnalysis(**data)
                            
                        # Metrics Row
                        m_col1, m_col2, m_col3 = st.columns(3)
                        total_throws = sum(s.total_throws for s in analysis.segments)
                        duration = analysis.video_metadata.duration
                        m_col1.metric("Total Throws", total_throws)
                        m_col2.metric("Duration", f"{duration:.2f}s")
                        m_col3.metric("Throw Rate", f"{total_throws/duration:.2f} t/s" if duration > 0 else "0 t/s")
                        
                        # Tabs for Video/Details
                        subtabs = st.tabs(["Video Overlay", "Segment Details"])
                        with subtabs[0]:
                            vid_out = os.path.join(output_dir, "debug_overlay.mp4")
                            if os.path.exists(vid_out):
                                st.video(vid_out)
                            else:
                                st.warning("Overlay video not found.")
                        
                        with subtabs[1]:
                            st.write(analysis.segments)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
                finally:
                    # Cleanup temp file
                    if video_path_for_pipeline and os.path.exists(video_path_for_pipeline):
                        os.unlink(video_path_for_pipeline)

with tabs[1]:
    st.header("Live Juggling Practice")
    st.markdown("Use your webcam to track juggling in real-time. Peaks (catches) are counted automatically.")
    
    # WebRTC Streamer
    # Use key to prevent component remounting
    
    # Shared Context for Voice Commands
    if "shared_ctx" not in st.session_state:
        st.session_state.shared_ctx = JugglingSharedContext()
    
    # Capture context locally to pass to worker threads safely
    shared_ctx = st.session_state.shared_ctx
        
    ctx = webrtc_streamer(
        key="live-juggling-v3",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: JugglingProcessor(shared_ctx),
        audio_processor_factory=lambda: JugglingAudioProcessor(shared_ctx),
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )
    
    # Check for voice command notifications and show toast
    last_cmd = shared_ctx.get_and_clear_last_voice_command()
    if last_cmd == "register_ball":
        st.toast("🎤 Voice: Registering ball...", icon="🎙️")
    elif last_cmd == "register_background":
        st.toast("🎤 Voice: Registering background...", icon="🏔️")
    elif last_cmd == "reset":
        st.toast("🎤 Voice: Resetting signatures...", icon="🔄")
    
    if ctx.video_processor:
        processor = ctx.video_processor
        
        # Detector Selection
        detector_type = st.radio("Technique", ["YOLO (AI)", "Visual Signature (Color)"], horizontal=True)
        active_type = "YOLO" if detector_type.startswith("YOLO") else "Signature"
        
        if active_type != processor.active_detector_type:
             processor.set_detector_type(active_type)
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Settings"):
            if active_type == "YOLO":
                st.caption("🚀 Performance Tuning")
                
                # Model Selection
                model_options = [
                    "yolo11l.mlpackage", "yolo11x.mlpackage", "yolov8l.mlpackage", 
                    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
                ]
                selected_model = st.selectbox("Model Size (Requires Download)", model_options, index=1) # Default 'yolo11x.mlpackage'
                if selected_model != processor.detector_yolo.model.ckpt_path and st.button("Load Model"):
                     processor.set_yolo_model(selected_model)
                     st.toast(f"Loaded {selected_model}!", icon="🤖")

                # Image Size
                imgsz = st.select_slider("Inference Resolution (Higher = More Accurate but Slower)", 
                                         options=[320, 480, 640, 960, 1280], value=480)
                if imgsz != processor.detector_yolo.imgsz:
                    processor.set_yolo_imgsz(imgsz)
                
                # Confidence
                conf = st.slider("Confidence Threshold", 0.05, 1.0, 0.1, step=0.05)
                processor.set_yolo_confidence(conf)
                
            bg_thresh = st.slider("Background Sensitivity (Lower = More Sensitive)", 5.0, 200.0, 50.0, step=5.0)
            processor.set_background_threshold(bg_thresh)
            
            if active_type == "Signature":
                conf = st.slider("Color Match Threshold", 0.1, 1.0, 0.8, step=0.05)
                processor.set_signature_confidence(conf)
             
        # Debug Toggle
        debug_mode = st.checkbox("🥽 Debug View (Show Background Mask)", value=False)
        processor.set_debug_mode(debug_mode)
        
        # Registration Controls
        col_reg1, col_reg2 = st.columns([1, 2])
        
        with col_reg1:
            reg_mode = st.toggle("🔴 Register New Object", value=False)
        
        if reg_mode:
            processor.set_registration_mode(True)
            st.info("Align your object in the yellow box.")
            
            if active_type == "YOLO":
                # Check what's currently detected
                last_obj = processor.last_detected_object
                if last_obj:
                    name, cid = last_obj
                    st.success(f"Target: **{name}** (ID: {cid})")
                    
                    if st.button("✅ Save Object"):
                        processor.add_class_id(cid)
                        st.toast(f"Registered {name}!", icon="✅")
                else:
                    st.warning("No object detected in center.")
                    
            elif active_type == "Signature":
                col_sig1, col_sig2 = st.columns(2)
                with col_sig1:
                    obj_name = st.text_input("Object Name", value="Orange")
                with col_sig2:
                    if st.button("📸 Capture / Add Angle"):
                        processor.trigger_signature_registration(obj_name)
                        st.toast(f"Adding pattern to {obj_name}...", icon="📸")
                
                # Background Registration
                if st.button("🏔️ Capture Background (Reduce False Positives)"):
                    processor.trigger_background_registration()
                    st.toast("Capturing background colors...", icon="🏔️")
                
                if processor.detector_signature.background_hist is not None:
                     st.success("✅ Background Model Active")
                else:
                     st.info("ℹ️ Tip: Capture Background to ignore wall/floor colors.")

                st.info("Tip: Rotate object and capture multiple times for better accuracy.")
                
        else:
            processor.set_registration_mode(False)
            
        # Display Active Classes / Signatures
        if active_type == "YOLO":
            classes = [processor.detector_yolo.class_names[i] for i in processor.allowed_class_ids]
            st.write(f"**Tracking:** {', '.join(classes)}")
        else:
             # Use safe getter
             sigs_map = processor.detector_signature.get_signatures()
             sigs = list(sigs_map.keys())
             
             if not sigs:
                 st.warning("No color signatures registered yet.")
             else:
                 st.write(f"**Tracking:** {', '.join(sigs)}")
                 
                 # Visualize Signatures
                 st.markdown("### 🎨 Color Signatures")
                 cols = st.columns(min(len(sigs), 3))
                 for idx, (label, hist) in enumerate(sigs_map.items()):
                     with cols[idx % len(cols)]:
                         if hist is None:
                             st.warning(f"No histogram for {label}")
                             continue
                         
                         # 1D Hue-only histogram visualization
                         hist_flat = hist.flatten()
                         n_bins = len(hist_flat)
                         
                         # Create a colorful bar chart
                         bar_height = 80
                         bar_width = max(200, n_bins * 6)
                         canvas = np.zeros((bar_height + 25, bar_width, 3), dtype=np.uint8)
                         
                         hist_norm = hist_flat / (hist_flat.max() + 1e-6)
                         bin_w = bar_width // n_bins
                         
                         for i in range(n_bins):
                             hue = int((i / n_bins) * 180)
                             bar_h = int(hist_norm[i] * bar_height)
                             color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                             color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
                             x1, x2 = i * bin_w, (i + 1) * bin_w
                             cv2.rectangle(canvas, (x1, bar_height - bar_h), (x2, bar_height), 
                                         (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), -1)
                         
                         # Labels
                         for name, hue in [("R", 0), ("Y", 30), ("G", 60), ("C", 90), ("B", 120), ("M", 150)]:
                             x = int((hue / 180) * bar_width)
                             cv2.putText(canvas, name, (x, bar_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                         
                         reg_info = processor.detector_signature.get_registration_info(label)
                         samples = reg_info['sample_count'] if reg_info else 0
                         st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"{label} ({samples} samples)", use_container_width=True)
                 
                 if st.button("❌ Reset All Signatures"):
                     processor.shared_context.set_reset_trigger()
                     st.rerun()

        
        # Stats
        st.metric("Session Peaks", processor.peak_count)

with tabs[2]:
    st.header(f"History for {selected_user_id}")
    if selected_user_id:
        history = get_user_performance_history(selected_user_id)
        if history:
            df = pd.DataFrame([
                {
                    "Date": pd.to_datetime(m.timestamp, unit="s"), 
                    "Throws": m.total_throws, 
                    "Duration": m.duration_seconds
                } 
                for m in history
            ])
            st.line_chart(df, x="Date", y="Throws")
            st.dataframe(df)
        else:
            st.info("No history found for this user.")
    else:
        st.warning("Please select a user to view history.")

