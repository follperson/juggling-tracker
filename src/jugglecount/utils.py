import cv2
import os
from .ingest import VideoReader
from loguru import logger

def extract_clip(video_path: str, t_start: float, t_end: float, output_path: str):
    """Extracts a clip from t_start to t_end and saves it to output_path."""
    reader = VideoReader(video_path)
    
    fps = reader.fps
    width = reader.width
    height = reader.height
    
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = reader.cap.read()
        if not ret:
            break
        out.write(frame)
        
    out.release()
    reader.release()
    logger.info(f"Extracted clip to: {output_path}")

def sample_frames(video_path: str, stride: int = 1) -> str:
    """Placeholder for frame sampling logic if needed later."""
    pass
