import cv2
import os
from typing import Iterator, Tuple
from loguru import logger

class VideoReader:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"Loaded video: {video_path}")
        logger.info(f"  FPS: {self.fps}, Frames: {self.frame_count}, Res: {self.width}x{self.height}, Duration: {self.duration:.2f}s")

    def __iter__(self) -> Iterator[Tuple[int, float, cv2.Mat]]:
        """Yields (frame_index, timestamp, frame)"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = i / self.fps
            yield i, timestamp, frame

    def get_frame(self, frame_index: int) -> Tuple[float, cv2.Mat]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_index}")
        timestamp = frame_index / self.fps
        return timestamp, frame

    def release(self):
        self.cap.release()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
