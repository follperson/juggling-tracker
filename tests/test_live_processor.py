import cv2
import numpy as np
import av
import unittest
from jugglecount.web.live_processor import JugglingProcessor

class TestLiveProcessor(unittest.TestCase):
    def test_processor_flow(self):
        print("Initializing Processor...")
        processor = JugglingProcessor()
        print("Processor Initialized.")
        
        # Create dummy frame (black image)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create AV Frame
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = 0

        
        print("Processing Frames...")
        for i in range(5):
            frame.pts = i * 33000 # 33ms
            result_frame = processor.recv(frame)
        print("Frames Processed.")
        
        result_img = result_frame.to_ndarray(format="bgr24")
        self.assertEqual(result_img.shape, (480, 640, 3))
        print("Shape Verified.")

if __name__ == "__main__":
    unittest.main()
