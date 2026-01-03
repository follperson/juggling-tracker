import av
import numpy as np
from typing import List
import os
import time
import speech_recognition as sr
from streamlit_webrtc import AudioProcessorBase
import threading
import logging
import queue

logger = logging.getLogger(__name__)

class JugglingSharedContext:
    def __init__(self):
        self.register_ball_trigger = False
        self.register_background_trigger = False
        self.reset_trigger = False
        self.last_voice_command: str = ""  # For UI notification
        self.lock = threading.Lock()

    def check_and_clear_ball_trigger(self) -> bool:
        with self.lock:
            if self.register_ball_trigger:
                self.register_ball_trigger = False
                return True
            return False

    def check_and_clear_background_trigger(self) -> bool:
        with self.lock:
            if self.register_background_trigger:
                self.register_background_trigger = False
                return True
            return False
            
    def check_and_clear_reset_trigger(self) -> bool:
        with self.lock:
            if self.reset_trigger:
                self.reset_trigger = False
                return True
            return False
    
    def get_and_clear_last_voice_command(self) -> str:
        """Get the last voice command and clear it (for one-time UI display)."""
        with self.lock:
            cmd = self.last_voice_command
            self.last_voice_command = ""
            return cmd

    def set_ball_trigger(self):
        with self.lock:
            self.register_ball_trigger = True
            self.last_voice_command = "register_ball"
            
    def set_background_trigger(self):
        with self.lock:
            self.register_background_trigger = True
            self.last_voice_command = "register_background"

    def set_reset_trigger(self):
        with self.lock:
            self.reset_trigger = True
            self.last_voice_command = "reset"


class JugglingAudioProcessor(AudioProcessorBase):
    def __init__(self, shared_context: JugglingSharedContext):
        self.shared_context = shared_context
        self.recognizer = sr.Recognizer()
        self.audio_buffer = bytearray()
        self.sample_rate = 16000 # WebRTC default usually 48k, often resampled
        self.buffer_lock = threading.Lock()
        self.frame_count = 0
        
        # Processing thread
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_audio_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("JugglingAudioProcessor initialized")

    async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        # Process batch of frames
        try:
            for frame in frames:
                # Convert to numpy
                sound = frame.to_ndarray()
                
                # Check original format properties
                is_int16_source = sound.dtype == np.int16 or sound.dtype == np.int32
                channels = len(frame.layout.channels)
                
                # Debug Check for Slow Audio
                # If PyAV returns (1, N) but channels=2, it is Interleaved Stereo
                if channels == 2 and sound.ndim == 2 and sound.shape[0] == 1:
                    # Reshape to (Samples, 2)
                    sound = sound.reshape(-1, 2)
                
                # If planar stereo (2, N)
                if sound.ndim == 2 and sound.shape[0] == 2:
                     sound = sound.T # Make (N, 2)

                # Normalize to (N, Channels) if (N, 2)
                # (Already handled by reshape above or transpose)
                
                # Mix to Mono (upcasts to float usually)
                if sound.ndim == 2 and sound.shape[1] == 2:
                    sound = np.mean(sound, axis=1)
                elif sound.ndim > 1:
                     # General case
                     sound = np.mean(sound, axis=1)
                    
                # Setup output
                sound_int16 = None
                
                # If source was already int16/int32, simple cast back (handling the float from mean)
                if is_int16_source:
                    sound_int16 = sound.astype(np.int16)
                else:
                    # Assume float32/64 source (-1.0 to 1.0)
                    sound_int16 = (sound * 32767).astype(np.int16)
                
                self.sample_rate = frame.sample_rate
                
                with self.buffer_lock:
                    self.audio_buffer.extend(sound_int16.tobytes())
            
            # Check buffer size after batch
            with self.buffer_lock:
                # 3.0 second buffer to capture full phrases like "Register Ball"
                bytes_per_sec = self.sample_rate * 2 
                threshold = bytes_per_sec * 3.0
                
                if len(self.audio_buffer) > threshold:
                    # print(f"DEBUG: Queuing Audio Batch {len(self.audio_buffer)} bytes")
                    data = bytes(self.audio_buffer)
                    self.command_queue.put((data, self.sample_rate))
                    self.audio_buffer = bytearray()
                    
        except Exception as e:
            logger.error(f"Error in recv_queued: {e}")
            print(f"Error in recv_queued: {e}")
            
        return frames

    def _process_audio_worker(self):
        logger.info("Audio worker started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for audio chunk
                audio_data_bytes, sample_rate = self.command_queue.get(timeout=1.0)
                duration = len(audio_data_bytes) / (sample_rate * 2)
                # print(f"DEBUG: Processing {duration:.1f}s of audio...")
                
                # create AudioData
                audio_data = sr.AudioData(audio_data_bytes, sample_rate, 2) # 2 bytes width = 16-bit
                
                try:
                    # Recognize (using Google Web Speech API - default key)
                    # NOTE: This requires internet.
                    text = self.recognizer.recognize_google(audio_data)
                    text = text.lower()
                    logger.info(f"Heard: {text}")
                    print(f"DEBUG: Heard '{text}'")
                    
                    if "ball" in text or "bowl" in text: # 'bowl' often misheard for ball
                         logger.info("Command: REGISTER BALL")
                         print("DEBUG: Command Triggered: REGISTER BALL")
                         self.shared_context.set_ball_trigger()
                         
                    if "background" in text or "back" in text and "ground" in text:
                         logger.info("Command: REGISTER BACKGROUND")
                         print("DEBUG: Command Triggered: REGISTER BACKGROUND")
                         self.shared_context.set_background_trigger()
                         
                    if "reset" in text:
                         logger.info("Command: RESET")
                         print("DEBUG: Command Triggered: RESET")
                         self.shared_context.set_reset_trigger()
                         
                except sr.UnknownValueError:
                    print("DEBUG: Speech not recognized (UnknownValueError)")
                    pass
                except sr.RequestError as e:
                    logger.error(f"Speech Service error: {e}")
                    print(f"DEBUG: Speech Service Error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio worker error: {e}")
                print(f"DEBUG: Worker Error: {e}")
