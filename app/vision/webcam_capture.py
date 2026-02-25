"""
Webcam Capture Module
Handles real-time video capture and frame processing
"""

import cv2
import numpy as np
import time
from typing import Optional, Callable, List
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class WebcamCapture:
    """
    Manage webcam capture and frame processing
    """

    def __init__(self, camera_id: int = 0):
        """
        Initialize webcam
        
        Args:
            camera_id: Camera device ID (default 0)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False

    def start(self) -> bool:
        """
        Start webcam capture
        
        Returns:
            True if successful
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS)
            
            self.is_running = True
            logger.info(f"Webcam {self.camera_id} started")
            return True
        except Exception as e:
            logger.error(f"Error starting webcam: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop webcam capture
        
        Returns:
            True if successful
        """
        try:
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
                self.is_running = False
                logger.info("Webcam stopped")
                return True
            return False
        except Exception as e:
            logger.error(f"Error stopping webcam: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read single frame from webcam
        
        Returns:
            Frame as numpy array or None
        """
        if not self.cap or not self.is_running:
            return None

        try:
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None

    def capture_frames(
        self, duration: int = 10, callback: Optional[Callable] = None
    ) -> List[np.ndarray]:
        """
        Capture frames for specified duration
        
        Args:
            duration: Duration in seconds
            callback: Optional callback function for each frame
            
        Returns:
            List of captured frames
        """
        if not self.start():
            return []

        frames = []
        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                frame = self.read_frame()
                if frame is None:
                    continue

                frames.append(frame)
                frame_count += 1

                # Call callback if provided
                if callback:
                    callback(frame, frame_count)

                # Quick exit with 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            logger.info(f"Captured {len(frames)} frames in {duration}s")
            return frames

        except Exception as e:
            logger.error(f"Error capturing frames: {e}")
            return frames
        finally:
            self.stop()

    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture single frame and return
        
        Returns:
            Frame or None
        """
        if not self.start():
            return None

        try:
            frame = self.read_frame()
            return frame
        finally:
            self.stop()

    def get_camera_info(self) -> dict:
        """
        Get camera information
        
        Returns:
            Dictionary with camera properties
        """
        if not self.cap:
            self.start()

        try:
            info = {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            }
            return info
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
            return {}

    def record_video(
        self, output_path: str, duration: int = 30
    ) -> bool:
        """
        Record video from webcam
        
        Args:
            output_path: Output video file path
            duration: Duration in seconds
            
        Returns:
            True if successful
        """
        if not self.start():
            return False

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                config.WEBCAM_FPS,
                (config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT),
            )

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < duration:
                frame = self.read_frame()
                if frame is None:
                    continue

                out.write(frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            out.release()
            logger.info(
                f"Video recorded: {output_path} ({frame_count} frames in {duration}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Error recording video: {e}")
            return False
        finally:
            self.stop()
