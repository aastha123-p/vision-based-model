[1mdiff --git a/app/vision/face_detector.py b/app/vision/face_detector.py[m
[1mindex 94ec752..5a37fd2 100644[m
[1m--- a/app/vision/face_detector.py[m
[1m+++ b/app/vision/face_detector.py[m
[36m@@ -1,71 +1,50 @@[m
 """[m
[31m-Face Detection and Landmark Extraction using MediaPipe with Haar Cascade Fallback[m
[32m+[m[32mFace Detection and Landmark Extraction using Haar Cascade[m
 Detects face presence, extracts facial landmarks for eye tracking and lip analysis[m
 """[m
 [m
 import cv2[m
[31m-import mediapipe as mp[m
[31m-from mediapipe.tasks import python[m
[31m-from mediapipe.tasks.python import vision[m
 import numpy as np[m
 from typing import Optional, Dict, List, Tuple[m
 [m
 [m
 class FaceDetector:[m
[31m-    """Detects faces and extracts facial landmarks using MediaPipe with fallback"""[m
[32m+[m[32m    """Detects faces and extracts facial landmarks using Haar Cascade"""[m
     [m
[31m-    def __init__(self, use_mediapipe=True):[m
[31m-        """Initialize face detection with MediaPipe and Haar Cascade fallback"""[m
[32m+[m[32m    def __init__(self, use_mediapipe=False):[m
[32m+[m[32m        """Initialize face detection with Haar Cascade"""[m
         self.use_mediapipe = use_mediapipe[m
         self.face_detector = None[m
         self.face_landmarker = None[m
         [m
[31m-        # Load Haar Cascade as fallback (always available)[m
[32m+[m[32m        # Load Haar Cascade classifiers (always available with OpenCV)[m
         self.haar_cascade = cv2.CascadeClassifier([m
             cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'[m
         )[m
         [m
[31m-        if use_mediapipe:[m
[31m-            self._try_load_mediapipe()[m
[32m+[m[32m        # Also load eye cascade for better eye detection[m
[32m+[m[32m        self.eye_cascade = cv2.CascadeClassifier([m
[32m+[m[32m            cv2.data.haarcascades + 'haarcascade_eye.xml'[m
[32m+[m[32m        )[m
[32m+[m[41m        [m
[32m+[m[32m        # Also load smile cascade[m
[32m+[m[32m        self.smile_cascade = cv2.CascadeClassifier([m
[32m+[m[32m            cv2.data.haarcascades + 'haarcascade_smile.xml'[m
[32m+[m[32m        )[m
         [m
[31m-        # Key landmark indices for analysis[m
[32m+[m[32m        print("[INFO] Haar Cascade FaceDetector initialized")[m
[32m+[m[41m        [m
[32m+[m[32m        # Key landmark indices for synthetic landmarks[m
         self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144][m
         self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380][m
         self.MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146][m
         self.FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136][m
         [m
[31m-        self.mediapipe_available = self.face_detector is not None[m
[31m-        [m
[31m-    def _try_load_mediapipe(self):[m
[31m-        """Try to load MediaPipe models"""[m
[31m-        try:[m
[31m-            base_options = python.BaseOptions(model_asset_path=None)[m
[31m-            options = vision.FaceDetectorOptions(base_options=base_options)[m
[31m-            self.face_detector = vision.FaceDetector.create_from_options(options)[m
[31m-            print("[INFO] MediaPipe FaceDetector loaded successfully")[m
[31m-        except Exception as e:[m
[31m-            print(f"[WARNING] MediaPipe FaceDetector failed: {e}")[m
[31m-            print("[WARNING] Falling back to Haar Cascade detector")[m
[31m-            self.face_detector = None[m
[31m-        [m
[31m-        try:[m
[31m-            base_options = python.BaseOptions(model_asset_path=None)[m
[31m-            landmarker_options = vision.FaceLandmarkerOptions([m
[31m-                base_options=base_options,[m
[31m-                output_face_blendshapes=False,[m
[31m-                output_facial_transformation_matrixes=False,[m
[31m-                num_faces=1[m
[31m-            )[m
[31m-            self.face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)[m
[31m-            print("[INFO] MediaPipe FaceLandmarker loaded successfully")[m
[31m-        except Exception as e:[m
[31m-            print(f"[WARNING] MediaPipe FaceLandmarker failed: {e}")[m
[31m-            print("[WARNING] Falling back to synthetic landmarks")[m
[31m-            self.face_landmarker = None[m
[32m+[m[32m        self.mediapipe_available = False[m
     [m
     def detect_face(self, frame: np.ndarray) -> Optional[Dict]:[m
         """[m
[31m-        Detect face in frame using MediaPipe or Haar Cascade[m
[32m+[m[32m        Detect face in frame using Haar Cascade[m
         [m
         Args:[m
             frame: Input image frame (BGR format)[m
[36m@@ -75,32 +54,7 @@[m [mclass FaceDetector:[m
         """[m
         h, w, _ = frame.shape[m
         [m
[31m-        # Try MediaPipe first[m
[31m-        if self.face_detector is not None:[m
[31m-            try:[m
[31m-                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[m
[31m-                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)[m
[31m-                detection_result = self.face_detector.detect(media_image)[m
[31m-                [m
[31m-                if detection_result.detections:[m
[31m-                    detection = detection_result.detections[0][m
[31m-                    bbox = detection.bounding_box[m
[31m-                    x = int(bbox.origin_x * w)[m
[31m-                    y = int(bbox.origin_y * h)[m
[31m-                    width = int(bbox.width * w)[m
[31m-                    height = int(bbox.height * h)[m
[31m-                    score = detection.categories[0].score if detection.categories else 0.5[m
[31m-                    [m
[31m-                    return {[m
[31m-                        'bbox': (x, y, width, height),[m
[31m-                        'confidence': score,[m
[31m-                        'keypoints': None,[m
[31m-                        'method': 'mediapipe'[m
[31m-                    }[m
[31m-            except Exception as e:[m
[31m-                print(f"[DEBUG] MediaPipe detection failed: {e}")[m
[31m-        [m
[31m-        # Fallback to Haar Cascade[m
[32m+[m[32m        # Use Haar Cascade for face detection[m
         try:[m
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[m
             faces = self.haar_cascade.detectMultiScale([m
[36m@@ -112,10 +66,12 @@[m [mclass FaceDetector:[m
             )[m
             [m
             if len(faces) > 0:[m
[31m-                (x, y, width, height) = faces[0]  # Get largest face[m
[32m+[m[32m                # Get largest face[m
[32m+[m[32m                largest_face = max(faces, key=lambda f: f[2] * f[3])[m
[32m+[m[32m                x, y, width, height = largest_face[m
                 return {[m
                     'bbox': (x, y, width, height),[m
[31m-                    'confidence': 0.7,  # Haar doesn't give confidence[m
[32m+[m[32m                    'confidence': 0.8,[m
                     'keypoints': None,[m
                     'method': 'haar_cascade'[m
                 }[m
[36m@@ -126,7 +82,7 @@[m [mclass FaceDetector:[m
     [m
     def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:[m
         """[m
[31m-        Extract facial landmarks[m
[32m+[m[32m        Extract facial landmarks using Haar Cascade + synthetic landmarks[m
         [m
         Args:[m
             frame: Input image frame (BGR format)[m
[36m@@ -134,70 +90,148 @@[m [mclass FaceDetector:[m
         Returns:[m
             Dictionary with landmark coordinates or None if face not detected[m
         """[m
[31m-        if self.face_landmarker is not None:[m
[31m-            try:[m
[31m-                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[m
[31m-                h, w, _ = frame.shape[m
[31m-                media_image = mp.Image(image_format=mp.