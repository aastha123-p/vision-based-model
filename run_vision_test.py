"""
Simple test runner for the Vision Module
Tests all components with minimal output
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.vision import (
    FaceDetector,
    EmotionDetector,
    EyeTracker,
    LipAnalyzer,
    VisionAnalyzer
)


def test_face_detector():
    """Test FaceDetector component"""
    print("\n[TEST] FaceDetector...")
    try:
        detector = FaceDetector()
        
        # Create a test image with a simulated face (white rectangle on black background)
        # This will trigger the synthetic landmark generation
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a "face-like" region (light gray rectangle)
        test_image[100:380, 200:440] = [180, 180, 180]
        
        # Test face detection
        result = detector.detect_face(test_image)
        
        # Test landmark extraction  
        landmarks = detector.extract_landmarks(test_image)
        
        print(f"  Face detected: {result is not None}")
        if result:
            print(f"  Method: {result.get('method')}")
            print(f"  BBox: {result.get('bbox')}")
        
        print(f"  Landmarks extracted: {landmarks is not None}")
        if landmarks:
            print(f"  Landmark method: {landmarks.get('method')}")
            print(f"  Left eye points: {len(landmarks.get('left_eye', []))}")
            print(f"  Right eye points: {len(landmarks.get('right_eye', []))}")
            print(f"  Mouth points: {len(landmarks.get('mouth', []))}")
        
        print("  ✓ FaceDetector test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ FaceDetector test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotion_detector():
    """Test EmotionDetector component"""
    print("\n[TEST] EmotionDetector...")
    try:
        detector = EmotionDetector()
        
        # Create a test face image (simulated)
        test_face = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Test emotion detection
        result = detector.detect_emotion(test_face)
        
        print(f"  Emotion detected: {result is not None}")
        if result:
            print(f"  Emotion: {result.get('emotion')}")
            print(f"  Confidence: {result.get('confidence'):.2f}")
        
        print("  ✓ EmotionDetector test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ EmotionDetector test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eye_tracker():
    """Test EyeTracker component"""
    print("\n[TEST] EyeTracker...")
    try:
        tracker = EyeTracker()
        
        # Create test eye landmarks
        test_eye_landmarks = [
            (100, 100),  # left corner
            (150, 100),  # right corner
            (115, 90),   # top
            (135, 90),   # top 2
            (115, 110),  # bottom
            (135, 110),  # bottom 2
        ]
        
        # Test EAR calculation
        ear = tracker.calculate_eye_aspect_ratio(test_eye_landmarks)
        
        # Test blink detection
        tracker.update_blink_status(ear)
        
        print(f"  Eye Aspect Ratio: {ear:.3f}")
        print(f"  Blink count: {tracker.blink_count}")
        
        print("  ✓ EyeTracker test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ EyeTracker test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lip_analyzer():
    """Test LipAnalyzer component"""
    print("\n[TEST] LipAnalyzer...")
    try:
        analyzer = LipAnalyzer()
        
        # Create test mouth landmarks
        test_mouth_landmarks = [
            (100, 100), (112, 100), (125, 100), (137, 100), (150, 100), (162, 100),
            (100, 120), (112, 118), (125, 115), (137, 115), (150, 118), (162, 120),
        ]
        
        # Test mouth openness
        openness = analyzer.calculate_mouth_openness(test_mouth_landmarks)
        
        # Test lip tension
        tension = analyzer.calculate_lip_tension(test_mouth_landmarks)
        
        print(f"  Mouth openness: {openness:.2f}")
        print(f"  Tension level: {tension.get('tension_level')}")
        tension_score = tension.get('tension_score')
        if tension_score is None:
            tension_score = 0.0
        print(f"  Tension score: {tension_score:.1f}")
        
        print("  ✓ LipAnalyzer test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ LipAnalyzer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_analyzer():
    """Test VisionAnalyzer component"""
    print("\n[TEST] VisionAnalyzer...")
    try:
        analyzer = VisionAnalyzer()
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:380, 200:440] = [180, 180, 180]
        
        # Test frame analysis
        result = analyzer.analyze_frame(test_frame)
        
        print(f"  Frame ID: {result.get('frame_id')}")
        print(f"  FPS: {result.get('fps'):.2f}")
        
        # Get session summary
        summary = analyzer.get_session_summary()
        print(f"  Frames processed: {summary.get('frames_processed')}")
        
        print("  ✓ VisionAnalyzer test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ VisionAnalyzer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_webcam_demo():
    """Run webcam demo for real-time testing"""
    print("\n[INFO] Starting webcam demo...")
    print("Press 'q' to quit")
    
    analyzer = VisionAnalyzer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ Could not open webcam")
        return False
    
    print("  ✓ Webcam opened")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ✗ Failed to grab frame")
            break
        
        # Analyze frame
        result = analyzer.analyze_frame(frame)
        
        # Draw results
        output = analyzer.draw_analysis_results(frame, result)
        
        cv2.imshow('Vision Demo', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("  ✓ Webcam demo ended")
    return True


def main():
    """Main function"""
    print("=" * 60)
    print("VISION MODULE TEST RUNNER")
    print("=" * 60)
    
    # Check if webcam mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == '--webcam':
        success = run_webcam_demo()
    else:
        # Run all tests
        results = {}
        
        results['FaceDetector'] = test_face_detector()
        results['EmotionDetector'] = test_emotion_detector()
        results['EyeTracker'] = test_eye_tracker()
        results['LipAnalyzer'] = test_lip_analyzer()
        results['VisionAnalyzer'] = test_vision_analyzer()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {name}: {status}")
        
        print("-" * 60)
        print(f"Total: {passed}/{total} tests passed")
        
        success = passed == total
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
