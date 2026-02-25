"""
Comprehensive test script for the Vision Module
Tests all components: FaceDetector, EmotionDetector, EyeTracker, LipAnalyzer, VisionAnalyzer
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.vision import (
    FaceDetector,
    EmotionDetector,
    EyeTracker,
    LipAnalyzer,
    VisionAnalyzer
)


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Testing Imports")
    print("="*60)

    try:
        print("✓ FaceDetector imported successfully")
        print("✓ EmotionDetector imported successfully")
        print("✓ EyeTracker imported successfully")
        print("✓ LipAnalyzer imported successfully")
        print("✓ VisionAnalyzer imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_face_detector():
    """Test FaceDetector component"""
    print("\n" + "="*60)
    print("TEST 2: Testing FaceDetector")
    print("="*60)

    try:
        detector = FaceDetector(use_mediapipe=True)
        print("✓ FaceDetector initialized")

        # Create a dummy image
        dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print(f"✓ Created dummy image with shape: {dummy_image.shape}")

        # Test face detection
        result = detector.detect_face(dummy_image)
        print(f"✓ Face detection executed: {result is not None}")
        if result:
            print(f"  - Detection method: {result.get('method')}")

        # Test landmark extraction
        landmarks = detector.extract_landmarks(dummy_image)
        print(f"✓ Landmark extraction executed: {landmarks is not None}")
        if landmarks:
            print(f"  - Extraction method: {landmarks.get('method')}")
            print(f"  - Number of landmarks: {len(landmarks.get('all_landmarks', []))}")

        return True
    except Exception as e:
        print(f"✗ FaceDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotion_detector():
    """Test EmotionDetector component"""
    print("\n" + "="*60)
    print("TEST 3: Testing EmotionDetector")
    print("="*60)

    try:
        detector = EmotionDetector()
        print("✓ EmotionDetector initialized")
        print(f"  - Model type: {detector.model_type}")
        print(f"  - Device: {detector.device}")

        # Create a dummy face image
        dummy_face = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        print(f"✓ Created dummy face image with shape: {dummy_face.shape}")

        # Test emotion detection
        emotion_result = detector.detect_emotion(dummy_face)
        print(f"✓ Emotion detection executed")
        print(f"  - Primary emotion: {emotion_result.get('primary_emotion')}")
        print(f"  - Confidence: {emotion_result.get('confidence'):.4f}")
        print(f"  - Is valid: {emotion_result.get('is_valid')}")

        # Test emotion intensity
        if emotion_result.get('is_valid'):
            intensity = detector.get_emotion_intensity(
                emotion_result['primary_emotion'],
                emotion_result['confidence']
            )
            print(f"  - Emotion intensity: {intensity:.2f}")

        return True
    except Exception as e:
        print(f"✗ EmotionDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eye_tracker():
    """Test EyeTracker component"""
    print("\n" + "="*60)
    print("TEST 4: Testing EyeTracker")
    print("="*60)

    try:
        tracker = EyeTracker()
        print("✓ EyeTracker initialized")

        # Create dummy eye landmarks
        dummy_eye_landmarks = [(50+i*10, 100+i*5, 0.0) for i in range(6)]
        print(f"✓ Created dummy eye landmarks: {len(dummy_eye_landmarks)} points")

        # Test eye aspect ratio calculation
        ear = tracker.calculate_eye_aspect_ratio(dummy_eye_landmarks)
        print(f"✓ Eye aspect ratio calculated: {ear:.4f}")

        # Test blink detection
        blink_result = tracker.detect_blink(dummy_eye_landmarks, dummy_eye_landmarks)
        print(f"✓ Blink detection executed")
        print(f"  - Left EAR: {blink_result['left_ear']:.4f}")
        print(f"  - Right EAR: {blink_result['right_ear']:.4f}")
        print(f"  - Is blinking: {blink_result['is_blink']}")
        print(f"  - Blink count: {blink_result['blink_count']}")

        # Test blink rate calculation
        blink_rate = tracker.calculate_blink_rate()
        print(f"✓ Blink rate calculated: {blink_rate:.2f} bpm")

        # Test pupil tracking
        pupil_pos = tracker.track_pupil_position(dummy_eye_landmarks)
        print(f"✓ Pupil position tracked: {pupil_pos}")

        return True
    except Exception as e:
        print(f"✗ EyeTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lip_analyzer():
    """Test LipAnalyzer component"""
    print("\n" + "="*60)
    print("TEST 5: Testing LipAnalyzer")
    print("="*60)

    try:
        analyzer = LipAnalyzer()
        print("✓ LipAnalyzer initialized")

        # Create dummy mouth landmarks
        dummy_mouth_landmarks = [(100+i*5, 150+i*3, 0.0) for i in range(12)]
        print(f"✓ Created dummy mouth landmarks: {len(dummy_mouth_landmarks)} points")

        # Test mouth openness calculation
        mouth_openness = analyzer.calculate_mouth_openness(dummy_mouth_landmarks)
        print(f"✓ Mouth openness calculated: {mouth_openness:.4f}")

        # Test lip tension calculation
        tension_result = analyzer.calculate_lip_tension(dummy_mouth_landmarks)
        print(f"✓ Lip tension calculated")
        print(f"  - Tension score: {tension_result.get('tension_score', 'N/A'):.4f}")
        print(f"  - Tension level: {tension_result.get('tension_level', 'unknown')}")

        return True
    except Exception as e:
        print(f"✗ LipAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_analyzer():
    """Test integrated VisionAnalyzer component"""
    print("\n" + "="*60)
    print("TEST 6: Testing VisionAnalyzer (Integrated)")
    print("="*60)

    try:
        analyzer = VisionAnalyzer()
        print("✓ VisionAnalyzer initialized")

        # Create a dummy frame
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print(f"✓ Created dummy frame with shape: {dummy_frame.shape}")

        # Test frame analysis
        analysis_result = analyzer.analyze_frame(dummy_frame)
        print(f"✓ Frame analysis executed")
        print(f"  - Frame ID: {analysis_result.get('frame_id')}")
        print(f"  - Face detected: {analysis_result.get('face_detected')}")
        print(f"  - FPS: {analysis_result.get('fps'):.2f}")

        if analysis_result.get('error'):
            print(f"  - Error: {analysis_result.get('error')}")

        # Test with a second frame
        dummy_frame2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        analysis_result2 = analyzer.analyze_frame(dummy_frame2)
        print(f"✓ Second frame analysis executed")
        print(f"  - Frame ID: {analysis_result2.get('frame_id')}")

        # Get session summary
        summary = analyzer.get_session_summary()
        print(f"✓ Session summary retrieved")
        print(f"  - Frames processed: {summary.get('frames_processed')}")
        print(f"  - Faces detected: {summary.get('faces_detected')}")

        # Test reset
        analyzer.reset()
        print(f"✓ Analyzer reset successfully")

        return True
    except Exception as e:
        print(f"✗ VisionAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VISION MODULE COMPREHENSIVE TEST SUITE")
    print("="*60)

    results = {}

    # Run all tests
    results['imports'] = test_imports()
    results['face_detector'] = test_face_detector()
    results['emotion_detector'] = test_emotion_detector()
    results['eye_tracker'] = test_eye_tracker()
    results['lip_analyzer'] = test_lip_analyzer()
    results['vision_analyzer'] = test_vision_analyzer()

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")

    print("-" * 60)
    print(f"Total: {total_tests} tests")
    print(f"Passed: {passed_tests} tests")
    print(f"Failed: {failed_tests} tests")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("="*60)

    # Return exit code
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
