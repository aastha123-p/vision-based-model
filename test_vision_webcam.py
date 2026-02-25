"""
Real-time Vision Testing Script
Tests the vision module with webcam input for face, emotion, eye, and lip detection
"""

import sys
import cv2
import numpy as np
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


def test_with_webcam():
    """Test vision module with webcam input"""
    print("\n" + "="*60)
    print("REAL-TIME VISION TEST WITH WEBCAM")
    print("="*60)
    
    # Initialize the integrated vision analyzer
    analyzer = VisionAnalyzer()
    
    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return False
    
    print("✓ Webcam opened successfully")
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("-" * 40)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to grab frame")
                break
            
            frame_count += 1
            
            # Analyze the frame
            result = analyzer.analyze_frame(frame)
            
            # Draw results on frame
            output_frame = analyzer.draw_analysis_results(frame, result)
            
            # Display face detection status
            if result['face_detected']:
                # Draw face bounding box
                if result.get('face_info'):
                    x, y, w, h = result['face_info']['bbox']
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display emotion
                if result.get('emotion'):
                    emotion = result['emotion']
                    print(f"Frame {frame_count}: Emotion: {emotion['primary']} ({emotion['confidence']:.2f})")
                
                # Display eye info
                if result.get('eyes'):
                    eyes = result['eyes']
                    print(f"  Eyes: Blink={eyes['blink_count']}, Strain={eyes['strain_level']}, Gaze={eyes['gaze_direction']}")
                
                # Display lip info
                if result.get('lips'):
                    lips = result['lips']
                    print(f"  Lips: Tension={lips['lip_tension_level']}, Speaking={lips['is_speaking']}")
            else:
                print(f"Frame {frame_count}: No face detected")
            
            # Show FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Vision Test", output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"vision_test_{frame_count}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"Saved screenshot: {filename}")
        
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Average FPS: {frame_count/elapsed:.1f}")
        
        summary = analyzer.get_session_summary()
        print(f"Faces detected: {summary['faces_detected']}")
        print(f"Total blinks: {summary['total_blinks']}")
        
    return True


def test_with_image():
    """Test vision module with a static image"""
    print("\n" + "="*60)
    print("VISION TEST WITH IMAGE FILE")
    print("="*60)
    
    # Initialize analyzer
    analyzer = VisionAnalyzer()
    
    # Try to load a test image
    test_images = [
        "test_face.jpg",
        "test_image.jpg",
        "data/test.jpg",
        "data/test_images/face.jpg"
    ]
    
    image_path = None
    for img in test_images:
        if Path(img).exists():
            image_path = img
            break
    
    if not image_path:
        print("No test image found. Using webcam instead...")
        return test_with_webcam()
    
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Could not load image: {image_path}")
        return False
    
    print(f"✓ Image loaded: {frame.shape}")
    
    # Analyze frame
    result = analyzer.analyze_frame(frame)
    
    # Draw results
    output_frame = analyzer.draw_analysis_results(frame, result)
    
    # Display results
    print("\n" + "-" * 40)
    print("ANALYSIS RESULTS:")
    print("-" * 40)
    print(f"Face detected: {result['face_detected']}")
    
    if result.get('face_info'):
        print(f"Face bbox: {result['face_info']['bbox']}")
        print(f"Face confidence: {result['face_info']['confidence']:.2f}")
    
    if result.get('emotion'):
        emotion = result['emotion']
        print(f"\nEmotion:")
        print(f"  Primary: {emotion['primary']}")
        print(f"  Confidence: {emotion['confidence']:.4f}")
        print(f"  Intensity: {emotion['intensity']:.2f}")
        print(f"  All emotions: {emotion['all_emotions']}")
    
    if result.get('eyes'):
        eyes = result['eyes']
        print(f"\nEyes:")
        print(f"  Blink count: {eyes['blink_count']}")
        print(f"  Blink rate: {eyes['blink_rate']:.1f} bpm")
        print(f"  Gaze direction: {eyes['gaze_direction']}")
        print(f"  Strain level: {eyes['strain_level']}")
        print(f"  Strain score: {eyes['strain_score']:.2f}")
    
    if result.get('lips'):
        lips = result['lips']
        print(f"\nLips:")
        print(f"  Tension level: {lips['lip_tension_level']}")
        print(f"  Tension score: {lips['lip_tension_score']:.2f}")
        print(f"  Mouth openness: {lips['mouth_openness']:.4f}")
        print(f"  Is speaking: {lips['is_speaking']}")
    
    if result.get('overall_strain'):
        strain = result['overall_strain']
        print(f"\nOverall Strain:")
        print(f"  Score: {strain['score']:.2f}")
        print(f"  Level: {strain['level']}")
    
    # Save output
    output_path = "vision_test_output.jpg"
    cv2.imwrite(output_path, output_frame)
    print(f"\n✓ Output saved to: {output_path}")
    
    # Show image (if GUI available)
    cv2.imshow("Vision Analysis Result", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def main():
    """Main function to choose test mode"""
    print("\n" + "="*60)
    print("VISION MODULE REAL-TIME TEST")
    print("="*60)
    print("\nSelect test mode:")
    print("1. Webcam test (real-time face, emotion, eye, lip detection)")
    print("2. Image test (static image analysis)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        return test_with_webcam()
    elif choice == '2':
        return test_with_image()
    else:
        print("Exiting...")
        return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
