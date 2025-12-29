import cv2
import numpy as np
import time
from models.gesture_recognition import GestureRecognizer
from utils.hand_tracking import HandTracker
import argparse

def test_realtime_gesture_recognition(model_path: str = None,
                                    camera_index: int = 0,
                                    display_window: bool = True,
                                    test_duration: int = 30):
    """
    Test real-time gesture recognition with webcam

    Args:
        model_path: Path to trained CNN model (optional)
        camera_index: Webcam device index
        display_window: Whether to show OpenCV window
        test_duration: Test duration in seconds
    """

    print("🔄 Initializing gesture recognition system...")

    # Initialize components
    gesture_recognizer = GestureRecognizer(
        use_cnn=True if model_path else False,
        use_lstm=False,
        cnn_model_path=model_path,
        confidence_threshold=0.5
    )

    hand_tracker = HandTracker()

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return

    print("✅ Camera opened successfully")
    print(f"Testing for {test_duration} seconds...")
    print("Make gestures in front of the camera. Press 'q' to quit early.")

    # Performance tracking
    frame_count = 0
    gesture_count = 0
    start_time = time.time()
    fps_history = []

    try:
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame_start_time = time.time()

            # Process frame
            hand_landmarks = hand_tracker.process_frame(frame)

            gesture_detected = False
            gesture_name = "none"
            confidence = 0.0

            if hand_landmarks:
                # Extract features
                features = hand_tracker.extract_single_hand_features(hand_landmarks)

                if features is not None:
                    # Recognize gesture
                    result = gesture_recognizer.recognize_static_gesture(features)

                    if result:
                        gesture_name, confidence = result
                        gesture_detected = True
                        gesture_count += 1

                # Draw landmarks
                frame = hand_tracker.draw_landmarks(frame, hand_landmarks)

            # Calculate FPS
            frame_time = time.time() - frame_start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(fps)

            # Display information on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gestures: {gesture_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if gesture_detected:
                cv2.putText(frame, f"Detected: {gesture_name}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display frame
            if display_window:
                cv2.imshow('Gesture Recognition Test', frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

            # Progress update every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                print(f"⏱️  {int(elapsed)}s elapsed - {gesture_count} gestures detected")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")

    finally:
        # Cleanup
        cap.release()
        if display_window:
            cv2.destroyAllWindows()

    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = np.mean(fps_history) if fps_history else 0
    gestures_per_minute = (gesture_count / total_time) * 60 if total_time > 0 else 0

    print("""
📊 TEST RESULTS""")
    print("=" * 40)
    print(f"Total frames processed: {frame_count}")
    print(f"Total gestures detected: {gesture_count}")
    print(f"Test duration: {total_time:.1f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Gestures per minute: {gestures_per_minute:.1f}")
    print(f"Detection rate: {(gesture_count/frame_count)*100:.1f}%" if frame_count > 0 else "0%")

    # Performance assessment
    if avg_fps >= 15:
        print("✅ Good performance (FPS >= 15)")
    elif avg_fps >= 10:
        print("⚠️ Acceptable performance (10 <= FPS < 15)")
    else:
        print("❌ Poor performance (FPS < 10)")

    if gestures_per_minute >= 10:
        print("✅ Good detection rate")
    elif gestures_per_minute >= 5:
        print("⚠️ Moderate detection rate")
    else:
        print("❌ Low detection rate - may need model tuning")

    return {
        'frames_processed': frame_count,
        'gestures_detected': gesture_count,
        'duration': total_time,
        'avg_fps': avg_fps,
        'gestures_per_minute': gestures_per_minute
    }

def benchmark_models(model_paths: list, camera_index: int = 0, test_duration: int = 10):
    """Benchmark multiple models"""
    results = {}

    for model_path in model_paths:
        model_name = model_path.split('/')[-1].replace('.h5', '')
        print(f"\n🔄 Testing model: {model_name}")

        result = test_realtime_gesture_recognition(
            model_path=model_path,
            camera_index=camera_index,
            display_window=False,  # Disable display for batch testing
            test_duration=test_duration
        )

        if result:
            results[model_name] = result

    # Print comparison
    print("""
📊 MODEL COMPARISON""")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Confidence':<12}")
    print("-" * 60)

    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['top_1']:<8.4f} {metrics['top_3']:<8.4f} {metrics['top_5']:<8.4f} {metrics['mean_confidence']:<12.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test real-time gesture recognition')
    parser.add_argument('--model', type=str,
                       help='Path to trained CNN model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--duration', type=int, default=30,
                       help='Test duration in seconds')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable OpenCV display window')
    parser.add_argument('--benchmark', nargs='+', type=str,
                       help='Paths to multiple models for comparison')

    args = parser.parse_args()

    if args.benchmark:
        # Benchmark multiple models
        benchmark_models(args.benchmark, args.camera, args.duration)
    else:
        # Test single model or rule-based
        test_realtime_gesture_recognition(
            model_path=args.model,
            camera_index=args.camera,
            display_window=not args.no_display,
            test_duration=args.duration
        )
