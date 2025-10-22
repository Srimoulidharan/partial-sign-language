import streamlit as st
import cv2
import numpy as np
import threading
import time
from PIL import Image
import io
import base64

# Import custom modules
from models.gesture_recognition import GestureRecognizer
from utils.hand_tracking import HandTracker
from utils.speech_processing import SpeechProcessor
from utils.nlp_cleanup import NLPCleaner
from utils.video_processing import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="Two-Way Sign Language Communication System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'gesture_recognizer' not in st.session_state:
    st.session_state.gesture_recognizer = GestureRecognizer()
if 'hand_tracker' not in st.session_state:
    st.session_state.hand_tracker = HandTracker()
if 'speech_processor' not in st.session_state:
    st.session_state.speech_processor = SpeechProcessor()
if 'nlp_cleaner' not in st.session_state:
    st.session_state.nlp_cleaner = NLPCleaner()
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Sign to Speech"
if 'translation_output' not in st.session_state:
    st.session_state.translation_output = ""
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []

def main():
    st.title("🤟 Two-Way Sign Language Communication System")
    st.markdown("Real-time sign language translation with gesture recognition and speech processing")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Mode selection
        mode = st.selectbox(
            "Translation Mode",
            ["Sign to Speech", "Speech to Sign"],
            index=0 if st.session_state.current_mode == "Sign to Speech" else 1
        )
        
        if mode != st.session_state.current_mode:
            st.session_state.current_mode = mode
            st.rerun()
        
        st.divider()
        
        # Camera controls
        st.subheader("📹 Camera Settings")
        camera_enabled = st.checkbox("Enable Camera", value=True)
        
        if camera_enabled:
            if st.button("🔴 Start/Stop Detection", type="primary"):
                st.session_state.is_running = not st.session_state.is_running
                st.rerun()
        
        # Detection sensitivity
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        st.divider()
        
        # Audio controls for Speech to Sign mode
        if st.session_state.current_mode == "Speech to Sign":
            st.subheader("🎤 Speech Input")
            if st.button("🎙️ Listen for Speech"):
                with st.spinner("Listening for speech..."):
                    speech_text = st.session_state.speech_processor.speech_to_text()
                    if speech_text:
                        cleaned_text = st.session_state.nlp_cleaner.clean_text(speech_text)
                        st.session_state.translation_output = cleaned_text
                        st.success(f"Recognized: {speech_text}")
                        st.rerun()
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.gesture_history = []
            st.session_state.translation_output = ""
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Video Feed")
        video_placeholder = st.empty()
        
        # Display current status
        status_container = st.container()
        with status_container:
            if st.session_state.is_running:
                st.success("🟢 Detection Active")
            else:
                st.info("🔴 Detection Stopped")
    
    with col2:
        st.subheader("📝 Translation Output")
        
        # Current mode display
        st.info(f"**Mode:** {st.session_state.current_mode}")
        
        # Translation output
        translation_container = st.container()
        with translation_container:
            if st.session_state.translation_output:
                st.markdown(f"**Latest Translation:**")
                st.text_area(
                    "",
                    value=st.session_state.translation_output,
                    height=100,
                    disabled=True,
                    key="translation_display"
                )
                
                # Text-to-speech button for Sign to Speech mode
                if st.session_state.current_mode == "Sign to Speech":
                    if st.button("🔊 Speak Translation"):
                        st.session_state.speech_processor.text_to_speech(
                            st.session_state.translation_output
                        )
            else:
                st.info("No translation available yet")
        
        # Gesture history
        st.subheader("📊 Recent Gestures")
        history_container = st.container()
        with history_container:
            if st.session_state.gesture_history:
                for i, gesture in enumerate(reversed(st.session_state.gesture_history[-10:])):
                    st.text(f"{i+1}. {gesture['gesture']} ({gesture['confidence']:.2f})")
            else:
                st.info("No gestures detected yet")
    
    # Video processing loop
    if camera_enabled and st.session_state.is_running:
        process_video_feed(video_placeholder, confidence_threshold)

def process_video_feed(video_placeholder, confidence_threshold):
    """Process video feed and perform gesture recognition"""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera connection.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    gesture_sequence = []
    
    try:
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for hand tracking
            hand_landmarks = st.session_state.hand_tracker.process_frame(frame)
            
            if hand_landmarks:
                # Draw hand landmarks
                annotated_frame = st.session_state.hand_tracker.draw_landmarks(
                    frame, hand_landmarks
                )
                
                # Extract features for gesture recognition
                features = st.session_state.hand_tracker.extract_features(hand_landmarks)
                
                if features is not None:
                    # Static gesture recognition (CNN)
                    static_prediction = st.session_state.gesture_recognizer.predict_static_gesture(
                        features, confidence_threshold
                    )
                    
                    if static_prediction:
                        gesture_name, confidence = static_prediction
                        
                        # Add to gesture history
                        gesture_entry = {
                            'gesture': gesture_name,
                            'confidence': confidence,
                            'timestamp': time.time()
                        }
                        st.session_state.gesture_history.append(gesture_entry)
                        
                        # Update translation output for Sign to Speech mode
                        if st.session_state.current_mode == "Sign to Speech":
                            # Build sentence from recent gestures
                            recent_gestures = [g['gesture'] for g in st.session_state.gesture_history[-5:]]
                            sentence = st.session_state.nlp_cleaner.build_sentence(recent_gestures)
                            st.session_state.translation_output = sentence
                        
                        # Draw gesture label on frame
                        cv2.putText(
                            annotated_frame,
                            f"{gesture_name} ({confidence:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                    
                    # Collect sequence for LSTM
                    gesture_sequence.append(features)
                    if len(gesture_sequence) > 30:  # Keep last 30 frames
                        gesture_sequence.pop(0)
                    
                    # Dynamic gesture recognition (LSTM) every 10 frames
                    if frame_count % 10 == 0 and len(gesture_sequence) >= 15:
                        dynamic_prediction = st.session_state.gesture_recognizer.predict_dynamic_gesture(
                            gesture_sequence, confidence_threshold
                        )
                        
                        if dynamic_prediction:
                            dynamic_gesture, confidence = dynamic_prediction
                            cv2.putText(
                                annotated_frame,
                                f"Dynamic: {dynamic_gesture} ({confidence:.2f})",
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2
                            )
            else:
                annotated_frame = frame
                # Clear gesture sequence when no hands detected
                gesture_sequence = []
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            frame_count += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.03)
            
    except Exception as e:
        st.error(f"❌ Error during video processing: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
