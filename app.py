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
        st.subheader("📷 Camera Feed")
        
        # Display current status
        if st.session_state.is_running:
            st.success("🟢 Detection Active")
        else:
            st.info("🔴 Detection Stopped - Click Start/Stop Detection to begin")
        
        # Camera input for Sign to Speech mode
        if camera_enabled and st.session_state.current_mode == "Sign to Speech":
            st.info("👋 Make a gesture in front of your camera")
            
            # Use camera input to capture frames
            camera_photo = st.camera_input(
                "Webcam Feed",
                key="camera_feed",
                disabled=not st.session_state.is_running
            )
            
            if camera_photo and st.session_state.is_running:
                # Process the captured frame
                process_camera_frame(camera_photo, confidence_threshold)
        
        elif st.session_state.current_mode == "Speech to Sign":
            st.info("🎤 Use the 'Listen for Speech' button in the sidebar to speak")
            
            # Display the recognized text as visual gestures
            if st.session_state.translation_output:
                st.markdown("### 📝 Recognized Gestures:")
                words = st.session_state.translation_output.lower().split()
                
                # Show gesture visualization
                gesture_cols = st.columns(min(len(words), 4))
                for idx, word in enumerate(words[:12]):  # Limit to 12 words
                    col_idx = idx % 4
                    with gesture_cols[col_idx]:
                        st.markdown(f"**{word}**")
                        # You could add gesture images/animations here
                        st.caption(f"Gesture: {word}")
    
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

def process_camera_frame(camera_photo, confidence_threshold):
    """Process a single frame from camera input and perform gesture recognition"""
    
    try:
        # Convert uploaded photo to OpenCV format
        from PIL import Image
        import io
        
        # Read the image
        image = Image.open(camera_photo)
        
        # Convert PIL image to numpy array
        frame = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process frame for hand tracking
        hand_landmarks = st.session_state.hand_tracker.process_frame(frame)
        
        if hand_landmarks:
            # Draw hand landmarks
            annotated_frame = st.session_state.hand_tracker.draw_landmarks(
                frame, hand_landmarks
            )
            
            # Extract features for gesture recognition (single hand for simplicity)
            features = st.session_state.hand_tracker.extract_single_hand_features(hand_landmarks)
            
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
                    
                    # Display detection result
                    st.success(f"✅ Detected: {gesture_name} (Confidence: {confidence:.2%})")
                else:
                    st.warning("⚠️ Hand detected but confidence too low")
        else:
            st.warning("⚠️ No hands detected in the image")
            
    except Exception as e:
        st.error(f"❌ Error processing camera frame: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
