# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# from utils.mediapipe_processor import MediaPipeProcessor
# from utils.speech_processing import SpeechProcessor
# from utils.nlp_cleanup import NLPCleaner
# from utils.speech_audio_processor import SpeechAudioProcessor
# from assets.gesture_svgs import get_svg_for_word

# # Set the Streamlit title and basic layout
# st.set_page_config(
#     page_title="Two-Way Sign Language Communication System",
#     layout="wide"
# )

# # Initialize session state
# if 'current_mode' not in st.session_state:
#     st.session_state.current_mode = "Sign to Speech"
# if 'translation_output' not in st.session_state:
#     st.session_state.translation_output = ""
# if 'speech_processor' not in st.session_state:
#     st.session_state.speech_processor = SpeechProcessor()
# if 'nlp_cleaner' not in st.session_state:
#     st.session_state.nlp_cleaner = NLPCleaner()

# st.title("🤟 Two-Way Sign Language Communication System")
# st.markdown("Real-time sign language translation with gesture recognition and speech processing")

# # Sidebar controls
# with st.sidebar:
#     st.header("Controls")
    
#     # Mode selection
#     mode = st.selectbox(
#         "Translation Mode",
#         ["Sign to Speech", "Speech to Sign"],
#         index=0 if st.session_state.current_mode == "Sign to Speech" else 1
#     )
    
#     if mode != st.session_state.current_mode:
#         st.session_state.current_mode = mode
#         st.rerun()
    
#     st.divider()
    
#     # Audio controls for Speech to Sign mode
#     if st.session_state.current_mode == "Speech to Sign":
#         st.subheader("🎤 Speech Input")
#         if st.button("🎙️ Listen for Speech"):
#             with st.spinner("Listening for speech..."):
#                 speech_text = st.session_state.speech_processor.speech_to_text()
#                 if speech_text:
#                     cleaned_text = st.session_state.nlp_cleaner.clean_text(speech_text)
#                     st.session_state.translation_output = cleaned_text
#                     st.success(f"Recognized: {speech_text}")
#                     st.rerun()
    
#     # Clear history
#     if st.button("Clear History"):
#         st.session_state.translation_output = ""
#         st.rerun()

# # Main content area
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("Camera Feed / Input")
    
#     if st.session_state.current_mode == "Sign to Speech":
#         st.info("Make a gesture in front of your camera")
        
#         # Implement the webrtc_stream component
#         webrtc_ctx = webrtc_streamer(
#             key="sign-language-recognition",
#             video_processor_factory=MediaPipeProcessor,
#             rtc_configuration={
#                 "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#             },
#             media_stream_constraints={
#                 "video": True,
#                 "audio": False
#             }
#         )

#         # Update session state from processor
#         if webrtc_ctx.video_processor:
#             prediction_result = webrtc_ctx.video_processor.prediction_result
#             if prediction_result is not None:
#                 label = prediction_result['label']
#                 confidence = prediction_result['confidence']
#                 st.session_state.translation_output = f"{label} (Confidence: {confidence:.2f})"
#                 st.success(f"Detected: {label} (Confidence: {confidence:.2f})")
#             else:
#                 st.info("Gathering sequence data...")
    
#     elif st.session_state.current_mode == "Speech to Sign":
#         st.info("Speak continuously - gestures will update in real-time")

#         # Implement audio streaming for continuous speech recognition
#         audio_ctx = webrtc_streamer(
#             key="speech-to-sign",
#             audio_processor_factory=SpeechAudioProcessor,
#             rtc_configuration={
#                 "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#             },
#             media_stream_constraints={
#                 "video": False,
#                 "audio": True
#             }
#         )

#         # Update session state from audio processor
#         if audio_ctx.audio_processor:
#             partial_text = audio_ctx.audio_processor.get_partial_text()
#             if partial_text:
#                 st.session_state.translation_output = partial_text
#                 st.markdown("### 🖐️ Sign Language Signs:")
#                 st.info("Real-time visual hand positions for your speech.")
#                 words = partial_text.lower().split()

#                 # Show gesture visualization with SVGs
#                 num_cols = min(len(words), 4)
#                 gesture_cols = st.columns(num_cols)
#                 displayed_words = words[:12]  # Limit to 12 words

#                 for idx, word in enumerate(displayed_words):
#                     col_idx = idx % num_cols
#                     with gesture_cols[col_idx]:
#                         svg = get_svg_for_word(word)
#                         st.markdown(f'<div style="text-align: center; margin: 10px;">{svg}</div>', unsafe_allow_html=True)
#                         st.caption(f"Sign for: {word}")
#             else:
#                 st.info("Start speaking to see gestures...")
#         else:
#             st.info("Initializing audio recognition...")

# with col2:
#     st.subheader("Translation Output")
    
#     # Current mode display
#     st.info(f"**Mode:** {st.session_state.current_mode}")
    
#     # Translation output
#     translation_container = st.container()
#     with translation_container:
#         if st.session_state.translation_output:
#             st.markdown("**Latest Translation:**")
#             st.text_area(
#                 "",
#                 value=st.session_state.translation_output,
#                 height=100,
#                 disabled=True,
#                 key="translation_display"
#             )
            
#             # Text-to-speech button for Sign to Speech mode
#             if st.session_state.current_mode == "Sign to Speech":
#                 if st.button("Speak Translation"):
#                     st.session_state.speech_processor.text_to_speech(
#                         st.session_state.translation_output
#                     )
#         else:
#             st.info("No translation available yet")
import streamlit as st
import time
import os
import cv2
import re
from streamlit_webrtc import webrtc_streamer

from utils.mediapipe_processor import MediaPipeProcessor
from utils.speech_audio_processor import SpeechAudioProcessor
from assets.gesture_svgs import get_svg_for_word, get_svg_for_gesture
from data.gesture_labels import GESTURE_TO_TEXT, TEXT_TO_GESTURE
from models.gesture_recognition import GestureRecognizer

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Two-Way Sign Language Communication System",
    layout="wide"
)

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
st.session_state.setdefault("mode", "Sign to Speech")
st.session_state.setdefault("translation_output", "")
st.session_state.setdefault("accumulated_gestures", [])
st.session_state.setdefault("gesture_recognizer", GestureRecognizer(use_lstm=True, lstm_model_path='models/lstm_model.h5'))

# -------------------------------------------------
# UI HEADER
# -------------------------------------------------
st.title("🤟 Two-Way Sign Language Communication System")
st.markdown("Real-time sign language and speech translation")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Controls")

    mode = st.selectbox(
        "Translation Mode",
        ["Sign to Speech", "Speech to Sign"],
        index=0 if st.session_state.mode == "Sign to Speech" else 1
    )

    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.translation_output = ""
        st.rerun()

    if st.button("Clear Sentence"):
        st.session_state.translation_output = ""
        st.session_state.accumulated_gestures = []
        st.rerun()

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
col1, col2 = st.columns([2, 1])

# -------------------------------------------------
# LEFT PANEL – INPUT
# -------------------------------------------------
with col1:
    st.subheader("Live Input")

    # -------- Sign to Speech --------
    if st.session_state.mode == "Sign to Speech":
        ctx = webrtc_streamer(
            key="sign-stream",
            video_processor_factory=MediaPipeProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            vp = ctx.video_processor
            if (
                vp.last_prediction
                and time.time() - vp.last_prediction_time < 1.0
            ):
                gesture_label = vp.last_prediction['label']
                confidence = vp.last_prediction['confidence']

                # Accumulate gestures
                if not st.session_state.accumulated_gestures or st.session_state.accumulated_gestures[-1][0] != gesture_label:
                    st.session_state.accumulated_gestures.append((gesture_label, confidence))
                    print(f"Accumulated gesture: {gesture_label} (confidence: {confidence:.2f})")
                    print(f"Current accumulated gestures: {st.session_state.accumulated_gestures}")

                # Build sentence
                sentence = st.session_state.gesture_recognizer.build_sentence_from_gestures(st.session_state.accumulated_gestures)
                print(f"Built sentence: {sentence}")
                st.session_state.translation_output = sentence
                st.rerun()  # Force UI update


    # -------- Speech to Sign --------
    else:
        ctx = webrtc_streamer(
            key="speech-stream",
            audio_processor_factory=SpeechAudioProcessor,
            media_stream_constraints={"video": False, "audio": True},
        )

        if ctx.audio_processor:
            text = ctx.audio_processor.get_partial_text()
            if text:
                st.session_state.translation_output = text

# -------------------------------------------------
# RIGHT PANEL – OUTPUT
# -------------------------------------------------
with col2:
    st.subheader("Translation Output")

    # ✅ FIXED: NON-EMPTY LABEL (ACCESSIBILITY SAFE)
    st.text_area(
        "Translation Output",
        value=st.session_state.translation_output,
        height=120,
        disabled=True,
        label_visibility="collapsed"
    )

    # -------- Gesture Rendering --------
    if st.session_state.mode == "Speech to Sign":
        text = st.session_state.translation_output.lower().strip()
        
        if text:
            st.markdown("### 🖐️ Gestures")
            
            # Clean text: remove extra punctuation but keep structure
            text_clean = re.sub(r'[^\w\s]', ' ', text)
            text_clean = ' '.join(text_clean.split())  # Normalize whitespace
            
            # First, try to match multi-word phrases (longest first)
            gestures_to_show = []
            matched_positions = set()
            
            # Sort phrases by length (longest first) to match "how are you" before "how"
            sorted_phrases = sorted(TEXT_TO_GESTURE.keys(), key=lambda x: (len(x.split()), -len(x)), reverse=True)
            
            # Match phrases first
            for phrase in sorted_phrases:
                phrase_words = phrase.split()
                if len(phrase_words) > 1:  # Multi-word phrases
                    # Use word boundary matching
                    pattern = r'\b' + re.escape(phrase) + r'\b'
                    matches = list(re.finditer(pattern, text_clean))
                    for match in matches:
                        start, end = match.span()
                        # Check if this position overlaps with already matched text
                        overlap = False
                        for pos_start, pos_end in matched_positions:
                            if not (end <= pos_start or start >= pos_end):
                                overlap = True
                                break
                        if not overlap:
                            gesture_name = TEXT_TO_GESTURE[phrase]
                            if gesture_name not in gestures_to_show:
                                gestures_to_show.append(gesture_name)
                                matched_positions.add((start, end))
            
            # Then match individual words from remaining text
            # Create a set of matched words from phrases
            matched_words = set()
            for phrase in sorted_phrases:
                if phrase in text_clean:
                    matched_words.update(phrase.split())
            
            # Match individual words
            words = text_clean.split()
            for word in words:
                word_clean = word.strip()
                if word_clean and word_clean not in matched_words:
                    if word_clean in TEXT_TO_GESTURE:
                        gesture_name = TEXT_TO_GESTURE[word_clean]
                        if gesture_name not in gestures_to_show:
                            gestures_to_show.append(gesture_name)
                    # Don't add unknown words as gestures - they'll show as text
            
            # If no gestures matched, show a message
            if not gestures_to_show:
                st.info("No matching gestures found. Try: hello, goodbye, thank you, yes, no, help, etc.")
            else:
                # Display gestures
                cols = st.columns(min(4, len(gestures_to_show)))
                
                for i, gesture in enumerate(gestures_to_show[:12]):
                    with cols[i % len(cols)]:
                        # Try to get SVG for gesture name first, then fallback to word
                        svg = get_svg_for_gesture(gesture) or get_svg_for_word(gesture)
                        if svg:
                            st.markdown(svg, unsafe_allow_html=True)
                            # Show readable text if available
                            display_text = GESTURE_TO_TEXT.get(gesture, gesture.replace('_', ' ').title())
                            st.caption(display_text)
