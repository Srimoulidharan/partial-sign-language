# import numpy as np
# from streamlit_webrtc import AudioProcessorBase
# import speech_recognition as sr
# import threading
# import time
# import streamlit as st

# class SpeechAudioProcessor(AudioProcessorBase):
#     """Real-time audio processor for continuous speech recognition"""

#     def __init__(self):
#         super().__init__()
#         self.recognizer = sr.Recognizer()
#         self.audio_buffer = []
#         self.is_listening = False
#         self.last_recognition_time = time.time()
#         self.recognition_interval = 2.0  # Recognize every 2 seconds
#         self.partial_text = ""
#         # Start background thread for recognition
#         self.recognition_thread = threading.Thread(target=self._continuous_recognition, daemon=True)
#         self.recognition_thread.start()

#     def recv(self, audio_frame):
#         # Convert audio frame to numpy array
#         audio_data = np.frombuffer(audio_frame.to_ndarray(), dtype=np.int16)
#         # Add to buffer
#         self.audio_buffer.append(audio_data)
#         # Keep buffer size manageable (last 5 seconds at 16kHz)
#         if len(self.audio_buffer) > 80:  # ~5 seconds
#             self.audio_buffer.pop(0)
#         return audio_frame

#     def _continuous_recognition(self):
#         while True:
#             try:
#                 current_time = time.time()
#                 if current_time - self.last_recognition_time > self.recognition_interval and self.audio_buffer:
#                     # Combine recent audio
#                     combined_audio = np.concatenate(self.audio_buffer[-40:])  # Last ~2.5 seconds
#                     # Convert to AudioData
#                     audio_data = sr.AudioData(combined_audio.tobytes(), 16000, 2)
#                     # Recognize speech
#                     try:
#                         text = self.recognizer.recognize_google(audio_data)
#                         if text and text != self.partial_text:
#                             self.partial_text = text
#                             # Update session state
#                             if 'translation_output' in st.session_state:
#                                 st.session_state.translation_output = text
#                             # Trigger rerun (if possible)
#                             # Note: Streamlit rerun might not work directly in thread
#                     except sr.UnknownValueError:
#                         pass  # No speech detected
#                     except sr.RequestError:
#                         pass  # API error
#                     self.last_recognition_time = current_time
#                 time.sleep(0.1)  # Small delay
#             except Exception as e:
#                 print(f"Error in recognition thread: {e}")
#                 time.sleep(1)

#     def get_partial_text(self):
#         return self.partial_text
import numpy as np
from streamlit_webrtc import AudioProcessorBase
import speech_recognition as sr
import threading
import time


class SpeechAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_buffer = []
        self.partial_text = ""
        self.last_time = time.time()

        threading.Thread(
            target=self._recognize_loop,
            daemon=True
        ).start()

    def recv(self, frame):
        audio = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
        self.audio_buffer.append(audio)

        if len(self.audio_buffer) > 80:
            self.audio_buffer.pop(0)

        return frame

    def _recognize_loop(self):
        while True:
            try:
                if time.time() - self.last_time > 2 and self.audio_buffer:
                    audio = np.concatenate(self.audio_buffer[-40:])
                    audio_data = sr.AudioData(audio.tobytes(), 16000, 2)

                    try:
                        text = self.recognizer.recognize_google(audio_data)
                        if text:
                            self.partial_text = text
                    except:
                        pass

                    self.last_time = time.time()

                time.sleep(0.1)
            except:
                time.sleep(1)

    def get_partial_text(self):
        return self.partial_text
