import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from av.video.frame import VideoFrame
import time
from models.rule_based_classifier import RuleBasedGestureClassifier

class MediaPipeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.classifier = RuleBasedGestureClassifier()

        self.last_prediction = None
        self.last_prediction_time = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Extract x,y coordinates for 21 landmarks (42 features)
            features = []
            for lm in hand.landmark:
                features.extend([lm.x, lm.y])

            features = np.array(features, dtype=np.float32)

            # Classify gesture
            result = self.classifier.classify_gesture(features)
            if result:
                gesture_label, confidence = result
                self.last_prediction = {
                    "label": gesture_label,
                    "confidence": confidence
                }
                self.last_prediction_time = time.time()

            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                img,
                hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )

        return VideoFrame.from_ndarray(img, format="bgr24")
