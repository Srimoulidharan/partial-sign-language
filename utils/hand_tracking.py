import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
import math

class HandTracker:
    """Hand tracking and landmark detection using MediaPipe"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Hand tracking
        
        Args:
            static_image_mode: Whether to process static images or video stream
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark connections for drawing
        self.landmark_connections = self.mp_hands.HAND_CONNECTIONS
        
        print(f"✅ Hand tracker initialized with {max_num_hands} max hands")
    
    def process_frame(self, frame: np.ndarray) -> Optional[List]:
        """
        Process a single frame and detect hand landmarks
        
        Args:
            frame: Input frame from camera
            
        Returns:
            List of hand landmarks or None if no hands detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks
            
            return None
            
        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks_list: List) -> np.ndarray:
        """
        Draw hand landmarks on the frame
        
        Args:
            frame: Input frame
            hand_landmarks_list: List of detected hand landmarks
            
        Returns:
            Annotated frame with landmarks
        """
        try:
            annotated_frame = frame.copy()
            
            for hand_landmarks in hand_landmarks_list:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.landmark_connections,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            return annotated_frame
            
        except Exception as e:
            print(f"❌ Error drawing landmarks: {str(e)}")
            return frame
    
    def extract_features(self, hand_landmarks_list: List) -> Optional[np.ndarray]:
        """
        Extract numerical features from hand landmarks
        
        Args:
            hand_landmarks_list: List of detected hand landmarks
            
        Returns:
            Feature array (42,) for single hand or (84,) for two hands
        """
        try:
            if not hand_landmarks_list:
                return None
            
            features = []
            
            # Process up to 2 hands
            for i, hand_landmarks in enumerate(hand_landmarks_list[:2]):
                hand_features = []
                
                # Extract x, y coordinates for each landmark (21 landmarks)
                for landmark in hand_landmarks.landmark:
                    hand_features.extend([landmark.x, landmark.y])
                
                features.extend(hand_features)
            
            # If only one hand, pad with zeros for consistency
            if len(hand_landmarks_list) == 1:
                features.extend([0.0] * 42)  # Pad for second hand
            
            return np.array(features[:84])  # Ensure consistent size
            
        except Exception as e:
            print(f"❌ Error extracting features: {str(e)}")
            return None
    
    def extract_single_hand_features(self, hand_landmarks_list: List) -> Optional[np.ndarray]:
        """
        Extract features from the first detected hand only
        
        Args:
            hand_landmarks_list: List of detected hand landmarks
            
        Returns:
            Feature array (42,) for single hand
        """
        try:
            if not hand_landmarks_list:
                return None
            
            # Use only the first hand
            hand_landmarks = hand_landmarks_list[0]
            features = []
            
            # Extract x, y coordinates for each landmark
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y])
            
            return np.array(features)
            
        except Exception as e:
            print(f"❌ Error extracting single hand features: {str(e)}")
            return None
    
    def calculate_hand_angles(self, hand_landmarks) -> List[float]:
        """
        Calculate angles between finger joints
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe
            
        Returns:
            List of calculated angles
        """
        try:
            angles = []
            landmarks = hand_landmarks.landmark
            
            # Define finger joint indices (thumb, index, middle, ring, pinky)
            finger_joints = [
                [1, 2, 3, 4],    # Thumb
                [5, 6, 7, 8],    # Index
                [9, 10, 11, 12], # Middle
                [13, 14, 15, 16], # Ring
                [17, 18, 19, 20]  # Pinky
            ]
            
            for finger in finger_joints:
                for i in range(len(finger) - 2):
                    # Get three consecutive points
                    p1 = landmarks[finger[i]]
                    p2 = landmarks[finger[i + 1]]
                    p3 = landmarks[finger[i + 2]]
                    
                    # Calculate angle
                    angle = self._calculate_angle(p1, p2, p3)
                    angles.append(angle)
            
            return angles
            
        except Exception as e:
            print(f"❌ Error calculating angles: {str(e)}")
            return []
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """
        Calculate angle between three points
        
        Args:
            p1, p2, p3: Three points (p2 is the vertex)
            
        Returns:
            Angle in degrees
        """
        try:
            # Vectors
            v1 = [p1.x - p2.x, p1.y - p2.y]
            v2 = [p3.x - p2.x, p3.y - p2.y]
            
            # Calculate angle using dot product
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                return 0.0
            
            cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
            
            angle_radians = math.acos(cos_angle)
            angle_degrees = math.degrees(angle_radians)
            
            return angle_degrees
            
        except Exception as e:
            return 0.0
    
    def get_hand_bounding_box(self, hand_landmarks, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box for detected hand
        
        Args:
            hand_landmarks: Hand landmarks
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Bounding box coordinates (x_min, y_min, x_max, y_max)
        """
        try:
            landmarks = hand_landmarks.landmark
            height, width = frame_shape[:2]
            
            x_coordinates = [landmark.x * width for landmark in landmarks]
            y_coordinates = [landmark.y * height for landmark in landmarks]
            
            x_min = int(min(x_coordinates))
            x_max = int(max(x_coordinates))
            y_min = int(min(y_coordinates))
            y_max = int(max(y_coordinates))
            
            return x_min, y_min, x_max, y_max
            
        except Exception as e:
            print(f"❌ Error calculating bounding box: {str(e)}")
            return None
    
    def is_hand_closed(self, hand_landmarks) -> bool:
        """
        Determine if hand is in a closed/fist position
        
        Args:
            hand_landmarks: Hand landmarks
            
        Returns:
            True if hand appears closed
        """
        try:
            landmarks = hand_landmarks.landmark
            
            # Check if fingertips are below their respective PIP joints
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
            finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
            
            closed_fingers = 0
            
            for tip, pip in zip(finger_tips, finger_pips):
                if landmarks[tip].y > landmarks[pip].y:  # Tip below PIP (closed)
                    closed_fingers += 1
            
            # Consider hand closed if 4 or more fingers are closed
            return closed_fingers >= 4
            
        except Exception as e:
            return False
    
    def get_gesture_confidence(self, hand_landmarks) -> float:
        """
        Calculate a confidence score for the detected gesture

        Args:
            hand_landmarks: Hand landmarks

        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            landmarks = hand_landmarks.landmark

            # Calculate confidence based on landmark visibility and consistency
            visibility_scores = []

            for landmark in landmarks:
                # MediaPipe provides visibility score for each landmark
                visibility = getattr(landmark, 'visibility', 1.0)
                visibility_scores.append(visibility)

            # Average visibility as confidence
            confidence = np.mean(visibility_scores)

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            print(f"❌ Error calculating confidence: {str(e)}")
            return 0.0

    def process_video_frames(self, video_path: str, max_frames: int = 100) -> List[np.ndarray]:
        """
        Process multiple frames from a video file for batch processing

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process

        Returns:
            List of processed frames (BGR format)
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame.copy())
                frame_count += 1

            cap.release()
            return frames

        except Exception as e:
            print(f"❌ Error processing video {video_path}: {str(e)}")
            return []

    def extract_batch_hand_features(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract hand features from multiple frames in batch

        Args:
            frames: List of frames to process

        Returns:
            List of hand feature arrays (42-dim vectors)
        """
        batch_features = []

        try:
            for frame in frames:
                # Process frame for hand landmarks
                hand_landmarks_list = self.process_frame(frame)

                if hand_landmarks_list:
                    # Extract features from first hand
                    features = self.extract_single_hand_features(hand_landmarks_list)
                    if features is not None:
                        batch_features.append(features)
                else:
                    # No hands detected, append zeros
                    batch_features.append(np.zeros(42))

            return batch_features

        except Exception as e:
            print(f"❌ Error in batch feature extraction: {str(e)}")
            return []

    def cleanup(self):

        """Clean up MediaPipe resources"""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
            print("✅ Hand tracker cleaned up")
        except Exception as e:
            print(f"❌ Error during cleanup: {str(e)}")
