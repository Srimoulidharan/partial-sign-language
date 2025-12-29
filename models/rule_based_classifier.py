import numpy as np
from typing import Optional, Tuple
import math

class RuleBasedGestureClassifier:
    """
    Rule-based gesture classifier that uses hand landmark analysis
    This provides functional gesture recognition for MVP without trained models
    """
    
    def __init__(self):
        self.gesture_rules = {
            'hello': self._is_waving_hand,
            'thumbs_up': self._is_thumbs_up,
            'ok': self._is_ok_sign,
            'stop': self._is_stop_sign,
            'peace': self._is_peace_sign,
            'pointing': self._is_pointing,
            'fist': self._is_fist,
            'open_hand': self._is_open_hand,
        }
        
    def classify_gesture(self, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Classify gesture based on hand landmark features
        
        Args:
            features: Hand landmark features (42,) array with x,y coordinates for 21 landmarks
            
        Returns:
            Tuple of (gesture_name, confidence) or None
        """
        try:
            if features.shape[0] != 42:
                return None
            
            # Reshape features into landmarks (21, 2)
            landmarks = features.reshape(21, 2)
            
            # Try each gesture rule
            best_gesture = None
            best_confidence = 0.0
            
            for gesture_name, rule_func in self.gesture_rules.items():
                confidence = rule_func(landmarks)
                if confidence > best_confidence and confidence > 0.6:  # Minimum threshold
                    best_confidence = confidence
                    best_gesture = gesture_name
            
            if best_gesture:
                return best_gesture, best_confidence
            
            # Fallback to finger counting for basic gestures
            num_fingers = self._count_extended_fingers(landmarks)
            if num_fingers >= 0:
                finger_gesture = self._map_finger_count_to_gesture(num_fingers)
                if finger_gesture:
                    return finger_gesture, 0.75
            
            return None
            
        except Exception as e:
            print(f"Error in rule-based classification: {str(e)}")
            return None
    
    def _count_extended_fingers(self, landmarks: np.ndarray) -> int:
        """Count the number of extended fingers"""
        try:
            # Landmark indices for fingertips and their base points
            # 0: Wrist, 4: Thumb tip, 8: Index tip, 12: Middle tip, 16: Ring tip, 20: Pinky tip
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 6, 10, 14, 18]  # PIP joints (one below tip)
            
            extended = 0
            
            # Check each finger
            for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                tip = landmarks[tip_idx]
                pip = landmarks[pip_idx]
                wrist = landmarks[0]
                
                # Calculate distances
                tip_to_wrist = np.linalg.norm(tip - wrist)
                pip_to_wrist = np.linalg.norm(pip - wrist)
                
                # Finger is extended if tip is farther from wrist than PIP
                if tip_to_wrist > pip_to_wrist * 1.1:  # 10% margin
                    extended += 1
            
            return extended
            
        except:
            return -1
    
    def _map_finger_count_to_gesture(self, count: int) -> Optional[str]:
        """Map finger count to gesture names"""
        mapping = {
            0: 'fist',
            1: 'pointing',
            2: 'peace',
            3: None,  # No specific gesture
            4: None,
            5: 'open_hand',
        }
        return mapping.get(count)
    
    def _is_open_hand(self, landmarks: np.ndarray) -> float:
        """Check if hand is fully open"""
        try:
            extended = self._count_extended_fingers(landmarks)
            if extended == 5:
                return 0.9
            elif extended == 4:
                return 0.7
            return 0.0
        except:
            return 0.0
    
    def _is_fist(self, landmarks: np.ndarray) -> float:
        """Check if hand is in a fist"""
        try:
            extended = self._count_extended_fingers(landmarks)
            if extended == 0:
                return 0.9
            elif extended == 1:
                return 0.6
            return 0.0
        except:
            return 0.0
    
    def _is_pointing(self, landmarks: np.ndarray) -> float:
        """Check if index finger is pointing"""
        try:
            # Index finger extended, others closed
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            wrist = landmarks[0]
            
            index_extended = np.linalg.norm(index_tip - wrist) > np.linalg.norm(index_pip - wrist) * 1.1
            middle_closed = np.linalg.norm(middle_tip - wrist) < np.linalg.norm(middle_pip - wrist) * 1.1
            
            if index_extended and middle_closed:
                return 0.85
            elif index_extended:
                return 0.65
            return 0.0
        except:
            return 0.0
    
    def _is_peace_sign(self, landmarks: np.ndarray) -> float:
        """Check for peace sign (V sign)"""
        try:
            extended = self._count_extended_fingers(landmarks)
            
            # Index and middle finger extended
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            wrist = landmarks[0]
            
            index_extended = np.linalg.norm(index_tip - wrist) > np.linalg.norm(landmarks[6] - wrist) * 1.1
            middle_extended = np.linalg.norm(middle_tip - wrist) > np.linalg.norm(landmarks[10] - wrist) * 1.1
            
            # Check if fingers are spread apart (V shape)
            finger_distance = np.linalg.norm(index_tip - middle_tip)
            
            if extended == 2 and index_extended and middle_extended and finger_distance > 0.05:
                return 0.9
            elif extended == 2:
                return 0.7
            return 0.0
        except:
            return 0.0
    
    def _is_thumbs_up(self, landmarks: np.ndarray) -> float:
        """Check for thumbs up gesture"""
        try:
            # Thumb extended upward, other fingers closed
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            # Thumb should be higher than wrist
            thumb_up = thumb_tip[1] < wrist[1]  # Y coordinate (lower is higher on screen)
            
            # Other fingers should be relatively closed
            extended = self._count_extended_fingers(landmarks)
            
            if thumb_up and extended <= 2:
                return 0.85
            elif thumb_up:
                return 0.6
            return 0.0
        except:
            return 0.0
    
    def _is_ok_sign(self, landmarks: np.ndarray) -> float:
        """Check for OK sign (thumb and index forming circle)"""
        try:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # Distance between thumb and index tip
            distance = np.linalg.norm(thumb_tip - index_tip)
            
            # They should be close together
            if distance < 0.04:  # Very close
                return 0.9
            elif distance < 0.06:  # Moderately close
                return 0.7
            return 0.0
        except:
            return 0.0
    
    def _is_stop_sign(self, landmarks: np.ndarray) -> float:
        """Check for stop sign (palm facing forward)"""
        try:
            # Hand fully open with palm facing forward
            extended = self._count_extended_fingers(landmarks)
            
            # All fingers should be extended
            if extended == 5:
                # Check if palm is facing forward (fingers pointing up)
                middle_tip = landmarks[12]
                middle_mcp = landmarks[9]  # Base of middle finger
                
                # If middle finger is pointing upward
                if middle_tip[1] < middle_mcp[1]:  # Y coordinate
                    return 0.85
                return 0.7
            return 0.0
        except:
            return 0.0
    
    def _is_waving_hand(self, landmarks: np.ndarray) -> float:
        """Check for waving hand (open palm)"""
        try:
            # Similar to open hand but can be at any angle
            extended = self._count_extended_fingers(landmarks)
            
            if extended >= 4:
                return 0.75
            return 0.0
        except:
            return 0.0
    
    def get_gesture_description(self, gesture_name: str) -> str:
        """Get description of how to make a gesture"""
        descriptions = {
            'hello': 'Wave with open palm',
            'thumbs_up': 'Extend thumb upward',
            'ok': 'Make circle with thumb and index finger',
            'stop': 'Open palm facing forward',
            'peace': 'Extend index and middle fingers in V shape',
            'pointing': 'Extend index finger only',
            'fist': 'Close all fingers',
            'open_hand': 'Extend all fingers',
        }
        return descriptions.get(gesture_name, 'Unknown gesture')
