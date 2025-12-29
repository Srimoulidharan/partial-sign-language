import os
import numpy as np
import pickle
from typing import Optional, Tuple, Dict, List
from models.cnn_model import CNNGestureModel
from models.lstm_model import LSTMGestureModel
from data.gesture_labels import GESTURE_TO_TEXT, get_gesture_info
from utils.nlp_cleanup import NLPCleaner

class GestureRecognizer:
    """Main gesture recognition interface integrating multiple models"""
    
    def __init__(self, 
                 use_cnn: bool = True,
                 use_lstm: bool = False,
                 cnn_model_path: Optional[str] = None,
                 lstm_model_path: Optional[str] = None,
                 gesture_labels_path: str = 'data/gesture_labels.py',
                 confidence_threshold: float = 0.5):
        """
        Initialize gesture recognizer
        
        Args:
            use_cnn: Whether to use CNN for static gestures
            use_lstm: Whether to use LSTM for dynamic gestures
            cnn_model_path: Path to trained CNN model
            lstm_model_path: Path to trained LSTM model
            gesture_labels_path: Path to gesture labels
            confidence_threshold: Minimum confidence for gesture detection
        """
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        self.confidence_threshold = confidence_threshold
        
        # Initialize models
        self.cnn_model = None
        self.lstm_model = None
        
        if use_cnn:
            self.cnn_model = CNNGestureModel()
            if cnn_model_path and os.path.exists(cnn_model_path):
                self.cnn_model.load_model(cnn_model_path)
                print(f"✅ CNN model loaded from {cnn_model_path}")
            else:
                print("⚠️ CNN model not loaded - will use rule-based fallback")
        
        if use_lstm:
            self.lstm_model = LSTMGestureModel()
            if lstm_model_path and os.path.exists(lstm_model_path):
                self.lstm_model.load_model(lstm_model_path)
                print(f"✅ LSTM model loaded from {lstm_model_path}")
            else:
                print("⚠️ LSTM model not loaded - dynamic gestures unavailable")
        
        # Load gesture labels
        self.gesture_to_text = GESTURE_TO_TEXT
        self.gesture_info = get_gesture_info
        
        # NLP cleaner for sentence building
        self.nlp_cleaner = NLPCleaner()
        
        # Rule-based fallback
        from models.rule_based_classifier import RuleBasedGestureClassifier
        self.rule_based_classifier = RuleBasedGestureClassifier()
        
        print("✅ GestureRecognizer initialized")
    
    def recognize_static_gesture(self, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Recognize a static gesture using CNN or rule-based fallback
        
        Args:
            features: Hand landmark features (42,)
            
        Returns:
            Tuple of (gesture_name, confidence) or None
        """
        if features.shape[0] != 42:
            return None
        
        # Try CNN model first
        if self.use_cnn and self.cnn_model and self.cnn_model.is_trained:
            try:
                # Reshape for prediction (add batch dimension)
                features_batch = features.reshape(1, -1)
                predictions = self.cnn_model.predict(features_batch, batch_size=1)
                
                confidence = np.max(predictions)
                if confidence >= self.confidence_threshold:
                    gesture_idx = np.argmax(predictions)
                    # Map index to gesture name (assumes model was trained with same order)
                    gesture_name = self._idx_to_gesture_name(gesture_idx)
                    return gesture_name, confidence
            except Exception as e:
                print(f"❌ CNN prediction error: {str(e)}")
        
        # Fallback to rule-based
        print("🔄 Using rule-based classifier (CNN unavailable)")
        return self.rule_based_classifier.classify_gesture(features)
    
    def recognize_dynamic_gesture(self, feature_sequence: List[np.ndarray]) -> Optional[Tuple[str, float]]:
        """
        Recognize a dynamic gesture sequence using LSTM
        
        Args:
            feature_sequence: List of hand landmark features [(42,), (42,), ...]
            
        Returns:
            Tuple of (gesture_name, confidence) or None
        """
        if not self.use_lstm or not self.lstm_model or not self.lstm_model.is_trained:
            print("⚠️ LSTM model not available for dynamic gestures")
            return None
        
        try:
            # Convert to numpy array
            sequence_array = np.array(feature_sequence)
            
            if len(sequence_array) == 0:
                return None
            
            # Pad/truncate to expected sequence length
            expected_length = self.lstm_model.sequence_length
            if len(sequence_array) < expected_length:
                padding = np.zeros((expected_length - len(sequence_array), 42))
                sequence_array = np.vstack([sequence_array, padding])
            else:
                sequence_array = sequence_array[:expected_length]
            
            # Reshape for prediction
            sequence_batch = sequence_array.reshape(1, expected_length, 42)
            
            predictions = self.lstm_model.predict([sequence_batch], batch_size=1)
            
            confidence = np.max(predictions)
            if confidence >= self.confidence_threshold:
                gesture_idx = np.argmax(predictions)
                gesture_name = self._idx_to_gesture_name(gesture_idx)
                return gesture_name, confidence
            
            return None
            
        except Exception as e:
            print(f"❌ LSTM prediction error: {str(e)}")
            return None
    
    def recognize_gestures_from_sequence(self, feature_sequence: List[np.ndarray], 
                                       window_size: int = 10) -> List[Tuple[str, float]]:
        """
        Recognize gestures from a sequence of features using sliding window
        
        Args:
            feature_sequence: List of hand landmark features
            window_size: Size of sliding window for static gesture detection
            
        Returns:
            List of recognized gestures with timestamps/confidence
        """
        recognized_gestures = []
        
        for i in range(0, len(feature_sequence) - window_size + 1, window_size // 2):
            window_features = feature_sequence[i:i + window_size]
            
            # Average features in window for static gesture
            avg_features = np.mean(window_features, axis=0)
            
            # Recognize static gesture
            gesture_result = self.recognize_static_gesture(avg_features)
            if gesture_result:
                gesture_name, confidence = gesture_result
                recognized_gestures.append({
                    'gesture': gesture_name,
                    'confidence': confidence,
                    'timestamp': i / len(feature_sequence),  # Normalized timestamp
                    'frame_range': (i, min(i + window_size, len(feature_sequence)))
                })
        
        return recognized_gestures
    
    def build_sentence_from_gestures(self, gestures: List[Tuple[str, float]]) -> str:
        """
        Build a natural language sentence from recognized gestures
        
        Args:
            gestures: List of (gesture_name, confidence) tuples
            
        Returns:
            Constructed sentence
        """
        if not gestures:
            return "No gestures recognized"
        
        # Extract gesture names with high confidence
        valid_gestures = [name for name, conf in gestures if conf > self.confidence_threshold]
        
        if not valid_gestures:
            return "Low confidence gestures"
        
        # Use NLP cleaner to build sentence
        sentence = self.nlp_cleaner.build_sentence(valid_gestures, self.confidence_threshold)
        
        return sentence if sentence else " ".join(valid_gestures)
    
    def get_gesture_description(self, gesture_name: str) -> Dict:
        """
        Get detailed information about a gesture
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            Dictionary with gesture information
        """
        text_translation = self.gesture_to_text.get(gesture_name, gesture_name.replace('_', ' '))
        info = self.gesture_info(gesture_name)
        
        return {
            'name': gesture_name,
            'text': text_translation,
            'description': info.get('description', 'No description available'),
            'category': info.get('category', 'unknown'),
            'difficulty': info.get('difficulty', 'unknown'),
            'is_static': info.get('is_static', True),
            'is_dynamic': info.get('is_dynamic', False)
        }
    
    def _idx_to_gesture_name(self, idx: int) -> str:
        """
        Convert model prediction index to gesture name
        (Assumes model was trained with gesture_to_idx from dataset)
        """
        # This needs to match the training order
        # In production, save the class mapping during training
        try:
            # Load from training metadata if available
            # For now, use a default mapping from gesture_labels
            default_classes = list(GESTURE_TO_TEXT.keys())
            if 0 <= idx < len(default_classes):
                return default_classes[idx]
            return f"unknown_{idx}"
        except:
            return "unknown"
    
    def save_model_info(self, save_dir: str):
        """Save model configuration and class mappings"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            model_info = {
                'use_cnn': self.use_cnn,
                'use_lstm': self.use_lstm,
                'confidence_threshold': self.confidence_threshold,
                'gesture_to_text': self.gesture_to_text,
                'num_classes': len(self.gesture_to_text),
                'class_names': list(self.gesture_to_text.keys()),
                'cnn_model_path': getattr(self.cnn_model, 'model_path', None) if self.cnn_model else None,
                'lstm_model_path': getattr(self.lstm_model, 'model_path', None) if self.lstm_model else None
            }
            
            info_path = os.path.join(save_dir, 'model_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"✅ Model info saved to {info_path}")
            
        except Exception as e:
            print(f"❌ Error saving model info: {str(e)}")
    
    def load_model_info(self, info_path: str):
        """Load model configuration from saved info"""
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            
            self.confidence_threshold = model_info.get('confidence_threshold', 0.5)
            self.gesture_to_text = dict(model_info.get('gesture_to_text', {}))
            
            print(f"✅ Model info loaded from {info_path}")
            
        except Exception as e:
            print(f"❌ Error loading model info: {str(e)}")

# Rule-based fallback (for when ML models are not available)
class RuleBasedRecognizer:
    """Wrapper for rule-based gesture recognition"""
    
    def __init__(self):
        from models.rule_based_classifier import RuleBasedGestureClassifier
        self.classifier = RuleBasedGestureClassifier()
    
    def recognize(self, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """Recognize gesture using rule-based method"""
        return self.classifier.classify_gesture(features)
    
    def get_description(self, gesture_name: str) -> str:
        """Get gesture description"""
        from models.rule_based_classifier import RuleBasedGestureClassifier
        classifier = RuleBasedGestureClassifier()
        return classifier.get_gesture_description(gesture_name)

if __name__ == "__main__":
    # Test the recognizer
    recognizer = GestureRecognizer(use_cnn=False, use_lstm=False)  # Use rule-based for testing
    
    # Test with sample features (42-dim)
    sample_features = np.random.rand(42)
    result = recognizer.recognize_static_gesture(sample_features)
    
    if result:
        gesture, conf = result
        print(f"Recognized: {gesture} (confidence: {conf:.2f})")
        print(f"Description: {recognizer.get_gesture_description(gesture)}")
    else:
        print("No gesture recognized")
