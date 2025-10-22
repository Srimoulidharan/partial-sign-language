import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from typing import Optional, Tuple, List
from data.gesture_labels import GESTURE_LABELS
from models.cnn_model import CNNGestureModel
from models.lstm_model import LSTMGestureModel

class GestureRecognizer:
    """Main gesture recognition system combining CNN and LSTM models"""
    
    def __init__(self):
        self.static_model = CNNGestureModel()
        self.dynamic_model = LSTMGestureModel()
        self.gesture_labels = GESTURE_LABELS
        self.feature_dim = 42  # 21 landmarks * 2 coordinates (x, y)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize both CNN and LSTM models"""
        try:
            # Build and compile static model
            self.static_model.build_model(
                input_shape=(self.feature_dim,),
                num_classes=len(self.gesture_labels)
            )
            
            # Build and compile dynamic model
            self.dynamic_model.build_model(
                input_shape=(None, self.feature_dim),  # Variable sequence length
                num_classes=len(self.gesture_labels)
            )
            
            print("✅ Gesture recognition models initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing models: {str(e)}")
            # Create fallback simple models
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create simple fallback models if initialization fails"""
        print("🔄 Creating fallback models...")
        
        # Simple static model
        self.static_model.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.feature_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.gesture_labels), activation='softmax')
        ])
        
        self.static_model.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Simple dynamic model
        self.dynamic_model.model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(None, self.feature_dim)),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(len(self.gesture_labels), activation='softmax')
        ])
        
        self.dynamic_model.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize with random weights (placeholder for real training)
        self._initialize_with_random_weights()
    
    def _initialize_with_random_weights(self):
        """Initialize models with random weights for demonstration"""
        # This would normally load pre-trained weights
        # For now, we'll use the randomly initialized weights
        
        # Create dummy data to initialize model weights
        dummy_static = np.random.random((1, self.feature_dim))
        dummy_dynamic = np.random.random((1, 15, self.feature_dim))
        
        # Make dummy predictions to initialize weights
        self.static_model.model.predict(dummy_static, verbose=0)
        self.dynamic_model.model.predict(dummy_dynamic, verbose=0)
        
        print("⚠️ Using randomly initialized weights - models need training for production use")
    
    def predict_static_gesture(self, features: np.ndarray, confidence_threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        Predict static gesture from hand landmark features
        
        Args:
            features: Hand landmark features (42,) array
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Tuple of (gesture_name, confidence) or None if below threshold
        """
        try:
            if features.shape != (self.feature_dim,):
                features = features.reshape((self.feature_dim,))
            
            # Normalize features
            features = self._normalize_features(features)
            
            # Make prediction
            features_batch = features.reshape(1, -1)
            predictions = self.static_model.model.predict(features_batch, verbose=0)
            
            # Get best prediction
            best_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][best_idx])
            
            if confidence >= confidence_threshold:
                gesture_name = self.gesture_labels[best_idx]
                return gesture_name, confidence
            
            return None
            
        except Exception as e:
            print(f"Error in static gesture prediction: {str(e)}")
            return None
    
    def predict_dynamic_gesture(self, sequence: List[np.ndarray], confidence_threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        Predict dynamic gesture from sequence of hand landmark features
        
        Args:
            sequence: List of hand landmark features
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Tuple of (gesture_name, confidence) or None if below threshold
        """
        try:
            if len(sequence) < 5:  # Need minimum sequence length
                return None
            
            # Convert to numpy array and normalize
            sequence_array = np.array(sequence)
            if sequence_array.shape[1] != self.feature_dim:
                return None
            
            # Normalize sequence
            sequence_normalized = np.array([self._normalize_features(frame) for frame in sequence_array])
            
            # Reshape for LSTM input
            sequence_batch = sequence_normalized.reshape(1, len(sequence_normalized), self.feature_dim)
            
            # Make prediction
            predictions = self.dynamic_model.model.predict(sequence_batch, verbose=0)
            
            # Get best prediction
            best_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][best_idx])
            
            if confidence >= confidence_threshold:
                gesture_name = f"Dynamic_{self.gesture_labels[best_idx]}"
                return gesture_name, confidence
            
            return None
            
        except Exception as e:
            print(f"Error in dynamic gesture prediction: {str(e)}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmark features
        
        Args:
            features: Raw hand landmark features
            
        Returns:
            Normalized features
        """
        try:
            # Simple min-max normalization
            features_norm = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
            return features_norm
        except:
            # Return original features if normalization fails
            return features
    
    def get_supported_gestures(self) -> List[str]:
        """Get list of supported gestures"""
        return self.gesture_labels.copy()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded models"""
        return {
            'static_model': {
                'type': 'CNN',
                'input_shape': (self.feature_dim,),
                'num_classes': len(self.gesture_labels),
                'trainable_params': self.static_model.model.count_params() if hasattr(self.static_model.model, 'count_params') else 'Unknown'
            },
            'dynamic_model': {
                'type': 'LSTM',
                'input_shape': (None, self.feature_dim),
                'num_classes': len(self.gesture_labels),
                'trainable_params': self.dynamic_model.model.count_params() if hasattr(self.dynamic_model.model, 'count_params') else 'Unknown'
            },
            'supported_gestures': self.gesture_labels
        }
