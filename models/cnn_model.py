import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional

class CNNGestureModel:
    """CNN model for static gesture recognition"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def build_model(self, input_shape: Tuple[int], num_classes: int, dropout_rate: float = 0.3):
        """
        Build CNN model architecture for static gesture recognition
        
        Args:
            input_shape: Shape of input features (feature_dim,)
            num_classes: Number of gesture classes
            dropout_rate: Dropout rate for regularization
        """
        try:
            # Since we're working with hand landmark features (not images),
            # we'll use a fully connected architecture optimized for feature vectors
            
            self.model = keras.Sequential([
                # Input layer
                layers.Dense(256, activation='relu', input_shape=input_shape, name='dense_1'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                
                # Hidden layers
                layers.Dense(128, activation='relu', name='dense_2'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                
                layers.Dense(64, activation='relu', name='dense_3'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                
                layers.Dense(32, activation='relu', name='dense_4'),
                layers.Dropout(dropout_rate),
                
                # Output layer
                layers.Dense(num_classes, activation='softmax', name='output')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
            )
            
            print(f"✅ CNN model built successfully with {input_shape} input shape and {num_classes} classes")
            
        except Exception as e:
            print(f"❌ Error building CNN model: {str(e)}")
            # Fallback to simpler model
            self._build_simple_model(input_shape, num_classes)
    
    def _build_simple_model(self, input_shape: Tuple[int], num_classes: int):
        """Build a simpler fallback model"""
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("⚠️ Using simplified CNN model")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32):
        """
        Train the CNN model
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            # Callbacks for training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Validation data
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            print("✅ CNN model training completed")
            
            return history
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            return None
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            results = self.model.evaluate(X_test, y_test, verbose=0)
            
            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = results[i]
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not built.")
        
        try:
            self.model.save(filepath)
            print(f"✅ Model saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is not None:
            self.model.summary()
        else:
            print("❌ Model not built yet")
    
    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        Get feature importance using gradient-based method
        
        Args:
            X_sample: Sample input for importance calculation
            
        Returns:
            Feature importance scores
        """
        if self.model is None:
            return np.array([])
        
        try:
            # Simple gradient-based importance
            with tf.GradientTape() as tape:
                X_tensor = tf.Variable(X_sample.astype(np.float32))
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)
                max_pred = tf.reduce_max(predictions)
            
            gradients = tape.gradient(max_pred, X_tensor)
            importance = tf.abs(gradients).numpy()
            
            return importance
            
        except Exception as e:
            print(f"❌ Error calculating feature importance: {str(e)}")
            return np.array([])
