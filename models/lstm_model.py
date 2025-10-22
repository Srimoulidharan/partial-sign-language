import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional

class LSTMGestureModel:
    """LSTM model for dynamic gesture recognition"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.sequence_length = 30  # Maximum sequence length
    
    def build_model(self, input_shape: Tuple, num_classes: int, 
                   lstm_units: int = 64, dropout_rate: float = 0.3):
        """
        Build LSTM model architecture for dynamic gesture recognition
        
        Args:
            input_shape: Shape of input sequences (sequence_length, feature_dim)
            num_classes: Number of gesture classes
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        try:
            self.model = keras.Sequential([
                # First LSTM layer with return sequences
                layers.LSTM(
                    lstm_units * 2, 
                    return_sequences=True, 
                    input_shape=input_shape,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name='lstm_1'
                ),
                layers.BatchNormalization(),
                
                # Second LSTM layer
                layers.LSTM(
                    lstm_units,
                    return_sequences=False,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name='lstm_2'
                ),
                layers.BatchNormalization(),
                
                # Dense layers for classification
                layers.Dense(64, activation='relu', name='dense_1'),
                layers.Dropout(dropout_rate),
                
                layers.Dense(32, activation='relu', name='dense_2'),
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
            
            print(f"✅ LSTM model built successfully with {input_shape} input shape and {num_classes} classes")
            
        except Exception as e:
            print(f"❌ Error building LSTM model: {str(e)}")
            # Fallback to simpler model
            self._build_simple_model(input_shape, num_classes)
    
    def _build_simple_model(self, input_shape: Tuple, num_classes: int):
        """Build a simpler fallback LSTM model"""
        self.model = keras.Sequential([
            layers.LSTM(32, input_shape=input_shape, dropout=0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("⚠️ Using simplified LSTM model")
    
    def preprocess_sequences(self, sequences: list, max_length: Optional[int] = None) -> np.ndarray:
        """
        Preprocess variable-length sequences for LSTM input
        
        Args:
            sequences: List of sequences with varying lengths
            max_length: Maximum sequence length (pad/truncate to this length)
            
        Returns:
            Preprocessed sequences array
        """
        if max_length is None:
            max_length = self.sequence_length
        
        processed_sequences = []
        
        for sequence in sequences:
            sequence = np.array(sequence)
            
            # Pad or truncate sequence
            if len(sequence) < max_length:
                # Pad with zeros
                padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
                padded_sequence = np.vstack([sequence, padding])
            else:
                # Truncate to max_length
                padded_sequence = sequence[-max_length:]
            
            processed_sequences.append(padded_sequence)
        
        return np.array(processed_sequences)
    
    def train(self, X_train: list, y_train: np.ndarray,
              X_val: Optional[list] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (list of variable-length sequences)
            y_train: Training labels (one-hot encoded)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            # Preprocess sequences
            X_train_processed = self.preprocess_sequences(X_train)
            X_val_processed = self.preprocess_sequences(X_val) if X_val is not None else None
            
            # Callbacks for training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=15,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6
                )
            ]
            
            # Validation data
            validation_data = (X_val_processed, y_val) if X_val_processed is not None and y_val is not None else None
            
            # Train model
            history = self.model.fit(
                X_train_processed, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            print("✅ LSTM model training completed")
            
            return history
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            return None
    
    def predict(self, sequences: list, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            sequences: Input sequences (list of variable-length sequences)
            batch_size: Batch size for prediction
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Preprocess sequences
        X_processed = self.preprocess_sequences(sequences)
        
        return self.model.predict(X_processed, batch_size=batch_size, verbose=0)
    
    def predict_single_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict a single sequence
        
        Args:
            sequence: Single sequence array
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Preprocess single sequence
        processed = self.preprocess_sequences([sequence])
        return self.model.predict(processed, verbose=0)[0]
    
    def evaluate(self, X_test: list, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            # Preprocess test sequences
            X_test_processed = self.preprocess_sequences(X_test)
            
            results = self.model.evaluate(X_test_processed, y_test, verbose=0)
            
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
            print(f"✅ LSTM model saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            print(f"✅ LSTM model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is not None:
            self.model.summary()
        else:
            print("❌ Model not built yet")
    
    def analyze_sequence_attention(self, sequence: np.ndarray) -> np.ndarray:
        """
        Analyze attention weights for a sequence (simplified version)
        
        Args:
            sequence: Input sequence
            
        Returns:
            Attention weights (simplified)
        """
        if self.model is None:
            return np.array([])
        
        try:
            # This is a simplified attention analysis
            # In a real implementation, you'd need attention layers
            processed = self.preprocess_sequences([sequence])
            
            # Get activations from LSTM layers
            lstm_model = keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('lstm_1').output
            )
            
            activations = lstm_model.predict(processed, verbose=0)[0]
            
            # Calculate simple attention weights based on activation magnitude
            attention_weights = np.mean(np.abs(activations), axis=1)
            attention_weights = attention_weights / np.sum(attention_weights)
            
            return attention_weights
            
        except Exception as e:
            print(f"❌ Error analyzing attention: {str(e)}")
            return np.array([])
