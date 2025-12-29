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
            
            print(f"CNN model built successfully with {input_shape} input shape and {num_classes} classes")
            
        except Exception as e:
            print(f" Error building CNN model: {str(e)}")
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

    def augment_features(self, features: np.ndarray,
                        translation_range: float = 0.05,
                        scale_range: float = 0.1,
                        rotation_range: float = 10.0,
                        noise_std: float = 0.01) -> np.ndarray:
        """
        Apply data augmentation to hand landmark features

        Args:
            features: Input features (batch_size, feature_dim) or (feature_dim,)
            translation_range: Max translation as fraction of coordinate range
            scale_range: Max scale factor variation
            rotation_range: Max rotation angle in degrees
            noise_std: Standard deviation for Gaussian noise

        Returns:
            Augmented features
        """
        try:
            # Handle single sample
            if features.ndim == 1:
                features = features.reshape(1, -1)

            augmented = features.copy()

            batch_size, feature_dim = augmented.shape

            # Reshape to (batch_size, 21, 2) for x,y coordinates
            coords = augmented.reshape(batch_size, 21, 2)

            for i in range(batch_size):
                # Random translation
                if translation_range > 0:
                    tx = np.random.uniform(-translation_range, translation_range)
                    ty = np.random.uniform(-translation_range, translation_range)
                    coords[i, :, 0] += tx
                    coords[i, :, 1] += ty

                # Random scaling
                if scale_range > 0:
                    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
                    coords[i] *= scale

                # Random rotation (around wrist landmark index 0)
                if rotation_range > 0:
                    angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
                    cos_a, sin_a = np.cos(angle), np.sin(angle)

                    # Translate to origin (wrist)
                    wrist = coords[i, 0].copy()
                    coords[i] -= wrist

                    # Rotate
                    x_rot = coords[i, :, 0] * cos_a - coords[i, :, 1] * sin_a
                    y_rot = coords[i, :, 0] * sin_a + coords[i, :, 1] * cos_a
                    coords[i, :, 0] = x_rot
                    coords[i, :, 1] = y_rot

                    # Translate back
                    coords[i] += wrist

                # Add Gaussian noise
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, coords[i].shape)
                    coords[i] += noise

            # Clip to valid range [0, 1] (assuming normalized coordinates)
            coords = np.clip(coords, 0, 1)

            # Reshape back
            augmented = coords.reshape(batch_size, feature_dim)

            return augmented if features.shape[0] > 1 else augmented[0]

        except Exception as e:
            print(f"❌ Error in data augmentation: {str(e)}")
            return features

    def train_with_augmentation(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                               epochs: int = 50, batch_size: int = 32,
                               augmentation_factor: int = 2,
                               use_early_stopping: bool = True,
                               use_lr_scheduler: bool = True) -> dict:
        """
        Train model with data augmentation

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            augmentation_factor: How many augmented versions per sample
            use_early_stopping: Whether to use early stopping
            use_lr_scheduler: Whether to use learning rate scheduler

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        try:
            # Create augmented dataset
            print(f"🔄 Generating augmented data (factor: {augmentation_factor})...")
            augmented_X = []
            augmented_y = []

            for i in range(len(X_train)):
                # Original sample
                augmented_X.append(X_train[i])
                augmented_y.append(y_train[i])

                # Augmented samples
                for _ in range(augmentation_factor):
                    aug_sample = self.augment_features(X_train[i])
                    augmented_X.append(aug_sample)
                    augmented_y.append(y_train[i])

            X_train_aug = np.array(augmented_X)
            y_train_aug = np.array(augmented_y)

            print(f"✅ Augmented dataset: {len(X_train_aug)} samples (original: {len(X_train)})")

            # Call regular training with augmented data
            return self.train(X_train_aug, y_train_aug, X_val, y_val, epochs, batch_size)

        except Exception as e:
            print(f"❌ Error in training with augmentation: {str(e)}")
            return {}

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      folds: int = 5, epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Perform k-fold cross-validation

        Args:
            X: Feature data
            y: Label data
            folds: Number of CV folds
            epochs: Epochs per fold
            batch_size: Batch size

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import classification_report

        try:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

            cv_results = {
                'fold_accuracies': [],
                'fold_losses': [],
                'avg_accuracy': 0.0,
                'std_accuracy': 0.0,
                'classification_reports': []
            }

            fold_idx = 1
            for train_idx, val_idx in skf.split(X, np.argmax(y, axis=1)):
                print(f"🔄 Training fold {fold_idx}/{folds}...")

                # Split data
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Build fresh model for each fold
                self.build_model(
                    input_shape=(X.shape[1],),
                    num_classes=y.shape[1]
                )

                # Train
                history = self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                   epochs, batch_size, verbose=0)

                # Evaluate
                if history:
                    val_accuracy = max(history.get('val_accuracy', [0]))
                    val_loss = min(history.get('val_loss', [float('inf')]))

                    cv_results['fold_accuracies'].append(val_accuracy)
                    cv_results['fold_losses'].append(val_loss)

                    # Detailed classification report
                    y_pred = self.predict(X_val_fold, batch_size=batch_size)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_true_classes = np.argmax(y_val_fold, axis=1)

                    report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
                    cv_results['classification_reports'].append(report)

                fold_idx += 1

            # Calculate averages
            if cv_results['fold_accuracies']:
                cv_results['avg_accuracy'] = np.mean(cv_results['fold_accuracies'])
                cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])

            print(f"✅ Cross-validation complete. Avg accuracy: {cv_results['avg_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            return cv_results

        except Exception as e:
            print(f"❌ Error in cross-validation: {str(e)}")
            return {}

    def hyperparameter_search(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             param_grid: dict, max_evals: int = 10) -> dict:
        """
        Perform hyperparameter search using random search

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            param_grid: Dictionary of parameter ranges
            max_evals: Maximum number of evaluations

        Returns:
            Best parameters and results
        """
        import random

        try:
            best_score = 0.0
            best_params = {}
            best_history = {}

            print(f"🔍 Starting hyperparameter search ({max_evals} evaluations)...")

            for eval_idx in range(max_evals):
                # Sample random parameters
                params = {}
                for param_name, param_range in param_grid.items():
                    if isinstance(param_range, list):
                        params[param_name] = random.choice(param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 2:
                        params[param_name] = random.uniform(param_range[0], param_range[1])

                print(f"Evaluation {eval_idx + 1}/{max_evals}: {params}")

                # Build model with sampled parameters
                self.build_model(
                    input_shape=(X_train.shape[1],),
                    num_classes=y_train.shape[1],
                    dropout_rate=params.get('dropout_rate', 0.3)
                )

                # Train
                history = self.train(
                    X_train, y_train, X_val, y_val,
                    epochs=params.get('epochs', 50),
                    batch_size=params.get('batch_size', 32),
                    verbose=0
                )

                # Evaluate
                if history and 'val_accuracy' in history:
                    val_accuracy = max(history['val_accuracy'])
                    if val_accuracy > best_score:
                        best_score = val_accuracy
                        best_params = params.copy()
                        best_history = history.copy()

            print(f"✅ Hyperparameter search complete. Best accuracy: {best_score:.4f}")
            print(f"Best parameters: {best_params}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_history': best_history
            }

        except Exception as e:
            print(f"❌ Error in hyperparameter search: {str(e)}")
            return {}
