import pickle
import numpy as np
import json
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SignLanguageDataset:
    """Dataset loader for sign language gesture recognition"""
    
    def __init__(self, processed_data_path: str = 'data/processed_wlasl/processed_features.pkl'):
        """
        Initialize dataset loader
        
        Args:
            processed_data_path: Path to processed features pickle file
        """
        self.processed_data_path = processed_data_path
        self.data = None
        self.metadata = None
        self.label_encoder = LabelEncoder()
        
        self.load_data()
    
    def load_data(self):
        """Load processed dataset"""
        try:
            with open(self.processed_data_path, 'rb') as f:
                self.data = pickle.load(f)
            
            # Load metadata
            metadata_path = self.processed_data_path.replace('processed_features.pkl', 'metadata.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            print(f"✅ Dataset loaded: {self.metadata['num_classes']} classes, {self.metadata['total_videos']} videos")
            
        except FileNotFoundError:
            print(f"❌ Processed data not found: {self.processed_data_path}")
            print("Run scripts/preprocess_dataset.py first")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
        
        return True
    
    def prepare_static_gesture_data(self, 
                                   test_size: float = 0.2,
                                   val_size: float = 0.1,
                                   random_state: int = 42) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare data for static gesture CNN training
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        if not self.data:
            return {}, {}, {}
        
        features = []
        labels = []
        
        # Extract single frames from sequences (use middle frame for static gestures)
        for gesture_name, video_sequences in self.data['features'].items():
            gesture_idx = self.data['gesture_to_idx'][gesture_name]
            
            for sequence in video_sequences:
                if sequence:  # Non-empty sequence
                    # Use middle frame for static gesture
                    middle_idx = len(sequence) // 2
                    features.append(sequence[middle_idx])
                    labels.append(gesture_idx)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        # Convert to one-hot encoding
        num_classes = len(np.unique(labels))
        y_train_onehot = np.eye(num_classes)[y_train]
        y_val_onehot = np.eye(num_classes)[y_val]
        y_test_onehot = np.eye(num_classes)[y_test]
        
        train_data = {'X': X_train, 'y': y_train_onehot}
        val_data = {'X': X_val, 'y': y_val_onehot}
        test_data = {'X': X_test, 'y': y_test_onehot}
        
        print(f"Static gesture data prepared:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Val: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        print(f"Feature shape: {X_train.shape[1:]}")
        
        return train_data, val_data, test_data
    
    def prepare_dynamic_gesture_data(self,
                                    sequence_length: int = 30,
                                    test_size: float = 0.2,
                                    val_size: float = 0.1,
                                    random_state: int = 42) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare data for dynamic gesture LSTM training
        
        Args:
            sequence_length: Length of sequences to pad/truncate to
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        if not self.data:
            return {}, {}, {}
        
        sequences = []
        labels = []
        
        # Use full sequences for dynamic gestures
        for gesture_name, video_sequences in self.data['features'].items():
            gesture_idx = self.data['gesture_to_idx'][gesture_name]
            
            for sequence in video_sequences:
                if len(sequence) >= 5:  # Minimum sequence length
                    # Pad or truncate sequence
                    if len(sequence) < sequence_length:
                        # Pad with zeros
                        padding = [np.zeros_like(sequence[0]) for _ in range(sequence_length - len(sequence))]
                        padded_sequence = sequence + padding
                    else:
                        # Truncate to sequence_length
                        padded_sequence = sequence[:sequence_length]
                    
                    sequences.append(np.array(padded_sequence))
                    labels.append(gesture_idx)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # Split data
        seq_temp, seq_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        seq_train, seq_val, y_train, y_val = train_test_split(
            seq_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        # Convert to one-hot encoding
        num_classes = len(np.unique(labels))
        y_train_onehot = np.eye(num_classes)[y_train]
        y_val_onehot = np.eye(num_classes)[y_val]
        y_test_onehot = np.eye(num_classes)[y_test]
        
        train_data = {'X': seq_train, 'y': y_train_onehot}
        val_data = {'X': seq_val, 'y': y_val_onehot}
        test_data = {'X': seq_test, 'y': y_test_onehot}
        
        print(f"Dynamic gesture data prepared:")
        print(f"Train: {seq_train.shape[0]} sequences")
        print(f"Val: {seq_val.shape[0]} sequences")
        print(f"Test: {seq_test.shape[0]} sequences")
        print(f"Sequence shape: {seq_train.shape[1:]}")
        
        return train_data, val_data, test_data
    
    def get_class_names(self) -> List[str]:
        """Get list of gesture class names"""
        if not self.data:
            return []
        return list(self.data['gesture_to_idx'].keys())
    
    def get_class_mapping(self) -> Dict[str, int]:
        """Get gesture name to index mapping"""
        if not self.data:
            return {}
        return self.data['gesture_to_idx'].copy()
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        if not self.metadata:
            return {}
        return self.metadata.copy()
    
    def create_data_generator(self, data_dict: Dict, batch_size: int = 32, shuffle: bool = True):
        """
        Create a data generator for training
        
        Args:
            data_dict: Dictionary with 'X' and 'y' keys
            batch_size: Batch size for generator
            shuffle: Whether to shuffle data
            
        Returns:
            Data generator function
        """
        X, y = data_dict['X'], data_dict['y']
        
        def generator():
            indices = np.arange(len(X))
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_indices = indices[start_idx:end_idx]
                
                yield X[batch_indices], y[batch_indices]
        
        return generator()

if __name__ == "__main__":
    # Test dataset loader
    dataset = SignLanguageDataset()
    
    if dataset.data:
        # Test static data preparation
        train_static, val_static, test_static = dataset.prepare_static_gesture_data()
        
        # Test dynamic data preparation
        train_dynamic, val_dynamic, test_dynamic = dataset.prepare_dynamic_gesture_data()
        
        print(f"Class names: {dataset.get_class_names()}")
        print(f"Dataset stats: {dataset.get_dataset_stats()}")
