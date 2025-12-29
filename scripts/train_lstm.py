import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.lstm_model import LSTMGestureModel

def train_lstm_model(dataset_path: str = 'data/dataset/SL/sequences',
                    model_save_path: str = 'models/trained_models',
                    epochs: int = 20,
                    batch_size: int = 32,
                    sequence_length: int = 40,
                    lstm_units: int = 64,
                    dropout_rate: float = 0.3):
    """
    Train LSTM model for dynamic gesture recognition
    Args:
        dataset_path: Path to processed dataset
        model_save_path: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size for training
        sequence_length: Length of sequences to pad/truncate to
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
    """
    # Create save directory
    os.makedirs(model_save_path, exist_ok=True)
    # Load dataset from .npy files
    print("🔄 Loading dataset from .npy sequences...")
    sequences = []
    labels = []
    class_names = []
    class_to_idx = {}
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path {dataset_path} not found. Run preprocessing first.")
        return
    for gesture in sorted(os.listdir(dataset_path)):
        gesture_path = os.path.join(dataset_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        if gesture not in class_to_idx:
            class_to_idx[gesture] = len(class_names)
            class_names.append(gesture)
        label = class_to_idx[gesture]
        for file in os.listdir(gesture_path):
            if file.endswith('.npy'):
                seq_path = os.path.join(gesture_path, file)
                seq = np.load(seq_path)
                sequences.append(seq)
                labels.append(label)
    if not sequences:
        print("❌ No sequences found. Run preprocessing first.")
        return
    sequences = np.array(sequences)
    labels = np.array(labels)
    num_classes = len(class_names)
    print(f"Loaded {len(sequences)} sequences from {num_classes} gesture classes: {class_names}")
    # Prepare data splits
    print("🔄 Preparing training data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences, labels, test_size=2000, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=2000, random_state=42, stratify=y_train_val
    )
    # One-hot encode labels for categorical crossentropy
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    train_data = {'X': X_train, 'y': y_train_onehot}
    val_data = {'X': X_val, 'y': y_val_onehot}
    test_data = {'X': X_test, 'y': y_test_onehot}
    # Initialize model
    lstm_model = LSTMGestureModel()
    # Build model
    print("🏗️ Building LSTM model...")
    input_shape = (sequence_length, sequences.shape[2])  # (sequence_length, feature_dim)
    lstm_model.build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    # Train model
    print("🚀 Training LSTM model...")
    history = lstm_model.train(
        X_train=train_data['X'].tolist(),  # Convert to list of sequences
        y_train=train_data['y'],
        X_val=val_data['X'].tolist(),
        y_val=val_data['y'],
        epochs=epochs,
        batch_size=batch_size
    )
    if history is None:
        print("❌ Training failed.")
        return
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_metrics = lstm_model.evaluate(
        X_test=test_data['X'].tolist(),
        y_test=test_data['y']
    )

    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lstm_gesture_model_{timestamp}.h5"
    model_path = os.path.join(model_save_path, model_filename)
    lstm_model.save_model(model_path)

    # Save training history and metadata
    training_info = {
        'timestamp': timestamp,
        'model_type': 'LSTM',
        'dataset': dataset_path,
        'num_classes': num_classes,
        'class_names': class_names,
        'input_shape': list(input_shape),
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate
        },
        'test_metrics': test_metrics,
        'model_path': model_path
    }

    # Save training info
    info_path = os.path.join(model_save_path, f"training_info_{timestamp}.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    # Plot training history
    if hasattr(history, 'history'):
        plot_training_history(history.history, model_save_path, timestamp)

    print("✅ LSTM training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training info saved to: {info_path}")

def plot_training_history(history: dict, save_dir: str, timestamp: str):
    """Plot training history"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Accuracy
        ax1.plot(history.get('accuracy', []), label='Train')
        ax1.plot(history.get('val_accuracy', []), label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss
        ax2.plot(history.get('loss', []), label='Train')
        ax2.plot(history.get('val_loss', []), label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # Top-k Accuracy
        if 'top_k_categorical_accuracy' in history:
            ax3.plot(history['top_k_categorical_accuracy'], label='Train')
            ax3.plot(history.get('val_top_k_categorical_accuracy', []), label='Validation')
            ax3.set_title('Top-k Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Top-k Accuracy')
            ax3.legend()

        # Learning rate (if available)
        if 'lr' in history:
            ax4.plot(history['lr'])
            ax4.set_title('Learning Rate')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"training_history_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training plot saved to: {plot_path}")

    except Exception as e:
        print(f"❌ Error plotting training history: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for gesture recognition')
    parser.add_argument('--dataset', type=str, default='data/dataset/SL/sequences',
                       help='Path to sequences directory')
    parser.add_argument('--save_dir', type=str, default='models/trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--seq_length', type=int, default=40,
                       help='Sequence length for LSTM')
    parser.add_argument('--lstm_units', type=int, default=64,
                       help='Number of LSTM units')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    args = parser.parse_args()

    train_lstm_model(
        dataset_path=args.dataset,
        model_save_path=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout
    )
