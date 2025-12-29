import os
import numpy as np
import argparse
from models.cnn_model import CNNGestureModel
from data.dataset_loader import SignLanguageDataset
import matplotlib.pyplot as plt
import json
from datetime import datetime

def train_cnn_model(dataset_path: str = 'data/processed_wlasl/processed_features.pkl',
                   model_save_path: str = 'models/trained_models',
                   epochs: int = 100,
                   batch_size: int = 32,
                   use_augmentation: bool = True,
                   augmentation_factor: int = 3,
                   use_cross_validation: bool = False,
                   cv_folds: int = 5,
                   use_hyperparameter_search: bool = False,
                   hp_max_evals: int = 10):
    """
    Train CNN model for static gesture recognition

    Args:
        dataset_path: Path to processed dataset
        model_save_path: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_augmentation: Whether to use data augmentation
        augmentation_factor: Augmentation factor
        use_cross_validation: Whether to perform cross-validation
        cv_folds: Number of CV folds
        use_hyperparameter_search: Whether to perform hyperparameter search
        hp_max_evals: Max hyperparameter evaluations
    """

    # Create save directory
    os.makedirs(model_save_path, exist_ok=True)

    # Load dataset
    print("🔄 Loading dataset...")
    dataset = SignLanguageDataset(dataset_path)

    if not dataset.data:
        print("❌ Failed to load dataset. Run preprocessing first.")
        return

    # Prepare data
    print("🔄 Preparing training data...")
    train_data, val_data, test_data = dataset.prepare_static_gesture_data()

    if not train_data or not val_data or not test_data:
        print("❌ Failed to prepare data.")
        return

    # Get class information
    class_names = dataset.get_class_names()
    num_classes = len(class_names)
    print(f"Training on {num_classes} gesture classes: {class_names}")

    # Initialize model
    cnn_model = CNNGestureModel()

    # Hyperparameter search
    if use_hyperparameter_search:
        print("🔍 Performing hyperparameter search...")
        param_grid = {
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 150]
        }

        hp_results = cnn_model.hyperparameter_search(
            train_data['X'], train_data['y'],
            val_data['X'], val_data['y'],
            param_grid, hp_max_evals
        )

        if hp_results:
            # Use best parameters
            best_params = hp_results['best_params']
            epochs = best_params.get('epochs', epochs)
            batch_size = best_params.get('batch_size', batch_size)
            dropout_rate = best_params.get('dropout_rate', 0.3)

            # Save hyperparameter results
            with open(os.path.join(model_save_path, 'hyperparameter_results.json'), 'w') as f:
                json.dump(hp_results, f, indent=2)
        else:
            dropout_rate = 0.3

    # Build model
    print("🏗️ Building CNN model...")
    cnn_model.build_model(
        input_shape=(train_data['X'].shape[1],),
        num_classes=num_classes,
        dropout_rate=dropout_rate if 'dropout_rate' in locals() else 0.3
    )

    # Cross-validation
    if use_cross_validation:
        print("🔄 Performing cross-validation...")
        X_combined = np.vstack([train_data['X'], val_data['X']])
        y_combined = np.vstack([train_data['y'], val_data['y']])

        cv_results = cnn_model.cross_validate(X_combined, y_combined, cv_folds, epochs, batch_size)

        # Save CV results
        with open(os.path.join(model_save_path, 'cross_validation_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=2)

        print(f"CV Results - Avg Accuracy: {cv_results['avg_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")

    # Train final model
    print("🚀 Training final model...")
    if use_augmentation:
        print(f"Using data augmentation (factor: {augmentation_factor})")
        history = cnn_model.train_with_augmentation(
            train_data['X'], train_data['y'],
            val_data['X'], val_data['y'],
            epochs=epochs,
            batch_size=batch_size,
            augmentation_factor=augmentation_factor
        )
    else:
        history = cnn_model.train(
            train_data['X'], train_data['y'],
            val_data['X'], val_data['y'],
            epochs=epochs,
            batch_size=batch_size
        )

    if history is None:
        print("❌ Training failed.")
        return

    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_metrics = cnn_model.evaluate(test_data['X'], test_data['y'])

    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"cnn_gesture_model_{timestamp}.h5"
    model_path = os.path.join(model_save_path, model_filename)
    cnn_model.save_model(model_path)

    # Save training history and metadata
    training_info = {
        'timestamp': timestamp,
        'model_type': 'CNN',
        'dataset': os.path.basename(dataset_path),
        'num_classes': num_classes,
        'class_names': class_names,
        'input_shape': list(train_data['X'].shape[1:]),
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'use_augmentation': use_augmentation,
            'augmentation_factor': augmentation_factor if use_augmentation else 0,
            'use_cross_validation': use_cross_validation,
            'cv_folds': cv_folds if use_cross_validation else 0,
            'use_hyperparameter_search': use_hyperparameter_search,
            'hp_max_evals': hp_max_evals if use_hyperparameter_search else 0
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

    print("✅ CNN training completed!")
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
    parser = argparse.ArgumentParser(description='Train CNN model for gesture recognition')
    parser.add_argument('--dataset', type=str, default='data/processed_wlasl/processed_features.pkl',
                       help='Path to processed dataset')
    parser.add_argument('--save_dir', type=str, default='models/trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--aug_factor', type=int, default=3,
                       help='Data augmentation factor')
    parser.add_argument('--cross_val', action='store_true',
                       help='Perform cross-validation')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--hp_search', action='store_true',
                       help='Perform hyperparameter search')
    parser.add_argument('--hp_evals', type=int, default=10,
                       help='Max hyperparameter evaluations')

    args = parser.parse_args()

    train_cnn_model(
        dataset_path=args.dataset,
        model_save_path=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=args.augmentation,
        augmentation_factor=args.aug_factor,
        use_cross_validation=args.cross_val,
        cv_folds=args.cv_folds,
        use_hyperparameter_search=args.hp_search,
        hp_max_evals=args.hp_evals
    )
