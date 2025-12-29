import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMGestureModel
import argparse

def evaluate_lstm_model(model_path: str, training_info_path: str, save_dir: str = 'models/evaluation_results'):
    """
    Evaluate trained LSTM model performance
    Args:
        model_path: Path to trained LSTM model
        training_info_path: Path to training info JSON
        save_dir: Directory to save evaluation results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load training info
    print("🔄 Loading training info...")
    with open(training_info_path, 'r') as f:
        training_info = json.load(f)
    
    class_names = training_info['class_names']
    num_classes = training_info['num_classes']
    dataset_path = training_info['dataset']
    sequence_length = training_info['training_params']['sequence_length']
    
    # Load dataset (similar to training)
    print("🔄 Loading dataset from .npy sequences...")
    sequences = []
    labels = []
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path {dataset_path} not found.")
        return
    
    for gesture in sorted(os.listdir(dataset_path)):
        gesture_path = os.path.join(dataset_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        if gesture not in class_to_idx:
            continue  # Skip if not in class_names
        label = class_to_idx[gesture]
        for file in os.listdir(gesture_path):
            if file.endswith('.npy'):
                seq_path = os.path.join(gesture_path, file)
                seq = np.load(seq_path)
                sequences.append(seq)
                labels.append(label)
    
    if not sequences:
        print("❌ No sequences found.")
        return
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Loaded {len(sequences)} sequences from {len(set(labels))} classes.")
    
    # Prepare test data (use 20% for evaluation)
    print("🔄 Preparing test data...")
    _, X_test, _, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # One-hot encode labels
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    test_data = {'X': X_test, 'y': y_test_onehot}
    
    # Load model
    print(f"🔄 Loading model from {model_path}...")
    lstm_model = LSTMGestureModel()
    lstm_model.load_model(model_path)
    
    # Get predictions
    print("🔄 Generating predictions...")
    y_pred_prob = lstm_model.predict(test_data['X'].tolist())
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(test_data['y'], axis=1)
    
    # Convert to class names
    y_true_names = [class_names[i] for i in y_true]
    y_pred_names = [class_names[i] for i in y_pred]
    
    # Calculate metrics
    print("📊 Calculating evaluation metrics...")
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Classification report
    report = classification_report(y_true_names, y_pred_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_names, y_pred_names, labels=class_names)
    
    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred[class_mask] == i)
            per_class_accuracy[class_name] = float(class_accuracy)
    
    # Top-k accuracy
    k_values = [1, 3, 5]
    top_k_accuracies = {}
    for k in k_values:
        if k <= num_classes:
            top_k_indices = np.argsort(y_pred_prob, axis=1)[:, -k:]
            top_k_correct = np.any(np.isin(top_k_indices, y_true.reshape(-1, 1)), axis=1)
            top_k_accuracy = np.mean(top_k_correct)
            top_k_accuracies[f'top_{k}'] = float(top_k_accuracy)
    
    # Confidence analysis
    max_confidences = np.max(y_pred_prob, axis=1)
    confidence_stats = {
        'mean': float(np.mean(max_confidences)),
        'std': float(np.std(max_confidences)),
        'min': float(np.min(max_confidences)),
        'max': float(np.max(max_confidences)),
        'median': float(np.median(max_confidences))
    }
    
    # Compile results
    evaluation_results = {
        'model_path': model_path,
        'dataset': dataset_path,
        'num_classes': num_classes,
        'test_samples': len(test_data['X']),
        'overall_accuracy': float(accuracy),
        'per_class_accuracy': per_class_accuracy,
        'top_k_accuracies': top_k_accuracies,
        'confidence_stats': confidence_stats,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Print results
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Test Samples: {len(test_data['X'])}")
    print(f"Number of Classes: {num_classes}")
    
    print("\nTop-k Accuracies:")
    for k, acc in top_k_accuracies.items():
        print(f"  {k}: {acc:.4f}")
    
    print("\nConfidence Statistics:")
    for stat, value in confidence_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print("\nPer-Class Accuracy (Top 10):")
    sorted_per_class = sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)[:10]
    for class_name, acc in sorted_per_class:
        print(f"  {class_name}: {acc:.4f}")
    
    # Save results
    timestamp = evaluation_results['timestamp']
    results_path = os.path.join(save_dir, f'lstm_evaluation_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    
    # Generate plots
    generate_evaluation_plots(evaluation_results, save_dir)
    
    return evaluation_results

def generate_evaluation_plots(results: dict, save_dir: str):
    """Generate evaluation plots"""
    try:
        timestamp = results['timestamp']
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Per-class accuracy bar plot (top 20)
        plt.figure(figsize=(15, 8))
        per_class_acc = results['per_class_accuracy']
        classes = list(per_class_acc.keys())
        accuracies = list(per_class_acc.values())
        
        sorted_indices = np.argsort(accuracies)[::-1]
        top_n = min(20, len(classes))
        top_classes = [classes[i] for i in sorted_indices[:top_n]]
        top_accuracies = [accuracies[i] for i in sorted_indices[:top_n]]
        
        bars = plt.bar(range(top_n), top_accuracies)
        plt.xlabel('Gesture Classes (Top 20)')
        plt.ylabel('Accuracy')
        plt.title('Top 20 Per-Class Accuracy')
        plt.xticks(range(top_n), top_classes, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for bar, acc in zip(bars, top_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'lstm_per_class_accuracy_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Skip full confusion matrix for 2000 classes, perhaps sample
        print("Skipping full confusion matrix due to large number of classes (2000). Consider subset evaluation.")
        
        print(f"✅ Evaluation plots saved to {save_dir}")
        
    except Exception as e:
        print(f"❌ Error generating plots: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LSTM model performance')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained LSTM model')
    parser.add_argument('--training_info', type=str, required=True,
                       help='Path to training info JSON')
    parser.add_argument('--save_dir', type=str, default='models/evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_lstm_model(args.model, args.training_info, args.save_dir)
