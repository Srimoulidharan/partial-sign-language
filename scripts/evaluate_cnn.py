import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import json
from models.cnn_model import CNNGestureModel
from data.dataset_loader import SignLanguageDataset
import argparse
def evaluate_cnn_model(model_path: str,
                      dataset_path: str = 'data/processed_wlasl/processed_features.pkl',
                      save_dir: str = 'models/evaluation_results'):
    """
    Evaluate trained CNN model performance
    Args:
        model_path: Path to trained CNN model
        dataset_path: Path to processed dataset
        save_dir: Directory to save evaluation results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    # Load dataset
    print("🔄 Loading dataset...")
    dataset = SignLanguageDataset(dataset_path)
    if not dataset.data:
        print("❌ Failed to load dataset.")
        return
    # Prepare test data
    print("🔄 Preparing test data...")
    _, _, test_data = dataset.prepare_static_gesture_data()
    
    if not test_data:
        print("❌ Failed to prepare test data.")
        return
    
    # Load model
    print(f"🔄 Loading model from {model_path}...")
    cnn_model = CNNGestureModel()
    cnn_model.load_model(model_path)
    
    # Get predictions
    print("🔄 Generating predictions...")
    y_pred_prob = cnn_model.predict(test_data['X'])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(test_data['y'], axis=1)
    
    # Get class names
    class_names = dataset.get_class_names()
    class_mapping = dataset.get_class_mapping()
    
    # Convert indices to class names
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
            per_class_accuracy[class_name] = class_accuracy
    
    # Top-k accuracy
    k_values = [1, 3, 5]
    top_k_accuracies = {}
    for k in k_values:
        if k <= len(class_names):
            # Get top-k predictions
            top_k_indices = np.argsort(y_pred_prob, axis=1)[:, -k:]
            top_k_correct = np.any(top_k_indices == y_true.reshape(-1, 1), axis=1)
            top_k_accuracy = np.mean(top_k_correct)
            top_k_accuracies[f'top_{k}'] = top_k_accuracy
    
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
        'dataset': os.path.basename(dataset_path),
        'num_classes': len(class_names),
        'test_samples': len(test_data['X']),
        'overall_accuracy': float(accuracy),
        'per_class_accuracy': per_class_accuracy,
        'top_k_accuracies': top_k_accuracies,
        'confidence_stats': confidence_stats,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    # Print results
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Test Samples: {len(test_data['X'])}")
    print(f"Number of Classes: {len(class_names)}")
    
    print("Top-k Accuracies:")
    for k, acc in top_k_accuracies.items():
        print(f"  {k}: {acc:.4f}")
    
    print("Confidence Statistics:")
    for stat, value in confidence_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print("Per-Class Accuracy:")
    for class_name, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {acc:.4f}")
    
    # Save results
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    
    # Generate plots
    generate_evaluation_plots(evaluation_results, save_dir)
    
    return evaluation_results

def generate_evaluation_plots(results: dict, save_dir: str):
    """Generate evaluation plots"""
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = np.array(results['confusion_matrix'])
        class_names = results['class_names']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True if len(class_names) <= 20 else False, 
                   fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class accuracy bar plot
        plt.figure(figsize=(12, 6))
        per_class_acc = results['per_class_accuracy']
        classes = list(per_class_acc.keys())
        accuracies = list(per_class_acc.values())
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        classes_sorted = [classes[i] for i in sorted_indices]
        accuracies_sorted = [accuracies[i] for i in sorted_indices]
        
        bars = plt.bar(range(len(classes_sorted)), accuracies_sorted)
        plt.xlabel('Gesture Classes')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies_sorted):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification report heatmap
        if 'classification_report' in results:
            report = results['classification_report']
            
            # Extract precision, recall, f1-score for each class
            classes = []
            precision = []
            recall = []
            f1_score = []
            
            for class_name in results['class_names']:
                if class_name in report:
                    classes.append(class_name)
                    precision.append(report[class_name]['precision'])
                    recall.append(report[class_name]['recall'])
                    f1_score.append(report[class_name]['f1-score'])
            
            if classes:
                plt.figure(figsize=(12, 8))
                metrics_data = np.array([precision, recall, f1_score]).T
                
                sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='RdYlGn',
                           xticklabels=['Precision', 'Recall', 'F1-Score'],
                           yticklabels=classes)
                plt.title('Classification Metrics by Class')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'classification_metrics.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✅ Evaluation plots saved to {save_dir}")
        
    except Exception as e:
        print(f"❌ Error generating plots: {str(e)}")

def compare_models(model_paths: list, dataset_path: str, save_dir: str):
    """Compare multiple trained models"""
    try:
        comparison_results = {}
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path).replace('.h5', '')
            print(f"\n🔄 Evaluating {model_name}...")
            
            results = evaluate_cnn_model(model_path, dataset_path, save_dir)
            if results:
                comparison_results[model_name] = {
                    'accuracy': results['overall_accuracy'],
                    'top_1': results['top_k_accuracies'].get('top_1', 0),
                    'top_3': results['top_k_accuracies'].get('top_3', 0),
                    'top_5': results['top_k_accuracies'].get('top_5', 0),
                    'mean_confidence': results['confidence_stats']['mean']
                }
        
        # Save comparison
        comparison_path = os.path.join(save_dir, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Print comparison table
        print("\n📊 MODEL COMPARISON")
        print("=" * 60)
        print("<15")
        print("-" * 60)
        
        for model_name, metrics in comparison_results.items():
            print("<15")
        
        print(f"\n✅ Comparison saved to {comparison_path}")
        
    except Exception as e:
        print(f"❌ Error in model comparison: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CNN model performance')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained CNN model')
    parser.add_argument('--dataset', type=str, default='data/processed_wlasl/processed_features.pkl',
                       help='Path to processed dataset')
    parser.add_argument('--save_dir', type=str, default='models/evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Paths to additional models for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        all_models = [args.model] + args.compare
        compare_models(all_models, args.dataset, args.save_dir)
    else:
        # Evaluate single model
        evaluate_cnn_model(args.model, args.dataset, args.save_dir)
