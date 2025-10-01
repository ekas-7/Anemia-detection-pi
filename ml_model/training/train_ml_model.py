"""
Training Script for ML Model
Trains Random Forest and Logistic Regression models with comprehensive metrics
"""

import numpy as np
from pathlib import Path
import time
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


class AnemiaModelTrainer:
    """
    Train and evaluate lightweight ML models for anemia detection
    """
    
    def __init__(self, data_dir='./data'):
        """
        Initialize trainer
        
        Args:
            data_dir: Data directory path
        """
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / 'features'
        self.models_dir = Path('./models')
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """
        Load extracted features for all splits
        
        Returns:
            Dictionary with train/val/test data
        """
        print("Loading extracted features...")
        
        data = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.features_dir / split
            
            if not split_dir.exists():
                print(f"Warning: {split} features not found")
                continue
            
            features = np.load(split_dir / 'features.npy')
            labels = np.load(split_dir / 'labels.npy')
            
            data[split] = {'features': features, 'labels': labels}
            
            print(f"  {split.capitalize()}: {features.shape[0]} samples, {features.shape[1]} features")
        
        return data
    
    def create_models(self):
        """
        Create ML model instances
        
        Returns:
            Dictionary of models
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                verbose=1
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear',
                verbose=1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                verbose=True
            )
        }
        
        return models
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # AUC-ROC
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['per_class_metrics'] = report
        
        return metrics
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val):
        """
        Train a single model and evaluate
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Trained model and results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print('='*60)
        
        # Training
        print("\nTraining...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        # Prediction on validation set
        print("\nEvaluating on validation set...")
        start_time = time.time()
        y_pred = model.predict(X_val)
        inference_time = (time.time() - start_time) / len(X_val) * 1000  # ms per sample
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_pred, y_prob)
        
        # Add timing information
        metrics['train_time_seconds'] = train_time
        metrics['inference_time_ms'] = inference_time
        
        # Print results
        print("\n--- Validation Results ---")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"Sensitivity:   {metrics['sensitivity']:.4f}")
        print(f"Specificity:   {metrics['specificity']:.4f}")
        print(f"F1-Score:      {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
        print(f"\nInference Time: {metrics['inference_time_ms']:.2f} ms per image")
        
        results = {
            'model_name': model_name,
            'metrics': metrics
        }
        
        return model, results
    
    def test_model(self, model_name, model, X_test, y_test):
        """
        Test model on test set with comprehensive metrics
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Test results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Testing {model_name.upper()} on Test Set")
        print('='*60)
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            confidence_scores = np.max(model.predict_proba(X_test), axis=1)
        else:
            y_prob = None
            confidence_scores = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        metrics['inference_time_ms'] = inference_time
        
        if confidence_scores is not None:
            metrics['mean_confidence'] = float(np.mean(confidence_scores))
            metrics['median_confidence'] = float(np.median(confidence_scores))
            metrics['min_confidence'] = float(np.min(confidence_scores))
            metrics['max_confidence'] = float(np.max(confidence_scores))
        
        # Print results
        print("\n--- Test Results ---")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"Sensitivity:   {metrics['sensitivity']:.4f}")
        print(f"Specificity:   {metrics['specificity']:.4f}")
        print(f"F1-Score:      {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
        
        if confidence_scores is not None:
            print(f"\nConfidence Scores:")
            print(f"  Mean:   {metrics['mean_confidence']:.4f}")
            print(f"  Median: {metrics['median_confidence']:.4f}")
            print(f"  Range:  [{metrics['min_confidence']:.4f}, {metrics['max_confidence']:.4f}]")
        
        print(f"\nInference Time: {metrics['inference_time_ms']:.2f} ms per image")
        
        return metrics
    
    def save_model(self, model, model_name):
        """
        Save trained model
        
        Args:
            model: Trained model
            model_name: Name of the model
        """
        model_path = self.models_dir / f'{model_name}_model.pkl'
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
    
    def save_results(self, results, filename='training_results.json'):
        """
        Save training results to JSON
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = self.models_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_serializable = convert_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"✓ Results saved to: {output_path}")
    
    def plot_confusion_matrices(self, results):
        """
        Plot confusion matrices for all models
        
        Args:
            results: Dictionary of results for all models
        """
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, res) in enumerate(results.items()):
            metrics = res['test_metrics']
            cm = np.array([
                [metrics['true_negatives'], metrics['false_positives']],
                [metrics['false_negatives'], metrics['true_positives']]
            ])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Anemia'],
                       yticklabels=['Normal', 'Anemia'])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {metrics["accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to: {self.models_dir / 'confusion_matrices.png'}")
        plt.close()
    
    def plot_metrics_comparison(self, results):
        """
        Plot comparison of metrics across models
        
        Args:
            results: Dictionary of results for all models
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
        model_names = list(results.keys())
        
        data = {metric: [] for metric in metrics_to_plot}
        
        for model_name in model_names:
            test_metrics = results[model_name]['test_metrics']
            for metric in metrics_to_plot:
                data[metric].append(test_metrics[metric])
        
        # Create bar plot
        x = np.arange(len(model_names))
        width = 0.13
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            offset = width * (i - len(metrics_to_plot)/2)
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to: {self.models_dir / 'metrics_comparison.png'}")
        plt.close()


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("ML Model Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = AnemiaModelTrainer(data_dir='./data')
    
    # Load data
    print("\n[Step 1] Loading data...")
    data = trainer.load_data()
    
    if 'train' not in data:
        print("\nError: Training data not found!")
        print("Please run extract_features.py first.")
        return
    
    X_train = data['train']['features']
    y_train = data['train']['labels']
    X_val = data['val']['features']
    y_val = data['val']['labels']
    X_test = data['test']['features']
    y_test = data['test']['labels']
    
    # Create models
    print("\n[Step 2] Creating models...")
    models = trainer.create_models()
    
    # Train all models
    print("\n[Step 3] Training models...")
    all_results = {}
    
    for model_name, model in models.items():
        # Train
        trained_model, train_results = trainer.train_model(
            model_name, model, X_train, y_train, X_val, y_val
        )
        
        # Test
        test_metrics = trainer.test_model(
            model_name, trained_model, X_test, y_test
        )
        
        # Save model
        trainer.save_model(trained_model, model_name)
        
        # Store results
        all_results[model_name] = {
            'validation_metrics': train_results['metrics'],
            'test_metrics': test_metrics
        }
    
    # Save results
    print("\n[Step 4] Saving results...")
    trainer.save_results(all_results, 'training_results.json')
    
    # Create visualizations
    print("\n[Step 5] Creating visualizations...")
    trainer.plot_confusion_matrices(all_results)
    trainer.plot_metrics_comparison(all_results)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    print("\nModel Performance on Test Set:")
    print(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'AUC-ROC':<10} {'Time (ms)':<10}")
    print("-" * 65)
    
    for model_name, results in all_results.items():
        metrics = results['test_metrics']
        print(f"{model_name.replace('_', ' ').title():<25} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['auc_roc']:<10.4f} "
              f"{metrics['inference_time_ms']:<10.2f}")
    
    print("\n" + "="*60)
    print("\nNext steps:")
    print("1. Check ./models/ for saved models and results")
    print("2. Run inference/predict.py to test inference")
    print("3. Deploy the best model to mobile app")


if __name__ == "__main__":
    main()
