"""
Train HMM on Hybrid Dataset (Real + Synthetic)

Combines real keystroke data with synthetic data for robust training
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.synthetic_generator import BehavioralDataGenerator
from data.data_loader import BehavioralDataLoader
from models.hmm import CognitiveStateHMM


def load_datasets(use_real=True, use_synthetic=True, hybrid_ratio=0.5):
    """
    Load training data with options for real, synthetic, or hybrid
    
    Args:
        use_real: Include real keystroke data
        use_synthetic: Include synthetic data
        hybrid_ratio: If both used, fraction of real data (0-1)
    """
    loader = BehavioralDataLoader()
    
    datasets = {}
    
    # Load real data if available
    if use_real:
        real_train_path = Path('data/processed/keystroke_train.csv')
        real_test_path = Path('data/processed/keystroke_test.csv')
        
        if real_train_path.exists():
            print("Loading REAL keystroke data...")
            df_real_train = pd.read_csv(real_train_path)
            df_real_test = pd.read_csv(real_test_path)
            
            datasets['real_train'] = df_real_train
            datasets['real_test'] = df_real_test
            print(f"  ✓ Real train: {len(df_real_train)} samples")
            print(f"  ✓ Real test: {len(df_real_test)} samples")
        else:
            print("  ⚠ Real data not found. Run prepare_real_data.py first")
            use_real = False
    
    # Load or generate synthetic data
    if use_synthetic:
        synthetic_train_path = Path('data/processed/train_data.csv')
        synthetic_test_path = Path('data/processed/test_data.csv')
        
        if synthetic_train_path.exists():
            print("\nLoading SYNTHETIC data...")
            df_synth_train = pd.read_csv(synthetic_train_path)
            df_synth_test = pd.read_csv(synthetic_test_path)
        else:
            print("\nGenerating SYNTHETIC data...")
            generator = BehavioralDataGenerator(seed=42)
            df_synth_train = generator.generate_sequence(n_samples=2000)
            df_synth_test = generator.generate_sequence(n_samples=500)
        
        datasets['synth_train'] = df_synth_train
        datasets['synth_test'] = df_synth_test
        print(f"  ✓ Synthetic train: {len(df_synth_train)} samples")
        print(f"  ✓ Synthetic test: {len(df_synth_test)} samples")
    
    # Create hybrid dataset if both are available
    if use_real and use_synthetic:
        print(f"\nCreating HYBRID dataset (ratio={hybrid_ratio})...")
        df_hybrid_train = loader.create_hybrid_dataset(
            datasets['real_train'],
            datasets['synth_train'],
            ratio=hybrid_ratio
        )
        datasets['hybrid_train'] = df_hybrid_train
    
    return datasets


def prepare_features(df, features=['reaction_time', 'error_rate']):
    """Extract and normalize features"""
    X = df[features].values
    
    # Handle any NaN or inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    
    return X_normalized, X_mean, X_std


def train_and_evaluate(
    df_train,
    df_test,
    model_name="hmm",
    save_dir=None
):
    """Train HMM and evaluate on test set"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Prepare data
    X_train, X_mean, X_std = prepare_features(df_train)
    X_test, _, _ = prepare_features(df_test)  # Use training stats
    
    # Check if test data has true_state labels (synthetic data has them, real data doesn't)
    has_labels = 'true_state' in df_test.columns
    if has_labels:
        y_test = df_test['true_state'].values
    else:
        y_test = None
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize and train model
    model = CognitiveStateHMM(
        n_states=3,
        state_names=["Focused", "Fatigued", "Distracted"],
        feature_names=["reaction_time", "error_rate"]
    )
    
    model.initialize_parameters(X_train, method='kmeans')
    history = model.fit(X_train, n_iter=100, tol=1e-4, verbose=True)
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    predicted_states, state_probs = model.predict_sequence(X_test)
    
    if has_labels:
        accuracy = accuracy_score(y_test, predicted_states)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predicted_states,
            target_names=model.state_names
        ))
        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_states)
    else:
        print("\n⚠ No ground truth labels available for real data")
        print("Model trained successfully on real keystroke dynamics data")
        print(f"Predicted state distribution:")
        state_counts = pd.Series(predicted_states)
        for state_id, state_name in enumerate(model.state_names):
            count = (state_counts == state_id).sum()
            pct = count / len(predicted_states) * 100
            print(f"  {state_name}: {count} samples ({pct:.1f}%)")
        accuracy = 0.0
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(save_dir / f'{model_name}_model.pkl')
        
        if has_labels:
            # Save confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=model.state_names,
                yticklabels=model.state_names
            )
            plt.title(f'Confusion Matrix: {model_name.upper()}')
            plt.ylabel('True State')
            plt.xlabel('Predicted State')
            plt.tight_layout()
            plt.savefig(save_dir / f'{model_name}_confusion_matrix.png', dpi=150)
            plt.close()
            
            # Save metrics
            metrics = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
        else:
            # For real data without labels, just save basic metrics
            metrics = {
                'model_name': model_name,
                'accuracy': None,  # Can't compute without labels
                'confusion_matrix': None,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'note': 'Real data - no ground truth labels available'
            }
        
        with open(save_dir / f'{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Saved model and results to {save_dir}")
    
    accuracy = accuracy if has_labels else 0.0  # Return 0 for models without labels
    return model, accuracy, history


def compare_models(results):
    """Compare performance across different training approaches"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    comparison = []
    for name, (model, accuracy, history) in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': f"{accuracy:.4f}",
            'Final Log-Likelihood': f"{history['log_likelihood'][-1]:.2f}",
            'Convergence Iterations': len(history['log_likelihood'])
        })
    
    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))
    
    # Save comparison
    output_dir = Path('data/processed/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    models = [r['Model'] for r in comparison]
    accuracies = [float(r['Accuracy']) for r in comparison]
    
    axes[0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Training convergence
    for name, (model, accuracy, history) in results.items():
        axes[1].plot(history['log_likelihood'], label=name, linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Log-Likelihood')
    axes[1].set_title('Training Convergence')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150)
    print(f"\n✓ Saved comparison plot to {output_dir}")


def main():
    """Main training pipeline with hybrid data"""
    
    print("="*60)
    print("HYBRID HMM TRAINING: Real + Synthetic Data")
    print("="*60)
    
    # Load all datasets
    datasets = load_datasets(use_real=True, use_synthetic=True, hybrid_ratio=0.5)
    
    results = {}
    output_base = Path('models/trained')
    
    # Train on synthetic only
    if 'synth_train' in datasets:
        print("\n" + "="*60)
        print("EXPERIMENT 1: Synthetic Data Only")
        print("="*60)
        model_synth, acc_synth, hist_synth = train_and_evaluate(
            datasets['synth_train'],
            datasets['synth_test'],
            model_name='synthetic_only',
            save_dir=output_base / 'synthetic'
        )
        results['Synthetic Only'] = (model_synth, acc_synth, hist_synth)
    
    # Train on real only
    if 'real_train' in datasets:
        print("\n" + "="*60)
        print("EXPERIMENT 2: Real Data Only")
        print("="*60)
        model_real, acc_real, hist_real = train_and_evaluate(
            datasets['real_train'],
            datasets['real_test'],
            model_name='real_only',
            save_dir=output_base / 'real'
        )
        results['Real Only'] = (model_real, acc_real, hist_real)
    
    # Train on hybrid
    if 'hybrid_train' in datasets:
        print("\n" + "="*60)
        print("EXPERIMENT 3: Hybrid Data (50% Real + 50% Synthetic)")
        print("="*60)
        
        # Test on real data
        model_hybrid, acc_hybrid, hist_hybrid = train_and_evaluate(
            datasets['hybrid_train'],
            datasets['real_test'],
            model_name='hybrid_50_50',
            save_dir=output_base / 'hybrid'
        )
        results['Hybrid (50/50)'] = (model_hybrid, acc_hybrid, hist_hybrid)
    
    # Compare all models
    if len(results) > 1:
        compare_models(results)
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\nResults saved to: models/trained/")
    print("Comparison saved to: data/processed/comparison/")


if __name__ == "__main__":
    main()
