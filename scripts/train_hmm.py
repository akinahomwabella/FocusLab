"""
Training Script for Cognitive State HMM

1. Generate synthetic behavioral data
2. Train HMM model
3. Evaluate performance
4. Save trained model
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data.synthetic_generator import BehavioralDataGenerator
from src.models.hmm import CognitiveStateHMM


def prepare_data(df: pd.DataFrame, features: list = None):
    """Prepare data for HMM training"""
    if features is None:
        features = ['reaction_time', 'error_rate']
    
    X = df[features].values
    
    # Standardize features for better numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    return X_normalized, X_mean, X_std


def evaluate_model(
    model: CognitiveStateHMM,
    X: np.ndarray,
    true_states: np.ndarray,
    save_dir: Path
):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Predict states
    predicted_states, state_probs = model.predict_sequence(X)
    
    # Accuracy
    accuracy = accuracy_score(true_states, predicted_states)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        true_states,
        predicted_states,
        target_names=model.state_names
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_states, predicted_states)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=model.state_names,
        yticklabels=model.state_names
    )
    plt.title('Confusion Matrix: True vs Predicted States')
    plt.ylabel('True State')
    plt.xlabel('Predicted State')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150)
    print(f"\n✓ Saved confusion matrix to {save_dir / 'confusion_matrix.png'}")
    
    # Plot state probabilities over time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Subplot 1: True vs Predicted
    time_steps = np.arange(len(true_states))
    axes[0].plot(time_steps, true_states, 'g-', label='True State', alpha=0.7, linewidth=2)
    axes[0].plot(time_steps, predicted_states, 'r--', label='Predicted State', alpha=0.7, linewidth=2)
    axes[0].set_ylabel('State')
    axes[0].set_title('True vs Predicted States Over Time')
    axes[0].legend()
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(model.state_names)
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: State probabilities
    for i, state_name in enumerate(model.state_names):
        axes[1].plot(time_steps, state_probs[:, i], label=state_name, linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('State Probabilities Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'state_inference_over_time.png', dpi=150)
    print(f"✓ Saved state inference plot to {save_dir / 'state_inference_over_time.png'}")
    
    # Calculate calibration metrics
    calibration_metrics = calculate_calibration(state_probs, true_states)
    print(f"\nCalibration Metrics:")
    print(f"  Brier Score: {calibration_metrics['brier_score']:.4f}")
    print(f"  Average Confidence: {calibration_metrics['avg_confidence']:.4f}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'calibration': calibration_metrics
    }


def calculate_calibration(probabilities: np.ndarray, true_states: np.ndarray) -> dict:
    """Calculate calibration metrics"""
    n_samples = len(true_states)
    
    # Brier score
    one_hot_true = np.zeros_like(probabilities)
    one_hot_true[np.arange(n_samples), true_states] = 1
    brier_score = np.mean(np.sum((probabilities - one_hot_true) ** 2, axis=1))
    
    # Average confidence in correct predictions
    correct_probs = probabilities[np.arange(n_samples), true_states]
    avg_confidence = np.mean(correct_probs)
    
    return {
        'brier_score': float(brier_score),
        'avg_confidence': float(avg_confidence)
    }


def plot_training_history(history: dict, save_dir: Path):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['log_likelihood'], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('HMM Training: Log-Likelihood Over Iterations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150)
    print(f"✓ Saved training history to {save_dir / 'training_history.png'}")


def visualize_learned_parameters(model: CognitiveStateHMM, save_dir: Path):
    """Visualize learned HMM parameters"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Transition matrix
    sns.heatmap(
        model.transition_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=model.state_names,
        yticklabels=model.state_names,
        ax=axes[0],
        cbar_kws={'label': 'Probability'}
    )
    axes[0].set_title('Learned Transition Matrix')
    axes[0].set_xlabel('Next State')
    axes[0].set_ylabel('Current State')
    
    # Emission parameters (means)
    emission_df = pd.DataFrame(
        model.emission_means,
        columns=model.feature_names,
        index=model.state_names
    )
    
    # Normalize for visualization
    emission_normalized = (emission_df - emission_df.min()) / (emission_df.max() - emission_df.min())
    
    sns.heatmap(
        emission_normalized,
        annot=emission_df.values,
        fmt='.2f',
        cmap='coolwarm',
        ax=axes[1],
        cbar_kws={'label': 'Normalized Value'}
    )
    axes[1].set_title('Learned Emission Parameters (Means)')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('State')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'learned_parameters.png', dpi=150)
    print(f"✓ Saved parameter visualization to {save_dir / 'learned_parameters.png'}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("FOCUSLAB: Training Cognitive State HMM")
    print("="*60)
    
    # Create output directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic behavioral data...")
    generator = BehavioralDataGenerator(seed=42)
    
    # Training data
    df_train = generator.generate_sequence(n_samples=2000, include_ground_truth=True)
    df_train.to_csv(output_dir / 'train_data.csv', index=False)
    
    # Test data (different seed for realistic evaluation)
    generator_test = BehavioralDataGenerator(seed=123)
    df_test = generator_test.generate_sequence(n_samples=500, include_ground_truth=True)
    df_test.to_csv(output_dir / 'test_data.csv', index=False)
    
    print(f"  ✓ Generated {len(df_train)} training samples")
    print(f"  ✓ Generated {len(df_test)} test samples")
    
    # Step 2: Prepare data
    print("\n[2/5] Preparing data...")
    X_train, X_mean, X_std = prepare_data(df_train)
    X_test, _, _ = prepare_data(df_test)  # Use training stats
    
    # Save normalization parameters
    norm_params = {
        'mean': X_mean.tolist(),
        'std': X_std.tolist(),
        'features': ['reaction_time', 'error_rate']
    }
    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    print("  ✓ Data normalized and ready")
    
    # Step 3: Train HMM
    print("\n[3/5] Training HMM model...")
    model = CognitiveStateHMM(
        n_states=3,
        state_names=["Focused", "Fatigued", "Distracted"],
        feature_names=["reaction_time", "error_rate"]
    )
    
    model.initialize_parameters(X_train, method='kmeans')
    history = model.fit(X_train, n_iter=100, tol=1e-4, verbose=True)
    
    print("  ✓ Model training completed")
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    metrics = evaluate_model(
        model,
        X_test,
        df_test['true_state'].values,
        output_dir
    )
    
    # Save evaluation metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Step 5: Save model and visualizations
    print("\n[5/5] Saving model and visualizations...")
    model.save(models_dir / 'cognitive_state_hmm.pkl')
    
    plot_training_history(history, output_dir)
    visualize_learned_parameters(model, output_dir)
    
    # Save model parameters as JSON for inspection
    params_summary = model.get_parameters_summary()
    with open(output_dir / 'model_parameters.json', 'w') as f:
        json.dump(params_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModel saved to: {models_dir / 'cognitive_state_hmm.pkl'}")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey Metrics:")
    print(f"  - Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Brier Score: {metrics['calibration']['brier_score']:.4f}")
    
    # Test online inference
    print("\n" + "="*60)
    print("Testing Online Inference")
    print("="*60)
    model.reset_belief()
    for i in range(5):
        result = model.predict_online(X_test[i])
        true_state = model.state_names[df_test['true_state'].values[i]]
        print(f"\nStep {i+1}:")
        print(f"  True State: {true_state}")
        print(f"  Predicted: {result['predicted_state']} (confidence: {result['confidence']:.3f})")
        print(f"  All probabilities: {result['state_probabilities']}")


if __name__ == "__main__":
    main()
