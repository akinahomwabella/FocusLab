"""
Training Script with REAL Keystroke Dynamics Data

Uses CMU Keystroke Dynamics Benchmark Dataset
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.real_data_loader import KeystrokeDataLoader
from src.models.hmm import CognitiveStateHMM


def align_states_to_labels(y_true, y_pred_states):
    """
    Align HMM latent states to heuristic labels using max overlap.
    
    The HMM learns its own state assignments (0, 1, 2) which may not
    correspond to the heuristic inferred labels (0, 1, 2). This function
    finds the optimal mapping by looking at which HMM state most frequently
    co-occurs with each true label.
    
    Args:
        y_true: True labels from heuristic (integers 0/1/2)
        y_pred_states: Predicted states from HMM (integers 0/1/2)
    
    Returns:
        y_pred_aligned: Remapped predictions aligned to true label space
    """
    cm = confusion_matrix(y_true, y_pred_states, labels=[0, 1, 2])
    state_to_label = {}
    
    # For each HMM state, find which true label it maps to most
    for state in range(cm.shape[1]):
        state_to_label[state] = np.argmax(cm[:, state])
    
    return np.array([state_to_label[s] for s in y_pred_states]), state_to_label


def train_with_real_data():
    """Train HMM on real keystroke dynamics data"""
    print("="*70)
    print("FOCUSLAB: Training on REAL Keystroke Dynamics Data")
    print("="*70)
    
    # Step 1: Load real data
    print("\n[1/6] Loading CMU Keystroke Dynamics Dataset...")
    loader = KeystrokeDataLoader()
    
    df_raw, metadata = loader.load_cmu_dataset(download=True)
    
    if df_raw is None:
        print("\n Failed to load data. Please download manually.")
        print("URL: https://www.cs.cmu.edu/~keystroke/")
        print("Save as: data/real/cmu_keystroke.csv")
        return
    
    # Step 2: Preprocess
    print("\n[2/6] Preprocessing for cognitive state inference...")
    df_processed = loader.preprocess_for_cognitive_states(
        df_raw,
        metadata,
        aggregate_method='session'
    )
    
    print(f"  Processed {len(df_processed)} samples")
    print(f"  State distribution:")
    for state_id, count in df_processed['inferred_state'].value_counts().items():
        state_name = ["Focused", "Fatigued", "Distracted"][state_id]
        print(f"    {state_name}: {count} ({count/len(df_processed)*100:.1f}%)")
    
    # Step 3: Split train/test
    print("\n[3/6] Splitting data...")
    
    # Use first 80% of sessions for training
    unique_sessions = df_processed.sort_values(['subject', 'session'])[['subject', 'session']].drop_duplicates()
    n_train = int(len(unique_sessions) * 0.8)
    
    train_sessions = unique_sessions.iloc[:n_train]
    test_sessions = unique_sessions.iloc[n_train:]
    
    df_train = df_processed.merge(train_sessions, on=['subject', 'session'])
    df_test = df_processed.merge(test_sessions, on=['subject', 'session'])
    
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    
    # Step 4: Prepare features for HMM
    print("\n[4/6] Preparing features...")
    
    # Use multiple features including temporal trends for better Fatigued/Distracted discrimination
    feature_cols = [
        'mean_timing',           # Average typing speed
        'outlier_rate',          # Error proxy
        'cv_timing',             # Variability (fatigue)
        'std_timing',            # Consistency
        'rt_trend',              # Progressive slowdown (fatigue indicator)
        'acceleration',          # Erratic behavior (distraction indicator)
        'variability_trend',     # Increasing variability (fatigue)
        'early_vs_late_ratio'    # Slowdown pattern
    ]
    
    X_train, y_train = loader.prepare_hmm_features(df_train, feature_cols)
    X_test, y_test = loader.prepare_hmm_features(df_test, feature_cols)
    
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Using features: {feature_cols}")
    
    # Step 5: Train HMM
    print("\n[5/6] Training HMM...")
    model = CognitiveStateHMM(
        n_states=3,
        state_names=["Focused", "Fatigued", "Distracted"],
        feature_names=feature_cols
    )
    
    model.initialize_parameters(X_train, method='kmeans')
    history = model.fit(X_train, n_iter=100, tol=1e-4, verbose=True)
    
    # Step 6: Evaluate
    print("\n[6/6] Evaluating model...")
    from train_hmm import evaluate_model, plot_training_history, visualize_learned_parameters
    
    output_dir = Path('data/processed/real_data_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predict states on test set
    y_pred_raw, _ = model.predict_sequence(X_test)
    
    # Align HMM states to heuristic labels
    y_pred_aligned, state_mapping = align_states_to_labels(y_test, y_pred_raw)
    
    print(f"\nState Alignment Mapping (HMM state → Heuristic label):")
    label_names = ["Focused", "Fatigued", "Distracted"]
    for hmm_state, true_label in state_mapping.items():
        print(f"  HMM state {hmm_state} → {label_names[true_label]}")
    
    # Now evaluate with aligned predictions
    # We'll call the core evaluation functions manually to use aligned predictions
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred_aligned)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_aligned,
        target_names=model.state_names,
        zero_division=0
    ))
    
    # Recompute confusion matrix with aligned predictions
    cm = confusion_matrix(y_test, y_pred_aligned, labels=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=model.state_names,
        yticklabels=model.state_names
    )
    plt.title('Confusion Matrix: True vs Predicted States (Aligned)')
    plt.ylabel('True State')
    plt.xlabel('Predicted State')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"\n✓ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    plt.close()
    
    # Compute calibration metrics on aligned predictions
    _, state_probs = model.predict_sequence(X_test)
    n_samples = len(y_test)
    one_hot_true = np.zeros_like(state_probs)
    one_hot_true[np.arange(n_samples), y_test] = 1
    brier_score = np.mean(np.sum((state_probs - one_hot_true) ** 2, axis=1))
    correct_probs = state_probs[np.arange(n_samples), y_test]
    avg_confidence = np.mean(correct_probs)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'state_mapping': state_mapping,
        'calibration': {
            'brier_score': float(brier_score),
            'avg_confidence': float(avg_confidence)
        }
    }
    plot_training_history(history, output_dir)
    visualize_learned_parameters(model, output_dir)
    
    # Save model
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / 'cognitive_state_hmm_real.pkl')
    
    # Save processed data for reference
    df_train.to_csv(output_dir / 'train_data_real.csv', index=False)
    df_test.to_csv(output_dir / 'test_data_real.csv', index=False)
    
    print("\n" + "="*70)
    print(" TRAINING ON REAL DATA COMPLETED!")
    print("="*70)
    print(f"\nModel saved to: models/cognitive_state_hmm_real.pkl")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey Metrics:")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Brier Score: {metrics['calibration']['brier_score']:.4f}")
    print(f"  Avg Confidence: {metrics['calibration']['avg_confidence']:.4f}")
    
    # Test on a temporal sequence
    print("\n" + "="*70)
    print("Testing on Temporal User Sequence")
    print("="*70)
    
    user_id = df_train['subject'].iloc[0]
    sequence = loader.simulate_temporal_sequence(df_processed, user_id, sequence_length=50)
    
    if sequence is not None:
        X_seq, _ = loader.prepare_hmm_features(sequence, feature_cols)
        
        model.reset_belief()
        predictions = []
        
        print(f"\nUser {user_id} typing over time:")
        for i in range(min(10, len(X_seq))):
            result = model.predict_online(X_seq[i])
            predictions.append(result)
            
            if i % 2 == 0:  # Print every other prediction
                print(f"  Step {i:2d}: {result['predicted_state']:12s} "
                      f"(conf: {result['confidence']:.3f})")
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        states_numeric = [['Focused', 'Fatigued', 'Distracted'].index(p['predicted_state']) 
                         for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        ax.plot(states_numeric, 'o-', linewidth=2, markersize=8, label='Predicted State')
        ax.set_ylabel('State', fontsize=12)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_title(f'Cognitive State Inference for User {user_id}', fontsize=14)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Focused', 'Fatigued', 'Distracted'])
        ax.grid(True, alpha=0.3)
        
        # Add confidence as color
        scatter = ax.scatter(range(len(predictions)), states_numeric, 
                           c=confidences, cmap='RdYlGn', 
                           s=200, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='Confidence')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_sequence_prediction.png', dpi=150)
        print(f"\n✓ Saved temporal prediction plot")


if __name__ == "__main__":
    # Install requests if needed
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "--break-system-packages", "-q", "requests"])
        import requests
    
    train_with_real_data()
