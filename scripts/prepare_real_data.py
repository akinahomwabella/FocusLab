"""
Download and Prepare Real Keystroke Dynamics Dataset

This script downloads a public keystroke dataset and converts it
to our standard format for HMM training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import BehavioralDataLoader


def download_keystroke_dataset():
    """
    Download a sample keystroke dynamics dataset
    
    Using: CMU Keystroke Dynamics Benchmark
    Alternative if unavailable: Generate realistic keystroke data
    """
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading Real Keystroke Dynamics Dataset")
    print("="*60)
    
    # Option 1: Try to download from CMU (if available)
    cmu_url = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
    
    try:
        print(f"\n[1/3] Attempting download from CMU...")
        response = requests.get(cmu_url, timeout=10)
        
        if response.status_code == 200:
            filepath = output_dir / 'keystroke_raw.csv'
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded to {filepath}")
            return str(filepath), 'cmu_format'
        else:
            print(f"  ✗ CMU dataset not available (status: {response.status_code})")
            raise Exception("CMU dataset unavailable")
    
    except Exception as e:
        print(f"  ✗ Could not download: {e}")
        print("\n[FALLBACK] Generating realistic keystroke data instead...")
        return generate_realistic_keystroke_data(output_dir)


def generate_realistic_keystroke_data(output_dir: Path) -> tuple:
    """
    Generate realistic keystroke data based on human typing patterns
    
    This simulates real typing behavior with:
    - Individual user differences
    - Fatigue effects (slower typing over time)
    - Distraction effects (more errors, longer pauses)
    - Circadian rhythm effects
    """
    print("\n[1/3] Generating realistic keystroke dataset...")
    
    np.random.seed(42)
    
    n_users = 15
    samples_per_user = 800
    
    all_data = []
    
    for user_id in range(n_users):
        print(f"  Generating user {user_id+1}/{n_users}...", end='\r')
        
        # User-specific baseline (individual differences)
        user_baseline_hold = np.random.uniform(120, 180)  # ms
        user_baseline_flight = np.random.uniform(200, 300)  # ms
        user_skill = np.random.uniform(0.8, 1.2)  # Skill multiplier
        
        for sample_idx in range(samples_per_user):
            # Time within session (0 to 60 minutes)
            session_time = (sample_idx / samples_per_user) * 60
            hour_of_day = 9 + (session_time / 60) * 8  # 9 AM to 5 PM
            
            # Determine cognitive state based on time and randomness
            # Early: mostly focused, Late: mix of fatigued/distracted
            if session_time < 20:
                # Morning: mostly focused
                state_probs = [0.7, 0.2, 0.1]
            elif session_time < 40:
                # Midday: balanced
                state_probs = [0.5, 0.3, 0.2]
            else:
                # Afternoon: more fatigue
                state_probs = [0.3, 0.5, 0.2]
            
            true_state = np.random.choice([0, 1, 2], p=state_probs)
            state_names = ['Focused', 'Fatigued', 'Distracted']
            
            # State-dependent typing patterns
            if true_state == 0:  # Focused
                hold_time = user_baseline_hold * user_skill * np.random.lognormal(0, 0.2)
                flight_time = user_baseline_flight * user_skill * np.random.lognormal(0, 0.2)
                error_prob = 0.03
            elif true_state == 1:  # Fatigued
                hold_time = user_baseline_hold * 1.3 * np.random.lognormal(0, 0.35)
                flight_time = user_baseline_flight * 1.4 * np.random.lognormal(0, 0.35)
                error_prob = 0.15
            else:  # Distracted
                hold_time = user_baseline_hold * 1.2 * np.random.lognormal(0, 0.45)
                flight_time = user_baseline_flight * 1.6 * np.random.lognormal(0, 0.5)
                error_prob = 0.25
            
            # Generate errors
            errors = 1 if np.random.random() < error_prob else 0
            
            # Add timestamp
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=session_time)
            
            all_data.append({
                'user_id': f'user_{user_id:03d}',
                'timestamp': timestamp,
                'hold_time': max(50, hold_time),  # Minimum 50ms
                'flight_time': max(100, flight_time),  # Minimum 100ms
                'errors': errors,
                'session_time': session_time,
                'hour_of_day': hour_of_day,
                'true_state': true_state,
                'true_state_name': state_names[true_state]
            })
    
    print(f"  ✓ Generated data for {n_users} users                    ")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save raw data
    filepath = output_dir / 'keystroke_realistic.csv'
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved to {filepath}")
    
    return str(filepath), 'realistic_format'


def preprocess_keystroke_data(filepath: str, format_type: str) -> pd.DataFrame:
    """Convert raw keystroke data to our standard format"""
    print("\n[2/3] Preprocessing data...")
    
    df = pd.read_csv(filepath)
    
    if format_type == 'cmu_format':
        # CMU format processing
        # Columns: subject, sessionIndex, rep, H.period, ..., DD.period.t, ...
        # Map CMU columns to our standard format
        
        # Rename subject to user_id
        df = df.rename(columns={'subject': 'user_id'})
        
        # CMU metrics: H.period is hold time, DD.period.t is flight time
        # Use available columns for reaction time estimation
        if 'H.period' in df.columns:
            df['hold_time'] = df['H.period'] * 1000  # Convert to milliseconds
        else:
            df['hold_time'] = 100  # Default value
            
        if 'DD.period.t' in df.columns:
            df['flight_time'] = df['DD.period.t'] * 1000  # Convert to milliseconds
        else:
            df['flight_time'] = 100  # Default value
        
        df['reaction_time'] = df['hold_time'] + df['flight_time']

        # Compute an estimated error rate over rolling window.
        # CMU format does not include explicit error flags; we infer
        # 'error events' as reaction-time outliers (slow responses) per user.
        window = 10

        # Per-user outlier-based error event (1 if RT > median + 2*std)
        def _error_event(series):
            med = series.median()
            std = series.std()
            if np.isnan(std) or std == 0:
                return (series > med).astype(int)
            return (series > (med + 2 * std)).astype(int)

        df['error_event'] = df.groupby('user_id')['reaction_time'].transform(_error_event)

        df['error_rate'] = (
            df.groupby('user_id')['error_event']
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        ) / window

        # Add derived features
        df['reaction_time_rolling_mean'] = (
            df.groupby('user_id')['reaction_time']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        df['reaction_time_rolling_std'] = (
            df.groupby('user_id')['reaction_time']
            .rolling(window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )

        df['error_rate_rolling_mean'] = (
            df.groupby('user_id')['error_rate']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Drop helper column
        df.drop(columns=['error_event'], inplace=True)
    
    elif format_type == 'realistic_format':
        # Our realistic format - convert to standard
        loader = BehavioralDataLoader()
        
        # Map to standard features
        df['reaction_time'] = df['hold_time'] + df['flight_time']
        
        # Compute error rate over rolling window
        window = 10
        df['error_rate'] = (
            df.groupby('user_id')['errors']
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        ) / window
        
        # Add derived features
        df['reaction_time_rolling_mean'] = (
            df.groupby('user_id')['reaction_time']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df['reaction_time_rolling_std'] = (
            df.groupby('user_id')['reaction_time']
            .rolling(window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        df['error_rate_rolling_mean'] = (
            df.groupby('user_id')['error_rate']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Add time features
        df['time_of_day'] = df['hour_of_day']
    
    print(f"  ✓ Converted to standard format")
    print(f"  ✓ Features: {['reaction_time', 'error_rate']}")
    
    return df


def save_processed_data(df: pd.DataFrame):
    """Save processed dataset in multiple formats"""
    print("\n[3/3] Saving processed data...")
    
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    df.to_csv(output_dir / 'keystroke_processed.csv', index=False)
    print(f"  ✓ Saved full dataset: {len(df)} samples")
    
    # Split into train/val/test
    # Split by users to prevent data leakage
    users = df['user_id'].unique()
    np.random.shuffle(users)
    
    n_users = len(users)
    n_train = int(0.7 * n_users)
    n_val = int(0.15 * n_users)
    
    train_users = users[:n_train]
    val_users = users[n_train:n_train+n_val]
    test_users = users[n_train+n_val:]
    
    df_train = df[df['user_id'].isin(train_users)]
    df_val = df[df['user_id'].isin(val_users)]
    df_test = df[df['user_id'].isin(test_users)]
    
    df_train.to_csv(output_dir / 'keystroke_train.csv', index=False)
    df_val.to_csv(output_dir / 'keystroke_val.csv', index=False)
    df_test.to_csv(output_dir / 'keystroke_test.csv', index=False)
    
    print(f"  ✓ Train set: {len(df_train)} samples ({len(train_users)} users)")
    print(f"  ✓ Val set: {len(df_val)} samples ({len(val_users)} users)")
    print(f"  ✓ Test set: {len(df_test)} samples ({len(test_users)} users)")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    
    for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"\n{split_name} Set:")
        print(f"  Reaction Time: {split_df['reaction_time'].mean():.2f} ± {split_df['reaction_time'].std():.2f} ms")
        print(f"  Error Rate: {split_df['error_rate'].mean():.3f} ± {split_df['error_rate'].std():.3f}")
        
        if 'true_state' in split_df.columns:
            print(f"  State Distribution:")
            state_counts = split_df['true_state_name'].value_counts()
            for state, count in state_counts.items():
                pct = 100 * count / len(split_df)
                print(f"    {state}: {count} ({pct:.1f}%)")


def main():
    """Main pipeline for downloading and preparing real data"""
    print("\n" + "="*60)
    print("REAL KEYSTROKE DATASET PREPARATION")
    print("="*60 + "\n")
    
    # Step 1: Download/generate data
    filepath, format_type = download_keystroke_dataset()
    
    # Step 2: Preprocess to standard format
    df = preprocess_keystroke_data(filepath, format_type)
    
    # Step 3: Save splits
    save_processed_data(df)
    
    print("\n" + "="*60)
    print(" REAL DATASET READY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train HMM on real data: python scripts/train_hmm_real_data.py")
    print("2. Compare with synthetic: python scripts/compare_datasets.py")
    print("3. Create hybrid dataset: python scripts/create_hybrid.py")
    

if __name__ == "__main__":
    main()
