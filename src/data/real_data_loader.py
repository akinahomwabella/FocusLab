"""
Real Data Loader for Keystroke Dynamics Datasets

Supports multiple public datasets:
1. CMU Keystroke Dynamics Benchmark Dataset
2. UCI MEU-Mobile KSD Dataset
3. EmoSurv Emotion Keystroke Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import requests
from io import StringIO
import zipfile


class KeystrokeDataLoader:
    """Load and preprocess real keystroke dynamics datasets"""
    
    def __init__(self, data_dir: str = "data/real"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_cmu_dataset(
        self,
        download: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load CMU Keystroke Dynamics Benchmark Dataset
        
        Dataset: 51 users typing password ".tie5Roanl" 400 times each
        Features: Hold time, Down-Down time, Up-Down time for each key
        
        Source: https://www.cs.cmu.edu/~keystroke/
        Paper: Killourhy & Maxion (2009)
        
        Returns:
            df: DataFrame with timing features
            metadata: Dataset information
        """
        csv_path = self.data_dir / "cmu_keystroke.csv"
        
        if not csv_path.exists() and download:
            print("Downloading CMU Keystroke Dynamics Dataset...")
            url = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(csv_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded to {csv_path}")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please manually download from: https://www.cs.cmu.edu/~keystroke/")
                return None, {}
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Data structure:
        # subject: user ID
        # sessionIndex: session number (0-7)
        # rep: repetition within session (0-49)
        # H.period, DD.period.t, etc.: timing features
        
        metadata = {
            'name': 'CMU Keystroke Dynamics Benchmark',
            'n_users': df['subject'].nunique(),
            'n_samples': len(df),
            'password': '.tie5Roanl',
            'features': [col for col in df.columns if 'H.' in col or 'DD.' in col or 'UD.' in col],
            'n_features': len([col for col in df.columns if 'H.' in col or 'DD.' in col or 'UD.' in col])
        }
        
        print(f"\nCMU Dataset Loaded:")
        print(f"  Users: {metadata['n_users']}")
        print(f"  Total samples: {metadata['n_samples']}")
        print(f"  Features: {metadata['n_features']}")
        
        return df, metadata
    
    def preprocess_for_cognitive_states(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        aggregate_method: str = 'session'
    ) -> pd.DataFrame:
        """
        Convert keystroke data to features suitable for cognitive state inference
        
        Args:
            df: Raw keystroke dataframe
            metadata: Dataset metadata
            aggregate_method: 'session' or 'user' level aggregation
            
        Returns:
            Processed DataFrame with cognitive-relevant features
        """
        timing_cols = metadata['features']
        
        if aggregate_method == 'session':
            # Aggregate by user and session
            group_cols = ['subject', 'sessionIndex']
        else:
            # Aggregate by user only
            group_cols = ['subject']
        
        # Calculate aggregate statistics per session/user
        agg_features = []
        
        for idx, group in df.groupby(group_cols):
            timing_data = group[timing_cols].values.flatten()
            
            # Remove NaN values
            timing_data = timing_data[~np.isnan(timing_data)]
            
            if len(timing_data) == 0:
                continue
            
            # Calculate temporal trend features
            trend_features = self._extract_temporal_trends(timing_data)
            
            features = {
                'subject': idx[0] if isinstance(idx, tuple) else idx,
                'session': idx[1] if isinstance(idx, tuple) else 0,
                
                # Reaction time proxies
                'mean_timing': np.mean(timing_data),
                'median_timing': np.median(timing_data),
                'std_timing': np.std(timing_data),
                
                # Variability (fatigue indicator)
                'cv_timing': np.std(timing_data) / (np.mean(timing_data) + 1e-6),  # coefficient of variation
                'iqr_timing': np.percentile(timing_data, 75) - np.percentile(timing_data, 25),
                
                # Speed indicators
                'min_timing': np.min(timing_data),
                'max_timing': np.max(timing_data),
                'range_timing': np.max(timing_data) - np.min(timing_data),
                
                # Error proxies (outliers as potential mistakes)
                'n_outliers': np.sum(np.abs(timing_data - np.mean(timing_data)) > 2 * np.std(timing_data)),
                'outlier_rate': np.sum(np.abs(timing_data - np.mean(timing_data)) > 2 * np.std(timing_data)) / len(timing_data),
                
                # Consistency (focus indicator)
                'q25': np.percentile(timing_data, 25),
                'q75': np.percentile(timing_data, 75),
                
                # Sample size for context
                'n_keystrokes': len(timing_data),
                
                # Temporal trend features (key for fatigue vs distraction)
                'rt_trend': trend_features['rt_trend'],
                'early_vs_late_ratio': trend_features['early_vs_late_ratio'],
                'acceleration': trend_features['acceleration'],
                'recovery_rate': trend_features['recovery_rate'],
                'variability_trend': trend_features['variability_trend'],
                'outlier_clustering': trend_features['outlier_clustering']
            }
            
            agg_features.append(features)
        
        processed_df = pd.DataFrame(agg_features)
        
        # Infer cognitive states based on heuristics
        # This is a simple heuristic - in reality, you'd have ground truth labels
        processed_df['inferred_state'] = self._infer_cognitive_state(processed_df)
        
        return processed_df
    
    def _extract_temporal_trends(self, timing_data: np.ndarray) -> dict:
        """
        Extract temporal trend features from keystroke sequence.
        
        Key insight:
        - Fatigued: Progressive slowdown (positive trend) + steady degradation
        - Distracted: Erratic timing (high variability in recent vs early parts)
        
        Returns:
            Dictionary with temporal trend features
        """
        if len(timing_data) < 10:
            # Not enough data for trends
            return {
                'rt_trend': 0.0,
                'early_vs_late_ratio': 1.0,
                'acceleration': 0.0,
                'recovery_rate': 0.0,
                'variability_trend': 0.0,
                'outlier_clustering': 0.0
            }
        
        # Split sequence into quarters
        q1_end = len(timing_data) // 4
        q2_end = len(timing_data) // 2
        q3_end = 3 * len(timing_data) // 4
        
        early = timing_data[:q1_end]
        late = timing_data[q3_end:]
        
        # FATIGUE INDICATOR: Progressive slowdown (positive trend)
        # Calculate slope: is RT increasing over time?
        time_indices = np.arange(len(timing_data), dtype=float)
        if np.std(time_indices) > 0:
            rt_trend = np.polyfit(time_indices, timing_data, 1)[0]
        else:
            rt_trend = 0.0
        
        # FATIGUE vs DISTRACTION: Early vs late ratio
        # Fatigued: slow early, slower late (high ratio)
        # Distracted: fast early, erratic late (ratio ~1 but high variance)
        early_mean = np.mean(early)
        late_mean = np.mean(late)
        early_vs_late_ratio = late_mean / (early_mean + 1e-6)
        
        # DISTRACTION INDICATOR: Acceleration (second derivative)
        # Erratic = rapidly changing speeds
        if len(timing_data) > 2:
            diffs = np.diff(timing_data)
            acceleration = np.mean(np.abs(np.diff(diffs)))  # Mean absolute second derivative
        else:
            acceleration = 0.0
        
        # FATIGUE INDICATOR: Recovery rate
        # Fatigued person doesn't recover; distracted person may
        # Look at the slope in second half vs first half
        first_half = timing_data[:len(timing_data)//2]
        second_half = timing_data[len(timing_data)//2:]
        
        if len(first_half) > 1 and len(second_half) > 1:
            first_trend = np.polyfit(np.arange(len(first_half)), first_half, 1)[0]
            second_trend = np.polyfit(np.arange(len(second_half)), second_half, 1)[0]
            # Positive recovery = trend improving (negative change)
            recovery_rate = first_trend - second_trend
        else:
            recovery_rate = 0.0
        
        # FATIGUE INDICATOR: Variability trend
        # Fatigue: consistent increase in std over time
        # Calculate rolling std
        win_size = max(3, len(timing_data) // 10)
        early_std = np.std(early)
        late_std = np.std(late)
        variability_trend = late_std - early_std  # Positive = increasing variability
        
        # DISTRACTION INDICATOR: Outlier clustering
        # Distracted: outliers clustered in time (Spikes of errors)
        # Fatigued: outliers distributed (gradual degradation)
        median = np.median(timing_data)
        std = np.std(timing_data)
        outlier_mask = np.abs(timing_data - median) > 2 * std
        
        if np.sum(outlier_mask) > 1:
            outlier_indices = np.where(outlier_mask)[0]
            outlier_gaps = np.diff(outlier_indices)
            # Low mean gap = clustered (distraction)
            # High mean gap = distributed (fatigue)
            outlier_clustering = -np.mean(outlier_gaps) if len(outlier_gaps) > 0 else 0.0
        else:
            outlier_clustering = 0.0
        
        return {
            'rt_trend': float(rt_trend),           # Positive = slowing down (fatigue indicator)
            'early_vs_late_ratio': float(early_vs_late_ratio),      # >1.1 = slowdown
            'acceleration': float(acceleration),    # High = erratic (distraction indicator)
            'recovery_rate': float(recovery_rate),  # Positive = getting worse (fatigue)
            'variability_trend': float(variability_trend),  # Positive = variability increasing
            'outlier_clustering': float(outlier_clustering)  # Negative = clustered (distraction)
        }
    
    def _infer_cognitive_state(self, df: pd.DataFrame) -> np.ndarray:
        """
        Heuristically infer cognitive states from keystroke features.
        
        Improved rules using temporal trends:
        - Focused: Low variability, low RT, no deterioration
        - Fatigued: Progressive slowdown + increasing variability (positive trends)
        - Distracted: Erratic timing + high acceleration (temporal instability)
        """
        states = []
        
        # Normalize features for comparison
        cv_norm = (df['cv_timing'] - df['cv_timing'].mean()) / (df['cv_timing'].std() + 1e-6)
        mean_norm = (df['mean_timing'] - df['mean_timing'].mean()) / (df['mean_timing'].std() + 1e-6)
        rt_trend_norm = (df['rt_trend'] - df['rt_trend'].mean()) / (df['rt_trend'].std() + 1e-6)
        accel_norm = (df['acceleration'] - df['acceleration'].mean()) / (df['acceleration'].std() + 1e-6)
        var_trend_norm = (df['variability_trend'] - df['variability_trend'].mean()) / (df['variability_trend'].std() + 1e-6)
        
        for i in range(len(df)):
            cv = cv_norm.iloc[i]
            mean = mean_norm.iloc[i]
            rt_trend = rt_trend_norm.iloc[i]
            acceleration = accel_norm.iloc[i]
            var_trend = var_trend_norm.iloc[i]
            
            # Decision logic using temporal patterns
            if cv < -0.3 and mean < 0.3 and rt_trend < 0.2:  
                # Low variability, normal speed, not slowing
                states.append(0)  # Focused
            elif rt_trend > 0.3 or var_trend > 0.4:
                # Progressive slowdown OR increasing variability
                states.append(1)  # Fatigued
            elif acceleration > 0.5 or cv > 1.0:
                # High erratic behavior OR very high variability
                states.append(2)  # Distracted
            elif mean > 0.4:
                # Generally slow
                states.append(1)  # Fatigued
            else:
                states.append(0)  # Default to Focused
        
        return np.array(states)
    
    def prepare_hmm_features(
        self,
        df: pd.DataFrame,
        feature_cols: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for HMM training
        
        Args:
            df: Processed dataframe
            feature_cols: Columns to use as features
            
        Returns:
            X: Feature matrix
            y: State labels (if available)
        """
        if feature_cols is None:
            # Default to reaction time and error rate proxies
            feature_cols = ['mean_timing', 'outlier_rate']
        
        X = df[feature_cols].values
        
        # Standardize
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Get labels if available
        if 'inferred_state' in df.columns:
            y = df['inferred_state'].values
        else:
            y = None
        
        return X, y
    
    def load_uci_mobile_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load UCI MEU-Mobile KSD Dataset
        
        Dataset: 56 users on mobile devices (Nexus 7)
        Features: Hold, Up-Down, Down-Down, Pressure, Finger-Area
        
        Source: https://archive.ics.uci.edu/dataset/399/
        """
        print("\nTo use UCI Mobile dataset:")
        print("1. Visit: https://archive.ics.uci.edu/dataset/399/")
        print("2. Download 'meu-mobile-keyboard.zip'")
        print("3. Extract to data/real/uci_mobile/")
        print("\nOr use Python:")
        print("  from ucimlrepo import fetch_ucirepo")
        print("  dataset = fetch_ucirepo(id=399)")
        
        return None, {}
    
    def simulate_temporal_sequence(
        self,
        df: pd.DataFrame,
        user_id: int,
        sequence_length: int = 100
    ) -> pd.DataFrame:
        """
        Create a temporal sequence from user data for HMM testing
        
        Simulates a user going through sessions over time with
        changing cognitive states
        """
        user_data = df[df['subject'] == user_id].sort_values('session')
        
        if len(user_data) == 0:
            print(f"No data for user {user_id}")
            return None
        
        # Interpolate to create smooth temporal sequence
        sequence = []
        
        for i in range(sequence_length):
            # Map to original data with some noise
            idx = int((i / sequence_length) * (len(user_data) - 1))
            sample = user_data.iloc[idx].copy()
            
            # Add temporal variation (simulate fatigue over time)
            fatigue_factor = 1 + (i / sequence_length) * 0.3  # 30% slowdown
            sample['mean_timing'] *= fatigue_factor
            sample['std_timing'] *= (1 + fatigue_factor * 0.2)
            
            # Recalculate derived features
            sample['cv_timing'] = sample['std_timing'] / sample['mean_timing']
            
            sequence.append(sample)
        
        sequence_df = pd.DataFrame(sequence)
        sequence_df['time_step'] = range(sequence_length)
        
        return sequence_df


def main():
    """Demonstrate data loading and preprocessing"""
    loader = KeystrokeDataLoader()
    
    print("="*60)
    print("LOADING REAL KEYSTROKE DYNAMICS DATA")
    print("="*60)
    
    # Load CMU dataset
    df_raw, metadata = loader.load_cmu_dataset(download=True)
    
    if df_raw is not None:
        # Preprocess for cognitive state inference
        print("\n" + "="*60)
        print("PREPROCESSING FOR COGNITIVE STATE INFERENCE")
        print("="*60)
        
        df_processed = loader.preprocess_for_cognitive_states(
            df_raw,
            metadata,
            aggregate_method='session'
        )
        
        print(f"\nProcessed {len(df_processed)} samples")
        print(f"Features: {df_processed.columns.tolist()}")
        print(f"\nState distribution:")
        print(df_processed['inferred_state'].value_counts())
        
        # Save processed data
        output_path = Path('data/processed/cmu_keystroke_processed.csv')
        df_processed.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed data to {output_path}")
        
        # Prepare for HMM
        X, y = loader.prepare_hmm_features(df_processed)
        print(f"\nHMM Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Create a temporal sequence for one user
        print("\n" + "="*60)
        print("CREATING TEMPORAL SEQUENCE")
        print("="*60)
        
        user_id = df_processed['subject'].iloc[0]
        sequence = loader.simulate_temporal_sequence(df_processed, user_id, sequence_length=200)
        
        if sequence is not None:
            sequence_path = Path('data/processed/cmu_user_sequence.csv')
            sequence.to_csv(sequence_path, index=False)
            print(f"✓ Created temporal sequence for user {user_id}")
            print(f"✓ Saved to {sequence_path}")
    
    print("\n" + "="*60)
    print("✅ DATA LOADING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
