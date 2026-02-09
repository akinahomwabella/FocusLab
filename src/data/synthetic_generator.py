"""
Synthetic Data Generator for Cognitive State Inference

Generates realistic behavioral signals (reaction times, error rates) 
for three cognitive states: Focused, Fatigued, Distracted
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class StateParameters:
    """Parameters defining behavioral distributions for each cognitive state"""
    name: str
    reaction_time_mean: float  # milliseconds
    reaction_time_std: float
    error_rate_mean: float  # probability [0, 1]
    error_rate_std: float
    

class BehavioralDataGenerator:
    """Generate synthetic behavioral data with realistic cognitive state transitions"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Define realistic state parameters based on cognitive science literature
        self.states = {
            0: StateParameters(
                name="Focused",
                reaction_time_mean=450,  # ms
                reaction_time_std=80,
                error_rate_mean=0.05,
                error_rate_std=0.02
            ),
            1: StateParameters(
                name="Fatigued", 
                reaction_time_mean=650,  # slower
                reaction_time_std=150,   # more variable
                error_rate_mean=0.18,
                error_rate_std=0.05
            ),
            2: StateParameters(
                name="Distracted",
                reaction_time_mean=750,  # slowest
                reaction_time_std=200,   # most variable
                error_rate_mean=0.25,
                error_rate_std=0.08
            )
        }
        
        # Transition probability matrix (realistic state transitions)
        # Rows: current state, Columns: next state
        # [Focused, Fatigued, Distracted]
        self.transition_matrix = np.array([
            [0.85, 0.10, 0.05],  # From Focused
            [0.15, 0.70, 0.15],  # From Fatigued
            [0.20, 0.20, 0.60]   # From Distracted
        ])
        
        # Initial state distribution
        self.initial_probs = np.array([0.7, 0.2, 0.1])  # Usually start focused
        
    def generate_sequence(
        self, 
        n_samples: int = 1000,
        include_ground_truth: bool = True
    ) -> pd.DataFrame:
        """
        Generate a sequence of behavioral observations
        
        Args:
            n_samples: Number of time steps to generate
            include_ground_truth: Whether to include true hidden states
            
        Returns:
            DataFrame with columns: timestamp, reaction_time, error_rate, [true_state]
        """
        # Generate hidden state sequence
        states = self._generate_state_sequence(n_samples)
        
        # Generate observations from states
        reaction_times = []
        error_rates = []
        
        for state in states:
            params = self.states[state]
            
            # Reaction time (lognormal distribution is more realistic)
            rt = np.random.lognormal(
                mean=np.log(params.reaction_time_mean),
                sigma=params.reaction_time_std / params.reaction_time_mean
            )
            reaction_times.append(rt)
            
            # Error rate (beta distribution bounded in [0,1])
            # Convert mean/std to alpha/beta parameters
            alpha, beta = self._beta_params_from_mean_std(
                params.error_rate_mean,
                params.error_rate_std
            )
            er = np.random.beta(alpha, beta)
            error_rates.append(er)
        
        # Create DataFrame
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='5s'),
            'reaction_time': reaction_times,
            'error_rate': error_rates,
        }
        
        if include_ground_truth:
            data['true_state'] = states
            data['true_state_name'] = [self.states[s].name for s in states]
        
        df = pd.DataFrame(data)
        
        # Add some realistic features
        df['reaction_time_rolling_mean'] = df['reaction_time'].rolling(10, min_periods=1).mean()
        df['error_rate_rolling_mean'] = df['error_rate'].rolling(10, min_periods=1).mean()
        df['time_of_day'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
        
        return df
    
    def _generate_state_sequence(self, n_samples: int) -> np.ndarray:
        """Generate hidden state sequence using transition matrix"""
        states = np.zeros(n_samples, dtype=int)
        
        # Initial state
        states[0] = np.random.choice(3, p=self.initial_probs)
        
        # Generate transitions
        for t in range(1, n_samples):
            current_state = states[t-1]
            states[t] = np.random.choice(3, p=self.transition_matrix[current_state])
        
        return states
    
    @staticmethod
    def _beta_params_from_mean_std(mean: float, std: float) -> Tuple[float, float]:
        """Convert mean and std to beta distribution alpha and beta parameters"""
        variance = std ** 2
        alpha = mean * ((mean * (1 - mean) / variance) - 1)
        beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)
        return max(alpha, 0.1), max(beta, 0.1)  # Ensure positive
    
    def generate_multiple_users(
        self,
        n_users: int = 10,
        samples_per_user: int = 500
    ) -> pd.DataFrame:
        """Generate data for multiple users with individual differences"""
        all_data = []
        
        for user_id in range(n_users):
            # Add individual variability
            self._add_user_variability()
            
            df = self.generate_sequence(samples_per_user)
            df['user_id'] = f'user_{user_id:03d}'
            all_data.append(df)
            
            # Reset to base parameters
            self.__init__(seed=42 + user_id)
        
        return pd.concat(all_data, ignore_index=True)
    
    def _add_user_variability(self):
        """Add slight individual differences to state parameters"""
        for state_id in self.states:
            self.states[state_id].reaction_time_mean *= np.random.uniform(0.9, 1.1)
            self.states[state_id].error_rate_mean *= np.random.uniform(0.9, 1.1)
    
    def save_metadata(self, filepath: str):
        """Save state parameters and transition matrix"""
        metadata = {
            'states': {
                state_id: {
                    'name': params.name,
                    'reaction_time_mean': params.reaction_time_mean,
                    'reaction_time_std': params.reaction_time_std,
                    'error_rate_mean': params.error_rate_mean,
                    'error_rate_std': params.error_rate_std,
                }
                for state_id, params in self.states.items()
            },
            'transition_matrix': self.transition_matrix.tolist(),
            'initial_probs': self.initial_probs.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Generate and save synthetic datasets"""
    generator = BehavioralDataGenerator(seed=42)
    
    # Generate single user sequence (for development/testing)
    print("Generating single user data...")
    df_single = generator.generate_sequence(n_samples=2000)
    
    # Generate multi-user data (for training/evaluation)
    print("Generating multi-user data...")
    df_multi = generator.generate_multiple_users(n_users=20, samples_per_user=1000)
    
    # Save datasets
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_single.to_csv(output_dir / 'single_user_sequence.csv', index=False)
    df_multi.to_csv(output_dir / 'multi_user_sequences.csv', index=False)
    
    # Save metadata
    generator.save_metadata(output_dir / 'data_generation_params.json')
    
    print(f"\n✅ Generated datasets:")
    print(f"  - Single user: {len(df_single)} samples")
    print(f"  - Multi user: {len(df_multi)} samples ({df_multi['user_id'].nunique()} users)")
    print(f"\nState distribution (single user):")
    print(df_single['true_state_name'].value_counts())
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
