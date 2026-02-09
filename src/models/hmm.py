"""
Hidden Markov Model for Cognitive State Inference

Implements HMM with Gaussian emissions for inferring latent cognitive states
from noisy behavioral signals (reaction time, error rate).
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple, Dict, Optional, List
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
import json


@dataclass
class HMMParameters:
    """Store HMM parameters"""
    n_states: int
    n_features: int
    transition_matrix: np.ndarray
    initial_probs: np.ndarray
    emission_means: np.ndarray  # Shape: (n_states, n_features)
    emission_covs: np.ndarray   # Shape: (n_states, n_features, n_features)
    state_names: List[str]
    feature_names: List[str]


class CognitiveStateHMM:
    """
    Hidden Markov Model for real-time cognitive state inference
    
    States: Focused (0), Fatigued (1), Distracted (2)
    Observations: [reaction_time, error_rate]
    """
    
    def __init__(
        self,
        n_states: int = 3,
        state_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.n_states = n_states
        self.state_names = state_names or ["Focused", "Fatigued", "Distracted"]
        self.feature_names = feature_names or ["reaction_time", "error_rate"]
        self.n_features = len(self.feature_names)
        
        # Model parameters (to be learned)
        self.transition_matrix = None
        self.initial_probs = None
        self.emission_means = None  # Mean of Gaussian for each state
        self.emission_covs = None   # Covariance of Gaussian for each state
        
        # For online inference
        self.current_belief = None  # Current state probability distribution
        
    def initialize_parameters(self, X: np.ndarray, method: str = 'kmeans'):
        """
        Initialize model parameters before training
        
        Args:
            X: Observations, shape (n_samples, n_features)
            method: 'kmeans' or 'random'
        """
        n_samples = X.shape[0]
        
        if method == 'kmeans':
            # Use k-means to initialize emission parameters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            self.emission_means = kmeans.cluster_centers_
            self.emission_covs = np.array([
                self._compute_robust_covariance(X[labels == i]) 
                for i in range(self.n_states)
            ])
        else:
            # Random initialization
            self.emission_means = np.random.randn(self.n_states, self.n_features)
            self.emission_covs = np.array([
                np.eye(self.n_features) for _ in range(self.n_states)
            ])
        
        # Initialize transition matrix (slightly favor staying in same state)
        self.transition_matrix = np.random.dirichlet(
            alpha=[5] + [1] * (self.n_states - 1),
            size=self.n_states
        )
        np.fill_diagonal(self.transition_matrix, 0.6)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Uniform initial distribution
        self.initial_probs = np.ones(self.n_states) / self.n_states
    
    def _compute_robust_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute covariance matrix robustly, handling degenerate cases"""
        if len(X) <= 1:
            # Not enough samples, return identity matrix
            return np.eye(self.n_features)
        
        try:
            cov = np.cov(X.T)
            # If cov is 1D (single feature), reshape it
            if cov.ndim == 1:
                cov = np.diag(cov)
            # Add regularization to ensure positive definiteness
            cov = cov + 1e-6 * np.eye(self.n_features)
            return cov
        except:
            # If anything fails, return regularized identity
            return (1e-3) * np.eye(self.n_features) + np.eye(self.n_features)
    
    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train HMM using Baum-Welch algorithm (EM for HMMs)
        
        Args:
            X: Observations, shape (n_samples, n_features)
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            Dictionary with training history
        """
        n_samples = X.shape[0]
        
        # Initialize if not already done
        if self.emission_means is None:
            self.initialize_parameters(X, method='kmeans')
        
        history = {'log_likelihood': []}
        prev_ll = -np.inf
        
        for iteration in range(n_iter):
            # E-step: Forward-backward algorithm
           
            alpha, scales = self._forward(X)
            beta = self._backward(X, scales)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(alpha)
            history['log_likelihood'].append(log_likelihood)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.2f}")
            
            # Check convergence
            if abs(log_likelihood - prev_ll) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            prev_ll = log_likelihood
            
            # Compute gamma (state posteriors) and xi (transition posteriors)
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(X, alpha, beta)
            
            # M-step: Update parameters
            self._update_parameters(X, gamma, xi)
        
        return history
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward algorithm
        
        Returns:
            alpha: Forward probabilities, shape (n_samples, n_states)
        """
        n_samples = X.shape[0]
        alpha = np.zeros((n_samples, self.n_states))
        scales = np.zeros(n_samples)
        
        # Initialization
        emission_probs = self._emission_probability(X[0])
        alpha[0] = self.initial_probs * emission_probs
        alpha[0] /= alpha[0].sum()  # Normalize to prevent underflow
        
        # Recursion
        for t in range(1, n_samples):
            emission_probs = self._emission_probability(X[t])
            for j in range(self.n_states):
                alpha[t, j] = emission_probs[j] * np.sum(
                    alpha[t-1] * self.transition_matrix[:, j]
                )
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        return alpha, scales
        
      
    
    def _backward(self, X: np.ndarray, scales: np.ndarray):
        """
        Backward algorithm
        
        Returns:
            beta: Backward probabilities, shape (n_samples, n_states)
        """
        n_samples = X.shape[0]
        beta = np.zeros((n_samples, self.n_states))
        
        # Initialization
        beta[-1] = 1.0
        
        # Recursion
        for t in range(n_samples - 2, -1, -1):
            emission_probs = self._emission_probability(X[t + 1])
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] * emission_probs * beta[t + 1]
                )
            beta[t] /= scales[t + 1]
        
        return beta
    
    def _emission_probability(self, x: np.ndarray) -> np.ndarray:
        """
        Compute emission probability for observation x
        
        Args:
            x: Single observation, shape (n_features,)
            
        Returns:
            Probability of x given each state, shape (n_states,)
        """
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            probs[i] = multivariate_normal.pdf(
                x,
                mean=self.emission_means[i],
                cov=self.emission_covs[i],
                allow_singular=True
            )
        return probs + 1e-10  # Prevent zeros
    
    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute state posteriors gamma[t, i] = P(state_t = i | X)"""
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
    
    def _compute_xi(
        self,
        X: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """
        Compute transition posteriors xi[t, i, j] = P(state_t = i, state_{t+1} = j | X)
        """
        n_samples = X.shape[0]
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        
        for t in range(n_samples - 1):
            emission_probs = self._emission_probability(X[t + 1])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.transition_matrix[i, j] *
                        emission_probs[j] *
                        beta[t + 1, j]
                    )
            xi[t] /= xi[t].sum()
        
        return xi
    
    def _update_parameters(
        self,
        X: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ):
        """M-step: Update all model parameters"""
        n_samples = X.shape[0]
        
        # Update initial probabilities
        self.initial_probs = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_matrix[i, j] = xi[:, i, j].sum() / gamma[:-1, i].sum()
        
        # Update emission parameters
        for i in range(self.n_states):
            gamma_sum = gamma[:, i].sum()
            
            # Update mean
            self.emission_means[i] = np.sum(
                gamma[:, i:i+1] * X, axis=0
            ) / gamma_sum
            
            # Update covariance
            diff = X - self.emission_means[i]
            self.emission_covs[i] = (
                np.dot((gamma[:, i:i+1] * diff).T, diff) / gamma_sum
            )
            # Add small regularization
            self.emission_covs[i] += 1e-6 * np.eye(self.n_features)
    
    def _compute_log_likelihood(self, alpha: np.ndarray) -> float:
        """Compute log-likelihood of observations"""
        return np.log(alpha.sum(axis=1)).sum()
    
    def predict_sequence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict most likely state sequence using Viterbi algorithm
        
        Args:
            X: Observations, shape (n_samples, n_features)
            
        Returns:
            states: Most likely state sequence
            probs: State probabilities at each time step
        """
        n_samples = X.shape[0]

        logA = np.log(self.transition_matrix + 1e-300)
        logpi = np.log(self.initial_probs + 1e-300)

        log_delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)

        logB0 = np.log(self._emission_probability(X[0]) + 1e-300)
        log_delta[0] = logpi + logB0

        for t in range(1, n_samples):
            logBt = np.log(self._emission_probability(X[t]) + 1e-300)
            for j in range(self.n_states):
                scores = log_delta[t - 1] + logA[:, j]
                psi[t, j] = int(np.argmax(scores))
                log_delta[t, j] = float(np.max(scores) + logBt[j])

        states = np.zeros(n_samples, dtype=int)
        states[-1] = int(np.argmax(log_delta[-1]))
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        alpha, _ = self._forward(X)
        return states, alpha
    
    def predict_online(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Online inference: Update belief with new observation
        
        Args:
            observation: Single observation, shape (n_features,)
            
        Returns:
            Dictionary with state probabilities and predicted state
        """
        if self.current_belief is None:
            # Initialize with prior
            self.current_belief = self.initial_probs.copy()
        
        # Predict step: Apply transition
        predicted_belief = self.transition_matrix.T @ self.current_belief
        
        # Update step: Incorporate observation
        emission_probs = self._emission_probability(observation)
        self.current_belief = predicted_belief * emission_probs
        self.current_belief /= self.current_belief.sum()
        
        # Get predicted state
        predicted_state = np.argmax(self.current_belief)
        
        return {
            'state_probabilities': {
                self.state_names[i]: float(self.current_belief[i])
                for i in range(self.n_states)
            },
            'predicted_state': self.state_names[predicted_state],
            'confidence': float(self.current_belief[predicted_state])
        }
    
    def reset_belief(self):
        """Reset online inference to initial state"""
        self.current_belief = None
    
    def save(self, filepath: str):
        """Save model to disk"""
        params = HMMParameters(
            n_states=self.n_states,
            n_features=self.n_features,
            transition_matrix=self.transition_matrix,
            initial_probs=self.initial_probs,
            emission_means=self.emission_means,
            emission_covs=self.emission_covs,
            state_names=self.state_names,
            feature_names=self.feature_names
        )
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CognitiveStateHMM':
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        model = cls(
            n_states=params.n_states,
            state_names=params.state_names,
            feature_names=params.feature_names
        )
        model.transition_matrix = params.transition_matrix
        model.initial_probs = params.initial_probs
        model.emission_means = params.emission_means
        model.emission_covs = params.emission_covs
        
        return model
    
    def get_parameters_summary(self) -> Dict:
        """Get human-readable summary of model parameters"""
        return {
            'n_states': self.n_states,
            'state_names': self.state_names,
            'transition_matrix': self.transition_matrix.tolist(),
            'emission_parameters': {
                self.state_names[i]: {
                    'mean': self.emission_means[i].tolist(),
                    'covariance': self.emission_covs[i].tolist()
                }
                for i in range(self.n_states)
            }
        }


if __name__ == "__main__":
    # Quick test
    print("Testing HMM implementation...")
    
    # Generate some dummy data
    np.random.seed(42)
    X_test = np.random.randn(100, 2)
    
    # Create and train model
    model = CognitiveStateHMM()
    model.initialize_parameters(X_test)
    history = model.fit(X_test, n_iter=50, verbose=True)
    
    # Test prediction
    states, probs = model.predict_sequence(X_test[:10])
    print(f"\nPredicted states: {states}")
    
    # Test online inference
    result = model.predict_online(X_test[0])
    print(f"\nOnline inference result: {result}")
    
    print("\n HMM implementation working!")
