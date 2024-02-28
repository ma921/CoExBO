import torch
import joblib
import pandas as pd
from ._prior import Uniform
from ._utils import TensorManager
from ._gp_classifier import preference_probability


class MonteCarloQuadrature(TensorManager):
    def __init__(self, model_pref, prior, n_mc=100, thresh=1000):
        """
        A class for Monte Carlo Quadrature.
        
        Args:
        - model_pref: gpytorch.models, the Dirichlet GP model for binary classification
        - prior: CoExBO._prior.BasePrior, the prior distribution over the domain.
        - n_mc: int, number of i.i.d. samples for Monte Carlo integration.
        - thresh: int, the threshold number of samples to decide whether we compute in paralell CPU cores via joblib.
        """
        TensorManager.__init__(self)
        self.model_pref = model_pref
        self.prior = prior
        self.prior_duel = Uniform(prior.bounds.repeat(1,2))
        self.bounds = prior.bounds
        self.n_dims = prior.bounds.shape[1]
        self.n_mc = n_mc
        self.X_mc = prior.sample(n_mc)
        self.thresh = thresh
        self.eps_weights = torch.finfo().eps
        
    def cleansing_weights(self, weights):
        """
        Remove anomalies from the computed weights
        
        Args:
        - weights: torch.tensor, weights
        
        Return:
        - weights: torch.tensor, the cleaned weights
        """
        weights[weights < self.eps_weights] = 0
        weights[weights.isinf()] = self.eps_weights
        weights[weights.isnan()] = self.eps_weights
        if not weights.sum() == 0:
            weights /= weights.sum()
        else:
            weights = torch.ones_like(weights)/len(weights)
        return weights.detach()
        
    def soft_copeland_score(self, X):
        """
        Compute the unnormalised soft Copeland score
        
        Args:
        - X: torch.tensor, an input
        
        Return:
        - Y_mean: torch.tensor, the unnormalised mean of soft Copeland score
        - Y_std: torch.tensor, the unnormalised standard deviation of soft Copeland score
        """
        n_X = len(X)
        X_pairwise = torch.vstack([
            torch.hstack([
                x.repeat(self.n_mc, 1), self.X_mc
            ]) for x in X
        ])
        y_probs_mean, y_probs_std = preference_probability(
            self.model_pref,
            X_pairwise,
            std=True,
        )
        Y_mean = y_probs_mean.reshape(n_X, self.n_mc).mean(axis=1)
        Y_std = y_probs_std.reshape(n_X, self.n_mc).mean(axis=1)
        return Y_mean, Y_std
    
    def check_input(self, X):
        """
        Check the input if it matches the desired format, otherwise standardise it.
        
        Args:
        - X: torch.tensor, an input before check/transform
        
        Return:
         - X: torch.tensor, an input after check/transform
        """
        if len(X.shape) == 1:
            if len(X) == self.n_dims:
                X = X.unsqueeze(0)
            else:
                X = X.unsqueeze(-1)
        return X
    
    def probability(self, X, mean=True, both=False):
        """
        Compute the unnormalised soft Copeland score.
        This function is the extended version of soft_copeland_score for flexibility.
        The name "probability" is misleading as this is unnormalised.
        
        Args:
        - X: torch.tensor, an input
        - mean: bool, return only mean soft Copeland score if true, otherwise return stddev.
        - both: bool, return only mean soft Copeland score if true, otherwise return both mean and stddev.
        
        Return:
        - Y_mean: torch.tensor, the unnormalised mean of soft Copeland score
        - Y_std: torch.tensor, the unnormalised standard deviation of soft Copeland score
        """
        X = self.check_input(X)
        if len(X) <= self.thresh:
            Y_mean, Y_std = self.soft_copeland_score(X)
        else:
            n_chunks = int(len(X) / self.thresh)
            Xs = torch.chunk(X, chunks=n_chunks, dim=0)
            result = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self.soft_copeland_score)(X) for X in Xs
            )
            Y_mean = torch.hstack([result[i][0] for i in range(n_chunks)])
            Y_std = torch.hstack([result[i][1] for i in range(n_chunks)])
        if both:
            return self.standardise_tensor(Y_mean), self.standardise_tensor(Y_std)
        else:
            if mean:
                return self.standardise_tensor(Y_mean)
            else:
                return self.standardise_tensor(Y_std)
        
    def normalising_constant(self):
        """
        Compute the normalising constant of soft-Copeland score to be proper probability.
        Yet, this is very time-consuming and should avoid as far as possible.
        """
        self.y_test_mean = self.probability(self.X_mc, mean=True)
        self.norm_const = self.y_test_mean.mean().item()
        
    def pdf(self, X):
        """
        Normalised soft-Copeland score.
        Yet, this is very time-consuming and should avoid as far as possible.
        
        Return:
        - Y_mean: torch.tensor, the normalised mean of soft Copeland score
        """
        if not hasattr(self, "norm_const"):
            self.normalising_constant()
        
        Y_mean = self.probability(X, mean=True)
        return Y_mean / self.norm_const
    
    def sample(self, n_samples, n_super=10000):
        """
        Sampling from the normalised soft-Copeland score using Sampling-importance Resampling.
        This is approximate sampling scheme with errors according to the number of supersamples.
        
        Args:
        - n_samples: int, number of samples
        - n_super: int, number of supersamples
        
        Return:
        - X_samples: torch.tensor, the (approximate) samples from the normalised soft Copeland score
        """
        X_super = self.prior.sample(n_super)
        w_super = self.cleansing_weights(self.probability(X_super, mean=True))
        idx = self.weighted_resampling(w_super, n_samples)
        return X_super[idx]
    