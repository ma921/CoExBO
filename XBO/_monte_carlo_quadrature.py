import torch
import joblib
import pandas as pd
from ._prior import Uniform
from ._weights import WeightsStabiliser
from ._utils import TensorManager
from ._gp_classifier import preference_probability


class MonteCarloQuadrature(WeightsStabiliser, TensorManager):
    def __init__(self, model_pref, prior, n_mc=100, thresh=1000):
        WeightsStabiliser.__init__(self)
        TensorManager.__init__(self)
        self.model_pref = model_pref
        self.prior = prior
        self.prior_duel = Uniform(prior.bounds.repeat(1,2))
        self.bounds = prior.bounds
        self.n_dims = prior.bounds.shape[1]
        self.n_mc = n_mc
        self.X_mc = prior.sample(n_mc)
        self.thresh = thresh
        
    def soft_copeland_score(self, X):
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
        if len(X.shape) == 1:
            if len(X) == self.n_dims:
                X = X.unsqueeze(0)
            else:
                X = X.unsqueeze(-1)
        return X
    
    def probability(self, X, mean=True, both=False):
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
        self.y_test_mean = self.probability(self.X_mc, mean=True)
        self.norm_const = self.y_test_mean.mean().item()
        
    def pdf(self, X):
        if not hasattr(self, "norm_const"):
            self.normalising_constant()
        
        Y_mean = self.probability(X, mean=True)
        return Y_mean / self.norm_const
    
    def sample(self, n_samples, n_super=10000):
        X_super = self.prior.sample(n_super)
        w_super = self.cleansing_weights(self.probability(X_super, mean=True))
        idx = self.weighted_resampling(w_super, n_samples)
        return X_super[idx]
    
    def explain(self, n_super=20000, corr=False):
        X_super = self.prior.sample(n_super)
        w_super = self.cleansing_weights(self.probability(X_super, mean=True))
        mean = w_super @ X_super
        covariance = 1 / (1 - w_super.pow(2).sum()) * (w_super * (X_super - mean).T @ (X_super - mean))
        std = covariance.diag().sqrt()
        #correlation = covariance / (std.unsqueeze(-1) @ std.unsqueeze(0))
        mode = X_super[w_super.argmax()]

        summary = pd.DataFrame(
            torch.vstack([mean, std, mode]),
            index=["mean", "std", "mode"],
            columns=["dim"+str(i) for i in range(len(mean))],
        )
        display(summary)
