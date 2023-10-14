import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._utils import TensorManager


class PiUCB(AnalyticAcquisitionFunction, TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        pi_augment=True,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.prior_pref = prior_pref
        self.pi_augment = pi_augment
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))
        
    def prior_gp(self, X):
        prior_mean = self.prior_pref.pdf(X)
        return prior_mean
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta.sqrt() * sigma
        if self.pi_augment:
            prior_mean = self.prior_gp(X)
            ucb = ucb * prior_mean.pow(self.gamma)
        return ucb


class ConstrainedPiUCB(AnalyticAcquisitionFunction, TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        radius,
        centre,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))
        self.register_buffer("radius", self.tensor(radius))
        self.centre = centre
        self.prior_pref = prior_pref
        
    def prior_gp(self, X):
        prior_mean = self.prior_pref.pdf(X)
        return prior_mean
    
    def constraint(self, X):
        n_X, _, n_dims = X.shape
        X_centred = X - self.centre.repeat(n_X, 1, 1)
        X_within_epsilon = self.standardise_tensor(
            X_centred.pow(2).sum(axis=2).squeeze() <= self.radius.pow(2)
        ).squeeze()
        return X_within_epsilon
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta.sqrt() * sigma
        prior_mean = self.prior_gp(X)
        X_within_epsilon = self.constraint(X)
        cpiucb = ucb * prior_mean.pow(self.gamma) * X_within_epsilon
        return cpiucb