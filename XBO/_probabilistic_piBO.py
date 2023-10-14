import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._utils import TensorManager

class ProbabilisticPiUCB(AnalyticAcquisitionFunction, TensorManager):
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
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))        
        self.pi_augment = pi_augment
        self.initialise(prior_pref, model)
        
    def initialise(self, prior_pref, model):
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
        self.E_y_obs = model.train_targets.mean()
        self.std_y_obs = model.train_targets.std()
        self.prior_pref = prior_pref
        
    def prior_gp(self, X):
        prior_mean, prior_std = self.prior_pref.probability(X, both=True)

        prior_mean_conv = (prior_mean - self.E_y_pref) / self.std_y_pref * self.std_y_obs + self.E_y_obs
        prior_std_conv = prior_std / self.std_y_pref * self.std_y_obs
        return prior_mean_conv, prior_std_conv
    
    def posterior_gp(self, X, likelihood_gp_mean, likelihood_gp_std):
        prior_gp_mean, prior_gp_std = self.prior_gp(X)
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean
        return posterior_gp_mean, posterior_gp_std
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        likelihood_gp_mean, likelihood_gp_std = self._mean_and_sigma(X)
        if self.pi_augment:
            posterior_gp_mean, posterior_gp_std = self.posterior_gp(X, likelihood_gp_mean, likelihood_gp_std)
            return posterior_gp_mean + self.beta.sqrt() * posterior_gp_std
        else:
            return likelihood_gp_mean + self.beta.sqrt() * likelihood_gp_std


class ConstrainedProbabilisticPiUCB(AnalyticAcquisitionFunction, TensorManager):
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
        self.initialise(prior_pref, model)
        
    def initialise(self, prior_pref, model):
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
        self.E_y_obs = model.train_targets.mean()
        self.std_y_obs = model.train_targets.std()
        self.prior_pref = prior_pref
        
    def prior_gp(self, X):
        prior_mean = self.prior_pref.probability(X, mean=True)
        prior_std = self.prior_pref.probability(X, mean=False)

        prior_mean_conv = (prior_mean - self.E_y_pref) / self.std_y_pref * self.std_y_obs + self.E_y_obs
        prior_std_conv = prior_std / self.std_y_pref * self.std_y_obs
        return prior_mean_conv, prior_std_conv
    
    def posterior_gp(self, X, likelihood_gp_mean, likelihood_gp_std):
        prior_gp_mean, prior_gp_std = self.prior_gp(X)
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean
        return posterior_gp_mean, posterior_gp_std
    
    def constraint(self, X):
        n_X, _, n_dims = X.shape
        X_centred = X - self.centre.repeat(n_X, 1, 1)
        X_within_epsilon = self.standardise_tensor(
            X_centred.pow(2).sum(axis=2).squeeze() <= self.radius.pow(2)
        ).squeeze()
        return X_within_epsilon
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        likelihood_gp_mean, likelihood_gp_std = self._mean_and_sigma(X)
        posterior_gp_mean, posterior_gp_std = self.posterior_gp(X, likelihood_gp_mean, likelihood_gp_std)
        piucb = posterior_gp_mean + self.beta.sqrt() * posterior_gp_std
        X_within_epsilon = self.constraint(X)
        return piucb * X_within_epsilon
