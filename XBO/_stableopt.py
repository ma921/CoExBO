import torch
import torch.distributions as D
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf
from ._utils import TensorManager
tm = TensorManager()

def sampling_from_hypersphere(center, radius, n_samples):
    # cited from https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    # https://baezortega.github.io/2018/10/14/hypersphere-sampling/
    ndim = center.shape[0]
    
    # sample n_samples points in d dimensions from a standard normal distribution
    samples = D.MultivariateNormal(
        tm.zeros(ndim),
        tm.ones(ndim).diag(),
    ).rsample(torch.Size([n_samples]))

    # make the samples lie on the surface of the unit hypersphere
    normalize_radii = torch.linalg.norm(samples, axis=1).unsqueeze(-1)
    samples /= normalize_radii

    # make the samples lie inside the hypersphere with the correct density
    uniform_points = D.Uniform(
        tm.zeros(1),
        tm.ones(1),
    ).rsample(torch.Size([n_samples]))
    new_radii = uniform_points.pow(1/ndim)
    samples *= new_radii

    # scale the points to have the correct radius and center
    samples = samples * radius + center
    return samples

class AdversariallyRobustUCB(UpperConfidenceBound):
    def __init__(
            self,
            radius,
            n_samples,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.n_samples = n_samples
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if len(X.shape) > 1:
            arucb = tm.null()
            for x in X:
                X_perturb = sampling_from_hypersphere(x, self.radius, self.n_samples)
                ucb_perturb = super().forward(X_perturb.unsqueeze(1))
                ucb_min = ucb_perturb.min()
                arucb = torch.cat([arucb, ucb_min.unsqueeze(0)], dim=0)
        else:
            X_perturb = sampling_from_hypersphere(X, self.radius, self.n_samples)
            ucb_perturb = super().forward(X_perturb.unsqueeze(1))
            arucb = ucb_perturb.min()
        return arucb

class AdversariallyRobustLCB:
    def __init__(self, model, beta, radius):
        self.model = model
        self.beta = tm.tensor(beta).sqrt()
        self.radius = radius
        
    def calc_lcb(self, X):
        with torch.no_grad():
            self.model.eval()
            self.model.likelihood.eval()
            pred = self.model.likelihood(self.model(X))
            lcb = pred.mean - self.beta * pred.variance
        return lcb
    
    def __call__(self, X_cand, return_positive=True):
        euclid_distance_matrix = torch.cdist(X_cand, X_cand)
        lcb = self.calc_lcb(X_cand)
        
        if return_positive:
            lcb_positive = lcb.max() - lcb
            return lcb_positive
        else:
            return lcb


class PriorAugmentedARUCB(UpperConfidenceBound):
    def __init__(
            self,
            radius,
            n_samples,
            prior,
            decay_beta,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.n_samples = n_samples
        self.prior = prior
        self.decay_beta = decay_beta
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if len(X.shape) > 1:
            arucb = tm.null()
            for x in X:
                X_perturb = sampling_from_hypersphere(x, self.radius, self.n_samples)
                ucb_perturb = super().forward(X_perturb.unsqueeze(1))
                pi = self.prior.pdf(X_perturb.squeeze()).pow(self.decay_beta)
                ucb_perturb = ucb_perturb * pi
                ucb_min = ucb_perturb.min()
                arucb = torch.cat([arucb, ucb_min.unsqueeze(0)], dim=0)
        else:
            X_perturb = sampling_from_hypersphere(X, self.radius, self.n_samples)
            ucb_perturb = super().forward(X_perturb.unsqueeze(1))
            pi = self.prior.pdf(X_perturb.squeeze()).pow(self.decay_beta)
            ucb_perturb = ucb_perturb * pi
            arucb = ucb_perturb.min()
        return arucb

class PriorAugmentedARLCB:
    def __init__(self, model, beta, radius, prior, decay_beta):
        self.model = model
        self.beta = tm.tensor(beta).sqrt()
        self.radius = radius
        self.prior = prior
        self.decay_beta = decay_beta
        
    def calc_lcb(self, X):
        with torch.no_grad():
            self.model.eval()
            self.model.likelihood.eval()
            pred = self.model.likelihood(self.model(X))
            lcb = pred.mean - self.beta * pred.variance
        return lcb
    
    def __call__(self, X_cand, return_positive=True):
        euclid_distance_matrix = torch.cdist(X_cand, X_cand)
        lcb = self.calc_lcb(X_cand)
        pi = self.prior.pdf(X_cand).pow(self.decay_beta)
        lcb_positive = (lcb.max() - lcb) * pi
        return lcb_positive
