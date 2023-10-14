import torch
import numpy as np
import pandas as pd
import gpytorch
from qpsolvers import solve_qp
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from ._utils import TensorManager, SafeTensorOperator
from ._prior import Uniform
from ._gp_classifier import preference_probability
from ._weights import WeightsStabiliser
from ._rchq import recombination
from ._kernel import Kernel

def set_rbf_model(X, Y):
    covar_module = ScaleKernel(RBFKernel())

    # Fit a GP model
    train_Y = Y
    train_Y = train_Y.view(-1).unsqueeze(1)
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    model = SingleTaskGP(
        X, 
        train_Y, 
        likelihood=likelihood, 
        covar_module=covar_module,
    )
    return model

def fit_model(model):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def set_and_fit_model(X_pairwise, y_probs):
    model = set_rbf_model(X_pairwise, y_probs)
    model = fit_model(model)
    return model

def remove_duplicates(A, B):
    # assert len(A) == len(B)
    n = len(A)
    cap = (A.repeat(len(B), 1) == B.repeat(len(B), 1).T).any(axis=1)
    return B[torch.logical_not(cap)]

class Transform(SafeTensorOperator):
    def __init__(self):
        SafeTensorOperator.__init__(self)
        
    def get_minmax_scaling(self, X):
        X = self.standardise_tensor(X)
        mins = X.min(axis=0).values
        maxs = X.max(axis=0).values
        return X, mins, maxs
    
    def get_gaussian_scaling(self, X):
        X = self.standardise_tensor(X)
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        return X, means, stds
    
    def normalize(self, X, mins, maxs):
        X = self.standardise_tensor(X)
        return (X - mins) / (maxs - mins)
    
    def unnormalize(self, X, mins, maxs):
        X = self.standardise_tensor(X)
        return (maxs - mins) * X + mins
    
    def standardize(self, X, means, stds):
        return (X - means) / stds
    
    def unstandardize(self, X, means, stds):
        return stds * X + means

class OptimizeAffine(TensorManager):
    def __init__(self, x_test, y_test, y_est, n_max=5):
        TensorManager.__init__(self)
        self.x_test = x_test
        self.y_test = y_test
        self.y_est = y_est
        self.n_max = n_max
        
    def objective(self, p):
        return (p[0] + p[1] * self.y_est - self.y_test).pow(2).sum()
    
    def closure(self):
        self.lbfgs.zero_grad()
        objective = self.objective(self.x_lbfgs)
        objective.backward()
        return objective
    
    def run(self):
        self.x_lbfgs = self.ones(2) * 0.5
        self.x_lbfgs.requires_grad = True
        
        self.lbfgs = torch.optim.LBFGS(
            [self.x_lbfgs],
            history_size=10, 
            max_iter=4, 
            line_search_fn="strong_wolfe",
        )
        
        for i in range(self.n_max):
            self.lbfgs.step(self.closure)
        return self.x_lbfgs.detach()
    
class BayesianQuadrature(WeightsStabiliser, Transform, OptimizeAffine):
    def __init__(self, model_pref, prior, n_rec=20000, n_mc=100, n_nys=200, n_augment=100):
        WeightsStabiliser.__init__(self)
        Transform.__init__(self)
        self.model_pref = model_pref
        self.prior = prior
        self.prior_duel = Uniform(prior.bounds.repeat(1,2))
        self.bounds = prior.bounds
        self.n_rec = n_rec
        model_mean, model_std = self.distill_model(model_pref, n_nys=n_nys, n_augment=n_augment)
        self.x_test, self.y_test_mean, self.y_test_std = self.generate_test_dataset(n_mc=n_mc)
        self.initialise(model_mean, model_std)
        
    def distill_model(self, model_pref, n_nys=200, n_augment=100):
        # 1. initial run
        X_pairwise = self.standardise_tensor(model_pref.train_inputs[0])
        y_probs_mean, y_probs_std = preference_probability(model_pref, X_pairwise, std=True)
        model_mean = set_and_fit_model(X_pairwise, self.standardise_tensor(y_probs_mean))
        model_std = set_and_fit_model(X_pairwise,  self.standardise_tensor(y_probs_std))
        
        # 2. data augmentaion
        X_aug = self.data_augmentation(model_pref, model_mean, model_std, n_nys=n_nys, n_augment=n_augment)
        y_aug_mean, y_aug_std = preference_probability(model_pref, X_aug, std=True)
        
        # 3. re-train models
        X_pairwise = torch.cat([X_pairwise, X_aug], dim=0)
        y_mean = torch.cat([y_probs_mean,  self.standardise_tensor(y_aug_mean)], dim=0)
        y_std = torch.cat([y_probs_std,  self.standardise_tensor(y_aug_std)], dim=0)
        model_mean = set_and_fit_model(X_pairwise, y_mean)
        model_std = set_and_fit_model(X_pairwise, y_std)
        return model_mean, model_std
    
    def data_augmentation(self, model_pref, model_mean, model_std, n_nys=200, n_augment=100):
        X_cand = self.prior_duel.sample(self.n_rec)
        X_nys = self.kmeans_resampling(X_cand, n_clusters=n_nys)
        
        w_cand_mean, w_cand_std = preference_probability(model_pref, X_cand, std=True)
        
        # 1. mean data augmentaion
        idx_mean, _ = recombination(
            X_cand,
            X_nys,
            n_augment,
            Kernel(model_mean),
            self.device,
            self.dtype,
            init_weights=self.cleansing_weights(w_cand_mean),
        )
        X_aug_mean = X_cand[idx_mean]
        
        # 2. variance data augmentation
        idx_std, _ = recombination(
            X_cand,
            X_nys,
            n_augment,
            Kernel(model_std),
            self.device,
            self.dtype,
            init_weights=self.cleansing_weights(w_cand_std),
        )
        idx_std = remove_duplicates(idx_mean, idx_std)
        X_aug_std = X_cand[idx_std]
        X_aug = torch.cat([X_aug_mean, X_aug_std], dim=0)
        return X_aug
    
    def generate_test_dataset(self, n_mc=100):
        x_test = self.prior.sample(n_mc)
        X_test = torch.vstack([
            torch.hstack([
                X.repeat(n_mc, 1), x_test
            ]) for X in x_test
        ])
        y_probs_mean, y_probs_std = preference_probability(
            self.model_pref,
            X_test,
            std=True,
        )
        y_test_mean = y_probs_mean.reshape(n_mc, n_mc).mean(axis=1)
        y_test_std = y_probs_std.reshape(n_mc, n_mc).mean(axis=1)
        return x_test, y_test_mean, y_test_std
    
    def compute_constants(self, model):
        mean_const = model.mean_module.constant.item()
        outputscale = model.covar_module.outputscale.item()
        lengthscale = model.covar_module.base_kernel.lengthscale.item()

        W = self.tensor(lengthscale).pow(2).repeat(int(self.n_dims * 2)).diag()
        v = outputscale * self.tensor(2 * torch.pi * W).det().sqrt()
        W = W[:self.n_dims, :self.n_dims]

        woodbury_factor = self.get_cache(model)
        weights = v * woodbury_factor
        norm_const = weights.sum() + mean_const
        return mean_const, W, weights, norm_const
    
    def compute_affine(self, n_max=5):
        y_est_mean = self.pdf_unnormalized(self.x_test, mean=True)
        OptimizeAffine.__init__(self, self.x_test, self.y_test_mean, y_est_mean, n_max=n_max)
        self.p_mean = self.run()
        y_est_std = self.pdf_unnormalized(self.x_test, mean=False)
        OptimizeAffine.__init__(self, self.x_test, self.y_test_std, y_est_std, n_max=n_max)
        self.p_std = self.run()
        
    def initialise(self, model_mean, model_std, n_max=5):
        Xobs = self.standardise_tensor(model_mean.train_inputs[0])
        self.n_obs, n_dims = Xobs.shape
        self.n_dims = int(n_dims / 2)
        
        if self.n_dims == 1:
            self.Xobs = Xobs[:,:self.n_dims].view(-1,1)
        else:
            self.Xobs = Xobs[:,:self.n_dims]

        self.mean_const_mean, self.W_mean, self.weights_mean, norm_const_mean = self.compute_constants(model_mean)
        self.mean_const_std, self.W_std, self.weights_std, norm_const_std = self.compute_constants(model_std)
        self.compute_affine(n_max=n_max)
        self.norm_const_mean = self.transform_to_bernoulli(self.p_mean, norm_const_mean, max=1)
        self.norm_const_std = self.transform_to_bernoulli(self.p_std, norm_const_std, max=0.25)
        
    def get_cache(self, model):
        try:
            woodbury_factor = model.prediction_strategy.mean_cache
        except:
            model.eval()
            mean = model.train_inputs[0].unsqueeze(0)
            model(mean)
            woodbury_factor = model.prediction_strategy.mean_cache
        return woodbury_factor
    
    def pdf_unnormalized(self, X, mean=True):
        if len(X.shape) == 1:
            X = X.view(-1,1)
        
        n_X, n_dims = X.shape
        
        x_AA = (
            self.Xobs.repeat(n_X, 1, 1) - X.unsqueeze(1)
        ).reshape(int(self.n_obs * n_X), self.n_dims)

        if mean:
            Npdfs_AA = self.safe_mvn_prob(
                self.zeros(self.n_dims),
                self.W_mean,
                x_AA,
            ).reshape(n_X, self.n_obs)
            soft_copeland_score_norm = Npdfs_AA @ self.weights_mean + self.mean_const_mean
            return soft_copeland_score_norm
        else:
            Npdfs_AA = self.safe_mvn_prob(
                self.zeros(self.n_dims),
                self.W_std,
                x_AA,
            ).reshape(n_X, self.n_obs)
            soft_copeland_score_norm = Npdfs_AA @ self.weights_std + self.mean_const_std
            return soft_copeland_score_norm
        
    def transform_to_bernoulli(self, p, y, max=1):
        return torch.clamp(p[0] + p[1] * y, min=0, max=max)
    
    def probability(self, X, mean=True):
        y = self.pdf_unnormalized(X, mean=mean)
        if mean:
            return self.transform_to_bernoulli(self.p_mean, y, max=1)
        else:
            return self.transform_to_bernoulli(self.p_std, y, max=0.25)
        
    def pdf(self, X):
        y = self.probability(X, mean=True)
        return y / self.norm_const_mean
    
    def sample(self, n_samples, n_super=10000):
        X_super = self.prior.sample(n_super)
        w_super = self.cleansing_weights(self.pdf(X_super, mean=True))
        idx = self.weighted_resampling(w_super, n_samples)
        return X_super[idx]
    
    def explain(self, n_super=20000, corr=False):
        X_super = self.prior.sample(n_super)
        w_super = self.cleansing_weights(self.pdf(X_super))
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
