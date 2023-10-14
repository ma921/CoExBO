import torch
import gpytorch
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.analytic import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from ._piBO import PiUCB, ConstrainedPiUCB
from ._probabilistic_piBO import ProbabilisticPiUCB, ConstrainedProbabilisticPiUCB


class DuelingAcquisitionFunction:
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        method="robust",
        hallucinate=False,
        n_restarts=10,
        raw_samples=512,
    ):
        self.model = model
        self.prior_pref = prior_pref
        self.beta = beta
        self.gamma = gamma
        self.method = method # select from ["ts", "nonmyopic", "dueling", "robust"]
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.bounds = prior_pref.bounds
        self.hallucinate = hallucinate
    
    def cleansing_input(self, X):
        X = X.squeeze()
        if len(X.shape) == 0:
            X = X.unsqueeze(0)
        return X
    
    def optimize_function(self, acqf):
        X_next, _ = optimize_acqf(
            acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
        )
        return self.cleansing_input(X_next)
    
    def parallel_ts(self, n_cand=10000):
        X_cand = self.prior_pref.prior.sample(n_cand)
        with gpytorch.settings.max_cholesky_size(float("inf")), torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_suggest = thompson_sampling(X_cand, num_samples=2)
        return X_suggest
    
    def hallucination(self, X):
        X_fantasy = X.unsqueeze(0)
        Y_fantasy = self.cleansing_input(
            self.model.posterior(X_fantasy).sample(torch.Size([1]))
        )
        model_fantasy = self.model.get_fantasy_model(X_fantasy,Y_fantasy)
        mll = ExactMarginalLogLikelihood(model_fantasy.likelihood, model_fantasy)
        fit_gpytorch_model(mll)
        return model_fantasy
    
    def nonmyopic(self):
        ucb = ProbabilisticPiUCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_opt = self.optimize_function(ucb)
        
        # fantasize
        model_fantasy = self.hallucination(X_opt)
        ucb_fantasize = ProbabilisticPiUCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_pess = self.optimize_function(ucb_fantasize)
        X_suggest = torch.vstack([X_opt, X_pess])
        return X_suggest
    
    def nonmyopic_pi(self):
        ucb = ProbabilisticPiUCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=True)
        X_opt = self.optimize_function(ucb)
        
        # fantasize
        model_fantasy = self.hallucination(X_opt)
        ucb_fantasize = ProbabilisticPiUCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=True)
        X_pess = self.optimize_function(ucb_fantasize)
        X_suggest = torch.vstack([X_opt, X_pess])
        return X_suggest
    
    def dueling(self):
        piucb = ProbabilisticPiUCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_pess = self.optimize_function(piucb)
        if self.hallucinate:
            model_fantasy = self.hallucination(X_pess)
            piucb = ProbabilisticPiUCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=True)
        else:
            piucb.pi_augment = True
        X_opt = self.optimize_function(piucb)
        X_suggest = torch.vstack([X_opt, X_pess])
        return X_suggest
    
    def __call__(self):
        if self.method == "ts":
            X_suggest = self.parallel_ts()
        elif self.method == "nonmyopic":
            X_suggest = self.nonmyopic()
        elif self.method == "nonmyopic_pi":
            X_suggest = self.nonmyopic()
        elif self.method == "dueling":
            X_suggest = self.dueling()
        else:
            raise ValueError('The method should be from ["ts", "nonmyopic", "dueling", "robust"]')
        return X_suggest.view(1,-1)

class PiBODuelingAcquisitionFunction:
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        method="dueling",
        n_restarts=10,
        raw_samples=512,
    ):
        self.model = model
        self.prior_pref = prior_pref
        self.beta = beta
        self.gamma = gamma
        self.method = method # select from ["dueling", "robust"]
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.bounds = prior_pref.bounds

    def cleansing_input(self, X):
        X = X.squeeze()
        if len(X.shape) == 0:
            X = X.unsqueeze(0)
        return X
    
    def optimize_function(self, acqf):
        X_next, _ = optimize_acqf(
            acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
        )
        return self.cleansing_input(X_next)
    
    def hallucination(self, X):
        X_fantasy = X.unsqueeze(0)
        Y_fantasy = self.cleansing_input(
            self.model.posterior(X_fantasy).sample(torch.Size([1]))
        )
        model_fantasy = self.model.get_fantasy_model(X_fantasy,Y_fantasy)
        mll = ExactMarginalLogLikelihood(model_fantasy.likelihood, model_fantasy)
        fit_gpytorch_model(mll)
        return model_fantasy
    
    def dueling(self):
        piucb = PiUCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_pess = self.optimize_function(piucb)
        model_fantasy = self.hallucination(X_pess)
        piucb = ProbabilisticPiUCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=True)
        X_opt = self.optimize_function(piucb)
        X_suggest = torch.vstack([X_opt, X_pess])
        return X_suggest
    
    def __call__(self):
        if self.method == "dueling":
            X_suggest = self.dueling()
        elif self.method == "robust":
            X_suggest = self.robust()
        else:
            raise ValueError('The method should be from ["dueling", "robust"]')
        return X_suggest.view(1,-1)
    
class BaselineDuelingAcquisitionFunction:
    def __init__(
        self,
        model,
        domain,
        beta,
        bounds,
        method="ts",
        n_restarts=10,
        raw_samples=512,
    ):
        self.model = model
        self.domain = domain
        self.beta = beta
        self.method = method # select from ["ts", "nonmyopic"]
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.bounds = bounds
        
    def cleansing_input(self, X):
        X = X.squeeze()
        if len(X.shape) == 0:
            X = X.unsqueeze(0)
        return X
    
    def optimize_function(self, acqf):
        X_next, _ = optimize_acqf(
            acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
        )
        return self.cleansing_input(X_next)
    
    def parallel_ts(self, n_cand=10000):
        X_cand = self.domain.sample(n_cand)
        with gpytorch.settings.max_cholesky_size(float("inf")), torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_suggest = thompson_sampling(X_cand, num_samples=2)
        return X_suggest
    
    def hallucination(self, X):
        X_fantasy = X.unsqueeze(0)
        Y_fantasy = self.cleansing_input(
            self.model.posterior(X_fantasy).sample(torch.Size([1]))
        )
        model_fantasy = self.model.get_fantasy_model(X_fantasy,Y_fantasy)
        mll = ExactMarginalLogLikelihood(model_fantasy.likelihood, model_fantasy)
        fit_gpytorch_model(mll)
        return model_fantasy
    
    def nonmyopic(self):
        ucb = UpperConfidenceBound(self.model, self.beta)
        X_opt = self.optimize_function(ucb)
        
        # fantasize
        model_fantasy = self.hallucination(X_opt)
        ucb_fantasize = UpperConfidenceBound(model_fantasy, self.beta)
        X_pess = self.optimize_function(ucb_fantasize)
        X_suggest = torch.vstack([X_opt, X_pess])
        return X_suggest
    
    def __call__(self):
        if self.method == "ts":
            X_suggest = self.parallel_ts()
        elif self.method == "nonmyopic":
            X_suggest = self.nonmyopic()
        else:
            raise ValueError('The method should be from ["ts", "nonmyopic", "dueling", "robust"]')
        return X_suggest.view(1,-1)