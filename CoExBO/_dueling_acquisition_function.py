import torch
import gpytorch
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.analytic import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from ._acquisition_function import CoExBO_UCB, PiBO_UCB


class BaseDuelingAcquisitionFunction:
    def __init__(
        self,
        model,
        bounds,
        n_restarts=10,
        raw_samples=512,
    ):
        """
        The base class for dueling acquisition function.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - bounds: torch.tensor, the bounds of the domain
        - n_restarts: int, number of restarts for acquisition function optimization for querying.
        - raw_samples: int, numner of initial samples for acquisition function optimization for querying.
        """
        self.model = model
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.bounds = bounds
        
    def cleansing_input(self, X):
        """
        Clensing the input to satisfy the desired format.
        
        Args:
        - X: torch.tensor, inputs before cleansing
        
        Return:
        - X: torch.tensor, inputs after cleansing
        """
        X = X.squeeze()
        if len(X.shape) == 0:
            X = X.unsqueeze(0)
        return X
    
    def optimize_function(self, acqf):
        """
        Optimising the acquisition funtion to find the next query.
        
        Args:
        - acqf: botorch.acquisition.AnalyticAcquisitionFunction, the acquisition function
        
        Return:
        - X_next: torch.tensor, the next query point
        """
        X_next, _ = optimize_acqf(
            acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
        )
        return self.cleansing_input(X_next)
    
    def hallucination(self, X):
        """
        Hallucinating the Gaussian process model
        
        Args:
        - X: int, the fantasy input to hallucinate
        
        Return:
        - model_fantasy: botorch.models.gp_regression.SingleTaskGP, the hallucinated GP model
        """
        X_fantasy = X.unsqueeze(0)
        Y_fantasy = self.cleansing_input(
            self.model.posterior(X_fantasy).sample(torch.Size([1]))
        )
        model_fantasy = self.model.get_fantasy_model(X_fantasy,Y_fantasy)
        mll = ExactMarginalLogLikelihood(model_fantasy.likelihood, model_fantasy)
        fit_gpytorch_model(mll)
        return model_fantasy

class DuelingAcquisitionFunction(BaseDuelingAcquisitionFunction):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        method="dueling",
        hallucinate=False,
        n_restarts=10,
        raw_samples=512,
    ):
        """
        Class for dueling acquisition function.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - gamma: float, decay hyperparameter in Eq.(7)
        - method: string, the acquisiton function. select from ["ts", "nonmyopic", "dueling"]. "dueling" is CoExBO AF.
        - hallucinate: bool, whether or not we condition the GP on the normal BO query point.
        - n_restarts: int, number of restarts for acquisition function optimization for querying.
        - raw_samples: int, numner of initial samples for acquisition function optimization for querying.
        """
        BaseDuelingAcquisitionFunction.__init__(self, model, prior_pref.bounds, n_restarts, raw_samples)
        self.prior_pref = prior_pref
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.hallucinate = hallucinate
    
    def parallel_ts(self, n_cand=10000):
        """
        The next pairwise query points by Thompson sampling
        
        Args:
        - n_cand: int, the number of candidates for Thompson sampling
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points by Thompson sampling
        """
        X_cand = self.prior_pref.prior.sample(n_cand)
        with gpytorch.settings.max_cholesky_size(float("inf")), torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_suggest = thompson_sampling(X_cand, num_samples=2)
        return X_suggest
    
    def nonmyopic(self):
        """
        Nonmyopic acquisition function by hallucination.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points by hallucination
        """
        ucb = CoExBO_UCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_ucb = self.optimize_function(ucb)
        
        # fantasize
        model_fantasy = self.hallucination(X_ucb)
        ucb_fantasize = CoExBO_UCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_hal = self.optimize_function(ucb_fantasize)
        X_suggest = torch.vstack([X_ucb, X_hal])
        return X_suggest
    
    def dueling(self):
        """
        CoExBO acquisition function.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points by CoExBO acquisition function
        """
        piucb = CoExBO_UCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_ucb = self.optimize_function(piucb)
        if self.hallucinate:
            # if selected, we hallucinate the GP on X_bo
            model_fantasy = self.hallucination(X_ucb)
            piucb = CoExBO_UCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=True)
        else:
            # otherwise not.
            piucb.pi_augment = True
        X_pref = self.optimize_function(piucb)
        X_suggest = torch.vstack([X_pref, X_ucb])
        return X_suggest
    
    def __call__(self):
        """
        Generate the pairwise candidate based on the selected acquisition function.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points
        """
        if self.method == "ts":
            X_suggest = self.parallel_ts()
        elif self.method == "nonmyopic":
            X_suggest = self.nonmyopic()
        elif self.method == "dueling":
            X_suggest = self.dueling()
        else:
            raise ValueError('The method should be from ["ts", "nonmyopic", "dueling"]')
        return X_suggest.view(1,-1)


class PiBODuelingAcquisitionFunction(BaseDuelingAcquisitionFunction):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        n_restarts=10,
        raw_samples=512,
    ):
        """
        Class for piBO dueling acquisition function.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - gamma: float, decay hyperparameter in Eq.(7)
        - n_restarts: int, number of restarts for acquisition function optimization for querying.
        - raw_samples: int, numner of initial samples for acquisition function optimization for querying.
        """
        BaseDuelingAcquisitionFunction.__init__(self, model, prior_pref.bounds, n_restarts, raw_samples)
        self.prior_pref = prior_pref
        self.beta = beta
        self.gamma = gamma
        self.method = "dueling"
    
    def dueling(self):
        """
        CoExBO (piBO) acquisition function.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points by CoExBO acquisition function
        """
        piucb = PiBO_UCB(self.model, self.prior_pref, self.beta, self.gamma, pi_augment=False)
        X_ucb = self.optimize_function(piucb)
        model_fantasy = self.hallucination(X_ucb)
        piucb = CoExBO_UCB(model_fantasy, self.prior_pref, self.beta, self.gamma, pi_augment=True)
        X_pi = self.optimize_function(piucb)
        X_suggest = torch.vstack([X_pi, X_ucb])
        return X_suggest
    
    def __call__(self):
        """
        Generate the pairwise candidate based on the piBO acquisition function.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points
        """
        if self.method == "dueling":
            X_suggest = self.dueling()
        else:
            raise ValueError('The method should be "dueling"')
        return X_suggest.view(1,-1)
    
class BaselineDuelingAcquisitionFunction(BaseDuelingAcquisitionFunction):
    def __init__(
        self,
        model,
        domain,
        beta,
        method="ts",
        n_restarts=10,
        raw_samples=512,
    ):
        """
        Class for baseline dueling acquisition function.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - domain: CoExBO._prior.BasePrior, the prior distribution over the domain.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - method: string, the acquisiton function. select from ["ts", "nonmyopic"].
        - n_restarts: int, number of restarts for acquisition function optimization for querying.
        - raw_samples: int, numner of initial samples for acquisition function optimization for querying.
        """
        BaseDuelingAcquisitionFunction.__init__(self, model, domain.bounds, n_restarts, raw_samples)
        self.domain = domain
        self.beta = beta
        self.method = method # select from ["ts", "nonmyopic"]
        
    def parallel_ts(self, n_cand=10000):
        """
        The next pairwise query points by Thompson sampling
        
        Args:
        - n_cand: int, the number of candidates for Thompson sampling
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points by Thompson sampling
        """
        X_cand = self.domain.sample(n_cand)
        with gpytorch.settings.max_cholesky_size(float("inf")), torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_suggest = thompson_sampling(X_cand, num_samples=2)
        return X_suggest
    
    def nonmyopic(self):
        """
        Nonmyopic acquisition function by hallucination.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points by hallucination
        """
        ucb = UpperConfidenceBound(self.model, self.beta)
        X_opt = self.optimize_function(ucb)
        
        # fantasize
        model_fantasy = self.hallucination(X_opt)
        ucb_fantasize = UpperConfidenceBound(model_fantasy, self.beta)
        X_pess = self.optimize_function(ucb_fantasize)
        X_suggest = torch.vstack([X_opt, X_pess])
        return X_suggest
    
    def __call__(self):
        """
        Generate the pairwise candidate based on the selected acquisition function.
        
        Return:
        - X_suggest: torch.tensor, the next pairwise query points
        """
        if self.method == "ts":
            X_suggest = self.parallel_ts()
        elif self.method == "nonmyopic":
            X_suggest = self.nonmyopic()
        else:
            raise ValueError('The method should be from ["ts", "nonmyopic"]')
        return X_suggest.view(1,-1)
