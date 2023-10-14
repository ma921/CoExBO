import time
import torch
import torch.distributions as D
from ._utils import TensorManager
from ._duel import DuelFeedback
from ._gp_regressor import set_and_fit_rbf_model
from ._monte_carlo_quadrature import MonteCarloQuadrature
from ._bayesian_quadrature import BayesianQuadrature
from ._dueling_acquisition_function import DuelingAcquisitionFunction, PiBODuelingAcquisitionFunction, BaselineDuelingAcquisitionFunction
from ._human_interface import HumanFeedback
from ._gp_classifier import gp_sample
from ._adaptive_gp import initial_set_agp_model, online_update_model


class AdaptiveXBO:
    def __init__(
        self, 
        domain, 
        true_function, 
        sigma=0,
        training_iter=200,
        n_mc_quadrature=100,
        n_restarts=10,
        raw_samples=512,
        quadrature_method="mc",
        acqf_method="robust",
        probabilistic_pi=True,
        hallucinate=True,
        adversarial=False,
    ):
        self.domain = domain
        self.true_function = true_function
        self.duel = DuelFeedback(domain, true_function)
        self.sigma = sigma # noise level of human feedback
        self.training_iter = training_iter
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.quadrature_method = quadrature_method # select from ["mc", "bq"]
        self.n_mc_quadrature = n_mc_quadrature
        self.learn_preference = acqf_method in ["dueling", "robust"]
        self.probabilistic_pi = probabilistic_pi
        self.hallucinate = hallucinate
        self.acqf_method = acqf_method
        self.adversarial = adversarial
        
    def initial_sampling(self, n_init_obj, n_init_pref):
        X = self.domain.sample(n_init_obj)
        Y = self.true_function(X.squeeze())
        X_pairwise, y_pairwise, y_pairwise_unsure = self.duel.sample_both(n_init_pref, sigma=self.sigma, in_loop=False)
        if self.adversarial:
            y_pairwise = 1 - y_pairwise
        dataset_obj = (X, Y)
        dataset_duel = (X_pairwise, y_pairwise, y_pairwise_unsure)
        return dataset_obj, dataset_duel
    
    def set_models(self, X, Y, dataset_duel, window_size=20, initial=False):
        model = set_and_fit_rbf_model(X, Y)
        if self.learn_preference:
            if initial:
                self.model_pref = initial_set_agp_model(
                    dataset_duel, 
                    self.duel.prior_duel,
                    window_size=window_size,
                    training_iter=self.training_iter,
                )
            else:
                self.model_pref = online_update_model(
                    self.model_pref,
                    dataset_duel,
                    self.duel.prior_duel,
                    window_size=window_size,
                    training_iter=self.training_iter,
                )
                
            if self.quadrature_method == "bq":
                prior_pref = BayesianQuadrature(self.model_pref, self.domain)
            else:
                prior_pref = MonteCarloQuadrature(
                    self.model_pref, 
                    self.domain, 
                    adaptive=True,
                    n_mc=10, #self.n_mc_quadrature,
                )
            return model, prior_pref
        else:
            return model
    
    def generate_pairwise_candidates(self, model, beta, prior_pref=None, gamma=None):
        if self.learn_preference:
            if self.probabilistic_pi:
                acqf = DuelingAcquisitionFunction(
                    model, 
                    prior_pref, 
                    beta, 
                    gamma,
                    method=self.acqf_method,
                    hallucinate=self.hallucinate,
                    n_restarts=self.n_restarts,
                    raw_samples=self.raw_samples,
                )
            else:
                acqf = PiBODuelingAcquisitionFunction(
                    model, 
                    prior_pref, 
                    beta, 
                    gamma,
                    method=self.acqf_method,
                    n_restarts=self.n_restarts,
                    raw_samples=self.raw_samples,
                )
        else:
            acqf = BaselineDuelingAcquisitionFunction(
                model,
                self.domain,
                beta,
                bounds=self.domain.bounds,
                method=self.acqf_method,
                n_restarts=self.n_restarts,
                raw_samples=self.raw_samples,
            )
        
        X_pairwise_next = acqf()
        dist = (X_pairwise_next[:,0] - X_pairwise_next[:,1]).pow(2).item()
        return X_pairwise_next, dist
    
    def query(self, X_pairwise_next):
        y_pairwise_next, y_pairwise_unsure_next = self.duel.feedback(X_pairwise_next, sigma=self.sigma, in_loop=True)
        if self.adversarial:
            y_pairwise_next = 1 - y_pairwise_next
        X_next = torch.chunk(X_pairwise_next, dim=1, chunks=2)[1 - y_pairwise_next]
        Y_next = self.true_function(X_next)
        return X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next
    
    def update_datasets(self, dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new):
        X, Y = dataset_obj
        X_next, Y_next = dataset_obj_new
        
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        dataset_obj = (X, Y)
        dataset_duel = self.duel.update_and_augment_data(dataset_duel, dataset_duel_new)
        return dataset_obj, dataset_duel
    
    def safe_sampling(self, model, u, n_samples=256):
        try:
            f = gp_sample(model, u)
            if torch.isnan(f).any():
                raise ValueError("Contains NaN")
            return f
        except:
            pred = model.posterior(u)
            mus = pred.loc.detach()
            sigmas = pred.stddev.detach()
            functions = []
            for mu, sigma in zip(mus, sigmas):
                f = D.Normal(mu, sigma).sample(torch.Size([n_samples]))
                functions.append(f)
            return torch.vstack(functions).T
    
    def compute_probability_of_improvement(self, model, X_pairwise):
        u, u_prime = torch.chunk(X_pairwise, dim=1, chunks=2)
        f0 = self.safe_sampling(model, u)
        f1 = self.safe_sampling(model, u_prime)
        functions = f0 - f1
        pi_f = D.Normal(0,1).cdf(functions.squeeze() / model.likelihood.noise.sqrt()).detach()
        pi = pi_f.mean(axis=0)
        pi_std = pi_f.std(axis=0)
        return pi, pi_std
    
    def posthoc_evaluation(self, dataset_obj, dataset_duel, dataset_duel_new):
        X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next = dataset_duel_new
        
        model = set_and_fit_rbf_model(X, Y)
        
        # estimate this time answer correctness
        pi_mean, pi_std = self.compute_probability_of_improvement(model, X_pairwise_next)
        if y_pairwise_next == 0:
            pi_mean = 1 - pi_mean
        print(f"Probability of correct selection: {pi_mean.item():.2e} ± {pi_std.item():.2e}")
        
        # estimate total answer correctness
        X_sure = X_pairwise[y_pairwise_unsure.bool()]
        Y_sure = y_pairwise[y_pairwise_unsure.bool()]
        pi, pi_std = self.compute_probability_of_improvement(model, X_sure)
        total_pi_mean = 1 - (pi - Y_sure).abs().mean()
        total_pi_std = pi_std.mean()
        print(f"Estimated total correct selection rate of sure samples: {total_pi_mean.item():.2e} ± {total_pi_std.item():.2e}")
        return pi_mean.item(), total_pi_mean.item()
    
    def __call__(self, dataset_obj, dataset_duel, beta, gamma, sigma=None, window_size=20, initial=False):
        if not sigma == None:
            self.sigma = sigma
        X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        tic = time.monotonic()
        # 1. XBO loop
        if self.learn_preference:
            model, prior_pref = self.set_models(X, Y, dataset_duel, window_size=window_size, initial=initial)
            X_pairwise_next, dist = self.generate_pairwise_candidates(
                model,  
                beta,
                prior_pref,
                gamma,
            )
        else:
            model = self.set_models(X, Y, X_pairwise, y_pairwise)
            X_pairwise_next, dist = self.generate_pairwise_candidates(
                model,
                beta,
            )
        tok = time.monotonic()   
        X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next = self.query(X_pairwise_next)
        overhead = tok - tic
        
        dataset_obj_new = (X_next, Y_next)
        dataset_duel_new = (X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next)
        dataset_obj, dataset_duel = self.update_datasets(
            dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new,
        )
        
        # 2. evaluate the process
        X, Y = dataset_obj
        best_obs = Y.max().item()
        pi, total_pi_mean = self.posthoc_evaluation(dataset_obj, dataset_duel, dataset_duel_new)
        #correct_answer_rate = self.duel.evaluate_correct_answer_rate(X_pairwise, y_pairwise)
        results = [overhead, best_obs, dist, pi, total_pi_mean]
        return dataset_obj, dataset_duel, results

        
class AdaptiveXBOwithHuman(HumanFeedback):
    def __init__(
        self, 
        domain, 
        true_function, 
        sigma=0,
        training_iter=200,
        n_mc_quadrature=100,
        n_restarts=10,
        raw_samples=512,
        quadrature_method="mc",
        acqf_method="dueling",
        probabilistic_pi=True,
        hallucinate=True,
        adversarial=False,
    ):
        HumanFeedback.__init__(self)
        self.domain = domain
        self.true_function = true_function
        self.duel = DuelFeedback(domain, true_function)
        self.sigma = sigma # noise level of human feedback
        self.training_iter = training_iter
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.quadrature_method = quadrature_method # select from ["mc", "bq"]
        self.n_mc_quadrature = n_mc_quadrature
        self.probabilistic_pi = probabilistic_pi
        self.hallucinate = hallucinate
        self.acqf_method = acqf_method
        self.adversarial = adversarial
        
    def query_to_human(self, X_pairwise, X=None, Y=None, model=None, prior_pref=None, beta=None, explanation=True):
        n_dims = int(X_pairwise.shape[1] / 2)
        if len(X_pairwise) == 1:
            self.display_pairwise_samples(X_pairwise)
            if explanation:
                self.explanation_flow(X_pairwise.reshape(2, n_dims), X, Y, model, prior_pref, beta)
            y_pairwise, y_pairwise_unsure = self.get_human_feedback()
        else:
            y_pairwise = []
            y_pairwise_unsure = []
            for epoch, X_pairwise_next in enumerate(X_pairwise):
                print("Epoch: "+str(epoch+1)+"/"+str(len(X_pairwise)))
                self.display_pairwise_samples(X_pairwise_next.unsqueeze(0), random=True)
                y_pairwise_next, y_pairwise_unsure_next = self.get_human_feedback(rand=True)
                y_pairwise.append(y_pairwise_next)
                y_pairwise_unsure.append(y_pairwise_unsure_next)
            y_pairwise = torch.cat(y_pairwise)
            y_pairwise_unsure = torch.cat(y_pairwise_unsure)
        return y_pairwise.long(), y_pairwise_unsure.long()
        
    def initial_sampling(self, n_init_obj, n_init_pref):
        X = self.domain.sample(n_init_obj)
        Y = self.true_function(X.squeeze())
        X_pairwise = self.duel.sample(n_init_pref)
        y_pairwise, y_pairwise_unsure = self.query_to_human(X_pairwise, explanation=False)
        X_pairwise, y_pairwise, y_pairwise_unsure = self.duel.data_augment(X_pairwise, y_pairwise, y_pairwise_unsure)
        
        dataset_obj = (X, Y)
        dataset_duel = (X_pairwise, y_pairwise, y_pairwise_unsure)
        return dataset_obj, dataset_duel
    
    def set_models(self, X, Y, dataset_duel, window_size=20, initial=False):
        model = set_and_fit_rbf_model(X, Y)
        
        if initial:
            self.model_pref = initial_set_agp_model(
                dataset_duel, 
                self.duel.prior_duel,
                window_size=window_size,
                training_iter=self.training_iter,
            )
        else:
            self.model_pref = online_update_model(
                self.model_pref,
                dataset_duel,
                self.duel.prior_duel,
                window_size=window_size,
                training_iter=self.training_iter,
            )
                
        if self.quadrature_method == "bq":
            prior_pref = BayesianQuadrature(self.model_pref, self.domain)
        else:
            prior_pref = MonteCarloQuadrature(
                self.model_pref, 
                self.domain, 
                adaptive=True,
                n_mc=10, #self.n_mc_quadrature,
            )
        return model, prior_pref
        
    def generate_pairwise_candidates(self, model, beta, prior_pref=None, gamma=None):
        if self.probabilistic_pi:
            acqf = DuelingAcquisitionFunction(
                model, 
                prior_pref, 
                beta, 
                gamma,
                method=self.acqf_method,
                hallucinate=self.hallucinate,
                n_restarts=self.n_restarts,
                raw_samples=self.raw_samples,
            )
        else:
            acqf = PiBODuelingAcquisitionFunction(
                model, 
                prior_pref, 
                beta, 
                gamma,
                method=self.acqf_method,
                n_restarts=self.n_restarts,
                raw_samples=self.raw_samples,
            )
        
        X_pairwise_next = acqf()
        dist = (X_pairwise_next[:,0] - X_pairwise_next[:,1]).pow(2).sqrt().item()
        return X_pairwise_next, dist
    
    def query(self, X_pairwise_next, X, Y, model, prior_pref, beta):
        y_pairwise_next, y_pairwise_unsure_next = self.query_to_human(X_pairwise_next, X, Y, model, prior_pref, beta)
        X_next = torch.chunk(X_pairwise_next, dim=1, chunks=2)[1 - y_pairwise_next]
        Y_next = self.true_function(X_next)
        return X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next
    
    def update_datasets(self, dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new):
        X, Y = dataset_obj
        X_next, Y_next = dataset_obj_new
        
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        dataset_obj = (X, Y)
        dataset_duel = self.duel.update_and_augment_data(dataset_duel, dataset_duel_new)
        return dataset_obj, dataset_duel
    
    def safe_sampling(self, model, u, n_samples=256):
        try:
            f = gp_sample(model, u)
            if torch.isnan(f).any():
                raise ValueError("Contains NaN")
            return f
        except:
            pred = model.posterior(u)
            mus = pred.loc.detach()
            sigmas = pred.stddev.detach()
            functions = []
            for mu, sigma in zip(mus, sigmas):
                f = D.Normal(mu, sigma).sample(torch.Size([n_samples]))
                functions.append(f)
            return torch.vstack(functions).T
    
    def compute_probability_of_improvement(self, model, X_pairwise):
        u, u_prime = torch.chunk(X_pairwise, dim=1, chunks=2)
        f0 = self.safe_sampling(model, u)
        f1 = self.safe_sampling(model, u_prime)
        functions = f0 - f1
        pi_f = D.Normal(0,1).cdf(functions.squeeze() / model.likelihood.noise.sqrt()).detach()
        pi = pi_f.mean(axis=0)
        pi_std = pi_f.std(axis=0)
        return pi, pi_std
    
    def posthoc_evaluation(self, dataset_obj, dataset_duel, dataset_duel_new):
        X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next = dataset_duel_new
        
        model = set_and_fit_rbf_model(X, Y)
        
        # estimate this time answer correctness
        pi_mean, pi_std = self.compute_probability_of_improvement(model, X_pairwise_next)
        if y_pairwise_next == 0:
            pi_mean = 1 - pi_mean
        print(f"Probability of correct selection: {pi_mean.item():.2e} ± {pi_std.item():.2e}")
        
        # estimate total answer correctness
        X_sure = X_pairwise[y_pairwise_unsure.bool()]
        Y_sure = y_pairwise[y_pairwise_unsure.bool()]
        pi, pi_std = self.compute_probability_of_improvement(model, X_sure)
        total_pi_mean = 1 - (pi - Y_sure).abs().mean()
        total_pi_std = pi_std.mean()
        print(f"Estimated total correct selection rate of sure samples: {total_pi_mean.item():.2e} ± {total_pi_std.item():.2e}")
        return pi_mean.item(), total_pi_mean.item()
    
    def __call__(self, dataset_obj, dataset_duel, beta, gamma, window_size=20, initial=False):
        X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        tic = time.monotonic()
        
        # 1. XBO loop
        print("training models...")
        model, prior_pref = self.set_models(X, Y, dataset_duel, window_size=window_size, initial=initial)
        print("generating candidates...")
        X_pairwise_next, dist = self.generate_pairwise_candidates(
            model,
            beta,
            prior_pref,
            gamma,
        )
        tok = time.monotonic()
        X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next = self.query(X_pairwise_next, X, Y, model, prior_pref, beta)
        overhead = tok - tic
        
        dataset_obj_new = (X_next, Y_next)
        dataset_duel_new = (X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next)
        dataset_obj, dataset_duel = self.update_datasets(
            dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new,
        )
        
        # 2. evaluate the process
        X, Y = dataset_obj
        best_obs = Y.max().item()
        pi, total_pi_mean = self.posthoc_evaluation(dataset_obj, dataset_duel, dataset_duel_new)
        #correct_answer_rate = self.duel.evaluate_correct_answer_rate(X_pairwise, y_pairwise)
        results = [overhead, best_obs, dist, pi, total_pi_mean]
        return dataset_obj, dataset_duel, results

class StateManager(TensorManager):
    def __init__(self, n_dims, beta_init=0.2, gamma_init=0.01, sigma_init=0.1, probabilistic_pi=True):
        TensorManager.__init__(self)
        self.n_dims = n_dims
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.sigma_init = sigma_init
        self.probabilistic_pi = probabilistic_pi
        
    def __call__(self, t):
        beta = self.beta_init * self.n_dims * self.sqrt(2*(1 + t)).item()
        sigma = self.sigma_init / (t+1)
        if self.probabilistic_pi:
            gamma = self.gamma_init * (t**2)
        else:
            gamma = self.gamma_init / (t+1)
        print(f"{t}) parameters: beta {beta:.3e} gamma {gamma:.3e} sigma {sigma:.3e}")
        return beta, gamma, sigma