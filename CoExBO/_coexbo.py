import time
import torch
import torch.distributions as D
from ._utils import TensorManager
from ._duel import DuelFeedback
from ._gp_regressor import set_and_fit_rbf_model
from ._gp_classifier import set_and_train_classifier, gp_sample
from ._monte_carlo_quadrature import MonteCarloQuadrature
from ._dueling_acquisition_function import DuelingAcquisitionFunction, PiBODuelingAcquisitionFunction, BaselineDuelingAcquisitionFunction
from ._human_interface import HumanFeedback


class CoExBO(HumanFeedback):
    def __init__(
        self, 
        domain, 
        true_function,
        feature_names=None,
        noisy=False,
        training_iter=200,
        n_mc_quadrature=100,
        n_restarts=10,
        raw_samples=512,
        explanation=True,
        acqf_method="dueling",
        probabilistic_pi=True,
        hallucinate=True,
        adversarial=False,
    ):
        """
        CoExBO main instance.
        
        Args:
        - domain: CoExBO._prior.BasePrior, the prior distribution over the domain.
        - true_function: class, the function that returns the true f values
        - feature_names: list, list of feature names
        - noisy: bool, whether or not the feedback contains noisy observations
        - training_iter: int, how many iterations we run SGD for training preference GP model.
        - n_mc_quadrature: int, how many samples we use to approximate the soft-Copeland score (human preference).
        - n_restarts: int, number of restarts for acquisition function optimization for querying.
        - raw_samples: int, numner of initial samples for acquisition function optimization for querying.
        - explanation: bool, whether or not we need the explanation feature.
        - acqf_method: string, the acquisiton function. select from ["ts", "nonmyopic", "dueling"]. "dueling" is CoExBO AF.
        - probabilistic_pi: bool, whether or not we use the uncertainty in prior estimation. "False" is "CoExBO (piBO)".
        - hallucinate: bool, whether or not we condition the GP on the normal BO query point.
        - adversarial: bool, (for test). If true, our selection will be reversed.
        """
        HumanFeedback.__init__(self, feature_names)
        self.domain = domain
        self.true_function = true_function
        self.noisy = noisy
        self.duel = DuelFeedback(domain, true_function, noisy=noisy)
        self.training_iter = training_iter
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.explanation = explanation
        self.n_mc_quadrature = n_mc_quadrature
        self.probabilistic_pi = probabilistic_pi
        self.hallucinate = hallucinate
        self.acqf_method = acqf_method
        self.adversarial = adversarial
        
    def query_to_human(self, X_pairwise, X=None, Y=None, model=None, prior_pref=None, beta=None, explanation=True):
        """
        Querying to human which candidates are preferred.
        
        Args:
        - X_pairwise: torch.tensor, a pairwise candidate
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - explanation: bool, whether or not we need the explanation feature.
        
        Return:
        - y_pairwise: torch.tensor, human selection results.
        - y_pairwise_unsure: torch.tensor, 1 if unsure, otherwise 0.
        """
        n_dims = int(X_pairwise.shape[1] / 2)
        if len(X_pairwise) == 1:
            # This triggers the case where we are running Human-in-the-loop experiments.
            self.display_pairwise_samples(X_pairwise)
            if explanation:
                self.explanation_flow(X_pairwise.reshape(2, n_dims), X, Y, model, prior_pref, beta)
            y_pairwise, y_pairwise_unsure = self.get_human_feedback()
        else:
            # This triggers the case where we are running the initial experiments.
            # We will ask human to collect the initial preference over random points.
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
        """
        Initial sampling.
        
        Args:
        - n_init_obj: int, number of intial samples for objective function.
        - n_init_pref: int, number of intial samples for human preference.
        
        Return:
        - dataset_obj: list, list of initial samples for objective function.
        - dataset_duel: list, list of initial samples for human preference.
        """
        X = self.domain.sample(n_init_obj)
        if self.noisy:
            Y, Y_true = self.true_function(X.squeeze())
        else:
            Y = self.true_function(X.squeeze())
        
        X_pairwise = self.duel.sample(n_init_pref)
        y_pairwise, y_pairwise_unsure = self.query_to_human(X_pairwise, explanation=False)
        X_pairwise, y_pairwise, y_pairwise_unsure = self.duel.data_augment(X_pairwise, y_pairwise, y_pairwise_unsure)
        
        if self.noisy:
            dataset_obj = (X, Y, Y_true)
        else:
            dataset_obj = (X, Y)
        dataset_duel = (X_pairwise, y_pairwise, y_pairwise_unsure)
        return dataset_obj, dataset_duel
    
    def set_models(self, X, Y, X_pairwise, y_pairwise):
        """
        Set models both for objective function and human preference.
        
        Args:
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - X_pairwise: torch.tensor, the observed pairwise candidates
        - y_pairwise: torch.tensor, the observed preference results
        
        Return:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        """
        model = set_and_fit_rbf_model(X, Y)
        model_pref = set_and_train_classifier(
            X_pairwise, 
            y_pairwise, 
            training_iter=self.training_iter,
        )
        prior_pref = MonteCarloQuadrature(model_pref, self.domain, n_mc=self.n_mc_quadrature)
        return model, prior_pref
    
    def generate_pairwise_candidates(self, model, beta, prior_pref=None, gamma=None):
        """
        Generate a pairwise candidate for the next query.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - gamma: float, decay hyperparameter in Eq.(7)
        
        Return:
        - X_pairwise_next: torch.tensor, a pairwise candidate for the next query
        - dist: float, Euclidean distance between the pairwise candidates to see how similar they are.
        """
        if self.probabilistic_pi:
            # (Default) Includes the uncertainty on expert preference elicitation
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
            # (For ablation study) does not include the uncertainty on expert preference elicitation
            # = vanilla piBO
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
        """
        Querying to both human users and true function.
        
        Args:
        - X_pairwise_next: torch.tensor, a pairwise candidate for the next query
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        
        Return:
        - X_next: torch.tensor, the observed input
        - Y_next: torch.tensor, the observed output (noisy)
        - Y_true_next: torch.tensor, the observed output (noiseless)
        - y_pairwise_next: torch.tensor, the observed preference result (sure)
        - y_pairwise_unsure_next: torch.tensor, the observed preference result (unsure)
        """
        y_pairwise_next, y_pairwise_unsure_next = self.query_to_human(X_pairwise_next, X, Y, model, prior_pref, beta, explanation=self.explanation)
        X_next = torch.chunk(X_pairwise_next, dim=1, chunks=2)[1 - y_pairwise_next]
        
        if self.noisy:
            Y_next, Y_true_next = self.true_function(X_next)
            return X_next, Y_next, Y_true_next, y_pairwise_next, y_pairwise_unsure_next
        else:
            Y_next = self.true_function(X_next)
            return X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next
    
    def update_datasets(self, dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new):
        """
        Merging old and new datasets for both objective and preference data.
        
        Args:
        - dataset_obj: list, list of the observed samples for objective function.
        - dataset_duel: list, list of the observed samples for human preference.
        - dataset_obj_new: list, list of the newly observed samples for objective function.
        - dataset_duel_new: list, list of the newly observed samples for human preference.
        
        Return:
        - dataset_obj: list, list of the merged samples for objective function.
        - dataset_duel: list, list of the merged samples for human preference.
        """
        if self.noisy:
            X, Y, Y_true = dataset_obj
            X_next, Y_next, Y_true_next = dataset_obj_new
        else:
            X, Y = dataset_obj
            X_next, Y_next = dataset_obj_new
        
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        if self.noisy:
            Y_true = torch.cat((Y_true, Y_true_next), dim=0)
            dataset_obj = (X, Y, Y_true)
        else:
            dataset_obj = (X, Y)
        dataset_duel = self.duel.update_and_augment_data(dataset_duel, dataset_duel_new)
        return dataset_obj, dataset_duel
    
    def safe_sampling(self, model, u, n_samples=256):
        """
        Safely sampling functions from Gaussian process model.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - u: torch.tensor, the input variable to condition the GP model.
        - n_samples: list, number of function samples.
        
        Return:
        - f: torch.tensor, function samples.
        """
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
        """
        Compute the probability of improvement. See Eqs.(89)-(90)
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - X_pairwise: torch.tensor, a pairwise candidate to investigate.
        
        Return:
        - pi: torch.tensor, the mean of the probability of improvement.
        - pi_std: torch.tensor, the standard deviation of the probability of improvement.
        """
        u, u_prime = torch.chunk(X_pairwise, dim=1, chunks=2)
        f0 = self.safe_sampling(model, u)
        f1 = self.safe_sampling(model, u_prime)
        functions = f0 - f1
        pi_f = D.Normal(0,1).cdf(functions.squeeze() / model.likelihood.noise.sqrt()).detach()
        pi = pi_f.mean(axis=0)
        pi_std = pi_f.std(axis=0)
        return pi, pi_std
    
    def posthoc_evaluation(self, dataset_obj, dataset_duel, dataset_duel_new):
        """
        Post-hoc evaluation of the human selection results.
        
        Args:
        - dataset_obj: list, list of the observed samples for objective function.
        - dataset_duel: list, list of the observed samples for human preference.
        - dataset_duel_new: list, list of the newly observed samples for human preference.
        
        Return:
        - pi_mean: flaot, the mean of the probability of correct selection.
        - total_pi_mean: float, the mean of the estimated total correct selection rate amongst sure samples.
        """
        if self.noisy:
            X, Y, Y_true = dataset_obj
        else:
            X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next = dataset_duel_new
        
        model = set_and_fit_rbf_model(X, Y)
        
        # estimate this time answer correctness
        pi_mean, pi_std = self.compute_probability_of_improvement(model, X_pairwise_next)
        if y_pairwise_next == 0:
            pi_mean = 1 - pi_mean
        if self.explanation:
            print(f"Probability of correct selection: {pi_mean.item():.2e} ± {pi_std.item():.2e}")
        
        # estimate total answer correctness
        X_sure = X_pairwise[y_pairwise_unsure.bool()]
        Y_sure = y_pairwise[y_pairwise_unsure.bool()]
        pi, pi_std = self.compute_probability_of_improvement(model, X_sure)
        total_pi_mean = 1 - (pi - Y_sure).abs().mean()
        total_pi_std = pi_std.mean()
        if self.explanation:
            print(f"Estimated total correct selection rate amongst sure samples: {total_pi_mean.item():.2e} ± {total_pi_std.item():.2e}")
        return pi_mean.item(), total_pi_mean.item()
    
    def __call__(self, dataset_obj, dataset_duel, beta, gamma):
        """
        Run CoExBO.
        Flow:
        1. Train models
        2. Generate a pairwise candidate
        3. Ask human which candidate to query with explanations.
        4. Query to true function.
        5. Update dataset
        6. evaluate the selection results.
        
        Args:
        - dataset_obj: list, list of the observed samples for objective function.
        - dataset_duel: list, list of the observed samples for human preference.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - gamma: float, decay hyperparameter in Eq.(7)
        
        Return:
        - results: list, list of the evaluation of the acquisition process.
        - dataset_obj: list, list of the updated samples for objective function.
        - dataset_duel: list, list of the updated samples for human preference.
        """
        self.duel.initialise_variance(dataset_obj)
        if self.noisy:
            X, Y, Y_true = dataset_obj
        else:
            X, Y = dataset_obj
        
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        tic = time.monotonic()
        
        # 1. CoExBO loop
        print("training models...")
        model, prior_pref = self.set_models(X, Y, X_pairwise, y_pairwise)
        print("generating candidates...")
        X_pairwise_next, dist = self.generate_pairwise_candidates(
            model,
            beta,
            prior_pref,
            gamma,
        )
        if self.noisy:
            (
                X_next, Y_next, Y_true_next, y_pairwise_next, y_pairwise_unsure_next
            ) = self.query(X_pairwise_next, X, Y, model, prior_pref, beta)
        else:
            (
                X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next
            ) = self.query(X_pairwise_next, X, Y, model, prior_pref, beta)
        
        tok = time.monotonic()
        overhead = tok - tic
        if self.noisy:
            dataset_obj_new = (X_next, Y_next, Y_true_next)
        else:
            dataset_obj_new = (X_next, Y_next)
        dataset_duel_new = (X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next)
        dataset_obj, dataset_duel = self.update_datasets(
            dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new,
        )
        
        # 2. evaluate the process
        if self.noisy:
            best_obs = dataset_obj[2].max().item()
        else:
            best_obs = dataset_obj[1].max().item()
        pi, total_pi_mean = self.posthoc_evaluation(dataset_obj, dataset_duel, dataset_duel_new)
        judge_correctness = self.duel.evaluate_correct_answer_rate(X_pairwise_next, y_pairwise_next)
        if self.explanation:
            print(f"Is your selection correct? Yes if 1: {judge_correctness}")
            print(f"Is your selection sure? Yes if 1: {y_pairwise_unsure_next.item()}")        
        results = [overhead, best_obs, dist, judge_correctness, y_pairwise_unsure_next.item()]
        return results, dataset_obj, dataset_duel


class CoExBOwithSimulation:
    def __init__(
        self, 
        domain, 
        true_function, 
        sigma=0,
        training_iter=200,
        n_mc_quadrature=100,
        n_restarts=10,
        raw_samples=512,
        acqf_method="dueling",
        probabilistic_pi=True,
        hallucinate=True,
        adversarial=False,
    ):
        """
        CoExBO main instance specialised for synthetic human response.
        
        Args:
        - domain: CoExBO._prior.BasePrior, the prior distribution over the domain.
        - true_function: class, the function that returns the true f values
        - sigma: float, Gaussian noise variance to the synthetic human selection process.
        - noisy: bool, whether or not the feedback contains noisy observations
        - training_iter: int, how many iterations we run SGD for training preference GP model.
        - n_mc_quadrature: int, how many samples we use to approximate the soft-Copeland score (human preference).
        - n_restarts: int, number of restarts for acquisition function optimization for querying.
        - raw_samples: int, numner of initial samples for acquisition function optimization for querying.
        - acqf_method: string, the acquisiton function. select from ["ts", "nonmyopic", "dueling"]. "dueling" is CoExBO AF.
        - probabilistic_pi: bool, whether or not we use the uncertainty in prior estimation. "False" is "CoExBO (piBO)".
        - hallucinate: bool, whether or not we condition the GP on the normal BO query point.
        - adversarial: bool, (for test). If true, our selection will be reversed.
        """
        self.domain = domain
        self.true_function = true_function
        self.duel = DuelFeedback(domain, true_function)
        self.sigma = sigma                                # noise level of synthetic human feedback
        self.training_iter = training_iter
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.n_mc_quadrature = n_mc_quadrature
        self.learn_preference = acqf_method in ["dueling"]
        self.probabilistic_pi = probabilistic_pi
        self.hallucinate = hallucinate
        self.acqf_method = acqf_method
        self.adversarial = adversarial
        
    def initial_sampling(self, n_init_obj, n_init_pref):
        """
        Initial sampling.
        
        Args:
        - n_init_obj: int, number of intial samples for objective function.
        - n_init_pref: int, number of intial samples for human preference.
        
        Return:
        - dataset_obj: list, list of initial samples for objective function.
        - dataset_duel: list, list of initial samples for human preference.
        """
        X = self.domain.sample(n_init_obj)
        Y = self.true_function(X.squeeze())
        dataset_obj = (X, Y)
        self.duel.initialise_variance(dataset_obj)
        
        X_pairwise, y_pairwise, y_pairwise_unsure = self.duel.sample_both(n_init_pref, sigma=self.sigma, in_loop=False)
        if self.adversarial:
            y_pairwise = 1 - y_pairwise
        dataset_duel = (X_pairwise, y_pairwise, y_pairwise_unsure)
        return dataset_obj, dataset_duel
    
    def set_models(self, X, Y, X_pairwise, y_pairwise):
        """
        Set models both for objective function and human preference.
        
        Args:
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - X_pairwise: torch.tensor, the observed pairwise candidates
        - y_pairwise: torch.tensor, the observed preference results
        
        Return:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        """
        model = set_and_fit_rbf_model(X, Y)
        if self.learn_preference:
            # CoExBO acquisition function is to learn human preference.
            model_pref = set_and_train_classifier(
                X_pairwise, 
                y_pairwise, 
                training_iter=self.training_iter,
            )
            prior_pref = MonteCarloQuadrature(model_pref, self.domain, n_mc=self.n_mc_quadrature)
            return model, prior_pref
        else:
            # The other benchmarking acquisition function is not to learn human preference.
            return model
    
    def generate_pairwise_candidates(self, model, beta, prior_pref=None, gamma=None):
        """
        Generate a pairwise candidate for the next query.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - gamma: float, decay hyperparameter in Eq.(7)
        
        Return:
        - X_pairwise_next: torch.tensor, a pairwise candidate for the next query
        - dist: float, Euclidean distance between the pairwise candidates to see how similar they are.
        """
        if self.learn_preference:
            # CoExBO acquisition function is to learn human preference.
            if self.probabilistic_pi:
                # (Default) Includes the uncertainty on expert preference elicitation
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
                # (For ablation study) does not include the uncertainty on expert preference elicitation
                # = vanilla piBO
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
            # The other benchmarking acquisition function is not to learn human preference.
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
        """
        Querying to both synthetic human response and true function.
        
        Args:
        - X_pairwise_next: torch.tensor, a pairwise candidate for the next query
        
        Return:
        - X_next: torch.tensor, the observed input
        - Y_next: torch.tensor, the observed output
        - y_pairwise_next: torch.tensor, the observed preference result (sure)
        - y_pairwise_unsure_next: torch.tensor, the observed preference result (unsure)
        """
        y_pairwise_next, y_pairwise_unsure_next = self.duel.feedback(X_pairwise_next, sigma=self.sigma, in_loop=True)
        if self.adversarial:
            y_pairwise_next = 1 - y_pairwise_next
        X_next = torch.chunk(X_pairwise_next, dim=1, chunks=2)[1 - y_pairwise_next]
        Y_next = self.true_function(X_next)
        return X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next
    
    def update_datasets(self, dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new):
        """
        Merging old and new datasets for both objective and preference data.
        
        Args:
        - dataset_obj: list, list of the observed samples for objective function.
        - dataset_duel: list, list of the observed samples for human preference.
        - dataset_obj_new: list, list of the newly observed samples for objective function.
        - dataset_duel_new: list, list of the newly observed samples for human preference.
        
        Return:
        - dataset_obj: list, list of the merged samples for objective function.
        - dataset_duel: list, list of the merged samples for human preference.
        """
        X, Y = dataset_obj
        X_next, Y_next = dataset_obj_new
        
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        dataset_obj = (X, Y)
        dataset_duel = self.duel.update_and_augment_data(dataset_duel, dataset_duel_new)
        return dataset_obj, dataset_duel
    
    def __call__(self, dataset_obj, dataset_duel, beta, gamma, sigma=None):
        """
        Run CoExBO.
        Flow:
        1. Train models
        2. Generate a pairwise candidate
        3. Query to synthetic human response function to select the candidate.
        4. Query to true function.
        5. Update dataset
        6. evaluate the selection results.
        
        Args:
        - dataset_obj: list, list of the observed samples for objective function.
        - dataset_duel: list, list of the observed samples for human preference.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - gamma: float, decay hyperparameter in Eq.(7)
        - sigma: float, Gaussian noise variance to the synthetic human selection process.
        
        Return:
        - result: list, list of the evaluation of the acquisition process.
        - dataset_obj: list, list of the updated samples for objective function.
        - dataset_duel: list, list of the updated samples for human preference.
        """
        self.duel.initialise_variance(dataset_obj)
        if not sigma == None:
            self.sigma = sigma
        X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        tic = time.monotonic()
        # 1. CoExBO loop
        if self.learn_preference:
            model, prior_pref = self.set_models(X, Y, X_pairwise, y_pairwise)
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
            
        X_next, Y_next, y_pairwise_next, y_pairwise_unsure_next = self.query(X_pairwise_next)
        tok = time.monotonic()
        overhead = tok - tic
        dataset_obj_new = (X_next, Y_next)
        dataset_duel_new = (X_pairwise_next, y_pairwise_next, y_pairwise_unsure_next)
        result = (overhead, dist)
        
        dataset_obj, dataset_duel, result = self.update_and_evaluate(result, dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new)
        return result, dataset_obj, dataset_duel
    
    def update_and_evaluate(self, result, dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new):
        """
        Update the dataset and perform post-hoc evaluation.
        
        Args:
        - result: list, list of the evaluation of the acquisition process.
        - dataset_obj: list, list of the observed samples for objective function.
        - dataset_duel: list, list of the observed samples for human preference.
        - dataset_obj_new: list, list of the newly observed samples for objective function.
        - dataset_duel_new: list, list of the newly observed samples for human preference.
        
        Return:
        - dataset_obj: list, list of the updated samples for objective function.
        - dataset_duel: list, list of the updated samples for human preference.
        - results: list, list of the evaluation of the acquisition process.
        """
        dataset_obj, dataset_duel = self.update_datasets(
            dataset_obj, dataset_duel, dataset_obj_new, dataset_duel_new,
        )
        X, Y = dataset_obj
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        overhead, dist = result
        
        # 2. evaluate the process
        X, Y = dataset_obj
        best_obs = Y.max().item()
        correct_answer_rate = self.duel.evaluate_correct_answer_rate(X_pairwise, y_pairwise)
        results = [overhead, best_obs, dist, correct_answer_rate]
        return dataset_obj, dataset_duel, results
    
class StateManager(TensorManager):
    def __init__(self, n_dims, beta_init=0.2, gamma_init=0.01, probabilistic_pi=True):
        """
        State parameter managers. The state parameters are beta and gamma.
        Beta increases over times followed by the common heuristics;
        βt = 0.2 d log(2t),
        where d is the dimension and t is the number of iteration.
        Gamma update follows Eq.(7) for CoExBO acquisition funciton,
        and gamma / t for the piBO acquisition function.
        
        Args:
        - n_dim: int, the number of dimension
        - beta_init: float, the initial value of beta
        - gamma_init: float, the initial value of gamma. 0.01 for CoExBO, 10 for piBO.
        - probabilistic_pi: bool, whether or not we use the uncertainty in prior estimation. "False" is "CoExBO (piBO)".
        """
        TensorManager.__init__(self)
        self.n_dims = n_dims
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.probabilistic_pi = probabilistic_pi
        
    def __call__(self, t):
        """
        Return the state parameters. Not that "t" starts from 0.
        
        Args:
        - t: int, the number of iterations
        
        Return:
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - gamma: float, decay hyperparameter in Eq.(7)
        """
        beta = self.beta_init * self.n_dims * self.sqrt(2*(1 + t)).item()
        if self.probabilistic_pi:
            gamma = self.gamma_init * (t**2)
        else:
            gamma = self.gamma_init / (t+1)
        print(f"{t}) parameters: beta {beta:.3e} gamma {gamma:.3e}")
        return beta, gamma