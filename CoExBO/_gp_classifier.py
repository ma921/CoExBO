import torch
import gpytorch
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RFFKernel, RBFKernel
from gpytorch.means import ConstantMean


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, base_kernel):
        """
        The Dirichlet GP model class.
        See details in the following GPyTorch tutorial.
        https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html
        
        Args:
        - train_x: torch.tensor, the training inputs
        - train_y: torch.tensor, the transformed label from categorical to continuous values.
        - likelihood: gpytorch.likelihoods.DirichletClassificationLikelihood, the Dirichlet likelihood
        - num_classes: int, number of classes. e.g. binary classification is two.
        - base_kernel: gpytorch.kernels, base kernel. We strongly recommend RFFKernel for fast computation.
        """
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            base_kernel(batch_shape=torch.Size((num_classes,)), num_samples=256),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        """
        Output the posterior predictive distribution
        
        Args:
        - x: torch.tensor, the inputs for test.
        
        Return:
        - pred: gpytorch.distributions.MultivariateNormal, the posterior predictive distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def set_gp_classifier(X_pairwise, y_pairwise):
    """
    Set up the binary classifier based on the pairwise inputs.

    Args:
    - X_pairwise: torch.tensor, the observed pairwise candidate.
    - y_pairwise: torch.tensor, the observed preference result

    Return:
    - model: gpytorch.models, the Dirichlet GP model for binary classification
    """
    likelihood = DirichletClassificationLikelihood(
        y_pairwise.long(),
        learn_additional_noise=True,
    )
    model = DirichletGPModel(
        X_pairwise.float(),
        likelihood.transformed_targets.float(),
        likelihood,
        num_classes=likelihood.num_classes,
        base_kernel=RFFKernel,
    )
    return model
    
def train_by_sgd(model, training_iter=10):
    """
    Training the Dirichlet GP model by SGD (Adam).

    Args:
    - model: gpytorch.models, the Dirichlet GP model before training
    - training_iter: int, the number of iterations for SGD training

    Return:
    - model: gpytorch.models, the Dirichlet GP model after training
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    train_x = mll.model.train_inputs[0]
    train_y = mll.model.train_targets
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, model.likelihood.transformed_targets).sum()
        loss.backward()
        optimizer.step()
    return model

def set_and_train_classifier(X_pairwise, y_pairwise, training_iter=50):
    """
    Set and Train the Dirichlet GP model in one go.

    Args:
    - X_pairwise: torch.tensor, the observed pairwise candidate.
    - y_pairwise: torch.tensor, the observed preference result
    - training_iter: int, the number of iterations for SGD training

    Return:
    - model: gpytorch.models, the Dirichlet GP model after training
    """
    model = set_gp_classifier(X_pairwise, y_pairwise)
    model = train_by_sgd(model, training_iter=training_iter)
    return model

def gp_sample(model, X, seed=0):
    """
    Sampling functions from the Dirichlet GP model.

    Args:
    - model: gpytorch.models, the Dirichlet GP model to sample
    - X: torch.tensor, the input to condition.
    - seed: int, random seed to fix

    Return:
    - pred_samples: torch.tensor, the sampled functions
    """
    torch.manual_seed(seed)
    model.eval()
    model.likelihood.eval()
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(X.float())
    with gpytorch.settings.fast_computations(covar_root_decomposition=True):
        pred_samples = test_dist.sample(torch.Size((256,))).exp()
    return pred_samples

def preference_probability(model, X, std=False):
    """
    Compute the preference probability.

    Args:
    - model: gpytorch.models, the Dirichlet GP model to sample
    - X: torch.tensor, the input to condition.
    - std: bool, whether or not we need the standard deviation outputs.

    Return:
    - prob_mean: torch.tensor, the mean of the probability of being Condorcet winner
    - prob_std: torch.tensor, the std of the probability of being Condorcet winner
    """
    pred_samples = gp_sample(model, X)
    prob_dist = (pred_samples / pred_samples.sum(-2, keepdim=True))
    f_samples = prob_dist[:,1,:]
    prob_mean = f_samples.mean(axis=0)
    if std:
        prob_std = (f_samples * (1 - f_samples)).std(axis=0)
        return prob_mean, prob_std
    else:
        return prob_mean