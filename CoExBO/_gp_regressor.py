import torch
import gpytorch
import warnings
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from ._utils import TensorManager
tm = TensorManager()

def set_rbf_model(X, Y):
    """
    Set up the Gaussian process regression model with the RBF kernel.

    Args:
    - X: torch.tensor, the observed inputs for the objective values.
    - Y: torch.tensor, the observed objective values.

    Return:
    - model: gpytorch.models, the GP regression model
    """
    base_kernel = RBFKernel()
    covar_module = ScaleKernel(base_kernel)

    # Fit a GP model
    train_Y = (Y - Y.mean()) / Y.std()
    train_Y = train_Y.view(-1).unsqueeze(1)
    likelihood = GaussianLikelihood()
    model = SingleTaskGP(X, train_Y, likelihood=likelihood, covar_module=covar_module)
    if tm.is_cuda():
        return model.cuda()
    else:
        return model

def optimise_model(model):
    """
    Fitting the GP regression model with L-BFGS-B optimizer.

    Args:
    - model: gpytorch.models, the GP regression model before training

    Return:
    - model: gpytorch.models, the GP regression model after training
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def set_and_fit_rbf_model(X, Y):
    """
    Set and fit the GP regression model in one go.

    Args:
    - X: torch.tensor, the observed inputs for the objective values.
    - Y: torch.tensor, the observed objective values.

    Return:
    - model: gpytorch.models, the GP regression model after training
    """
    model = set_rbf_model(X, Y)
    model = optimise_model(model)
    return model

def predict(test_x, model):
    """
    Compute the posterior predictive distribution

    Args:
    - test_x: torch.tensor, the inputs for compute the predictive distribution.
    - model: gpytorch.models, the GP regression model.

    Return:
    - pred: gpytorch.models, gpytorch.distributions.MultivariateNormal, the posterior predictive distribution
    """
    model.eval()
    model.likelihood.eval()

    try:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = model.likelihood(model(test_x))
    except:
        warnings.warn("Cholesky failed. Adding more jitter...")
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(float=1e-2):
            pred = model.likelihood(model(test_x))
    return pred

