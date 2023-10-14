import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from ._utils import TensorManager
tm = TensorManager()

def set_rbf_model(X, Y):
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
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def set_and_fit_rbf_model(X, Y):
    model = set_rbf_model(X, Y)
    model = optimise_model(model)
    return model

def predict(test_x, model):
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

