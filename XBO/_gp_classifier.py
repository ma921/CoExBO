import torch
import gpytorch
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RFFKernel, RBFKernel
from gpytorch.means import ConstantMean


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, base_kernel):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            base_kernel(batch_shape=torch.Size((num_classes,)), num_samples=256),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def set_gp_classifier(X_pairwise, y_pairwise):
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
    model = set_gp_classifier(X_pairwise, y_pairwise)
    model = train_by_sgd(model, training_iter=training_iter)
    return model

def gp_sample(model, X, seed=0):
    torch.manual_seed(seed)
    model.eval()
    model.likelihood.eval()
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(X.float())
    with gpytorch.settings.fast_computations(covar_root_decomposition=True):
        pred_samples = test_dist.sample(torch.Size((256,))).exp()
    return pred_samples

def preference_probability(model, X, std=False, both=False):
    pred_samples = gp_sample(model, X)
    prob_dist = (pred_samples / pred_samples.sum(-2, keepdim=True))
    f_samples = prob_dist[:,1,:]
    prob_mean = f_samples.mean(axis=0)
    if std:
        prob_std = (f_samples * (1 - f_samples)).std(axis=0)
        return prob_mean, prob_std
    else:
        return prob_mean