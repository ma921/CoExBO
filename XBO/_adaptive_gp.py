import torch
import pyro
import gpytorch
import torch.distributions as D
from gpytorch.likelihoods import DirichletClassificationLikelihood
from .AGP.AdaptiveSparseGPRegression import AdaptiveSparseGPRegression 

def preprocessing(dataset_duel, window_size=20, initial=False):
    X_pairwise, Y_pairwise, Y_pairwise_unsure = dataset_duel
    likelihood = DirichletClassificationLikelihood(
        Y_pairwise.long(),
        learn_additional_noise=True,
    )
    
    idx_sure = Y_pairwise_unsure.bool().logical_not()
    if initial:
        X_agp_init = X_pairwise[idx_sure][:window_size]
        Y_agp_init = likelihood.transformed_targets[:,idx_sure][:, :window_size]
        X_agp_learn = X_pairwise[idx_sure][window_size:]
        Y_agp_learn = likelihood.transformed_targets[:,idx_sure][:, window_size:]
        return X_agp_init, Y_agp_init, X_agp_learn, Y_agp_learn
    else:
        X_agp_learn = X_pairwise[idx_sure][-2:]
        Y_agp_learn = likelihood.transformed_targets[:,idx_sure][:, -2:]
        return X_agp_learn, Y_agp_learn

def initial_set_agp_model(dataset_duel, prior_duel, window_size=20, training_iter=100):
    # initialize the inducing inputs in interval [0,1] 
    X_agp_init, Y_agp_init, X_agp_learn, Y_agp_learn = preprocessing(
        dataset_duel, window_size=window_size, initial=True,
    )
    pyro.clear_param_store()
    kernel = pyro.contrib.gp.kernels.RBF(input_dim=X_agp_init.shape[-1])
    n_inducing_points = window_size
    inducing_points = prior_duel.sample(n_inducing_points)

    # Define the model
    osgpr = AdaptiveSparseGPRegression(
        X_agp_init,
        Y_agp_init, 
        kernel,
        Xu=inducing_points,
        lamb=0.98,
        jitter=1.0e-3,
    )
    # Initialize the model
    osgpr.batch_update(num_steps=training_iter)
    
    for t, (x, y) in enumerate(zip(X_agp_learn, Y_agp_learn.T)):
        loss = osgpr.online_update(
            x.unsqueeze(0), 
            y.unsqueeze(0), 
            L=window_size, 
            M=window_size, 
            num_steps=1, 
            perc_th = 0.01,
        )
    return osgpr

def online_update_model(osgpr, dataset_duel, prior_duel, window_size=20, training_iter=100):
    # initialize the inducing inputs in interval [0,1] 
    X_agp_learn, Y_agp_learn = preprocessing(
        dataset_duel, window_size=window_size, initial=False,
    )
    for t, (x, y) in enumerate(zip(X_agp_learn, Y_agp_learn.T)):
        loss = osgpr.online_update(
            x.unsqueeze(0), 
            y.unsqueeze(0), 
            L=window_size, 
            M=window_size, 
            num_steps=10, 
            perc_th=0.01,
        )
    return osgpr

def gp_sample(osgpr, X, seed=0):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    with torch.no_grad():
        pred, cov = osgpr(X, full_cov=True, noiseless=False)
    pred = gpytorch.distributions.MultivariateNormal(pred, cov)
    pred_samples = pred.sample(torch.Size([256])).exp()
    return pred_samples

def preference_probability(osgpr, X, std=False, both=False):
    pred_samples = gp_sample(osgpr, X)
    prob_dist = (pred_samples / pred_samples.sum(-2, keepdim=True))
    f_samples = prob_dist[:,1,:]
    prob_mean = f_samples.mean(axis=0)
    if std:
        prob_std = (f_samples * (1 - f_samples)).std(axis=0)
        return prob_mean, prob_std
    else:
        return prob_mean