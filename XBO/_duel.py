import torch
import torch.distributions as D
from ._prior import Uniform
from ._utils import TensorManager


class DuelFeedback(TensorManager):
    def __init__(
        self, 
        prior_init, 
        true_function,
        n_cand=20000,
    ):
        super().__init__()
        self.prior_init = prior_init
        self.prior_duel = Uniform(prior_init.bounds.repeat(1,2))
        self.true_function = true_function

    def initialise_variance(self, dataset_obj):
        X, Y = dataset_obj
        self.Y_var = Y.var()
    
    def feedback(self, X_pairwise, sigma=0, in_loop=True):
        u, u_prime = torch.chunk(X_pairwise, dim=1, chunks=2)
        if not sigma == 0:
            noise = D.Normal(0, sigma*self.Y_var).sample(torch.Size([len(u)]))
            noise_prime = D.Normal(0, sigma*self.Y_var).sample(torch.Size([len(u)]))
        else:
            noise = self.zeros(len(u))
            noise_prime = self.zeros(len(u_prime))
        
        y = self.true_function(u.squeeze())
        y_prime = self.true_function(u_prime.squeeze())
        
        y_pairwise = (y + noise > y_prime + noise_prime).long()
        thresh = sigma * self.Y_var
        bool_unsure = ((y - y_prime).abs().pow(2) <= thresh)
        y_pairwise_unsure = bool_unsure.long()
        if in_loop:
            y_pairwise[bool_unsure] = 1

        return y_pairwise, y_pairwise_unsure

    def augmented_feedback(self, X_pairwise, sigma=0, in_loop=True):
        y_pairwise, y_pairwise_unsure = self.feedback(X_pairwise, sigma=sigma, in_loop=in_loop)
        X_pairwise, y_pairwise, y_pairwise_unsure = self.data_augment(X_pairwise, y_pairwise, y_pairwise_unsure)
        return X_pairwise, y_pairwise, y_pairwise_unsure

    def swap_columns(self, X_pairwise):
        dim = int(X_pairwise.shape[-1] / 2)
        index = torch.cat([torch.arange(dim,2*dim), torch.arange(dim)])
        return X_pairwise[:,index]

    def data_augment(self, X_pairwise, y_pairwise, y_pairwise_unsure):
        X_pairwise_swap = self.swap_columns(X_pairwise)
        X_cat = torch.vstack([X_pairwise, X_pairwise_swap])
        #if y_pairwise_unsure == None:
        #    y_pairwise_swap = 1 - y_pairwise
        #else:
        # make dataset contradicting when unsure
        y_pairwise_swap = abs(y_pairwise_unsure - y_pairwise)
        y_cat = torch.cat([y_pairwise, y_pairwise_swap], dim=0)
        y_unsure = torch.cat([y_pairwise_unsure, y_pairwise_unsure], dim=0)
        return X_cat, y_cat, y_unsure
    
    def sample(self, n_sample):
        X_pairwise = self.prior_duel.sample(n_sample)
        return X_pairwise

    def sample_both(self, n_init, sigma=0, in_loop=True):
        X_pairwise = self.sample(n_init)
        X_pairwise, y_pairwise, y_pairwise_unsure = self.augmented_feedback(X_pairwise, sigma=sigma, in_loop=in_loop)
        return X_pairwise, y_pairwise, y_pairwise_unsure

    def update_and_augment_data(self, dataset_duel, dataset_duel_new):
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        X_pairwise_new, y_pairwise_new, y_pairwise_unsure_new = dataset_duel_new
        X_pairwise_new, y_pairwise_new, y_pairwise_unsure_new = self.data_augment(X_pairwise_new, y_pairwise_new, y_pairwise_unsure_new)
        X_cat = torch.vstack([X_pairwise, X_pairwise_new])
        y_cat = torch.cat([y_pairwise, y_pairwise_new], dim=0)
        y_unsure = torch.cat([y_pairwise_unsure, y_pairwise_unsure_new], dim=0)
        dataset_duel_updated = (X_cat, y_cat, y_unsure)
        return dataset_duel_updated
    
    def evaluate_correct_answer_rate(self, X_pairwise, y_pairwise):
        y_true, _ = self.feedback(X_pairwise, sigma=0)
        n_wrong = (y_pairwise - y_true).abs().sum().item()
        n_correct = (1 - n_wrong)
        return n_correct