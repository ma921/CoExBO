import torch
import pickle
import numpy as np
import pandas as pd
from CoExBO._utils import TensorManager
from CoExBO._gp_regressor import set_and_fit_rbf_model, predict

class ElectrolyteSearch(TensorManager):
    def __init__(
        self, 
        path_data="./experiments/AEM_training_data.csv",
        path_prior="./experiments/electrolyte_prior.pickle",
        feature_names=["LiPF6","EC","DMC"],
        sigma=3.,
    ):
        """
        Electrolyte Search problem
        
        Args:
        - path_data: string, the path to the training data
        - path_prior: string, the path to the experts' duel data
        - feature_names: list, list of feature names
        - sigma: float, Gaussian noise variance in experimental data feedback
        """
        TensorManager.__init__(self)
        self.loading_data(path_data)
        self.prior_duel = self.load_prior_duel(path_prior)
        self.feature_names = feature_names
        self.train_gp(self.X, self.Y)
        self.noise = torch.distributions.Normal(0, sigma)
        
    def loading_data(self, path):
        """
        loading training data for interpolation
        
        Args:
        - path: string, the path to the training data
        """
        df = pd.read_csv(path, index_col=0)
        self.X = self.from_numpy(np.array(df.iloc[:,:3]))
        self.Y = self.from_numpy(np.array(df.iloc[:,3]))
        self.bounds = torch.vstack([
            self.X.min(axis=0).values, 
            self.X.max(axis=0).values,
        ])
        
    def load_prior_duel(self, path):
        """
        loading the experts' duel data
        
        Args:
        - path: string, the path to experts' duel data
        
        Return:
        - dataset_duel: list, the list of experts' duel data
        """
        with open(path, 'rb') as handle:
            dataset_duel = pickle.load(handle)
        return dataset_duel
    
    def train_gp(self, X, Y):
        """
        Training the Gaussian process regression model to interpolate values.
        
        Args:
        - X: torch.tensor, the input variables
        - Y: torch.tensor, the output variables
        """
        self.model = set_and_fit_rbf_model(X, Y)
        
    def __call__(self, X):
        """
        Return the experimental values estimated by GP.
        
        Args:
        - X: torch.tensor, the input variables
        
        Return:
        - f + epsilon: torch.tensor, the noisy output variables
        - f: torch.tensor, the noiseless output variables
        """
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        pred = predict(X, self.model)
        f = pred.loc * self.Y.var() + self.Y.mean()
        epsilon = self.noise.sample(torch.Size([len(X)]))
        return f + epsilon, f