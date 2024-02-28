import copy
import torch
import torch.distributions as D
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ._utils import TensorManager


class BasePrior(ABC, TensorManager):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample(self, X):
        r"""Sampling from the prior"""
        pass
    
    @abstractmethod
    def pdf(self, X):
        r"""Return the probability density function of the prior"""
        pass

class Uniform(BasePrior):
    def __init__(self, bounds):
        """
        Uniform prior class
        
        Args:
        - bounds: torch.tensor, the lower and upper bounds for each dimension
        """
        super().__init__() # call TensorManager
        self.bounds = self.standardise_tensor(bounds)
        self.n_dims = self.bounds.shape[1]
        self.type = "continuous"
        
    def sample(self, n_samples, qmc=True):
        """
        Sampling from Uniform prior
        
        Args:
        - n_samples: int, the number of initial samples
        - qmc: bool, sampling from Sobol sequence if True, otherwise simply Monte Carlo sampling.
        
        Return:
        - samples: torch.tensor, the samples from uniform prior
        """
        random_samples = self.rand(self.n_dims, n_samples, qmc=qmc)
        samples = self.bounds[0].unsqueeze(0) + (
            self.bounds[1] - self.bounds[0]
        ).unsqueeze(0) * random_samples
        return samples
    
    def pdf(self, samples):
        """
        The probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        _pdf = self.ones(len(samples)) * (1/(self.bounds[1] - self.bounds[0])).prod()
        _ood = torch.logical_or(
            (samples >= self.bounds[1]).any(axis=1), 
            (samples <= self.bounds[0]).any(axis=1),
        ).logical_not()
        return self.standardise_tensor(_pdf * _ood)
    
    def logpdf(self, samples):
        """
        The log probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the log PDF over samples
        """
        _logpdf = self.ones(len(samples)) * (1/(self.bounds[1] - self.bounds[0])).prod().log()
        _ood = torch.logical_or(
            (samples >= self.bounds[1]).any(axis=1), 
            (samples <= self.bounds[0]).any(axis=1),
        ).logical_not()
        return self.standardise_tensor(_logpdf * _ood)
